# -*- coding: utf-8 -*-
"""
Экономика накопителя по ценовым категориям 3–6 на сохранённых прогнозах.

Работает офлайн по каталогу прогона: читает forecast_series_test.csv
(прогнозы всех моделей и факт с отметками времени) и не переобучает ни одной
модели. Прежде вся экономика сводилась к одной цифре за один месяц по
трёхзонному тарифу; здесь для каждого источника прогноза считаются:

  * счёт и экономия по месяцам в выбранной ценовой категории;
  * доверительный интервал годовой экономии — блочный бутстреп по неделям
    суточных вкладов (экономия раскладывается по суткам точно, см.
    daily_savings);
  * чувствительность к ёмкости накопителя, его цене и ставкам мощности.

Контроллеры — суточный MPC по прогнозу и оптимум при известном будущем
(optimization/controllers.py). Разница между ними и есть цена ошибки
прогноза и ограничений суточного планирования.

Запуск:
    python -m analysis.economics results/runs/<каталог> --category 4
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from optimization.controllers import (
    Battery, evaluate_schedule, mpc_day_ahead, perfect_foresight_ru,
)
from optimization.tariffs_ru import RuTariff, calendar_frame, peak_hours_of_region

logger = logging.getLogger("smart_grid.analysis.economics")

ACTUAL_COLUMN = "__actual__"
ORACLE = "Идеальный прогноз"
UPPER_BOUND = "Оптимум при известном будущем"


# ══════════════════════════════════════════════════════════════════════════════
# ДАННЫЕ И НАКОПИТЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

def load_forecast_series(run_dir: str, split: str = "test") -> pd.DataFrame:
    path = os.path.join(run_dir, f"forecast_series_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Нет {path}. Ряды прогнозов сохраняются агрегатным режимом с блоком "
            "накопителя; прогоны до 28.09.2026 писали их в корень results/.")
    frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
    if ACTUAL_COLUMN not in frame.columns and "actual" in frame.columns:
        frame = frame.rename(columns={"actual": ACTUAL_COLUMN})
    if ACTUAL_COLUMN not in frame.columns:
        raise ValueError(f"В {path} нет колонки факта {ACTUAL_COLUMN}")
    return frame.dropna().reset_index(drop=True)


def battery_for_load(peak_kw: float, capex_rub_per_kwh: Optional[float] = None,
                     capacity_scale: float = 1.0) -> Battery:
    """
    Накопитель того же типоразмера, что в конвейере: мощность — доля пика,
    ёмкость — несколько часов этой мощности, износ — из цены и ресурса.
    """
    from config import Config, RealWorldReference as ref

    capex = ref.BATTERY_CAPEX_RUB_PER_KWH if capex_rub_per_kwh is None else capex_rub_per_kwh
    power = peak_kw * ref.BATTERY_POWER_SHARE_OF_PEAK * capacity_scale
    capacity = power * ref.BATTERY_DURATION_HOURS
    cost = capex * capacity
    return Battery(
        capacity=capacity, max_power=power, capex_rub=cost,
        round_trip_efficiency=ref.BATTERY_ROUND_TRIP_EFF,
        min_soc=Config.BATTERY_MIN_SOC, max_soc=Config.BATTERY_MAX_SOC,
        cycle_cost_per_kwh=cost / (capacity * ref.BATTERY_REF_DOD * ref.BATTERY_CYCLE_LIFE),
        annual_om_share=Config.BATTERY_OM_SHARE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# РАЗЛОЖЕНИЕ ЭКОНОМИИ ПО СУТКАМ
# ══════════════════════════════════════════════════════════════════════════════

def daily_savings(baseline: np.ndarray, grid: np.ndarray, charged: np.ndarray,
                  timestamps, tariff: RuTariff, battery: Battery,
                  plan: Optional[np.ndarray] = None,
                  holidays: Optional[Sequence[bool]] = None) -> pd.Series:
    """
    Чистая экономия по суткам; сумма равна итогу evaluate_schedule.

    Разложение точное, потому что все компоненты счёта аддитивны по суткам:
    энергия — почасово, обе мощности — как среднее по рабочим дням месяца,
    то есть каждый рабочий день несёт 1/n_рд месячной ставки. Именно это
    позволяет строить доверительный интервал бутстрепом по суткам.
    """
    from optimization.controllers import _marginal_energy_price

    baseline = np.asarray(baseline, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n = len(grid)
    cal = calendar_frame(timestamps, tariff, holidays)
    price = _marginal_energy_price(tariff, n)
    peaks = peak_hours_of_region(baseline, cal)

    hourly = (baseline - grid) * price - np.asarray(charged) * battery.cycle_cost_per_kwh
    hourly -= battery.capex_rub * battery.annual_om_share / 8760.0
    if tariff.with_plan and plan is not None:
        p = np.asarray(plan, dtype=np.float64)
        dev = lambda x: (tariff.deviation_up_rate * np.clip(x - p, 0, None)
                         + tariff.deviation_down_rate * np.clip(p - x, 0, None))
        hourly += dev(baseline) - dev(grid)
    frame = cal.assign(v=hourly, base=baseline, grid=grid, peak=peaks)
    out = frame.groupby("day")["v"].sum()

    for month, g in frame.groupby("month", sort=True):
        share = g["day"].nunique() / pd.Period(month).days_in_month
        work = g[g["working"]]
        n_wd = work["day"].nunique()
        if not n_wd:
            continue
        gen = tariff.gen_capacity_rate * share / n_wd
        pk = work[work["peak"]]
        out.loc[pk["day"]] += gen * (pk["base"] - pk["grid"]).to_numpy()
        if tariff.two_rate_network:
            net = tariff.net_capacity_rate * share / n_wd
            win = work[work["in_window"]].groupby("day")
            diff = win["base"].max() - win["grid"].max()
            out.loc[diff.index] += net * diff.to_numpy()
    return out


def block_bootstrap_annual(daily: pd.Series, n_boot: int = 2000, block_days: int = 7,
                           seed: int = 0, level: float = 0.90) -> Dict[str, float]:
    """
    Доверительный интервал годовой экономии: блоки по неделе суток.

    Блоки сохраняют недельный цикл и зависимость соседних суток; бутстреп по
    отдельным суткам сузил бы интервал. Блоки циклические, поэтому каждые
    сутки попадают в выборку одинаково часто. Результат — 365 × средняя суточная
    экономия.
    """
    values = daily.to_numpy(dtype=np.float64)
    n = len(values)
    if n == 0:
        return {"annual_mean": float("nan"), "annual_lo": float("nan"),
                "annual_hi": float("nan"), "n_days": 0}
    rng = np.random.RandomState(seed)
    # Не меньше трёх блоков: при одном блоке длиной во весь ряд каждая
    # бутстреп-выборка совпадала бы с исходной, и интервал схлопывался в точку.
    b = max(1, min(block_days, n // 3))
    n_blocks = int(np.ceil(n / b))
    # Циклические блоки: начало — любые сутки, конец блока переходит в начало
    # ряда. У обычных блоков с началом не позже n − b последние сутки попадали
    # в выборку реже прочих, и на коротком периоде с выбросом в конце интервал
    # смещался так, что не содержал собственного среднего.
    offsets = np.arange(b)
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[i] = values[idx].mean()
    alpha = (1.0 - level) / 2
    return {"annual_mean": float(values.mean() * 365),
            "annual_lo": float(np.quantile(means, alpha) * 365),
            "annual_hi": float(np.quantile(means, 1 - alpha) * 365),
            "n_days": int(n)}


# ══════════════════════════════════════════════════════════════════════════════
# ОЦЕНКА ИСТОЧНИКОВ ПРОГНОЗА
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sources(frame: pd.DataFrame, tariff: RuTariff, battery: Battery,
                     sources: Optional[List[str]] = None,
                     holidays: Optional[Sequence[bool]] = None,
                     n_boot: int = 2000) -> Dict[str, pd.DataFrame]:
    """
    Экономия каждого источника прогноза при суточном MPC и верхняя граница.

    План для категорий 5 и 6 — прогноз сетевой нагрузки с учётом расписания
    накопителя, то есть отклонения возникают только из-за ошибки прогноза.
    """
    ts = frame["timestamp"]
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype=np.float64)
    if sources is None:
        sources = [c for c in frame.columns if c not in ("timestamp", ACTUAL_COLUMN)]
    forecasts = {ORACLE: actual, **{s: frame[s].to_numpy(dtype=np.float64) for s in sources}}

    summary, monthly, days = [], [], {}
    schedules = {}
    for name, fc in forecasts.items():
        schedules[name] = (mpc_day_ahead(fc, actual, ts, tariff, battery, holidays), fc)
    schedules[UPPER_BOUND] = (perfect_foresight_ru(actual, ts, tariff, battery, holidays),
                              actual)

    from optimization.tariffs_ru import monthly_bill
    base_bill = monthly_bill(actual, ts, tariff, plan=actual, region_load=actual,
                             holidays=holidays)
    for name, (sched, fc) in schedules.items():
        plan = fc + (sched["grid"] - actual) if tariff.with_plan else None
        res = evaluate_schedule(sched["grid"], sched["charged"], actual, ts, tariff,
                                battery, plan=plan, holidays=holidays)
        d = daily_savings(actual, sched["grid"], sched["charged"], ts, tariff, battery,
                          plan=plan, holidays=holidays)
        boot = block_bootstrap_annual(d, n_boot=n_boot)
        mae = float(np.mean(np.abs(fc - actual)))
        summary.append({"source": name, "forecast_MAE": mae, **res, **boot})
        days[name] = d
        bill = monthly_bill(sched["grid"], ts, tariff, plan=plan, region_load=actual,
                            holidays=holidays)
        for (_, b0), (_, b1) in zip(base_bill.iterrows(), bill.iterrows()):
            monthly.append({"source": name, "month": b0["month"],
                            "gross_savings": b0["total"] - b1["total"]})

    table = pd.DataFrame(summary).sort_values("net_savings", ascending=False)
    bound = table.loc[table["source"] == UPPER_BOUND, "net_savings"].iloc[0]
    table["share_of_bound"] = table["net_savings"] / bound if bound > 0 else np.nan
    return {"summary": table.reset_index(drop=True),
            "monthly": pd.DataFrame(monthly),
            "daily": pd.DataFrame(days)}


def sensitivity(frame: pd.DataFrame, tariff: RuTariff, source: str,
                capacity_scales=(0.5, 1.0, 2.0),
                capex_per_kwh=(12_000.0, 16_000.0, 24_000.0),
                capacity_rate_scales=(0.7, 1.0, 1.3),
                holidays: Optional[Sequence[bool]] = None) -> pd.DataFrame:
    """
    Годовая экономия и окупаемость на сетке параметров.

    Для каждой точки заново решаются и MPC по прогнозу source, и верхняя
    граница: при другой ёмкости или ставке мощности меняется само расписание.
    """
    from dataclasses import replace

    ts = frame["timestamp"]
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype=np.float64)
    fc = actual if source == ORACLE else frame[source].to_numpy(dtype=np.float64)
    peak = float(actual.max())
    rows = []
    for scale in capacity_scales:
        for capex in capex_per_kwh:
            battery = battery_for_load(peak, capex, scale)
            for rate_scale in capacity_rate_scales:
                t = replace(tariff, gen_capacity_rate=tariff.gen_capacity_rate * rate_scale,
                            net_capacity_rate=tariff.net_capacity_rate * rate_scale)
                for label, sched in (
                        (source, mpc_day_ahead(fc, actual, ts, t, battery, holidays)),
                        (UPPER_BOUND, perfect_foresight_ru(actual, ts, t, battery, holidays))):
                    r = evaluate_schedule(sched["grid"], sched["charged"], actual, ts, t,
                                          battery, holidays=holidays)
                    rows.append({"controller": label, "capacity_scale": scale,
                                 "capacity_kwh": battery.capacity,
                                 "capex_rub_per_kwh": capex,
                                 "capacity_rate_scale": rate_scale,
                                 "annual_net_savings": r["annual_net_savings"],
                                 "payback_years": r["payback_years"]})
    return pd.DataFrame(rows)


def markdown_report(summary: pd.DataFrame, tariff: RuTariff, battery: Battery,
                    sens: Optional[pd.DataFrame]) -> str:
    lines = [f"# Экономика накопителя, ценовая категория {tariff.category}\n",
             f"Накопитель: {battery.capacity:.0f} кВт·ч, {battery.max_power:.0f} кВт, "
             f"CAPEX {battery.capex_rub / 1e6:.1f} млн руб. Ставки и окна часов СО — "
             "примерные, перед использованием заменить на тарифы региона.\n",
             "| Источник | MAE прогноза | Экономия за период, руб | Год, руб "
             "(90% ДИ) | Доля от оптимума | Окупаемость, лет |",
             "|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        payback = "—" if not np.isfinite(r["payback_years"]) else f"{r['payback_years']:.1f}"
        share = "—" if not np.isfinite(r.get("share_of_bound", np.nan)) else \
            f"{100 * r['share_of_bound']:.0f}%"
        lines.append(
            f"| {r['source']} | {r['forecast_MAE']:.1f} | {r['net_savings']:,.0f} "
            f"| {r['annual_mean']:,.0f} ({r['annual_lo']:,.0f} … {r['annual_hi']:,.0f}) "
            f"| {share} | {payback} |".replace(",", " "))
    lines.append("\n*Интервал — блочный бутстреп по неделям суточных вкладов в экономию.*\n")
    n_days = int(summary["n_days"].max()) if "n_days" in summary else 0
    if n_days < 21:
        lines.append(f"*Период — {n_days} суток: меньше трёх недель, интервал "
                     "ориентировочный.*\n")
    if sens is not None and len(sens):
        lines.append("## Чувствительность (годовая экономия, руб)\n")
        pivot = sens.pivot_table(index=["controller", "capacity_scale", "capex_rub_per_kwh"],
                                 columns="capacity_rate_scale", values="annual_net_savings")
        cols = list(pivot.columns)
        lines.append("| Контроллер | Ёмкость × | CAPEX, руб/кВт·ч | "
                     + " | ".join(f"ставки мощности × {c}" for c in cols) + " |")
        lines.append("|---|---|---|" + "---|" * len(cols))
        for (ctrl, scale, capex), row in pivot.iterrows():
            lines.append(f"| {ctrl} | {scale} | {capex:,.0f} | ".replace(",", " ")
                         + " | ".join(f"{v:,.0f}".replace(",", " ") for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_report(run_dir: str, category: int = 4, split: str = "test",
                 tariff_params: Optional[dict] = None,
                 sensitivity_source: Optional[str] = None,
                 out: Optional[str] = None, n_boot: int = 2000) -> Dict[str, pd.DataFrame]:
    """Считает экономику по каталогу прогона и пишет таблицы и отчёт рядом."""
    params = dict(tariff_params or {})
    params["category"] = category
    tariff = RuTariff(**params)

    frame = load_forecast_series(run_dir, split)
    battery = battery_for_load(float(frame[ACTUAL_COLUMN].max()))
    result = evaluate_sources(frame, tariff, battery, n_boot=n_boot)

    out = out or os.path.join(run_dir, f"economics_ru_cat{category}")
    os.makedirs(out, exist_ok=True)
    result["summary"].to_csv(os.path.join(out, "summary.csv"), index=False, encoding="utf-8-sig")
    result["monthly"].to_csv(os.path.join(out, "monthly.csv"), index=False, encoding="utf-8-sig")
    result["daily"].to_csv(os.path.join(out, "daily.csv"), encoding="utf-8-sig")

    sens = None
    if sensitivity_source:
        sens = sensitivity(frame, tariff, sensitivity_source)
        sens.to_csv(os.path.join(out, "sensitivity.csv"), index=False, encoding="utf-8-sig")
        result["sensitivity"] = sens
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(result["summary"], tariff, battery, sens))
    result["out_dir"] = out
    logger.info("Экономика по категории %d: %s", category, out)
    return result


def write_panel_report(run_dir: str, category: int = 4,
                       tariff_params: Optional[dict] = None,
                       out: Optional[str] = None, n_boot: int = 500) -> pd.DataFrame:
    """
    Экономика по каждому клиенту панели: суточный MPC по прогнозу лучшей
    модели и оптимум при известном будущем.

    Вместо одной цифры по одному ряду получается распределение по клиентам:
    у одних накопитель окупается, у других нет, и заказчику нужно видеть
    именно это. Накопитель для каждого клиента — того же типоразмера
    относительно его собственного пика.
    """
    params = dict(tariff_params or {})
    params["category"] = category
    tariff = RuTariff(**params)
    path = os.path.join(run_dir, "forecast_series_panel_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет {path}: ряды по клиентам пишут panel-smoke и panel-fast")
    long = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])

    rows = []
    for key, g in long.groupby("series", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        actual = g["actual"].to_numpy(dtype=np.float64)
        if actual.max() <= 0:
            continue
        battery = battery_for_load(float(actual.max()))
        ts = g["timestamp"]
        for label, sched in (
                ("MPC по прогнозу", mpc_day_ahead(g["forecast"].to_numpy(), actual, ts,
                                                  tariff, battery)),
                (UPPER_BOUND, perfect_foresight_ru(actual, ts, tariff, battery))):
            r = evaluate_schedule(sched["grid"], sched["charged"], actual, ts, tariff, battery)
            d = daily_savings(actual, sched["grid"], sched["charged"], ts, tariff, battery)
            boot = block_bootstrap_annual(d, n_boot=n_boot)
            rows.append({"series": key, "controller": label,
                         "peak_kw": float(actual.max()),
                         "forecast_MAE": float(np.mean(np.abs(g["forecast"] - actual))),
                         **r, **boot})
    table = pd.DataFrame(rows)
    out = out or os.path.join(run_dir, f"economics_ru_cat{category}_panel")
    os.makedirs(out, exist_ok=True)
    table.to_csv(os.path.join(out, "by_client.csv"), index=False, encoding="utf-8-sig")
    mpc = table[table["controller"] == "MPC по прогнозу"]
    lines = [f"# Экономика по клиентам панели, категория {category}\n",
             f"Клиентов: {len(mpc)}. Окупаемость до 10 лет по MPC: "
             f"{int((mpc['payback_years'] <= 10).sum())}; экономия положительна у "
             f"{int((mpc['annual_net_savings'] > 0).sum())}.\n",
             "| Клиент | Пик, кВт | MAE прогноза | Год по MPC, руб (90% ДИ) | "
             "Год при известном будущем, руб | Окупаемость по MPC, лет |",
             "|---|---|---|---|---|---|"]
    ub = table[table["controller"] == UPPER_BOUND].set_index("series")
    for _, r in mpc.sort_values("annual_net_savings", ascending=False).iterrows():
        payback = "—" if not np.isfinite(r["payback_years"]) else f"{r['payback_years']:.1f}"
        lines.append(
            f"| {r['series']} | {r['peak_kw']:.0f} | {r['forecast_MAE']:.2f} "
            f"| {r['annual_mean']:,.0f} ({r['annual_lo']:,.0f} … {r['annual_hi']:,.0f}) "
            f"| {ub.loc[r['series'], 'annual_net_savings']:,.0f} | {payback} |".replace(",", " "))
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run_dir")
    parser.add_argument("--category", type=int, default=4, choices=[3, 4, 5, 6])
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--tariff-json", default=None,
                        help="JSON с полями RuTariff: ставки и окна часов региона")
    parser.add_argument("--sensitivity-source", default=None,
                        help="Источник прогноза для анализа чувствительности")
    parser.add_argument("--out", default=None, help="Каталог вывода (по умолчанию "
                        "<run_dir>/economics_ru_cat<N>)")
    args = parser.parse_args(argv)

    params = {}
    if args.tariff_json:
        with open(args.tariff_json, encoding="utf-8") as f:
            params = json.load(f)
        if "peak_windows" in params:
            params["peak_windows"] = {int(k): tuple(v) for k, v in params["peak_windows"].items()}
    result = write_report(args.run_dir, args.category, args.split, params,
                          args.sensitivity_source, args.out)
    out = result["out_dir"]
    print(result["summary"][["source", "net_savings", "annual_mean", "annual_lo",
                             "annual_hi", "share_of_bound"]].to_string(index=False))
    print(f"Результаты: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
