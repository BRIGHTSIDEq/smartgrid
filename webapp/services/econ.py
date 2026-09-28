# -*- coding: utf-8 -*-
"""
Экономика накопителя по загруженной выгрузке.

Модель проекта прогнозирует только ряды, на которых обучалась, поэтому для
выгрузки нового клиента прогноз строится простыми способами из
analysis.value_calculator: «как вчера», «как неделю назад», профиль часа
недели. Заказчик видит, сколько накопитель даёт уже при таком прогнозе и
сколько добавил бы точный.

Весь расчёт — вызовы analysis.economics.evaluate_sources и
optimization.controllers; здесь только подготовка кадра и сборка результата
для страницы. «Что если» считается той же функцией с изменённым тарифом или
накопителем, поэтому центральная точка совпадает с итогом до рубля.
(analysis.economics.sensitivity для этого не годится: она не передаёт план
категорий 5–6, и центр расходился бы с итогом.)
"""

import glob
import hashlib
import os
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.economics import (
    ACTUAL_COLUMN, ORACLE, UPPER_BOUND, battery_for_load, evaluate_sources,
)
from analysis.value_calculator import PERFECT, simple_forecasts
from optimization.controllers import mpc_day_ahead
from optimization.tariffs_ru import DEFAULT_PEAK_WINDOWS, RuTariff, monthly_bill
from webapp.services.errors import DataError

TOTAL = "Сумма выбранных приборов"
MIN_EVAL_DAYS = 7
MAX_DROPPED_SHARE = 0.10      # сколько суток с пропусками можно исключить без отказа


class Stopped(Exception):
    """Расчёт остановлен пользователем."""


def _check_stop(job: Any) -> None:
    if job is not None and getattr(job, "stop", None) is not None and job.stop.is_set():
        raise Stopped()
MIN_ANNUAL_DAYS = 14
DEFAULT_CAPEX = 16_000.0

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def code_fingerprint() -> str:
    """
    Отпечаток расчётного кода для ключа кэша.

    Без него после любой правки экономики кэш на защите показал бы старые
    цифры, посчитанные прежним кодом.
    """
    files = sorted(glob.glob(os.path.join(ROOT, "analysis", "economics.py"))
                   + glob.glob(os.path.join(ROOT, "analysis", "value_calculator.py"))
                   + glob.glob(os.path.join(ROOT, "optimization", "*.py"))
                   + glob.glob(os.path.join(ROOT, "data", "connector.py"))
                   + glob.glob(os.path.join(ROOT, "webapp", "formatting.py"))
                   + glob.glob(os.path.join(ROOT, "webapp", "services", "econ.py")))
    h = hashlib.sha1()
    for path in files:
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
# ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════

def series_for(clean: pd.DataFrame, meters: Sequence[str]) -> Dict[str, Any]:
    """
    Почасовой ряд по выбранным приборам.

    Сумма берётся только по часам, где есть все выбранные приборы: пропуск
    одного прибора иначе выглядел бы как провал нагрузки всего объекта.
    Число выпавших часов возвращается для паспорта расчёта.
    """
    meters = [m for m in meters if m]
    if not meters:
        raise DataError("Не выбраны приборы", "", "Отметьте хотя бы один прибор.")
    part = clean[clean["series"].isin(meters)]
    if part.empty:
        raise DataError("Приборы не найдены", ", ".join(meters))
    wide = part.pivot_table(index="timestamp", columns="series", values="consumption")
    total = wide.sum(axis=1, min_count=wide.shape[1]).sort_index().asfreq("h")
    # Период — общий для всех выбранных приборов: прибор, подключённый позже
    # или закончившийся раньше, иначе давал бы «пропуск» в начале или конце.
    first, last = total.first_valid_index(), total.last_valid_index()
    if first is None:
        raise DataError("У выбранных приборов нет общих часов", ", ".join(meters),
                        "Выберите приборы с пересекающимися периодами.")
    total = total.loc[first:last]
    return {"series": total, "dropped_hours": int(total.isna().sum()),
            "meters": list(wide.columns)}


def evaluation_days(series: pd.Series) -> int:
    """
    Сколько последних суток оценивать.

    Первые четыре недели — разгон: на них строится профиль часа недели. Если
    данных меньше шести недель, разгон — две недели.
    """
    total_days = int(series.dropna().index.normalize().nunique())
    warmup = 28 if total_days >= 42 else 14
    return total_days - warmup


def economics_frame(series: pd.Series, eval_days: Optional[int] = None) -> pd.DataFrame:
    """
    Кадр в формате forecast_series_test.csv: timestamp, прогнозы, факт.

    Пропуски в оцениваемом периоде — отказ с датами: линейная задача
    накопителя на пропусках не решается, а молча выкинуть часы значило бы
    исказить суточные максимумы.
    """
    eval_days = evaluation_days(series) if eval_days is None else eval_days
    if eval_days < MIN_EVAL_DAYS:
        raise DataError("Слишком короткая выгрузка",
                        f"после разгона остаётся {max(eval_days, 0)} суток",
                        f"Нужно не меньше {MIN_EVAL_DAYS} суток сверх двух–четырёх "
                        "недель истории.")
    try:
        forecasts = simple_forecasts(series, eval_days)
    except ValueError as exc:
        raise DataError("Недостаточно истории", str(exc))
    actual = forecasts.pop(PERFECT)
    frame = pd.DataFrame({"timestamp": actual.index,
                          **{k: v.to_numpy() for k, v in forecasts.items()},
                          ACTUAL_COLUMN: actual.to_numpy()})
    # Пропуск в истории портит прогноз «как вчера» на следующие сутки и «как
    # неделю назад» — через неделю, хотя сами эти сутки в порядке. Там, где
    # опереться не на что, берётся профиль часа недели — так поступил бы и
    # диспетчер. Исключаются только сутки без самого факта.
    profile = "Профиль часа недели"
    filled = 0
    if profile in frame.columns:
        for col in forecasts:
            if col != profile:
                gap = frame[col].isna() & frame[profile].notna()
                filled += int(gap.sum())
                frame.loc[gap, col] = frame.loc[gap, profile]
    day = frame["timestamp"].dt.normalize()
    bad_days = sorted(day[frame.drop(columns="timestamp").isna().any(axis=1)].unique())
    total_days = day.nunique()
    if bad_days and len(bad_days) > MAX_DROPPED_SHARE * total_days:
        shown = ", ".join(pd.Timestamp(d).strftime("%d.%m.%Y") for d in bad_days[:4])
        shown += f" и ещё {len(bad_days) - 4}" if len(bad_days) > 4 else ""
        raise DataError("В периоде оценки много суток с пропусками",
                        f"{len(bad_days)} из {total_days}: {shown}",
                        "Исключите прибор с длинным пропуском или выберите другие "
                        "приборы: короткие пропуски до 3 ч заполняются, длинные — нет.")
    # Немногие сутки с пропуском (обрыв связи, сутки после него в прогнозе
    # «как вчера») исключаются целиком: задача накопителя решается по
    # суткам, а неполные месяцы счёт и так взвешивает по числу суток.
    frame = frame[~day.isin(bad_days)].reset_index(drop=True)
    frame.attrs["dropped_days"] = [pd.Timestamp(d).strftime("%d.%m.%Y") for d in bad_days]
    frame.attrs["filled_hours"] = filled
    frame.attrs["warmup_from"] = series.first_valid_index().strftime("%d.%m.%Y")
    return frame


def selection_days(total_days: int) -> int:
    """
    Сколько первых суток периода отдать под выбор простого прогноза.

    Прогноз, лучший на том же отрезке, где потом считается экономия, получает
    фору: среди трёх кандидатов выбирается тот, кому повезло именно здесь.
    Поэтому выбор делается на первых сутках, а экономия — на остальных.
    На коротких выгрузках отрезать нечего: выбор остаётся на всём периоде,
    и паспорт расчёта это называет.
    """
    if total_days >= 84:
        return 28
    if total_days >= 28:
        return 14
    return 0


def selection_split(frame: pd.DataFrame):
    """(отрезок выбора или None, отрезок оценки) — по целым суткам."""
    days = frame["timestamp"].dt.normalize()
    unique = days.drop_duplicates().reset_index(drop=True)
    k = selection_days(len(unique))
    if k == 0:
        return None, frame
    cut = unique.iloc[k]
    return (frame[days < cut].reset_index(drop=True),
            frame[days >= cut].reset_index(drop=True))


def forecast_sources(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in ("timestamp", ACTUAL_COLUMN)]


# ══════════════════════════════════════════════════════════════════════════════
# ТАРИФ И НАКОПИТЕЛЬ ИЗ ФОРМЫ
# ══════════════════════════════════════════════════════════════════════════════

TARIFF_FIELDS = ("energy_price", "gen_capacity_rate", "net_capacity_rate",
                 "net_energy_rate", "net_single_rate", "deviation_up_rate",
                 "deviation_down_rate")


def _number(value: Any) -> float:
    return float(str(value).replace(" ", "").replace(" ", "")
                 .replace(" ", "").replace(",", "."))


TARIFF_LABELS = {
    "energy_price": "электроэнергия", "gen_capacity_rate": "покупка мощности",
    "net_capacity_rate": "передача: содержание сетей", "net_energy_rate": "передача: оплата потерь",
    "net_single_rate": "передача", "deviation_up_rate": "взяли больше плана",
    "deviation_down_rate": "взяли меньше плана",
}


def tariff_fields(category: int) -> List[str]:
    """Ставки, которые участвуют в счёте этой категории."""
    fields = ["energy_price", "gen_capacity_rate"]
    fields += ["net_capacity_rate", "net_energy_rate"] if category in (4, 6) else ["net_single_rate"]
    if category in (5, 6):
        fields += ["deviation_up_rate", "deviation_down_rate"]
    return fields


def tariff_from_form(form: Dict[str, Any], strict: bool = False) -> RuTariff:
    """
    Параметры формы → RuTariff; неверные значения — понятная ошибка.

    strict — для расчёта из формы: пустая ставка своей категории или пустые
    часы пика — ошибка, а не молчаливая подстановка примера (пользователь,
    стёрший поле, мог иметь в виду ноль).
    """
    try:
        base = RuTariff(category=int(form.get("category", 4)))
        needed = tariff_fields(base.category)
        fields: Dict[str, Any] = {}
        for name in TARIFF_FIELDS:
            value = form.get(name)
            if value in (None, ""):
                if strict and name in needed:
                    raise ValueError(f"заполните ставку «{TARIFF_LABELS[name]}»")
                continue
            fields[name] = _number(value)
            if fields[name] < 0:
                raise ValueError(f"ставка не может быть отрицательной: {value}")
        hours = form.get("peak_hours")
        if hours not in (None, ""):
            fields["peak_windows"] = {m: hours_to_window(hours) for m in range(1, 13)}
        elif strict:
            raise ValueError("не выбраны плановые часы пиковой нагрузки")
        return replace(base, **fields)
    except (TypeError, ValueError) as exc:
        raise DataError("Неверные параметры тарифа", str(exc))


def hours_to_window(hours: Any) -> tuple:
    """
    «8,9,…,20» из шкалы суток → окно [8, 21).

    Окно одно на сутки: RuTariff задаёт плановые часы непрерывным
    промежутком. Разрыв в выделении — ошибка, а не молчаливое склеивание.
    """
    values = sorted({int(h) for h in str(hours).split(",") if str(h).strip() != ""})
    if not values:
        raise ValueError("не выбраны плановые часы пиковой нагрузки")
    if values != list(range(values[0], values[-1] + 1)):
        raise ValueError("плановые часы должны идти подряд")
    if values[0] < 0 or values[-1] > 23:
        raise ValueError("час вне суток")
    return (values[0], values[-1] + 1)


def example_fields(tariff: RuTariff) -> List[str]:
    """
    Какие значения тарифа остались примерными — поимённо.

    Прежде проверка была «всё или ничего»: одна изменённая ставка снимала
    пометку со всего тарифа, хотя остальные оставались примерными.
    """
    default = RuTariff(category=tariff.category)
    out = [TARIFF_LABELS[f] for f in tariff_fields(tariff.category)
           if getattr(tariff, f) == getattr(default, f)]
    if tariff.peak_windows == DEFAULT_PEAK_WINDOWS:
        out.append("часы пика")
    return out


def is_example_tariff(tariff: RuTariff) -> bool:
    return bool(example_fields(tariff))


# ══════════════════════════════════════════════════════════════════════════════
# РАСЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

def _row(summary: pd.DataFrame, name: str, capex: float) -> Dict[str, Any]:
    r = summary[summary["source"] == name].iloc[0].to_dict()
    lo, hi = r["annual_lo"], r["annual_hi"]
    r["payback_lo"] = capex / hi if hi > 0 else float("inf")
    r["payback_hi"] = capex / lo if lo > 0 else float("inf")
    return r


def choose_source(sel: pd.DataFrame, tariff: RuTariff, battery: Any) -> Dict[str, Any]:
    """Лучший простой прогноз по чистой экономии на отрезке выбора."""
    names = forecast_sources(sel)
    res = evaluate_sources(sel, tariff, battery, sources=names, n_boot=1)["summary"]
    ranked = res[res["source"].isin(names)].sort_values("net_savings", ascending=False)
    return {"best": ranked.iloc[0]["source"],
            "ranking": [{"source": r["source"], "net_savings": float(r["net_savings"]),
                         "mae": float(r["forecast_MAE"])} for _, r in ranked.iterrows()]}


def peak_hour_risk(frame: pd.DataFrame, tariff: RuTariff, grid: np.ndarray) -> Dict[str, Any]:
    """
    Экономия на покупке мощности при другом часе пика региона, в год.

    Покупная мощность оплачивается по нагрузке в час пика региона, а его
    публикует АТС уже после месяца. Расчёт берёт час по нагрузке самого
    объекта — это самое выгодное для накопителя допущение. Здесь тот же счёт
    пересчитывается для каждого планового часа по очереди: «в среднем» —
    если час пика региона равновероятно любой из них, «в худшем» — если
    каждый раз самый неудачный. Остальные части счёта от часа пика не зависят.
    """
    from data.generator import holiday_flags
    ts = frame["timestamp"]
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype=np.float64)
    holidays = holiday_flags(pd.DatetimeIndex(ts))
    scale = 8760.0 / len(actual)

    def saving(region: np.ndarray) -> float:
        before = monthly_bill(actual, ts, tariff, region_load=region, holidays=holidays)
        after = monthly_bill(grid, ts, tariff, region_load=region, holidays=holidays)
        return float(before["gen_capacity_cost"].sum() - after["gen_capacity_cost"].sum()) * scale

    hour = ts.dt.hour.to_numpy()
    hours = sorted({h for m in range(1, 13) for h in range(*tariff.peak_windows[m])})
    # Ряд «субъекта» с максимумом ровно в час h: окна СО по месяцам могут
    # различаться, и тогда выбирается ближайший к h час внутри окна.
    by_hour = {h: saving(-np.abs(hour - h).astype(np.float64)) for h in hours}
    worst = min(by_hour, key=by_hour.get)
    return {"own": saving(actual), "mean": float(np.mean(list(by_hour.values()))),
            "worst": by_hour[worst], "worst_hour": int(worst),
            "by_hour": [{"hour": int(h), "annual": v} for h, v in by_hour.items()]}


def run_economics(frame: pd.DataFrame, tariff: RuTariff, capex_per_kwh: float = DEFAULT_CAPEX,
                  power_share: float = 0.20, n_boot: int = 2000,
                  job: Any = None) -> Dict[str, Any]:
    """
    Экономия по простым прогнозам, точный прогноз, потолок и расписание.

    Накопитель подбирается по максимуму всей выгрузки, прогноз для плана
    заряда — на отрезке выбора, все цифры страницы — на отрезке оценки.
    """
    battery = battery_for_load(float(frame[ACTUAL_COLUMN].max()), capex_per_kwh,
                               power_share / 0.20)
    attrs = dict(frame.attrs)
    sel, frame = selection_split(frame)
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype=np.float64)
    if job is not None:
        job.stage, job.progress = "выбор прогноза для плана заряда", 0.05
    chosen = choose_source(sel, tariff, battery) if sel is not None else None
    _check_stop(job)
    if job is not None:
        job.stage, job.progress = "планирование заряда по каждому прогнозу", 0.15
    result = evaluate_sources(frame, tariff, battery, n_boot=n_boot)
    _check_stop(job)
    summary = result["summary"]
    if chosen is not None:
        best = chosen["best"]
    else:
        simple = summary[~summary["source"].isin([ORACLE, UPPER_BOUND])]
        best = simple.sort_values("net_savings", ascending=False).iloc[0]["source"]

    if job is not None:
        job.stage, job.progress = "расписание накопителя для графика", 0.8
    sched = mpc_day_ahead(frame[best].to_numpy(dtype=np.float64), actual,
                          frame["timestamp"], tariff, battery)
    _check_stop(job)
    if job is not None:
        job.stage, job.progress = "другой час пика региона", 0.9
    risk = peak_hour_risk(frame, tariff, sched["grid"])
    n_days = int(summary["n_days"].max())
    months = sorted({t.month for t in frame["timestamp"]})
    monthly = result["monthly"]

    b = _row(summary, best, battery.capex_rub)
    for key in ("mean", "worst"):
        risk[f"annual_{key}"] = b["annual_mean"] - (risk["own"] - risk[key])
    parts = [("Мощность (покупка)", b["saved_gen_capacity_cost"]),
             ("Сетевая мощность", b["saved_net_capacity_cost"]),
             ("Энергия и передача", b["saved_energy_cost"] + b["saved_net_energy_cost"])]
    if tariff.with_plan:
        parts.append(("Отклонения от плана", b["saved_deviation_cost"]))
    if not tariff.two_rate_network:
        parts = [x for x in parts if x[0] != "Сетевая мощность"]
    parts += [("Износ батареи", -b["degradation"]), ("Обслуживание", -b["om_cost"])]
    return {
        "parts": [{"label": k, "value": float(v)} for k, v in parts],
        "parts_max": max([abs(float(v)) for _, v in parts] + [1.0]),
        "battery": {"capacity_kwh": battery.capacity, "power_kw": battery.max_power,
                    "capex_rub": battery.capex_rub, "capex_per_kwh": capex_per_kwh,
                    "power_share": power_share},
        "tariff": {"category": tariff.category, "example": is_example_tariff(tariff),
                   "example_fields": example_fields(tariff),
                   "window": list(tariff.peak_windows[1])},
        "dropped_days": attrs.get("dropped_days", []),
        "filled_hours": attrs.get("filled_hours", 0),
        "warmup_from": attrs.get("warmup_from"),
        "best_source": best,
        "best": _row(summary, best, battery.capex_rub),
        "sources": [_row(summary, s, battery.capex_rub) for s in summary["source"]],
        "monthly": _monthly(monthly[monthly["source"] == best], frame),
        "schedule": {
            "t": (frame["timestamp"].astype("int64") // 10**9).tolist(),
            "load": np.round(actual, 2).tolist(),
            "grid": np.round(sched["grid"], 2).tolist(),
            "soc": np.round(sched["soc"], 2).tolist(),
            "window": list(tariff.peak_windows[1]),
        },
        "n_days": n_days,
        "annual_shown": n_days >= MIN_ANNUAL_DAYS,
        "peak_risk": risk,
        "selection": ({"days": int(sel["timestamp"].dt.normalize().nunique()),
                       "period": [sel["timestamp"].iloc[0].strftime("%d.%m.%Y"),
                                  sel["timestamp"].iloc[-1].strftime("%d.%m.%Y")],
                       "ranking": chosen["ranking"]} if chosen is not None else None),
        "profile": _average_day(frame, sched["grid"]),
        "winter_included": any(m in (11, 12, 1, 2) for m in months),
        "period": [frame["timestamp"].iloc[0].strftime("%d.%m.%Y"),
                   frame["timestamp"].iloc[-1].strftime("%d.%m.%Y")],
    }


def _average_day(frame: pd.DataFrame, grid: np.ndarray) -> Dict[str, Any]:
    """
    Средние рабочие сутки: нагрузка и взятое из сети по часам.

    Годовой график на 8 тысяч точек не читается с одного взгляда; средние
    будни показывают главное — где накопитель срезает и где заряжается.
    """
    from data.generator import holiday_flags
    ts = pd.DatetimeIndex(frame["timestamp"])
    working = (ts.dayofweek < 5) & ~np.asarray(holiday_flags(ts), dtype=bool)
    part = pd.DataFrame({"hour": ts.hour, "load": frame[ACTUAL_COLUMN].to_numpy(),
                         "grid": grid})[working]
    by_hour = part.groupby("hour")[["load", "grid"]].mean().reindex(range(24))
    return {"load": np.round(by_hour["load"].to_numpy(), 2).tolist(),
            "grid": np.round(by_hour["grid"].to_numpy(), 2).tolist()}


def _monthly(rows: pd.DataFrame, frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Экономия по месяцам с пометкой неполных: иначе короткий месяц выглядит провалом."""
    days = frame["timestamp"].dt.normalize().groupby(frame["timestamp"].dt.to_period("M")).nunique()
    out = []
    for _, r in rows.iterrows():
        period = pd.Period(r["month"])
        covered = int(days.get(period, 0))
        out.append({"month": r["month"], "gross_savings": float(r["gross_savings"]),
                    "days": covered, "partial": covered < period.days_in_month})
    return out


WHAT_IF = (
    ("Размер накопителя", "capacity", (0.5, 2.0)),
    ("Цена накопителя", "capex", (0.75, 1.5)),
    ("Ставки мощности", "rates", (0.7, 1.3)),
)


def what_if_points(capex_per_kwh: float = DEFAULT_CAPEX) -> List[Dict[str, Any]]:
    """
    Центр и по две точки на фактор: семь расчётов вместо сетки из 27.

    Цена накопителя — доли от цены пользователя: при заданных 12 000 ₽
    прежняя точка «12 000» совпадала с центром, и расчёт тратился зря.
    """
    points = [{"factor": "Как сейчас", "kind": "center", "value": 1.0}]
    for label, kind, values in WHAT_IF:
        points += [{"factor": label, "kind": kind,
                    "value": v * capex_per_kwh if kind == "capex" else v} for v in values]
    return points


def run_what_if(frame: pd.DataFrame, tariff: RuTariff, source: str,
                capex_per_kwh: float = DEFAULT_CAPEX, power_share: float = 0.20,
                points: Optional[List[Dict[str, Any]]] = None, n_boot: int = 500,
                job: Any = None, stop: Any = None) -> List[Dict[str, Any]]:
    """
    Годовая экономия и окупаемость при изменении одного фактора.

    Каждая точка — evaluate_sources на кадре с одним источником прогноза;
    центральная точка тем самым равна итогу страницы.
    """
    points = points or what_if_points(capex_per_kwh)
    peak = float(frame[ACTUAL_COLUMN].max())
    one = selection_split(frame)[1][["timestamp", source, ACTUAL_COLUMN]]
    out = []
    for i, p in enumerate(points):
        if stop is not None and stop.is_set():
            break
        if job is not None:
            job.stage, job.progress = f"вариант {i + 1} из {len(points)}", i / len(points)
        t, capex, scale = tariff, capex_per_kwh, power_share / 0.20
        if p["kind"] == "capacity":
            scale *= p["value"]
        elif p["kind"] == "capex":
            capex = p["value"]
        elif p["kind"] == "rates":
            t = replace(tariff, gen_capacity_rate=tariff.gen_capacity_rate * p["value"],
                        net_capacity_rate=tariff.net_capacity_rate * p["value"])
        battery = battery_for_load(peak, capex, scale)
        res = evaluate_sources(one, t, battery, sources=[source], n_boot=n_boot)
        r = _row(res["summary"], source, battery.capex_rub)
        out.append({**p, "annual": r["annual_mean"], "annual_lo": r["annual_lo"],
                    "annual_hi": r["annual_hi"], "net": r["net_savings"],
                    "payback": r["payback_years"], "capacity_kwh": battery.capacity})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ФИНАНСЫ
# ══════════════════════════════════════════════════════════════════════════════

FINANCE_DEFAULTS = {"discount": 0.12, "life": 12, "growth": 0.05}


def cash_flows(annual: float, capex: float, discount: float, life: int,
               growth: float) -> Dict[str, Any]:
    """
    Дисконтированные потоки, NPV, IRR и дисконтированная окупаемость.

    Экономия первого года — годовая оценка расчёта; дальше она растёт вместе
    со ставками (индексация тарифов). Износ батареи уже вычтен из экономии
    как стоимость циклов, поэтому отдельной замены в потоках нет.
    """
    life = max(1, int(life))
    flows = [annual * (1 + growth) ** (t - 1) for t in range(1, life + 1)]
    cumulative, total = [-capex], -capex
    for t, f in enumerate(flows, start=1):
        total += f / (1 + discount) ** t
        cumulative.append(total)
    payback = None
    for t in range(1, len(cumulative)):
        if cumulative[t] >= 0:
            prev = cumulative[t - 1]
            payback = t - 1 + (-prev) / (cumulative[t] - prev)
            break

    def npv_at(r: float) -> float:
        return -capex + sum(f / (1 + r) ** t for t, f in enumerate(flows, start=1))

    irr = None
    if sum(flows) > capex:
        lo, hi = 0.0, 10.0
        if npv_at(hi) < 0:
            for _ in range(80):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if npv_at(mid) > 0 else (lo, mid)
            irr = (lo + hi) / 2
    return {"npv": cumulative[-1], "irr": irr, "payback": payback, "cumulative": cumulative}


def finance(result: Dict[str, Any], discount: float, life: int, growth: float) -> Dict[str, Any]:
    """NPV и IRR для оценки и обоих краёв разброса годовой экономии."""
    b, capex = result["best"], result["battery"]["capex_rub"]
    out = {k: cash_flows(b[f"annual_{k}"], capex, discount, life, growth)
           for k in ("mean", "lo", "hi")}
    out["params"] = {"discount": discount, "life": int(life), "growth": growth}
    return out


def finance_params(form: Dict[str, Any]) -> Dict[str, Any]:
    """Поля формы в процентах и годах → доли; пустое поле — значение по умолчанию."""
    out = dict(FINANCE_DEFAULTS)
    try:
        for key, scale in (("discount", 100.0), ("growth", 100.0), ("life", 1.0)):
            value = form.get(key)
            if value not in (None, ""):
                out[key] = _number(value) / scale
    except ValueError as exc:
        raise DataError("Неверные финансовые параметры", str(exc))
    out["life"] = int(round(out["life"]))
    if not (0 <= out["discount"] < 1 and 1 <= out["life"] <= 30 and -0.5 < out["growth"] < 1):
        raise DataError("Неверные финансовые параметры",
                        "ставка 0–100 %, срок службы 1–30 лет, индексация от −50 до 100 %")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# ПАСПОРТ РАСЧЁТА
# ══════════════════════════════════════════════════════════════════════════════

def passport(result: Dict[str, Any], meters: List[str], dropped_hours: int) -> List[Dict[str, str]]:
    """
    «На чём основан расчёт»: только сработавшие условия, каждое — простыми словами.

    Уровни: «Факт» (ваши данные), «Допущение» (что мы предположили),
    «Нужно ваше значение» (что стоит заменить своим).
    """
    rows = [{"level": "Факт", "kind": "fact",
             "text": f"Экономия посчитана по {result['n_days']} суткам ({result['period'][0]}–"
                     f"{result['period'][1]}), приборов: {len(meters)}."
                     + (f" {dropped_hours} ч, где был не каждый прибор, в сумму не вошли."
                        if dropped_hours else "")}]
    if result.get("warmup_from"):
        rows.append({"level": "Факт", "kind": "fact",
                     "text": f"Данные с {result['warmup_from']} до начала периода ушли на историю "
                             "для прогноза: профилю часа недели нужны прошлые недели."})
    dropped = result.get("dropped_days") or []
    if dropped:
        shown = ", ".join(dropped[:5]) + (f" и ещё {len(dropped) - 5}" if len(dropped) > 5 else "")
        rows.append({"level": "Факт", "kind": "fact",
                     "text": f"Исключены сутки с пропусками данных: {shown}."})
    if result.get("filled_hours"):
        rows.append({"level": "Факт", "kind": "fact",
                     "text": f"Где из-за пропуска в истории не было прогноза «как вчера» или «как "
                             f"неделю назад» ({result['filled_hours']} ч), взят профиль часа недели."})
    if result["n_days"] < 365:
        text = f"Экономия за год пересчитана по {result['n_days']} суткам"
        if not result["winter_included"]:
            text += "; зима в них не вошла, а зимой нагрузка и экономия обычно выше"
        rows.append({"level": "Допущение", "kind": "assume", "text": text + "."})
    sel = result.get("selection")
    if sel:
        rows.append({"level": "Факт", "kind": "fact",
                     "text": f"Прогноз для плана заряда выбран на первых {sel['days']} сутках "
                             f"({sel['period'][0]}–{sel['period'][1]}), экономия посчитана на "
                             "остальных — выбор не подогнан под результат."})
    else:
        rows.append({"level": "Допущение", "kind": "assume",
                     "text": "Данных мало, поэтому прогноз выбран на том же периоде, где "
                             "посчитана экономия: цифра может быть немного завышена."})
    risk = result.get("peak_risk")
    text = ("Час пика региона мы оценили по вашей нагрузке. Точный час публикует "
            "биржа (АТС) после окончания месяца")
    if risk and result.get("annual_shown"):
        from webapp.formatting import rub
        text += (f". Если он придётся на любой из плановых часов, экономия в год — около "
                 f"{rub(risk['annual_mean'])}, в самом неудачном случае — {rub(risk['annual_worst'])}.")
    else:
        text += "; если он придётся на другое время, экономия на покупке мощности будет меньше."
    rows.append({"level": "Допущение", "kind": "assume", "text": text})
    rows.append({"level": "Допущение", "kind": "assume",
                 "text": "Цена энергии одна на все часы. Выигрыш на разнице цен по часам не "
                         "учтён, поэтому оценка осторожная."})
    if result["tariff"]["category"] in (5, 6):
        rows.append({"level": "Допущение", "kind": "assume",
                     "text": "План на сутки — прогноз нагрузки вместе с планом заряда. Накопитель "
                             "его выполняет, но не гасит ошибку прогноза нагрузки, поэтому ставки "
                             "отклонений почти не меняют экономию, хотя входят в счёт."})
    if result["tariff"]["example"]:
        fields = result["tariff"].get("example_fields") or []
        what = ("Ставки и часы пика — примерные" if len(fields) > 2 or not fields
                else "Примерные значения: " + ", ".join(fields))
        rows.append({"level": "Нужно ваше значение", "kind": "need",
                     "text": f"{what}. Введите свои из счёта, и расчёт станет расчётом для "
                             "вашего объекта."})
    if result["n_days"] < 21:
        rows.append({"level": "Допущение", "kind": "assume",
                     "text": f"Экономия оценена всего по {result['n_days']} суткам — разброс "
                             "ориентировочный. Для надёжной цифры нужны данные от трёх месяцев."})
    return rows
