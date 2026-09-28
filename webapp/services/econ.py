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
from optimization.tariffs_ru import DEFAULT_PEAK_WINDOWS, RuTariff
from webapp.services.errors import DataError

TOTAL = "Сумма выбранных приборов"
MIN_EVAL_DAYS = 7
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
                   + glob.glob(os.path.join(ROOT, "optimization", "*.py"))
                   + glob.glob(os.path.join(ROOT, "data", "connector.py"))
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
    gaps = frame[frame.drop(columns="timestamp").isna().any(axis=1)]["timestamp"]
    if len(gaps):
        days = sorted({t.strftime("%d.%m.%Y") for t in gaps})
        shown = ", ".join(days[:4]) + (f" и ещё {len(days) - 4}" if len(days) > 4 else "")
        raise DataError("В периоде оценки есть пропуски", f"сутки: {shown}",
                        "Исключите прибор с длинным пропуском или выберите другие "
                        "приборы: короткие пропуски до 3 ч заполняются, длинные — нет.")
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# ТАРИФ И НАКОПИТЕЛЬ ИЗ ФОРМЫ
# ══════════════════════════════════════════════════════════════════════════════

TARIFF_FIELDS = ("energy_price", "gen_capacity_rate", "net_capacity_rate",
                 "net_energy_rate", "net_single_rate", "deviation_up_rate",
                 "deviation_down_rate")


def _number(value: Any) -> float:
    return float(str(value).replace(" ", "").replace(" ", "")
                 .replace(" ", "").replace(",", "."))


def tariff_from_form(form: Dict[str, Any]) -> RuTariff:
    """Параметры формы → RuTariff; неверные значения — понятная ошибка."""
    try:
        base = RuTariff(category=int(form.get("category", 4)))
        fields: Dict[str, Any] = {}
        for name in TARIFF_FIELDS:
            value = form.get(name)
            if value not in (None, ""):
                fields[name] = _number(value)
                if fields[name] < 0:
                    raise ValueError(f"ставка не может быть отрицательной: {value}")
        hours = form.get("peak_hours")
        if hours not in (None, ""):
            fields["peak_windows"] = {m: hours_to_window(hours) for m in range(1, 13)}
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


def is_example_tariff(tariff: RuTariff) -> bool:
    default = RuTariff(category=tariff.category)
    same = all(getattr(tariff, f) == getattr(default, f) for f in TARIFF_FIELDS)
    return same and tariff.peak_windows == DEFAULT_PEAK_WINDOWS


# ══════════════════════════════════════════════════════════════════════════════
# РАСЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

def _row(summary: pd.DataFrame, name: str, capex: float) -> Dict[str, Any]:
    r = summary[summary["source"] == name].iloc[0].to_dict()
    lo, hi = r["annual_lo"], r["annual_hi"]
    r["payback_lo"] = capex / hi if hi > 0 else float("inf")
    r["payback_hi"] = capex / lo if lo > 0 else float("inf")
    return r


def run_economics(frame: pd.DataFrame, tariff: RuTariff, capex_per_kwh: float = DEFAULT_CAPEX,
                  power_share: float = 0.20, n_boot: int = 2000,
                  job: Any = None) -> Dict[str, Any]:
    """Экономия по простым прогнозам, точный прогноз, потолок и расписание."""
    actual = frame[ACTUAL_COLUMN].to_numpy(dtype=np.float64)
    battery = battery_for_load(float(actual.max()), capex_per_kwh, power_share / 0.20)
    if job is not None:
        job.stage, job.progress = "планирование заряда по каждому прогнозу", 0.1
    result = evaluate_sources(frame, tariff, battery, n_boot=n_boot)
    summary = result["summary"]
    simple = summary[~summary["source"].isin([ORACLE, UPPER_BOUND])]
    best = simple.sort_values("net_savings", ascending=False).iloc[0]["source"]

    if job is not None:
        job.stage, job.progress = "расписание накопителя для графика", 0.85
    sched = mpc_day_ahead(frame[best].to_numpy(dtype=np.float64), actual,
                          frame["timestamp"], tariff, battery)
    n_days = int(summary["n_days"].max())
    months = sorted({t.month for t in frame["timestamp"]})
    monthly = result["monthly"]

    b = _row(summary, best, battery.capex_rub)
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
                   "window": list(tariff.peak_windows[1])},
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
        "winter_included": any(m in (11, 12, 1, 2) for m in months),
        "period": [frame["timestamp"].iloc[0].strftime("%d.%m.%Y"),
                   frame["timestamp"].iloc[-1].strftime("%d.%m.%Y")],
    }


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
    ("Ёмкость накопителя", "capacity", (0.5, 2.0), "×{:g}"),
    ("Цена накопителя", "capex", (12_000.0, 24_000.0), "{:,.0f} ₽/кВт·ч"),
    ("Ставки мощности", "rates", (0.7, 1.3), "{:+.0%}"),
)


def what_if_points() -> List[Dict[str, Any]]:
    """Центр и по две точки на фактор: семь расчётов вместо сетки из 27."""
    points = [{"factor": "Как сейчас", "kind": "center", "value": 1.0}]
    for label, kind, values, _ in WHAT_IF:
        points += [{"factor": label, "kind": kind, "value": v} for v in values]
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
    points = points or what_if_points()
    one = frame[["timestamp", source, ACTUAL_COLUMN]]
    peak = float(frame[ACTUAL_COLUMN].max())
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
# ПАСПОРТ РАСЧЁТА
# ══════════════════════════════════════════════════════════════════════════════

def passport(result: Dict[str, Any], meters: List[str], dropped_hours: int) -> List[Dict[str, str]]:
    """
    «На чём основан расчёт»: только сработавшие условия, каждое — простыми словами.

    Уровни: «Факт» (ваши данные), «Допущение» (что мы предположили),
    «Нужно ваше значение» (что стоит заменить своим).
    """
    rows = [{"level": "Факт", "kind": "fact",
             "text": f"Ваши данные за {result['n_days']} суток ({result['period'][0]}–"
                     f"{result['period'][1]}), приборов: {len(meters)}."
                     + (f" {dropped_hours} ч, где был не каждый прибор, в сумму не вошли."
                        if dropped_hours else "")}]
    if result["n_days"] < 365:
        text = f"Экономия за год пересчитана по {result['n_days']} суткам"
        if not result["winter_included"]:
            text += "; зима в них не вошла, а зимой нагрузка и экономия обычно выше"
        rows.append({"level": "Допущение", "kind": "assume", "text": text + "."})
    rows.append({"level": "Допущение", "kind": "assume",
                 "text": "Час пика региона мы оценили по вашей нагрузке. Точный час публикует "
                         "биржа (АТС) после окончания месяца; если он придётся на другое время, "
                         "экономия на покупке мощности будет меньше."})
    if result["tariff"]["example"]:
        rows.append({"level": "Нужно ваше значение", "kind": "need",
                     "text": "Ставки и часы пика — примерные. Введите свои из счёта, и расчёт "
                             "станет расчётом для вашего объекта."})
    if result["n_days"] < 21:
        rows.append({"level": "Допущение", "kind": "assume",
                     "text": "Меньше трёх недель данных — диапазон ориентировочный."})
    return rows
