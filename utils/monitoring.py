# -*- coding: utf-8 -*-
"""
Ежедневный контроль качества прогноза в эксплуатации.

Модель, прошедшая отбор, со временем деградирует: меняется состав нагрузки,
подключаются новые объекты, ломаются приборы учёта. Здесь прогнозы,
выданные за прошедшие сутки, сравниваются с фактом, и по каждому ряду
считается скользящая MASE за последние window_days. Тревога поднимается,
если она выросла больше чем в degradation раз относительно эталона —
MASE того же ряда на тесте при отборе модели.

MASE выбрана потому, что она безразмерна: один порог годится и для фидера
на 50 кВт, и для подстанции на 50 МВт.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def daily_quality(forecasts: pd.DataFrame, actuals: pd.DataFrame,
                  naive_scale: Dict[str, float], reference_mase: Dict[str, float],
                  window_days: int = 7, degradation: float = 1.3,
                  min_coverage: float = 0.9) -> pd.DataFrame:
    """
    Parameters
    ----------
    forecasts : series, timestamp, forecast — выданные прогнозы.
    actuals   : series, timestamp, consumption — факт.
    naive_scale : знаменатель MASE по каждому ряду (ошибка «как вчера» на
        обучающей части, как при оценке модели).
    reference_mase : MASE ряда на тесте при отборе модели.

    Returns
    -------
    По строке на ряд и сутки: MAE и MASE за сутки, скользящая MASE, доля часов
    с фактом, статус ("норма", "деградация", "нет данных").
    """
    merged = forecasts.merge(actuals, on=["series", "timestamp"], how="left")
    merged["day"] = pd.to_datetime(merged["timestamp"]).dt.normalize()
    merged["abs_err"] = (merged["forecast"] - merged["consumption"]).abs()

    rows = []
    for (key, day), g in merged.groupby(["series", "day"], sort=True):
        coverage = float(g["consumption"].notna().mean())
        mae = float(g["abs_err"].mean()) if coverage else np.nan
        scale = naive_scale.get(key, np.nan)
        rows.append({"series": key, "day": day, "coverage": coverage, "MAE": mae,
                     "MASE": mae / scale if scale and np.isfinite(scale) else np.nan})
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["MASE_rolling"] = (table.groupby("series")["MASE"]
                                  .transform(lambda s: s.rolling(window_days, min_periods=1).mean()))

    def status(r):
        if r["coverage"] < min_coverage:
            return "нет данных"
        ref = reference_mase.get(r["series"], np.nan)
        if np.isfinite(ref) and r["MASE_rolling"] > degradation * ref:
            return "деградация"
        return "норма"

    table["reference_MASE"] = table["series"].map(reference_mase)
    table["status"] = table.apply(status, axis=1)
    return table


def alerts(table: pd.DataFrame) -> pd.DataFrame:
    """Последние сутки по каждому ряду, где статус не «норма»."""
    if table.empty:
        return table
    last = table.sort_values("day").groupby("series").tail(1)
    return last[last["status"] != "норма"].reset_index(drop=True)
