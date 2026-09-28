# -*- coding: utf-8 -*-
"""
Демонстрационная выгрузка АСКУЭ без интернета и без обучения моделей.

Формат повторяет типичную выгрузку сбытовой компании (как в
clients/example.yaml): широкая таблица «Дата и время» × приборы учёта, шаг
30 минут, средняя мощность за интервал в кВт, метка — КОНЕЦ интервала,
разделитель «;», десятичная запятая, кодировка cp1251, столбец «Итого».

Нагрузка берётся из генератора панели проекта. В неё намеренно заложены
дефекты, которые должен найти отчёт о качестве:

  ТП-3 Ф-2 — объект подключён только через 20 суток (нули до подключения);
  ТП-7 Ф-1 — 30 часов без связи (пустые ячейки);
  ТП-7 Ф-4 — застывшие показания 8 часов подряд;
  ТП-12 Ф-1 — одно отрицательное значение (ошибка знака);
  все — две повторённые строки.

    python -m webapp.demo.build_demo [каталог]
"""

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

METERS = ["ТП-3 Ф-1", "ТП-3 Ф-2", "ТП-7 Ф-1", "ТП-7 Ф-4", "ТП-12 Ф-1", "ТП-12 Ф-2"]
DEFECTS: Dict[str, str] = {
    "ТП-3 Ф-2": "нули до подключения",
    "ТП-7 Ф-1": "пропуск 30 ч",
    "ТП-7 Ф-4": "застывшие показания",
    "ТП-12 Ф-1": "отрицательное значение",
}


def build_demo_frame(days: int = 365, seed: int = 11) -> pd.DataFrame:
    """Широкая получасовая таблица в кВт с меткой конца интервала."""
    from data.panel import generate_panel_data

    df, _ = generate_panel_data(days=days, n_cities=1, feeders_per_city=len(METERS),
                                seed=seed, start_date="2025-01-01")
    df = df.sort_values(["feeder_id", "timestamp"])
    hourly = df.pivot(index="timestamp", columns="feeder_id", values="consumption")
    hourly.columns = METERS

    # Две получасовые мощности на час; противофазный шум сохраняет энергию часа.
    rng = np.random.RandomState(seed)
    wiggle = rng.normal(0, 0.02, size=hourly.shape)
    first = hourly * (1 + wiggle)
    second = hourly * (1 - wiggle)
    ends_first = hourly.index + pd.Timedelta(minutes=30)
    ends_second = hourly.index + pd.Timedelta(minutes=60)
    half = pd.concat([first.set_axis(ends_first), second.set_axis(ends_second)]).sort_index()

    # Дефекты — в долях периода, чтобы демо любой длины содержало их все.
    start = half.index[0]
    at = lambda share: start + pd.Timedelta(days=int(days * share))
    half.loc[half.index < at(0.055), "ТП-3 Ф-2"] = 0.0
    gap = (half.index >= at(0.6)) & (half.index < at(0.6) + pd.Timedelta(hours=30))
    half.loc[gap, "ТП-7 Ф-1"] = np.nan
    flat_from = at(0.165) + pd.Timedelta(hours=9)
    flat = (half.index >= flat_from) & (half.index < flat_from + pd.Timedelta(hours=8))
    half.loc[flat, "ТП-7 Ф-4"] = round(float(half.loc[flat, "ТП-7 Ф-4"].iloc[0]), 1)
    half.loc[at(0.25) + pd.Timedelta(hours=13), "ТП-12 Ф-1"] *= -1

    half = half.round(2)
    half["Итого"] = half[METERS].sum(axis=1, min_count=1).round(2)
    dup = half.iloc[[500, 501]]
    half = pd.concat([half, dup]).sort_index(kind="stable")
    half.index.name = "Дата и время"
    return half


def write_demo_export(path: str, days: int = 365, seed: int = 11) -> str:
    frame = build_demo_frame(days, seed)
    out = frame.reset_index()
    out["Дата и время"] = out["Дата и время"].dt.strftime("%d.%m.%Y %H:%M")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out.to_csv(path, sep=";", decimal=",", index=False, encoding="cp1251")
    return path


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "webapp_workspace/demo"
    print(write_demo_export(os.path.join(target_dir, "askue_demo.csv")))
