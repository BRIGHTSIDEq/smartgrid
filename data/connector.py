# -*- coding: utf-8 -*-
"""
Загрузка выгрузок приборов учёта и отчёт о качестве данных.

Выгрузки АСКУЭ и сбытовых компаний различаются форматом, и каждое различие
незаметно портит прогноз, если его не учесть. Те же ловушки уже встретились
на наборе UCI (data/uci.py):

  * метка времени обозначает КОНЕЦ интервала или его начало;
  * значение — мощность в кВт или энергия в кВт·ч за интервал;
  * шаг — 15 или 30 минут, а не час;
  * «широкий» формат (колонка на прибор) или «длинный» (строка на значение);
  * нули до подключения объекта, пропуски связи, повторы строк, застывшие
    показания, отрицательные значения при ошибке знака.

read_meter_export приводит выгрузку к длинному почасовому виду, а
quality_report перечисляет найденные проблемы по каждому ряду: решать,
исправлять их или исключать ряд, должен человек, который знает объект.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ExportFormat:
    """
    Описание выгрузки.

    wide=True — колонка времени и по колонке на прибор; иначе длинный формат
    с колонками series_col и value_col.
    """

    timestamp_col: str = "timestamp"
    series_col: str = "series"
    value_col: str = "value"
    wide: bool = False
    unit: str = "kWh"               # "kWh" — энергия за интервал, "kW" — мощность
    stamp_at: str = "end"           # "end" — метка конца интервала, "start" — начала
    interval_minutes: Optional[int] = None   # None — определить по данным
    sep: str = ","
    decimal: str = "."
    encoding: str = "utf-8-sig"
    dayfirst: bool = True
    exclude_columns: List[str] = field(default_factory=list)


def _infer_interval(stamps: pd.Series) -> int:
    diffs = pd.to_datetime(stamps).sort_values().diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Не удалось определить шаг выгрузки: меньше двух отметок времени")
    minutes = int(diffs.mode().iloc[0].total_seconds() // 60)
    if minutes not in (1, 5, 10, 15, 30, 60):
        raise ValueError(f"Необычный шаг выгрузки {minutes} мин — задайте interval_minutes явно")
    return minutes


def read_meter_export(path_or_frame, fmt: Optional[ExportFormat] = None) -> pd.DataFrame:
    """
    Выгрузка → длинная таблица series, timestamp (начало часа), consumption (кВт·ч),
    intervals (сколько интервалов попало в час).

    Час с неполным числом интервалов не выбрасывается, а помечается: так
    пропуск связи виден в отчёте, а не превращается в «низкое потребление».
    """
    fmt = fmt or ExportFormat()
    if isinstance(path_or_frame, pd.DataFrame):
        raw = path_or_frame.copy()
    else:
        raw = pd.read_csv(path_or_frame, sep=fmt.sep, decimal=fmt.decimal,
                          encoding=fmt.encoding)
    if fmt.timestamp_col not in raw.columns:
        raise ValueError(f"Нет колонки времени {fmt.timestamp_col!r}")
    raw[fmt.timestamp_col] = pd.to_datetime(raw[fmt.timestamp_col], dayfirst=fmt.dayfirst)

    if fmt.wide:
        value_cols = [c for c in raw.columns
                      if c != fmt.timestamp_col and c not in fmt.exclude_columns]
        long = raw.melt(id_vars=[fmt.timestamp_col], value_vars=value_cols,
                        var_name="series", value_name="value")
    else:
        for col in (fmt.series_col, fmt.value_col):
            if col not in raw.columns:
                raise ValueError(f"Нет колонки {col!r}")
        long = raw.rename(columns={fmt.series_col: "series", fmt.value_col: "value"})
    long = long.rename(columns={fmt.timestamp_col: "timestamp"})
    long["series"] = long["series"].astype(str)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    minutes = fmt.interval_minutes or _infer_interval(long["timestamp"])
    step = pd.Timedelta(minutes=minutes)
    start = long["timestamp"] - step if fmt.stamp_at == "end" else long["timestamp"]
    energy = long["value"] * (minutes / 60.0) if fmt.unit.lower() == "kw" else long["value"]
    long = long.assign(hour=start.dt.floor("h"), energy=energy)

    hourly = (long.groupby(["series", "hour"])
                  .agg(consumption=("energy", lambda v: v.sum(min_count=1)),
                       intervals=("energy", "count"),
                       duplicates=("timestamp", lambda t: int(t.duplicated().sum())))
                  .reset_index().rename(columns={"hour": "timestamp"}))
    hourly["expected_intervals"] = 60 // minutes
    return hourly


def _runs(mask: np.ndarray) -> List[int]:
    """Длины серий подряд идущих True."""
    runs, n = [], 0
    for m in mask:
        if m:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def quality_report(hourly: pd.DataFrame, flat_hours: int = 6, zero_hours: int = 24,
                   outlier_z: float = 6.0, min_coverage: float = 0.95) -> pd.DataFrame:
    """
    Проблемы данных по каждому ряду.

    Колонки: покрытие часов, пропущенные часы, неполные часы, повторы,
    отрицательные значения, серии нулей и застывших показаний, выбросы
    (устойчивая оценка относительно медианы того же часа недели), нули в
    начале ряда (объект ещё не подключён) и итоговое заключение.
    """
    rows = []
    for key, g in hourly.groupby("series", sort=True):
        g = g.sort_values("timestamp")
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        s = g.set_index("timestamp")["consumption"].reindex(full)
        present = s.notna().to_numpy()
        values = s.to_numpy(dtype=np.float64)

        lead_zeros = 0
        for v in values:
            if v == 0 or np.isnan(v):
                lead_zeros += 1
            else:
                break
        body = values[lead_zeros:]
        zero_runs = _runs(np.nan_to_num(body, nan=1.0) == 0)
        same = np.r_[False, np.diff(body) == 0] & (body != 0) & ~np.isnan(body)
        flat_runs = [r + 1 for r in _runs(same)]

        how = pd.Series(body, index=full[lead_zeros:])
        key_how = np.asarray(how.index.dayofweek * 24 + how.index.hour)
        med = how.groupby(key_how).transform("median")
        mad = (how - med).abs().groupby(key_how).transform("median") * 1.4826
        z = ((how - med).abs() / mad.replace(0, np.nan)).to_numpy()

        incomplete = int((g["intervals"] < g["expected_intervals"]).sum())
        coverage = float(present.mean())
        row = {
            "series": key,
            "start": full[0], "end": full[-1], "hours": len(full),
            "coverage": coverage,
            "missing_hours": int((~present).sum()),
            "longest_gap_hours": max(_runs(~present), default=0),
            "incomplete_hours": incomplete,
            "duplicate_rows": int(g["duplicates"].sum()),
            "negative_values": int(np.nansum(values < 0)),
            "leading_zero_hours": lead_zeros,
            "zero_runs_over_limit": sum(r >= zero_hours for r in zero_runs),
            "flat_runs_over_limit": sum(r >= flat_hours for r in flat_runs),
            "outliers": int(np.nansum(z > outlier_z)),
        }
        problems = []
        if coverage < min_coverage:
            problems.append(f"покрытие {100 * coverage:.1f}%")
        if row["longest_gap_hours"] > 24:
            problems.append(f"пропуск {row['longest_gap_hours']} ч подряд")
        if row["negative_values"]:
            problems.append("отрицательные значения")
        if row["duplicate_rows"]:
            problems.append("повторы строк")
        if row["zero_runs_over_limit"]:
            problems.append("длительные нули")
        if row["flat_runs_over_limit"]:
            problems.append("застывшие показания")
        row["verdict"] = "пригоден" if not problems else "проверить: " + ", ".join(problems)
        rows.append(row)
    return pd.DataFrame(rows)


def clean_hourly(hourly: pd.DataFrame, max_fill_hours: int = 3) -> pd.DataFrame:
    """
    Минимальная очистка перед обучением.

    Нули до подключения объекта отрезаются, отрицательные значения и неполные
    часы становятся пропусками, пропуски до max_fill_hours заполняются
    линейной интерполяцией. Длинные пропуски остаются пропусками: заполнять
    сутки выдуманной нагрузкой нельзя, такой ряд нужно обрезать или исключить.
    """
    out = []
    for key, g in hourly.groupby("series", sort=True):
        g = g.sort_values("timestamp")
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq="h")
        g = g.set_index("timestamp").reindex(full)
        s = g["consumption"].where(g["intervals"] >= g["expected_intervals"])
        s = s.where(s >= 0)
        nonzero = np.flatnonzero(s.fillna(0).to_numpy() > 0)
        if not len(nonzero):
            continue
        s = s.iloc[nonzero[0]:]
        # interpolate(limit=k) заполнил бы первые k часов и ДЛИННОГО пропуска;
        # здесь заполняются только пропуски, целиком не длиннее k часов.
        gap = s.isna()
        run_id = (gap != gap.shift()).cumsum()
        run_len = gap.groupby(run_id).transform("sum")
        short = gap & (run_len <= max_fill_hours)
        s = s.where(~short, s.interpolate(limit_area="inside"))
        out.append(pd.DataFrame({"series": key, "timestamp": s.index,
                                 "consumption": s.to_numpy()}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["series", "timestamp", "consumption"])
