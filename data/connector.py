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
    time_col: Optional[str] = None  # время отдельным столбцом («Дата» + «Время»)
    skip_rows: int = 0              # строки шапки отчёта до заголовка таблицы


_ISO = r"^\d{4}-\d{1,2}-\d{1,2}"
_MIDNIGHT = r"\b24:00(?::00)?$"


def parse_stamps(values: pd.Series, dayfirst: bool = True) -> pd.Series:
    """
    Дата и время выгрузки → datetime; неразобранное — NaT.

    Три ловушки реальных выгрузок:
      * ISO-даты («2025-03-01 00:00») с dayfirst=True pandas разбирает по
        формату, угаданному из первой строки с днём ≤ 12, и переставляет день
        и месяц во всём столбце — поэтому ISO разбирается отдельно, строго;
      * «24:00» — конец суток, то есть 00:00 следующих;
      * смещение пояса («+03:00») отбрасывается: выгрузка АСКУЭ записана в
        местном времени, и часы пика задаются в нём же.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.dt.tz_localize(None) if getattr(values.dt, "tz", None) else values
    s = values.astype(str).str.strip()
    s = s.str.replace(r"(?<=\d)T(?=\d)", " ", regex=True)
    s = s.str.replace(r"\s*(Z|[+-]\d{2}:?\d{2})$", "", regex=True)
    midnight = s.str.contains(_MIDNIGHT, regex=True)
    s = s.str.replace(_MIDNIGHT, "00:00", regex=True)
    iso = s.str.match(_ISO)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], format="ISO8601", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], dayfirst=dayfirst, errors="coerce")
    out[midnight] = out[midnight] + pd.Timedelta(days=1)
    return out


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
                          encoding=fmt.encoding, skiprows=fmt.skip_rows)
    if fmt.timestamp_col not in raw.columns:
        raise ValueError(f"Нет колонки времени {fmt.timestamp_col!r}")
    if fmt.time_col:
        if fmt.time_col not in raw.columns:
            raise ValueError(f"Нет колонки времени {fmt.time_col!r}")
        raw[fmt.timestamp_col] = (raw[fmt.timestamp_col].astype(str).str.strip() + " "
                                  + raw[fmt.time_col].astype(str).str.strip())
        raw = raw.drop(columns=[fmt.time_col])
    stamps = parse_stamps(raw[fmt.timestamp_col], fmt.dayfirst)
    if stamps.isna().any():
        bad = raw.loc[stamps.isna(), fmt.timestamp_col].astype(str).iloc[0]
        raise ValueError(f"Не удалось разобрать дату и время: {bad!r}")
    raw[fmt.timestamp_col] = stamps

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

    # Отрицательные значения считаются ДО сложения в час: минус в одном
    # получасе гасится плюсом соседнего, и в почасовом ряду ошибка знака
    # становилась невидимой. Повторы — тоже по исходным строкам.
    long["negative"] = (long["energy"] < 0).astype(int)
    long["repeat"] = long.duplicated(["series", "timestamp"]).astype(int)
    keys = ["series", "hour"]
    # Повтор строки не должен удваивать энергию часа: энергия и число
    # интервалов считаются по строкам без повторов, а сами повторы — отдельно.
    unique = long[long["repeat"] == 0].groupby(keys)
    hourly = pd.DataFrame({
        "consumption": unique["energy"].sum(min_count=1),
        "intervals": unique["energy"].count(),
        "negatives": unique["negative"].sum(),
        "duplicates": long.groupby(keys)["repeat"].sum(),
    }).reset_index().rename(columns={"hour": "timestamp"})
    hourly["expected_intervals"] = 60 // minutes
    # Больше значений в часе, чем допускает шаг, — шаг указан неверно (5 мин
    # вместо 15): энергия часа завысилась бы в разы, а отчёт о качестве этого
    # не заметил бы. Повторы строк уже исключены, так что это не они.
    extra = hourly["intervals"] > hourly["expected_intervals"]
    if extra.any():
        worst = int(hourly.loc[extra, "intervals"].max())
        raise ValueError(f"В часе встречается {worst} значений, а при шаге {minutes} мин их "
                         f"не больше {60 // minutes} — проверьте шаг выгрузки")
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
                   outlier_z: float = 6.0, min_coverage: float = 0.95,
                   holidays: bool = True) -> pd.DataFrame:
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
        # Эталон — тот же час недели в соседних девяти неделях, а не за весь
        # ряд: летняя нагрузка отличается от годовой медианы того же часа, и
        # у промышленного фидера с узким разбросом весь сезон выглядел бы
        # выбросом (сотни ложных срабатываний на демонстрационной выгрузке).
        local = lambda v: v.rolling(9, center=True, min_periods=3).median()
        med = how.groupby(key_how).transform(local)
        mad = (how - med).abs().groupby(key_how).transform(local) * 1.4826
        # Нижняя граница разброса — 5% типичного уровня ряда: на ровном
        # участке MAD близок к нулю, и любое колебание выглядело бы выбросом.
        floor = 0.05 * float(np.nanmedian(np.abs(body))) if len(body) else 0.0
        mad = mad.clip(lower=floor if floor > 0 else None)
        z = ((how - med).abs() / mad.replace(0, np.nan)).to_numpy()
        # Праздник — не дефект данных: производство в новогодние дни падает
        # вдвое, и без этого исключения промышленный фидер получал сотни
        # «нетипичных значений», которые пользователь принял бы за ошибки учёта.
        if holidays is not False and len(how):
            from data.generator import holiday_flags
            z = np.where(holiday_flags(how.index), np.nan, z)

        # Неполный час — часть интервалов есть, часть нет. Полностью пустой
        # час уже учтён как пропущенный и второй раз не считается.
        incomplete = int(((g["intervals"] > 0)
                          & (g["intervals"] < g["expected_intervals"])).sum())
        coverage = float(present.mean())
        row = {
            "series": key,
            "start": full[0], "end": full[-1], "hours": len(full),
            "coverage": coverage,
            "missing_hours": int((~present).sum()),
            "longest_gap_hours": max(_runs(~present), default=0),
            "incomplete_hours": incomplete,
            "duplicate_rows": int(g["duplicates"].sum()),
            "negative_values": int(g["negatives"].sum()) if "negatives" in g.columns
                               else int(np.nansum(values < 0)),
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
        # Повторы строк сами по себе ряд не портят: read_meter_export считает
        # энергию часа без повторов. Они остаются в отчёте для сведения.
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
        if "negatives" in g.columns:
            s = s.where(g["negatives"].fillna(0) == 0)
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
