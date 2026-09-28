# -*- coding: utf-8 -*-
"""
Определение формата выгрузки и загрузка через data.connector.

sniff_format только ПОДСКАЗЫВАЕТ значения формы: кодировку, разделитель,
десятичный знак, столбец времени, широкий или длинный формат, шаг. Две вещи
из данных надёжно не определить, и их выбирает пользователь:

  * значения — средняя мощность за интервал (кВт) или энергия (кВт·ч);
    ошибка даёт двукратное расхождение на получасовых данных;
  * метка времени — начало или конец интервала; ошибка сдвигает час
    максимума, а значит и попадание в час пика.

Разбор выгрузки делает data.connector.read_meter_export — тот же код, что у
командных утилит, поэтому веб и командная строка не расходятся.
"""

import csv
import io
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from data.connector import ExportFormat, read_meter_export

SNIFF_BYTES = 64 * 1024
TOTAL_WORDS = ("итого", "сумма", "всего", "total")


def _decode(raw: bytes) -> (str, str):
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


def _looks_like_time(values: List[str]) -> bool:
    sample = [v for v in values if v.strip()][:20]
    if not sample:
        return False
    parsed = pd.to_datetime(pd.Series(sample), dayfirst=True, errors="coerce")
    return parsed.notna().mean() > 0.8 and bool(re.search(r"\d", sample[0]))


def _numeric_share(values: List[str], decimal: str) -> float:
    sample = [v.strip() for v in values if v.strip()][:50]
    if not sample:
        return 0.0
    ok = 0
    for v in sample:
        v = v.replace(" ", "").replace(" ", "")
        if decimal == ",":
            v = v.replace(",", ".")
        try:
            float(v)
            ok += 1
        except ValueError:
            pass
    return ok / len(sample)


def sniff_format(path: str) -> Dict[str, Any]:
    """
    Подсказки для формы формата и предпросмотр первых строк.

    Возвращает {"format": {...поля ExportFormat...}, "columns": [...],
    "preview": [[...], ...], "series_count": int, "notes": [...]}.
    """
    with open(path, "rb") as f:
        raw = f.read(SNIFF_BYTES)
    text, encoding = _decode(raw)
    lines = text.splitlines()
    if len(lines) > 1 and not text.endswith("\n"):
        lines = lines[:-1]                  # последняя строка могла обрезаться
    sample = "\n".join(lines[:200])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        sep = ";" if sample.count(";") >= sample.count(",") else ","
    rows = list(csv.reader(io.StringIO(sample), delimiter=sep))
    if not rows:
        raise ValueError("В файле нет строк")
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]

    decimal = "," if sep != "," and re.search(r"\d+,\d+", sample) else "."
    columns = [c.strip() for c in header]
    by_col = {c: [r[i] if i < len(r) else "" for r in body] for i, c in enumerate(columns)}

    time_col = next((c for c in columns if _looks_like_time(by_col[c])), columns[0])
    numeric = [c for c in columns if c != time_col and _numeric_share(by_col[c], decimal) > 0.9]
    other = [c for c in columns if c not in numeric and c != time_col]
    wide = len(numeric) >= 2 or not other
    excluded = [c for c in numeric if any(w in c.lower() for w in TOTAL_WORDS)]

    fmt: Dict[str, Any] = {
        "timestamp_col": time_col, "wide": wide, "sep": sep, "decimal": decimal,
        "encoding": "cp1251" if encoding == "cp1251" else "utf-8-sig",
        "exclude_columns": excluded, "unit": None, "stamp_at": None,
        "interval_minutes": None,
    }
    if not wide:
        fmt["series_col"] = other[0] if other else "series"
        fmt["value_col"] = numeric[0] if numeric else "value"

    stamps = pd.to_datetime(pd.Series(by_col[time_col]), dayfirst=True, errors="coerce").dropna()
    notes = []
    diffs = stamps.drop_duplicates().sort_values().diff().dropna()
    if len(diffs):
        minutes = int(diffs.mode().iloc[0].total_seconds() // 60)
        if minutes in (5, 10, 15, 30, 60):
            fmt["interval_minutes"] = minutes
    if excluded:
        notes.append(f"Столбцы {', '.join(excluded)} похожи на итоговые и исключены из расчёта.")

    series_count = (len(numeric) - len(excluded)) if wide else \
        len({r[columns.index(fmt["series_col"])] for r in body if len(r) > 1}) if other else 0
    return {"format": fmt, "columns": columns, "preview": [columns] + body[:12],
            "series_count": series_count, "notes": notes}


def build_format(form: Dict[str, Any]) -> ExportFormat:
    """Значения формы → ExportFormat с проверкой обязательных решений."""
    missing = []
    if form.get("unit") not in ("kW", "kWh"):
        missing.append("единицы значений (кВт или кВт·ч)")
    if form.get("stamp_at") not in ("start", "end"):
        missing.append("к чему относится метка времени (начало или конец интервала)")
    if missing:
        raise ValueError("Укажите " + " и ".join(missing))
    interval = form.get("interval_minutes")
    return ExportFormat(
        timestamp_col=form["timestamp_col"],
        series_col=form.get("series_col") or "series",
        value_col=form.get("value_col") or "value",
        wide=bool(form.get("wide")),
        unit=form["unit"], stamp_at=form["stamp_at"],
        interval_minutes=int(interval) if interval else None,
        sep=form.get("sep") or ";", decimal=form.get("decimal") or ",",
        encoding=form.get("encoding") or "utf-8-sig",
        exclude_columns=list(form.get("exclude_columns") or []),
    )


def read_raw(path: str, fmt: ExportFormat) -> (pd.DataFrame, ExportFormat):
    """
    Читает таблицу выгрузки; при ошибке кодировки повторяет в cp1251.

    Для двух кодировок, в которых реально приходят выгрузки (utf-8 и cp1251),
    автоматический повтор надёжен, и спрашивать пользователя незачем.
    """
    from dataclasses import replace

    try:
        raw = pd.read_csv(path, sep=fmt.sep, decimal=fmt.decimal, encoding=fmt.encoding,
                          dtype=str)
    except UnicodeDecodeError:
        other = "cp1251" if fmt.encoding.lower().replace("-", "") != "cp1251" else "utf-8-sig"
        fmt = replace(fmt, encoding=other)
        raw = pd.read_csv(path, sep=fmt.sep, decimal=fmt.decimal, encoding=fmt.encoding,
                          dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw, fmt


def load_hourly(path: str, fmt: ExportFormat) -> Dict[str, Any]:
    """
    Разбор выгрузки в почасовую длинную таблицу существующим коннектором.

    Строки, где время не разбирается (например, «Итого» в конце), отбрасываются
    до коннектора: иначе разбор дат падал бы с невнятной ошибкой. Число
    отброшенных строк и их начало возвращаются для сообщения пользователю.
    """
    raw, fmt = read_raw(path, fmt)
    if fmt.timestamp_col not in raw.columns:
        raise ValueError(f"Нет столбца времени «{fmt.timestamp_col}»")
    stamps = pd.to_datetime(raw[fmt.timestamp_col], dayfirst=fmt.dayfirst, errors="coerce")
    bad = raw[stamps.isna()]
    raw = raw[stamps.notna()].copy()
    if raw.empty:
        raise ValueError("Ни в одной строке не удалось разобрать дату и время")
    raw[fmt.timestamp_col] = stamps[stamps.notna()]
    value_cols = [c for c in raw.columns if c != fmt.timestamp_col
                  and (not fmt.wide or c not in fmt.exclude_columns)
                  and (fmt.wide or c == fmt.value_col)]
    for col in value_cols:
        raw[col] = pd.to_numeric(raw[col].str.replace(" ", "", regex=False)
                                 .str.replace(" ", "", regex=False)
                                 .str.replace(",", ".", regex=False)
                                 if fmt.decimal == "," else raw[col], errors="coerce")
    if fmt.wide:
        raw = raw.drop(columns=[c for c in fmt.exclude_columns if c in raw.columns])
    hourly = read_meter_export(raw, fmt)
    skipped = [str(v)[:40] for v in bad.iloc[:, 0].tolist()[:3]]
    return {"hourly": hourly, "format": fmt, "skipped_rows": int(len(bad)),
            "skipped_examples": skipped}


def sanity_summary(hourly: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Проверка здравым смыслом: сутки с наибольшим потреблением и их максимум.

    Пользователь сравнивает это со своим представлением об объекте: ошибка
    кВт/кВт·ч на получасовых данных даёт расхождение ровно вдвое.
    """
    if hourly.empty:
        return None
    total = hourly.groupby("timestamp")["consumption"].sum()
    daily = total.groupby(total.index.normalize()).sum()
    day = daily.idxmax()
    within = total[total.index.normalize() == day]
    return {"day": day, "day_kwh": float(daily.max()),
            "peak_kwh": float(within.max()), "peak_hour": within.idxmax(),
            "mean_daily_kwh": float(daily.mean())}
