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
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from data.connector import ExportFormat, parse_stamps, read_meter_export

SNIFF_BYTES = 64 * 1024
TOTAL_WORDS = ("итого", "сумма", "всего", "total")


def _decode(raw: bytes) -> (str, str):
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1"), "latin-1"


TIME_ONLY = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
SPACES = ("\u00a0", "\u202f", "\u2009", " ")


def _looks_like_time(values: List[str]) -> bool:
    sample = [v for v in values if v.strip()][:20]
    if not sample or all(TIME_ONLY.match(v.strip()) for v in sample):
        return False
    parsed = parse_stamps(pd.Series(sample))
    return parsed.notna().mean() > 0.8 and bool(re.search(r"\d", sample[0]))


def _looks_like_clock(values: List[str]) -> bool:
    """Столбец только со временем суток: «00:30», «1:00», «24:00»."""
    sample = [v.strip() for v in values if v.strip()][:20]
    return bool(sample) and sum(bool(TIME_ONLY.match(v)) for v in sample) / len(sample) > 0.9


def _header_row(lines: List[str], sep_guess: str) -> int:
    """
    Номер строки заголовка таблицы.

    Выгрузки АСКУЭ часто начинаются с шапки отчёта («Отчёт по точкам учёта…,
    период…»). Заголовок — первая строка, в которой столько же полей, сколько
    в большинстве строк ниже.
    """
    counts = [line.count(sep_guess) for line in lines[:200]]
    body = [c for c in counts if c > 0]
    if not body:
        return 0
    modal = max(set(body), key=body.count)
    return next((i for i, c in enumerate(counts) if c == modal), 0)


def _clean_number(v: str) -> str:
    for s in SPACES:
        v = v.replace(s, "")
    return v


def _numeric_share(values: List[str], decimal: str) -> float:
    sample = [v.strip() for v in values if v.strip()][:50]
    if not sample:
        return 0.0
    ok = 0
    for v in sample:
        v = _clean_number(v)
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
    guess = max(";,\t", key=lambda d: sum(line.count(d) for line in lines[:200]))
    skip = _header_row(lines, guess)
    lines = lines[skip:]
    sample = "\n".join(lines[:200])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        sep = guess
    rows = list(csv.reader(io.StringIO(sample), delimiter=sep))
    if not rows:
        raise ValueError("В файле нет строк")
    header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]

    decimal = "," if sep != "," and re.search(r"\d+,\d+", sample) else "."
    columns = [c.strip() for c in header]
    by_col = {c: [r[i] if i < len(r) else "" for r in body] for i, c in enumerate(columns)}

    time_col = next((c for c in columns if _looks_like_time(by_col[c])), columns[0])
    # «Дата» и «Время» отдельными столбцами: без склейки все значения суток
    # легли бы на 00:00, и расход суток сложился бы в один час.
    clock_col = next((c for c in columns if c != time_col and _looks_like_clock(by_col[c])), None)
    stamp_values = ([f"{d} {t}" for d, t in zip(by_col[time_col], by_col[clock_col])]
                    if clock_col else by_col[time_col])
    rest = [c for c in columns if c not in (time_col, clock_col)]
    empty = [c for c in rest if not any(v.strip() for v in by_col[c])]
    numeric = [c for c in rest if c not in empty and _numeric_share(by_col[c], decimal) > 0.9]
    other = [c for c in rest if c not in numeric and c not in empty]
    wide = len(numeric) >= 2 or not other
    excluded = [c for c in numeric if any(w in c.lower() for w in TOTAL_WORDS)] + empty

    fmt: Dict[str, Any] = {
        "timestamp_col": time_col, "time_col": clock_col, "skip_rows": skip,
        "wide": wide, "sep": sep, "decimal": decimal,
        "encoding": "cp1251" if encoding == "cp1251" else "utf-8-sig",
        "exclude_columns": excluded, "unit": None, "stamp_at": None,
        "interval_minutes": None,
    }
    if not wide:
        fmt["series_col"] = other[0] if other else "series"
        fmt["value_col"] = numeric[0] if numeric else "value"

    stamps = parse_stamps(pd.Series(stamp_values)).dropna()
    notes = []
    diffs = stamps.drop_duplicates().sort_values().diff().dropna()
    if len(diffs):
        minutes = int(diffs.mode().iloc[0].total_seconds() // 60)
        if minutes in (5, 10, 15, 30, 60):
            fmt["interval_minutes"] = minutes
    if excluded:
        notes.append(f"Столбцы {', '.join(excluded)} похожи на итоговые или пустые и "
                     "исключены из расчёта.")
    if skip:
        notes.append(f"Первые {skip} строки — шапка отчёта, таблица начинается ниже.")
    if clock_col:
        notes.append(f"Дата и время записаны в разных столбцах («{time_col}» и «{clock_col}»).")

    series_count = (len(numeric) - len([c for c in excluded if c in numeric])) if wide else \
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
    if interval and int(interval) not in (5, 10, 15, 30, 60):
        raise ValueError("Шаг выгрузки должен быть 5, 10, 15, 30 или 60 минут")
    return ExportFormat(
        timestamp_col=form["timestamp_col"],
        time_col=form.get("time_col") or None,
        skip_rows=int(form.get("skip_rows") or 0),
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
                          dtype=str, skiprows=fmt.skip_rows)
    except UnicodeDecodeError:
        other = "cp1251" if fmt.encoding.lower().replace("-", "") != "cp1251" else "utf-8-sig"
        fmt = replace(fmt, encoding=other)
        raw = pd.read_csv(path, sep=fmt.sep, decimal=fmt.decimal, encoding=fmt.encoding,
                          dtype=str, skiprows=fmt.skip_rows)
    raw.columns = [str(c).strip() for c in raw.columns]
    # «;» в конце строки даёт безымянный пустой столбец — это не прибор.
    raw = raw[[c for c in raw.columns if not (c.startswith("Unnamed:") and raw[c].isna().all())]]
    return raw, fmt


def load_hourly(path: str, fmt: ExportFormat) -> Dict[str, Any]:
    """
    Разбор выгрузки в почасовую длинную таблицу существующим коннектором.

    Строки, где время не разбирается (например, «Итого» в конце), отбрасываются
    до коннектора: иначе разбор дат падал бы с невнятной ошибкой. Число
    отброшенных строк и их начало возвращаются для сообщения пользователю.
    """
    from dataclasses import replace

    raw, fmt = read_raw(path, fmt)
    if fmt.timestamp_col not in raw.columns:
        raise ValueError(f"Нет столбца времени «{fmt.timestamp_col}»")
    text = raw[fmt.timestamp_col].fillna("").astype(str)
    if fmt.time_col:
        if fmt.time_col not in raw.columns:
            raise ValueError(f"Нет столбца времени «{fmt.time_col}»")
        text = text.str.strip() + " " + raw[fmt.time_col].fillna("").astype(str).str.strip()
        raw = raw.drop(columns=[fmt.time_col])
        fmt = replace(fmt, time_col=None)
    stamps = parse_stamps(text, fmt.dayfirst)
    bad = text[stamps.isna()]
    raw = raw[stamps.notna()].copy()
    if raw.empty:
        raise ValueError("Ни в одной строке не удалось разобрать дату и время")
    raw[fmt.timestamp_col] = stamps[stamps.notna()]
    value_cols = [c for c in raw.columns if c != fmt.timestamp_col
                  and (not fmt.wide or c not in fmt.exclude_columns)
                  and (fmt.wide or c == fmt.value_col)]
    for col in value_cols:
        values = raw[col].fillna("").astype(str).str.replace(r"[\s\u00a0\u202f\u2009]", "", regex=True)
        if fmt.decimal == ",":
            values = values.str.replace(",", ".", regex=False)
        raw[col] = pd.to_numeric(values, errors="coerce")
    if fmt.wide:
        raw = raw.drop(columns=[c for c in fmt.exclude_columns if c in raw.columns])
    hourly = read_meter_export(raw, fmt)
    skipped = [str(v).strip()[:40] or "пустая строка" for v in bad.tolist()[:3]]
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


def normalize_upload(path: str) -> Optional[str]:
    """
    Приводит загруженный файл к CSV; возвращает путь нового файла или None.

    Сбытовые компании и АСКУЭ часто выдают выгрузку в Excel. Первый лист
    .xlsx сохраняется как CSV с «;», и дальше работает обычный разбор —
    тот же, что для CSV. Двоичный файл другого вида отклоняется сразу, с
    понятной причиной, а не ложным советом сменить кодировку.
    """
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"PK\x03\x04"):
        try:
            sheet = pd.read_excel(path, header=None, dtype=str, sheet_name=0)
        except Exception as exc:              # noqa: BLE001
            raise ValueError(f"Не удалось прочитать файл Excel: {exc}")
        target = os.path.splitext(path)[0] + ".csv"
        sheet.dropna(how="all").to_csv(target, sep=";", index=False, header=False,
                                       encoding="utf-8-sig")
        return target
    if head.startswith(b"\xd0\xcf\x11\xe0"):
        raise ValueError("Это файл старого формата Excel (.xls) — сохраните его как .xlsx или CSV")
    if b"\x00" in head:
        raise ValueError("Файл не похож на таблицу CSV: в нём двоичные данные")
    return None
