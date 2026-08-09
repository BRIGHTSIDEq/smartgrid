# -*- coding: utf-8 -*-
"""
data/uci.py — адаптер набора UCI ElectricityLoadDiagrams20112014.

ЗАЧЕМ НУЖНЫ РЕАЛЬНЫЕ ДАННЫЕ
───────────────────────────
Основной вывод работы — линейные модели и градиентный бустинг не уступают
глубоким последовательностным — получен на синтетике, которую проект
генерирует сам. Генератор складывает нагрузку из аддитивных компонент, и такое
устройство само по себе благоприятно для линейных моделей. Возражение
очевидное, и ответить на него можно только данными, которых проект не
создавал.

Набор содержит потребление 370 клиентов Португалии с 2011 по 2014 год с шагом
15 минут. Ряды сильно разномасштабны и имеют разные даты подключения — то
есть проверяют ровно те свойства конвейера, которые на сбалансированной
синтетике не проявляются.

ЧТО В ЭТИХ ДАННЫХ УСТРОЕНО НЕОЧЕВИДНО
─────────────────────────────────────
1. Метка времени — КОНЕЦ интервала. Значение «2011-01-01 00:15:00» описывает
   промежуток 00:00–00:15, а не начинающийся в 00:15. Присвоение такой метке
   часа напрямую сдвигает весь суточный профиль на 15 минут и портит
   календарные признаки.

2. Значения — мощность в кВт, а не энергия. Переход к кВт·ч требует умножения
   на длительность интервала (0.25 ч), иначе масштаб завышается вчетверо.

3. Переход на летнее время не выброшен из сетки, а закодирован дефектами. В
   мартовский день перевода час с 01:00 до 02:00 заполнен нулями у ВСЕХ
   клиентов; в октябрьский тот же час содержит потребление за два часа сразу.
   Без обработки это регулярные выбросы, повторяющиеся ежегодно.

4. Клиенты, подключённые позже начала наблюдений, имеют нули до момента
   подключения. Это не нулевое потребление, а отсутствие объекта, и обучение
   на таких участках означает подгонку под несуществующие данные.

Погоды в наборе нет. Конвейер это допускает: погодные каналы подключаются
только при наличии соответствующих колонок. Подменять отсутствующую погоду
синтезированной нельзя — это вернуло бы ту самую зависимость от собственного
генератора, ради ухода от которой набор и берётся.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_grid.data.uci")

# Официальная страница набора. Файл не скачивается автоматически: загрузка
# внешних данных — отдельное решение, которое принимает пользователь.
UCI_URL = ("https://archive.ics.uci.edu/static/public/321/"
           "electricityloaddiagrams20112014.zip")
UCI_FILENAME = "LD2011_2014.txt"

QUARTER_HOURS_PER_HOUR = 4
INTERVAL_HOURS = 0.25          # метка описывает 15-минутный промежуток


@dataclass
class UCISeriesSpec:
    """
    Характеристики ряда, выведенные ИЗ ДАННЫХ, а не заданные извне.

    Все величины считаются только по обучающему отрезку: признак, посчитанный
    по всему ряду, переносит в обучение сведения о тесте, и утечка такого рода
    не проявляется ни в одной проверке форм.
    """

    city_id: str
    feeder_id: str
    feeder_type: str
    n_hours: int
    mean_kwh: float
    static: Dict[str, float]

    def static_features(self) -> Dict[str, float]:
        return dict(self.static)


# ══════════════════════════════════════════════════════════════════════════════
# ЧТЕНИЕ ИСХОДНОГО ФАЙЛА
# ══════════════════════════════════════════════════════════════════════════════

def read_uci_raw(path: str, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Читает LD2011_2014.txt в широкую таблицу: индекс — метка, колонки — клиенты.

    Разделитель — точка с запятой, десятичный знак — запятая: числа с точкой
    прочитались бы как строки, и вся таблица молча стала бы объектной.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл {path} не найден. Набор не скачивается автоматически: "
            f"загрузите архив с {UCI_URL} и распакуйте {UCI_FILENAME}."
        )

    positions = None
    if usecols is not None:
        # Позиции вычисляются по заголовку, а не предполагаются: клиенты в
        # файле идут в произвольном порядке, и срез по первым N колонкам дал
        # бы не тот набор рядов, который запрошен.
        header = list(pd.read_csv(path, sep=";", nrows=0).columns)
        missing = [c for c in usecols if c not in header]
        if missing:
            raise KeyError(f"В файле нет колонок: {', '.join(map(str, missing))}")
        positions = [0] + [header.index(c) for c in usecols]

    frame = pd.read_csv(
        path, sep=";", decimal=",", index_col=0,
        usecols=positions, low_memory=False,
    )
    frame.index = pd.to_datetime(frame.index)
    frame = frame.astype(np.float64)
    frame.index.name = "interval_end"

    logger.info("UCI: прочитано %d меток × %d клиентов (%s … %s)",
                len(frame), frame.shape[1], frame.index[0], frame.index[-1])
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# ПЕРЕХОД К ЧАСОВОМУ ШАГУ
# ══════════════════════════════════════════════════════════════════════════════

def quarter_hours_to_hourly(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Сводит 15-минутную мощность (кВт) к часовой энергии (кВт·ч).

    Метка интервала переносится на его НАЧАЛО вычитанием 15 минут. Иначе
    последний отсчёт часа («01:00:00», описывающий 00:45–01:00) попал бы в
    следующий час, и суточный профиль сместился бы у всех рядов одинаково —
    ошибка, незаметная в метриках, но искажающая связь с календарём.
    """
    starts = wide.index - pd.Timedelta(minutes=60 // QUARTER_HOURS_PER_HOUR)
    energy = wide.mul(INTERVAL_HOURS)          # кВт × 0.25 ч = кВт·ч
    energy.index = starts.floor("h")
    hourly = energy.groupby(level=0).sum()
    hourly.index.name = "timestamp"

    expected = pd.date_range(hourly.index[0], hourly.index[-1], freq="h")
    if len(hourly) != len(expected):
        missing = len(expected) - len(hourly)
        logger.warning("UCI: в часовой сетке не хватает %d отсчётов, добавлены как NaN",
                       missing)
        hourly = hourly.reindex(expected)
        hourly.index.name = "timestamp"

    return hourly


def _dst_transition_days(years: Sequence[int]) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
    """Последние воскресенья марта и октября — дни перевода часов в ЕС."""
    march, october = [], []
    for year in years:
        for month, bucket in ((3, march), (10, october)):
            days = pd.date_range(f"{year}-{month:02d}-01",
                                 periods=31, freq="D")
            days = days[days.month == month]
            bucket.append(days[days.dayofweek == 6][-1])
    return march, october


def repair_dst_artifacts(hourly: pd.DataFrame,
                         october_factor: float = 2.0) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Устраняет дефекты дней перевода часов.

    В марте час 01:00 заполнен нулями у всех клиентов, в октябре он содержит
    потребление за два часа. Правка применяется ТОЛЬКО если данные
    подтверждают ожидаемую картину: слепая коррекция по календарю испортила бы
    набор, если бы его когда-то уже привели в порядок.

    Мартовский час восстанавливается средним соседних часов, октябрьский
    делится пополам. Оба случая возвращаются отдельным списком: это правки
    данных, и они обязаны попасть в отчёт, а не остаться внутри загрузчика.
    """
    fixed = hourly.copy()
    report: List[Dict[str, Any]] = []
    years = sorted({int(t.year) for t in hourly.index})
    march, october = _dst_transition_days(years)

    daily_mean = float(np.nanmean(hourly.to_numpy()))

    for day in march:
        stamp = day + pd.Timedelta(hours=1)
        if stamp not in fixed.index:
            continue
        row = fixed.loc[stamp]
        if float(np.nanmax(np.abs(row.to_numpy()))) > 1e-9:
            report.append({"дата": str(stamp), "тип": "март",
                           "действие": "пропущена — ожидаемых нулей нет"})
            continue
        before = fixed.index.get_loc(stamp) - 1
        after = fixed.index.get_loc(stamp) + 1
        if before < 0 or after >= len(fixed):
            continue
        fixed.loc[stamp] = (fixed.iloc[before].to_numpy()
                            + fixed.iloc[after].to_numpy()) / 2.0
        report.append({"дата": str(stamp), "тип": "март",
                       "действие": "нулевой час заменён средним соседних"})

    for day in october:
        stamp = day + pd.Timedelta(hours=1)
        if stamp not in fixed.index:
            continue
        row_sum = float(np.nansum(fixed.loc[stamp].to_numpy()))
        neighbours = []
        idx = fixed.index.get_loc(stamp)
        for shift in (-1, 1):
            j = idx + shift
            if 0 <= j < len(fixed):
                neighbours.append(float(np.nansum(fixed.iloc[j].to_numpy())))
        reference = float(np.mean(neighbours)) if neighbours else daily_mean
        if reference <= 0 or row_sum < 1.5 * reference:
            report.append({"дата": str(stamp), "тип": "октябрь",
                           "действие": "пропущена — удвоения не обнаружено"})
            continue
        fixed.loc[stamp] = fixed.loc[stamp].to_numpy() / october_factor
        report.append({"дата": str(stamp), "тип": "октябрь",
                       "действие": "сдвоенный час разделён пополам"})

    for item in report:
        logger.info("UCI DST %s: %s — %s", item["тип"], item["дата"], item["действие"])
    return fixed, report


# ══════════════════════════════════════════════════════════════════════════════
# ОТБОР РЯДОВ
# ══════════════════════════════════════════════════════════════════════════════

def select_active_series(hourly: pd.DataFrame, start: str, end: str,
                         min_nonzero_share: float = 0.98,
                         max_series: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Оставляет клиентов, действующих на всём запрошенном окне.

    Нули в начале ряда означают, что клиент ещё не подключён, а не что он
    потреблял ноль. Обучение на таком участке — подгонка под отсутствующий
    объект, поэтому ряды с недостаточной долей ненулевых часов отбрасываются
    целиком, а не «зашиваются».

    Границы окна общие для всех рядов: конвейер делит выборку по времени
    одинаково для всей панели, и ряды разной длины нарушили бы это разбиение.
    """
    window = hourly.loc[str(start):str(end)]
    if window.empty:
        raise ValueError(f"Окно {start} … {end} не пересекается с данными "
                         f"({hourly.index[0]} … {hourly.index[-1]})")

    kept, rejected = [], []
    for col in window.columns:
        values = window[col].to_numpy(np.float64)
        share = float(np.mean(np.isfinite(values) & (values > 0.0)))
        if share >= min_nonzero_share:
            kept.append(col)
        else:
            rejected.append({"клиент": col, "доля ненулевых": round(share, 4)})

    if not kept:
        raise ValueError(
            f"Ни один ряд не активен на всём окне {start} … {end}: "
            f"порог доли ненулевых часов {min_nonzero_share}")

    # Отбор крупнейших, а не первых по алфавиту: номер клиента в наборе
    # произволен, и срез «первых N» дал бы случайную по составу панель.
    if max_series is not None and len(kept) > max_series:
        order = window[kept].mean().sort_values(ascending=False)
        kept = list(order.index[:max_series])

    logger.info("UCI: отобрано %d рядов из %d (отсеяно %d по активности)",
                len(kept), window.shape[1], len(rejected))
    return window[kept], rejected


# ══════════════════════════════════════════════════════════════════════════════
# СТАТИЧЕСКИЕ ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def build_static_features(series: pd.Series, train_end: int) -> Dict[str, float]:
    """
    Выводит постоянные характеристики ряда по ОБУЧАЮЩЕМУ отрезку.

    Ограничение обучающим отрезком принципиально: средний уровень или форма
    профиля, посчитанные по всему ряду, содержат сведения о тестовом периоде.
    Такая утечка не нарушает ни одной проверки размерностей и обнаруживается
    только по неправдоподобно высокому качеству.
    """
    train = series.iloc[:train_end]
    values = train.to_numpy(np.float64)
    mean = float(np.nanmean(values))
    if not np.isfinite(mean) or mean <= 0:
        mean = 1e-9

    hours = train.index.hour.to_numpy()
    weekday = train.index.dayofweek.to_numpy()

    profile = np.array([np.nanmean(values[hours == h]) if np.any(hours == h) else mean
                        for h in range(24)])
    peak_hour = int(np.argmax(profile))
    angle = 2.0 * np.pi * peak_hour / 24.0

    weekend = values[weekday >= 5]
    workday = values[weekday < 5]
    weekend_ratio = (float(np.nanmean(weekend)) / float(np.nanmean(workday))
                     if weekend.size and workday.size and np.nanmean(workday) > 0 else 1.0)

    night = values[(hours >= 0) & (hours < 6)]

    return {
        "static_log_mean_kwh": float(np.log1p(mean)),
        "static_load_factor": float(mean / max(float(np.nanmax(values)), 1e-9)),
        "static_cv": float(np.nanstd(values) / mean),
        "static_weekend_ratio": float(weekend_ratio),
        "static_night_share": float(np.nanmean(night) / mean) if night.size else 1.0,
        "static_peak_hour_sin": float(np.sin(angle)),
        "static_peak_hour_cos": float(np.cos(angle)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ПРАЗДНИКИ ПОРТУГАЛИИ
# ══════════════════════════════════════════════════════════════════════════════

def _easter_sunday(year: int) -> pd.Timestamp:
    """Пасха по григорианскому календарю (алгоритм Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def portuguese_holidays(years: Sequence[int]) -> set:
    """
    Государственные праздники Португалии, включая переходящие.

    Календарь берётся португальский, а не российский: набор описывает
    потребление в Португалии, и чужие праздничные дни пометили бы обычные
    рабочие сутки как выходные, внеся систематическую ошибку в признак.
    """
    fixed = [(1, 1), (4, 25), (5, 1), (6, 10), (8, 15),
             (10, 5), (11, 1), (12, 1), (12, 8), (12, 25)]
    days = set()
    for year in years:
        for month, day in fixed:
            days.add(pd.Timestamp(year=year, month=month, day=day).date())
        easter = _easter_sunday(year)
        days.add((easter - pd.Timedelta(days=2)).date())      # Страстная пятница
        days.add(easter.date())
        days.add((easter + pd.Timedelta(days=60)).date())     # Тело Господне
        days.add((easter - pd.Timedelta(days=47)).date())     # Карнавал
    return days


# ══════════════════════════════════════════════════════════════════════════════
# СБОРКА ПАНЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def uci_to_panel(path: str, start: str = "2012-01-01", end: str = "2014-12-31",
                 max_series: Optional[int] = 32, train_ratio: float = 0.70,
                 min_nonzero_share: float = 0.98,
                 usecols: Optional[Sequence[str]] = None,
                 ) -> Tuple[pd.DataFrame, List[UCISeriesSpec], Dict[str, Any]]:
    """
    Приводит набор UCI к тому же виду, что и синтетическая панель.

    Возвращает (df, specs, report). Формат df совпадает с выходом
    generate_panel_data по обязательным колонкам, поэтому дальше используется
    тот же конвейер без ветвлений: любое расхождение в обработке синтетики и
    реальных данных сделало бы их результаты несравнимыми.

    Погодных колонок нет — они и не добавляются. Предобработка подключает
    погодные каналы по факту наличия колонок.
    """
    wide = read_uci_raw(path, usecols=usecols)
    hourly = quarter_hours_to_hourly(wide)
    hourly, dst_report = repair_dst_artifacts(hourly)
    window, rejected = select_active_series(
        hourly, start, end, min_nonzero_share=min_nonzero_share,
        max_series=max_series)

    n_hours = len(window)
    train_end = int(n_hours * train_ratio)
    if train_end < 24:
        raise ValueError(f"Обучающий отрезок слишком короток: {train_end} ч")

    holidays = portuguese_holidays(sorted({int(t.year) for t in window.index}))
    index = window.index
    hour = index.hour.to_numpy().astype(np.int8)
    weekday = index.dayofweek.to_numpy().astype(np.int8)
    is_holiday = np.array([t.date() in holidays for t in index], dtype=np.int8)
    day_of_year = index.dayofyear.to_numpy().astype(np.int16)

    frames: List[pd.DataFrame] = []
    specs: List[UCISeriesSpec] = []

    for col in window.columns:
        series = window[col]
        static = build_static_features(series, train_end)
        values = series.to_numpy(np.float64)

        # Пропуски заполняются значением сутками ранее, а не нулём и не
        # средним: суточный ход — сильнейшая закономерность в этих рядах, и
        # любая другая подстановка внесла бы искусственный провал.
        gaps = ~np.isfinite(values)
        if gaps.any():
            for pos in np.where(gaps)[0]:
                donor = pos - 24
                values[pos] = values[donor] if donor >= 0 and np.isfinite(values[donor]) \
                    else float(np.nanmedian(values))

        frame = pd.DataFrame({
            "timestamp": index,
            "city_id": "PT",
            "feeder_id": str(col),
            "feeder_type": "unknown",
            "consumption": values.astype(np.float32),
            "hour": hour, "weekday": weekday,
            "is_weekend": (weekday >= 5).astype(np.int8),
            "is_holiday": is_holiday, "day_of_year": day_of_year,
        })
        for name, value in static.items():
            frame[name] = np.float32(value)
        frames.append(frame)

        specs.append(UCISeriesSpec(
            city_id="PT", feeder_id=str(col), feeder_type="unknown",
            n_hours=n_hours, mean_kwh=float(np.nanmean(values)), static=static))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["city_id", "feeder_id", "timestamp"]).reset_index(drop=True)

    report = {
        "источник": os.path.basename(path),
        "окно": f"{index[0]} … {index[-1]}",
        "часов на ряд": n_hours,
        "рядов отобрано": len(specs),
        "рядов отсеяно": len(rejected),
        "правки DST": dst_report,
        "пропусков заполнено": int(sum(
            int(np.sum(~np.isfinite(window[c].to_numpy()))) for c in window.columns)),
        "разброс средних, раз": float(
            max(s.mean_kwh for s in specs) / max(min(s.mean_kwh for s in specs), 1e-9)),
    }
    logger.info("UCI: панель собрана — %d рядов × %d ч, разброс средних %.1f×",
                len(specs), n_hours, report["разброс средних, раз"])
    return df, specs, report
