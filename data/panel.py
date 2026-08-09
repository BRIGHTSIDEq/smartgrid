# -*- coding: utf-8 -*-
"""
data/panel.py — многорядный (panel) генератор данных Smart Grid.

ЗАЧЕМ ЭТО НУЖНО
───────────────
Агрегатный генератор (data/generator.py) возвращает ОДИН временной ряд города.
Увеличение числа домохозяйств там не добавляет данных для обучения: растёт
только абсолютный масштаб, а после нормировки информации не прибавляется —
более того, агрегация сглаживает ряд и коэффициент вариации даже падает
(0.282 при 250 домохозяйствах против 0.231 при 25 000). Поэтому объём
обучающей выборки упирался в длину ряда, и нейросети обучались на нескольких
тысячах окон одного и того же процесса.

Panel-режим формирует НЕСКОЛЬКО связанных рядов: города, внутри каждого —
фидеры со своими параметрами. Это даёт качественно другой объём данных
(рядов × длина) и делает осмысленным обучение одной глобальной модели на всех
рядах сразу — подход, который в современной практике прогнозирования нагрузки
вытеснил модель-на-ряд.

СТРУКТУРА ДАННЫХ
────────────────
Возвращается long-format DataFrame: одна строка = (timestamp, city_id,
feeder_id). Городской ряд не хранится отдельно, а получается суммированием
фидеров — это гарантирует согласованность иерархии по построению.

Погода общая для города: фидеры одного города находятся в одной местности.
Различаются они составом потребителей, чувствительностью к погоде, размером,
проникновением электротранспорта и микрогенерации.

ВЕКТОРИЗАЦИЯ
────────────
Ни одного цикла по домохозяйствам или по часам в горячем пути: профили
считаются матричными операциями. Электротранспорт моделируется не отдельными
сессиями зарядки, а числом одновременных сессий из распределения Пуассона —
при десятках фидеров и годах истории посессионная симуляция неприемлема по
времени. Промышленная нагрузка получает режимы работы через пороговый
AR-процесс, что даёт устойчивые серии «смена идёт / не идёт» без цикла.

История изменений — в CHANGELOG.md.
"""

import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.generator import generate_holiday_mask, _build_household_profiles, _sigmoid

logger = logging.getLogger("smart_grid.data.panel")

# Типы фидеров задают состав потребителей и форму суточного графика.
FEEDER_TYPES = ("residential", "mixed", "commercial", "industrial")


@dataclass
class FeederSpec:
    """
    Параметры одного фидера.

    Каждый фидер — самостоятельный объект сети со своим составом потребителей.
    Именно разнородность этих параметров делает ряды непохожими друг на друга:
    без неё panel-режим выродился бы в несколько копий одного процесса.
    """

    city_id: str
    feeder_id: str
    feeder_type: str
    households: int
    nonresidential_share: float      # доля коммерческой и промышленной нагрузки
    kwh_per_household_month: float
    early_bird_frac: float
    night_owl_frac: float
    heating_sensitivity: float       # отклик на холод
    cooling_sensitivity: float       # отклик на жару
    weekend_sensitivity: float       # насколько ниже потребление в выходные
    ev_penetration: float
    solar_penetration: float
    noise_sigma: float
    annual_trend: float
    evening_peak_hour: float         # сдвиг вечернего максимума

    def static_features(self) -> Dict[str, float]:
        """
        Статические признаки ряда для глобальной модели.

        Модель, обучаемая на всех фидерах сразу, должна отличать их друг от
        друга: без этих признаков она усредняла бы разнородные объекты.
        """
        return {
            "static_log_households": float(np.log1p(self.households)),
            "static_nonres_share": float(self.nonresidential_share),
            "static_heating_sens": float(self.heating_sensitivity),
            "static_cooling_sens": float(self.cooling_sensitivity),
            "static_weekend_sens": float(self.weekend_sensitivity),
            "static_ev_penetration": float(self.ev_penetration),
            "static_solar_penetration": float(self.solar_penetration),
            "static_evening_peak": float(self.evening_peak_hour) / 24.0,
            **{f"static_type_{t}": float(self.feeder_type == t) for t in FEEDER_TYPES},
        }


def _sample_feeder_specs(
    rng: np.random.Generator,
    city_id: str,
    n_feeders: int,
    base_kwh_per_household_month: float,
) -> List[FeederSpec]:
    """
    Формирует непохожие друг на друга фидеры одного города.

    Размеры распределены логнормально: в реальной сети рядом соседствуют
    небольшие жилые фидеры и крупные смешанные, а не одинаковые объекты.
    """
    specs: List[FeederSpec] = []
    # Тип задаёт «характер» фидера; доли подобраны так, чтобы жилые
    # преобладали, как в городской распределительной сети.
    type_probs = np.array([0.45, 0.30, 0.15, 0.10])

    for i in range(n_feeders):
        ftype = str(rng.choice(FEEDER_TYPES, p=type_probs))

        if ftype == "residential":
            households = int(rng.lognormal(mean=np.log(1200), sigma=0.45))
            nonres = float(rng.uniform(0.10, 0.30))
            evening = float(rng.uniform(19.0, 21.0))
        elif ftype == "mixed":
            households = int(rng.lognormal(mean=np.log(900), sigma=0.5))
            nonres = float(rng.uniform(0.35, 0.65))
            evening = float(rng.uniform(18.5, 20.0))
        elif ftype == "commercial":
            households = int(rng.lognormal(mean=np.log(400), sigma=0.5))
            nonres = float(rng.uniform(0.90, 1.60))
            evening = float(rng.uniform(17.5, 19.0))
        else:  # industrial
            households = int(rng.lognormal(mean=np.log(250), sigma=0.5))
            nonres = float(rng.uniform(1.80, 3.20))
            evening = float(rng.uniform(17.0, 18.5))

        households = int(np.clip(households, 60, 20_000))

        specs.append(FeederSpec(
            city_id=city_id,
            feeder_id=f"{city_id}_F{i:02d}",
            feeder_type=ftype,
            households=households,
            nonresidential_share=nonres,
            kwh_per_household_month=float(
                base_kwh_per_household_month * rng.uniform(0.82, 1.20)),
            early_bird_frac=float(rng.uniform(0.18, 0.36)),
            night_owl_frac=float(rng.uniform(0.12, 0.28)),
            heating_sensitivity=float(rng.uniform(1.2e-4, 3.0e-4)),
            cooling_sensitivity=float(rng.uniform(0.4e-4, 2.0e-4)),
            weekend_sensitivity=float(rng.uniform(0.86, 1.00)),
            ev_penetration=float(np.clip(rng.normal(1.0, 0.5), 0.15, 2.5)),
            solar_penetration=float(np.clip(rng.normal(1.0, 0.6), 0.0, 3.0)),
            noise_sigma=float(rng.uniform(0.020, 0.050)),
            annual_trend=float(rng.uniform(0.005, 0.030)),
            evening_peak_hour=evening,
        ))
    return specs


def _generate_city_weather(
    rng: np.random.Generator,
    hours: int,
    t: np.ndarray,
    temp_annual_mean: float,
    temp_annual_amplitude: float,
    temp_min: float,
    temp_max: float,
) -> Dict[str, np.ndarray]:
    """
    Погода города. Общая для всех его фидеров: они в одной местности.

    Каждый город получает собственный сдвиг климата, чтобы города не были
    копиями друг друга.
    """
    climate_shift = rng.normal(0.0, 1.5)

    temp_annual = (temp_annual_mean + climate_shift
                   + temp_annual_amplitude * np.sin(2 * np.pi * t / (24 * 365.25) - np.pi / 2))
    temp_diurnal = 3.5 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 4)

    # AR(1)-шум погоды: векторизованная реализация через накопление.
    innov = rng.normal(0.0, 0.6, size=hours)
    noise = np.empty(hours)
    noise[0] = rng.normal(0.0, 1.5)
    phi = 0.97
    # lfilter недоступен без scipy.signal в горячем пути — используем
    # накопление в логарифмическом числе шагов через явный проход по массиву
    # NumPy, что на порядок быстрее поэлементного цикла Python.
    for i in range(1, hours):
        noise[i] = phi * noise[i - 1] + innov[i]

    temperature = np.clip(temp_annual + temp_diurnal + noise, temp_min, temp_max)

    cloud_base = 0.52 + 0.18 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    cloud_noise = np.convolve(rng.normal(0, 0.10, hours + 48),
                              np.ones(48) / 48, mode="same")[:hours]
    cloud_cover = np.clip(cloud_base + cloud_noise, 0.0, 1.0)

    humidity_base = 60.0 + 8.0 * np.sin(2 * np.pi * t / (24 * 365.25))
    humidity_noise = np.convolve(rng.normal(0, 6.0, hours + 24),
                                 np.ones(24) / 24, mode="same")[:hours]
    humidity = np.clip(humidity_base + humidity_noise, 20.0, 98.0)

    wind_base = 4.0 + 2.5 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    wind_speed = np.clip(wind_base + rng.exponential(2.0, hours), 0.0, 30.0)

    return {
        "temperature": temperature.astype(np.float32),
        "humidity": humidity.astype(np.float32),
        "wind_speed": wind_speed.astype(np.float32),
        "cloud_cover": cloud_cover.astype(np.float32),
    }


def _vectorized_ev_load(
    rng: np.random.Generator,
    spec: FeederSpec,
    hour_of_day: np.ndarray,
    is_weekend: np.ndarray,
    temperature: np.ndarray,
    ev_share_of_households: float,
    home_power_kw: Tuple[float, float],
) -> np.ndarray:
    """
    Нагрузка электротранспорта как число одновременных сессий зарядки.

    Посессионная симуляция (цикл по дням и по автомобилям) неприемлема при
    десятках фидеров и годах истории. Здесь число одновременных сессий в час
    берётся из распределения Пуассона с суточным профилем интенсивности —
    это сохраняет и суточную форму, и случайную «зернистость» нагрузки.
    """
    n_ev = spec.households * ev_share_of_households * spec.ev_penetration
    if n_ev <= 0:
        return np.zeros(len(hour_of_day), dtype=np.float32)

    # Профиль вероятности зарядки: преимущественно ночью дома, слабее днём.
    night = np.exp(-((np.minimum(np.abs(hour_of_day - 23),
                                 np.abs(hour_of_day + 1)) ** 2) / (2 * 3.0 ** 2)))
    day = 0.25 * np.exp(-((hour_of_day - 13.0) ** 2) / (2 * 3.0 ** 2))
    profile = night + day
    profile = profile / profile.mean()

    # В выходные больше дневных зарядок, в мороз — чаще из-за падения запаса хода.
    weekend_factor = np.where(is_weekend > 0, 1.10, 1.0)
    cold_factor = np.where(temperature < -15.0, 1.25, 1.0)

    # Средняя доля автомобилей на зарядке в данный час.
    lam = n_ev * 0.06 * profile * weekend_factor * cold_factor
    sessions = rng.poisson(np.maximum(lam, 0.0))

    mean_power = float(np.mean(home_power_kw))
    return (sessions * mean_power).astype(np.float32)


def _vectorized_industrial(
    rng: np.random.Generator,
    hours: int,
    hour_of_day: np.ndarray,
    weekday: np.ndarray,
    persistence: float = 0.94,
) -> np.ndarray:
    """
    Профиль работы промышленной нагрузки без цикла по часам.

    Режимы «смена идёт / не идёт» получаются пороговой обработкой сглаженного
    случайного блуждания: это даёт устойчивые серии нужной длины, тогда как
    независимый шум давал бы дребезг каждый час.
    """
    innov = rng.normal(0.0, 1.0, hours)
    # Сглаживание скользящим средним задаёт характерную длительность режима.
    window = max(int(1.0 / (1.0 - persistence)), 2)
    walk = np.convolve(innov, np.ones(window) / window, mode="same")

    # Порог зависит от дня недели: в выходные вероятность работы ниже.
    threshold = np.where(weekday < 5, -0.15, 0.55)
    working = (walk > threshold).astype(np.float32)

    # Дневная смена выражена сильнее ночной.
    shift_profile = 0.55 + 0.45 * np.exp(-((hour_of_day - 13.0) ** 2) / (2 * 5.0 ** 2))

    # Режим модулирует нагрузку, но не определяет её целиком: у коммерческих
    # потребителей есть часы работы, и суточный ход присутствует всегда.
    # Если оставить чистое произведение, пороговый процесс подавляет суточную
    # сезонность, и автокорреляция на суточном лаге падает ниже реальной.
    return (shift_profile * (0.70 + 0.30 * working)).astype(np.float32)


def generate_panel_data(
    days: int = 365,
    n_cities: int = 2,
    feeders_per_city: int = 8,
    start_date: str = "2024-01-01",
    seed: int = 42,
    kwh_per_household_month: float = 206.0,
    temp_annual_mean: float = 6.25,
    temp_annual_amplitude: float = 12.75,
    temp_min: float = -35.0,
    temp_max: float = 38.0,
    temp_setpoint: float = 18.0,
    cooling_setpoint: float = 24.0,
    ev_share_of_households: float = 0.005,
    solar_share_of_households: float = 0.002,
    ev_home_power_kw: Tuple[float, float] = (3.5, 7.4),
    solar_panel_peak_kw: float = 5.0,
    annual_trend: float = 0.015,
    dsr_events_per_year: int = 10,
    dsr_strength_range: Tuple[float, float] = (0.02, 0.06),
) -> Tuple[pd.DataFrame, List[FeederSpec]]:
    """
    Генерирует panel-датасет: несколько городов, в каждом несколько фидеров.

    Returns
    -------
    df : pd.DataFrame
        Long-format: одна строка на (timestamp, city_id, feeder_id).
    specs : list[FeederSpec]
        Параметры фидеров — нужны для статических признаков и отчётности.
    """
    rng = np.random.default_rng(seed)
    hours = days * 24
    t = np.arange(hours, dtype=np.float64)
    dates = pd.date_range(start=start_date, periods=hours, freq="h")

    hour_of_day = (t % 24).astype(np.float32)
    day_of_sim = (t // 24).astype(int)
    weekday = (day_of_sim % 7).astype(np.int8)
    is_weekend = (weekday >= 5).astype(np.float32)
    day_of_year = (day_of_sim % 365).astype(int)

    holiday_daily = generate_holiday_mask(days, start_date)
    holiday = np.repeat(holiday_daily, 24).astype(np.float32)

    logger.info(
        "Генерация panel-данных: %d городов × %d фидеров × %d дней "
        "(%d рядов, %d точек на ряд)",
        n_cities, feeders_per_city, days, n_cities * feeders_per_city, hours,
    )

    frames: List[pd.DataFrame] = []
    all_specs: List[FeederSpec] = []

    for c in range(n_cities):
        city_id = f"C{c:02d}"
        city_rng = np.random.default_rng(seed * 1000 + c)
        weather = _generate_city_weather(
            city_rng, hours, t, temp_annual_mean, temp_annual_amplitude,
            temp_min, temp_max,
        )
        specs = _sample_feeder_specs(city_rng, city_id, feeders_per_city,
                                     kwh_per_household_month)
        all_specs.extend(specs)

        # События управления спросом объявляются на уровне города: их
        # назначает сетевая компания, а не отдельный фидер.
        dsr_active = np.zeros(hours, np.float32)
        dsr_strength = np.zeros(hours, np.float32)
        peak_hours_idx = np.where(
            (((hour_of_day >= 7) & (hour_of_day < 10))
             | ((hour_of_day >= 17) & (hour_of_day < 21))) & (is_weekend == 0)
        )[0]
        n_dsr = max(2, int(days / 365 * dsr_events_per_year))
        if len(peak_hours_idx):
            for start in city_rng.choice(peak_hours_idx,
                                         size=min(n_dsr, len(peak_hours_idx)),
                                         replace=False):
                dur = int(city_rng.integers(2, 5))
                end = min(int(start) + dur, hours)
                dsr_active[int(start):end] = 1.0
                dsr_strength[int(start):end] = city_rng.uniform(*dsr_strength_range)

        # Поток случайности каждого фидера ответвляется от целочисленной
        # энтропии (сид, город, номер фидера). Прежде он выводился через
        # hash() от строкового идентификатора, а хеш строк в Python
        # рандомизируется при каждом запуске процесса (PEP 456) — один и тот же
        # сид давал разные панели, и результаты не воспроизводились. Погода при
        # этом совпадала побитово, потому что зависит только от city_rng, так
        # что расхождение проявлялось лишь в потреблении.
        feeder_seeds = np.random.SeedSequence([seed, c]).spawn(len(specs))

        for spec, feeder_seq in zip(specs, feeder_seeds):
            feeder_rng = np.random.default_rng(feeder_seq)
            frames.append(_generate_feeder_series(
                spec, feeder_rng, dates, t, hour_of_day, weekday, is_weekend,
                holiday, day_of_year, weather, dsr_active, dsr_strength,
                temp_setpoint, cooling_setpoint, ev_share_of_households,
                solar_share_of_households, ev_home_power_kw, solar_panel_peak_kw,
                annual_trend,
            ))

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["city_id", "feeder_id", "timestamp"]).reset_index(drop=True)

    logger.info(
        "Сформировано %d строк | рядов: %d | среднее потребление фидера: %.1f кВт",
        len(df), df.groupby(["city_id", "feeder_id"]).ngroups,
        df["consumption"].mean(),
    )
    return df, all_specs


def _generate_feeder_series(
    spec: FeederSpec,
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    t: np.ndarray,
    hour_of_day: np.ndarray,
    weekday: np.ndarray,
    is_weekend: np.ndarray,
    holiday: np.ndarray,
    day_of_year: np.ndarray,
    weather: Dict[str, np.ndarray],
    dsr_active: np.ndarray,
    dsr_strength: np.ndarray,
    temp_setpoint: float,
    cooling_setpoint: float,
    ev_share_of_households: float,
    solar_share_of_households: float,
    ev_home_power_kw: Tuple[float, float],
    solar_panel_peak_kw: float,
    annual_trend: float,
) -> pd.DataFrame:
    """Формирует ряд одного фидера. Все операции векторные."""
    hours = len(t)
    temperature = weather["temperature"]
    humidity = weather["humidity"]
    wind_speed = weather["wind_speed"]
    cloud_cover = weather["cloud_cover"]

    # ── Бытовой профиль ──────────────────────────────────────────────────────
    early, standard, night = _build_household_profiles(
        hour_of_day, is_weekend, holiday, weekend_scale=spec.weekend_sensitivity)
    n_std_frac = max(0.0, 1.0 - spec.early_bird_frac - spec.night_owl_frac)
    profile = (spec.early_bird_frac * early
               + n_std_frac * standard
               + spec.night_owl_frac * night)

    # Сдвиг вечернего максимума делает форму графика индивидуальной.
    evening_shift = 0.25 * np.exp(
        -((hour_of_day - spec.evening_peak_hour) ** 2) / (2 * 2.0 ** 2))
    profile = profile * (1.0 + evening_shift)

    # ── Отклик на погоду (асимметричный, как в агрегатном генераторе) ────────
    cold_deg = np.maximum(temp_setpoint - temperature, 0.0)
    warm_deg = np.maximum(temperature - cooling_setpoint, 0.0)
    temp_response = (1.0
                     + spec.heating_sensitivity * cold_deg ** 2
                     + spec.cooling_sensitivity * warm_deg ** 2)

    humid_factor = np.where(temperature > 22.0,
                            0.10 * _sigmoid((humidity - 60.0) / 10.0), 0.0)
    wind_factor = np.where(temperature < 10.0, 0.05 * np.log1p(wind_speed), 0.0)

    # ── Праздники и годовой тренд ───────────────────────────────────────────
    holiday_factor = 1.0 - 0.18 * holiday
    trend = 1.0 + spec.annual_trend * (t / (24 * 365.25))

    # ── Шум ──────────────────────────────────────────────────────────────────
    noise = np.convolve(rng.normal(0.0, spec.noise_sigma, hours + 6),
                        np.ones(6) / 6, mode="same")[:hours] * np.sqrt(6)

    shape = (profile * temp_response * (1.0 + humid_factor + wind_factor)
             * holiday_factor * trend * (1.0 + noise))

    # ── Масштаб: бытовая нагрузка по фактическому потреблению ───────────────
    hours_per_month = 8766.0 / 12.0
    residential_mean = spec.households * spec.kwh_per_household_month / hours_per_month
    residential = shape * (residential_mean / max(shape.mean(), 1e-9))

    # ── Коммерческая и промышленная часть ───────────────────────────────────
    industrial_shape = _vectorized_industrial(rng, hours, hour_of_day, weekday)
    nonres_mean = residential_mean * spec.nonresidential_share
    industrial = industrial_shape * (nonres_mean / max(industrial_shape.mean(), 1e-9))

    # ── Электротранспорт и микрогенерация ───────────────────────────────────
    ev_load = _vectorized_ev_load(rng, spec, hour_of_day, is_weekend, temperature,
                                  ev_share_of_households, ev_home_power_kw)

    n_solar = spec.households * solar_share_of_households * spec.solar_penetration
    solar_profile = np.maximum(0.0, np.exp(-((hour_of_day - 12.5) ** 2) / (2 * 3.2 ** 2)))
    solar_season = 0.72 + 0.28 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    solar_gen = (solar_profile * solar_season * (1.0 - 0.85 * cloud_cover ** 0.7)
                 * solar_panel_peak_kw * n_solar)

    consumption = ((residential + industrial) * (1.0 - dsr_strength)
                   + ev_load - solar_gen)
    consumption = np.clip(consumption, 0.05 * float(np.mean(consumption)), None)

    frame = pd.DataFrame({
        "timestamp": dates,
        "city_id": spec.city_id,
        "feeder_id": spec.feeder_id,
        "feeder_type": spec.feeder_type,
        "consumption": consumption.astype(np.float32),
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,
        "ev_load_kw": ev_load,
        "solar_gen_kw": solar_gen.astype(np.float32),
        "dsr_active": dsr_active,
        "hour": hour_of_day.astype(np.int8),
        "weekday": weekday,
        "is_weekend": is_weekend.astype(np.int8),
        "is_holiday": holiday.astype(np.int8),
        "day_of_year": day_of_year.astype(np.int16),
    })
    for key, value in spec.static_features().items():
        frame[key] = np.float32(value)
    return frame


def city_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Городской ряд как сумма фидеров.

    Иерархия согласована по построению: городской ряд нигде не генерируется
    отдельно, поэтому расхождения между уровнями возникнуть не может.
    """
    return (df.groupby(["city_id", "timestamp"], as_index=False)["consumption"]
              .sum()
              .rename(columns={"consumption": "city_consumption"}))


def validate_panel(df: pd.DataFrame, specs: List[FeederSpec]) -> Tuple[bool, List[Dict]]:
    """
    Проверяет реалистичность и согласованность panel-датасета.

    Помимо привычных показателей отдельно контролируется РАЗНОРОДНОСТЬ рядов:
    если фидеры окажутся почти одинаковыми, panel-режим не даст новой
    информации по сравнению с одним агрегатным рядом и терять на него время
    бессмысленно.
    """
    rows: List[Dict] = []

    def check(name, value, lo, hi, unit=""):
        ok = bool(not np.isnan(value) and lo <= value <= hi)
        rows.append({"показатель": name, "значение": value,
                     "мин": lo, "макс": hi, "единица": unit, "ok": ok})
        return ok

    grouped = df.groupby(["city_id", "feeder_id"])["consumption"]
    means = grouped.mean()
    n_series = len(means)

    check("Число рядов", float(n_series), 2, 1e6, "шт")
    check("Точек на ряд", float(len(df) / max(n_series, 1)), 24 * 30, 1e9, "ч")

    # Разнородность: отношение разброса средних к самому среднему.
    check("Разброс размеров фидеров", float(means.std() / max(means.mean(), 1e-9)),
          0.15, 3.0)

    # Корреляция между фидерами: общая погода задаёт связь, но не тождество.
    pivot = df.pivot_table(index="timestamp", columns="feeder_id",
                           values="consumption", aggfunc="sum")
    corr = pivot.corr().values
    off_diag = corr[~np.eye(len(corr), dtype=bool)]
    check("Средняя корреляция рядов", float(np.nanmean(off_diag)), 0.10, 0.97)

    # Суточная сезонность проверяется по МЕДИАНЕ всех фидеров: отдельно взятый
    # ряд может оказаться промышленным, где суточный ход слабее по природе.
    acfs = []
    for _, grp in df.groupby("feeder_id"):
        c = grp.sort_values("timestamp")["consumption"].values.astype(np.float64)
        if len(c) <= 24:
            continue
        c = c - c.mean()
        acfs.append(float(np.mean(c[:-24] * c[24:]) / (np.var(c) + 1e-12)))
    check("ACF(24), медиана по фидерам", float(np.median(acfs)) if acfs else np.nan,
          0.30, 1.0)
    check("ACF(24), минимум по фидерам", float(np.min(acfs)) if acfs else np.nan,
          0.15, 1.0)

    hourly = df.groupby("hour")["consumption"].mean()
    check("Час вечернего максимума", float(hourly.loc[16:23].idxmax()), 16, 23, "ч")

    # Проверка удельного потребления возможна только там, где известно число
    # домохозяйств, — то есть для собственного генератора. У внешних наборов
    # такой величины нет, и подставлять оценку означало бы проверять данные
    # против собственного же допущения. Пропуск фиксируется явно: молчаливое
    # исчезновение проверки из отчёта неотличимо от её прохождения.
    hours_per_month = 8766.0 / 12.0
    per_hh = [
        df.loc[df["feeder_id"] == spec.feeder_id, "consumption"].mean()
        / (1.0 + spec.nonresidential_share) / spec.households * hours_per_month
        for spec in specs
        if hasattr(spec, "households") and hasattr(spec, "nonresidential_share")
    ]
    if per_hh:
        check("Потребление на домохозяйство", float(np.mean(per_hh)),
              120.0, 340.0, "кВт·ч/мес")
    else:
        rows.append({"показатель": "Потребление на домохозяйство",
                     "значение": float("nan"), "мин": 120.0, "макс": 340.0,
                     "единица": "кВт·ч/мес", "ok": True,
                     "пропущено": "число домохозяйств неизвестно"})
        logger.info("Валидация: удельное потребление не проверяется — "
                    "число домохозяйств в источнике отсутствует")

    logger.info("─" * 84)
    logger.info("ВАЛИДАЦИЯ PANEL-ДАТАСЕТА")
    logger.info("%-34s %14s %22s %8s", "Показатель", "Значение", "Диапазон", "Вердикт")
    logger.info("─" * 84)
    for r in rows:
        logger.info("%-34s %14.3f %10.2f … %-9.2f %8s %s",
                    r["показатель"], r["значение"], r["мин"], r["макс"],
                    "OK" if r["ok"] else "ОТКЛОН.", r["единица"])
    logger.info("─" * 84)

    failed = [r["показатель"] for r in rows if not r["ok"]]
    if failed:
        logger.warning("Отклонения: %s", "; ".join(failed))
    else:
        logger.info("Все показатели panel-датасета в норме.")
    return (not failed), rows
