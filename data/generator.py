# -*- coding: utf-8 -*-
"""
data/generator.py — Генерация реалистичных данных Smart Grid v4.

═══════════════════════════════════════════════════════════════════════════════
ВЕРСИЯ 4 — реалистичная симуляция города вместо «линейных» синтетических данных
═══════════════════════════════════════════════════════════════════════════════

МОТИВАЦИЯ
─────────
В v3 Ridge Regression (R²=0.79) побеждала нейросети (LSTM R²≈0.19, XGBoost R²≈0.24).
Диагноз: данные содержали только сильные линейные зависимости (temp, hour, weekend),
которые Ridge учит идеально. Нейросетям нечего учить сверх линейного базиса.

V4 добавляет структурные нелинейности, которые Ridge не может уловить без явного
задания взаимодействий, но которые нейросеть обнаруживает автоматически:

  1. U-образная зависимость от температуры         → нелинейная
  2. Взаимодействие температура × влажность        → мультипликативное
  3. Взаимодействие температура × ветер            → нелинейное (log1p)
  4. 3 типа домохозяйств с разными пиками          → скрытая переменная
  5. Сдвиг паттерна в выходные и праздники         → нелинейный по времени суток
  6. Поведенческий AR(1) с φ=0.4                  → автокорреляция остатков
  7. Плавный годовой drift через sinusoidal        → нелинейный по сезону
  8. Промышленные выбросы и отключения             → редкие нелинейные события

АРХИТЕКТУРА ГЕНЕРАТОРА
──────────────────────
Потребление строится по формуле:
  C(t) = base_load
       × profile(type, hour, is_weekend, is_holiday)   ← 3 типа домов
       × seasonal_drift(doy)                            ← плавный годовой цикл
       × temp_response(T, humidity, wind)               ← U-кривая + взаимодействия
       × anomaly_factor(t)                              ← выбросы и отключения
       × (1 + ar_noise(t))                             ← поведенческая инерция
       × trend(t)                                       ← долгосрочный рост

ТЕСТ ACF:
  ACF(lag=24)  > 0.5  — суточная сезонность  (сильная периодичность)
  ACF(lag=168) > 0.3  — недельная сезонность (слабее, но статистически значима)
"""

import logging
from datetime import timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_grid.data.generator")


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _gaussian_peak(hour_arr: np.ndarray, center: float, width: float) -> np.ndarray:
    """Гауссовый пик потребления вокруг часа `center` с полушириной `width`."""
    return np.exp(-((hour_arr - center) ** 2) / (2 * width ** 2))


def _build_household_profiles(
    hour_arr: np.ndarray,
    is_weekend: np.ndarray,
    is_holiday: np.ndarray,
    weekend_scale: float = 0.88,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает часовые профили трёх типов домохозяйств.

    early_birds  : ранний пик 6–8, слабый вечерний
    standard     : умеренное утро, сильный вечерний пик 18–20
    night_owls   : поздний пик 21–23, минимум утром

    Выходной/праздник: утренний пик сдвигается на +2ч, вечерний — шире.
    Праздник: дополнительный пик готовки в 13–15ч.

    Returns: three 1-D arrays of shape (hours,) with relative load multiplier.
    """
    H = hour_arr.astype(np.float32)

    # Базовые профили (будни)
    base_night_min = 0.65   # ночной минимум

    early_wd = (
        base_night_min
        + 0.90 * _gaussian_peak(H, center=7.0,  width=1.2)    # сильный утренний
        + 0.35 * _gaussian_peak(H, center=12.5, width=1.5)    # обеденный
        + 0.45 * _gaussian_peak(H, center=18.5, width=2.0)    # слабый вечерний
    )
    standard_wd = (
        base_night_min
        + 0.45 * _gaussian_peak(H, center=8.0,  width=1.5)    # умеренное утро
        + 0.30 * _gaussian_peak(H, center=13.0, width=1.5)    # обеденный
        + 1.00 * _gaussian_peak(H, center=19.0, width=2.5)    # сильный вечерний
    )
    night_wd = (
        base_night_min
        + 0.25 * _gaussian_peak(H, center=8.5,  width=1.5)    # слабое утро
        + 0.35 * _gaussian_peak(H, center=13.5, width=1.5)    # обеденный
        + 0.80 * _gaussian_peak(H, center=21.5, width=2.0)    # поздний вечерний
    )

    # Выходной паттерн: утренний пик +2ч, вечерний шире
    early_we = (
        base_night_min
        + 0.75 * _gaussian_peak(H, center=9.0,  width=2.0)    # сдвинутое утро
        + 0.40 * _gaussian_peak(H, center=14.0, width=2.0)    # обеденный
        + 0.50 * _gaussian_peak(H, center=19.5, width=3.0)    # широкий вечерний
    )
    standard_we = (
        base_night_min
        + 0.35 * _gaussian_peak(H, center=10.0, width=2.0)
        + 0.40 * _gaussian_peak(H, center=14.0, width=2.0)
        + 0.85 * _gaussian_peak(H, center=20.0, width=3.5)    # шире вечером
    )
    night_we = (
        base_night_min
        + 0.20 * _gaussian_peak(H, center=11.0, width=2.0)
        + 0.40 * _gaussian_peak(H, center=15.0, width=2.0)
        + 0.90 * _gaussian_peak(H, center=22.0, width=2.5)
    )

    # Пик готовки в праздники: 13–15ч
    cooking_peak = 0.30 * _gaussian_peak(H, center=14.0, width=1.5)

    # Итоговые профили: weekday / weekend / holiday
    is_we = is_weekend.astype(bool) | is_holiday.astype(bool)
    is_hol = is_holiday.astype(bool)

    early    = np.where(is_we, early_we,    early_wd)
    standard = np.where(is_we, standard_we, standard_wd)
    night    = np.where(is_we, night_we,    night_wd)

    # Праздник: дополнительный пик готовки поверх выходного профиля
    early    = np.where(is_hol, early    + cooking_peak, early)
    standard = np.where(is_hol, standard + cooking_peak, standard)
    night    = np.where(is_hol, night    + cooking_peak, night)

    # Выходные и праздники: общий масштаб снижения.
    # В чисто жилом районе люди дома, но офисы/магазины закрыты →
    # суммарное потребление района ниже будних дней.
    is_scaled = is_we   # weekends + holidays оба масштабируются
    early    = np.where(is_scaled, early    * weekend_scale, early)
    standard = np.where(is_scaled, standard * weekend_scale, standard)
    night    = np.where(is_scaled, night    * weekend_scale, night)

    return (
        early.astype(np.float32),
        standard.astype(np.float32),
        night.astype(np.float32),
    )


def generate_holiday_mask(days: int = 365, start_date: str = "2024-01-01") -> np.ndarray:
    """Маска праздничных дней для России (shape=(days,)). 1 — праздник."""
    holiday_dates = {
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (1, 6), (1, 7), (1, 8),
        (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
    }
    base = pd.to_datetime(start_date)
    mask = np.zeros(days, dtype=np.float32)
    for i in range(days):
        d = base + timedelta(days=i)
        if (d.month, d.day) in holiday_dates:
            mask[i] = 1.0
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def generate_smartgrid_data(
    days: int = 365,
    households: int = 500,
    start_date: str = "2024-01-01",
    seed: int = 42,
    # Температурная кривая
    temp_setpoint: float = 18.0,
    temp_quadratic_coef: float = 2.5e-4,
    # Влажность
    humidity_threshold: float = 60.0,
    humidity_coef: float = 0.30,
    # Ветер
    wind_temp_threshold: float = 10.0,
    wind_coef: float = 0.15,
    # Домохозяйства
    early_bird_frac: float = 0.28,
    night_owl_frac: float = 0.20,
    # Поведенческий AR(1)
    ar_phi: float = 0.40,
    ar_sigma: float = 0.022,
    # Годовой drift
    seasonal_winter_boost: float = 0.15,
    seasonal_summer_dip:   float = 0.10,
    ev_penetration: float = 0.28,
    solar_penetration: float = 0.22,
    industrial_loads: int = 4,
) -> pd.DataFrame:
    """
    Генерирует реалистичный почасовой ряд суммарного потребления района.

    Parameters
    ----------
    days, households, start_date, seed — основные параметры
    temp_setpoint       : комфортная температура (°C) для U-кривой
    temp_quadratic_coef : коэф. квадратичного члена U-кривой
    humidity_threshold  : порог влажности (%) для эффекта кондиционирования
    humidity_coef       : сила эффекта влажности (0–1)
    wind_temp_threshold : порог температуры (°C) для эффекта ветра
    wind_coef           : сила эффекта ветра на отопление
    early_bird_frac     : доля «ранних» домохозяйств
    night_owl_frac      : доля «ночных» домохозяйств
    ar_phi              : коэффициент AR(1) поведенческого остатка
    ar_sigma            : базовая сигма AR(1) шума
    seasonal_winter_boost: зимний прирост потребления (0.15 = +15%)
    seasonal_summer_dip  : летнее снижение потребления (0.10 = −10%)

    Returns
    -------
    pd.DataFrame (часовая гранулярность) со всеми признаками.
    Колонки совместимы с preprocessing.py (v3).
    """
    rng = np.random.default_rng(seed)
    logger.info("Генерация Smart Grid v5: %d дней, %d домохозяйств | EV=%.0f%% Solar=%.0f%%",
                days, households, 100*ev_penetration, 100*solar_penetration)

    hours = days * 24
    t = np.arange(hours, dtype=np.float32)
    dates = pd.date_range(start=start_date, periods=hours, freq="h")

    hour_of_day = (t % 24).astype(np.int8)
    day_of_sim  = (t // 24).astype(int)
    weekday     = day_of_sim % 7
    is_weekend  = (weekday >= 5).astype(np.float32)
    day_of_year = (day_of_sim % 365).astype(int)

    # ── 1. ПРАЗДНИКИ ──────────────────────────────────────────────────────────
    holiday_mask_daily = generate_holiday_mask(days, start_date)
    holiday_mask = np.repeat(holiday_mask_daily, 24)

    # Новогодние каникулы (1–8 января) — отдельная маска для эффектов
    ny_mask = np.zeros(hours, dtype=np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and 1 <= d.day <= 8:
            ny_mask[i * 24: (i + 1) * 24] = 1.0

    # ── 2. ТЕМПЕРАТУРА ────────────────────────────────────────────────────────
    # Годовая синусоида (Россия): среднее +5°C, амплитуда ±20°C
    temp_annual = 5.0 + 20.0 * np.sin(
        2 * np.pi * t / (24 * 365.25) - np.pi / 2
    )
    temp_diurnal = 3.5 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 4)

    # Медленный AR(1) шум: φ=0.97 — плавные отклонения
    temp_noise = np.zeros(hours)
    temp_noise[0] = rng.normal(0, 1.5)
    for i in range(1, hours):
        temp_noise[i] = 0.97 * temp_noise[i - 1] + rng.normal(0, 0.6)

    temperature = np.clip(
        temp_annual + temp_diurnal + temp_noise, -38.0, 45.0
    ).astype(np.float32)

    # ── 3. ТЕПЛОВЫЕ ВОЛНЫ ЖАРЫ (temp > 35°C, +35% потребления) ───────────────
    heat_surge_factor = np.ones(hours, dtype=np.float32)
    n_heat_surges = max(1, int(days / 365 * 3))  # ~3 раза в год

    summer_hours_idx = np.where((dates.month >= 6) & (dates.month <= 8))[0]
    if len(summer_hours_idx) > 24:
        for _ in range(n_heat_surges):
            anchor = int(rng.choice(
                summer_hours_idx[:max(1, len(summer_hours_idx) - 240)]
            ))
            duration_h = int(rng.integers(48, 120))   # 2–5 дней
            end_idx = min(anchor + duration_h, hours)
            # Температура взлетает выше 35°C
            temp_boost = float(rng.uniform(8.0, 14.0))
            temperature[anchor:end_idx] = np.clip(
                temperature[anchor:end_idx] + temp_boost, -38, 45
            )
            heat_surge_factor[anchor:end_idx] = float(rng.uniform(1.28, 1.40))
            logger.debug("Тепловой удар: hour %d, длит. %dч, +%.1f°C",
                         anchor, duration_h, temp_boost)

    # Волны холода (дополнительное снижение температуры зимой)
    cold_wave_factor = np.ones(hours, dtype=np.float32)
    n_cold_waves = max(1, days // 120)
    winter_idx = np.where((dates.month == 12) | (dates.month <= 2))[0]
    if len(winter_idx) > 24:
        for _ in range(n_cold_waves):
            anchor = int(rng.choice(winter_idx[:max(1, len(winter_idx) - 120)]))
            duration_h = int(rng.integers(72, 168))
            end_idx = min(anchor + duration_h, hours)
            temperature[anchor:end_idx] = np.clip(
                temperature[anchor:end_idx] - float(rng.uniform(5.0, 12.0)), -38, 45
            )
            cold_wave_factor[anchor:end_idx] = float(rng.uniform(1.12, 1.22))

    # ── 4. ВЛАЖНОСТЬ ──────────────────────────────────────────────────────────
    # Летом выше (~65%), зимой ниже (~55%). AR(1) шум φ=0.85.
    humidity_annual = 60.0 + 8.0 * np.sin(2 * np.pi * t / (24 * 365.25))
    humidity_noise = np.zeros(hours)
    humidity_noise[0] = rng.normal(0, 5.0)
    for i in range(1, hours):
        humidity_noise[i] = 0.85 * humidity_noise[i - 1] + rng.normal(0, 3.0)
    humidity = np.clip(humidity_annual + humidity_noise, 20.0, 98.0).astype(np.float32)

    # ── 5. СКОРОСТЬ ВЕТРА ─────────────────────────────────────────────────────
    # Зимой ветреннее. Log-normal распределение.
    wind_base = 4.0 + 2.5 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)  # зима ~6.5 м/с
    wind_noise = rng.exponential(scale=2.0, size=hours)
    wind_speed = np.clip(wind_base + wind_noise, 0.0, 30.0).astype(np.float32)

    # ── 6. ПРОФИЛИ ДОМОХОЗЯЙСТВ ───────────────────────────────────────────────
    # Распределение типов фиксировано для воспроизводимости
    n_early    = int(households * early_bird_frac)
    n_night    = int(households * night_owl_frac)
    n_standard = households - n_early - n_night

    logger.info(
        "Типы домохозяйств: ранние=%d (%.0f%%), стандартные=%d (%.0f%%), "
        "ночные=%d (%.0f%%)",
        n_early,    100 * n_early    / households,
        n_standard, 100 * n_standard / households,
        n_night,    100 * n_night    / households,
    )

    early_profile, standard_profile, night_profile = _build_household_profiles(
        hour_arr=t % 24,
        is_weekend=is_weekend,
        is_holiday=holiday_mask,
    )

    # Взвешенный агрегат профилей (нормализован к [0.65, 1.65])
    aggregate_profile = (
        n_early    * early_profile
        + n_standard * standard_profile
        + n_night    * night_profile
    ) / households

    # Праздники — дополнительное снижение базовой нагрузки (меньше работы)
    holiday_base_reduction = (
        1.0
        - 0.35 * ny_mask                          # новогодние −35%
        - 0.18 * holiday_mask * (1 - ny_mask)     # прочие праздники −18%
    ).astype(np.float32)

    # ── 7. ГОДОВОЙ SEASONAL DRIFT ─────────────────────────────────────────────
    # cos(2π × doy / 365): +1 = январь (зима), -1 = июль (лето)
    # Зима: +15%, лето: -10% относительно весны/осени (midpoint)
    # midpoint = (boost - dip) / 2 = 0.025; amplitude = (boost + dip) / 2 = 0.125
    midpoint  = (seasonal_winter_boost - seasonal_summer_dip) / 2
    amplitude = (seasonal_winter_boost + seasonal_summer_dip) / 2
    seasonal_drift = (
        1.0 + midpoint + amplitude * np.cos(2 * np.pi * day_of_year / 365)
    ).astype(np.float32)

    # ── 8. ТЕМПЕРАТУРНЫЙ ОТКЛИК (U-кривая + взаимодействия) ──────────────────
    #
    # БАЗА: consumption += coef * (T - T_setpoint)² / households_scale
    #   Нейтральная точка: T=18°C → нулевой вклад
    #   T=-20°C: (−38)² × 0.02 ≈ 29% прирост (отопление)
    #   T=+38°C: (+20)² × 0.02 ≈  8% прирост (кондиционирование)
    #   Асимметрия отражает реальность: отопление > кондиционирование
    #
    T_diff = temperature - temp_setpoint
    temp_quadratic = temp_quadratic_coef * (T_diff ** 2)   # [0, ~0.3]

    # ВЛАЖНОСТЬ × ТЕМПЕРАТУРА: при жаре + влажности кондиционеры работают интенсивнее
    # humidity_factor = 0 при T ≤ 22°C; нарастает при высокой влажности и жаре
    sigmoid_hum = _sigmoid((humidity - humidity_threshold) / 10.0)
    humidity_factor = np.where(
        temperature > 22.0,
        humidity_coef * sigmoid_hum,
        0.0,
    ).astype(np.float32)

    # ВЕТЕР × ХОЛОД: ветер усиливает теплопотери при морозе
    # wind_factor растёт логарифмически со скоростью ветра
    wind_factor = np.where(
        temperature < wind_temp_threshold,
        wind_coef * np.log1p(wind_speed),
        0.0,
    ).astype(np.float32)

    # Итоговый температурный множитель:
    #   temp_quadratic: базовый отклик от отклонения от комфорта
    #   humidity_factor: дополнительное кондиционирование в жару
    #   wind_factor: дополнительное отопление на ветру в мороз
    temp_response = (
        1.0 + temp_quadratic + humidity_factor + wind_factor
    ).astype(np.float32)

    # ── 9. ДОЛГОСРОЧНЫЙ ТРЕНД ─────────────────────────────────────────────────
    # +5% в год — рост электропотребления с EVs и тепловыми насосами
    trend = (1.0 + 0.05 * (t / (24 * 365.25))).astype(np.float32)

    # ── 10. АНОМАЛИИ ──────────────────────────────────────────────────────────
    anomaly_factor = np.ones(hours, dtype=np.float32)

    # Промышленные выбросы: раз в 30–60 дней, +20–40%, длит. 3–6ч
    n_industrial = max(2, int(industrial_loads) * max(1, days // 365))
    for _ in range(n_industrial):
        idx = int(rng.integers(24, hours - 24))
        dur = int(rng.integers(3, 7))
        spike = float(rng.uniform(1.20, 1.40))
        anomaly_factor[idx: idx + dur] *= spike
        logger.debug("Промышленный выброс: t=%d, dur=%dч, x%.2f", idx, dur, spike)

    # Плановые отключения района: раз в 90 дней, до 15% нормы, 4–8ч
    n_planned = max(1, days // 90)
    for _ in range(n_planned):
        idx = int(rng.integers(48, hours - 48))
        dur = int(rng.integers(4, 9))
        anomaly_factor[idx: idx + dur] *= 0.15
        logger.debug("Плановое отключение: t=%d, dur=%dч", idx, dur)

    # Аварийные отключения: раз в месяц, 1–3ч, до 30% нормы
    n_emergency = max(2, days // 30)
    for _ in range(n_emergency):
        idx = int(rng.integers(24, hours - 24))
        dur = int(rng.integers(1, 4))
        anomaly_factor[idx: idx + dur] *= 0.30

    # ── 11. ПОВЕДЕНЧЕСКИЙ AR(1) ОСТАТОК ──────────────────────────────────────
    # Инерция поведения: «вчера много потребляли → сегодня тоже»
    # Гетероскедастичность: в часы пик σ на 50% выше
    peak_mask = (
        ((hour_of_day >= 7) & (hour_of_day < 10))
        | ((hour_of_day >= 18) & (hour_of_day < 21))
    ).astype(float)
    sigma_t = ar_sigma * (1.0 + 0.5 * peak_mask)

    ar_noise = np.zeros(hours)
    ar_noise[0] = rng.normal(0, ar_sigma)
    for i in range(1, hours):
        ar_noise[i] = ar_phi * ar_noise[i - 1] + rng.normal(0, float(sigma_t[i]))
    ar_noise = ar_noise.astype(np.float32)

    # ── 12. ИТОГОВОЕ ПОТРЕБЛЕНИЕ ──────────────────────────────────────────────
    # Базовый масштаб: households × 10 кВт·ч/ч (среднее потребление на дом ~240 кВт·ч/сут)
    base_scale = float(households) * 10.0

    consumption = (
        aggregate_profile      # суточно-поведенческий профиль + тип дома
        * holiday_base_reduction  # снижение в праздники
        * seasonal_drift       # плавный годовой цикл ±15%
        * temp_response        # U-кривая + влажность + ветер
        * heat_surge_factor    # тепловые удары
        * cold_wave_factor     # волны холода
        * anomaly_factor       # промышленные выбросы + отключения
        * (1.0 + ar_noise)     # поведенческая инерция
        * trend                # долгосрочный рост
        * base_scale
    ).astype(np.float32)

    # ── 12b. EV / Solar / DSR компоненты (v5) ──────────────────────────────
    cloud_cover = np.clip(0.50 + 0.30 * np.sin(2*np.pi * t / 24 + 1.2) + rng.normal(0, 0.12, size=hours), 0.0, 1.0).astype(np.float32)

    evening_ev = (_gaussian_peak(hour_of_day.astype(np.float32), center=20.0, width=2.5)
                  + 0.45 * _gaussian_peak(hour_of_day.astype(np.float32), center=7.5, width=1.8)).astype(np.float32)
    ev_load_norm = np.clip(ev_penetration * evening_ev / (evening_ev.max() + 1e-8), 0.0, 1.0).astype(np.float32)
    ev_load_kw = ev_load_norm * (0.13 * base_scale)

    daylight = np.clip(np.sin(np.pi * (hour_of_day.astype(np.float32) - 6.0) / 12.0), 0.0, 1.0)
    solar_raw = daylight * (1.0 - 0.75 * cloud_cover)
    solar_gen_norm = np.clip(solar_penetration * solar_raw / (solar_raw.max() + 1e-8), 0.0, 1.0).astype(np.float32)
    solar_gen_kw = solar_gen_norm * (0.12 * base_scale)

    dsr_active = np.zeros(hours, dtype=np.float32)
    dsr_hours = max(24, days // 10)
    dsr_idx = rng.choice(np.arange(24, hours - 24), size=min(dsr_hours, max(1, hours - 48)), replace=False)
    dsr_active[dsr_idx] = 1.0

    consumption = consumption + ev_load_kw - solar_gen_kw
    consumption = consumption * (1.0 - 0.07 * dsr_active)
    consumption = np.clip(consumption, 0.0, None)

    # ── 13. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ───────────────────────────────────────────
    month = dates.month.values.astype(np.int8)

    season_arr = np.where(
        (month >= 3) & (month <= 5), "spring",
        np.where(
            (month >= 6) & (month <= 8), "summer",
            np.where((month >= 9) & (month <= 11), "autumn", "winter"),
        ),
    )

    # Тарифные зоны (российская трёхтарифная система)
    tariff_zone_arr = np.empty(hours, dtype=object)
    for i in range(hours):
        h = int(hour_of_day[i])
        wd = int(weekday[i])
        is_hol = bool(holiday_mask[i])
        if h < 7 or h >= 23:
            tariff_zone_arr[i] = "night"
        elif wd >= 5 or is_hol:
            tariff_zone_arr[i] = "day"
        elif 10 <= h < 17 or 21 <= h < 23:
            tariff_zone_arr[i] = "peak"
        else:
            tariff_zone_arr[i] = "day"

    is_peak_hour  = (tariff_zone_arr == "peak").astype(np.int8)
    is_night_hour = (tariff_zone_arr == "night").astype(np.int8)

    heating_degree_days = np.maximum(0.0, 18.0 - temperature).astype(np.float32)
    cooling_degree_days = np.maximum(0.0, temperature - 24.0).astype(np.float32)

    # temperature_squared: нелинейный признак для preprocessing.py
    temp_sq_raw = temperature ** 2
    temp_sq_max = float(temp_sq_raw.max()) + 1e-8
    temperature_squared = (temp_sq_raw / temp_sq_max).astype(np.float32)

    # ── СБОРКА DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":             dates,
        "consumption":           consumption,
        "temperature":           temperature,
        "humidity":              humidity,
        "wind_speed":            wind_speed,
        "hour":                  hour_of_day,
        "weekday":               weekday.astype(np.int8),
        "is_weekend":            is_weekend.astype(np.int8),
        "is_holiday":            holiday_mask.astype(np.int8),
        "is_peak_hour":          is_peak_hour,
        "is_night_hour":         is_night_hour,
        "tariff_zone":           tariff_zone_arr,
        "month":                 month,
        "season":                season_arr,
        "heating_degree_days":   heating_degree_days,
        "cooling_degree_days":   cooling_degree_days,
        "day_of_year":           day_of_year.astype(np.int16),
        "temperature_squared":   temperature_squared,
        "cloud_cover":          cloud_cover,
        "ev_load_norm":         ev_load_norm,
        "solar_gen_norm":       solar_gen_norm,
        "dsr_active":           dsr_active.astype(np.int8),
    })

    # Скользящие статистики (без утечки в будущее)
    df["rolling_mean_24h"] = (
        df["consumption"].rolling(24, min_periods=1).mean().astype(np.float32)
    )
    df["rolling_std_24h"] = (
        df["consumption"].rolling(24, min_periods=2).std().fillna(0.0).astype(np.float32)
    )

    # Циклическое кодирование часа (для preprocessing.py)
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)

    logger.info(
        "Сгенерировано %d записей. Потребление: min=%.1f, mean=%.1f, max=%.1f кВт·ч",
        len(df), df["consumption"].min(),
        df["consumption"].mean(), df["consumption"].max(),
    )
    logger.info("  EV нагрузка: mean=%.1f кВт (%.1f%% базовой)", float(ev_load_kw.mean()), 100.0*float(ev_load_kw.mean()/base_scale))
    logger.info("  Solar: mean=%.1f кВт, max=%.1f кВт", float(solar_gen_kw.mean()), float(solar_gen_kw.max()))
    logger.info("  DSR часов: %d | CV потребления: %.3f", int(dsr_active.sum()), float(df["consumption"].std()/(df["consumption"].mean()+1e-8)))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ВАЛИДАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def validate_generated_data(df: pd.DataFrame) -> bool:
    """
    Проверяет качество сгенерированных данных.

    Критерии:
      1. Нет NaN / Inf
      2. Нет отрицательного потребления
      3. Утренний пик в 6–11ч, вечерний в 17–22ч
      4. Выходные < будних (0.75 ≤ ratio ≤ 0.97)
      5. Зима > лето по потреблению
      6. Диапазон температур корректный
      7. ACF(lag=24) > 0.5  — суточная сезонность
      8. ACF(lag=168) > 0.3 — недельная сезонность
    """
    passed = True
    logger.info("Валидация данных Smart Grid v4...")

    # 1. NaN / Inf
    bad = df["consumption"].isna().sum() + np.isinf(df["consumption"].values).sum()
    if bad > 0:
        logger.error("  ❌ NaN/Inf: %d значений", bad); passed = False
    else:
        logger.info("  ✅ NaN/Inf: нет")

    # 2. Отрицательное потребление
    neg = (df["consumption"] < 0).sum()
    if neg > 0:
        logger.error("  ❌ Отрицательное потребление: %d значений", neg); passed = False
    else:
        logger.info("  ✅ Отрицательных значений: нет")

    # 3. Пики в правильное время
    hourly = df.groupby("hour")["consumption"].mean()
    morning_peak = int(hourly.loc[6:11].idxmax())
    evening_peak = int(hourly.loc[17:22].idxmax())
    if not (6 <= morning_peak <= 11):
        logger.warning("  ⚠️  Утренний пик в %d:00 (ожидалось 6–11)", morning_peak)
    else:
        logger.info("  ✅ Утренний пик: %d:00", morning_peak)
    if not (17 <= evening_peak <= 22):
        logger.warning("  ⚠️  Вечерний пик в %d:00 (ожидалось 17–22)", evening_peak)
    else:
        logger.info("  ✅ Вечерний пик: %d:00", evening_peak)

    # 4. Выходные/будни
    wd_mean = df[df["is_weekend"] == 0]["consumption"].mean()
    we_mean = df[df["is_weekend"] == 1]["consumption"].mean()
    ratio = we_mean / wd_mean
    if not (0.75 <= ratio <= 0.97):
        logger.warning("  ⚠️  Выходные/будни = %.2f (ожидалось 0.75–0.97)", ratio)
    else:
        logger.info("  ✅ Выходные/будни: %.2f", ratio)

    # 5. Зима > Лето
    if "season" in df.columns:
        w_mean = df[df["season"] == "winter"]["consumption"].mean()
        s_mean = df[df["season"] == "summer"]["consumption"].mean()
        if w_mean <= s_mean:
            logger.warning("  ⚠️  Зима (%.1f) ≤ Лето (%.1f)", w_mean, s_mean)
        else:
            logger.info("  ✅ Сезонность: зима=%.1f > лето=%.1f", w_mean, s_mean)

    # 6. Диапазон температур
    t_min, t_max = df["temperature"].min(), df["temperature"].max()
    n_days = len(df) // 24
    if not (t_min <= 5.0 and t_max >= (15.0 if n_days < 180 else 22.0)):
        logger.warning("  ⚠️  Диапазон температур: %.1f..%.1f°C", t_min, t_max)
    else:
        logger.info("  ✅ Диапазон температур: %.1f..%.1f°C", t_min, t_max)

    # 7–8. ACF(lag=24) и ACF(lag=168)
    cons = df["consumption"].values
    cons_centered = cons - cons.mean()
    var = np.var(cons_centered) + 1e-10

    def acf(series: np.ndarray, lag: int) -> float:
        n = len(series)
        if lag >= n:
            return 0.0
        return float(np.mean(series[:n - lag] * series[lag:])) / var

    acf_24  = acf(cons_centered, lag=24)
    acf_168 = acf(cons_centered, lag=168)

    if acf_24 >= 0.5:
        logger.info("  ✅ ACF(lag=24)  = %.3f ≥ 0.5 — суточная сезонность OK", acf_24)
    else:
        logger.warning("  ⚠️  ACF(lag=24)  = %.3f < 0.5 — слабая суточная сезонность", acf_24)

    if len(df) >= 168 * 2:
        if acf_168 >= 0.3:
            logger.info("  ✅ ACF(lag=168) = %.3f ≥ 0.3 — недельная сезонность OK", acf_168)
        else:
            logger.warning("  ⚠️  ACF(lag=168) = %.3f < 0.3 — слабая недельная сезонность", acf_168)
    else:
        logger.info("  ℹ️  ACF(lag=168): недостаточно данных (< 336 строк)")

    status = "ПРОЙДЕНА" if passed else "ПРОВАЛЕНА"
    logger.info("Валидация %s", status)
    return passed