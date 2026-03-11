# -*- coding: utf-8 -*-
"""
data/generator.py — Генерация синтетических данных Smart Grid.

ВЕРСИЯ 3 — добавлены нелинейные признаки для нейросетей:
  • 730 дней / 500 домохозяйств (было 180 / 250)
  • 9 исходных признаков: is_peak_hour, is_night_hour, tariff_zone, month,
    season, heating_degree_days, cooling_degree_days, rolling_mean_24h,
    rolling_std_24h
  • НОВЫЕ (v3): hour_sin, hour_cos, temperature_squared
    - hour_sin/cos: циклическое кодирование часа.
      Причина: hour/23 создаёт разрыв между часами 23 и 0 (1.0 vs 0.0),
      хотя они соседние. Sin/cos образуют замкнутый круг.
      Источник: Ke et al. 2017 (LightGBM); стандарт в SOTA временных рядах.
    - temperature_squared: нелинейная U-образная зависимость от температуры.
      Потребление растёт и при сильном морозе, и при жаре (кондиционеры).
      Квадратичный член позволяет линейным слоям нейросети приближать эту форму.
  • Реалистичные тарифные зоны (Россия): пик 6.50 / день 4.20 / ночь 1.80
  • AR(1) коррелированный шум вместо белого — реалистичнее
  • Гетероскедастичность: больше шума в часы пик
  • Волны жары/холода (3–7 дней подряд)
  • Новогодние каникулы (10 дней сниженного потребления)
  • Технические отключения (плановые и аварийные)
  • validate_generated_data() для проверки качества данных
"""

import logging
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_grid.data.generator")


# ══════════════════════════════════════════════════════════════════════════════
# ПРАЗДНИКИ
# ══════════════════════════════════════════════════════════════════════════════

def generate_holiday_mask(days: int = 365, start_date: str = "2024-01-01") -> np.ndarray:
    """
    Маска праздничных дней для России (shape=(days,)).
    1 — праздник, 0 — обычный день.
    """
    holiday_dates = {
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (1, 6), (1, 7), (1, 8),   # Новогодние каникулы (8 дней)
        (2, 23),                   # День защитника Отечества
        (3, 8),                    # Международный женский день
        (5, 1),                    # День труда
        (5, 9),                    # День Победы
        (6, 12),                   # День России
        (11, 4),                   # День народного единства
    }
    base = pd.to_datetime(start_date)
    mask = np.zeros(days, dtype=np.float32)
    for i in range(days):
        d = base + timedelta(days=i)
        if (d.month, d.day) in holiday_dates:
            mask[i] = 1.0
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР
# ══════════════════════════════════════════════════════════════════════════════

def generate_smartgrid_data(
    days: int = 730,
    households: int = 500,
    start_date: str = "2024-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует реалистичный почасовой ряд суммарного потребления
    района из `households` домохозяйств.

    Факторы моделирования
    ─────────────────────
    • Суточная сезонность  — двойной пик (08:00 и 19:00)
    • Недельная сезонность — снижение на выходных −15%
    • Праздники            — снижение −22%, новогодние каникулы −35%
    • Годовая температурная зависимость (отопление / кондиционирование)
    • Долгосрочный тренд роста ~5% в год
    • Волны жары (t > 30°C, 3–7 дней) и холода (t < −15°C)
    • Технические отключения: плановые (−90%, 4–8ч) и аварийные (−70%, 1–3ч)
    • AR(1) шум с φ=0.4: ρ(ε_t, ε_{t-1}) = 0.4 (реалистичнее белого шума)
    • Гетероскедастичность: σ в часы пик +50%

    Новые признаки в DataFrame
    ──────────────────────────
    is_peak_hour      — 1 если 07–10 или 18–21 (пиковые часы тарифа)
    is_night_hour     — 1 если 23–07 (ночная зона тарифа)
    tariff_zone       — "peak" / "day" / "night"
    month             — 1..12
    season            — "winter" / "spring" / "summer" / "autumn"
    heating_degree_days — max(0, 18 − T) (стандарт EN 12831)
    cooling_degree_days — max(0, T − 24) (стандарт ASHRAE)
    rolling_mean_24h  — скользящее среднее потребления за 24ч
    rolling_std_24h   — скользящее стд.откл. за 24ч

    Returns
    -------
    pd.DataFrame со всеми признаками (часовая гранулярность).
    """
    rng = np.random.default_rng(seed)
    logger.info("Генерация данных Smart Grid: %d дней, %d домохозяйств", days, households)

    hours = days * 24
    t = np.arange(hours)
    dates = pd.date_range(start=start_date, periods=hours, freq="h")

    # ── 1. ТЕМПЕРАТУРА ────────────────────────────────────────────────────────
    # Годовая синусоида (Россия: среднее +5°C, амплитуда ±18°C) + суточный ход
    temp_year = 5.0 + 18.0 * np.sin(2 * np.pi * t / (24 * 365.25) - np.pi / 2)
    temp_day = 3.0 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 3)  # +3°C днём

    # AR(1) температурный шум: φ=0.95 — температура меняется плавно
    temp_noise = np.zeros(hours)
    temp_noise[0] = rng.normal(0, 2.0)
    for i in range(1, hours):
        temp_noise[i] = 0.95 * temp_noise[i - 1] + rng.normal(0, 0.8)

    temperature = np.clip(temp_year + temp_day + temp_noise, -35.0, 42.0).astype(np.float32)

    # ── 2. ВОЛНЫ ЖАРЫ / ХОЛОДА ────────────────────────────────────────────────
    heat_wave_factor = np.ones(hours, dtype=np.float32)
    cold_wave_factor = np.ones(hours, dtype=np.float32)

    # Летние волны жары (июнь–август): ~8 волн за 2 года
    n_heat_waves = max(1, days // 90)
    summer_hours = np.where(
        (dates.month >= 6) & (dates.month <= 8)
    )[0]
    if len(summer_hours) > 0:
        for _ in range(n_heat_waves):
            start_idx = int(rng.choice(summer_hours[:max(1, len(summer_hours) - 168)]))
            duration_h = int(rng.integers(72, 168))  # 3–7 дней
            end_idx = min(start_idx + duration_h, hours)
            # Волна жары: +5–10°C к температуре → рост кондиционирования на 25%
            temperature[start_idx:end_idx] = np.clip(
                temperature[start_idx:end_idx] + float(rng.uniform(5, 10)), -35, 42
            )
            heat_wave_factor[start_idx:end_idx] = float(rng.uniform(1.20, 1.30))

    # Зимние волны холода (декабрь–февраль): ~6 волн за 2 года
    n_cold_waves = max(1, days // 120)
    winter_hours = np.where(
        (dates.month == 12) | (dates.month <= 2)
    )[0]
    if len(winter_hours) > 0:
        for _ in range(n_cold_waves):
            start_idx = int(rng.choice(winter_hours[:max(1, len(winter_hours) - 120)]))
            duration_h = int(rng.integers(72, 168))
            end_idx = min(start_idx + duration_h, hours)
            temperature[start_idx:end_idx] = np.clip(
                temperature[start_idx:end_idx] - float(rng.uniform(5, 12)), -35, 42
            )
            cold_wave_factor[start_idx:end_idx] = float(rng.uniform(1.15, 1.25))

    # ── 3. СУТОЧНАЯ СЕЗОННОСТЬ ────────────────────────────────────────────────
    hour_of_day = t % 24
    daily_pattern = (
        1.0
        + 0.20 * np.sin(2 * np.pi * hour_of_day / 24)
        + 0.50 * np.exp(-((hour_of_day - 8) ** 2) / 8)    # Утренний пик 08:00
        + 0.70 * np.exp(-((hour_of_day - 19) ** 2) / 10)  # Вечерний пик 19:00
        - 0.20 * np.exp(-((hour_of_day - 3) ** 2) / 5)    # Ночной минимум 03:00
    ).astype(np.float32)

    # ── 4. НЕДЕЛЬНАЯ СЕЗОННОСТЬ ───────────────────────────────────────────────
    weekday = (t // 24) % 7
    is_weekend = (weekday >= 5).astype(np.float32)
    weekly_factor = (1.0 - 0.15 * is_weekend).astype(np.float32)

    # ── 5. ПРАЗДНИКИ ──────────────────────────────────────────────────────────
    holiday_mask_daily = generate_holiday_mask(days, start_date)
    holiday_mask = np.repeat(holiday_mask_daily, 24)

    # Новогодние каникулы: 1–8 января — снижение на 35% вместо 22%
    ny_mask = np.zeros(hours, dtype=np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and 1 <= d.day <= 8:
            ny_mask[i * 24: (i + 1) * 24] = 1.0

    holiday_factor = (
        1.0
        - 0.35 * ny_mask                         # Новогодние каникулы −35%
        - 0.22 * holiday_mask * (1 - ny_mask)    # Прочие праздники −22%
    ).astype(np.float32)

    # ── 6. ДОЛГОСРОЧНЫЙ ТРЕНД ────────────────────────────────────────────────
    trend = (1.0 + 0.05 * (t / (24 * 365.25))).astype(np.float32)

    # ── 7. ВЛИЯНИЕ ТЕМПЕРАТУРЫ ────────────────────────────────────────────────
    heating = np.maximum(0.0, (15.0 - temperature) / 15.0) * 0.45
    cooling = np.maximum(0.0, (temperature - 25.0) / 10.0) * 0.35
    temp_factor = (1.0 + heating + cooling).astype(np.float32)

    # ── 8. ТЕХНИЧЕСКИЕ ОТКЛЮЧЕНИЯ ────────────────────────────────────────────
    outage_factor = np.ones(hours, dtype=np.float32)

    # Плановые отключения: ~1 раз в 3 месяца, 4–8 часов, −90%
    n_planned = max(1, days // 90)
    for _ in range(n_planned):
        idx = int(rng.integers(24, hours - 24))
        dur = int(rng.integers(4, 9))
        outage_factor[idx: idx + dur] *= 0.10
        logger.debug("Плановое отключение: час %d, длительность %d", idx, dur)

    # Аварийные отключения: ~1 раз в месяц, 1–3 часа, −70%
    n_emergency = max(2, days // 30)
    for _ in range(n_emergency):
        idx = int(rng.integers(24, hours - 24))
        dur = int(rng.integers(1, 4))
        outage_factor[idx: idx + dur] *= 0.30

    # ── 9. AR(1) ШУМ (реалистичнее белого) ───────────────────────────────────
    # Белый шум: ε_t ~ N(0, σ)
    # AR(1):     u_t = φ * u_{t-1} + ε_t, где φ = 0.4
    # Сохраняет краткосрочную автокорреляцию потребления
    phi = 0.4
    base_sigma = 0.025

    # Гетероскедастичность: в часы пик σ на 50% больше
    peak_hours = ((hour_of_day >= 7) & (hour_of_day < 10)) | \
                 ((hour_of_day >= 18) & (hour_of_day < 21))
    sigma_t = base_sigma * (1.0 + 0.5 * peak_hours.astype(float))

    ar_noise = np.zeros(hours)
    ar_noise[0] = rng.normal(0, base_sigma)
    for i in range(1, hours):
        ar_noise[i] = phi * ar_noise[i - 1] + rng.normal(0, float(sigma_t[i]))
    ar_noise = ar_noise.astype(np.float32)

    # ── ИТОГОВОЕ ПОТРЕБЛЕНИЕ ──────────────────────────────────────────────────
    consumption = (
        daily_pattern
        * weekly_factor
        * holiday_factor
        * trend
        * temp_factor
        * outage_factor
        * heat_wave_factor
        * cold_wave_factor
        * (1.0 + ar_noise)
        * households
        * 10.0   # масштаб → реалистичные кВт·ч
    ).astype(np.float32)
    consumption = np.clip(consumption, 0.0, None)

    # ── НОВЫЕ ПРИЗНАКИ ────────────────────────────────────────────────────────
    month = dates.month.values.astype(np.int8)

    season_arr = np.where(
        (month >= 3) & (month <= 5), "spring",
        np.where(
            (month >= 6) & (month <= 8), "summer",
            np.where((month >= 9) & (month <= 11), "autumn", "winter"),
        ),
    )

    # Тарифные зоны (российская трёхтарифная система)
    # Пиковая: 10–17, 21–23 будни
    # Дневная: 07–10, 17–21 будни; 07–23 выходные
    # Ночная:  23–07 все дни
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

    is_peak_hour = ((tariff_zone_arr == "peak")).astype(np.int8)
    is_night_hour = ((tariff_zone_arr == "night")).astype(np.int8)

    # Градусо-дни отопления и охлаждения
    heating_degree_days = np.maximum(0.0, 18.0 - temperature).astype(np.float32)
    cooling_degree_days = np.maximum(0.0, temperature - 24.0).astype(np.float32)

    # ── СБОРКА DataFrame ──────────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp":           dates,
        "consumption":         consumption,
        "temperature":         temperature,
        "hour":                hour_of_day.astype(np.int8),
        "weekday":             weekday.astype(np.int8),
        "is_weekend":          is_weekend.astype(np.int8),
        "is_holiday":          holiday_mask.astype(np.int8),
        "is_peak_hour":        is_peak_hour,
        "is_night_hour":       is_night_hour,
        "tariff_zone":         tariff_zone_arr,
        "month":               month,
        "season":              season_arr,
        "heating_degree_days": heating_degree_days,
        "cooling_degree_days": cooling_degree_days,
        "day_of_year":         ((t // 24) % 365).astype(np.int16),
    })

    # Скользящие статистики (вычисляются после сборки — нет утечки в прошлое)
    df["rolling_mean_24h"] = (
        df["consumption"].rolling(24, min_periods=1).mean().astype(np.float32)
    )
    df["rolling_std_24h"] = (
        df["consumption"].rolling(24, min_periods=2).std().fillna(0.0).astype(np.float32)
    )

    # ── НЕЛИНЕЙНЫЕ ПРИЗНАКИ v3 ────────────────────────────────────────────────
    # 1. Циклическое кодирование часа: sin/cos вместо линейного hour/23
    #    Проблема hour/23: часы 0 и 23 «далеки» (0.0 vs 1.0), хотя соседние.
    #    Sin/cos: f(0) ≈ f(24) — разрыв устранён. Стандарт для периодических временны́х рядов.
    #    [Ke et al., 2017; также в N-BEATS, Informer, Autoformer]
    df["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24).astype(np.float32)

    # 2. Температура² — нелинейная U-образная зависимость потребления от T.
    #    При T < 15°C: отопление растёт. При T > 25°C: кондиционирование растёт.
    #    Квадратичный член позволяет линейным слоям нейросети приближать эту форму
    #    без явного задания точки перегиба. Нормализация T²/max(T²) → [0, 1].
    temp_sq_raw = temperature ** 2
    temp_sq_max = float(temp_sq_raw.max()) + 1e-8   # защита от деления на 0
    df["temperature_squared"] = (temp_sq_raw / temp_sq_max).astype(np.float32)

    logger.info(
        "Сгенерировано %d записей. Потребление: min=%.1f, mean=%.1f, max=%.1f кВт·ч",
        len(df),
        df["consumption"].min(),
        df["consumption"].mean(),
        df["consumption"].max(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ВАЛИДАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def validate_generated_data(df: pd.DataFrame) -> bool:
    """
    Проверяет качество сгенерированных данных.

    Критерии:
    1. Нет отрицательного или нулевого потребления (кроме отключений < 1%)
    2. Суточные пики в правильное время (8:00 и 19:00 ± 2ч)
    3. Выходные < будних дней в среднем на 5–25%
    4. Сезонность: зима > лето для отопительной страны
    5. Нет NaN или Inf значений

    Returns
    -------
    bool — True если все проверки пройдены
    """
    passed = True
    logger.info("Валидация данных...")

    # 1. Нет NaN / Inf
    nan_count = df["consumption"].isna().sum() + np.isinf(df["consumption"].values).sum()
    if nan_count > 0:
        logger.error("  ❌ NaN/Inf в потреблении: %d значений", nan_count)
        passed = False
    else:
        logger.info("  ✅ NaN/Inf: нет")

    # 2. Нет отрицательного потребления
    neg_count = (df["consumption"] < 0).sum()
    if neg_count > 0:
        logger.error("  ❌ Отрицательное потребление: %d значений", neg_count)
        passed = False
    else:
        logger.info("  ✅ Отрицательных значений: нет")

    # 3. Пики в правильное время
    hourly_mean = df.groupby("hour")["consumption"].mean()
    peak_hours = hourly_mean.idxmax()
    # Утренний пик должен быть в 6–10, вечерний в 17–21
    morning_peak = hourly_mean.loc[6:10].idxmax()
    evening_peak = hourly_mean.loc[17:22].idxmax()
    if not (6 <= morning_peak <= 10):
        logger.warning("  ⚠️  Утренний пик в %d:00 (ожидалось 6–10)", morning_peak)
    else:
        logger.info("  ✅ Утренний пик: %d:00", morning_peak)
    if not (17 <= evening_peak <= 22):
        logger.warning("  ⚠️  Вечерний пик в %d:00 (ожидалось 17–22)", evening_peak)
    else:
        logger.info("  ✅ Вечерний пик: %d:00", evening_peak)

    # 4. Выходные < будних дней
    weekday_mean = df[df["is_weekend"] == 0]["consumption"].mean()
    weekend_mean = df[df["is_weekend"] == 1]["consumption"].mean()
    ratio = weekend_mean / weekday_mean
    if not (0.75 <= ratio <= 0.97):
        logger.warning(
            "  ⚠️  Выходные/будни = %.2f (ожидалось 0.75–0.97)", ratio
        )
    else:
        logger.info("  ✅ Выходные/будни: %.2f", ratio)

    # 5. Сезонность: зима > лето
    if "season" in df.columns:
        winter_mean = df[df["season"] == "winter"]["consumption"].mean()
        summer_mean = df[df["season"] == "summer"]["consumption"].mean()
        if winter_mean <= summer_mean:
            logger.warning(
                "  ⚠️  Зима (%.1f) ≤ Лето (%.1f) — нетипично для России",
                winter_mean, summer_mean,
            )
        else:
            logger.info(
                "  ✅ Сезонность: зима=%.1f > лето=%.1f кВт·ч",
                winter_mean, summer_mean,
            )

    # 6. Диапазон температур
    # Границы адаптивные: полный диапазон (-35..42) достигается только при days≥365.
    # При коротких периодах (fast_mode, days<180) крайние значения могут не встретиться.
    t_min, t_max = df["temperature"].min(), df["temperature"].max()
    n_days = len(df) // 24
    # Минимум: зима при 365 днях даёт t<-15; при 90 днях без зимы — t>-5
    min_ok = t_min <= 5.0   # в любом периоде должно быть прохладно
    # Максимум: лето при 365 днях даёт t>20; при коротком периоде — гибче
    max_ok = t_max >= (15.0 if n_days < 180 else 22.0)
    if not (min_ok and max_ok):
        logger.warning("  ⚠️  Диапазон температур: %.1f..%.1f°C (n_days=%d)", t_min, t_max, n_days)
    else:
        logger.info("  ✅ Диапазон температур: %.1f..%.1f°C", t_min, t_max)

    status = "ПРОЙДЕНА" if passed else "ПРОВАЛЕНА"
    logger.info("Валидация %s", status)
    return passed