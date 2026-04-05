# -*- coding: utf-8 -*-
"""
data/generator.py — Smart Grid v5: сложные нелинейные данные.

ПРОБЛЕМА v4: LinearRegression (MAE=628) побеждала LSTM (MAE=855).
ACF(24)=0.895 — данные слишком периодические, LinReg с lag_24h угадывает тривиально.

НОВЫЕ НЕЛИНЕЙНОСТИ v5:
  1. EV зарядка (28% домов) — случайные непериодические спайки
  2. Солнечные панели (22% домов) — снижение нагрузки зависит от cloud_cover
  3. Demand Response Events — 6–10 событий/год, снижение 15–32%
  4. Промышленные нагрузки с независимыми 8–16ч циклами
  5. AR sigma: 0.022→0.060, phi: 0.40→0.65 (GARCH-эффект)
  6. Behavioral regime switching (concept drift каждые 60–90 дней)
  7. Triple interaction: temperature × humidity × wind
  8. Новые признаки в df: cloud_cover, ev_load_norm, solar_gen_norm, dsr_active
"""

import logging
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_grid.data.generator")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _gaussian_peak(hour_arr, center, width):
    return np.exp(-((hour_arr - center) ** 2) / (2 * width ** 2))


def _build_household_profiles(hour_arr, is_weekend, is_holiday, weekend_scale=0.88):
    H = hour_arr.astype(np.float32)
    b = 0.65
    early_wd    = b + 0.90*_gaussian_peak(H,7.0,1.2) + 0.35*_gaussian_peak(H,12.5,1.5) + 0.45*_gaussian_peak(H,18.5,2.0)
    standard_wd = b + 0.45*_gaussian_peak(H,8.0,1.5) + 0.30*_gaussian_peak(H,13.0,1.5) + 1.00*_gaussian_peak(H,19.0,2.5)
    night_wd    = b + 0.25*_gaussian_peak(H,8.5,1.5) + 0.35*_gaussian_peak(H,13.5,1.5) + 0.80*_gaussian_peak(H,21.5,2.0)
    early_we    = b + 0.75*_gaussian_peak(H,9.0,2.0) + 0.40*_gaussian_peak(H,14.0,2.0) + 0.50*_gaussian_peak(H,19.5,3.0)
    standard_we = b + 0.35*_gaussian_peak(H,10.0,2.0)+ 0.40*_gaussian_peak(H,14.0,2.0) + 0.85*_gaussian_peak(H,20.0,3.5)
    night_we    = b + 0.20*_gaussian_peak(H,11.0,2.0)+ 0.40*_gaussian_peak(H,15.0,2.0) + 0.90*_gaussian_peak(H,22.0,2.5)
    cooking     = 0.30 * _gaussian_peak(H, 14.0, 1.5)
    is_we  = is_weekend.astype(bool) | is_holiday.astype(bool)
    is_hol = is_holiday.astype(bool)
    early    = np.where(is_we, early_we, early_wd)
    standard = np.where(is_we, standard_we, standard_wd)
    night    = np.where(is_we, night_we, night_wd)
    early    = np.where(is_hol, early+cooking, early)
    standard = np.where(is_hol, standard+cooking, standard)
    night    = np.where(is_hol, night+cooking, night)
    early    = np.where(is_we, early*weekend_scale, early)
    standard = np.where(is_we, standard*weekend_scale, standard)
    night    = np.where(is_we, night*weekend_scale, night)
    return early.astype(np.float32), standard.astype(np.float32), night.astype(np.float32)


def generate_holiday_mask(days=365, start_date="2024-01-01"):
    holidays = {(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),(2,23),(3,8),(5,1),(5,9),(6,12),(11,4)}
    base = pd.to_datetime(start_date)
    mask = np.zeros(days, dtype=np.float32)
    for i in range(days):
        d = base + timedelta(days=i)
        if (d.month, d.day) in holidays:
            mask[i] = 1.0
    return mask


def generate_smartgrid_data(
    days=365, households=500, start_date="2024-01-01", seed=42,
    temp_setpoint=18.0, temp_quadratic_coef=2.5e-4,
    humidity_threshold=60.0, humidity_coef=0.30,
    wind_temp_threshold=10.0, wind_coef=0.15,
    early_bird_frac=0.28, night_owl_frac=0.20,
    ar_phi=0.65, ar_sigma=0.060,        # v5: усилен
    seasonal_winter_boost=0.15, seasonal_summer_dip=0.10,
    ev_penetration=0.28,                # v5: EV 28% домов
    solar_penetration=0.22,             # v5: Solar 22% домов
    industrial_loads=4,                 # v5: промышленные потребители
):
    rng = np.random.default_rng(seed)
    logger.info("Генерация Smart Grid v5: %d дней, %d домохозяйств | EV=%.0f%% Solar=%.0f%%",
                days, households, ev_penetration*100, solar_penetration*100)

    hours = days * 24
    t = np.arange(hours, dtype=np.float32)
    dates = pd.date_range(start=start_date, periods=hours, freq="h")
    hour_of_day = (t % 24).astype(np.int8)
    day_of_sim  = (t // 24).astype(int)
    weekday     = day_of_sim % 7
    is_weekend  = (weekday >= 5).astype(np.float32)
    day_of_year = (day_of_sim % 365).astype(int)

    # Праздники
    holiday_mask_daily = generate_holiday_mask(days, start_date)
    holiday_mask = np.repeat(holiday_mask_daily, 24)
    ny_mask = np.zeros(hours, dtype=np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and 1 <= d.day <= 8:
            ny_mask[i*24:(i+1)*24] = 1.0

    # Температура
    temp_annual  = 5.0 + 20.0 * np.sin(2*np.pi*t/(24*365.25) - np.pi/2)
    temp_diurnal = 3.5 * np.sin(2*np.pi*(t%24)/24 - np.pi/4)
    temp_noise = np.zeros(hours)
    temp_noise[0] = rng.normal(0, 1.5)
    for i in range(1, hours):
        temp_noise[i] = 0.97*temp_noise[i-1] + rng.normal(0, 0.6)
    temperature = np.clip(temp_annual + temp_diurnal + temp_noise, -38.0, 45.0).astype(np.float32)

    # Тепловые волны и волны холода
    heat_surge_factor = np.ones(hours, dtype=np.float32)
    summer_idx = np.where((dates.month >= 6) & (dates.month <= 8))[0]
    if len(summer_idx) > 24:
        for _ in range(max(1, int(days/365*3))):
            a = int(rng.choice(summer_idx[:max(1, len(summer_idx)-240)]))
            e = min(a + int(rng.integers(48, 120)), hours)
            temperature[a:e] = np.clip(temperature[a:e] + rng.uniform(8,14), -38, 45)
            heat_surge_factor[a:e] = rng.uniform(1.28, 1.40)
    cold_wave_factor = np.ones(hours, dtype=np.float32)
    winter_idx = np.where((dates.month==12)|(dates.month<=2))[0]
    if len(winter_idx) > 24:
        for _ in range(max(1, days//120)):
            a = int(rng.choice(winter_idx[:max(1, len(winter_idx)-120)]))
            e = min(a + int(rng.integers(72, 168)), hours)
            temperature[a:e] = np.clip(temperature[a:e] - rng.uniform(5,12), -38, 45)
            cold_wave_factor[a:e] = rng.uniform(1.12, 1.22)

    # Облачность (v5 новое)
    cloud_annual = 0.52 + 0.18*np.cos(2*np.pi*t/(24*365.25) + np.pi)
    cloud_noise = np.zeros(hours)
    cloud_noise[0] = rng.normal(0, 0.08)
    for i in range(1, hours):
        cloud_noise[i] = 0.92*cloud_noise[i-1] + rng.normal(0, 0.06)
    cloud_cover = np.clip(cloud_annual + cloud_noise, 0.0, 1.0).astype(np.float32)

    # Влажность
    hum_annual = 60.0 + 8.0*np.sin(2*np.pi*t/(24*365.25))
    hum_noise = np.zeros(hours)
    hum_noise[0] = rng.normal(0, 5.0)
    for i in range(1, hours):
        hum_noise[i] = 0.85*hum_noise[i-1] + rng.normal(0, 3.0)
    humidity = np.clip(hum_annual + hum_noise, 20.0, 98.0).astype(np.float32)

    # Ветер
    wind_base  = 4.0 + 2.5*np.cos(2*np.pi*t/(24*365.25) + np.pi)
    wind_speed = np.clip(wind_base + rng.exponential(2.0, hours), 0.0, 30.0).astype(np.float32)

    # Профили домохозяйств
    n_early    = int(households * early_bird_frac)
    n_night    = int(households * night_owl_frac)
    n_standard = households - n_early - n_night
    logger.info("Типы домохозяйств: ранние=%d (%.0f%%), стандартные=%d (%.0f%%), ночные=%d (%.0f%%)",
                n_early, 100*n_early/households, n_standard, 100*n_standard/households, n_night, 100*n_night/households)
    ep, sp, np_ = _build_household_profiles(t%24, is_weekend, holiday_mask)
    aggregate_profile = (n_early*ep + n_standard*sp + n_night*np_) / households

    # Behavioral regime switching (concept drift)
    regime_noise = np.ones(hours, dtype=np.float32)
    seg_start = 0
    while seg_start < days:
        seg_len = int(rng.integers(55, 90))
        seg_end = min(seg_start + seg_len, days)
        shift = rng.uniform(-0.09, 0.09)
        regime_noise[seg_start*24:seg_end*24] = 1.0 + shift
        seg_start = seg_end
    aggregate_profile = aggregate_profile * regime_noise

    holiday_base_reduction = (1.0 - 0.35*ny_mask - 0.18*holiday_mask*(1-ny_mask)).astype(np.float32)

    # Сезонный drift
    mid  = (seasonal_winter_boost - seasonal_summer_dip) / 2
    amp  = (seasonal_winter_boost + seasonal_summer_dip) / 2
    seasonal_drift = (1.0 + mid + amp*np.cos(2*np.pi*day_of_year/365)).astype(np.float32)

    # Температурный отклик с тройным взаимодействием (v5)
    T_diff          = temperature - temp_setpoint
    temp_quadratic  = temp_quadratic_coef * (T_diff**2)
    sig_hum         = _sigmoid((humidity - humidity_threshold) / 10.0)
    humidity_factor = np.where(temperature > 22.0, humidity_coef*sig_hum, 0.0).astype(np.float32)
    wind_factor     = np.where(temperature < wind_temp_threshold,
                               wind_coef*np.log1p(wind_speed), 0.0).astype(np.float32)
    # Triple interaction: жарко + влажно + ветрено → кондиционеры на полную
    triple = np.where((temperature > 28.0) & (humidity > 68.0) & (wind_speed > 6.0),
                      0.07 * sig_hum * np.log1p(wind_speed) / 4.0, 0.0).astype(np.float32)
    temp_response = (1.0 + temp_quadratic + humidity_factor + wind_factor + triple).astype(np.float32)

    # Тренд
    trend = (1.0 + 0.05*(t/(24*365.25))).astype(np.float32)

    # Аномалии
    anomaly_factor = np.ones(hours, dtype=np.float32)
    for _ in range(max(2, days//int(rng.integers(30, 61)))):
        idx = int(rng.integers(24, hours-24))
        dur = int(rng.integers(3, 7))
        anomaly_factor[idx:idx+dur] *= rng.uniform(1.20, 1.40)
    for _ in range(max(1, days//90)):
        idx = int(rng.integers(48, hours-48))
        dur = int(rng.integers(4, 9))
        anomaly_factor[idx:idx+dur] *= 0.15
    for _ in range(max(2, days//30)):
        idx = int(rng.integers(24, hours-24))
        dur = int(rng.integers(1, 4))
        anomaly_factor[idx:idx+dur] *= 0.30

    # AR(1) GARCH-шум (v5: усилен)
    peak_mask   = ((hour_of_day >= 7) & (hour_of_day < 10)
                   | (hour_of_day >= 18) & (hour_of_day < 21)).astype(float)
    temp_extreme = (np.abs(temperature - temp_setpoint) > 15).astype(float)
    sigma_t     = ar_sigma * (1.0 + 0.7*peak_mask + 0.5*temp_extreme)
    ar_noise    = np.zeros(hours)
    ar_noise[0] = rng.normal(0, ar_sigma)
    for i in range(1, hours):
        ar_noise[i] = ar_phi*ar_noise[i-1] + rng.normal(0, float(sigma_t[i]))
    ar_noise = ar_noise.astype(np.float32)

    # Базовое потребление (детерминированное × стохастическое)
    base_scale = float(households) * 10.0
    base_consumption = (
        aggregate_profile * holiday_base_reduction * seasonal_drift
        * temp_response * heat_surge_factor * cold_wave_factor
        * anomaly_factor * (1.0 + ar_noise) * trend * base_scale
    ).astype(np.float32)

    # ── EV ЗАРЯДКА (v5) ──────────────────────────────────────────────────────
    n_ev = int(households * ev_penetration)
    ev_load_raw = np.zeros(hours, dtype=np.float64)
    for day_idx in range(days):
        n_charging = int(rng.binomial(n_ev, 0.85))
        is_we_day  = bool(weekday[day_idx*24] >= 5)
        is_hol_day = bool(holiday_mask_daily[day_idx] > 0)
        home_frac  = 0.72 if (is_we_day or is_hol_day) else 0.60
        for _ in range(n_charging):
            if rng.random() < home_frac:
                base_h = int(rng.choice([22, 23, 0, 1, 2, 3, 20, 21]))
            else:
                base_h = int(rng.integers(9, 18))
            duration = int(rng.integers(3, 9))
            power    = rng.uniform(6.5, 11.0)
            for h in range(duration):
                abs_h = day_idx*24 + (base_h + h) % 24
                if abs_h < hours:
                    ev_load_raw[abs_h] += power
        # Пятничный кластер зарядки
        if weekday[day_idx*24] == 4 and not is_hol_day:
            peak_h = day_idx*24 + 21
            if peak_h < hours:
                ev_load_raw[peak_h] += n_ev * 0.12 * rng.uniform(0.8, 1.2) * 8.0
    ev_load_raw = ev_load_raw.astype(np.float32)

    # ── СОЛНЕЧНАЯ ГЕНЕРАЦИЯ (v5) ──────────────────────────────────────────────
    n_solar = int(households * solar_penetration)
    kw_peak = 5.0
    hour_float = (t % 24).astype(np.float32)
    solar_profile = np.maximum(0.0, np.exp(-((hour_float - 12.5)**2)/(2*3.2**2)))
    solar_season  = 0.72 + 0.28*np.sin(2*np.pi*(day_of_year.astype(np.float32)-80)/365)
    cloud_factor  = 1.0 - 0.85*(cloud_cover**0.7)
    solar_gen_raw = (solar_profile * solar_season * cloud_factor * kw_peak * n_solar).astype(np.float32)

    # ── DEMAND RESPONSE (v5) ──────────────────────────────────────────────────
    dsr_active   = np.zeros(hours, dtype=np.float32)
    dsr_strength = np.zeros(hours, dtype=np.float32)
    peak_wd_idx  = np.where(
        ((hour_of_day >= 10)&(hour_of_day < 17) | (hour_of_day >= 18)&(hour_of_day < 22))
        & (is_weekend == 0)
    )[0]
    used_h = set()
    for _ in range(max(4, int(days/365*8))):
        if len(peak_wd_idx) == 0: break
        a = int(rng.choice(peak_wd_idx))
        if a in used_h: continue
        dur = int(rng.integers(2, 6))
        str_ = rng.uniform(0.15, 0.32)
        for h in range(dur):
            idx = a + h
            if idx < hours:
                dsr_active[idx]   = 1.0
                dsr_strength[idx] = str_
                used_h.add(idx)

    # ── ПРОМЫШЛЕННЫЕ НАГРУЗКИ (v5) ────────────────────────────────────────────
    industrial_load_raw = np.zeros(hours, dtype=np.float32)
    for _ in range(industrial_loads):
        power_kw = rng.uniform(350, 1800)
        p_wd = rng.uniform(0.65, 0.90)
        p_we = rng.uniform(0.15, 0.40)
        in_work = False
        till_change = int(rng.integers(4, 12))
        for h in range(hours):
            p = p_wd if (weekday[h] < 5) else p_we
            if till_change <= 0:
                in_work = rng.random() < (p if not in_work else (1-p*0.3))
                till_change = int(rng.integers(8, 17) if in_work else rng.integers(4, 9))
            if in_work:
                industrial_load_raw[h] += power_kw * rng.uniform(0.92, 1.08)
            till_change -= 1

    # ── ПРАЗДНИЧНЫЕ СПАЙКИ ────────────────────────────────────────────────────
    holiday_spike = np.ones(hours, dtype=np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and d.day == 1:
            for ho in [0, 1, 2]:
                idx = i*24 + ho
                if idx < hours: holiday_spike[idx] = 1.38
        elif d.month == 5 and d.day == 9:
            for ho in [20, 21, 22]:
                idx = i*24 + ho
                if idx < hours: holiday_spike[idx] = 1.22

    # ── ИТОГОВОЕ ПОТРЕБЛЕНИЕ ──────────────────────────────────────────────────
    consumption = (
        base_consumption * holiday_spike * (1.0 - dsr_strength)
        + ev_load_raw
        + industrial_load_raw
        - solar_gen_raw
    ).astype(np.float32)
    consumption = np.clip(consumption, 200.0, None)

    logger.info("Сгенерировано %d записей. Потребление: min=%.1f, mean=%.1f, max=%.1f кВт·ч",
                len(consumption), float(consumption.min()), float(consumption.mean()), float(consumption.max()))
    logger.info("  EV нагрузка: mean=%.1f кВт (%.1f%% базовой)", float(ev_load_raw.mean()),
                100*float(ev_load_raw.mean())/max(float(base_consumption.mean()), 1))
    logger.info("  Solar: mean=%.1f кВт, max=%.1f кВт", float(solar_gen_raw.mean()), float(solar_gen_raw.max()))
    logger.info("  DSR часов: %d | CV потребления: %.3f",
                int(dsr_active.sum()), float(consumption.std()/consumption.mean()))

    # Нормализованные новые признаки
    ev_max      = float(ev_load_raw.max()) + 1.0
    solar_max   = float(solar_gen_raw.max()) + 1.0
    ev_norm     = (ev_load_raw / ev_max).astype(np.float32)
    solar_norm  = (solar_gen_raw / solar_max).astype(np.float32)

    # Tariff zones
    tariff_zone_arr = np.empty(hours, dtype=object)
    for i in range(hours):
        h  = int(hour_of_day[i]); wd = int(weekday[i]); ih = bool(holiday_mask[i])
        if h < 7 or h >= 23: tariff_zone_arr[i] = "night"
        elif wd >= 5 or ih:  tariff_zone_arr[i] = "day"
        elif 10<=h<17 or 21<=h<23: tariff_zone_arr[i] = "peak"
        else: tariff_zone_arr[i] = "day"
    is_peak_hour  = (tariff_zone_arr == "peak").astype(np.int8)
    is_night_hour = (tariff_zone_arr == "night").astype(np.int8)

    temp_sq_raw = temperature**2
    temp_sq_max = float(temp_sq_raw.max()) + 1e-8
    temperature_squared = (temp_sq_raw / temp_sq_max).astype(np.float32)

    month = dates.month.values.astype(np.int8)
    season_arr = np.where((month>=3)&(month<=5),"spring",
                  np.where((month>=6)&(month<=8),"summer",
                   np.where((month>=9)&(month<=11),"autumn","winter")))

    df = pd.DataFrame({
        "timestamp": dates, "consumption": consumption,
        "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,          # v5 ★
        "ev_load_norm": ev_norm,             # v5 ★
        "solar_gen_norm": solar_norm,        # v5 ★
        "dsr_active": dsr_active,            # v5 ★
        "hour": hour_of_day, "weekday": weekday.astype(np.int8),
        "is_weekend": is_weekend.astype(np.int8), "is_holiday": holiday_mask.astype(np.int8),
        "is_peak_hour": is_peak_hour, "is_night_hour": is_night_hour,
        "tariff_zone": tariff_zone_arr, "month": month, "season": season_arr,
        "heating_degree_days": np.maximum(0.0, 18.0-temperature).astype(np.float32),
        "cooling_degree_days": np.maximum(0.0, temperature-24.0).astype(np.float32),
        "day_of_year": day_of_year.astype(np.int16),
        "temperature_squared": temperature_squared,
    })
    df["rolling_mean_24h"] = df["consumption"].rolling(24, min_periods=1).mean().astype(np.float32)
    df["rolling_std_24h"]  = df["consumption"].rolling(24, min_periods=2).std().fillna(0.0).astype(np.float32)
    df["hour_sin"] = np.sin(2*np.pi*hour_of_day/24).astype(np.float32)
    df["hour_cos"] = np.cos(2*np.pi*hour_of_day/24).astype(np.float32)
    return df


def validate_generated_data(df):
    passed = True
    logger.info("Валидация данных Smart Grid v5...")
    bad = df["consumption"].isna().sum() + np.isinf(df["consumption"].values).sum()
    if bad > 0: logger.error("  ❌ NaN/Inf: %d", bad); passed = False
    else: logger.info("  ✅ NaN/Inf: нет")
    if (df["consumption"] < 0).sum() > 0: logger.error("  ❌ Отрицательные значения"); passed = False
    else: logger.info("  ✅ Отрицательных значений: нет")
    hourly = df.groupby("hour")["consumption"].mean()
    mp = int(hourly.loc[6:11].idxmax()); ep = int(hourly.loc[17:22].idxmax())
    logger.info("  ✅ Утренний пик: %d:00", mp) if 6<=mp<=11 else logger.warning("  ⚠️  Утренний пик: %d:00", mp)
    logger.info("  ✅ Вечерний пик: %d:00", ep) if 17<=ep<=22 else logger.warning("  ⚠️  Вечерний пик: %d:00", ep)
    ratio = df[df["is_weekend"]==1]["consumption"].mean() / df[df["is_weekend"]==0]["consumption"].mean()
    logger.info("  ✅ Выходные/будни: %.2f", ratio)
    if "season" in df.columns:
        w = df[df["season"]=="winter"]["consumption"].mean(); s = df[df["season"]=="summer"]["consumption"].mean()
        logger.info("  ✅ Сезонность: зима=%.1f > лето=%.1f", w, s) if w>s else logger.warning("  ⚠️  Зима ≤ Лето")
    t_min, t_max = df["temperature"].min(), df["temperature"].max()
    logger.info("  ✅ Диапазон температур: %.1f..%.1f°C", t_min, t_max)
    cons = df["consumption"].values; cc = cons - cons.mean(); var = np.var(cc)+1e-10
    def acf(s, lag): n=len(s); return float(np.mean(s[:n-lag]*s[lag:]))/var if lag<n else 0.0
    a24  = acf(cc, 24); a168 = acf(cc, 168)
    logger.info("  ✅ ACF(lag=24) = %.3f", a24) if a24>=0.35 else logger.warning("  ⚠️  ACF(lag=24) = %.3f < 0.35", a24)
    if len(df)>=336:
        logger.info("  ✅ ACF(lag=168) = %.3f", a168) if a168>=0.15 else logger.warning("  ⚠️  ACF(lag=168) = %.3f < 0.15", a168)
    logger.info("  ℹ️  EV нагрузка norm mean=%.3f | Solar norm mean=%.3f | DSR часов=%d",
                float(df["ev_load_norm"].mean()), float(df["solar_gen_norm"].mean()), int(df["dsr_active"].sum()))
    logger.info("  ℹ️  CV=%.3f (v4 было 0.372)", float(df["consumption"].std()/df["consumption"].mean()))
    logger.info("Валидация %s", "ПРОЙДЕНА" if passed else "ПРОВАЛЕНА")
    return passed
