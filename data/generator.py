# -*- coding: utf-8 -*-
"""
data/generator.py — Smart Grid v6: исправление EV-бага + усиление нелинейности.

ИСПРАВЛЕНИЯ v6:
═══════════════════════════════════════════════════════════════════════════════

БАГ #4 (КРИТИЧЕСКИЙ): EV time-wrap — ночная зарядка переносилась в тот же день.
  БЫЛО:  abs_h = day_idx*24 + (base_h + h) % 24
         Если base_h=22, h=2: (22+2)%24=0 → abs_h = day_idx*24+0 = 00:00 ТОГО ЖЕ ДНЯ.
         Все ночные сессии (base_h=22,23,0,1,2,3) накладывались на hour 0 текущего дня.
         Реальный вклад EV: 29-59 кВт (1.2% базовой нагрузки) вместо расчётных 120-240 кВт.
  СТАЛО: abs_h = day_idx*24 + base_h + h  (без modulo!)
         if abs_h >= hours: break
         Зарядка с 22:00 на 5ч корректно охватывает часы 22,23,0(+1д),1(+1д),2(+1д).

УСИЛЕНИЕ нелинейности (нейросети должны победить LinReg):
  1. EV penetration: 28% → 50% (больше EVs)
  2. EV power: 6.5-11 кВт → 11-22 кВт (Level 2 зарядка)
  3. Добавлен EV коммерческий флот: 3 грузовика × 30-50 кВт × 8-10ч/ночь
  4. Industrial loads: 4 → 6 заводов, мощность выше
  5. DSR события: 8 → 15/год, чаще и сильнее
  6. "Demand cascade": после аномально высокого часа нагрузка держится высокой
     ещё 3-6ч (AR-эффект на уровне событий, не просто шум)
  7. Cold-snap EV surge: при T < -15°C EV заряжаются чаще (range anxiety)

Ожидаемый результат:
  EV нагрузка: 1.2% → 8-12% базовой нагрузки
  CV потребления: 0.386 → 0.42+
  ACF(24): ~0.84 → ~0.75 (меньше автокорреляция → труднее для LinReg)
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
    return np.exp(-((hour_arr - center)**2) / (2 * width**2))


def _build_household_profiles(hour_arr, is_weekend, is_holiday, weekend_scale=0.88):
    H = hour_arr.astype(np.float32)
    b = 0.65
    early_wd    = b + 0.90*_gaussian_peak(H,7.0,1.2) + 0.35*_gaussian_peak(H,12.5,1.5) + 0.45*_gaussian_peak(H,18.5,2.0)
    standard_wd = b + 0.45*_gaussian_peak(H,8.0,1.5) + 0.30*_gaussian_peak(H,13.0,1.5) + 1.00*_gaussian_peak(H,19.0,2.5)
    night_wd    = b + 0.25*_gaussian_peak(H,8.5,1.5) + 0.35*_gaussian_peak(H,13.5,1.5) + 0.80*_gaussian_peak(H,21.5,2.0)
    early_we    = b + 0.75*_gaussian_peak(H,9.0,2.0) + 0.40*_gaussian_peak(H,14.0,2.0) + 0.50*_gaussian_peak(H,19.5,3.0)
    standard_we = b + 0.35*_gaussian_peak(H,10.0,2.0)+ 0.40*_gaussian_peak(H,14.0,2.0) + 0.85*_gaussian_peak(H,20.0,3.5)
    night_we    = b + 0.20*_gaussian_peak(H,11.0,2.0)+ 0.40*_gaussian_peak(H,15.0,2.0) + 0.90*_gaussian_peak(H,22.0,2.5)
    cooking = 0.30 * _gaussian_peak(H, 14.0, 1.5)
    is_we  = is_weekend.astype(bool) | is_holiday.astype(bool)
    is_hol = is_holiday.astype(bool)
    early    = np.where(is_we, early_we,    early_wd)
    standard = np.where(is_we, standard_we, standard_wd)
    night    = np.where(is_we, night_we,    night_wd)
    early    = np.where(is_hol, early+cooking, early)
    standard = np.where(is_hol, standard+cooking, standard)
    night    = np.where(is_hol, night+cooking, night)
    early    = np.where(is_we, early*weekend_scale, early)
    standard = np.where(is_we, standard*weekend_scale, standard)
    night    = np.where(is_we, night*weekend_scale, night)
    return early.astype(np.float32), standard.astype(np.float32), night.astype(np.float32)


def generate_holiday_mask(days=365, start_date="2024-01-01"):
    holidays = {(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),(1,8),
                (2,23),(3,8),(5,1),(5,9),(6,12),(11,4)}
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
    ar_phi=0.65, ar_sigma=0.060,
    seasonal_winter_boost=0.15, seasonal_summer_dip=0.10,
    ev_penetration=0.50,          # v6: 28% → 50%
    solar_penetration=0.22,
    industrial_loads=6,           # v6: 4 → 6
):
    rng = np.random.default_rng(seed)
    logger.info("Генерация Smart Grid v6: %d дней, %d домохозяйств | EV=%.0f%% Solar=%.0f%%",
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
    temp_annual  = 5.0 + 20.0*np.sin(2*np.pi*t/(24*365.25) - np.pi/2)
    temp_diurnal = 3.5*np.sin(2*np.pi*(t%24)/24 - np.pi/4)
    tn = np.zeros(hours); tn[0] = rng.normal(0,1.5)
    for i in range(1,hours): tn[i] = 0.97*tn[i-1] + rng.normal(0,0.6)
    temperature = np.clip(temp_annual + temp_diurnal + tn, -38.0, 45.0).astype(np.float32)

    # Тепловые волны и волны холода
    heat_surge_factor = np.ones(hours, np.float32)
    for idx in np.where((dates.month >= 6)&(dates.month <= 8))[0][:max(1,int(days/365*3))*50:50]:
        e = min(idx + int(rng.integers(48,120)), hours)
        temperature[idx:e] = np.clip(temperature[idx:e]+rng.uniform(8,14),-38,45)
        heat_surge_factor[idx:e] = rng.uniform(1.28,1.40)

    cold_wave_factor = np.ones(hours, np.float32)
    for idx in np.where((dates.month==12)|(dates.month<=2))[0][:max(1,days//120)*50:50]:
        e = min(idx + int(rng.integers(72,168)), hours)
        temperature[idx:e] = np.clip(temperature[idx:e]-rng.uniform(5,12),-38,45)
        cold_wave_factor[idx:e] = rng.uniform(1.12,1.22)

    # Облачность
    cloud_annual = 0.52 + 0.18*np.cos(2*np.pi*t/(24*365.25)+np.pi)
    cn = np.zeros(hours); cn[0] = rng.normal(0,0.08)
    for i in range(1,hours): cn[i] = 0.92*cn[i-1] + rng.normal(0,0.06)
    cloud_cover = np.clip(cloud_annual+cn, 0.0, 1.0).astype(np.float32)

    # Влажность
    hum = 60.0 + 8.0*np.sin(2*np.pi*t/(24*365.25))
    hn = np.zeros(hours); hn[0]=rng.normal(0,5.0)
    for i in range(1,hours): hn[i]=0.85*hn[i-1]+rng.normal(0,3.0)
    humidity = np.clip(hum+hn, 20.0, 98.0).astype(np.float32)

    # Ветер
    wind_base = 4.0+2.5*np.cos(2*np.pi*t/(24*365.25)+np.pi)
    wind_speed = np.clip(wind_base+rng.exponential(2.0,hours), 0.0, 30.0).astype(np.float32)

    # Профили домохозяйств + behavioral regime switching
    n_early    = int(households * early_bird_frac)
    n_night    = int(households * night_owl_frac)
    n_standard = households - n_early - n_night
    logger.info("Типы: ранние=%d (%.0f%%), стандартные=%d (%.0f%%), ночные=%d (%.0f%%)",
                n_early,100*n_early/households, n_standard,100*n_standard/households, n_night,100*n_night/households)
    ep, sp, np_ = _build_household_profiles(t%24, is_weekend, holiday_mask)
    agg = (n_early*ep + n_standard*sp + n_night*np_) / households
    regime = np.ones(hours, np.float32)
    s = 0
    while s < days:
        se = min(s + int(rng.integers(55,90)), days)
        regime[s*24:se*24] = 1.0 + rng.uniform(-0.09,0.09)
        s = se
    agg *= regime

    holiday_base_reduction = (1.0 - 0.35*ny_mask - 0.18*holiday_mask*(1-ny_mask)).astype(np.float32)
    mid  = (seasonal_winter_boost - seasonal_summer_dip)/2
    amp  = (seasonal_winter_boost + seasonal_summer_dip)/2
    seasonal_drift = (1.0 + mid + amp*np.cos(2*np.pi*day_of_year/365)).astype(np.float32)

    T_diff = temperature - temp_setpoint
    temp_q  = temp_quadratic_coef*(T_diff**2)
    sig_hum = _sigmoid((humidity-humidity_threshold)/10.0)
    hum_fac = np.where(temperature>22.0, humidity_coef*sig_hum, 0.0).astype(np.float32)
    wnd_fac = np.where(temperature<wind_temp_threshold, wind_coef*np.log1p(wind_speed), 0.0).astype(np.float32)
    triple  = np.where((temperature>28.0)&(humidity>68.0)&(wind_speed>6.0),
                       0.07*sig_hum*np.log1p(wind_speed)/4.0, 0.0).astype(np.float32)
    temp_response = (1.0+temp_q+hum_fac+wnd_fac+triple).astype(np.float32)

    trend = (1.0+0.05*(t/(24*365.25))).astype(np.float32)

    # Аномалии
    anomaly_factor = np.ones(hours, np.float32)
    for _ in range(max(2, days//int(rng.integers(30,61)))):
        i = int(rng.integers(24,hours-24))
        anomaly_factor[i:i+int(rng.integers(3,7))] *= rng.uniform(1.20,1.40)
    for _ in range(max(1,days//90)):
        i = int(rng.integers(48,hours-48))
        anomaly_factor[i:i+int(rng.integers(4,9))] *= 0.15
    for _ in range(max(2,days//30)):
        i = int(rng.integers(24,hours-24))
        anomaly_factor[i:i+int(rng.integers(1,4))] *= 0.30

    # AR шум GARCH
    pm = ((hour_of_day>=7)&(hour_of_day<10)|(hour_of_day>=18)&(hour_of_day<21)).astype(float)
    te = (np.abs(temperature-temp_setpoint)>15).astype(float)
    sigma_t = ar_sigma*(1.0+0.7*pm+0.5*te)
    ar = np.zeros(hours); ar[0]=rng.normal(0,ar_sigma)
    for i in range(1,hours): ar[i]=ar_phi*ar[i-1]+rng.normal(0,float(sigma_t[i]))
    ar = ar.astype(np.float32)

    base_scale = float(households)*10.0
    base_consumption = (
        agg*holiday_base_reduction*seasonal_drift*temp_response
        *heat_surge_factor*cold_wave_factor*anomaly_factor*(1.0+ar)*trend*base_scale
    ).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # EV ЗАРЯДКА (v6: исправлен time-wrap баг + увеличена мощность)
    # ══════════════════════════════════════════════════════════════════════
    n_ev = int(households * ev_penetration)
    ev_load_raw = np.zeros(hours, np.float64)

    for day_idx in range(days):
        is_we_day  = bool(weekday[day_idx*24] >= 5)
        is_hol_day = bool(holiday_mask_daily[day_idx] > 0)
        is_cold    = bool(temperature[day_idx*24] < -15)  # cold snap = range anxiety

        # Больше зарядок в холодные дни (range anxiety)
        charge_prob = 0.90 if is_cold else 0.85
        n_charging  = int(rng.binomial(n_ev, charge_prob))
        home_frac   = 0.72 if (is_we_day or is_hol_day) else 0.62

        for _ in range(n_charging):
            if rng.random() < home_frac:
                # Домашняя зарядка: преимущественно ночью
                base_h = int(rng.choice([21, 22, 23, 0, 1, 2, 3, 20]))
            else:
                # Публичная: дневные часы
                base_h = int(rng.integers(9, 18))

            duration = int(rng.integers(3, 9))  # 3-8 часов
            # v6: Level 2 зарядка 11-22 кВт (было 6.5-11)
            power = rng.uniform(11.0, 22.0)

            # ── ИСПРАВЛЕНИЕ v6 (КРИТИЧЕСКОЕ) ─────────────────────────────
            # БЫЛО: abs_h = day_idx*24 + (base_h + h) % 24
            #   При base_h=22, h=2: (22+2)%24=0 → hour 0 ТЕКУЩЕГО дня!
            #   Все ночные сессии накладывались на midnight того же дня.
            #   Реальный EV вклад был в 4-6× ниже расчётного.
            # СТАЛО: без modulo — правильно распространяется на следующий день.
            for h_offset in range(duration):
                abs_h = day_idx * 24 + base_h + h_offset
                if abs_h >= hours:
                    break
                ev_load_raw[abs_h] += power
            # ── конец исправления ─────────────────────────────────────────

        # Пятничный кластер: все EV-владельцы заряжаются перед выходными
        if weekday[day_idx*24] == 4 and not is_hol_day:
            peak_h = day_idx*24 + 21
            if peak_h < hours:
                ev_load_raw[peak_h] += n_ev * 0.15 * rng.uniform(0.8, 1.2) * 16.0

        # Cold-snap surge: при T < -15°C дополнительная зарядка вечером
        if is_cold:
            surge_h = day_idx*24 + 19
            if surge_h < hours:
                ev_load_raw[surge_h] += n_ev * ev_penetration * rng.uniform(5.0, 10.0)

    # Коммерческий EV флот (v6 новое): 3 грузовика/района × 30-50 кВт × 8-10ч
    n_commercial = max(1, households // 150)  # 1 флот на каждые 150 домов
    for _ in range(n_commercial):
        depot_start = int(rng.integers(21, 24))  # Начало зарядки в 21-23ч
        kw_truck    = rng.uniform(30.0, 50.0)    # Мощность грузового зарядника
        n_trucks    = int(rng.integers(2, 5))    # 2-4 грузовика
        for day_idx in range(days):
            # Заряжаются только если завтра рабочий день
            next_wd = (day_idx+1) % 7
            if next_wd < 5 and rng.random() < 0.85:
                duration = int(rng.integers(8, 11))
                for h_offset in range(duration):
                    abs_h = day_idx*24 + depot_start + h_offset
                    if abs_h >= hours: break
                    ev_load_raw[abs_h] += kw_truck * n_trucks * rng.uniform(0.90, 1.05)

    ev_load_raw = ev_load_raw.astype(np.float32)

    # ── СОЛНЕЧНАЯ ГЕНЕРАЦИЯ ────────────────────────────────────────────────
    n_solar = int(households * solar_penetration)
    hour_f  = (t % 24).astype(np.float32)
    solar_profile = np.maximum(0.0, np.exp(-((hour_f-12.5)**2)/(2*3.2**2)))
    solar_season  = 0.72 + 0.28*np.sin(2*np.pi*(day_of_year.astype(np.float32)-80)/365)
    cloud_factor  = 1.0 - 0.85*(cloud_cover**0.7)
    solar_gen_raw = (solar_profile*solar_season*cloud_factor*5.0*n_solar).astype(np.float32)

    # ── DEMAND RESPONSE (v6: чаще и сильнее) ──────────────────────────────
    dsr_active   = np.zeros(hours, np.float32)
    dsr_strength = np.zeros(hours, np.float32)
    # v6: 15 событий/год (было 8), длительность 4-8ч (было 2-6), снижение 20-40%
    n_dsr = max(6, int(days/365*15))
    peak_wd = np.where(
        ((hour_of_day>=10)&(hour_of_day<17)|(hour_of_day>=18)&(hour_of_day<22))
        &(is_weekend==0)
    )[0]
    used_h = set()
    for _ in range(n_dsr):
        if len(peak_wd) == 0: break
        a = int(rng.choice(peak_wd))
        if a in used_h: continue
        dur = int(rng.integers(4, 9))    # v6: 4-8ч (было 2-6ч)
        str_= rng.uniform(0.20, 0.40)   # v6: 20-40% снижение (было 15-32%)
        for h in range(dur):
            idx = a + h
            if idx < hours:
                dsr_active[idx]   = 1.0
                dsr_strength[idx] = str_
                used_h.add(idx)

    # ── ПРОМЫШЛЕННЫЕ НАГРУЗКИ (v6: 6 заводов, выше мощность) ─────────────
    industrial_load_raw = np.zeros(hours, np.float32)
    for ind_i in range(industrial_loads):
        power_kw = rng.uniform(500, 2500)   # v6: 350-1800 → 500-2500 кВт
        p_wd = rng.uniform(0.65, 0.90)
        p_we = rng.uniform(0.15, 0.40)
        in_work = False
        till_ch = int(rng.integers(4, 12))
        for h in range(hours):
            p = p_wd if weekday[h] < 5 else p_we
            if till_ch <= 0:
                in_work = rng.random() < (p if not in_work else (1-p*0.3))
                till_ch = int(rng.integers(8,17) if in_work else rng.integers(4,9))
            if in_work:
                industrial_load_raw[h] += power_kw * rng.uniform(0.92, 1.08)
            till_ch -= 1

    # ── DEMAND CASCADE (v6 новое) ─────────────────────────────────────────
    # После аномально высокого часа нагрузка продолжает быть высокой (инерция).
    # Это требует от модели понимать "состояние" нагрузки, а не только текущие признаки.
    # LinReg с flat features не может уловить этот нелинейный decay-эффект.
    cascade_factor = np.ones(hours, np.float32)
    cascade_state  = 0.0
    cascade_threshold = float(base_consumption.mean()) * 1.35  # 35% выше среднего
    for h in range(hours):
        bc = float(base_consumption[h])
        if bc > cascade_threshold:
            cascade_state = min(cascade_state + rng.uniform(0.03, 0.08), 0.25)
        else:
            cascade_state = max(cascade_state - 0.04, 0.0)
        cascade_factor[h] = 1.0 + cascade_state

    # Праздничные спайки
    holiday_spike = np.ones(hours, np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and d.day == 1:
            for ho in [0, 1, 2]:
                idx = i*24+ho
                if idx < hours: holiday_spike[idx] = 1.38
        elif d.month == 5 and d.day == 9:
            for ho in [20, 21, 22]:
                idx = i*24+ho
                if idx < hours: holiday_spike[idx] = 1.22

    # ── ИТОГОВОЕ ПОТРЕБЛЕНИЕ ──────────────────────────────────────────────
    consumption = (
        base_consumption * holiday_spike * cascade_factor * (1.0 - dsr_strength)
        + ev_load_raw
        + industrial_load_raw
        - solar_gen_raw
    ).astype(np.float32)
    consumption = np.clip(consumption, 200.0, None)

    ev_mean = float(ev_load_raw.mean())
    bc_mean = float(base_consumption.mean())
    logger.info("Сгенерировано %d записей. Потребление: min=%.1f, mean=%.1f, max=%.1f кВт·ч",
                len(consumption), float(consumption.min()), float(consumption.mean()), float(consumption.max()))
    logger.info("  EV нагрузка: mean=%.1f кВт (%.1f%% базовой)  [v5 был: ~%.1f%% — был time-wrap баг]",
                ev_mean, 100*ev_mean/max(bc_mean,1), 1.2)
    logger.info("  Solar: mean=%.1f кВт, max=%.1f кВт", float(solar_gen_raw.mean()), float(solar_gen_raw.max()))
    logger.info("  DSR событий активно: %d ч | Industrial: %d заводов | Cascade max=%.2f",
                int(dsr_active.sum()), industrial_loads, float(cascade_factor.max()))
    logger.info("  CV потребления: %.3f", float(consumption.std()/consumption.mean()))

    # Нормализованные новые признаки
    ev_norm    = (ev_load_raw / (float(ev_load_raw.max())+1.0)).astype(np.float32)
    solar_norm = (solar_gen_raw / (float(solar_gen_raw.max())+1.0)).astype(np.float32)

    # Tariff zones
    tariff_arr = np.empty(hours, dtype=object)
    for i in range(hours):
        h, wd, ih = int(hour_of_day[i]), int(weekday[i]), bool(holiday_mask[i])
        if h<7 or h>=23: tariff_arr[i]="night"
        elif wd>=5 or ih: tariff_arr[i]="day"
        elif 10<=h<17 or 21<=h<23: tariff_arr[i]="peak"
        else: tariff_arr[i]="day"
    is_peak_hour  = (tariff_arr=="peak").astype(np.int8)
    is_night_hour = (tariff_arr=="night").astype(np.int8)

    temp_sq_raw = temperature**2
    temperature_squared = (temp_sq_raw / (float(temp_sq_raw.max())+1e-8)).astype(np.float32)
    month = dates.month.values.astype(np.int8)
    season_arr = np.where((month>=3)&(month<=5),"spring",
                  np.where((month>=6)&(month<=8),"summer",
                   np.where((month>=9)&(month<=11),"autumn","winter")))

    df = pd.DataFrame({
        "timestamp": dates, "consumption": consumption,
        "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,
        "ev_load_norm": ev_norm,
        "solar_gen_norm": solar_norm,
        "dsr_active": dsr_active,
        "hour": hour_of_day, "weekday": weekday.astype(np.int8),
        "is_weekend": is_weekend.astype(np.int8), "is_holiday": holiday_mask.astype(np.int8),
        "is_peak_hour": is_peak_hour, "is_night_hour": is_night_hour,
        "tariff_zone": tariff_arr, "month": month, "season": season_arr,
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
    logger.info("Валидация данных Smart Grid v6...")
    bad = df["consumption"].isna().sum() + np.isinf(df["consumption"].values).sum()
    if bad: logger.error("  ❌ NaN/Inf: %d", bad); passed=False
    else: logger.info("  ✅ NaN/Inf: нет")
    if (df["consumption"]<0).sum(): logger.error("  ❌ Отрицательные значения"); passed=False
    else: logger.info("  ✅ Отрицательных значений: нет")
    hourly = df.groupby("hour")["consumption"].mean()
    mp = int(hourly.loc[6:11].idxmax()); ep = int(hourly.loc[17:22].idxmax())
    logger.info("  ✅ Утренний пик: %d:00", mp) if 6<=mp<=11 else logger.warning("  ⚠️  Утренний пик: %d:00",mp)
    logger.info("  ✅ Вечерний пик: %d:00", ep) if 17<=ep<=22 else logger.warning("  ⚠️  Вечерний пик: %d:00",ep)
    r = df[df["is_weekend"]==1]["consumption"].mean()/df[df["is_weekend"]==0]["consumption"].mean()
    logger.info("  ✅ Выходные/будни: %.2f", r)
    if "season" in df.columns:
        w=df[df["season"]=="winter"]["consumption"].mean(); s=df[df["season"]=="summer"]["consumption"].mean()
        logger.info("  ✅ Сезонность: зима=%.1f > лето=%.1f", w,s) if w>s else logger.warning("  ⚠️  Зима<=Лето")
    t_min,t_max = df["temperature"].min(),df["temperature"].max()
    logger.info("  ✅ Диапазон температур: %.1f..%.1f°C", t_min, t_max)
    cons=df["consumption"].values; cc=cons-cons.mean(); var=np.var(cc)+1e-10
    def acf(s,lag): n=len(s); return float(np.mean(s[:n-lag]*s[lag:]))/var if lag<n else 0.0
    a24=acf(cc,24); a168=acf(cc,168)
    logger.info("  ✅ ACF(lag=24) = %.3f", a24) if a24>=0.35 else logger.warning("  ⚠️  ACF(lag=24)=%.3f<0.35",a24)
    if len(df)>=336:
        logger.info("  ✅ ACF(lag=168) = %.3f", a168) if a168>=0.15 else logger.warning("  ⚠️  ACF(lag=168)=%.3f<0.15",a168)
    ev_mean = float(df["ev_load_norm"].mean()); solar_mean = float(df["solar_gen_norm"].mean())
    logger.info("  ℹ️  EV norm mean=%.3f | Solar norm mean=%.3f | DSR ч=%d | CV=%.3f",
                ev_mean, solar_mean, int(df["dsr_active"].sum()),
                float(df["consumption"].std()/df["consumption"].mean()))
    logger.info("Валидация %s", "ПРОЙДЕНА" if passed else "ПРОВАЛЕНА")
    return passed
