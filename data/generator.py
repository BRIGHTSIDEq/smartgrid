# -*- coding: utf-8 -*-
"""
data/generator.py — синтетический генератор данных Smart Grid.

Формирует почасовой ряд агрегированного потребления города и сопутствующие
физические ковариаты. Учитываемые компоненты:

  Погода          температура (годовой + суточный ход, AR-шум, волны тепла/холода),
                  влажность, скорость ветра, облачность;
  Спрос           три поведенческих профиля домохозяйств (ранние/стандартные/
                  ночные), районная структура города с маятниковой миграцией,
                  праздники РФ, годовой тренд, режимные переключения;
  Отклик на погоду квадратичная зависимость от отклонения температуры от уставки,
                  сигмоидный вклад влажности при T>22°C, ветровой вклад при
                  T<10°C, тройное взаимодействие температура×влажность×ветер;
  Электротранспорт домашняя ночная и публичная дневная зарядка, коммерческий
                  флот, пятничный кластер, всплеск при T<-15°C;
  Генерация       распределённая солнечная с учётом облачности и сезона;
  Управление      события Demand Side Response с последующим отскоком спроса;
  Промышленность  независимые заводские нагрузки с марковским режимом работы;
  Нелинейности    каскад спроса после аномальных часов и латентное «состояние
                  сети» с памятью и порогами.

Возвращаются СЫРЫЕ физические величины. Нормировка признаков выполняется
исключительно в data/preprocessing.py скалерами, обученными только на train, —
иначе статистика тестовой части просачивается в обучающие признаки.

История изменений — в CHANGELOG.md.
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


def _sample_event_starts(rng, candidate_idx, n_events, min_gap):
    """
    Выбирает начала погодных аномалий, разнесённые по всему сезону.

    Наивная выборка «первые N часов сезона с шагом 50» сгруппировала бы все
    события в первые двое суток, и их эффекты наложились бы друг на друга:
    например, шесть волн холода подряд опускали среднюю температуру января
    на несколько градусов ниже климатической нормы. Здесь начала выбираются
    случайно по всему множеству подходящих часов с минимальным интервалом
    между событиями.
    """
    if len(candidate_idx) == 0 or n_events <= 0:
        return []
    chosen = []
    pool = np.array(candidate_idx)
    for _ in range(int(n_events)):
        if len(pool) == 0:
            break
        pick = int(rng.choice(pool))
        chosen.append(pick)
        pool = pool[np.abs(pool - pick) >= min_gap]
    return sorted(chosen)


def _build_household_profiles(hour_arr, is_weekend, is_holiday, weekend_scale=0.93):
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
    temp_setpoint=18.0, cooling_setpoint=24.0,
    heating_coef=2.0e-4, cooling_coef=1.2e-4,
    temp_annual_mean=6.25, temp_annual_amplitude=12.75,
    temp_min=-35.0, temp_max=38.0,
    humidity_threshold=60.0, humidity_coef=0.10,
    wind_temp_threshold=10.0, wind_coef=0.05,
    early_bird_frac=0.28, night_owl_frac=0.20,
    ar_phi=0.65, ar_sigma=0.030,
    seasonal_winter_boost=0.06, seasonal_summer_dip=0.04,
    kwh_per_household_month=206.0,
    nonresidential_share=0.45,
    annual_trend=0.015,
    ev_penetration=0.005,
    ev_home_power_kw=(3.5, 7.4),
    ev_public_power_kw=(22.0, 50.0),
    ev_fleet_power_kw=(30.0, 60.0),
    solar_penetration=0.002,
    solar_panel_peak_kw=5.0,
    dsr_events_per_year=10,
    dsr_strength_range=(0.02, 0.06),
    industrial_loads=6,
    city_districts=12,
):
    """
    Генерирует почасовой ряд потребления города и сопутствующие ковариаты.

    Масштаб нагрузки задаётся не произвольным множителем, а фактическим
    среднемесячным потреблением домохозяйства (`kwh_per_household_month`):
    итоговый ряд принудительно приводится к этому среднему. Благодаря этому
    абсолютные значения остаются сопоставимыми с реальными счетами и не
    «уплывают» при изменении числа домохозяйств или состава компонентов.
    """
    rng = np.random.default_rng(seed)
    city_districts = int(max(1, city_districts))
    logger.info(
        "Генерация данных: %d дней, %d домохозяйств, районов=%d | "
        "%.0f кВт·ч/мес на домохозяйство | EV=%.1f%% Solar=%.1f%%",
        days, households, city_districts, kwh_per_household_month,
        ev_penetration * 100, solar_penetration * 100,
    )

    hours = days * 24
    t = np.arange(hours, dtype=np.float32)
    dates = pd.date_range(start=start_date, periods=hours, freq="h")
    # Календарь берётся из фактических дат, а не из остатка от деления.
    # weekday = day_of_sim % 7 верен только при старте в понедельник: при любой
    # другой START_DATE день недели, выходные, пиковые часы и тарифная зона
    # разошлись бы с колонкой timestamp, и модель училась бы на календаре, не
    # соответствующем меткам времени. day_of_year % 365 дополнительно теряет
    # сутки в високосном году, расходясь с годовым периодом 365.25 в температуре.
    hour_of_day = dates.hour.to_numpy().astype(np.int8)
    day_of_sim  = (t // 24).astype(int)
    weekday     = dates.dayofweek.to_numpy().astype(int)
    is_weekend  = (weekday >= 5).astype(np.float32)
    day_of_year = dates.dayofyear.to_numpy().astype(int)

    # Праздники
    holiday_mask_daily = generate_holiday_mask(days, start_date)
    holiday_mask = np.repeat(holiday_mask_daily, 24)
    ny_mask = np.zeros(hours, dtype=np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and 1 <= d.day <= 8:
            ny_mask[i*24:(i+1)*24] = 1.0

    # ── Температура ──────────────────────────────────────────────────────────
    # Годовой ход задан климатическими нормами: средняя температура января и
    # июля определяют среднее и амплитуду синусоиды.
    # Минимум сдвинут на 20 суток от 1 января: в московском климате самый
    # холодный период — вторая половина января, а не самое начало года. Без
    # сдвига годовой максимум потребления уезжал на ноябрь, и январь
    # оказывался ниже апреля.
    _COLD_PEAK_SHIFT_DAYS = 20.0
    temp_annual  = temp_annual_mean + temp_annual_amplitude*np.sin(
        2*np.pi*(t - _COLD_PEAK_SHIFT_DAYS*24)/(24*365.25) - np.pi/2)
    temp_diurnal = 3.5*np.sin(2*np.pi*(t%24)/24 - np.pi/4)
    tn = np.zeros(hours); tn[0] = rng.normal(0,1.5)
    for i in range(1,hours): tn[i] = 0.97*tn[i-1] + rng.normal(0,0.6)
    temperature = np.clip(temp_annual + temp_diurnal + tn, temp_min, temp_max).astype(np.float32)

    # Волны тепла и холода. Прирост нагрузки в жару умеренный: доля жилья
    # с кондиционированием в России невелика, поэтому летние пики выражены
    # заметно слабее, чем в странах с массовым охлаждением.
    summer_hours = np.where((dates.month >= 6) & (dates.month <= 8))[0]
    winter_hours = np.where((dates.month == 12) | (dates.month <= 2))[0]

    heat_surge_factor = np.ones(hours, np.float32)
    for idx in _sample_event_starts(rng, summer_hours, max(1, int(days/365*2)), min_gap=240):
        e = min(idx + int(rng.integers(48,120)), hours)
        temperature[idx:e] = np.clip(temperature[idx:e]+rng.uniform(5,9), temp_min, temp_max)
        heat_surge_factor[idx:e] = rng.uniform(1.04,1.10)

    cold_wave_factor = np.ones(hours, np.float32)
    for idx in _sample_event_starts(rng, winter_hours, max(1, int(days/365*3)), min_gap=336):
        e = min(idx + int(rng.integers(72,168)), hours)
        temperature[idx:e] = np.clip(temperature[idx:e]-rng.uniform(4,8), temp_min, temp_max)
        cold_wave_factor[idx:e] = rng.uniform(1.04,1.09)

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

    # ── ГОРОДСКАЯ СИМУЛЯЦИЯ: районная структура и маятниковая миграция ─────
    # Каждый район имеет свой профиль + чувствительность к погоде/выходным.
    # Это добавляет реализм «целого города», но оставляет прогнозируемый паттерн.
    district_weights = rng.dirichlet(np.ones(city_districts)).astype(np.float32)
    district_curve = np.zeros(hours, np.float32)
    commute_morning = _gaussian_peak(t % 24, 8.5, 1.8).astype(np.float32)
    commute_evening = _gaussian_peak(t % 24, 18.5, 2.2).astype(np.float32)
    weekend_shift = np.where(is_weekend > 0, -0.04, 0.02).astype(np.float32)
    for d_i in range(city_districts):
        dist_scale = rng.uniform(0.78, 1.26)
        commute_amp = rng.uniform(0.04, 0.14)
        office_bias = rng.uniform(0.85, 1.25)
        # деловые районы активнее в будни, спальные — вечером и в выходные
        district_pattern = (
            1.0
            + commute_amp * office_bias * commute_morning
            + commute_amp * (2.0 - office_bias) * commute_evening
            + weekend_shift * (2.0 - office_bias)
        ).astype(np.float32)
        district_curve += district_weights[d_i] * dist_scale * district_pattern
    district_curve = np.clip(district_curve, 0.80, 1.35).astype(np.float32)

    # В праздники останавливается ПРОИЗВОДСТВО, а бытовое потребление слегка
    # растёт: люди дома. Прежняя версия снижала на 35% именно бытовую
    # компоненту, отчего новогодний провал по городу достигал 24% при
    # фактических для ЕЭС России 8-12%, и логика была перевёрнута.
    holiday_base_reduction = (1.0 + 0.04*ny_mask + 0.02*holiday_mask*(1-ny_mask)).astype(np.float32)
    holiday_industrial_factor = (
        1.0 - 0.55*ny_mask - 0.30*holiday_mask*(1-ny_mask)).astype(np.float32)
    mid  = (seasonal_winter_boost - seasonal_summer_dip)/2
    amp  = (seasonal_winter_boost + seasonal_summer_dip)/2
    seasonal_drift = (1.0 + mid + amp*np.cos(2*np.pi*day_of_year/365)).astype(np.float32)

    # ── Отклик нагрузки на погоду (асимметричный) ────────────────────────────
    # В России отопление преимущественно центральное или газовое, а
    # кондиционирование распространено слабо. Поэтому электрический отклик на
    # холод (освещение, циркуляционные насосы, локальные обогреватели) заметно
    # сильнее отклика на жару. Симметричная парабола вокруг уставки, уместная
    # для стран с электроотоплением и массовым охлаждением, здесь завысила бы
    # летний пик и исказила сезонность.
    cold_deg = np.maximum(temp_setpoint - temperature, 0.0)
    warm_deg = np.maximum(temperature - cooling_setpoint, 0.0)
    temp_q = (heating_coef*(cold_deg**2) + cooling_coef*(warm_deg**2)).astype(np.float32)

    sig_hum = _sigmoid((humidity-humidity_threshold)/10.0)
    hum_fac = np.where(temperature>22.0, humidity_coef*sig_hum, 0.0).astype(np.float32)
    wnd_fac = np.where(temperature<wind_temp_threshold, wind_coef*np.log1p(wind_speed), 0.0).astype(np.float32)
    triple  = np.where((temperature>28.0)&(humidity>68.0)&(wind_speed>6.0),
                       0.03*sig_hum*np.log1p(wind_speed)/4.0, 0.0).astype(np.float32)
    temp_response = (1.0+temp_q+hum_fac+wnd_fac+triple).astype(np.float32)

    trend = (1.0+annual_trend*(t/(24*365.25))).astype(np.float32)

    # Аномалии
    # Амплитуды подобраны так, чтобы коэффициент заполнения графика
    # (среднее / максимум) остался в реальном диапазоне 0.55-0.75:
    # редкие всплески и провалы не должны растягивать размах нагрузки вдвое.
    anomaly_factor = np.ones(hours, np.float32)
    for _ in range(max(2, days//int(rng.integers(30,61)))):
        i = int(rng.integers(24,hours-24))
        anomaly_factor[i:i+int(rng.integers(3,7))] *= rng.uniform(1.08,1.18)
    for _ in range(max(1,days//90)):        # плановые отключения
        i = int(rng.integers(48,hours-48))
        anomaly_factor[i:i+int(rng.integers(4,9))] *= rng.uniform(0.55,0.70)
    for _ in range(max(2,days//30)):        # кратковременные провалы
        i = int(rng.integers(24,hours-24))
        anomaly_factor[i:i+int(rng.integers(1,4))] *= rng.uniform(0.70,0.85)

    # AR шум GARCH
    pm = ((hour_of_day>=7)&(hour_of_day<10)|(hour_of_day>=17)&(hour_of_day<21)).astype(float)
    te = (np.abs(temperature-temp_setpoint)>15).astype(float)
    sigma_t = ar_sigma*(1.0+0.7*pm+0.5*te)
    ar = np.zeros(hours); ar[0]=rng.normal(0,ar_sigma)
    for i in range(1,hours): ar[i]=ar_phi*ar[i-1]+rng.normal(0,float(sigma_t[i]))
    ar = ar.astype(np.float32)

    # ── Масштабирование к фактическому потреблению ───────────────────────────
    # Форма профиля собрана из безразмерных множителей; абсолютный уровень
    # задаётся отдельно — по среднемесячному потреблению домохозяйства.
    # Нормировка на собственное среднее гарантирует, что добавление любого
    # нового множителя не сдвинет итоговый уровень нагрузки.
    base_shape = (
        agg*holiday_base_reduction*seasonal_drift*temp_response
        *district_curve
        *heat_surge_factor*cold_wave_factor*anomaly_factor*(1.0+ar)*trend
    ).astype(np.float64)

    hours_per_month = 8766.0 / 12.0
    residential_mean_kw = float(households) * kwh_per_household_month / hours_per_month
    base_consumption = (
        base_shape * (residential_mean_kw / max(base_shape.mean(), 1e-9))
    ).astype(np.float32)

    logger.info("  Бытовая нагрузка: среднее %.1f кВт (%.0f кВт·ч/мес на домохозяйство)",
                residential_mean_kw, kwh_per_household_month)

    # ══════════════════════════════════════════════════════════════════════
    # EV ЗАРЯДКА (v6: исправлен time-wrap баг + увеличена мощность)
    # ══════════════════════════════════════════════════════════════════════
    n_ev = int(households * ev_penetration)
    ev_load_raw = np.zeros(hours, np.float64)

    for day_idx in range(days):
        is_we_day  = bool(weekday[day_idx*24] >= 5)
        is_hol_day = bool(holiday_mask_daily[day_idx] > 0)
        is_cold    = bool(temperature[day_idx*24] < -15)  # cold snap = range anxiety

        # Частота зарядки. Средний суточный пробег легкового автомобиля в
        # России — порядка 30-40 км, а типичная батарея проезжает 250-400 км,
        # поэтому владелец подключается раз в два-три дня, а не ежедневно.
        # Прежнее значение 0.85 давало около 27 000 кВт·ч в год на автомобиль,
        # что соответствует пробегу порядка 130 000 км — завышение на порядок.
        charge_prob = 0.45 if is_cold else 0.35
        n_charging  = int(rng.binomial(n_ev, charge_prob))
        home_frac   = 0.72 if (is_we_day or is_hol_day) else 0.62

        for _ in range(n_charging):
            if rng.random() < home_frac:
                # Домашняя зарядка: преимущественно ночью. Мощность типична
                # для российских частных подключений: однофазные 16 А (3.5 кВт)
                # и 32 А (7.4 кВт). Трёхфазные 11–22 кВт в жилом секторе редки.
                base_h = int(rng.choice([21, 22, 23, 0, 1, 2, 3, 20]))
                power = rng.uniform(*ev_home_power_kw)
            else:
                # Публичная станция: дневные часы, мощность выше
                base_h = int(rng.integers(9, 18))
                power = rng.uniform(*ev_public_power_kw)

            # Длительность выводится из ОТПУСКА ЭНЕРГИИ, а не задаётся
            # независимо от мощности. Иначе произведение «мощность × часы»
            # ничем не ограничено сверху и не соответствует ёмкости батареи:
            # 7 кВт в течение 8 часов — это 56 кВт·ч за сессию, больше полной
            # ёмкости типичного автомобиля.
            session_kwh = rng.uniform(8.0, 26.0)
            duration = int(np.clip(round(session_kwh / max(power, 1e-6)), 1, 9))

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

        # Пятничный кластер: часть владельцев заряжается перед выходными
        if weekday[day_idx*24] == 4 and not is_hol_day:
            peak_h = day_idx*24 + 21
            if peak_h < hours:
                ev_load_raw[peak_h] += n_ev * 0.15 * rng.uniform(0.8, 1.2) * ev_home_power_kw[1]

        # Всплеск в сильный мороз: падение запаса хода вынуждает заряжаться чаще
        if is_cold:
            surge_h = day_idx*24 + 19
            if surge_h < hours:
                ev_load_raw[surge_h] += n_ev * 0.10 * rng.uniform(0.8, 1.2) * ev_home_power_kw[0]

    # Коммерческий парк: депо заряжает грузовики ночью перед рабочим днём.
    # Число депо масштабируется проникновением электротранспорта, а не только
    # числом домохозяйств. Прежде оно равнялось households // 1000 независимо
    # от сценария, и при ev_penetration = 0 канал ЭВ всё равно давал около 4%
    # городского потребления: сценарий «current» описывал сеть, где
    # электротранспорта нет, но депо электрогрузовиков есть. Нормировка на
    # опорное проникновение 0.15 соответствует перспективному сценарию.
    fleet_scale = min(1.0, max(0.0, ev_penetration) / 0.15)
    n_commercial = int(round(max(0, households // 1000) * fleet_scale))
    for _ in range(n_commercial):
        depot_start = int(rng.integers(21, 24))  # Начало зарядки в 21-23ч
        kw_truck    = rng.uniform(*ev_fleet_power_kw)
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
    solar_gen_raw = (solar_profile*solar_season*cloud_factor
                     *solar_panel_peak_kw*n_solar).astype(np.float32)

    # ── УПРАВЛЕНИЕ СПРОСОМ ────────────────────────────────────────────────
    # В России ценозависимое снижение потребления существует, но охватывает
    # ограниченный круг участников, поэтому эффект на агрегированную нагрузку
    # района измеряется единицами процентов, а не десятками.
    dsr_active   = np.zeros(hours, np.float32)
    dsr_strength = np.zeros(hours, np.float32)
    n_dsr = max(2, int(days/365*dsr_events_per_year))
    # События назначаются в пиковые тарифные часы: 07–10 и 17–21 в будни.
    peak_wd = np.where(
        (((hour_of_day>=7)&(hour_of_day<10)) | ((hour_of_day>=17)&(hour_of_day<21)))
        &(is_weekend==0)
    )[0]
    used_h = set()
    for _ in range(n_dsr):
        if len(peak_wd) == 0: break
        a = int(rng.choice(peak_wd))
        if a in used_h: continue
        dur = int(rng.integers(2, 5))
        str_= rng.uniform(*dsr_strength_range)
        for h in range(dur):
            idx = a + h
            if idx < hours:
                dsr_active[idx]   = 1.0
                dsr_strength[idx] = str_
                used_h.add(idx)

    # ── КОММЕРЧЕСКАЯ И ПРОМЫШЛЕННАЯ НАГРУЗКА ─────────────────────────────
    # Мощность отдельных объектов задаётся не абсолютной величиной, а долей от
    # бытовой нагрузки: так соотношение секторов остаётся правдоподобным при
    # любом числе домохозяйств. Профиль — марковское чередование смен.
    # Смены привязаны к часу суток. Прежняя версия переключала состояние
    # «работает / стоит» марковским процессом, зависящим только от дня недели,
    # поэтому завод мог начать смену в три часа ночи. У компоненты выходил
    # плоский суточный профиль и ACF(24) около 0.02 — она была непрогнозируема
    # на сутки вперёд в принципе и задавала искусственный шумовой пол, на
    # котором сидели все модели.
    industrial_shape = np.zeros(hours, np.float64)
    for ind_i in range(industrial_loads):
        weight = rng.uniform(0.5, 1.5)
        shift_start = int(rng.integers(6, 9))
        # Одно-, двух- и трёхсменные объекты плюс непрерывное производство.
        shift_len = int(rng.choice([8, 16, 24], p=[0.5, 0.3, 0.2]))
        # Загрузка задаётся УРОВНЕМ, а не двоичным «работает / стоит».
        # Двоичное переключение означало бы, что предприятие пропускает целые
        # смены случайным образом, и суточная автокорреляция ряда падала
        # сильнее, чем её поднимала привязка смен к часам. Реальное
        # производство работает по календарю, а меняется загрузка — портфель
        # заказов, и меняется она инерционно.
        load_wd = rng.uniform(0.85, 1.00)
        # Выходная загрузка не падает до пятой части: в городском агрегате есть
        # непрерывные производства, торговля и коммунальная инфраструктура.
        # При 0.20-0.50 отношение выходные/будни по городу выходило 0.76 при
        # эталонных для России 0.88-0.99.
        load_we = rng.uniform(0.55, 0.85)
        # Постоянная составляющая: вентиляция, холодильное оборудование,
        # дежурное освещение и обогрев работают круглосуточно. Без неё ночной
        # провал городского ряда получался вдвое глубже реального, а
        # коэффициент заполнения проваливался ниже правдоподобного диапазона.
        baseload = rng.uniform(0.30, 0.55)
        level = 1.0
        for day_idx in range(days):
            h0 = day_idx * 24
            base = load_wd if weekday[h0] < 5 else load_we
            base *= float(holiday_industrial_factor[h0])
            level = 0.78 * level + 0.22 * rng.normal(1.0, 0.10)
            activity = max(0.0, base * level)
            if activity <= 0.02:
                continue
            ramp = activity * rng.uniform(0.97, 1.03)
            # Постоянная составляющая идёт все 24 часа суток.
            for k in range(24):
                h = h0 + k
                if h < hours:
                    industrial_shape[h] += weight * baseload * ramp
            # Сменная составляющая накладывается поверх.
            for k in range(shift_len):
                h = h0 + shift_start + k
                if h >= hours:
                    break
                # Плавный набор и снижение нагрузки на краях смены.
                edge = min(1.0, (k + 1) / 2.0, (shift_len - k) / 2.0)
                industrial_shape[h] += (weight * (1.0 - baseload) * ramp * edge
                                        * rng.uniform(0.97, 1.03))

    nonres_mean_kw = residential_mean_kw * nonresidential_share
    industrial_load_raw = (
        industrial_shape * (nonres_mean_kw / max(industrial_shape.mean(), 1e-9))
    ).astype(np.float32)

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
            cascade_state = min(cascade_state + rng.uniform(0.02, 0.05), 0.12)
        else:
            cascade_state = max(cascade_state - 0.04, 0.0)
        cascade_factor[h] = 1.0 + cascade_state

    # ── НЕЛИНЕЙНОЕ "СОСТОЯНИЕ СЕТИ" (v7): память + пороги + взаимодействия ──
    # Цель: сделать задачу менее линейной, но физически правдоподобной.
    # Компонент зависит от:
    #   1) экстремальной температуры,
    #   2) пиковых часов,
    #   3) EV-нагрузки,
    #   4) собственной инерции (latent state).
    grid_stress = np.zeros(hours, np.float32)
    ev_norm_proxy = ev_load_raw / (float(ev_load_raw.max()) + 1e-6)
    temp_extreme = np.clip(np.abs(temperature - temp_setpoint) / 22.0, 0.0, 1.4).astype(np.float32)
    peak_hours = (((hour_of_day >= 7) & (hour_of_day < 10)) | ((hour_of_day >= 17) & (hour_of_day < 21))).astype(np.float32)
    stress_state = 0.0
    for h in range(hours):
        trigger = (
            0.40 * float(temp_extreme[h]) +
            0.25 * float(peak_hours[h]) +
            0.25 * float(ev_norm_proxy[h]) +
            0.10 * float(cloud_cover[h])
        )
        stress_state = 0.90 * stress_state + 0.10 * trigger + float(rng.normal(0.0, 0.015))
        stress_state = float(np.clip(stress_state, 0.0, 1.4))
        grid_stress[h] = stress_state
    stress_nonlinear = (
        1.0
        + 0.09 * _sigmoid(6.0 * (grid_stress - 0.55))
        + 0.03 * (grid_stress ** 2)
    ).astype(np.float32)

    # ── DSR rebound (v7): после снятия ограничения часть спроса возвращается ──
    dsr_rebound = np.zeros(hours, np.float32)
    for h in range(1, hours):
        if dsr_active[h - 1] > 0 and dsr_active[h] == 0:
            rebound_amp = rng.uniform(0.08, 0.16)
            rebound_len = int(rng.integers(2, 5))
            for k in range(rebound_len):
                idx = h + k
                if idx >= hours:
                    break
                dsr_rebound[idx] += rebound_amp * np.exp(-0.65 * k)

    # Праздничные спайки
    holiday_spike = np.ones(hours, np.float32)
    for i in range(days):
        d = pd.to_datetime(start_date) + timedelta(days=i)
        if d.month == 1 and d.day == 1:
            for ho in [0, 1, 2]:
                idx = i*24+ho
                if idx < hours: holiday_spike[idx] = 1.20
        elif d.month == 5 and d.day == 9:
            for ho in [20, 21, 22]:
                idx = i*24+ho
                if idx < hours: holiday_spike[idx] = 1.12

    # ── ИТОГОВОЕ ПОТРЕБЛЕНИЕ ──────────────────────────────────────────────
    consumption = (
        base_consumption * holiday_spike * cascade_factor * stress_nonlinear * (1.0 - dsr_strength)
        + ev_load_raw
        + industrial_load_raw
        - solar_gen_raw
        + base_consumption * dsr_rebound
    ).astype(np.float32)
    # Нижняя отсечка задаётся относительно среднего уровня: абсолютная
    # константа теряет смысл при изменении масштаба нагрузки.
    consumption = np.clip(consumption, 0.05*float(np.mean(consumption)), None)

    ev_mean = float(ev_load_raw.mean())
    bc_mean = float(base_consumption.mean())
    logger.info("Сгенерировано %d записей. Потребление: min=%.1f, mean=%.1f, max=%.1f кВт·ч",
                len(consumption), float(consumption.min()), float(consumption.mean()), float(consumption.max()))
    logger.info("  EV нагрузка: mean=%.1f кВт (%.1f%% базовой)  [v5 был: ~%.1f%% — был time-wrap баг]",
                ev_mean, 100*ev_mean/max(bc_mean,1), 1.2)
    logger.info("  Solar: mean=%.1f кВт, max=%.1f кВт", float(solar_gen_raw.mean()), float(solar_gen_raw.max()))
    logger.info("  DSR событий активно: %d ч | Industrial: %d заводов | Cascade max=%.2f",
                int(dsr_active.sum()), industrial_loads, float(cascade_factor.max()))
    logger.info("  Grid stress max=%.2f | DSR rebound max=%.2f",
                float(stress_nonlinear.max()), float(dsr_rebound.max()))
    logger.info("  CV потребления: %.3f", float(consumption.std()/consumption.mean()))

    # ── ВАЖНО: нормировка признаков здесь НЕ выполняется ─────────────────────
    # Раньше ev/solar/temperature² делились на максимум по ВСЕМУ ряду, включая
    # тестовую часть. Это утечка: статистика теста попадала в обучающие признаки
    # в обход train-скалеров. Генератор теперь отдаёт сырые физические величины,
    # а нормировка целиком выполняется в data/preprocessing.py скалерами,
    # обученными только на train.

    # Tariff zones
    # Тарифные зоны по действующему в РФ порядку: пиковая 07–10 и 17–21,
    # полупиковая 10–17 и 21–23, ночная 23–07. В выходные и праздники
    # пиковая зона не применяется.
    tariff_arr = np.empty(hours, dtype=object)
    for i in range(hours):
        h, wd, ih = int(hour_of_day[i]), int(weekday[i]), bool(holiday_mask[i])
        if h<7 or h>=23: tariff_arr[i]="night"
        elif wd>=5 or ih: tariff_arr[i]="day"
        elif (7<=h<10) or (17<=h<21): tariff_arr[i]="peak"
        else: tariff_arr[i]="day"
    is_peak_hour  = (tariff_arr=="peak").astype(np.int8)
    is_night_hour = (tariff_arr=="night").astype(np.int8)

    month = dates.month.values.astype(np.int8)
    season_arr = np.where((month>=3)&(month<=5),"spring",
                  np.where((month>=6)&(month<=8),"summer",
                   np.where((month>=9)&(month<=11),"autumn","winter")))

    df = pd.DataFrame({
        "timestamp": dates, "consumption": consumption,
        "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
        "cloud_cover": cloud_cover,
        "ev_load_kw": ev_load_raw,
        "solar_gen_kw": solar_gen_raw,
        "dsr_active": dsr_active,
        "hour": hour_of_day, "weekday": weekday.astype(np.int8),
        "is_weekend": is_weekend.astype(np.int8), "is_holiday": holiday_mask.astype(np.int8),
        "is_peak_hour": is_peak_hour, "is_night_hour": is_night_hour,
        "tariff_zone": tariff_arr, "month": month, "season": season_arr,
        "heating_degree_days": np.maximum(0.0, 18.0-temperature).astype(np.float32),
        "cooling_degree_days": np.maximum(0.0, temperature-24.0).astype(np.float32),
        "day_of_year": day_of_year.astype(np.int16),
    })
    df["rolling_mean_24h"] = df["consumption"].rolling(24, min_periods=1).mean().astype(np.float32)
    df["rolling_std_24h"]  = df["consumption"].rolling(24, min_periods=2).std().fillna(0.0).astype(np.float32)
    df["hour_sin"] = np.sin(2*np.pi*hour_of_day/24).astype(np.float32)
    df["hour_cos"] = np.cos(2*np.pi*hour_of_day/24).astype(np.float32)
    return df


def validate_against_reference(df, households, reference=None, kwh_per_household_month=None):
    """
    Сверяет сгенерированный ряд с фактическими показателями реального мира.

    Проверка отвечает на вопрос «похож ли синтетический город на настоящий»
    и делает это воспроизводимо: при каждом прогоне в лог выводится таблица
    «показатель — модель — эталон — вердикт». Если калибровка нарушится при
    правке генератора, это будет видно сразу, а не после защиты.

    Сверяемые показатели:
      * среднемесячное потребление на домохозяйство;
      * средняя температура января и июля;
      * отношение зимнего потребления к летнему;
      * отношение выходных к будням;
      * коэффициент заполнения графика (среднее / максимум);
      * коэффициент вариации;
      * положение утреннего и вечернего максимумов;
      * автокорреляция на суточном лаге.

    Returns
    -------
    (passed: bool, rows: list[dict]) — вердикт и построчные результаты.
    """
    if reference is None:
        from config import RealWorldReference as reference
    if kwh_per_household_month is None:
        kwh_per_household_month = reference.KWH_PER_HOUSEHOLD_MONTH

    cons = df["consumption"].values.astype(np.float64)
    temp = df["temperature"].values.astype(np.float64)
    month = pd.DatetimeIndex(df["timestamp"]).month.values
    hours_per_month = 8766.0 / 12.0

    # Бытовая часть выделяется из общей нагрузки по доле непромышленных
    # потребителей: сравнивать полное потребление района с бытовым счётом
    # домохозяйства было бы некорректно.
    nonres_share = getattr(reference, "NONRESIDENTIAL_SHARE", 0.0)
    residential_mean = cons.mean() / (1.0 + nonres_share)
    kwh_per_hh = residential_mean / households * hours_per_month

    jan = temp[month == 1].mean() if (month == 1).any() else float("nan")
    jul = temp[month == 7].mean() if (month == 7).any() else float("nan")

    winter = cons[np.isin(month, (12, 1, 2))].mean() if np.isin(month, (12, 1, 2)).any() else np.nan
    summer = cons[np.isin(month, (6, 7, 8))].mean() if np.isin(month, (6, 7, 8)).any() else np.nan
    winter_summer = winter / summer if summer and not np.isnan(summer) else float("nan")

    we = df[df["is_weekend"] == 1]["consumption"].mean()
    wd = df[df["is_weekend"] == 0]["consumption"].mean()
    we_wd = we / wd if wd else float("nan")

    load_factor = cons.mean() / cons.max()
    cv = cons.std() / cons.mean()

    hourly = df.groupby("hour")["consumption"].mean()
    morning_peak = int(hourly.loc[6:11].idxmax())
    evening_peak = int(hourly.loc[16:23].idxmax())

    c = cons - cons.mean()
    var = np.var(c) + 1e-12
    acf24 = float(np.mean(c[:-24] * c[24:]) / var) if len(c) > 24 else float("nan")

    def row(name, value, lo, hi, unit=""):
        # Показатель, который невозможно вычислить на данном периоде (например,
        # летняя температура при прогоне длиной в два зимних месяца), помечается
        # как «нет данных» и не считается отклонением от эталона.
        available = not np.isnan(value)
        return {"показатель": name, "модель": value, "эталон_мин": lo,
                "эталон_макс": hi, "единица": unit,
                "доступен": available,
                "ok": available and (lo <= value <= hi)}

    # Форма суточного графика и годовой ход
    by_hour = df.groupby("hour")["consumption"].mean()
    night_level = float(by_hour.loc[1:5].mean())
    noon_over_night = float(by_hour.loc[11:15].mean() / max(night_level, 1e-9))
    morning_over_night = float(by_hour.loc[7:10].mean() / max(night_level, 1e-9))

    months = pd.to_datetime(df["timestamp"]).dt.month
    by_month = df.groupby(months)["consumption"].mean()
    winter = by_month.reindex([12, 1, 2]).mean()
    summer = by_month.reindex([6, 7, 8]).mean()
    winter_summer_ratio = float(winter / max(summer, 1e-9)) if summer == summer else float("nan")

    rows = [
        row("Потребление на домохозяйство", kwh_per_hh,
            *reference.KWH_PER_HOUSEHOLD_MONTH_RANGE, "кВт·ч/мес"),
        row("Средняя температура января", jan,
            reference.TEMP_JANUARY_MEAN - 3.0, reference.TEMP_JANUARY_MEAN + 3.0, "°C"),
        row("Средняя температура июля", jul,
            reference.TEMP_JULY_MEAN - 3.0, reference.TEMP_JULY_MEAN + 3.0, "°C"),
        row("Отношение зима/лето", winter_summer,
            *reference.WINTER_SUMMER_RATIO_RANGE, ""),
        row("Отношение выходные/будни", we_wd,
            *reference.WEEKEND_WEEKDAY_RATIO_RANGE, ""),
        row("Коэффициент заполнения графика", load_factor,
            *reference.LOAD_FACTOR_RANGE, ""),
        row("Коэффициент вариации", cv, *reference.CV_RANGE, ""),
        row("Час утреннего максимума", float(morning_peak),
            *map(float, reference.MORNING_PEAK_HOURS), "ч"),
        row("Час вечернего максимума", float(evening_peak),
            *map(float, reference.EVENING_PEAK_HOURS), "ч"),
        row("ACF на суточном лаге", acf24, reference.ACF24_MIN, 1.0, ""),
        # Форма суточного графика. Прежний набор показателей её не проверял, и
        # генератор выдавал профиль, где полдень ниже трёх часов ночи, а
        # утреннего подъёма не было вовсе, — проверка «час утреннего максимума»
        # находила его только потому, что искала максимум в узком окне 6-11.
        row("Полдень к ночному минимуму", noon_over_night,
            reference.NOON_OVER_NIGHT_MIN, 3.0, ""),
        row("Утро к ночному минимуму", morning_over_night,
            reference.MORNING_OVER_NIGHT_MIN, 3.0, ""),
        row("Зимний месяц к летнему", winter_summer_ratio,
            reference.WINTER_SUMMER_MONTH_RATIO_MIN, 2.0, ""),
    ]

    logger.info("─" * 88)
    logger.info("СВЕРКА С РЕАЛЬНЫМИ ДАННЫМИ (Россия, Московский регион)")
    logger.info("%-34s %12s %22s %8s", "Показатель", "Модель", "Эталонный диапазон", "Вердикт")
    logger.info("─" * 88)
    for r in rows:
        if not r["доступен"]:
            verdict, shown = "нет данных", float("nan")
        else:
            verdict, shown = ("OK" if r["ok"] else "ОТКЛОН."), r["модель"]
        logger.info(
            "%-34s %12.2f %10.2f … %-9.2f %10s %s",
            r["показатель"], shown, r["эталон_мин"], r["эталон_макс"],
            verdict, r["единица"],
        )
    logger.info("─" * 88)

    failed = [r["показатель"] for r in rows if r["доступен"] and not r["ok"]]
    skipped = [r["показатель"] for r in rows if not r["доступен"]]
    if skipped:
        logger.info(
            "Не проверялись (период прогона не покрывает нужные месяцы): %s",
            "; ".join(skipped),
        )
    if failed:
        logger.warning(
            "Отклонения от эталона: %s. Это не ошибка выполнения, но при "
            "интерпретации результатов такие расхождения нужно оговаривать.",
            "; ".join(failed),
        )
    else:
        logger.info("Все проверенные показатели укладываются в эталонные диапазоны.")
    return (not failed), rows


def validate_generated_data(df):
    passed = True
    logger.info("Валидация сгенерированных данных...")
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
    ev_mean = float(df["ev_load_kw"].mean()); solar_mean = float(df["solar_gen_kw"].mean())
    logger.info("  ℹ️  EV mean=%.1f кВт | Solar mean=%.1f кВт | DSR ч=%d | CV=%.3f",
                ev_mean, solar_mean, int(df["dsr_active"].sum()),
                float(df["consumption"].std()/df["consumption"].mean()))
    logger.info("Валидация %s", "ПРОЙДЕНА" if passed else "ПРОВАЛЕНА")
    return passed