# -*- coding: utf-8 -*-
"""
data/preprocessing.py — Подготовка мультивариантных данных.

Формирует матрицу из 26 признаков на каждый час и нарезает её скользящими
окнами (history → horizon).

ПРИНЦИП РАЗДЕЛЕНИЯ ДАННЫХ (защита от утечки):
  1. Разбиение строго хронологическое: train → val → test, без перемешивания.
  2. ВСЕ скалеры обучаются только на train-части и лишь применяются к val/test.
     Это касается не только потребления, но и температуры, влажности, ветра,
     облачности, скользящего СКО, нагрузки электротранспорта и солнечной
     генерации — любая нормировка «на максимум по всему ряду» означала бы, что
     статистика теста участвует в обучении.
  3. Лаговые колонки строятся до разбиения — это корректно, поскольку лаг
     смотрит строго в прошлое и на границе сплитов не заглядывает вперёд.
  4. Окна нарезаются внутри каждого сплита независимо, поэтому целевые значения
     одного сплита никогда не попадают в историю другого.

Состав 26 признаков (порядок фиксирован, индексы используются в models/baseline.py):
   0 consumption      9 tariff_zone     18 dow_sin
   1 hour_sin        10 day_of_year     19 dow_cos
   2 hour_cos        11 humidity        20 month_sin
   3 is_peak_hour    12 wind_speed      21 month_cos
   4 is_night_hour   13 rolling_mean_24h 22 cloud_cover
   5 is_weekend      14 rolling_std_24h 23 ev_load
   6 is_holiday      15 load_lag_24h    24 solar_gen
   7 temperature     16 load_lag_48h    25 dsr_active
   8 temperature²    17 load_lag_168h

История изменений — в CHANGELOG.md.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("smart_grid.data.preprocessing")

N_FEATURES: int = 26


def _encode_tariff_zone(df):
    if "tariff_zone" in df.columns:
        mapping = {"night": 0.0, "day": 0.5, "peak": 1.0}
        return df["tariff_zone"].map(mapping).fillna(0.5).values.astype(np.float32)
    hours   = df["hour"].values
    weekday = df["weekday"].values if "weekday" in df.columns else np.zeros(len(df))
    enc = np.full(len(df), 0.5, dtype=np.float32)
    enc[(hours < 7) | (hours >= 23)] = 0.0
    enc[(((hours >= 10) & (hours < 17)) | ((hours >= 21) & (hours < 23))) & (weekday < 5)] = 1.0
    return enc


def _add_lag_columns(df):
    df = df.copy()
    for lag in (24, 48, 168):
        df[f"load_lag_{lag}h"] = df["consumption"].shift(lag).fillna(0.0)
    return df


def _build_feature_matrix(
    df, cons_scaler, temp_scaler,
    humidity_scaler=None, wind_scaler=None,
    rolling_std_scaler=None, cloud_scaler=None,
    ev_scaler=None, solar_scaler=None,
    temp_sq_max=None,
):
    N = len(df)
    def scale(col, scaler): return scaler.transform(df[col].values.reshape(-1,1)).flatten().astype(np.float32)
    def zeros(): return np.zeros(N, np.float32)

    cons     = scale("consumption", cons_scaler)
    h        = df["hour"].values.astype(np.float32)
    hour_sin = np.sin(2*np.pi*h/24).astype(np.float32)
    hour_cos = np.cos(2*np.pi*h/24).astype(np.float32)
    is_peak  = df["is_peak_hour"].values.astype(np.float32)  if "is_peak_hour"  in df.columns else zeros()
    is_night = df["is_night_hour"].values.astype(np.float32) if "is_night_hour" in df.columns else zeros()
    is_wend  = df["is_weekend"].values.astype(np.float32)
    is_hol   = df["is_holiday"].values.astype(np.float32)
    temp     = scale("temperature", temp_scaler) if "temperature" in df.columns else zeros()

    # temperature² нормируется на максимум ПО TRAIN (temp_sq_max), а не по всему
    # ряду: иначе диапазон тестовых температур влиял бы на обучающие признаки.
    if "temperature" in df.columns:
        t_raw = df["temperature"].values.astype(np.float32)
        if temp_sq_max is None:
            temp_sq_max = float((t_raw**2).max()) + 1e-8
            logger.warning("temp_sq_max не передан → нормировка по текущему сплиту.")
        temp_sq = (t_raw**2 / temp_sq_max).clip(0, 1.5).astype(np.float32)
    else:
        temp_sq = zeros()

    tariff_enc = _encode_tariff_zone(df)

    if "day_of_year" in df.columns:
        doy_norm = df["day_of_year"].values.astype(np.float32) / 364.0
        doy_vals = df["day_of_year"].values.astype(np.float32)
    else:
        doy_norm = doy_vals = zeros()

    humidity = (humidity_scaler.transform(df["humidity"].values.reshape(-1,1)).flatten().astype(np.float32)
                if "humidity" in df.columns and humidity_scaler is not None else zeros())
    wind     = (wind_scaler.transform(df["wind_speed"].values.reshape(-1,1)).flatten().astype(np.float32)
                if "wind_speed" in df.columns and wind_scaler is not None else zeros())

    if "rolling_mean_24h" in df.columns:
        roll_mean = cons_scaler.transform(df["rolling_mean_24h"].values.reshape(-1,1)).flatten().clip(0,1).astype(np.float32)
    else:
        roll_mean = cons.copy()

    if "rolling_std_24h" in df.columns and rolling_std_scaler is not None:
        roll_std = rolling_std_scaler.transform(df["rolling_std_24h"].values.reshape(-1,1)).flatten().clip(0,1).astype(np.float32)
    elif "rolling_std_24h" in df.columns:
        rs = df["rolling_std_24h"].values.astype(np.float32)
        roll_std = (rs / (float(rs.max())+1e-8)).astype(np.float32)
    else:
        roll_std = zeros()

    def get_lag(lag_h):
        col = f"load_lag_{lag_h}h"
        if col in df.columns:
            return cons_scaler.transform(df[col].values.reshape(-1,1)).flatten().clip(0,1.5).astype(np.float32)
        logger.warning("Колонка %s не найдена → нули.", col)
        return zeros()

    lag_24h  = get_lag(24)
    lag_48h  = get_lag(48)
    lag_168h = get_lag(168)

    if "weekday" in df.columns:
        wd = df["weekday"].values.astype(np.float32)
    elif "timestamp" in df.columns:
        wd = pd.to_datetime(df["timestamp"]).dt.dayofweek.values.astype(np.float32)
    else:
        wd = zeros()
    dow_sin   = np.sin(2*np.pi*wd/7).astype(np.float32)
    dow_cos   = np.cos(2*np.pi*wd/7).astype(np.float32)
    month_sin = np.sin(2*np.pi*(doy_vals-1)/365).astype(np.float32)
    month_cos = np.cos(2*np.pi*(doy_vals-1)/365).astype(np.float32)

    cloud = (cloud_scaler.transform(df["cloud_cover"].values.reshape(-1,1)).flatten().clip(0,1).astype(np.float32)
             if "cloud_cover" in df.columns and cloud_scaler is not None
             else (df["cloud_cover"].values.clip(0,1).astype(np.float32) if "cloud_cover" in df.columns else zeros()))

    # EV и солнечная генерация: нормировка train-скалерами по сырым кВт.
    def scale_optional(col, scaler):
        if col in df.columns and scaler is not None:
            return scaler.transform(df[col].values.reshape(-1,1)).flatten().clip(0,1.5).astype(np.float32)
        return zeros()

    ev    = scale_optional("ev_load_kw", ev_scaler)
    solar = scale_optional("solar_gen_kw", solar_scaler)
    dsr   = df["dsr_active"].values.clip(0,1).astype(np.float32) if "dsr_active" in df.columns else zeros()

    return np.stack([
        cons, hour_sin, hour_cos, is_peak, is_night,
        is_wend, is_hol, temp, temp_sq, tariff_enc, doy_norm,
        humidity, wind, roll_mean, roll_std,
        lag_24h, lag_48h, lag_168h,
        dow_sin, dow_cos, month_sin, month_cos,
        cloud, ev, solar, dsr,
    ], axis=1).astype(np.float32)


def _make_multivariate_windows(features, history, horizon):
    N, nf = features.shape
    n = N - history - horizon + 1
    if n <= 0:
        raise ValueError(f"Недостаточно данных: N={N}, history={history}, horizon={horizon}")
    ri = np.arange(n)[:, None]
    ci = np.arange(history)[None, :]
    X  = features[ri + ci]
    yi = ri + history + np.arange(horizon)[None, :]
    Y  = features[yi, 0]
    return X.astype(np.float32), Y.astype(np.float32)


def prepare_data(df, history_length=48, forecast_horizon=24,
                 train_ratio=0.70, val_ratio=0.15):
    """Полный пайплайн подготовки данных v7 (26 признаков)."""
    df = _add_lag_columns(df)
    logger.info("Лаговые колонки добавлены: load_lag_24h, load_lag_48h, load_lag_168h")

    total     = len(df)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))
    df_train  = df.iloc[:train_end].copy()
    df_val    = df.iloc[train_end:val_end].copy()
    df_test   = df.iloc[val_end:].copy()

    raw_train = df_train["consumption"].values.astype(np.float32)
    raw_val   = df_val["consumption"].values.astype(np.float32)
    raw_test  = df_test["consumption"].values.astype(np.float32)

    cons_scaler = MinMaxScaler((0,1)).fit(raw_train.reshape(-1,1))
    temp_scaler = MinMaxScaler((0,1))
    if "temperature" in df.columns:
        temp_scaler.fit(df_train["temperature"].values.reshape(-1,1))
    humidity_scaler   = (MinMaxScaler((0,1)).fit(df_train["humidity"].values.reshape(-1,1))
                         if "humidity" in df.columns else None)
    wind_scaler       = (MinMaxScaler((0,1)).fit(df_train["wind_speed"].values.reshape(-1,1))
                         if "wind_speed" in df.columns else None)
    rolling_std_scaler= (MinMaxScaler((0,1)).fit(df_train["rolling_std_24h"].values.reshape(-1,1))
                         if "rolling_std_24h" in df.columns else None)
    cloud_scaler      = (MinMaxScaler((0,1)).fit(df_train["cloud_cover"].values.reshape(-1,1))
                         if "cloud_cover" in df.columns else None)
    ev_scaler         = (MinMaxScaler((0,1)).fit(df_train["ev_load_kw"].values.reshape(-1,1))
                         if "ev_load_kw" in df.columns else None)
    solar_scaler      = (MinMaxScaler((0,1)).fit(df_train["solar_gen_kw"].values.reshape(-1,1))
                         if "solar_gen_kw" in df.columns else None)
    temp_sq_max       = None
    if "temperature" in df.columns:
        temp_sq_max = float((df_train["temperature"].values.astype(np.float32)**2).max()) + 1e-8

    scaled_train = cons_scaler.transform(raw_train.reshape(-1,1)).flatten()
    scaled_val   = cons_scaler.transform(raw_val.reshape(-1,1)).flatten()
    scaled_test  = cons_scaler.transform(raw_test.reshape(-1,1)).flatten()

    kw = dict(humidity_scaler=humidity_scaler, wind_scaler=wind_scaler,
              rolling_std_scaler=rolling_std_scaler, cloud_scaler=cloud_scaler,
              ev_scaler=ev_scaler, solar_scaler=solar_scaler,
              temp_sq_max=temp_sq_max)
    feat_train = _build_feature_matrix(df_train, cons_scaler, temp_scaler, **kw)
    feat_val   = _build_feature_matrix(df_val,   cons_scaler, temp_scaler, **kw)
    feat_test  = _build_feature_matrix(df_test,  cons_scaler, temp_scaler, **kw)

    X_train, Y_train = _make_multivariate_windows(feat_train, history_length, forecast_horizon)
    X_val,   Y_val   = _make_multivariate_windows(feat_val,   history_length, forecast_horizon)
    X_test,  Y_test  = _make_multivariate_windows(feat_test,  history_length, forecast_horizon)

    # Знаменатель MASE считается ТОЛЬКО по обучающей части (Hyndman & Koehler):
    # ошибка сезонно-наивного прогноза на train. Использовать для этого тест
    # нельзя — метрика перестанет быть независимой от оцениваемых данных.
    from utils.metrics import seasonal_naive_scale
    mase_scale = seasonal_naive_scale(raw_train, season_length=24)

    logger.info("Данные подготовлены | train=%d val=%d test=%d | n_features=%d",
                len(X_train), len(X_val), len(X_test), feat_train.shape[1])
    logger.info("X_train.shape: %s  Y_train.shape: %s", X_train.shape, Y_train.shape)
    logger.info("Скалеры (обучены только на train): humidity=%s wind=%s rolling_std=%s "
                "cloud=%s ev=%s solar=%s",
                "ok" if humidity_scaler else "нет", "ok" if wind_scaler else "нет",
                "ok" if rolling_std_scaler else "нет", "ok" if cloud_scaler else "нет",
                "ok" if ev_scaler else "нет", "ok" if solar_scaler else "нет")
    logger.info("MASE scale (сезонно-наивный на train, m=24): %.2f кВт·ч", mase_scale)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
        "scaled_train": scaled_train, "scaled_val": scaled_val, "scaled_test": scaled_test,
        "scaler": cons_scaler, "temp_scaler": temp_scaler,
        "humidity_scaler": humidity_scaler, "wind_scaler": wind_scaler,
        "rolling_std_scaler": rolling_std_scaler, "cloud_scaler": cloud_scaler,
        "ev_scaler": ev_scaler, "solar_scaler": solar_scaler,
        "temp_sq_max": temp_sq_max,
        "mase_scale": mase_scale,
        "n_features": feat_train.shape[1],
        "history_length": history_length, "forecast_horizon": forecast_horizon,
        "train_end_idx": train_end, "val_end_idx": val_end,
        "test_start_idx": val_end + history_length,
        "timestamps": df["timestamp"].values,
    }


def validate_data_integrity(data):
    """
    Проверяет корректность подготовленных данных.

    Проверяются: отсутствие NaN/Inf, согласованность размеров X и Y, непустота
    сплитов, хронологический порядок индексов и диапазон целевых значений.

    О диапазоне Y. Скалер обучен на train, поэтому Y_train лежит в [0, 1], а
    val/test могут выходить за верхнюю границу: в данных есть годовой тренд
    (+5%/год), и зимние пики второго года выше максимума первого. Это не ошибка,
    поэтому превышение до 1.50 даёт предупреждение, а не исключение. Значения
    выше 1.50 означают уже реальную проблему масштабирования и прерывают работу.
    """
    log = logging.getLogger("smart_grid.data.preprocessing")

    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        Y = data[f"Y_{split}"]

        assert not np.any(np.isnan(X)), f"NaN в X_{split}"
        assert not np.any(np.isinf(X)), f"Inf в X_{split}"
        assert not np.any(np.isnan(Y)), f"NaN в Y_{split}"
        assert X.shape[0] == Y.shape[0], f"Размер X_{split} != Y_{split}"
        assert X.shape[0] > 0, f"Пустой сплит {split}"

        y_min = float(Y.min())
        y_max = float(Y.max())

        if y_min < -0.05:
            log.warning("Y_%s min=%.4f < -0.05 — отрицательные значения (аномалии?)", split, y_min)
        if y_max > 1.50:
            log.error("Y_%s max=%.4f > 1.50 — слишком большие значения, проверьте scaler", split, y_max)
            raise AssertionError(f"Y_{split} max={y_max:.4f} > 1.50 — критическая проблема масштабирования")
        elif y_max > 1.05:
            log.warning(
                "Y_%s max=%.4f > 1.05 — тест/вал данные превышают train-max "
                "(нормально для данных с трендом)", split, y_max
            )

    assert data["train_end_idx"] < data["val_end_idx"], "train/val порядок нарушен"
    assert data["val_end_idx"] < data["test_start_idx"], "val/test порядок нарушен"

    log.info(
        "validate_data_integrity ПРОЙДЕНА | train=%d val=%d test=%d | Y in [%.3f, %.3f]",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
        float(data["Y_train"].min()), float(data["Y_train"].max()),
    )


def inverse_scale(scaler, data):
    """Обратное масштабирование consumption в оригинальный масштаб."""
    original_shape = data.shape
    flat = data.flatten().reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(original_shape)


def reconstruct_day_ahead_series(predictions, forecast_horizon=24, n_hours=None):
    """
    Собирает НЕПРЕРЫВНЫЙ прогнозный ряд из перекрывающихся окон.

    Окна нарезаны со сдвигом 1 час, поэтому просто развернуть матрицу (N, H) в
    вектор нельзя — соседние элементы относятся к разным моментам времени.
    Берутся окна с шагом H (0-е, H-е, 2H-е, …), и их горизонты стыкуются встык.
    Получается ровно то, что видит диспетчер: раз в сутки строится прогноз на
    следующие H часов, и по нему принимаются решения.

    Parameters
    ----------
    predictions : np.ndarray, shape (N, H) — прогноз в исходном масштабе.
    forecast_horizon : int — H.
    n_hours : int, optional — требуемая длина ряда (обрезка).

    Returns
    -------
    np.ndarray, shape (k·H,) — непрерывный почасовой прогноз. Первый элемент
    соответствует моменту data["test_start_idx"] в исходном ряде.
    """
    preds = np.atleast_2d(np.asarray(predictions))
    n_windows, horizon = preds.shape
    if horizon != forecast_horizon:
        logger.warning("Горизонт прогноза %d != ожидаемого %d", horizon, forecast_horizon)

    blocks = [preds[i] for i in range(0, n_windows, horizon)]
    if not blocks:
        return np.asarray([], dtype=np.float32)

    series = np.concatenate(blocks).astype(np.float32)
    if n_hours is not None:
        series = series[:n_hours]
    return series


def actual_series_for_forecast(data, n_hours):
    """
    Фактическое потребление, соответствующее ряду из reconstruct_day_ahead_series.

    Прогноз для окна i относится к моменту raw_test[history + i], поэтому факт
    отсчитывается со сдвига history_length внутри тестовой части.
    """
    history = int(data["history_length"])
    return np.asarray(data["raw_test"][history:history + n_hours], dtype=np.float64)