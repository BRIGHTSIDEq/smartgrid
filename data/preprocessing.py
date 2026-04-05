# -*- coding: utf-8 -*-
"""
data/preprocessing.py — Подготовка мультивариантных данных для нейросетей.

ВЕРСИЯ 6 (v9 пайплайна) — 26 признаков

ИЗМЕНЕНИЯ v6:
  + 4 новых признака из generator v5:
      22 cloud_cover_scaled   — облачность [0,1] (влияет на solar)
      23 ev_load_norm         — нагрузка от EV [0,1] (нелинейный компонент)
      24 solar_gen_norm       — солнечная генерация [0,1] (снижает потребление)
      25 dsr_active           — DSR флаг 0/1 (события demand response)
  
  Итого признаков: 22 → 26

ПОЛНЫЙ СПИСОК 26 ПРИЗНАКОВ:
  0  consumption_scaled
  1  hour_sin
  2  hour_cos
  3  is_peak_hour
  4  is_night_hour
  5  is_weekend
  6  is_holiday
  7  temperature_scaled
  8  temperature_squared
  9  tariff_zone_enc
  10 day_of_year_norm
  11 humidity_scaled
  12 wind_speed_scaled
  13 rolling_mean_scaled
  14 rolling_std_scaled
  15 load_lag_24h_scaled
  16 load_lag_48h_scaled
  17 load_lag_168h_scaled
  18 dow_sin
  19 dow_cos
  20 month_sin
  21 month_cos
  22 cloud_cover_scaled     ★ v6
  23 ev_load_norm           ★ v6
  24 solar_gen_norm         ★ v6
  25 dsr_active             ★ v6
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("smart_grid.data.preprocessing")

N_FEATURES: int = 26


def _encode_tariff_zone(df):
    if "tariff_zone" in df.columns:
        mapping = {"night": 0.0, "day": 0.5, "peak": 1.0}
        return df["tariff_zone"].map(mapping).fillna(0.5).values.astype(np.float32)
    hours = df["hour"].values
    weekday = df["weekday"].values if "weekday" in df.columns else np.zeros(len(df))
    enc = np.full(len(df), 0.5, dtype=np.float32)
    night = (hours < 7) | (hours >= 23)
    peak  = (((hours >= 10) & (hours < 17)) | ((hours >= 21) & (hours < 23))) & (weekday < 5)
    enc[night] = 0.0; enc[peak] = 1.0
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
    temp_sq_max=None,
):
    N = len(df)

    # 0. Consumption
    cons = cons_scaler.transform(df["consumption"].values.reshape(-1,1)).flatten().astype(np.float32)

    # 1-2. Hour sin/cos
    h = df["hour"].values.astype(np.float32)
    hour_sin = np.sin(2*np.pi*h/24).astype(np.float32)
    hour_cos = np.cos(2*np.pi*h/24).astype(np.float32)

    # 3-6. Binary flags
    is_peak    = df["is_peak_hour"].values.astype(np.float32)  if "is_peak_hour"  in df.columns else np.zeros(N, np.float32)
    is_night   = df["is_night_hour"].values.astype(np.float32) if "is_night_hour" in df.columns else np.zeros(N, np.float32)
    is_weekend = df["is_weekend"].values.astype(np.float32)
    is_holiday = df["is_holiday"].values.astype(np.float32)

    # 7. Temperature
    temp = temp_scaler.transform(df["temperature"].values.reshape(-1,1)).flatten().astype(np.float32) \
           if "temperature" in df.columns else np.zeros(N, np.float32)

    # 8. Temperature²
    if "temperature_squared" in df.columns:
        temp_sq = df["temperature_squared"].values.astype(np.float32)
    elif "temperature" in df.columns:
        if temp_sq_max is None:
            t_raw = df["temperature"].values.astype(np.float32)
            temp_sq_max = float((t_raw**2).max()) + 1e-8
        t_raw = df["temperature"].values.astype(np.float32)
        temp_sq = (t_raw**2 / temp_sq_max).clip(0, 1.5).astype(np.float32)
    else:
        temp_sq = np.zeros(N, np.float32)

    # 9. Tariff zone
    tariff_enc = _encode_tariff_zone(df)

    # 10. Day of year
    if "day_of_year" in df.columns:
        doy_norm = df["day_of_year"].values.astype(np.float32) / 364.0
        doy_vals = df["day_of_year"].values.astype(np.float32)
    else:
        doy_norm = doy_vals = np.zeros(N, np.float32)

    # 11. Humidity
    humidity = humidity_scaler.transform(df["humidity"].values.reshape(-1,1)).flatten().astype(np.float32) \
               if ("humidity" in df.columns and humidity_scaler is not None) else np.zeros(N, np.float32)

    # 12. Wind speed
    wind = wind_scaler.transform(df["wind_speed"].values.reshape(-1,1)).flatten().astype(np.float32) \
           if ("wind_speed" in df.columns and wind_scaler is not None) else np.zeros(N, np.float32)

    # 13. Rolling mean
    if "rolling_mean_24h" in df.columns:
        rolling_mean = cons_scaler.transform(df["rolling_mean_24h"].values.reshape(-1,1)).flatten().clip(0,1).astype(np.float32)
    else:
        rolling_mean = cons.copy()

    # 14. Rolling std
    if "rolling_std_24h" in df.columns and rolling_std_scaler is not None:
        rolling_std = rolling_std_scaler.transform(df["rolling_std_24h"].values.reshape(-1,1)).flatten().clip(0,1).astype(np.float32)
    elif "rolling_std_24h" in df.columns:
        rs = df["rolling_std_24h"].values.astype(np.float32)
        rolling_std = (rs / (float(rs.max())+1e-8)).astype(np.float32)
    else:
        rolling_std = np.zeros(N, np.float32)

    # 15-17. Load lags
    def get_lag(lag_h):
        col = f"load_lag_{lag_h}h"
        if col in df.columns:
            return cons_scaler.transform(df[col].values.reshape(-1,1)).flatten().clip(0,1.5).astype(np.float32)
        logger.warning("Колонка %s не найдена → нули.", col)
        return np.zeros(N, np.float32)

    lag_24h  = get_lag(24)
    lag_48h  = get_lag(48)
    lag_168h = get_lag(168)

    # 18-19. Day of week sin/cos
    if "weekday" in df.columns:
        wd = df["weekday"].values.astype(np.float32)
    elif "timestamp" in df.columns:
        wd = pd.to_datetime(df["timestamp"]).dt.dayofweek.values.astype(np.float32)
    else:
        wd = np.zeros(N, np.float32)
    dow_sin = np.sin(2*np.pi*wd/7).astype(np.float32)
    dow_cos = np.cos(2*np.pi*wd/7).astype(np.float32)

    # 20-21. Month sin/cos
    month_sin = np.sin(2*np.pi*(doy_vals-1)/365).astype(np.float32)
    month_cos = np.cos(2*np.pi*(doy_vals-1)/365).astype(np.float32)

    # ── v6: новые признаки 22-25 ──────────────────────────────────────────────

    # 22. Cloud cover
    if "cloud_cover" in df.columns and cloud_scaler is not None:
        cloud_scaled = cloud_scaler.transform(df["cloud_cover"].values.reshape(-1,1)).flatten().clip(0,1).astype(np.float32)
    elif "cloud_cover" in df.columns:
        cloud_scaled = df["cloud_cover"].values.clip(0,1).astype(np.float32)
    else:
        cloud_scaled = np.zeros(N, np.float32)

    # 23. EV load norm (уже нормализован в generator)
    ev_load = df["ev_load_norm"].values.clip(0,1).astype(np.float32) \
              if "ev_load_norm" in df.columns else np.zeros(N, np.float32)

    # 24. Solar gen norm (уже нормализован в generator)
    solar_gen = df["solar_gen_norm"].values.clip(0,1).astype(np.float32) \
                if "solar_gen_norm" in df.columns else np.zeros(N, np.float32)

    # 25. DSR active (бинарный флаг)
    dsr = df["dsr_active"].values.clip(0,1).astype(np.float32) \
          if "dsr_active" in df.columns else np.zeros(N, np.float32)

    return np.stack([
        cons, hour_sin, hour_cos, is_peak, is_night,
        is_weekend, is_holiday, temp, temp_sq, tariff_enc, doy_norm,
        humidity, wind, rolling_mean, rolling_std,
        lag_24h, lag_48h, lag_168h,
        dow_sin, dow_cos, month_sin, month_cos,
        cloud_scaled, ev_load, solar_gen, dsr,   # 22-25 ★ v6
    ], axis=1).astype(np.float32)


def _make_multivariate_windows(features, history, horizon):
    N, n_features = features.shape
    n_samples = N - history - horizon + 1
    if n_samples <= 0:
        raise ValueError(f"Недостаточно данных: N={N}, history={history}, horizon={horizon}")
    row_idx = np.arange(n_samples)[:, None]
    col_idx = np.arange(history)[None, :]
    X = features[row_idx + col_idx]
    y_start = np.arange(n_samples) + history
    y_idx   = y_start[:, None] + np.arange(horizon)[None, :]
    Y = features[y_idx, 0]
    return X.astype(np.float32), Y.astype(np.float32)


def prepare_data(df, history_length=48, forecast_horizon=24,
                 train_ratio=0.70, val_ratio=0.15):
    """Полный пайплайн подготовки данных v6 (26 признаков)."""
    # v5+: добавляем лаги ДО сплита
    df = _add_lag_columns(df)
    logger.info("Лаговые колонки добавлены: load_lag_24h, load_lag_48h, load_lag_168h")

    total     = len(df)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    raw_train = df_train["consumption"].values.astype(np.float32)
    raw_val   = df_val["consumption"].values.astype(np.float32)
    raw_test  = df_test["consumption"].values.astype(np.float32)

    # Fit скалеров ТОЛЬКО на train
    cons_scaler = MinMaxScaler((0,1)).fit(raw_train.reshape(-1,1))
    temp_scaler = MinMaxScaler((0,1))
    if "temperature" in df.columns:
        temp_scaler.fit(df_train["temperature"].values.reshape(-1,1))

    humidity_scaler = None
    if "humidity" in df.columns:
        humidity_scaler = MinMaxScaler((0,1)).fit(df_train["humidity"].values.reshape(-1,1))

    wind_scaler = None
    if "wind_speed" in df.columns:
        wind_scaler = MinMaxScaler((0,1)).fit(df_train["wind_speed"].values.reshape(-1,1))

    rolling_std_scaler = None
    if "rolling_std_24h" in df.columns:
        rolling_std_scaler = MinMaxScaler((0,1)).fit(df_train["rolling_std_24h"].values.reshape(-1,1))

    cloud_scaler = None
    if "cloud_cover" in df.columns:
        cloud_scaler = MinMaxScaler((0,1)).fit(df_train["cloud_cover"].values.reshape(-1,1))

    temp_sq_max = None
    if "temperature" in df.columns and "temperature_squared" not in df.columns:
        t_train = df_train["temperature"].values.astype(np.float32)
        temp_sq_max = float((t_train**2).max()) + 1e-8

    scaled_train = cons_scaler.transform(raw_train.reshape(-1,1)).flatten()
    scaled_val   = cons_scaler.transform(raw_val.reshape(-1,1)).flatten()
    scaled_test  = cons_scaler.transform(raw_test.reshape(-1,1)).flatten()

    kw = dict(humidity_scaler=humidity_scaler, wind_scaler=wind_scaler,
              rolling_std_scaler=rolling_std_scaler, cloud_scaler=cloud_scaler,
              temp_sq_max=temp_sq_max)
    feat_train = _build_feature_matrix(df_train, cons_scaler, temp_scaler, **kw)
    feat_val   = _build_feature_matrix(df_val,   cons_scaler, temp_scaler, **kw)
    feat_test  = _build_feature_matrix(df_test,  cons_scaler, temp_scaler, **kw)

    actual_n_features = feat_train.shape[1]
    X_train, Y_train = _make_multivariate_windows(feat_train, history_length, forecast_horizon)
    X_val,   Y_val   = _make_multivariate_windows(feat_val,   history_length, forecast_horizon)
    X_test,  Y_test  = _make_multivariate_windows(feat_test,  history_length, forecast_horizon)

    logger.info("Данные подготовлены v6 | train=%d val=%d test=%d | n_features=%d",
                len(X_train), len(X_val), len(X_test), actual_n_features)
    logger.info("X_train.shape: %s  Y_train.shape: %s", X_train.shape, Y_train.shape)
    logger.info("Скалеры: humidity=%s wind=%s rolling_std=%s cloud=%s",
                "ok" if humidity_scaler else "нет",
                "ok" if wind_scaler else "нет",
                "ok" if rolling_std_scaler else "нет",
                "ok" if cloud_scaler else "нет")

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
        "scaled_train": scaled_train, "scaled_val": scaled_val, "scaled_test": scaled_test,
        "scaler": cons_scaler, "temp_scaler": temp_scaler,
        "humidity_scaler": humidity_scaler, "wind_scaler": wind_scaler,
        "rolling_std_scaler": rolling_std_scaler, "cloud_scaler": cloud_scaler,
        "n_features": actual_n_features,
        "train_end_idx": train_end, "val_end_idx": val_end,
        "test_start_idx": val_end + history_length,
        "timestamps": df["timestamp"].values,
    }


def validate_data_integrity(data):
    log = logging.getLogger("smart_grid.data.preprocessing")
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]; Y = data[f"Y_{split}"]
        assert not np.any(np.isnan(X)), f"NaN в X_{split}"
        assert not np.any(np.isinf(X)), f"Inf в X_{split}"
        assert not np.any(np.isnan(Y)), f"NaN в Y_{split}"
        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] > 0, f"Пустой сплит {split}"
        y_max = float(Y.max()); y_min = float(Y.min())
        assert y_min >= -0.05, f"Y_{split} min={y_min:.4f}"
        assert y_max <= 1.05, f"Y_{split} max={y_max:.4f}"
    assert data["train_end_idx"] < data["val_end_idx"]
    assert data["val_end_idx"] < data["test_start_idx"]
    log.info("validate_data_integrity ПРОЙДЕНА | train=%d val=%d test=%d | Y in [%.3f, %.3f]",
             len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
             float(data["Y_train"].min()), float(data["Y_train"].max()))


def inverse_scale(scaler, data):
    original_shape = data.shape
    flat = data.flatten().reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(original_shape)
