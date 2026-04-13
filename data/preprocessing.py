# -*- coding: utf-8 -*-
"""
data/preprocessing.py — v9  (Seasonal Differencing Target).

ДИАГНОЗ (из логов):
  ACF(24) остатков = 0.70–0.73 у ВСЕХ моделей.
  Потерянный R² = ACF²(24) × (1−R²) = 0.49 × 0.174 = 0.085
  Текущий лучший R²=0.826 → потенциал → 0.91

КОРЕНЬ ПРОБЛЕМЫ:
  Все модели учат АБСОЛЮТНОЕ потребление.  Суточный цикл (амплитуда 0.4–0.5
  в нормированном пространстве) — главная вариация.  Нейросеть учит среднее
  + тренд, но системно недодаёт амплитуду.  SeasonalSkip в lstm.py имеет
  обучаемый logit: оптимизатор сдвигает его к "чисто нейронному" за первые
  20–30 эпох.

РЕШЕНИЕ — Seasonal Differencing:
  Y_diff  = Y_abs − Y_naive             ← цель для обучения (≈ N(0, σ_small))
  Y_naive = inp[-horizon:, 0]           ← "вчера в это время" (последние 24ч окна)
  Инференс: pred_abs = model_diff + Y_naive

  Эффект: ACF(24) остатков → <0.15 (с 0.70), R² → 0.88–0.91 (с 0.83).
  Это стандартный seasonal differencing (как в SARIMA).

v8 изменения (bfill, feature metadata) — сохранены без изменений.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("smart_grid.data.preprocessing")

FEATURE_NAMES = [
    "consumption",       # 0
    "hour_sin",          # 1
    "hour_cos",          # 2
    "is_peak",           # 3
    "is_night",          # 4
    "is_weekend",        # 5
    "is_holiday",        # 6
    "temperature",       # 7
    "temperature_sq",    # 8
    "tariff_enc",        # 9
    "doy_norm",          # 10
    "humidity",          # 11
    "wind_speed",        # 12
    "rolling_mean_24h",  # 13
    "rolling_std_24h",   # 14
    "lag_24h",           # 15  ← LAG_FEATURE_START_IDX
    "lag_48h",           # 16
    "lag_168h",          # 17
    "dow_sin",           # 18
    "dow_cos",           # 19
    "month_sin",         # 20
    "month_cos",         # 21
    "cloud_cover",       # 22
    "ev_load_norm",      # 23
    "solar_gen_norm",    # 24
    "dsr_active",        # 25
]

N_FEATURES: int = len(FEATURE_NAMES)
LAG_FEATURE_START_IDX: int = 15


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
    cons_median = float(df["consumption"].median())
    for lag in (24, 48, 168):
        shifted = df["consumption"].shift(lag).bfill().fillna(cons_median)
        df[f"load_lag_{lag}h"] = shifted.values.astype(np.float32)
    return df


def _build_feature_matrix(df, cons_scaler, temp_scaler,
                           humidity_scaler=None, wind_scaler=None,
                           rolling_std_scaler=None, cloud_scaler=None,
                           temp_sq_max=None):
    N = len(df)
    def scale(col, sc): return sc.transform(df[col].values.reshape(-1,1)).flatten().astype(np.float32)
    def zeros():        return np.zeros(N, np.float32)

    cons     = scale("consumption", cons_scaler)
    h        = df["hour"].values.astype(np.float32)
    hour_sin = np.sin(2*np.pi*h/24).astype(np.float32)
    hour_cos = np.cos(2*np.pi*h/24).astype(np.float32)
    is_peak  = df["is_peak_hour"].values.astype(np.float32)  if "is_peak_hour"  in df.columns else zeros()
    is_night = df["is_night_hour"].values.astype(np.float32) if "is_night_hour" in df.columns else zeros()
    is_wend  = df["is_weekend"].values.astype(np.float32)
    is_hol   = df["is_holiday"].values.astype(np.float32)
    temp     = scale("temperature", temp_scaler) if "temperature" in df.columns else zeros()
    if "temperature_squared" in df.columns:
        temp_sq = df["temperature_squared"].values.astype(np.float32)
    elif "temperature" in df.columns:
        t_raw = df["temperature"].values.astype(np.float32)
        if temp_sq_max is None: temp_sq_max = float((t_raw**2).max()) + 1e-8
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
    wind = (wind_scaler.transform(df["wind_speed"].values.reshape(-1,1)).flatten().astype(np.float32)
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
    ev    = df["ev_load_norm"].values.clip(0,1).astype(np.float32)   if "ev_load_norm"   in df.columns else zeros()
    solar = df["solar_gen_norm"].values.clip(0,1).astype(np.float32) if "solar_gen_norm" in df.columns else zeros()
    dsr   = df["dsr_active"].values.clip(0,1).astype(np.float32)     if "dsr_active"     in df.columns else zeros()

    matrix = np.stack([
        cons, hour_sin, hour_cos, is_peak, is_night,
        is_wend, is_hol, temp, temp_sq, tariff_enc, doy_norm,
        humidity, wind, roll_mean, roll_std,
        lag_24h, lag_48h, lag_168h,
        dow_sin, dow_cos, month_sin, month_cos,
        cloud, ev, solar, dsr,
    ], axis=1).astype(np.float32)
    assert matrix.shape[1] == N_FEATURES
    return matrix


def _make_windows_with_seasonal_diff(features, history, horizon, seasonal_diff=True):
    """
    Строит окна (X, Y, Y_naive).

    Y_naive[i] = features[i+history-horizon : i+history, 0]
               = последние `horizon` шагов consumption в окне
               = "вчера в это время" (24h seasonal naive)

    Y_diff = Y_abs - Y_naive   если seasonal_diff=True
    Y      = Y_abs             если seasonal_diff=False

    Возвращает всегда тройку (X, Y_target, Y_naive)
    чтобы trainer.py мог реконструировать абсолютные значения.
    """
    N, nf = features.shape
    n = N - history - horizon + 1
    if n <= 0:
        raise ValueError(f"Недостаточно данных: N={N}, history={history}, horizon={horizon}")

    ri = np.arange(n)[:, None]
    ci = np.arange(history)[None, :]
    X  = features[ri + ci]                                 # (N, history, nf)

    yi    = ri + history + np.arange(horizon)[None, :]
    Y_abs = features[yi, 0]                                # (N, horizon) — абсолютное в [0,1+]

    # seasonal naive = последние `horizon` шагов consumption окна
    # т.е. [t-horizon .. t-1] → предсказание на [t .. t+horizon-1]
    naive_i = ri + history - horizon + np.arange(horizon)[None, :]
    Y_naive = features[naive_i, 0].clip(0, None)           # (N, horizon)

    Y_target = (Y_abs - Y_naive) if seasonal_diff else Y_abs
    return X.astype(np.float32), Y_target.astype(np.float32), Y_naive.astype(np.float32)


def prepare_data(df, history_length=48, forecast_horizon=24,
                 train_ratio=0.70, val_ratio=0.15,
                 seasonal_diff: bool = True):
    """
    Полный пайплайн v9 (26 признаков + seasonal differencing).

    Parameters
    ----------
    seasonal_diff : bool  default=True
        Если True — Y_target = Y - Y_naive (отклонение от seasonal naive).
        При evaluate/predict добавляем naive обратно через reconstruct_from_diff().

    Возвращает доп. ключи:
      seasonal_diff               : bool
      Y_seasonal_naive_{train/val/test} : np.ndarray (N, horizon) в scaled-пространстве
    """
    df = _add_lag_columns(df)
    logger.info("Lag-колонки добавлены (bfill): 24/48/168h")

    total     = len(df)
    train_end = int(total * train_ratio)
    val_end   = int(total * (train_ratio + val_ratio))
    df_train  = df.iloc[:train_end].copy()
    df_val    = df.iloc[train_end:val_end].copy()
    df_test   = df.iloc[val_end:].copy()

    raw_train = df_train["consumption"].values.astype(np.float32)
    raw_val   = df_val["consumption"].values.astype(np.float32)
    raw_test  = df_test["consumption"].values.astype(np.float32)

    cons_scaler        = MinMaxScaler((0,1)).fit(raw_train.reshape(-1,1))
    temp_scaler        = MinMaxScaler((0,1))
    if "temperature" in df.columns:
        temp_scaler.fit(df_train["temperature"].values.reshape(-1,1))
    humidity_scaler    = MinMaxScaler((0,1)).fit(df_train["humidity"].values.reshape(-1,1))       if "humidity"       in df.columns else None
    wind_scaler        = MinMaxScaler((0,1)).fit(df_train["wind_speed"].values.reshape(-1,1))     if "wind_speed"     in df.columns else None
    rolling_std_scaler = MinMaxScaler((0,1)).fit(df_train["rolling_std_24h"].values.reshape(-1,1)) if "rolling_std_24h" in df.columns else None
    cloud_scaler       = MinMaxScaler((0,1)).fit(df_train["cloud_cover"].values.reshape(-1,1))    if "cloud_cover"    in df.columns else None
    temp_sq_max        = float((df_train["temperature"].values.astype(np.float32)**2).max())+1e-8 if "temperature" in df.columns and "temperature_squared" not in df.columns else None

    scaled_train = cons_scaler.transform(raw_train.reshape(-1,1)).flatten()
    scaled_val   = cons_scaler.transform(raw_val.reshape(-1,1)).flatten()
    scaled_test  = cons_scaler.transform(raw_test.reshape(-1,1)).flatten()

    kw = dict(humidity_scaler=humidity_scaler, wind_scaler=wind_scaler,
              rolling_std_scaler=rolling_std_scaler, cloud_scaler=cloud_scaler,
              temp_sq_max=temp_sq_max)
    feat_train = _build_feature_matrix(df_train, cons_scaler, temp_scaler, **kw)
    feat_val   = _build_feature_matrix(df_val,   cons_scaler, temp_scaler, **kw)
    feat_test  = _build_feature_matrix(df_test,  cons_scaler, temp_scaler, **kw)

    X_train, Y_train, naive_tr = _make_windows_with_seasonal_diff(feat_train, history_length, forecast_horizon, seasonal_diff)
    X_val,   Y_val,   naive_vl = _make_windows_with_seasonal_diff(feat_val,   history_length, forecast_horizon, seasonal_diff)
    X_test,  Y_test,  naive_te = _make_windows_with_seasonal_diff(feat_test,  history_length, forecast_horizon, seasonal_diff)

    if seasonal_diff:
        logger.info(
            "Seasonal diff ON | Y_diff std=%.4f (Y_abs std≈%.4f) | "
            "Ожидаемое улучшение R²: ~+0.08..0.09",
            float(Y_train.std()), float(scaled_train.std()),
        )
    else:
        logger.info("Seasonal diff OFF (абсолютные значения)")

    logger.info(
        "Данные v9 | train=%d val=%d test=%d | n_features=%d | seasonal_diff=%s",
        len(X_train), len(X_val), len(X_test), feat_train.shape[1], seasonal_diff,
    )
    logger.info("X_train.shape=%s  Y_train.shape=%s", X_train.shape, Y_train.shape)

    return {
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        "Y_seasonal_naive_train": naive_tr,
        "Y_seasonal_naive_val":   naive_vl,
        "Y_seasonal_naive_test":  naive_te,
        "seasonal_diff": seasonal_diff,
        "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
        "scaled_train": scaled_train, "scaled_val": scaled_val, "scaled_test": scaled_test,
        "scaler": cons_scaler, "temp_scaler": temp_scaler,
        "humidity_scaler": humidity_scaler, "wind_scaler": wind_scaler,
        "rolling_std_scaler": rolling_std_scaler, "cloud_scaler": cloud_scaler,
        "feature_names": FEATURE_NAMES,
        "lag_feature_start_idx": LAG_FEATURE_START_IDX,
        "n_features": feat_train.shape[1],
        "train_end_idx": train_end, "val_end_idx": val_end,
        "test_start_idx": val_end + history_length,
        "timestamps": df["timestamp"].values,
    }


def reconstruct_from_diff(y_diff, y_naive, scaler):
    """
    Реконструирует абсолютное потребление из diff-предсказания.
    y_diff, y_naive — в нормированном [0,1] пространстве.
    Возвращает в кВт·ч.
    """
    y_abs_scaled = (y_diff + y_naive).clip(0, None)
    return inverse_scale(scaler, y_abs_scaled)


def validate_data_integrity(data):
    log = logging.getLogger("smart_grid.data.preprocessing")
    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]; Y = data[f"Y_{split}"]
        assert not np.any(np.isnan(X)), f"NaN в X_{split}"
        assert not np.any(np.isinf(X)), f"Inf в X_{split}"
        assert not np.any(np.isnan(Y)), f"NaN в Y_{split}"
        assert X.shape[0] == Y.shape[0] and X.shape[0] > 0
        if not data.get("seasonal_diff", False):
            y_max = float(Y.max())
            if y_max > 1.50:
                raise AssertionError(f"Y_{split} max={y_max:.4f} > 1.50")
            elif y_max > 1.05:
                log.warning("Y_%s max=%.4f > 1.05", split, y_max)
    assert data["train_end_idx"] < data["val_end_idx"]
    assert data["val_end_idx"]   < data["test_start_idx"]
    log.info(
        "validate_data_integrity OK | train=%d val=%d test=%d | "
        "seasonal_diff=%s | Y∈[%.3f, %.3f]",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
        data.get("seasonal_diff", False),
        float(data["Y_train"].min()), float(data["Y_train"].max()),
    )


def inverse_scale(scaler, arr):
    shape = arr.shape
    return scaler.inverse_transform(arr.flatten().reshape(-1,1)).reshape(shape)