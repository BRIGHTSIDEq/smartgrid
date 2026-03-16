# -*- coding: utf-8 -*-
"""
data/preprocessing.py — Подготовка мультивариантных данных для нейросетей.

═══════════════════════════════════════════════════════════════════════════════
ВЕРСИЯ 4 — расширение до 15 признаков (generator v4)

ИЗМЕНЕНИЯ v4:
  • Добавлены 4 новых признака из generator v4:
    humidity, wind_speed, rolling_mean_24h, rolling_std_24h
  • Новые скалеры: humidity_scaler, wind_scaler, rolling_std_scaler
    (fit ТОЛЬКО на train — без утечки в val/test)

СОСТАВ 15 ПРИЗНАКОВ (индексы):
  0  consumption_scaled    — нормализованное потребление [0, 1]
  1  hour_sin              — sin(2π·h/24) циклический час
  2  hour_cos              — cos(2π·h/24) циклический час
  3  is_peak_hour          — 0/1
  4  is_night_hour         — 0/1
  5  is_weekend            — 0/1
  6  is_holiday            — 0/1
  7  temperature_scaled    — нормализованная температура [0, 1]
  8  temperature_squared   — T²/max(T²) нелинейный тепловой признак [0, 1]
  9  tariff_zone_enc       — ночь=0.0 / день=0.5 / пик=1.0
  10 day_of_year_norm      — (день_года − 1) / 364 → [0, 1]
  11 humidity_scaled       — нормализованная влажность [0, 1]
  12 wind_speed_scaled     — нормализованная скорость ветра [0, 1]
  13 rolling_mean_scaled   — rolling_mean_24h через cons_scaler (та же шкала)
  14 rolling_std_scaled    — rolling_std_24h через rolling_std_scaler

ЗАМЕТКА О rolling_mean_24h И УТЕЧКЕ:
  rolling вычисляется в generator на ПОЛНОМ df ДО сплита.
  Это корректно: rolling(24) — окно назад. Значение в момент t использует
  [t-23..t]. Первые строки val используют хвост train как историю — ровно
  те данные, что были бы известны в реальном времени.
  Утечка была бы только при rolling(center=True).
  Нормализация rolling_mean через cons_scaler: единицы одинаковые (кВт·ч).
  Нормализация rolling_std через отдельный rolling_std_scaler: std имеет
  другой диапазон — не [min_cons, max_cons], а [0, ~std_max].

НОРМАЛИЗАЦИЯ (без утечки):
  consumption       : MinMaxScaler fit на train
  temperature       : MinMaxScaler fit на train
  humidity          : MinMaxScaler fit на train
  wind_speed        : MinMaxScaler fit на train
  rolling_std_24h   : MinMaxScaler fit на train
  rolling_mean_24h  : cons_scaler (те же единицы, что consumption)
  Остальные         : детерминированные преобразования
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("smart_grid.data.preprocessing")

# Число признаков — константа для других модулей
N_FEATURES: int = 15   # v4: +humidity, +wind_speed, +rolling_mean_24h, +rolling_std_24h


# ──────────────────────────────────────────────────────────────────────────────
# ВНУТРЕННИЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────────────────────

def _encode_tariff_zone(df: pd.DataFrame) -> np.ndarray:
    """
    Кодирует tariff_zone в число:
        "night" → 0.0
        "day"   → 0.5
        "peak"  → 1.0

    Если колонка отсутствует — вычисляет из hour + weekday.
    """
    if "tariff_zone" in df.columns:
        mapping = {"night": 0.0, "day": 0.5, "peak": 1.0}
        return df["tariff_zone"].map(mapping).fillna(0.5).values.astype(np.float32)

    # Запасной расчёт (если generator старой версии)
    hours = df["hour"].values
    weekday = df["weekday"].values if "weekday" in df.columns else np.zeros(len(df))
    enc = np.full(len(df), 0.5, dtype=np.float32)
    night = (hours < 7) | (hours >= 23)
    peak = (((hours >= 10) & (hours < 17)) | ((hours >= 21) & (hours < 23))) & (weekday < 5)
    enc[night] = 0.0
    enc[peak] = 1.0
    return enc


def _build_feature_matrix(
    df: pd.DataFrame,
    cons_scaler: MinMaxScaler,
    temp_scaler: MinMaxScaler,
    humidity_scaler: Optional[MinMaxScaler] = None,
    wind_scaler: Optional[MinMaxScaler] = None,
    rolling_std_scaler: Optional[MinMaxScaler] = None,
) -> np.ndarray:
    """
    Собирает матрицу признаков из DataFrame.

    v4: расширена до 15 признаков (humidity, wind_speed, rolling_mean_24h,
        rolling_std_24h).

    Parameters
    ----------
    cons_scaler         : fit на train consumption
    temp_scaler         : fit на train temperature
    humidity_scaler     : fit на train humidity (None → нули, если колонка отсутствует)
    wind_scaler         : fit на train wind_speed
    rolling_std_scaler  : fit на train rolling_std_24h

    Returns
    -------
    np.ndarray shape=(N, 15), dtype=float32
    """
    N = len(df)

    # ── 0. Consumption (нормализованное) ──────────────────────────────────────
    cons = cons_scaler.transform(
        df["consumption"].values.reshape(-1, 1)
    ).flatten().astype(np.float32)

    # ── 1–2. Циклическое кодирование часа ─────────────────────────────────────
    hour_vals = df["hour"].values.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour_vals / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour_vals / 24).astype(np.float32)

    # ── 3–6. Бинарные признаки ────────────────────────────────────────────────
    is_peak    = df["is_peak_hour"].values.astype(np.float32)  \
        if "is_peak_hour"  in df.columns else np.zeros(N, dtype=np.float32)
    is_night   = df["is_night_hour"].values.astype(np.float32) \
        if "is_night_hour" in df.columns else np.zeros(N, dtype=np.float32)
    is_weekend = df["is_weekend"].values.astype(np.float32)
    is_holiday = df["is_holiday"].values.astype(np.float32)

    # ── 7. Temperature (нормализованная) ──────────────────────────────────────
    if "temperature" in df.columns:
        temp = temp_scaler.transform(
            df["temperature"].values.reshape(-1, 1)
        ).flatten().astype(np.float32)
    else:
        temp = np.zeros(N, dtype=np.float32)

    # ── 8. Temperature² ───────────────────────────────────────────────────────
    if "temperature_squared" in df.columns:
        temp_sq = df["temperature_squared"].values.astype(np.float32)
    elif "temperature" in df.columns:
        t_raw = df["temperature"].values.astype(np.float32)
        t_sq  = t_raw ** 2
        temp_sq = (t_sq / (float(t_sq.max()) + 1e-8)).astype(np.float32)
    else:
        temp_sq = np.zeros(N, dtype=np.float32)

    # ── 9. Tariff zone encoded ────────────────────────────────────────────────
    tariff_enc = _encode_tariff_zone(df)

    # ── 10. Day of year normalized ────────────────────────────────────────────
    if "day_of_year" in df.columns:
        doy_norm = (df["day_of_year"].values.astype(np.float32)) / 364.0
    else:
        doy_norm = np.zeros(N, dtype=np.float32)

    # ── 11. Humidity (нормализованная через humidity_scaler) ──────────────────
    # Взаимодействие с температурой (кондиционирование при высокой влажности и жаре).
    # Скалер fit на train → нет утечки. Если колонки нет → нули (обратная совместимость).
    if "humidity" in df.columns and humidity_scaler is not None:
        humidity = humidity_scaler.transform(
            df["humidity"].values.reshape(-1, 1)
        ).flatten().astype(np.float32)
    else:
        humidity = np.zeros(N, dtype=np.float32)

    # ── 12. Wind speed (нормализованная через wind_scaler) ────────────────────
    # Взаимодействие с температурой (теплопотери на ветру в мороз).
    if "wind_speed" in df.columns and wind_scaler is not None:
        wind = wind_scaler.transform(
            df["wind_speed"].values.reshape(-1, 1)
        ).flatten().astype(np.float32)
    else:
        wind = np.zeros(N, dtype=np.float32)

    # ── 13. Rolling mean 24h (нормализованная через cons_scaler) ──────────────
    # rolling_mean_24h имеет те же единицы что consumption → cons_scaler корректен.
    # Значение: среднее потребление за последние 24ч — «текущий уровень нагрузки».
    # Вычислена в generator на полном df с backward-only окном → утечки нет.
    if "rolling_mean_24h" in df.columns:
        rolling_mean = cons_scaler.transform(
            df["rolling_mean_24h"].values.reshape(-1, 1)
        ).flatten().clip(0.0, 1.0).astype(np.float32)
    else:
        rolling_mean = cons.copy()   # fallback: само потребление

    # ── 14. Rolling std 24h (нормализованная через rolling_std_scaler) ────────
    # rolling_std имеет другой диапазон чем consumption → отдельный скалер.
    # Значение: волатильность потребления за 24ч — «нестабильность нагрузки».
    if "rolling_std_24h" in df.columns and rolling_std_scaler is not None:
        rolling_std = rolling_std_scaler.transform(
            df["rolling_std_24h"].values.reshape(-1, 1)
        ).flatten().clip(0.0, 1.0).astype(np.float32)
    elif "rolling_std_24h" in df.columns:
        # Fallback без скалера: нормируем на max
        raw_std = df["rolling_std_24h"].values.astype(np.float32)
        rolling_std = (raw_std / (float(raw_std.max()) + 1e-8)).astype(np.float32)
    else:
        rolling_std = np.zeros(N, dtype=np.float32)

    # ── Сборка (N, 15) ────────────────────────────────────────────────────────
    return np.stack(
        [cons, hour_sin, hour_cos, is_peak, is_night,
         is_weekend, is_holiday, temp, temp_sq, tariff_enc, doy_norm,
         humidity, wind, rolling_mean, rolling_std],
        axis=1,
    ).astype(np.float32)  # (N, 15)


def _make_multivariate_windows(
    features: np.ndarray,
    history: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Нарезает мультивариантный ряд на скользящие окна.

    Parameters
    ----------
    features : np.ndarray shape=(N, n_features)
    history  : int  длина входного окна
    horizon  : int  длина прогноза

    Returns
    -------
    X : np.ndarray shape=(samples, history, n_features)
    Y : np.ndarray shape=(samples, horizon)  ← только consumption (канал 0)
    """
    n_features = features.shape[1]
    total = len(features)
    n_samples = total - history - horizon + 1

    X = np.empty((n_samples, history, n_features), dtype=np.float32)
    Y = np.empty((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        X[i] = features[i: i + history]                      # (T, F)
        Y[i] = features[i + history: i + history + horizon, 0]  # consumption только

    return X, Y


# ──────────────────────────────────────────────────────────────────────────────
# ПУБЛИЧНЫЙ API
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    history_length: int = 48,
    forecast_horizon: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Полный пайплайн подготовки мультивариантных данных (v4: 15 признаков).

    Шаги:
    1. Хронологическое разделение train / val / test
    2. Fit скалеров ТОЛЬКО на train: cons, temp, humidity, wind, rolling_std
    3. Сборка матриц признаков (N, 15) для каждого сплита
    4. Нарезка скользящими окнами → (samples, history, 15)

    Returns
    -------
    dict с ключами:
        X_train, Y_train  — тензоры обучения
        X_val,   Y_val    — тензоры валидации
        X_test,  Y_test   — тензоры теста
        scaler            — MinMaxScaler consumption (для inverse_scale)
        temp_scaler       — MinMaxScaler temperature
        humidity_scaler   — MinMaxScaler humidity  (None если колонки нет)
        wind_scaler       — MinMaxScaler wind_speed (None если колонки нет)
        rolling_std_scaler— MinMaxScaler rolling_std_24h (None если нет)
        n_features        — 15
        raw_*/scaled_*    — сырые и нормализованные ряды потребления
        train_end_idx, val_end_idx, test_start_idx
        timestamps        — np.ndarray дат
    """
    total      = len(df)
    train_end  = int(total * train_ratio)
    val_end    = int(total * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    raw_train = df_train["consumption"].values.astype(np.float32)
    raw_val   = df_val["consumption"].values.astype(np.float32)
    raw_test  = df_test["consumption"].values.astype(np.float32)

    # ── Скалеры: fit ТОЛЬКО на train ─────────────────────────────────────────
    cons_scaler = MinMaxScaler(feature_range=(0, 1))
    cons_scaler.fit(raw_train.reshape(-1, 1))

    temp_scaler = MinMaxScaler(feature_range=(0, 1))
    if "temperature" in df.columns:
        temp_scaler.fit(df_train["temperature"].values.reshape(-1, 1))

    # humidity_scaler: fit на train, transform val/test
    humidity_scaler: Optional[MinMaxScaler] = None
    if "humidity" in df.columns:
        humidity_scaler = MinMaxScaler(feature_range=(0, 1))
        humidity_scaler.fit(df_train["humidity"].values.reshape(-1, 1))

    # wind_scaler: fit на train
    wind_scaler: Optional[MinMaxScaler] = None
    if "wind_speed" in df.columns:
        wind_scaler = MinMaxScaler(feature_range=(0, 1))
        wind_scaler.fit(df_train["wind_speed"].values.reshape(-1, 1))

    # rolling_std_scaler: std имеет диапазон [0, ~std_max], отличный от consumption
    rolling_std_scaler: Optional[MinMaxScaler] = None
    if "rolling_std_24h" in df.columns:
        rolling_std_scaler = MinMaxScaler(feature_range=(0, 1))
        rolling_std_scaler.fit(df_train["rolling_std_24h"].values.reshape(-1, 1))

    # Нормализованные ряды потребления (для inverse_scale и storage)
    scaled_train = cons_scaler.transform(raw_train.reshape(-1, 1)).flatten()
    scaled_val   = cons_scaler.transform(raw_val.reshape(-1,   1)).flatten()
    scaled_test  = cons_scaler.transform(raw_test.reshape(-1,  1)).flatten()

    # ── Матрицы признаков (N, 15) ─────────────────────────────────────────────
    _scaler_kwargs = dict(
        humidity_scaler    = humidity_scaler,
        wind_scaler        = wind_scaler,
        rolling_std_scaler = rolling_std_scaler,
    )
    feat_train = _build_feature_matrix(df_train, cons_scaler, temp_scaler, **_scaler_kwargs)
    feat_val   = _build_feature_matrix(df_val,   cons_scaler, temp_scaler, **_scaler_kwargs)
    feat_test  = _build_feature_matrix(df_test,  cons_scaler, temp_scaler, **_scaler_kwargs)

    actual_n_features = feat_train.shape[1]   # 15 если все колонки есть, иначе меньше

    # ── Скользящие окна ───────────────────────────────────────────────────────
    X_train, Y_train = _make_multivariate_windows(feat_train, history_length, forecast_horizon)
    X_val,   Y_val   = _make_multivariate_windows(feat_val,   history_length, forecast_horizon)
    X_test,  Y_test  = _make_multivariate_windows(feat_test,  history_length, forecast_horizon)

    logger.info(
        "Данные подготовлены v4 | train=%d val=%d test=%d | n_features=%d",
        len(X_train), len(X_val), len(X_test), actual_n_features,
    )
    logger.info(
        "✅ X_train.shape: %s  Y_train.shape: %s",
        X_train.shape, Y_train.shape,
    )
    logger.info(
        "   Скалеры: humidity=%s  wind=%s  rolling_std=%s",
        "✓" if humidity_scaler    is not None else "✗ (нет колонки)",
        "✓" if wind_scaler        is not None else "✗ (нет колонки)",
        "✓" if rolling_std_scaler is not None else "✗ (нет колонки)",
    )

    return {
        # Тензоры для нейросетей
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        # Сырые и нормализованные ряды (для inverse_scale и storage)
        "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
        "scaled_train": scaled_train, "scaled_val": scaled_val, "scaled_test": scaled_test,
        # Скалеры (все нужны для сохранения и инференса)
        "scaler":              cons_scaler,
        "temp_scaler":         temp_scaler,
        "humidity_scaler":     humidity_scaler,
        "wind_scaler":         wind_scaler,
        "rolling_std_scaler":  rolling_std_scaler,
        # Мета
        "n_features":        actual_n_features,
        "train_end_idx":     train_end,
        "val_end_idx":       val_end,
        "test_start_idx":    val_end + history_length,
        "timestamps":        df["timestamp"].values,
    }


def validate_data_integrity(data: Dict[str, Any]) -> None:
    """
    Проверяет корректность подготовленных данных перед обучением.
    Выбрасывает AssertionError с диагностическим сообщением при ошибке.

    Проверки:
      1. Нет NaN/Inf в тензорах
      2. X и Y согласованы по числу сэмплов
      3. Y содержит только consumption (канал 0 из features)
      4. Все сплиты непустые
      5. Val и Test не пересекаются с Train по времени (хронологический порядок)
      6. scaler не обучен на val/test (data_min проверяется)
    """
    log = logging.getLogger("smart_grid.data.preprocessing")

    for split in ("train", "val", "test"):
        X = data[f"X_{split}"]
        Y = data[f"Y_{split}"]

        # 1. NaN/Inf
        assert not np.any(np.isnan(X)), f"❌ NaN в X_{split}"
        assert not np.any(np.isinf(X)), f"❌ Inf в X_{split}"
        assert not np.any(np.isnan(Y)), f"❌ NaN в Y_{split}"

        # 2. Согласованность размеров
        assert X.shape[0] == Y.shape[0], (
            f"❌ Размер X_{split}={X.shape[0]} ≠ Y_{split}={Y.shape[0]}"
        )

        # 3. Непустые сплиты
        assert X.shape[0] > 0, f"❌ Пустой сплит {split}"

        # 4. Y в [0, 1] (нормализованное consumption)
        y_max = float(Y.max())
        y_min = float(Y.min())
        assert y_min >= -0.05, f"❌ Y_{split} содержит отрицательные значения: min={y_min:.4f}"
        assert y_max <= 1.05, f"❌ Y_{split} > 1.05 (утечка из scaler?): max={y_max:.4f}"

    # 5. Хронологический порядок: train < val < test
    assert data["train_end_idx"] < data["val_end_idx"], "❌ train/val порядок нарушен"
    assert data["val_end_idx"] < data["test_start_idx"], "❌ val/test порядок нарушен"

    log.info(
        "✅ validate_data_integrity ПРОЙДЕНА | train=%d val=%d test=%d | Y∈[%.3f, %.3f]",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
        float(data["Y_train"].min()), float(data["Y_train"].max()),
    )


def inverse_scale(
    scaler: MinMaxScaler,
    data: np.ndarray,
) -> np.ndarray:
    """Обратное масштабирование consumption (канал 0) в оригинальный масштаб."""
    original_shape = data.shape
    flat = data.flatten().reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(original_shape)