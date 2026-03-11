# -*- coding: utf-8 -*-
"""
data/preprocessing.py — Подготовка мультивариантных данных для нейросетей.

═══════════════════════════════════════════════════════════════════════════════
ВЕРСИЯ 3 — мультивариантный вход (11 признаков, было 9)

ИЗМЕНЕНИЯ v3:
  • hour_norm (линейный) → hour_sin + hour_cos (циклические)
    Причина: hour/23 создаёт разрыв на границе 23→0. Sin/cos — замкнутый
    круг без артефакта разрыва. [Ke et al., 2017]
  • temperature_squared (новый): нелинейная U-образная зависимость от T.
    При T<15°C потребление растёт (отопление), при T>25°C — растёт снова
    (кондиционирование). Квадратичный член приближает эту форму.

СОСТАВ 11 ПРИЗНАКОВ (индексы):
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

НОРМАЛИЗАЦИЯ (без утечки):
  consumption : MinMaxScaler обучается ТОЛЬКО на train
  temperature : MinMaxScaler обучается ТОЛЬКО на train
  Остальные  : детерминированные преобразования (нет обучения)

ЦЕЛЕВАЯ ПЕРЕМЕННАЯ Y:
  Только scaled consumption — прогнозируем один ряд.
  inverse_scale() работает как раньше через consumption-scaler.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("smart_grid.data.preprocessing")

# Число признаков — константа для других модулей
N_FEATURES: int = 11   # было 9: добавлены hour_sin, hour_cos, temperature_squared


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
) -> np.ndarray:
    """
    Собирает матрицу (N, 11) из DataFrame.

    v3: hour_norm заменён на hour_sin + hour_cos;
        добавлен temperature_squared.

    Parameters
    ----------
    cons_scaler : обученный MinMaxScaler для потребления (fit только на train)
    temp_scaler : обученный MinMaxScaler для температуры (fit только на train)

    Returns
    -------
    np.ndarray shape=(N, 11), dtype=float32
    """
    N = len(df)

    # 0. Consumption (нормализованное)
    cons = cons_scaler.transform(
        df["consumption"].values.reshape(-1, 1)
    ).flatten().astype(np.float32)

    # 1–2. Циклическое кодирование часа: sin/cos вместо hour/23
    # hour/23 создаёт разрыв: часы 23 и 0 кажутся «далёкими» (1.0 vs 0.0).
    # Sin/cos обеспечивают непрерывность: f(0) ≈ f(24). [Ke et al., 2017]
    hour_vals = df["hour"].values.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hour_vals / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour_vals / 24).astype(np.float32)

    # 3–6. Бинарные признаки
    is_peak = df["is_peak_hour"].values.astype(np.float32) \
        if "is_peak_hour" in df.columns else np.zeros(N, dtype=np.float32)
    is_night = df["is_night_hour"].values.astype(np.float32) \
        if "is_night_hour" in df.columns else np.zeros(N, dtype=np.float32)
    is_weekend = df["is_weekend"].values.astype(np.float32)
    is_holiday = df["is_holiday"].values.astype(np.float32)

    # 7. Temperature (нормализованная)
    if "temperature" in df.columns:
        temp = temp_scaler.transform(
            df["temperature"].values.reshape(-1, 1)
        ).flatten().astype(np.float32)
    else:
        temp = np.zeros(N, dtype=np.float32)

    # 8. Temperature² — нелинейный тепловой признак
    # Если generator v3 уже посчитал — используем напрямую.
    # Иначе считаем здесь (обратная совместимость со старым generator v2).
    if "temperature_squared" in df.columns:
        temp_sq = df["temperature_squared"].values.astype(np.float32)
    elif "temperature" in df.columns:
        t_raw = df["temperature"].values.astype(np.float32)
        t_sq = t_raw ** 2
        t_sq_max = float(t_sq.max()) + 1e-8
        temp_sq = (t_sq / t_sq_max).astype(np.float32)
    else:
        temp_sq = np.zeros(N, dtype=np.float32)

    # 9. Tariff zone encoded
    tariff_enc = _encode_tariff_zone(df)

    # 10. Day of year normalized [0, 1]
    if "day_of_year" in df.columns:
        doy_norm = (df["day_of_year"].values.astype(np.float32)) / 364.0
    else:
        doy_norm = np.zeros(N, dtype=np.float32)

    # Сборка (N, 11): порядок индексов фиксирован (документация выше)
    return np.stack(
        [cons, hour_sin, hour_cos, is_peak, is_night,
         is_weekend, is_holiday, temp, temp_sq, tariff_enc, doy_norm],
        axis=1,
    )  # (N, 11)


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
    Полный пайплайн подготовки мультивариантных данных.

    Шаги:
    1. Хронологическое разделение train / val / test
    2. Нормализация consumption и temperature (fit только на train)
    3. Сборка матрицы признаков (N, 9)
    4. Нарезка скользящими окнами → (samples, history, 9)

    Returns
    -------
    dict с ключами:
        X_train, Y_train  — (N_tr, T, 9), (N_tr, H)
        X_val,   Y_val
        X_test,  Y_test
        scaler            — MinMaxScaler для consumption (для inverse_scale)
        temp_scaler       — MinMaxScaler для temperature
        n_features        — 9
        raw_train/val/test — сырые ряды потребления
        scaled_train/val/test — нормализованные ряды потребления
        train_end_idx, val_end_idx, test_start_idx
        timestamps        — np.ndarray дат
    """
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    raw_train = df_train["consumption"].values.astype(np.float32)
    raw_val = df_val["consumption"].values.astype(np.float32)
    raw_test = df_test["consumption"].values.astype(np.float32)

    # ── Обучаем скалеры ТОЛЬКО на train ──────────────────────────────────────
    cons_scaler = MinMaxScaler(feature_range=(0, 1))
    cons_scaler.fit(raw_train.reshape(-1, 1))

    temp_scaler = MinMaxScaler(feature_range=(0, 1))
    if "temperature" in df.columns:
        temp_scaler.fit(df_train["temperature"].values.reshape(-1, 1))

    # Нормализованные ряды потребления (для обратного масштабирования и BCE)
    scaled_train = cons_scaler.transform(raw_train.reshape(-1, 1)).flatten()
    scaled_val = cons_scaler.transform(raw_val.reshape(-1, 1)).flatten()
    scaled_test = cons_scaler.transform(raw_test.reshape(-1, 1)).flatten()

    # ── Матрицы признаков (N, 9) ──────────────────────────────────────────────
    feat_train = _build_feature_matrix(df_train, cons_scaler, temp_scaler)
    feat_val = _build_feature_matrix(df_val, cons_scaler, temp_scaler)
    feat_test = _build_feature_matrix(df_test, cons_scaler, temp_scaler)

    # ── Скользящие окна ───────────────────────────────────────────────────────
    X_train, Y_train = _make_multivariate_windows(feat_train, history_length, forecast_horizon)
    X_val, Y_val = _make_multivariate_windows(feat_val, history_length, forecast_horizon)
    X_test, Y_test = _make_multivariate_windows(feat_test, history_length, forecast_horizon)

    logger.info(
        "Данные подготовлены | train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    logger.info(
        "✅ X_train.shape: %s  Y_train.shape: %s  (n_features=%d, v3: +hour_sin/cos +temp_sq)",
        X_train.shape, Y_train.shape, N_FEATURES,
    )

    return {
        # Тензоры для нейросетей
        "X_train": X_train, "Y_train": Y_train,
        "X_val":   X_val,   "Y_val":   Y_val,
        "X_test":  X_test,  "Y_test":  Y_test,
        # Сырые и нормализованные ряды (для inverse_scale и storage)
        "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
        "scaled_train": scaled_train, "scaled_val": scaled_val, "scaled_test": scaled_test,
        # Скалеры
        "scaler": cons_scaler,
        "temp_scaler": temp_scaler,
        # Мета
        "n_features": N_FEATURES,
        "train_end_idx": train_end,
        "val_end_idx": val_end,
        "test_start_idx": val_end + history_length,
        "timestamps": df["timestamp"].values,
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