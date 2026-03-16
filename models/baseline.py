# -*- coding: utf-8 -*-
"""
models/baseline.py — Базовые модели: XGBoost с лаг-признаками, LinearRegression.

═══════════════════════════════════════════════════════════════════════════════
РЕРАЙТ v5: выравнивание информационного контекста XGBoost vs LinearRegression
═══════════════════════════════════════════════════════════════════════════════

ПРОБЛЕМА (v4): неравный информационный контекст
───────────────────────────────────────────────
LinearRegression (FlattenWrapper) получает 48×15=720 сырых признаков — прямой
доступ ко всему 48-часовому окну ВСЕХ 15 каналов. MAE=665.
XGBoost (LagFeaturesWrapper v4) получал 71 признак из канала 0 + ковариаты
последнего шага (channel[−1, 1:14]). Только потребление имело историю 48 шагов,
остальные 14 каналов — одна точка. MAE=1023.

Разрыв MAE объяснялся не слабостью алгоритма, а меньшим объёмом информации.

РЕШЕНИЕ v5: добавить rolling_mean(окна 6,12,24,48) и rolling_std(окна 6,24)
  ПО ВСЕМ 15 КАНАЛАМ.

  Это даёт XGBoost сопоставимый контекст:
    rolling_mean(w, ch) ≈ среднее значение канала ch за последние w часов
    rolling_std(w, ch)  ≈ волатильность канала ch за w часов

  Для непрерывных каналов (temp, humidity, wind, rolling_mean_24h) это прямые
  физические признаки. Для бинарных (is_weekend, is_holiday) rolling_mean даёт
  «частоту события за последние w часов» — тоже осмысленный признак.

СОСТАВ ПРИЗНАКОВ v5 (при T=48, F=15):
  ┌─ 48 лагов потребления (канал 0, все T шагов)
  ├─ 4 rolling mean cons (окна 3,6,12,24) — перекрытие с новыми, но XGBoost игнорирует
  ├─ 2 rolling std  cons (окна 6,24)
  ├─ 3 min/max/range cons последние 24ч
  ├─ 4 трендовых дельты (delta_1h, delta_3h, delta_24h, ratio_24h)
  ├─ 14 ковариат последнего шага (каналы 1–14)
  ├─ 15×4 rolling mean по всем каналам (окна 6,12,24,48)  ★ НОВОЕ
  └─ 15×2 rolling std  по всем каналам (окна 6,24)        ★ НОВОЕ

  Итого: 48 + 4 + 2 + 3 + 4 + 14 + 60 + 30 = 165 признаков
  Соотношение 165:6100 = 1:37 — безопасно с регуляризацией XGBoost.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

logger = logging.getLogger("smart_grid.models.baseline")


# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ПОСТРОЕНИЯ ЛАГ-ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════

def build_lag_features(X: np.ndarray) -> np.ndarray:
    """
    Преобразует 3D окно (N, T, F) в 2D матрицу признаков (N, ~165).

    v5 vs v4: добавлены rolling_mean(6,12,24,48) и rolling_std(6,24)
    ПО ВСЕМ F КАНАЛАМ — выравнивает информационный контекст с LinearRegression.

    LinearRegression получает 48×15=720 сырых значений (весь тензор).
    XGBoost v5 получает 165 агрегированных — сопоставимая информационная
    ёмкость при лучшей инвариантности к шуму.

    Состав (при T=48, F=15):
      ЧАСТЬ 1 — потребление (канал 0), всё окно:
        48 лагов + 4 rolling mean(3,6,12,24) + 2 rolling std(6,24)
        + 3 (min/max/range 24ч) + 4 тренда = 61 признак

      ЧАСТЬ 2 — ковариаты последнего шага:
        F-1 = 14 признаков (каналы 1..F-1)

      ЧАСТЬ 3 ★ NEW — rolling_mean(6,12,24,48) × F каналов = 4F признаков
      ЧАСТЬ 4 ★ NEW — rolling_std(6,24) × F каналов = 2F признаков

    Итого: 61 + 14 + 60 + 30 = 165 признаков (при F=15).

    Примечание: rolling по каналу 0 в частях 3–4 дублирует часть 1,
    но XGBoost через feature importance самостоятельно занулит дубликаты.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, F)

    Returns
    -------
    features : np.ndarray, shape (N, 61 + (F-1) + 6*F)
    """
    N, T, F = X.shape
    cons = X[:, :, 0]   # (N, T) — нормализованное потребление

    feature_list = []

    # ── ЧАСТЬ 1: Потребление (канал 0) ────────────────────────────────────────

    # 1a. Все T лагов потребления
    for i in range(T):
        feature_list.append(cons[:, i])

    # 1b. Скользящие агрегаты потребления
    for window in [3, 6, 12, 24]:
        w = min(window, T)
        feature_list.append(cons[:, -w:].mean(axis=1))
    for window in [6, 24]:
        w = min(window, T)
        feature_list.append(cons[:, -w:].std(axis=1))

    # 1c. Min / max / range за 24ч
    w24 = min(24, T)
    cons_24 = cons[:, -w24:]
    feature_list.append(cons_24.min(axis=1))
    feature_list.append(cons_24.max(axis=1))
    feature_list.append(cons_24.max(axis=1) - cons_24.min(axis=1))

    # 1d. Трендовые дельты
    feature_list.append(cons[:, -1] - cons[:, -2]  if T >= 2  else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] - cons[:, -4]  if T >= 4  else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] - cons[:, -25] if T >= 25 else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] / (cons[:, -25] + 1e-8) if T >= 25 else np.ones(N, np.float32))

    # ── ЧАСТЬ 2: Ковариаты последнего шага (каналы 1..F-1) ────────────────────
    for ch in range(1, F):
        feature_list.append(X[:, -1, ch])

    # ── ЧАСТЬ 3: rolling_mean(6,12,24,48) по ВСЕМ F каналам ★ ────────────────
    # Физический смысл:
    #   channel=0 (cons):       среднее потребление за w часов
    #   channel=7 (temp):       средняя температура за w часов
    #   channel=11 (humidity):  средняя влажность за w часов
    #   channel=12 (wind):      средний ветер за w часов
    #   channel=3/4 (is_peak):  доля пиковых часов за w часов
    # LinearRegression видит эти значения напрямую в сыром окне.
    # Здесь мы даём XGBoost агрегированный эквивалент.
    for window in [6, 12, 24, 48]:
        w = min(window, T)
        for ch in range(F):
            feature_list.append(X[:, -w:, ch].mean(axis=1))

    # ── ЧАСТЬ 4: rolling_std(6,24) по ВСЕМ F каналам ★ ──────────────────────
    # Std температуры за 6ч = вариабельность погоды в ближайшие часы.
    # Std потребления за 24ч = амплитуда суточного профиля.
    for window in [6, 24]:
        w = min(window, T)
        for ch in range(F):
            feature_list.append(X[:, -w:, ch].std(axis=1))

    result = np.stack(feature_list, axis=1).astype(np.float32)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ОБЁРТКИ
# ══════════════════════════════════════════════════════════════════════════════

class LagFeaturesWrapper:
    """
    Обёртка для XGBoost: 3D (N,T,F) → лаг-признаки (N,~31) → fit/predict.
    Использует MultiOutputRegressor для независимого обучения по каждому шагу.
    """

    def __init__(self, estimator: BaseEstimator, name: str = "") -> None:
        self.estimator = estimator
        self.name = name or type(estimator).__name__

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "LagFeaturesWrapper":
        X2d = build_lag_features(X_train)
        self.estimator.fit(X2d, Y_train)
        logger.info(
            "%s обучен | lag-features: %d | Y: %s",
            self.name, X2d.shape[1], str(Y_train.shape),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2d = build_lag_features(X)
        return self.estimator.predict(X2d)

    def __repr__(self) -> str:
        return f"LagFeaturesWrapper({self.name})"


class FlattenWrapper:
    """
    Разворачивает 3D тензоры (N, T, F) → 2D (N, T×F) для sklearn.
    Используется для LinearRegression (Ridge работает хорошо на полном окне).
    """

    def __init__(self, estimator: BaseEstimator, name: str = "") -> None:
        self.estimator = estimator
        self.name = name or type(estimator).__name__

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "FlattenWrapper":
        X2d = X_train.reshape(X_train.shape[0], -1)
        self.estimator.fit(X2d, Y_train)
        logger.info("%s обучен | X_flat: %d фич", self.name, X2d.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2d = X.reshape(X.shape[0], -1)
        return self.estimator.predict(X2d)

    def __repr__(self) -> str:
        return f"FlattenWrapper({self.name})"


# ══════════════════════════════════════════════════════════════════════════════
# ПУБЛИЧНЫЙ API
# ══════════════════════════════════════════════════════════════════════════════

def build_linear_regression(alpha: float = 1.0) -> FlattenWrapper:
    """
    Ridge regression на полном сглаженном окне (N, 528).
    Ridge нечувствителен к мультиколлинеарности — сырое окно ему подходит.
    alpha=1.0 — умеренная регуляризация.
    """
    return FlattenWrapper(Ridge(alpha=alpha), name="LinearRegression")


def build_xgboost(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.75,
    colsample_bytree: float = 0.80,
    min_child_weight: int = 10,
    seed: int = 42,
) -> LagFeaturesWrapper:
    """
    XGBoost на расширенных лаг-признаках (~165 фич) + MultiOutputRegressor.

    ИЗМЕНЕНИЯ v5 vs v4:
      v4: 71 признак — только потребление имело историю 48 шагов,
          остальные 14 каналов — одна точка на последнем шаге.
      v5: 165 признаков — rolling_mean(6,12,24,48) и rolling_std(6,24)
          добавлены ПО ВСЕМ 15 КАНАЛАМ. XGBoost теперь видит историю
          температуры, влажности, ветра — те же данные что LinearRegression,
          но в агрегированном виде.

      colsample_bytree=0.80: 165×0.8≈132 признака/дерево — сохраняем разнообразие.
      min_child_weight=10: с 165 признаками пространство разбивается точнее,
                           листья могут быть чуть меньше чем при 71 признаке.
      Соотношение 165:6100 = 1:37 — безопасно при reg_alpha+reg_lambda.
    """
    base_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        tree_method="hist",
    )

    multi_model = MultiOutputRegressor(base_model, n_jobs=-1)

    logger.info(
        "XGBoost v5 (FullLags + RollingAllChannels + MultiOutput) | "
        "~165 признаков | 24 независимых модели | n_est=%d depth=%d min_cw=%d",
        n_estimators, max_depth, min_child_weight,
    )
    return LagFeaturesWrapper(multi_model, name="XGBoost")