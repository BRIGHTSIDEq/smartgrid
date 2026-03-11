# -*- coding: utf-8 -*-
"""
models/baseline.py — Базовые модели: XGBoost, LinearRegression.

ИСПРАВЛЕНИЕ v2:
  XGBoost при n_features=9, history=48: X_flat имеет 9×48 = 432 колонки.
  Старые настройки (n_estimators=200, max_depth=6, colsample=0.80) давали underfitting:
    - 200 деревьев мало для 432-мерного пространства
    - colsample=0.80 → 432×0.8 = 346 фич/дерево → переобучение на коррелирующих фичах

  Новые настройки:
    - n_estimators=500  — больше деревьев (lazy ensemble)
    - max_depth=5       — чуть меньше (снижаем переобучение)
    - colsample=0.40    — 432×0.4 = 173 фич/дерево (лучше разнообразие)
    - subsample=0.80    — без изменений
    - min_child_weight=5 — защита от мелких листьев на 432 фичах
"""

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import xgboost as xgb

logger = logging.getLogger("smart_grid.models.baseline")


class FlattenWrapper:
    """
    Разворачивает 3D тензоры (N, T, F) → 2D (N, T×F) для sklearn/xgboost.
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
        n_feat = X2d.shape[1]
        logger.info("%s обучен | X_flat: %d фич", self.name, n_feat)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2d = X.reshape(X.shape[0], -1)
        return self.estimator.predict(X2d)

    def __repr__(self) -> str:
        return f"FlattenWrapper({self.name})"


def build_linear_regression(alpha: float = 1.0) -> FlattenWrapper:
    """
    Ridge regression вместо OLS LinearRegression.
    Причина: при 432 фичах обычный OLS может быть нестабилен (мультиколлинеарность).
    Ridge с alpha=1.0 даёт более устойчивое решение.
    """
    return FlattenWrapper(Ridge(alpha=alpha), name="LinearRegression")


def build_xgboost(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    subsample: float = 0.80,
    colsample_bytree: float = 0.40,
    min_child_weight: int = 5,
    seed: int = 42,
) -> FlattenWrapper:
    """
    XGBoost настроен под 432 входных фичи (9 признаков × 48 шагов).

    Ключевые параметры для высокой размерности:
      colsample_bytree=0.40: каждое дерево видит ~173 из 432 фич
        → снижает корреляцию между деревьями, улучшает ансамбль
      min_child_weight=5:   листья требуют минимум 5 наблюдений
        → защита от переобучения на редких паттернах
      n_estimators=500:     lazy ensemble — больше деревьев компенсируют
        малый размер каждого дерева (max_depth=5 vs 6)
    """
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
        # early_stopping_rounds: УБРАН.
        # Причина: при multi-output (24 targets) eval_metric усредняется по 24 головам →
        # шумный сигнал val_loss → преждевременная остановка на ~50-100 деревьях вместо 500.
        # При 528 фичах и 50 деревьях = underfitting. С 500 деревьями + lr=0.05 модель
        # уже достаточно регуляризована через shrinkage (ранняя остановка не нужна).
    )
    return FlattenWrapper(model, name="XGBoost")