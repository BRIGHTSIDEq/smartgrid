# -*- coding: utf-8 -*-
"""
Классические статистические модели и ансамбль прогнозов.

Оба метода были в курсовой. Ошибкой там был не выбор методов, а выбор
победителя по тесту; здесь они возвращены на ту же методику, что и остальные
модели: параметры подбираются только по валидации.

HoltWintersAdditive — экспоненциальное сглаживание ETS(A,N,A) с суточным
сезоном. Классический базлайн для нагрузки: если нейросеть его не бьёт, её
сложность не оправдана.

ValidationWeightedEnsemble — взвешенное среднее прогнозов обученных моделей,
веса по Бейтсу–Грейнджеру (обратно пропорциональны MSE на валидации).
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("smart_grid.models.statistical")

CH_CONSUMPTION = 0


# ══════════════════════════════════════════════════════════════════════════════
# ETS(A,N,A)
# ══════════════════════════════════════════════════════════════════════════════

def holt_winters_forecast(history: np.ndarray, horizon: int, alpha: float,
                          gamma: float, season: int = 24) -> np.ndarray:
    """
    Прогноз аддитивного Хольта–Винтерса без тренда для каждой строки history.

    Состояние инициализируется по первому сезону окна: уровень — среднее,
    сезонные компоненты — отклонения от него. Затем рекурсия проходит по
    остальной истории:

        l_t = α (x_t − s_{t−m}) + (1 − α) l_{t−1}
        s_t = γ (x_t − l_t) + (1 − γ) s_{t−m}
        ŷ_{T+h} = l_T + s_{T+h−m·k}

    Вычисление векторизовано по окнам: цикл идёт только по времени внутри
    окна, поэтому прогноз для сотни тысяч окон занимает доли секунды.
    """
    x = np.asarray(history, dtype=np.float64)
    n, T = x.shape
    if T < season:
        raise ValueError(f"Окно {T} ч короче сезона {season} ч")

    level = x[:, :season].mean(axis=1)
    seasonal = np.empty((n, T), dtype=np.float64)
    seasonal[:, :season] = x[:, :season] - level[:, None]
    for t in range(season, T):
        s_prev = seasonal[:, t - season]
        new_level = alpha * (x[:, t] - s_prev) + (1.0 - alpha) * level
        seasonal[:, t] = gamma * (x[:, t] - new_level) + (1.0 - gamma) * s_prev
        level = new_level

    idx = [T - season + (h % season) for h in range(horizon)]
    return (level[:, None] + seasonal[:, idx]).astype(np.float32)


class HoltWintersAdditive:
    """
    ETS(A,N,A) с суточным сезоном; α и γ выбираются по валидации.

    Подбор по валидации, а не максимизацией правдоподобия на train, сделан
    намеренно: так же выбираются alpha у Ridge и число деревьев у XGBoost, и
    сравнение моделей идёт при одинаковом доступе к данным.
    """

    GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

    def __init__(self, season: int = 24, grid: Sequence[float] = GRID) -> None:
        self.season = int(season)
        self.grid = tuple(grid)
        self.alpha_: Optional[float] = None
        self.gamma_: Optional[float] = None
        self.horizon: Optional[int] = None
        self.val_mae_: Optional[float] = None

    def fit(self, X_train, Y_train, X_val=None, Y_val=None) -> "HoltWintersAdditive":
        self.horizon = int(Y_train.shape[1])
        if X_val is None or Y_val is None:
            raise ValueError("HoltWintersAdditive подбирает α и γ по валидации; "
                             "без неё параметры не выбираются")
        hist = X_val[:, :, CH_CONSUMPTION]
        best = (np.inf, None, None)
        for a in self.grid:
            for g in self.grid:
                pred = holt_winters_forecast(hist, self.horizon, a, g, self.season)
                mae = float(np.mean(np.abs(pred - Y_val)))
                if mae < best[0]:
                    best = (mae, a, g)
        self.val_mae_, self.alpha_, self.gamma_ = best
        logger.info("ETS(A,N,A): α=%.2f γ=%.2f (MAE_val=%.5f, перебрано %d пар)",
                    self.alpha_, self.gamma_, self.val_mae_, len(self.grid) ** 2)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("Модель не обучена: вызовите fit() перед predict().")
        return holt_winters_forecast(X[:, :, CH_CONSUMPTION], self.horizon,
                                     self.alpha_, self.gamma_, self.season)


def build_ets() -> HoltWintersAdditive:
    return HoltWintersAdditive()


# ══════════════════════════════════════════════════════════════════════════════
# АНСАМБЛЬ С ВЕСАМИ ПО ВАЛИДАЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def bates_granger_weights(val_mse: Dict[str, float]) -> Dict[str, float]:
    """
    Веса, обратно пропорциональные MSE на валидации (Bates & Granger, 1969).

    Схема выбрана вместо подгонки весов регрессией: одна статистика на модель
    почти не переобучается на валидации, тогда как свободные веса
    подстраиваются под её шум и завышают качество ансамбля ровно на той
    выборке, по которой выбирается лучшая модель.
    """
    finite = {k: v for k, v in val_mse.items() if np.isfinite(v) and v > 0}
    if not finite:
        raise ValueError("Нет моделей с конечной ошибкой на валидации")
    inv = {k: 1.0 / v for k, v in finite.items()}
    total = sum(inv.values())
    return {k: w / total for k, w in inv.items()}


class ValidationWeightedEnsemble:
    """
    Взвешенное среднее прогнозов уже обученных моделей.

    Члены ансамбля — объекты с методом predict(X) в масштабе обучения (обёртки
    ModelTrainer). Веса считаются по валидации, поэтому в отбор лучшей модели
    ансамбль не допускается: его валидационная оценка смещена в его пользу.
    """

    def __init__(self, members: List[Tuple[str, Any]]) -> None:
        self.members = list(members)
        self.weights_: Dict[str, float] = {}

    def fit(self, X_train, Y_train, X_val=None, Y_val=None) -> "ValidationWeightedEnsemble":
        if X_val is None or Y_val is None:
            raise ValueError("Веса ансамбля считаются по валидации")
        if len(self.members) < 2:
            raise ValueError("Для ансамбля нужно хотя бы две модели")
        mse = {name: float(np.mean((np.asarray(m.predict(X_val)) - Y_val) ** 2))
               for name, m in self.members}
        self.weights_ = bates_granger_weights(mse)
        self.members = [(n, m) for n, m in self.members if n in self.weights_]
        logger.info("Ансамбль: веса по валидации %s",
                    ", ".join(f"{n}={w:.3f}" for n, w in self.weights_.items()))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("Ансамбль не обучен: вызовите fit() перед predict().")
        out = None
        for name, member in self.members:
            part = self.weights_[name] * np.asarray(member.predict(X), dtype=np.float64)
            out = part if out is None else out + part
        return out.astype(np.float32)
