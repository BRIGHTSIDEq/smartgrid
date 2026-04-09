# -*- coding: utf-8 -*-
"""
utils/metrics.py — Функции расчёта метрик качества прогноза.
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger("smart_grid.utils.metrics")


def _prepare_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true).flatten()
    p = np.asarray(y_pred).flatten()

    if t.size == 0 or p.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays.")
    if t.size != p.size:
        raise ValueError(
            f"y_true and y_pred must contain the same number of elements: "
            f"{t.size} != {p.size}."
        )

    return t, p


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _prepare_arrays(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _prepare_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    t, p = _prepare_arrays(y_true, y_pred)
    return float(np.mean(np.abs((t - p) / (np.abs(t) + eps))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = _prepare_arrays(y_true, y_pred)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> Dict[str, float]:
    """
    Вычисляет MAE, RMSE, MAPE и R² одним вызовом.

    Parameters
    ----------
    y_true, y_pred : np.ndarray  — фактические и предсказанные значения
    model_name : str             — имя для логирования

    Returns
    -------
    dict {"MAE": ..., "RMSE": ..., "MAPE": ..., "R2": ...}
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    if model_name:
        logger.info(
            "%-15s | MAE=%8.2f | RMSE=%8.2f | MAPE=%6.2f%% | R²=%6.4f",
            model_name,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["MAPE"],
            metrics["R2"],
        )
    return metrics
