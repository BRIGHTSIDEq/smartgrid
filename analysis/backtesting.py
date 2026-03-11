# -*- coding: utf-8 -*-
"""
analysis/backtesting.py — Скользящий бэктестинг.

ИСПРАВЛЕНИЕ: убран plt.show() — блокирует выполнение в headless-среде.
Используем matplotlib Agg backend (задан в trainer.py), только savefig().
"""

import logging
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils.metrics import compute_all_metrics
from data.preprocessing import inverse_scale

logger = logging.getLogger("smart_grid.analysis.backtesting")


def run_backtesting(
    model: Any,
    data: Dict[str, Any],
    n_windows: int = 8,
    plots_dir: str = "results/plots",
    model_name: str = "Model",
    save: bool = True,
) -> Dict[str, List[float]]:
    """
    Скользящий бэктестинг на непересекающихся окнах тестовой выборки.

    Returns dict {"MAE": [...], "RMSE": [...], "MAPE": [...], "R2": [...]}
    """
    os.makedirs(plots_dir, exist_ok=True)

    X_test = data["X_test"]
    Y_test = data["Y_test"]
    scaler = data["scaler"]
    total = len(X_test)
    N_WINDOWS = 10 
    if total < n_windows:
        n_windows = max(1, min(N_WINDOWS, total // 20))
        logger.warning("n_windows → %d (размер теста)", n_windows)

    window_size = total // n_windows
    metrics_history: Dict[str, List[float]] = {"MAE": [], "RMSE": [], "MAPE": [], "R2": []}

    logger.info("Бэктестинг %s: %d окон по ~%d шагов", model_name, n_windows, window_size)

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else total
        X_w, Y_w = X_test[start:end], Y_test[start:end]

        try:
            if isinstance(model, tf.keras.Model):
                Y_pred_s = model.predict(X_w, verbose=0)
            else:
                Y_pred_s = model.predict(X_w)

            Y_true = inverse_scale(scaler, Y_w)
            Y_pred = inverse_scale(scaler, Y_pred_s)
            m = compute_all_metrics(Y_true, Y_pred)
            for k in metrics_history:
                metrics_history[k].append(m[k])
            logger.info("  Окно %2d/%d | MAE=%.2f MAPE=%.2f%%",
                        w + 1, n_windows, m["MAE"], m["MAPE"])
        except Exception as exc:
            logger.error("Ошибка в окне %d: %s", w, exc)

    # ── График (без plt.show()) ───────────────────────────────────────────────
    if metrics_history["MAE"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Бэктестинг — {model_name}", fontsize=13, fontweight="bold")
        ws = list(range(1, len(metrics_history["MAE"]) + 1))

        axes[0].plot(ws, metrics_history["MAE"], "o-", label="MAE")
        axes[0].plot(ws, metrics_history["RMSE"], "s-", label="RMSE")
        axes[0].set_title("MAE / RMSE по окнам")
        axes[0].set_xlabel("Окно")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ws, metrics_history["MAPE"], "^-", color="orange")
        axes[1].set_title("MAPE по окнам")
        axes[1].set_xlabel("Окно")
        axes[1].set_ylabel("MAPE (%)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            path = os.path.join(plots_dir, f"backtesting_{model_name.replace(' ', '_')}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("График бэктестинга: %s", path)
        plt.close(fig)

    logger.info(
        "Итого по бэктестингу %s: MAE=%.2f±%.2f  MAPE=%.2f±%.2f%%",
        model_name,
        np.mean(metrics_history["MAE"]), np.std(metrics_history["MAE"]),
        np.mean(metrics_history["MAPE"]), np.std(metrics_history["MAPE"]),
    )
    return metrics_history
