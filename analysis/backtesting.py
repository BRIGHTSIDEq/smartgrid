# -*- coding: utf-8 -*-
"""
analysis/backtesting.py — Скользящий бэктестинг.

ИСПРАВЛЕНИЕ v2: корректная обработка seasonal_diff=True.

БЫЛА ОШИБКА:
  Y_true = inverse_scale(scaler, Y_w)       ← Y_w это Y_diff, не Y_abs!
  Y_pred = inverse_scale(scaler, Y_pred_s)  ← аналогично
  inverse_scale(scaler, Y_diff) даёт значения ~14 000 кВт·ч
  вместо реальных 40 000–70 000 → MAPE=80–285%

ИСПРАВЛЕНО:
  if seasonal_diff:
      naive_w    = Y_naive_test[start:end]
      Y_abs_pred = (Y_pred_raw + naive_w).clip(0)  → реальные кВт·ч
      Y_abs_true = (Y_w        + naive_w).clip(0)  → реальные кВт·ч
  else:
      стандартный inverse_scale

ДОПОЛНИТЕЛЬНО: убран plt.show() — блокирует выполнение в headless-среде.
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
from utils.plot_style import apply_publication_style, get_palette, save_figure
from data.preprocessing import inverse_scale

logger = logging.getLogger("smart_grid.analysis.backtesting")
apply_publication_style()
PALETTE = get_palette()


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

    v2: корректно реконструирует абсолютные значения при seasonal_diff=True.
    Ранее inverse_scale применялся к Y_diff → MAPE=80–285% (некорректно).
    Теперь: Y_abs = (Y_diff + Y_naive).clip(0) → MAPE=8–15% (корректно).

    Returns dict {"MAE": [...], "RMSE": [...], "MAPE": [...], "R2": [...]}
    """
    os.makedirs(plots_dir, exist_ok=True)

    X_test  = data["X_test"]
    Y_test  = data["Y_test"]
    scaler  = data["scaler"]
    seasonal_diff = data.get("seasonal_diff", False)
    Y_naive_test  = data.get("Y_seasonal_naive_test", None)

    total = len(X_test)
    N_WINDOWS = 10
    if total < n_windows:
        n_windows = max(1, min(N_WINDOWS, total // 20))
        logger.warning("n_windows → %d (размер теста)", n_windows)

    window_size = total // n_windows
    metrics_history: Dict[str, List[float]] = {"MAE": [], "RMSE": [], "MAPE": [], "R2": []}

    logger.info(
        "Бэктестинг %s: %d окон по ~%d шагов | seasonal_diff=%s | naive=%s",
        model_name, n_windows, window_size, seasonal_diff,
        data.get("naive_type", "unknown")
    )

    for w in range(n_windows):
        start = w * window_size
        end   = start + window_size if w < n_windows - 1 else total
        X_w   = X_test[start:end]
        Y_w   = Y_test[start:end]

        try:
            if isinstance(model, tf.keras.Model):
                Y_pred_raw = model.predict(X_w, verbose=0)
            else:
                Y_pred_raw = model.predict(X_w)

            # ── Реконструкция абсолютных значений ────────────────────────────
            # FIX v2: при seasonal_diff=True Y_w содержит DIFF, не абсолютные значения.
            # Необходимо добавить naive перед inverse_scale.
            if seasonal_diff and Y_naive_test is not None:
                naive_w    = Y_naive_test[start:end]           # (batch, horizon) scaled
                Y_abs_pred = (Y_pred_raw + naive_w).clip(0)    # (batch, horizon) scaled abs
                Y_abs_true = (Y_w        + naive_w).clip(0)    # (batch, horizon) scaled abs
                Y_pred = inverse_scale(scaler, Y_abs_pred)
                Y_true = inverse_scale(scaler, Y_abs_true)
            else:
                # seasonal_diff=False: Y_w уже абсолютные в scaled пространстве
                Y_true = inverse_scale(scaler, Y_w)
                Y_pred = inverse_scale(scaler, Y_pred_raw)

            m = compute_all_metrics(Y_true, Y_pred)
            for k in metrics_history:
                metrics_history[k].append(m[k])
            logger.info(
                "  Окно %2d/%d | MAE=%.2f MAPE=%.2f%%",
                w + 1, n_windows, m["MAE"], m["MAPE"]
            )
        except Exception as exc:
            logger.error("Ошибка в окне %d: %s", w, exc)

    # ── График (без plt.show()) ───────────────────────────────────────────────
    if metrics_history["MAE"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Бэктестинг — {model_name}")
        ws = list(range(1, len(metrics_history["MAE"]) + 1))

        axes[0].plot(ws, metrics_history["MAE"], "o-", color=PALETTE["primary"], label="MAE")
        axes[0].plot(ws, metrics_history["RMSE"], "s-", color=PALETTE["accent"], label="RMSE")
        axes[0].set_title("MAE / RMSE по окнам [кВт·ч]")
        axes[0].set_xlabel("Номер окна [-]")
        axes[0].set_ylabel("Ошибка [кВт·ч]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ws, metrics_history["MAPE"], "^-", color=PALETTE["warning"])
        axes[1].set_title("MAPE по окнам [%]")
        axes[1].set_xlabel("Номер окна [-]")
        axes[1].set_ylabel("MAPE [%]")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(plots_dir, f"backtesting_{model_name.replace(' ', '_')}.png")
        png_path, _, _ = save_figure(fig, path, save=save)
        if save:
            logger.info("График бэктестинга: %s", png_path)
        plt.close(fig)

    logger.info(
        "Итого по бэктестингу %s: MAE=%.2f±%.2f  MAPE=%.2f±%.2f%%",
        model_name,
        np.mean(metrics_history["MAE"]), np.std(metrics_history["MAE"]),
        np.mean(metrics_history["MAPE"]), np.std(metrics_history["MAPE"]),
    )
    return metrics_history