# -*- coding: utf-8 -*-
"""
analysis/backtesting.py — Проверка устойчивости прогноза во времени.

Здесь две РАЗНЫЕ процедуры, которые нельзя путать:

1. `analyze_stability_over_windows` — разбивает тестовую выборку на несколько
   последовательных окон и считает метрики уже обученной модели в каждом.
   Это анализ УСТОЙЧИВОСТИ: показывает, как качество плавает от периода к
   периоду (например, растёт зимой). Переобучения нет, новых данных модель не
   видит. Бэктестингом в общепринятом смысле это не является.

2. `run_rolling_origin_backtest` — настоящий walk-forward: для каждой точки
   отсечения модель обучается ЗАНОВО только на данных до этой точки и
   оценивается на следующем блоке. Так воспроизводится реальная эксплуатация,
   где модель периодически переобучают на накопленной истории. Процедура
   дорогая (N переобучений), поэтому применяется к одной-двум лучшим моделям.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utils.metrics import compute_all_metrics
from data.preprocessing import inverse_scale, prepare_data

logger = logging.getLogger("smart_grid.analysis.backtesting")

from utils.visualization import save_figure


# ══════════════════════════════════════════════════════════════════════════════
# 1. УСТОЙЧИВОСТЬ ПО ОКНАМ ТЕСТА (без переобучения)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stability_over_windows(
    model: Any,
    data: Dict[str, Any],
    n_windows: int = 8,
    plots_dir: str = "results/plots",
    model_name: str = "Model",
    save: bool = True,
) -> Dict[str, List[float]]:
    """
    Метрики обученной модели на последовательных непересекающихся окнах теста.

    Returns dict {"MAE": [...], "RMSE": [...], "MAPE": [...], "R2": [...]}
    """
    os.makedirs(plots_dir, exist_ok=True)

    X_test = data["X_test"]
    Y_test = data["Y_test"]
    scaler = data["scaler"]
    total = len(X_test)

    if total < n_windows * 20:
        n_windows = max(1, min(n_windows, total // 20))
        logger.warning("Размер теста мал → число окон снижено до %d", n_windows)

    window_size = max(1, total // n_windows)
    metrics_history: Dict[str, List[float]] = {"MAE": [], "RMSE": [], "MAPE": [], "R2": []}

    logger.info("Анализ устойчивости %s: %d окон по ~%d шагов (без переобучения)",
                model_name, n_windows, window_size)

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else total
        X_w, Y_w = X_test[start:end], Y_test[start:end]
        if len(X_w) == 0:
            continue

        try:
            if isinstance(model, tf.keras.Model):
                Y_pred_s = model.predict(X_w, verbose=0)
            else:
                Y_pred_s = model.predict(X_w)

            m = compute_all_metrics(inverse_scale(scaler, Y_w),
                                    inverse_scale(scaler, Y_pred_s))
            for k in metrics_history:
                metrics_history[k].append(m[k])
            logger.info("  Окно %2d/%d | MAE=%.2f MAPE=%.2f%%",
                        w + 1, n_windows, m["MAE"], m["MAPE"])
        except Exception as exc:
            logger.error("Ошибка в окне %d: %s", w, exc)

    if metrics_history["MAE"]:
        _plot_window_metrics(
            metrics_history, model_name, plots_dir, save,
            title=f"Устойчивость по окнам теста — {model_name}",
            xlabel="Окно теста", fname=f"stability_{model_name.replace(' ', '_')}.png",
        )
        logger.info(
            "Устойчивость %s: MAE=%.2f±%.2f  MAPE=%.2f±%.2f%%  (разброс %.1f%% от среднего)",
            model_name,
            np.mean(metrics_history["MAE"]), np.std(metrics_history["MAE"]),
            np.mean(metrics_history["MAPE"]), np.std(metrics_history["MAPE"]),
            100 * np.std(metrics_history["MAE"]) / max(np.mean(metrics_history["MAE"]), 1e-9),
        )
    return metrics_history


# ══════════════════════════════════════════════════════════════════════════════
# 2. НАСТОЯЩИЙ ROLLING-ORIGIN БЭКТЕСТИНГ (с переобучением)
# ══════════════════════════════════════════════════════════════════════════════

def run_rolling_origin_backtest(
    build_fn: Callable[[], Any],
    df: pd.DataFrame,
    model_name: str = "Model",
    n_origins: int = 4,
    history_length: int = 48,
    forecast_horizon: int = 24,
    initial_train_share: float = 0.55,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Walk-forward с расширяющимся окном обучения.

    Схема (n_origins=3):
        origin 1: train [0 .. 55%]  → test (55% .. 70%]
        origin 2: train [0 .. 70%]  → test (70% .. 85%]
        origin 3: train [0 .. 85%]  → test (85% .. 100%]

    На каждом шаге модель создаётся ЗАНОВО через build_fn() и обучается только
    на доступной к этому моменту истории — заглянуть вперёд невозможно.
    Разброс метрик между origin-ами показывает, насколько результат на одном
    фиксированном тестовом периоде вообще воспроизводим.

    Parameters
    ----------
    build_fn : callable
        Фабрика НОВОЙ необученной модели (без аргументов).
    df : pd.DataFrame
        Полный датасет (тот же, что подаётся в prepare_data).
    initial_train_share : float
        Доля данных, доступная для обучения на первом origin.

    Returns
    -------
    {"origins": [...], "MAE": [...], "RMSE": [...], "MAPE": [...], "R2": [...],
     "mean_MAE": float, "std_MAE": float}
    """
    from models.trainer import ModelTrainer   # локальный импорт: избегаем цикла

    os.makedirs(plots_dir, exist_ok=True)
    total = len(df)
    step = (1.0 - initial_train_share) / n_origins

    results: Dict[str, Any] = {"origins": [], "MAE": [], "RMSE": [],
                               "MAPE": [], "R2": [], "MASE": []}

    logger.info("=" * 70)
    logger.info("ROLLING-ORIGIN БЭКТЕСТИНГ: %s | %d origin-ов, переобучение на каждом",
                model_name, n_origins)
    logger.info("=" * 70)

    # Каждый из трёх блоков (train, val, test) должен вмещать хотя бы одно окно.
    min_block = history_length + forecast_horizon + 1

    for k in range(n_origins):
        train_share = initial_train_share + k * step
        test_share = train_share + step
        end_idx = int(total * min(test_share, 1.0))
        df_k = df.iloc[:end_idx].copy()

        # Внутри среза: доступная история делится на train и validation,
        # последний блок — тест, который модель не видела.
        avail = int(total * train_share)
        val_rows = max(int(0.10 * avail), min_block)
        train_rows = avail - val_rows
        test_rows = end_idx - avail

        if train_rows < min_block or test_rows < min_block:
            logger.warning(
                "Origin %d/%d пропущен: блоки слишком малы для окна %d+%d "
                "(train=%d, val=%d, test=%d строк). Увеличьте объём данных "
                "или уменьшите число origin-ов.",
                k + 1, n_origins, history_length, forecast_horizon,
                train_rows, val_rows, test_rows,
            )
            continue

        train_ratio = train_rows / end_idx
        val_ratio = val_rows / end_idx

        try:
            data_k = prepare_data(
                df_k, history_length=history_length,
                forecast_horizon=forecast_horizon,
                train_ratio=train_ratio, val_ratio=val_ratio,
            )
            logger.info(
                "Origin %d/%d | обучение до часа %d (%.0f%%) | тест %d окон",
                k + 1, n_origins, avail, 100 * train_share, len(data_k["X_test"]),
            )

            trainer = ModelTrainer(build_fn(), f"{model_name}_origin{k+1}",
                                   plots_dir=plots_dir)
            trainer.train(data_k, epochs=epochs, batch_size=batch_size,
                          patience=patience)
            m = trainer.evaluate(data_k, split="test", run_residual_diagnostics=False)

            results["origins"].append(k + 1)
            for key in ("MAE", "RMSE", "MAPE", "R2"):
                results[key].append(m[key])
            results["MASE"].append(m.get("MASE", float("nan")))

            logger.info("Origin %d: MAE=%.2f  MAPE=%.2f%%  R²=%.4f  MASE=%s",
                        k + 1, m["MAE"], m["MAPE"], m["R2"],
                        f"{m['MASE']:.3f}" if "MASE" in m else "н/д")
        except Exception as exc:
            logger.error("Origin %d провален: %s", k + 1, exc)

    if results["MAE"]:
        results["mean_MAE"] = float(np.mean(results["MAE"]))
        results["std_MAE"] = float(np.std(results["MAE"]))
        logger.info("─" * 70)
        logger.info(
            "ИТОГ rolling-origin %s: MAE=%.2f±%.2f | MAPE=%.2f±%.2f%% | R²=%.4f±%.4f",
            model_name,
            results["mean_MAE"], results["std_MAE"],
            np.mean(results["MAPE"]), np.std(results["MAPE"]),
            np.mean(results["R2"]), np.std(results["R2"]),
        )
        _plot_window_metrics(
            results, model_name, plots_dir, save,
            title=f"Rolling-origin бэктестинг — {model_name}",
            xlabel="Origin (точка отсечения)",
            fname=f"backtesting_{model_name.replace(' ', '_')}.png",
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ОБЩИЙ ГРАФИК
# ══════════════════════════════════════════════════════════════════════════════

def _plot_window_metrics(
    metrics: Dict[str, List[float]],
    model_name: str,
    plots_dir: str,
    save: bool,
    title: str,
    xlabel: str,
    fname: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    xs = list(range(1, len(metrics["MAE"]) + 1))

    axes[0].plot(xs, metrics["MAE"], "o-", label="MAE")
    axes[0].plot(xs, metrics["RMSE"], "s-", label="RMSE")
    axes[0].axhline(float(np.mean(metrics["MAE"])), color="gray", ls="--", lw=1,
                    label=f"Средний MAE = {np.mean(metrics['MAE']):.0f}")
    axes[0].set_title("MAE / RMSE")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("кВт·ч")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, metrics["MAPE"], "^-", color="orange")
    axes[1].set_title("MAPE")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("MAPE (%)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(plots_dir, fname)
        save_figure(fig, path, dpi=150)
        logger.info("График сохранён: %s", path)
    plt.close(fig)


# Обратная совместимость со старым именем (использовалось в main.py).
def run_backtesting(
    model: Any,
    data: Dict[str, Any],
    n_windows: int = 8,
    plots_dir: str = "results/plots",
    model_name: str = "Model",
    save: bool = True,
) -> Dict[str, List[float]]:
    """Устаревшее имя. Используйте analyze_stability_over_windows()."""
    logger.warning(
        "run_backtesting() переименована в analyze_stability_over_windows(): "
        "процедура не переобучает модель и бэктестингом не является."
    )
    return analyze_stability_over_windows(
        model, data, n_windows=n_windows, plots_dir=plots_dir,
        model_name=model_name, save=save,
    )
