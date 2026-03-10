# -*- coding: utf-8 -*-
"""
utils/visualization.py — Утилиты для построения графиков.
"""

import logging
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data.preprocessing import inverse_scale
from optimization.storage import StorageResult

logger = logging.getLogger("smart_grid.utils.visualization")

plt.style.use("seaborn-v0_8-darkgrid")


# ── Кривые обучения Keras ─────────────────────────────────────────────────────

def plot_training_history(
    history,
    model_name: str = "Model",
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """Рисует Loss / MAE / MAPE по эпохам."""
    os.makedirs(plots_dir, exist_ok=True)
    h = history.history
    metrics_to_plot = [k for k in ("loss", "mae", "mape") if k in h]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 4))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics_to_plot):
        ax.plot(h[m], label=f"Train {m.upper()}")
        if f"val_{m}" in h:
            ax.plot(h[f"val_{m}"], label=f"Val {m.upper()}")
        ax.set_title(f"{model_name} — {m.upper()}", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        fig.savefig(
            os.path.join(plots_dir, f"training_{model_name.replace(' ', '_')}.png"),
            dpi=150, bbox_inches="tight",
        )
    pass  # plt.show() убран: headless Agg backend
    plt.close(fig)


# ── Сравнение прогнозов ───────────────────────────────────────────────────────

def plot_predictions_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_steps: int = 168,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """Наносит факт и прогнозы нескольких моделей на один график."""
    os.makedirs(plots_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 5))
    t = np.arange(n_steps)
    ax.plot(t, y_true.flatten()[:n_steps], "k-", lw=2, label="Факт")
    colors = plt.cm.tab10.colors
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(t, pred.flatten()[:n_steps], lw=1.5,
                color=colors[i % 10], label=name, alpha=0.85)
    ax.set_title("Сравнение прогнозов моделей", fontweight="bold")
    ax.set_xlabel("Шаг прогноза")
    ax.set_ylabel("Потребление (кВт·ч)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(plots_dir, "predictions_comparison.png"),
                    dpi=150, bbox_inches="tight")
    pass  # plt.show() убран: headless Agg backend
    plt.close(fig)


# ── Сравнение метрик моделей ──────────────────────────────────────────────────

def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """Барчарт метрик MAE / RMSE / MAPE / R² по всем моделям."""
    os.makedirs(plots_dir, exist_ok=True)
    model_names = list(metrics.keys())
    metric_names = ["MAE", "RMSE", "MAPE", "R2"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Сравнение метрик моделей", fontsize=14, fontweight="bold")
    colors = sns.color_palette("husl", len(model_names))

    for ax, m in zip(axes, metric_names):
        values = [metrics[n].get(m, 0) for n in model_names]
        bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor="black")
        ax.set_title(m, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(plots_dir, "metrics_comparison.png"),
                    dpi=150, bbox_inches="tight")
    pass  # plt.show() убран: headless Agg backend
    plt.close(fig)


# ── Оптимизация накопителя ────────────────────────────────────────────────────

def plot_storage_result(
    result: StorageResult,
    forecast: np.ndarray,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """Четыре субграфика: цены, SOC, энергопотоки, затраты."""
    os.makedirs(plots_dir, exist_ok=True)
    hours = np.arange(len(forecast))
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    fig.suptitle("Оптимизация накопителя энергии", fontsize=14, fontweight="bold")

    # 1. Цены и действия
    ax = axes[0]
    ax.plot(hours, result.prices, lw=2, color="black", label="Цена руб/кВт·ч")
    for i, act in enumerate(result.actions):
        if act == "charge":
            ax.axvspan(i, i + 1, alpha=0.25, color="green")
        elif act == "discharge":
            ax.axvspan(i, i + 1, alpha=0.25, color="red")
    ax.set_title("Тарифные зоны и действия накопителя (зелёный=заряд, красный=разряд)")
    ax.set_ylabel("Руб/кВт·ч")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SOC
    ax = axes[1]
    soc_arr = np.array(result.battery_levels[:-1])
    ax.plot(hours, soc_arr, lw=2.5, color="purple", marker="o", ms=3)
    ax.fill_between(hours, 0, soc_arr, alpha=0.3, color="purple")
    ax.set_title("State of Charge накопителя")
    ax.set_ylabel("кВт·ч")
    ax.grid(True, alpha=0.3)

    # 3. Энергопотоки
    ax = axes[2]
    ax.plot(hours, forecast, lw=2, color="blue", label="Спрос", alpha=0.8)
    ax.plot(hours, result.energy_from_grid, lw=1.5, color="green",
            ls="--", label="Из сети (с накопителем)")
    ax.fill_between(hours, result.energy_from_grid, forecast,
                    where=(np.array(result.energy_from_grid) < forecast),
                    alpha=0.3, color="green", label="Разряд → экономия")
    ax.fill_between(hours, forecast, result.energy_from_grid,
                    where=(np.array(result.energy_from_grid) > forecast),
                    alpha=0.3, color="red", label="Зарядка (+нагрузка)")
    ax.set_title("Энергопотоки")
    ax.set_ylabel("кВт·ч")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Затраты
    ax = axes[3]
    baseline_costs = forecast * result.prices
    w = 0.4
    ax.bar(hours - w / 2, baseline_costs, w, alpha=0.6, label="Без накопителя", color="coral")
    ax.bar(hours + w / 2, result.hourly_costs, w, alpha=0.8, label="С накопителем", color="lightgreen")
    cumulative_net = np.cumsum(baseline_costs - np.array(result.hourly_costs))
    ax2 = ax.twinx()
    ax2.plot(hours, cumulative_net, color="darkgreen", lw=2.5, label="Накопленная экономия")
    ax2.set_ylabel("Накопленная экономия (руб)", color="darkgreen")
    ax.set_title(
        f"Затраты: чистая экономия = {result.net_savings:.1f} руб"
        f" ({result.net_savings_pct:.1f}%)"
    )
    ax.set_xlabel("Час")
    ax.set_ylabel("Руб")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(plots_dir, "storage_optimization.png"),
                    dpi=150, bbox_inches="tight")
    pass  # plt.show() убран: headless Agg backend
    plt.close(fig)
