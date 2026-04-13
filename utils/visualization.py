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
from utils.plot_style import apply_publication_style, get_palette, save_figure

logger = logging.getLogger("smart_grid.utils.visualization")
apply_publication_style()
PALETTE = get_palette()


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
        unit = "[%]" if m == "mape" else "[кВт·ч]" if m in ("loss", "mae") else "[-]"
        ax.set_title(f"{model_name} — {m.upper()} {unit}")
        ax.set_xlabel("Эпоха [-]")
        ax.set_ylabel(f"{m.upper()} {unit}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, os.path.join(plots_dir, f"training_{model_name.replace(' ', '_')}.png"), save=save)
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
    ax.plot(t, y_true.flatten()[:n_steps], color=PALETTE["baseline"], lw=2, label="Факт")
    colors = sns.color_palette("colorblind", max(10, len(predictions)))
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(t, pred.flatten()[:n_steps], lw=1.5,
                color=colors[i % 10], label=name, alpha=0.85)
    ax.set_title("Сравнение прогнозов моделей [кВт·ч]")
    ax.set_xlabel("Шаг прогноза [ч]")
    ax.set_ylabel("Потребление [кВт·ч]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "predictions_comparison.png"), save=save)
    plt.close(fig)


# ── Сравнение метрик моделей ──────────────────────────────────────────────────

def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """Барчарт метрик MAE / RMSE / sMAPE / R² по всем моделям."""
    os.makedirs(plots_dir, exist_ok=True)
    model_names = list(metrics.keys())
    metric_names = ["MAE", "RMSE", "sMAPE", "R2"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Сравнение метрик моделей")
    colors = sns.color_palette("colorblind", len(model_names))

    for ax, m in zip(axes, metric_names):
        values = [metrics[n].get(m, 0) for n in model_names]
        bars = ax.bar(model_names, values, color=colors, alpha=0.8, edgecolor="black")
        unit = "[%]" if m == "sMAPE" else "[-]" if m == "R2" else "[кВт·ч]"
        ax.set_title(f"{m} {unit}")
        ax.set_ylabel(unit)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "metrics_comparison.png"), save=save)
    plt.close(fig)


def plot_scientific_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """Графики для научного отчёта: факт-vs-прогноз и распределение ошибки."""
    os.makedirs(plots_dir, exist_ok=True)
    y_t = y_true.flatten()
    y_p = y_pred.flatten()
    err = y_p - y_t

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Scatter факт-прогноз
    axes[0].scatter(y_t, y_p, s=4, alpha=0.30, color=PALETTE["primary"])
    diag_min = min(y_t.min(), y_p.min())
    diag_max = max(y_t.max(), y_p.max())
    axes[0].plot([diag_min, diag_max], [diag_min, diag_max], "--", color=PALETTE["negative"], lw=1.5)
    axes[0].set_title(f"{model_name}: факт vs прогноз")
    axes[0].set_xlabel("Факт [кВт·ч]")
    axes[0].set_ylabel("Прогноз [кВт·ч]")
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    axes[1].hist(err, bins=60, color=PALETTE["highlight"], alpha=0.75, edgecolor=PALETTE["baseline"])
    axes[1].axvline(np.mean(err), color=PALETTE["negative"], linestyle="--", lw=1.5, label="MBE")
    axes[1].set_title(f"{model_name}: распределение ошибки")
    axes[1].set_xlabel("Ошибка прогноза [кВт·ч]")
    axes[1].set_ylabel("Частота [-]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, os.path.join(plots_dir, f"scientific_diagnostics_{model_name}.png"), save=save)
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
    fig.suptitle("Оптимизация накопителя энергии")

    # 1. Цены и действия
    ax = axes[0]
    ax.plot(hours, result.prices, lw=2, color=PALETTE["baseline"], label="Цена [руб/кВт·ч]")
    for i, act in enumerate(result.actions):
        if act == "charge":
            ax.axvspan(i, i + 1, alpha=0.25, color=PALETTE["positive"])
        elif act == "discharge":
            ax.axvspan(i, i + 1, alpha=0.25, color=PALETTE["negative"])
    ax.set_title("Тарифные зоны и действия накопителя (зелёный=заряд, красный=разряд)")
    ax.set_xlabel("Час [ч]")
    ax.set_ylabel("Тариф [руб/кВт·ч]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SOC
    ax = axes[1]
    soc_arr = np.array(result.battery_levels[:-1])
    ax.plot(hours, soc_arr, lw=2.5, color=PALETTE["highlight"], marker="o", ms=3)
    ax.fill_between(hours, 0, soc_arr, alpha=0.3, color=PALETTE["highlight"])
    ax.set_title("State of Charge накопителя")
    ax.set_xlabel("Час [ч]")
    ax.set_ylabel("SOC [кВт·ч]")
    ax.grid(True, alpha=0.3)

    # 3. Энергопотоки
    ax = axes[2]
    ax.plot(hours, forecast, lw=2, color=PALETTE["primary"], label="Спрос [кВт·ч]", alpha=0.8)
    ax.plot(hours, result.energy_from_grid, lw=1.5, color=PALETTE["secondary"],
            ls="--", label="Из сети (с накопителем)")
    ax.fill_between(hours, result.energy_from_grid, forecast,
                    where=(np.array(result.energy_from_grid) < forecast),
                    alpha=0.3, color=PALETTE["positive"], label="Разряд → экономия")
    ax.fill_between(hours, forecast, result.energy_from_grid,
                    where=(np.array(result.energy_from_grid) > forecast),
                    alpha=0.3, color=PALETTE["negative"], label="Зарядка (+нагрузка)")
    ax.set_title("Энергопотоки")
    ax.set_xlabel("Час [ч]")
    ax.set_ylabel("Энергия [кВт·ч]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Затраты
    ax = axes[3]
    baseline_costs = forecast * result.prices
    w = 0.4
    ax.bar(hours - w / 2, baseline_costs, w, alpha=0.6, label="Без накопителя", color=PALETTE["accent"])
    ax.bar(hours + w / 2, result.hourly_costs, w, alpha=0.8, label="С накопителем", color=PALETTE["secondary"])
    cumulative_net = np.cumsum(baseline_costs - np.array(result.hourly_costs))
    ax2 = ax.twinx()
    ax2.plot(hours, cumulative_net, color=PALETTE["primary"], lw=2.5, label="Накопленная экономия")
    ax2.set_ylabel("Накопленная экономия [руб]", color=PALETTE["primary"])
    ax.set_title(
        f"Затраты: чистая экономия = {result.net_savings:.1f} руб"
        f" ({result.net_savings_pct:.1f}%)"
    )
    ax.set_xlabel("Час [ч]")
    ax.set_ylabel("Затраты [руб]")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, os.path.join(plots_dir, "storage_optimization.png"), save=save)
    plt.close(fig)
