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


def save_figure(fig, path: str, dpi: int = 150, attempts: int = 3) -> bool:
    """
    Сохраняет график, переживая кратковременную блокировку файла.

    На Windows только что созданный PNG может быть на доли секунды захвачен
    антивирусом или службой индексации, и запись падает с OSError. Для
    двухчасового прогона это означало бы потерю результатов на шаге построения
    графиков, поэтому запись повторяется, а окончательная неудача не прерывает
    пайплайн: график вторичен по отношению к метрикам.

    Returns
    -------
    bool — удалось ли сохранить файл.
    """
    import time

    for attempt in range(1, attempts + 1):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            return True
        except OSError as exc:
            if attempt == attempts:
                logger.error(
                    "График не сохранён после %d попыток: %s (%s). "
                    "Расчёт продолжается — метрики не затронуты.",
                    attempts, path, exc,
                )
                return False
            logger.warning("Не удалось сохранить %s (попытка %d/%d): %s",
                           path, attempt, attempts, exc)
            time.sleep(0.5 * attempt)
    return False


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
        save_figure(fig, os.path.join(plots_dir, f"training_{model_name.replace(' ', '_')}.png"), dpi=150)
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
        save_figure(fig, os.path.join(plots_dir, "predictions_comparison.png"), dpi=150)
    plt.close(fig)


# ── Сравнение метрик моделей ──────────────────────────────────────────────────

def plot_metrics_comparison(
    metrics: Dict[str, Dict[str, float]],
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Барчарт метрик по всем моделям, отсортированный по MAE.

    На панели MASE проводится красная линия на уровне 1.0 — граница
    практической ценности: столбцы выше неё означают, что модель не превзошла
    сезонно-наивный прогноз.
    """
    os.makedirs(plots_dir, exist_ok=True)
    model_names = sorted(metrics.keys(), key=lambda n: metrics[n].get("MAE", 9e18))
    has_mase = any("MASE" in m for m in metrics.values())
    metric_names = ["MAE", "RMSE", "MAPE", "R2"] + (["MASE"] if has_mase else [])

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    fig.suptitle("Сравнение метрик моделей (тестовая выборка)",
                 fontsize=14, fontweight="bold")
    colors = sns.color_palette("husl", len(model_names))

    for ax, m in zip(np.atleast_1d(axes), metric_names):
        values = [metrics[n].get(m, 0) for n in model_names]
        bars = ax.bar(model_names, values, color=colors, alpha=0.85, edgecolor="black")
        ax.set_title(m, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        if m == "MASE":
            ax.axhline(1.0, color="red", ls="--", lw=2,
                       label="Сезонно-наивный прогноз")
            ax.legend(fontsize=8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save:
        save_figure(fig, os.path.join(plots_dir, "metrics_comparison.png"), dpi=150)
    plt.close(fig)


# ── Деградация прогноза по шагам горизонта ───────────────────────────────────

def plot_metrics_by_horizon(
    per_horizon: Dict[str, Dict[str, List[float]]],
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Кривые «шаг горизонта → ошибка» для всех моделей.

    Усреднённый MAE скрывает главное: прогноз на 1 час вперёд и на 24 часа —
    задачи разной сложности. Расхождение кривых показывает, какая модель
    выигрывает именно на дальних шагах.

    Parameters
    ----------
    per_horizon : {имя модели: результат utils.metrics.metrics_by_horizon}
    """
    os.makedirs(plots_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Деградация прогноза с ростом горизонта",
                 fontsize=14, fontweight="bold")
    colors = plt.cm.tab10.colors

    for i, (name, m) in enumerate(per_horizon.items()):
        axes[0].plot(m["h"], m["MAE"], marker="o", ms=3, lw=1.6,
                     color=colors[i % 10], label=name)
        axes[1].plot(m["h"], m["R2"], marker="s", ms=3, lw=1.6,
                     color=colors[i % 10], label=name)

    axes[0].set_title("MAE по шагам горизонта")
    axes[0].set_xlabel("Шаг прогноза, ч вперёд")
    axes[0].set_ylabel("MAE, кВт·ч")
    axes[1].set_title("R² по шагам горизонта")
    axes[1].set_xlabel("Шаг прогноза, ч вперёд")
    axes[1].set_ylabel("R²")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        save_figure(fig, os.path.join(plots_dir, "metrics_by_horizon.png"), dpi=150)
    plt.close(fig)


# ── Точность против вычислительной стоимости ─────────────────────────────────

def plot_accuracy_vs_cost(
    metrics: Dict[str, Dict[str, float]],
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Диаграмма «время обучения → MAE» в логарифмическом масштабе по оси X.

    Отвечает на практический вопрос работы: оправдывают ли тяжёлые архитектуры
    свою вычислительную стоимость. Точки левее и ниже — лучше.
    """
    os.makedirs(plots_dir, exist_ok=True)
    names = [n for n in metrics if "train_time_sec" in metrics[n]]
    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, n in enumerate(names):
        t = max(float(metrics[n]["train_time_sec"]), 0.1)
        mae = float(metrics[n]["MAE"])
        params = int(metrics[n].get("n_params", 0))
        size = 80 + (params / 20000.0 if params else 0)
        ax.scatter(t, mae, s=min(size, 900), color=colors[i % 10],
                   alpha=0.75, edgecolor="black", zorder=3)
        ax.annotate(n, (t, mae), textcoords="offset points", xytext=(8, 6),
                    fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Время обучения, сек (лог. шкала)")
    ax.set_ylabel("MAE на тесте, кВт·ч")
    ax.set_title("Точность против вычислительной стоимости\n"
                 "(размер точки ∝ числу параметров)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        save_figure(fig, os.path.join(plots_dir, "accuracy_vs_cost.png"), dpi=150)
    plt.close(fig)


# ── Цена ошибки прогноза для накопителя ──────────────────────────────────────

def plot_forecast_value(
    storage_results: Dict[str, "StorageResult"],
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Столбчатая диаграмма чистой экономии накопителя по источникам прогноза.

    Верхняя пунктирная линия — результат при идеальном прогнозе. Расстояние до
    неё и есть денежная стоимость ошибки конкретной модели.
    """
    os.makedirs(plots_dir, exist_ok=True)
    names = sorted(storage_results, key=lambda k: -storage_results[k].net_savings)
    values = [storage_results[n].net_savings for n in names]
    oracle = storage_results.get("Идеальный прогноз")
    oracle_val = oracle.net_savings if oracle is not None else max(values)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#2e7d32" if n == "Идеальный прогноз" else "#1976d2" for n in names]
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="black")

    ax.axhline(oracle_val, color="red", ls="--", lw=2,
               label=f"Идеальный прогноз = {oracle_val:,.0f} руб".replace(",", " "))
    for bar, v in zip(bars, values):
        shortfall = oracle_val - v
        label = f"{v:,.0f}".replace(",", " ")
        if shortfall > 1:
            label += f"\n(−{shortfall:,.0f})".replace(",", " ")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Чистая экономия за горизонт, руб")
    ax.set_title("Экономический эффект накопителя в зависимости от источника прогноза\n"
                 "(стратегия срезки пика)", fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        save_figure(fig, os.path.join(plots_dir, "forecast_value.png"), dpi=150)
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
        save_figure(fig, os.path.join(plots_dir, "storage_optimization.png"), dpi=150)
    plt.close(fig)
