# -*- coding: utf-8 -*-
"""
analysis/eda.py — Исследовательский анализ данных (EDA).
"""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.plot_style import apply_publication_style, get_palette, save_figure

logger = logging.getLogger("smart_grid.analysis.eda")
apply_publication_style()
PALETTE = get_palette()


def run_eda(
    df: pd.DataFrame,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Выполняет полный EDA временного ряда потребления.

    Графики:
    1. Временные паттерны (месяц / неделя / сутки)
    2. Профили потребления (суточный, недельный, тепловая карта, гистограмма)
    3. Декомпозиция: тренд + сезонность + остатки
    4. ACF / PACF
    5. Зависимость от температуры
    """
    os.makedirs(plots_dir, exist_ok=True)
    logger.info("Запуск EDA...")
    c = df["consumption"].values

    # ── Описательная статистика ───────────────────────────────────────────────
    logger.info("\n%s", df["consumption"].describe().to_string())
    logger.info(
        "CV=%.3f  Skew=%.3f  Kurt=%.3f",
        df["consumption"].std() / df["consumption"].mean(),
        df["consumption"].skew(),
        df["consumption"].kurtosis(),
    )

    # ── 1. Временные паттерны ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(df["timestamp"][: 24 * 30], c[: 24 * 30], color=PALETTE["primary"])
    axes[0].set_title("Потребление электроэнергии: первый месяц [кВт·ч]")
    axes[0].set_xlabel("Время [timestamp]")
    axes[0].set_ylabel("Потребление [кВт·ч]")
    axes[1].plot(df["timestamp"][: 24 * 7], c[: 24 * 7], color=PALETTE["warning"])
    axes[1].set_title("Недельный паттерн потребления [кВт·ч]")
    axes[1].set_xlabel("Время [timestamp]")
    axes[1].set_ylabel("Потребление [кВт·ч]")
    axes[2].plot(df["timestamp"][:24], c[:24], color=PALETTE["secondary"], marker="o")
    axes[2].set_title("Суточный паттерн потребления [кВт·ч]")
    axes[2].set_xlabel("Время [timestamp]")
    axes[2].set_ylabel("Потребление [кВт·ч]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, plots_dir, "01_timeseries_patterns.png", save)

    # ── 2. Профили ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hp = df.groupby("hour")["consumption"].mean()
    axes[0, 0].plot(hp.index, hp.values, marker="o", lw=2)
    axes[0, 0].fill_between(hp.index, hp.values, alpha=0.3)
    axes[0, 0].set_title("Суточный профиль (среднее) [кВт·ч]")
    axes[0, 0].set_xlabel("Час суток [ч]")
    axes[0, 0].set_ylabel("Потребление [кВт·ч]")
    axes[0, 0].grid(True, alpha=0.3)

    wp = df.groupby("weekday")["consumption"].mean()
    day_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    axes[0, 1].bar(range(7), wp.values, color=PALETTE["primary"], alpha=0.85)
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(day_names)
    axes[0, 1].set_title("Недельный профиль (среднее) [кВт·ч]")
    axes[0, 1].set_xlabel("День недели")
    axes[0, 1].set_ylabel("Потребление [кВт·ч]")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    pivot = df.pivot_table(values="consumption", index="hour", columns="weekday", aggfunc="mean")
    sns.heatmap(pivot, cmap="viridis", ax=axes[1, 0], cbar_kws={"label": "Потребление [кВт·ч]"})
    axes[1, 0].set_title("Тепловая карта: час × день недели [кВт·ч]")
    axes[1, 0].set_xlabel("День недели")
    axes[1, 0].set_ylabel("Час суток [ч]")

    axes[1, 1].hist(c, bins=60, color=PALETTE["highlight"], alpha=0.7, edgecolor=PALETTE["baseline"])
    axes[1, 1].axvline(np.mean(c), color=PALETTE["negative"], ls="--", lw=1.5, label="Среднее")
    axes[1, 1].axvline(np.median(c), color=PALETTE["positive"], ls="--", lw=1.5, label="Медиана")
    axes[1, 1].set_title("Распределение потребления [кВт·ч]")
    axes[1, 1].set_xlabel("Потребление [кВт·ч]")
    axes[1, 1].set_ylabel("Плотность [1/кВт·ч]")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, plots_dir, "02_consumption_profiles.png", save)

    # ── 3. Декомпозиция ───────────────────────────────────────────────────────
    logger.info("Декомпозиция временного ряда...")
    n_decomp = min(len(c), 24 * 365)
    try:
        decomp = seasonal_decompose(c[:n_decomp], model="additive", period=24)
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        for ax, series, title in zip(
            axes,
            [decomp.observed, decomp.trend, decomp.seasonal[: 24 * 7], decomp.resid],
            ["Исходный ряд", "Тренд", "Сезонность (первая неделя)", "Остатки"],
        ):
            ax.plot(series, lw=0.8)
            ax.set_title(f"{title} [кВт·ч]")
            ax.set_xlabel("Временной индекс [ч]")
            ax.set_ylabel("Амплитуда [кВт·ч]")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, plots_dir, "03_decomposition.png", save)
    except Exception as exc:
        logger.warning("Декомпозиция не удалась: %s", exc)

    # ── 4. ACF / PACF ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    n_acf = min(len(c), 24 * 60)
    plot_acf(c[:n_acf], lags=200, ax=axes[0])
    axes[0].set_title("ACF: автокорреляционная функция")
    axes[0].set_xlabel("Лаг [ч]")
    axes[0].set_ylabel("ACF [-]")
    plot_pacf(c[:n_acf], lags=50, ax=axes[1])
    axes[1].set_title("PACF: частичная автокорреляция")
    axes[1].set_xlabel("Лаг [ч]")
    axes[1].set_ylabel("PACF [-]")
    plt.tight_layout()
    _save(fig, plots_dir, "04_acf_pacf.png", save)

    # ── 5. Температурная зависимость ──────────────────────────────────────────
    if "temperature" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        hb = ax.hexbin(
            df["temperature"], c,
            gridsize=40, cmap="viridis", mincnt=1,
        )
        plt.colorbar(hb, label="Плотность наблюдений [-]")
        temp_bins = pd.cut(df["temperature"], bins=36)
        temp_med = df.groupby(temp_bins, observed=True)["consumption"].median()
        temp_p10 = df.groupby(temp_bins, observed=True)["consumption"].quantile(0.1)
        temp_p90 = df.groupby(temp_bins, observed=True)["consumption"].quantile(0.9)
        temp_x = np.array([b.mid for b in temp_med.index], dtype=np.float32)
        ax.plot(temp_x, temp_med.values, color=PALETTE["accent"], lw=2, label="Медиана")
        ax.fill_between(temp_x, temp_p10.values, temp_p90.values, color=PALETTE["accent"], alpha=0.2, label="P10-P90")
        ax.set_xlabel("Температура [°C]")
        ax.set_ylabel("Потребление [кВт·ч]")
        ax.set_title("Температурная зависимость (hexbin + доверительный диапазон)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, plots_dir, "05_temperature_dependency.png", save)

    logger.info("EDA завершён. Графики сохранены в: %s", plots_dir)


def _save(fig: plt.Figure, plots_dir: str, name: str, save: bool) -> None:
    path = os.path.join(plots_dir, name)
    png_path, pdf_path, svg_path = save_figure(fig, path, save=save)
    if save:
        logger.debug("График сохранён: %s | %s | %s", png_path, pdf_path, svg_path)
    plt.close(fig)
