# -*- coding: utf-8 -*-
"""
utils/academic_plots.py — Графики для курсовой работы.

Содержит четыре новых визуализации:

1. plot_mc_dropout_uncertainty()
   График прогноза LSTM с доверительным интервалом ±1σ и ±2σ.
   Строится из 30 forward pass с training=True (MC Dropout).

2. plot_feature_importance_xgboost()
   Горизонтальный barplot топ-N важнейших признаков XGBoost
   с группировкой по типу (лаги / rolling / ковариаты).

3. plot_hpo_comparison()
   Таблица-график: метрики до и после HPO по всем моделям.
   Показывает прирост в % для каждого параметра.

4. plot_sarima_vs_neural()
   Сравнение SARIMA с нейросетями на одном графике — наглядный
   аргумент «статистика vs глубокое обучение».
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from utils.plot_style import apply_publication_style, get_palette, save_figure

logger = logging.getLogger("smart_grid.utils.academic_plots")
apply_publication_style()
PALETTE = get_palette()


# ══════════════════════════════════════════════════════════════════════════════
# 1. MC DROPOUT — ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ
# ══════════════════════════════════════════════════════════════════════════════

def plot_mc_dropout_uncertainty(
    y_true: np.ndarray,
    mc_mean: np.ndarray,
    mc_std: np.ndarray,
    model_name: str = "LSTM",
    n_steps: int = 72,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Строит график прогноза с доверительными интервалами из MC Dropout.

    Parameters
    ----------
    y_true   : np.ndarray (N, H) — истинные значения в кВт·ч
    mc_mean  : np.ndarray (N, H) — среднее по MC-выборкам в кВт·ч
    mc_std   : np.ndarray (N, H) — стд по MC-выборкам в кВт·ч
    n_steps  : int — кол-во часов для отображения

    Интерпретация для курсовой:
      - Широкий интервал = модель «не уверена» → аномальное событие
      - Узкий интервал = высокая уверенность → типичный паттерн
      - Покрытие: ~68% точек должно попасть в ±1σ, ~95% в ±2σ
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Разворачиваем multi-step в хронологический ряд (single origin)
    H = y_true.shape[1] if y_true.ndim == 2 else 24
    n_origins = max(1, (n_steps + H - 1) // H)

    def flatten_origins(arr):
        rows = [arr[i] for i in range(min(n_origins, len(arr)))]
        return np.concatenate(rows)[:n_steps]

    yt  = flatten_origins(y_true)
    ym  = flatten_origins(mc_mean)
    ys  = flatten_origins(mc_std)
    t   = np.arange(len(yt))

    # Считаем покрытие
    in_1sigma = np.mean(np.abs(yt - ym) <= ys)
    in_2sigma = np.mean(np.abs(yt - ym) <= 2 * ys)
    mean_uncertainty = float(ys.mean())

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        f"{model_name} — Прогноз с доверительными интервалами (MC Dropout, 30 выборок)",
        fontsize=13, fontweight="bold",
    )

    # ── График 1: Прогноз + интервалы ─────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, ym - 2*ys, ym + 2*ys,
                    alpha=0.18, color=PALETTE["primary"], label="±2σ (95% теор.)")
    ax.fill_between(t, ym - ys,   ym + ys,
                    alpha=0.35, color=PALETTE["primary"], label="±1σ (68% теор.)")
    ax.plot(t, yt, color=PALETTE["baseline"], lw=2.0, label="Факт", zorder=5)
    ax.plot(t, ym, color=PALETTE["accent"],   lw=1.8, label="MC-среднее", zorder=4)

    ax.set_xlabel("Шаг прогноза [ч]")
    ax.set_ylabel("Потребление [кВт·ч]")
    ax.set_title(
        f"Покрытие: ±1σ={in_1sigma:.1%}  ±2σ={in_2sigma:.1%}  "
        f"| Средняя неопределённость: ±{mean_uncertainty:.0f} кВт·ч"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── График 2: Динамика неопределённости ───────────────────────────────────
    ax2 = axes[1]
    ax2.plot(t, ys, color=PALETTE["warning"], lw=1.8, label="σ (неопределённость)")
    ax2.fill_between(t, 0, ys, alpha=0.3, color=PALETTE["warning"])

    # Отмечаем часы с высокой неопределённостью (>P90)
    p90 = np.percentile(ys, 90)
    high_unc = t[ys > p90]
    if len(high_unc) > 0:
        ax2.scatter(high_unc, ys[ys > p90],
                    color=PALETTE["negative"], s=20, zorder=5,
                    label=f"Высокая неопределённость (>P90={p90:.0f} кВт·ч)")

    ax2.axhline(ys.mean(), color=PALETTE["secondary"], ls="--", lw=1.5,
                label=f"Среднее σ={ys.mean():.0f} кВт·ч")
    ax2.set_xlabel("Шаг прогноза [ч]")
    ax2.set_ylabel("Стд. отклонение σ [кВт·ч]")
    ax2.set_title("Динамика неопределённости прогноза")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, f"mc_dropout_uncertainty_{model_name}.png")
    save_figure(fig, path, save=save)
    plt.close(fig)
    logger.info(
        "MC Dropout график: %s | покрытие ±1σ=%.1f%% ±2σ=%.1f%% | "
        "ср.неопределённость=±%.0f кВт·ч",
        path, in_1sigma * 100, in_2sigma * 100, mean_uncertainty
    )


def compute_mc_predictions(
    model,
    X: np.ndarray,
    scaler,
    n_samples: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Выполняет MC Dropout inference и возвращает mean/std в кВт·ч.

    Parameters
    ----------
    model    : tf.keras.Model — модель с Dropout слоями
    X        : np.ndarray — входной тензор
    scaler   : MinMaxScaler — для обратного масштабирования
    n_samples: int — кол-во MC-выборок

    Returns
    -------
    mc_mean_kwh : np.ndarray (N, H)
    mc_std_kwh  : np.ndarray (N, H)
    """
    from data.preprocessing import inverse_scale

    preds_scaled = np.stack(
        [model(X, training=True).numpy() for _ in range(n_samples)],
        axis=0,
    )  # (n_samples, N, H)

    mean_scaled = preds_scaled.mean(axis=0)
    std_scaled  = preds_scaled.std(axis=0)

    mc_mean_kwh = inverse_scale(scaler, mean_scaled)
    mc_std_kwh  = inverse_scale(scaler, mean_scaled + std_scaled) - mc_mean_kwh

    return mc_mean_kwh, mc_std_kwh


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE IMPORTANCE XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance_xgboost(
    importance_dict: Dict[str, np.ndarray],
    top_n: int = 25,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Горизонтальный barplot важности признаков XGBoost с цветовой группировкой.

    Группы признаков (по цвету):
      🔵 Синий  — лаги потребления (cons_lag_Nh)
      🟠 Оранжевый — rolling статистики потребления
      🟢 Зелёный — ковариаты последнего шага (температура, час, EV и т.д.)
      🔴 Красный — rolling по всем каналам

    Parameters
    ----------
    importance_dict : dict из LagFeaturesWrapper.get_feature_importance()
    top_n           : int — показать топ-N признаков
    """
    os.makedirs(plots_dir, exist_ok=True)

    names       = importance_dict["names"][:top_n]
    importances = importance_dict["importances"][:top_n]

    # Нормируем для читаемости
    imp_pct = importances / (importances.sum() + 1e-8) * 100

    # Определяем группу для цвета
    def _group_color(name: str) -> str:
        n = name.lower()
        if n.startswith("cons_lag"):
            return PALETTE["primary"]      # лаги потребления
        elif n.startswith("cons_") or n.startswith("delta") or n.startswith("ratio"):
            return PALETTE["warning"]      # rolling/тренды потребления
        elif n.startswith("last_"):
            return PALETTE["secondary"]    # ковариаты последнего шага
        else:
            return PALETTE["highlight"]    # rolling по всем каналам

    colors = [_group_color(n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, top_n * 0.35)))
    fig.suptitle(
        "XGBoost — Важность признаков (усреднено по 24 горизонтам прогноза)",
        fontsize=13, fontweight="bold",
    )

    # ── График 1: Barplot ─────────────────────────────────────────────────────
    ax = axes[0]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, imp_pct, color=colors, edgecolor="white", height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Важность признака [%]")
    ax.set_title(f"Топ-{top_n} признаков")
    ax.grid(True, alpha=0.3, axis="x")

    # Подписи значений
    for i, (v, c) in enumerate(zip(imp_pct, colors)):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=7, color="black")

    # Легенда групп
    legend_patches = [
        mpatches.Patch(color=PALETTE["primary"],   label="Лаги потребления"),
        mpatches.Patch(color=PALETTE["warning"],   label="Rolling/тренды потребления"),
        mpatches.Patch(color=PALETTE["secondary"], label="Ковариаты (темп., час, EV...)"),
        mpatches.Patch(color=PALETTE["highlight"], label="Rolling по всем каналам"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    # ── График 2: Накопленная важность ────────────────────────────────────────
    ax2 = axes[1]
    all_imp   = importance_dict["all_importances"]
    cumulative = np.cumsum(np.sort(all_imp)[::-1]) / (all_imp.sum() + 1e-8) * 100
    x_feat    = np.arange(1, len(cumulative) + 1)

    ax2.plot(x_feat, cumulative, color=PALETTE["primary"], lw=2)
    ax2.axhline(80, color=PALETTE["negative"], ls="--", lw=1.5, label="80%")
    ax2.axhline(95, color=PALETTE["accent"],   ls="--", lw=1.5, label="95%")

    # Находим сколько признаков нужно для 80% и 95%
    n80 = int(np.searchsorted(cumulative, 80)) + 1
    n95 = int(np.searchsorted(cumulative, 95)) + 1
    ax2.axvline(n80, color=PALETTE["negative"], ls=":", lw=1, alpha=0.7)
    ax2.axvline(n95, color=PALETTE["accent"],   ls=":", lw=1, alpha=0.7)
    ax2.text(n80, 40, f"n={n80}", color=PALETTE["negative"], fontsize=9, ha="center")
    ax2.text(n95, 55, f"n={n95}", color=PALETTE["accent"],   fontsize=9, ha="center")

    ax2.set_xlabel("Кол-во признаков (по убыванию важности)")
    ax2.set_ylabel("Накопленная важность [%]")
    ax2.set_title("Кривая накопленной важности")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(100, len(cumulative)))

    plt.tight_layout()
    path = os.path.join(plots_dir, "feature_importance_xgboost.png")
    save_figure(fig, path, save=save)
    plt.close(fig)
    logger.info("Feature importance график: %s", path)
    logger.info(
        "  80%% важности = %d признаков | 95%% = %d признаков (всего %d)",
        n80, n95, len(all_imp)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. HPO СРАВНЕНИЕ ДО/ПОСЛЕ
# ══════════════════════════════════════════════════════════════════════════════

# Результаты ДО HPO (из логов v14/v15, первый запуск)
_RESULTS_BEFORE_HPO = {
    "WeightedEnsemble":  {"MAE": 6941.78, "R2": 0.7827, "MAPE": 9.60},
    "XGBoost":           {"MAE": 7177.11, "R2": 0.7684, "MAPE": 9.87},
    "LinearRegression":  {"MAE": 7276.66, "R2": 0.7734, "MAPE": 10.31},
    "PatchTST":          {"MAE": 7268.86, "R2": 0.7668, "MAPE": 10.11},
    "iTransformer":      {"MAE": 7413.28, "R2": 0.7598, "MAPE": 10.37},
    "TFT-Lite":          {"MAE": 7622.10, "R2": 0.7515, "MAPE": 10.74},
    "LSTM":              {"MAE": 8084.62, "R2": 0.7125, "MAPE": 11.20},
}


def plot_hpo_comparison(
    results_after: Dict[str, Dict[str, float]],
    results_before: Optional[Dict[str, Dict[str, float]]] = None,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Визуализирует улучшение метрик после HPO.

    Parameters
    ----------
    results_after  : dict {model_name: {MAE, R2, MAPE}} — после HPO
    results_before : dict — до HPO (если None — берём из лог-констант)
    """
    os.makedirs(plots_dir, exist_ok=True)

    before = results_before or _RESULTS_BEFORE_HPO

    # Находим общие модели
    common_models = [m for m in results_after if m in before]
    if not common_models:
        logger.warning("HPO comparison: нет общих моделей для сравнения")
        return

    mae_before = [before[m]["MAE"]  for m in common_models]
    mae_after  = [results_after[m]["MAE"]  for m in common_models]
    r2_before  = [before[m]["R2"]   for m in common_models]
    r2_after   = [results_after[m]["R2"]   for m in common_models]

    delta_mae_pct = [(b - a) / b * 100 for b, a in zip(mae_before, mae_after)]
    delta_r2_abs  = [a - b            for b, a in zip(r2_before,  r2_after)]

    x = np.arange(len(common_models))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Сравнение метрик до и после автоматического подбора гиперпараметров (HPO)",
        fontsize=13, fontweight="bold",
    )

    # ── MAE до/после ──────────────────────────────────────────────────────────
    ax = axes[0]
    b1 = ax.bar(x - width/2, mae_before, width, label="До HPO",
                color=PALETTE["accent"], alpha=0.8, edgecolor="white")
    b2 = ax.bar(x + width/2, mae_after,  width, label="После HPO",
                color=PALETTE["primary"], alpha=0.8, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Weighted", "W.") for m in common_models],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MAE [кВт·ч]")
    ax.set_title("MAE — чем меньше, тем лучше")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    # ── Δ MAE % ───────────────────────────────────────────────────────────────
    ax2 = axes[1]
    bar_colors = [PALETTE["positive"] if d > 0 else PALETTE["negative"]
                  for d in delta_mae_pct]
    bars = ax2.bar(x, delta_mae_pct, color=bar_colors, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("Weighted", "W.") for m in common_models],
                        rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Снижение MAE [%]")
    ax2.set_title("Улучшение MAE после HPO\n(+зелёный = лучше, -красный = хуже)")
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, d in zip(bars, delta_mae_pct):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.1 if d >= 0 else -0.3),
                 f"{d:+.1f}%", ha="center", va="bottom", fontsize=8)

    # ── R² до/после ───────────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.bar(x - width/2, r2_before, width, label="До HPO",
            color=PALETTE["accent"], alpha=0.8, edgecolor="white")
    ax3.bar(x + width/2, r2_after,  width, label="После HPO",
            color=PALETTE["primary"], alpha=0.8, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace("Weighted", "W.") for m in common_models],
                        rotation=20, ha="right", fontsize=8)
    ax3.set_ylabel("R² [-]")
    ax3.set_title("R² — чем больше, тем лучше")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.set_ylim(0.65, min(1.0, max(r2_before + r2_after) + 0.03))

    plt.tight_layout()
    path = os.path.join(plots_dir, "hpo_comparison.png")
    save_figure(fig, path, save=save)
    plt.close(fig)
    logger.info("HPO comparison график: %s", path)

    # Текстовая таблица в лог
    logger.info("\n%s", "=" * 75)
    logger.info("СРАВНЕНИЕ ДО/ПОСЛЕ HPO:")
    logger.info("%-20s %10s %10s %8s %10s %10s %8s",
                "Модель", "MAE до", "MAE после", "ΔMAE%",
                "R² до", "R² после", "ΔR²")
    logger.info("-" * 75)
    for m, mb, ma, db, rb, ra, dr in zip(
            common_models, mae_before, mae_after, delta_mae_pct,
            r2_before, r2_after, delta_r2_abs):
        logger.info("%-20s %10.2f %10.2f %+7.1f%% %10.4f %10.4f %+7.4f",
                    m, mb, ma, db, rb, ra, dr)
    logger.info("=" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SARIMA VS НЕЙРОСЕТИ
# ══════════════════════════════════════════════════════════════════════════════

def plot_sarima_vs_neural(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    n_steps: int = 96,
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Сравнительный график SARIMA и нейросетей.

    Специально выделяет SARIMA жирной пунктирной линией — наглядно
    показывает разрыв между статистическим и нейросетевым подходами.
    """
    os.makedirs(plots_dir, exist_ok=True)

    H = y_true.shape[1] if y_true.ndim == 2 else 24
    n_origins = max(1, (n_steps + H - 1) // H)

    def flatten_origins(arr):
        rows = [arr[i] for i in range(min(n_origins, len(arr)))]
        return np.concatenate(rows)[:n_steps]

    yt = flatten_origins(y_true)
    t  = np.arange(len(yt))

    # Стили для каждой модели
    model_styles = {
        "SARIMA":          {"color": "#E74C3C", "lw": 2.5, "ls": "--",  "zorder": 8},
        "WeightedEnsemble":{"color": "#2C3E50", "lw": 2.5, "ls": "-",   "zorder": 9},
        "XGBoost":         {"color": "#27AE60", "lw": 1.8, "ls": "-",   "zorder": 7},
        "LinearRegression":{"color": "#8E44AD", "lw": 1.5, "ls": "-.",  "zorder": 6},
        "PatchTST":        {"color": "#2980B9", "lw": 1.5, "ls": "-",   "zorder": 6},
        "iTransformer":    {"color": "#F39C12", "lw": 1.5, "ls": "-",   "zorder": 6},
        "LSTM":            {"color": "#16A085", "lw": 1.5, "ls": "-",   "zorder": 6},
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle(
        "SARIMA vs Нейросети: сравнение точности прогнозирования",
        fontsize=13, fontweight="bold",
    )

    # ── График 1: Прогнозы ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, yt, color="black", lw=2.2, label="Факт", zorder=10)

    for name, pred in predictions.items():
        arr = flatten_origins(pred)
        style = model_styles.get(name, {"color": "gray", "lw": 1.2, "ls": "-", "zorder": 5})
        ax.plot(t, arr[:len(yt)], label=name, **style)

    ax.set_xlabel("Шаг прогноза [ч]")
    ax.set_ylabel("Потребление [кВт·ч]")
    ax.set_title("Прогнозы всех моделей (SARIMA выделен пунктиром)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── График 2: Ошибки ──────────────────────────────────────────────────────
    ax2 = axes[1]
    for name, pred in predictions.items():
        arr = flatten_origins(pred)
        err = np.abs(arr[:len(yt)] - yt)
        style = model_styles.get(name, {"color": "gray", "lw": 1.2, "ls": "-", "zorder": 5})
        style_err = {k: v for k, v in style.items() if k != "zorder"}
        ax2.plot(t, err, label=name, alpha=0.85, **style_err)

    ax2.set_xlabel("Шаг прогноза [ч]")
    ax2.set_ylabel("|Ошибка| [кВт·ч]")
    ax2.set_title("Абсолютная ошибка |факт − прогноз|")
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "sarima_vs_neural.png")
    save_figure(fig, path, save=save)
    plt.close(fig)
    logger.info("SARIMA vs Neural график: %s", path)