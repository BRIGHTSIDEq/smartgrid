# -*- coding: utf-8 -*-
"""
analysis/residuals.py — Анализ остатков прогноза.
Тесты: ADF, KPSS, Ljung-Box, Durbin-Watson, нормальность.

ИСПРАВЛЕНИЕ v2: защита от константных/близко-константных остатков.
  adfuller() и kpss() бросают ValueError("Invalid input, x is constant")
  когда residuals.std() ≈ 0. Это происходит когда best_pred ≈ y_true
  (почти идеальная модель) или когда best_pred является константой
  (ошибка в пайплайне — например WeightedEnsemble без predict_absolute).

  Решение: проверяем std до вызова тестов, логируем предупреждение.
"""

import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from utils.plot_style import apply_publication_style, get_palette, save_figure

logger = logging.getLogger("smart_grid.analysis.residuals")
apply_publication_style()
PALETTE = get_palette()

# Минимальный std для запуска статистических тестов
# При std < порога остатки считаются константными
_MIN_STD_FOR_TESTS = 1.0   # 1 кВт·ч — практически ноль для нашей задачи


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    plots_dir: str = "results/plots",
    save: bool = True,
) -> Dict[str, float]:
    """
    Полный анализ остатков: визуализация + статистические тесты.

    v2: защита от ValueError при константных остатках.
    Если std(residuals) < 1 кВт·ч — тесты пропускаются с предупреждением.

    Parameters
    ----------
    y_true, y_pred : np.ndarray  — в кВт·ч
    model_name     : str
    plots_dir      : str
    save           : bool

    Returns
    -------
    dict с p-values всех тестов (nan если тест пропущен).
    """
    os.makedirs(plots_dir, exist_ok=True)
    residuals = y_true.flatten() - y_pred.flatten()
    n = len(residuals)
    res_std = float(residuals.std())

    logger.info("Анализ остатков модели: %s (n=%d)", model_name, n)
    logger.info("  Остатки: mean=%.2f, std=%.2f, min=%.2f, max=%.2f кВт·ч",
                float(residuals.mean()), res_std,
                float(residuals.min()), float(residuals.max()))

    # ── Проверка на константность ─────────────────────────────────────────────
    if res_std < _MIN_STD_FOR_TESTS:
        logger.warning(
            "⚠️  Остатки практически константны (std=%.4f кВт·ч < %.1f). "
            "Статистические тесты пропущены. "
            "Проверьте корректность предсказания модели '%s'.",
            res_std, _MIN_STD_FOR_TESTS, model_name,
        )
        return {
            "adf_p": float("nan"), "kpss_p": float("nan"),
            "lb_p_10": float("nan"), "lb_p_20": float("nan"),
            "durbin_watson": float("nan"), "normality_p": float("nan"),
        }

    # ── Тест ADF (стационарность) ─────────────────────────────────────────────
    adf_p = float("nan")
    try:
        adf_stat, adf_p, *_ = adfuller(residuals)
        logger.info("ADF тест: stat=%.4f p=%.4f (%s)",
                    adf_stat, adf_p,
                    "стационарны" if adf_p < 0.05 else "не стационарны")
    except ValueError as exc:
        logger.warning("ADF тест не удался: %s", exc)

    # ── Тест KPSS ─────────────────────────────────────────────────────────────
    kpss_p = float("nan")
    try:
        kpss_stat, kpss_p, *_ = kpss(residuals, regression="c", nlags="auto")
        logger.info("KPSS тест: stat=%.4f p=%.4f", kpss_stat, kpss_p)
    except (ValueError, Exception) as exc:
        logger.warning("KPSS тест не удался: %s", exc)

    # ── Тест Ljung-Box (автокорреляция) ───────────────────────────────────────
    lb_p_10 = lb_p_20 = float("nan")
    try:
        lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        lb_p_10 = float(lb_result["lb_pvalue"].iloc[0])
        lb_p_20 = float(lb_result["lb_pvalue"].iloc[1])
        logger.info("Ljung-Box lag=10: p=%.4f  lag=20: p=%.4f", lb_p_10, lb_p_20)
    except Exception as exc:
        logger.warning("Ljung-Box не удался: %s", exc)

    # ── Durbin-Watson ─────────────────────────────────────────────────────────
    dw_stat = float("nan")
    try:
        dw_stat = durbin_watson(residuals)
        logger.info("Durbin-Watson: %.4f (2.0=нет автокорреляции)", dw_stat)
    except Exception as exc:
        logger.warning("Durbin-Watson не удался: %s", exc)

    # ── Нормальность (Shapiro-Wilk или Jarque-Bera) ───────────────────────────
    norm_p = float("nan")
    try:
        if n <= 5000:
            sw_stat, sw_p = stats.shapiro(residuals[:5000])
            logger.info("Shapiro-Wilk: stat=%.4f p=%.4f", sw_stat, sw_p)
            norm_p = sw_p
        else:
            jb_stat, jb_p = stats.jarque_bera(residuals)
            logger.info("Jarque-Bera: stat=%.4f p=%.4f", jb_stat, jb_p)
            norm_p = jb_p
    except Exception as exc:
        logger.warning("Тест нормальности не удался: %s", exc)

    # ── Визуализация ──────────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Анализ остатков — {model_name}")

        axes[0, 0].plot(residuals, color=PALETTE["primary"], alpha=0.8)
        axes[0, 0].axhline(0, color=PALETTE["negative"], ls="--", lw=1)
        axes[0, 0].set_title("Остатки во времени [кВт·ч]")
        axes[0, 0].set_xlabel("Временной индекс [ч]")
        axes[0, 0].set_ylabel("Остаток [кВт·ч]")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(residuals, bins=60, density=True, color=PALETTE["primary"], alpha=0.7)
        xr = np.linspace(residuals.min(), residuals.max(), 200)
        axes[0, 1].plot(xr, stats.norm.pdf(xr, residuals.mean(), res_std),
                        color=PALETTE["negative"], lw=2, label="N(μ,σ)")
        axes[0, 1].set_title("Распределение остатков [кВт·ч]")
        axes[0, 1].set_xlabel("Остаток [кВт·ч]")
        axes[0, 1].set_ylabel("Плотность [1/кВт·ч]")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        stats.probplot(residuals, plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q график остатков")
        axes[1, 0].set_xlabel("Теоретические квантили [-]")
        axes[1, 0].set_ylabel("Выборочные квантили [кВт·ч]")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].scatter(y_pred.flatten(), residuals, alpha=0.3, s=2)
        axes[1, 1].axhline(0, color=PALETTE["negative"], ls="--", lw=1)
        axes[1, 1].set_xlabel("Предсказание [кВт·ч]")
        axes[1, 1].set_ylabel("Остаток [кВт·ч]")
        axes[1, 1].set_title("Остатки vs предсказания [кВт·ч]")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(plots_dir, f"residuals_{model_name.replace(' ', '_')}.png")
        png_path, _, _ = save_figure(fig, path, save=save)
        if save:
            logger.info("График остатков: %s", png_path)
        plt.close(fig)
    except Exception as exc:
        logger.warning("Не удалось построить графики остатков: %s", exc)

    return {
        "adf_p": adf_p,
        "kpss_p": kpss_p,
        "lb_p_10": lb_p_10,
        "lb_p_20": lb_p_20,
        "durbin_watson": dw_stat,
        "normality_p": norm_p,
    }