# -*- coding: utf-8 -*-
"""
analysis/residuals.py — Анализ остатков прогноза.
Тесты: ADF, KPSS, Ljung-Box, Durbin-Watson, нормальность.
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

logger = logging.getLogger("smart_grid.analysis.residuals")


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    plots_dir: str = "results/plots",
    save: bool = True,
) -> Dict[str, float]:
    """
    Полный анализ остатков: визуализация + статистические тесты.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    model_name     : str
    plots_dir      : str
    save           : bool

    Returns
    -------
    dict с p-values всех тестов.
    """
    os.makedirs(plots_dir, exist_ok=True)
    residuals = y_true.flatten() - y_pred.flatten()
    n = len(residuals)

    logger.info("Анализ остатков модели: %s (n=%d)", model_name, n)

    # ── Тест ADF (стационарность) ─────────────────────────────────────────────
    adf_stat, adf_p, *_ = adfuller(residuals)
    logger.info("ADF тест: stat=%.4f p=%.4f (%s)",
                adf_stat, adf_p, "стационарны" if adf_p < 0.05 else "не стационарны")

    # ── Тест KPSS ─────────────────────────────────────────────────────────────
    try:
        kpss_stat, kpss_p, *_ = kpss(residuals, regression="c", nlags="auto")
        logger.info("KPSS тест: stat=%.4f p=%.4f", kpss_stat, kpss_p)
    except Exception:
        kpss_p = np.nan
        logger.warning("KPSS тест не удался")

    # ── Тест Ljung-Box (автокорреляция) ──────────────────────────────────────
    lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    lb_p_10 = float(lb_result["lb_pvalue"].iloc[0])
    lb_p_20 = float(lb_result["lb_pvalue"].iloc[1])
    logger.info("Ljung-Box lag=10: p=%.4f  lag=20: p=%.4f", lb_p_10, lb_p_20)

    # ── Durbin-Watson ─────────────────────────────────────────────────────────
    dw_stat = durbin_watson(residuals)
    logger.info("Durbin-Watson: %.4f (2.0=нет автокорреляции)", dw_stat)

    # ── Нормальность (Shapiro-Wilk или Jarque-Bera) ───────────────────────────
    if n <= 5000:
        sw_stat, sw_p = stats.shapiro(residuals[:5000])
        logger.info("Shapiro-Wilk: stat=%.4f p=%.4f", sw_stat, sw_p)
        norm_p = sw_p
    else:
        jb_stat, jb_p = stats.jarque_bera(residuals)
        logger.info("Jarque-Bera: stat=%.4f p=%.4f", jb_stat, jb_p)
        norm_p = jb_p

    # ── Визуализация ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Анализ остатков — {model_name}", fontsize=14, fontweight="bold")

    # Остатки во времени
    axes[0, 0].plot(residuals, lw=0.6, alpha=0.7)
    axes[0, 0].axhline(0, color="red", ls="--", lw=1)
    axes[0, 0].set_title("Остатки во времени")
    axes[0, 0].set_ylabel("Остаток")
    axes[0, 0].grid(True, alpha=0.3)

    # Гистограмма + нормальное распределение
    axes[0, 1].hist(residuals, bins=60, density=True, color="steelblue", alpha=0.7)
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    axes[0, 1].plot(xr, stats.norm.pdf(xr, residuals.mean(), residuals.std()),
                    "r-", lw=2, label="N(μ,σ)")
    axes[0, 1].set_title("Распределение остатков")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals, plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # Факт vs прогноз
    axes[1, 1].scatter(y_pred.flatten(), residuals, alpha=0.3, s=2)
    axes[1, 1].axhline(0, color="red", ls="--", lw=1)
    axes[1, 1].set_xlabel("Предсказание")
    axes[1, 1].set_ylabel("Остаток")
    axes[1, 1].set_title("Остатки vs Предсказания")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(plots_dir, f"residuals_{model_name.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("График остатков: %s", path)
    plt.close(fig)

    return {
        "adf_p": adf_p,
        "kpss_p": kpss_p,
        "lb_p_10": lb_p_10,
        "lb_p_20": lb_p_20,
        "durbin_watson": dw_stat,
        "normality_p": norm_p,
    }
