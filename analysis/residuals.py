# -*- coding: utf-8 -*-
"""
analysis/residuals.py — Анализ остатков прогноза.
Тесты: ADF, KPSS, Ljung-Box, Durbin-Watson, нормальность.

ВАЖНО о форме данных. Прогноз имеет форму (N, H): N перекрывающихся окон со
сдвигом 1 час, H шагов горизонта. Развёрнутый в один вектор массив НЕ является
временным рядом: элемент k = i·H + h соответствует моменту i + h, поэтому
соседние элементы скачут по времени, а «лаг 24» равен одному часу.
Прогонять по такому вектору ADF/KPSS/Ljung-Box/DW бессмысленно.

Настоящий почасовой ряд — это срез при фиксированном шаге горизонта:
residuals[:, h] индексируется номером окна, а окна отстоят ровно на час.
Все тесты ниже выполняются по такому срезу (по умолчанию h=1, прогноз
на час вперёд).
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

from utils.visualization import save_figure


def extract_residual_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon_step: int = 0,
) -> np.ndarray:
    """
    Возвращает корректный почасовой ряд остатков — срез при фиксированном
    шаге горизонта.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N, H) либо (N,)
    horizon_step : int
        Индекс шага горизонта (0 = прогноз на 1 час вперёд).
    """
    resid = np.atleast_2d(np.asarray(y_true) - np.asarray(y_pred))
    if resid.shape[0] == 1 and resid.shape[1] > 1:
        return resid.ravel().astype(np.float64)
    h = int(np.clip(horizon_step, 0, resid.shape[1] - 1))
    return resid[:, h].astype(np.float64)


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    plots_dir: str = "results/plots",
    save: bool = True,
    horizon_step: int = 0,
) -> Dict[str, float]:
    """
    Полный анализ остатков: визуализация + статистические тесты.

    Тесты выполняются по срезу с фиксированным шагом горизонта (см. docstring
    модуля), гистограмма и Q-Q строятся по всем остаткам целиком.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N, H)
    horizon_step   : int — шаг горизонта для тестов на автокорреляцию

    Returns
    -------
    dict с p-values всех тестов.
    """
    os.makedirs(plots_dir, exist_ok=True)
    residuals = extract_residual_series(y_true, y_pred, horizon_step)
    residuals_all = (np.asarray(y_true) - np.asarray(y_pred)).flatten()
    n = len(residuals)

    logger.info(
        "Анализ остатков модели: %s | ряд для тестов: n=%d (срез h=%d), "
        "всего остатков=%d",
        model_name, n, horizon_step + 1, len(residuals_all),
    )

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
    if len(residuals_all) <= 5000:
        sw_stat, sw_p = stats.shapiro(residuals_all[:5000])
        logger.info("Shapiro-Wilk: stat=%.4f p=%.4f", sw_stat, sw_p)
        norm_p = sw_p
    else:
        jb_stat, jb_p = stats.jarque_bera(residuals_all)
        logger.info("Jarque-Bera: stat=%.4f p=%.4f", jb_stat, jb_p)
        norm_p = jb_p

    # ── Визуализация ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Анализ остатков — {model_name}", fontsize=14, fontweight="bold")

    # Остатки во времени (корректный почасовой ряд: срез h=horizon_step+1)
    axes[0, 0].plot(residuals, lw=0.6, alpha=0.7)
    axes[0, 0].axhline(0, color="red", ls="--", lw=1)
    axes[0, 0].set_title(f"Остатки во времени (прогноз на {horizon_step + 1} ч вперёд)")
    axes[0, 0].set_xlabel("Час")
    axes[0, 0].set_ylabel("Остаток, кВт·ч")
    axes[0, 0].grid(True, alpha=0.3)

    # Гистограмма + нормальное распределение (по всем шагам горизонта)
    axes[0, 1].hist(residuals_all, bins=60, density=True, color="steelblue", alpha=0.7)
    xr = np.linspace(residuals_all.min(), residuals_all.max(), 200)
    axes[0, 1].plot(xr, stats.norm.pdf(xr, residuals_all.mean(), residuals_all.std()),
                    "r-", lw=2, label="N(μ,σ)")
    axes[0, 1].set_title("Распределение остатков (все шаги горизонта)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals_all, plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # Остатки vs предсказания
    axes[1, 1].scatter(np.asarray(y_pred).flatten(), residuals_all, alpha=0.3, s=2)
    axes[1, 1].axhline(0, color="red", ls="--", lw=1)
    axes[1, 1].set_xlabel("Предсказание, кВт·ч")
    axes[1, 1].set_ylabel("Остаток, кВт·ч")
    axes[1, 1].set_title("Остатки vs Предсказания")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(plots_dir, f"residuals_{model_name.replace(' ', '_')}.png")
        save_figure(fig, path, dpi=150)
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