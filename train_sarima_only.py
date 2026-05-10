# -*- coding: utf-8 -*-
"""
train_sarima_only.py — Обучение и оценка только SARIMA (исправленная версия).

Два бага в оригинальном скрипте:

БАГ 1 — Обучение только на raw_train:
  Тестовые окна начинаются через 2628 шагов после конца train.
  SARIMA прогнозировала «прошлое» относительно test → MBE=-31493.
  Исправление: обучаем на raw_train + raw_val.

БАГ 2 — predict() вызывал forecast(24) N=2413 раз с одной точки:
  Statsmodels forecast() каждый раз стартует с конца обучающего ряда.
  Все 2413 окон получали одинаковый прогноз → MAPE=40%, R²=-2.35.
  Исправление: один forecast(history+N+23), нарезается на окна.

Использование:
    python train_sarima_only.py
"""

import logging
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from data.generator import load_or_generate_smartgrid_data
from data.preprocessing import prepare_data, inverse_scale
from models.baseline import build_sarima
from utils.metrics import compute_all_metrics
from utils.plot_style import apply_publication_style, get_palette, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sarima_standalone")

apply_publication_style()
PALETTE = get_palette()


# ══════════════════════════════════════════════════════════════════════════════
# ИСПРАВЛЕННАЯ ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# Обходит сломанный predict() в SARIMAWrapper без правки baseline.py
# ══════════════════════════════════════════════════════════════════════════════

def sarima_predict_fixed(fitted_model, X: np.ndarray, horizon: int = 24) -> np.ndarray:
    """
    Правильное предсказание SARIMA для N тестовых окон.

    Вместо N вызовов forecast(24) с одной точки генерирует один длинный
    forecast(history + N + horizon - 1) и нарезает его на окна:
        окно i = long_forecast[history+i : history+i+horizon]

    Это корректно, потому что тестовые окна сдвинуты на 1 шаг друг
    относительно друга — ровно как срезы одного длинного ряда.
    """
    N = len(X)
    history_len = int(X.shape[1]) if X.ndim == 3 else 192
    total_steps = history_len + N + horizon - 1

    logger.info(
        "Генерация прогноза: %d шагов (history=%d + N_test=%d + horizon=%d − 1)",
        total_steps, history_len, N, horizon,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long_fc = fitted_model.forecast(steps=total_steps)

    predictions = np.stack([
        np.maximum(long_fc[history_len + i : history_len + i + horizon], 0)
        for i in range(N)
    ]).astype(np.float32)

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# ГРАФИК
# ══════════════════════════════════════════════════════════════════════════════

def plot_sarima_result(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    plots_dir: str,
    n_steps: int = 168,
) -> None:
    """Прогноз vs факт + остатки на первых n_steps часах тест-периода."""
    H = y_true.shape[1]
    n_origins = max(1, (n_steps + H - 1) // H)

    yt = np.concatenate([y_true[i] for i in range(n_origins)])[:n_steps]
    yp = np.concatenate([y_pred[i] for i in range(n_origins)])[:n_steps]
    t  = np.arange(len(yt))
    residuals = yt - yp

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(
        f"SARIMA(1,1,1)(1,1,1,24) | MAE={metrics['MAE']:.0f} кВт·ч"
        f"  MAPE={metrics['MAPE']:.1f}%  R²={metrics['R2']:.4f}",
        fontsize=13, fontweight="bold",
    )

    axes[0].plot(t, yt, color=PALETTE["baseline"], lw=2.0, label="Факт",   zorder=5)
    axes[0].plot(t, yp, color=PALETTE["accent"],   lw=1.8, label="SARIMA",
                 linestyle="--", zorder=4)
    axes[0].fill_between(t, yt, yp, alpha=0.2, color=PALETTE["warning"], label="Ошибка")
    axes[0].set_ylabel("Потребление [кВт·ч]")
    axes[0].set_title("Прогноз SARIMA — первые 168 часов тестового периода")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, residuals, color=PALETTE["primary"], lw=1.5, alpha=0.8)
    axes[1].axhline(0, color="black", lw=1.0, ls="--")
    axes[1].axhline(
        residuals.mean(), color=PALETTE["negative"], lw=1.5, ls="--",
        label=f"MBE = {residuals.mean():.0f} кВт·ч",
    )
    axes[1].fill_between(t, 0, residuals, alpha=0.2, color=PALETTE["primary"])
    axes[1].set_xlabel("Шаг прогноза [ч]")
    axes[1].set_ylabel("Остаток [кВт·ч]")
    axes[1].set_title("Остатки (факт − SARIMA)")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "sarima_fixed_vs_fact.png")
    save_figure(fig, path, save=True)
    plt.close(fig)
    logger.info("График сохранён: %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ПАЙПЛАЙН
# ══════════════════════════════════════════════════════════════════════════════

def main():
    Config.set_optimal_mode()
    Config.create_dirs()

    # ── 1. Данные из кэша ──────────────────────────────────────────────────────
    logger.info("[1/5] Загрузка данных из кэша...")
    df = load_or_generate_smartgrid_data(
        csv_path=Config.GENERATED_DATA_CSV,
        force_regenerate=False,
        days=Config.DAYS,
        households=Config.HOUSEHOLDS,
        start_date=Config.START_DATE,
        seed=Config.SEED,
        coefficients=Config.get_generator_coefficients(),
    )
    logger.info(
        "Загружено %d строк | mean=%.0f  std=%.0f кВт·ч",
        len(df), df["consumption"].mean(), df["consumption"].std(),
    )

    # ── 2. Подготовка ──────────────────────────────────────────────────────────
    logger.info("[2/5] Подготовка данных (history=%d)...", Config.HISTORY_LENGTH)
    data = prepare_data(
        df,
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        seasonal_diff=False,
    )
    scaler    = data["scaler"]
    X_test    = data["X_test"]
    Y_test    = data["Y_test"]

    logger.info(
        "Train=%d  Val=%d  Test=%d окон",
        len(data["X_train"]), len(data["X_val"]), len(X_test),
    )

    # ── 3. Обучение на train+val ───────────────────────────────────────────────
    # ИСПРАВЛЕНИЕ БАГ 1: объединяем train и val, чтобы конец обучающего ряда
    # совпадал с началом тестового периода.
    logger.info("[3/5] Обучение SARIMA(1,1,1)(1,1,1,24) на train+val...")
    raw_trainval = np.concatenate(
        [data["raw_train"], data["raw_val"]]
    ).astype(np.float64)

    logger.info(
        "Длина обучающего ряда: %d точек  (train=%d + val=%d)",
        len(raw_trainval), len(data["raw_train"]), len(data["raw_val"]),
    )

    sarima = build_sarima(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        refit=False,
    )
    sarima.fit(
        data["X_train"],   # нужен только для совместимости интерфейса
        data["Y_train"],
        raw_series=raw_trainval,
    )
    logger.info("SARIMA обучена. AIC=%.2f", sarima._fitted_model.aic)

    # ── 4. Предсказание (исправленная логика) ──────────────────────────────────
    logger.info("[4/5] Предсказание на тестовой выборке...")
    # ИСПРАВЛЕНИЕ БАГ 2: используем sarima_predict_fixed вместо sarima.predict()
    y_pred_kwh = sarima_predict_fixed(
        fitted_model=sarima._fitted_model,
        X=X_test,
        horizon=Config.FORECAST_HORIZON,
    )
    y_true_kwh = inverse_scale(scaler, Y_test)

    # ── 5. Метрики, таблица, график ────────────────────────────────────────────
    logger.info("[5/5] Оценка результатов...")
    metrics = compute_all_metrics(y_true_kwh, y_pred_kwh, model_name="SARIMA_fixed")

    logger.info("")
    logger.info("=" * 60)
    logger.info("SARIMA (исправленная) — тестовая выборка:")
    logger.info("  MAE    = %8.2f кВт·ч", metrics["MAE"])
    logger.info("  RMSE   = %8.2f кВт·ч", metrics["RMSE"])
    logger.info("  MAPE   = %8.2f %%",    metrics["MAPE"])
    logger.info("  sMAPE  = %8.2f %%",    metrics["sMAPE"])
    logger.info("  R²     = %8.4f",       metrics["R2"])
    logger.info("  MBE    = %8.2f кВт·ч", metrics["MBE"])
    logger.info("=" * 60)

    # Сравнение с нейросетями из последнего run.log
    REFERENCE = {
        "WeightedEnsemble": {"MAE": 6880.17, "R2": 0.7875, "MAPE":  9.51},
        "XGBoost":          {"MAE": 6990.25, "R2": 0.7793, "MAPE":  9.67},
        "LinearRegression": {"MAE": 7260.07, "R2": 0.7744, "MAPE": 10.31},
        "iTransformer":     {"MAE": 7298.82, "R2": 0.7685, "MAPE": 10.30},
        "PatchTST":         {"MAE": 7492.95, "R2": 0.7555, "MAPE": 10.49},
        "LSTM":             {"MAE": 7747.82, "R2": 0.7414, "MAPE": 10.80},
        "TFT-Lite":         {"MAE": 7932.51, "R2": 0.7287, "MAPE": 10.93},
        "SARIMA":           {
            "MAE":  metrics["MAE"],
            "R2":   metrics["R2"],
            "MAPE": metrics["MAPE"],
        },
    }

    logger.info("")
    logger.info("Итоговое сравнение всех моделей:")
    logger.info("%-22s %9s %9s %8s", "Модель", "MAE", "MAPE%", "R²")
    logger.info("-" * 54)
    for name, m in sorted(REFERENCE.items(), key=lambda x: x[1]["MAE"]):
        marker = " ◄ SARIMA" if name == "SARIMA" else ""
        logger.info(
            "%-22s %9.2f %8.2f%% %8.4f%s",
            name, m["MAE"], m["MAPE"], m["R2"], marker,
        )
    logger.info("=" * 60)

    plot_sarima_result(y_true_kwh, y_pred_kwh, metrics, Config.PLOTS_DIR)


if __name__ == "__main__":
    main()