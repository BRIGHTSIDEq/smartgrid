# -*- coding: utf-8 -*-
"""
main.py — Точка входа. Оркестрирует полный пайплайн Smart Grid.

Порядок выполнения:
  1.  Конфигурация и логирование
  2.  Генерация данных
  3.  EDA
  4.  Подготовка данных (без data-leakage)
  5.  Построение моделей: LSTM, VanillaTransformer, PatchTST, XGBoost, LinearReg
  6.  Обучение через единый интерфейс ModelTrainer
  7.  Сравнение всех моделей (таблица метрик)
  8.  Визуализация весов внимания лучшего Transformer
  9.  Анализ остатков лучшей модели
  10. Бэктестинг
  11. Оптимизация накопителя (трёхтарифная система, учёт деградации)
  12. Экспорт лучшей модели
"""

import logging
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — без GUI, plt.show() не блокирует

from config import Config

# ── Фиксация случайности ДО импорта tensorflow ───────────────────────────────
np.random.seed(Config.SEED)
random.seed(Config.SEED)

import tensorflow as tf
tf.random.set_seed(Config.SEED)

from data.generator import generate_smartgrid_data, validate_generated_data
from data.preprocessing import prepare_data, inverse_scale, validate_data_integrity
from analysis.eda import run_eda
from analysis.residuals import analyze_residuals
from analysis.backtesting import run_backtesting

from models.lstm import build_lstm_model
from models.transformer import (
    build_vanilla_transformer,
    build_patchtst,
    count_parameters,
)
from models.baseline import build_linear_regression, build_xgboost
from models.trainer import ModelTrainer, compare_trainers

from optimization.storage import simulate_storage, compare_strategies
from utils.visualization import (
    plot_training_history,
    plot_predictions_comparison,
    plot_metrics_comparison,
    plot_storage_result,
)
from utils.deployment import export_model_bundle

logger = Config.setup_logging()


def main() -> None:
    logger.info("=" * 70)
    logger.info("  СИСТЕМА ПРОГНОЗИРОВАНИЯ ЭНЕРГОПОТРЕБЛЕНИЯ — SMART GRID")
    logger.info("=" * 70)

    # ── 1. Режим (раскомментируй нужный) ─────────────────────────────────────
    # Config.set_fast_mode()        # 365 дней, 150 эпох — быстрая отладка
    Config.set_optimal_mode()       # 365 дней, 200 эпох — по умолчанию ✅
    # Config.set_full_mode()          # 365 дней, 300 эпох — финальный запуск
    Config.create_dirs()

    # ── 2. Данные ─────────────────────────────────────────────────────────────
    logger.info("\n[1/10] Генерация данных...")
    df = generate_smartgrid_data(
        days=Config.DAYS,
        households=Config.HOUSEHOLDS,
        start_date=Config.START_DATE,
        seed=Config.SEED,
        temp_setpoint=Config.GEN_TEMP_SETPOINT,
        temp_quadratic_coef=Config.GEN_TEMP_QUADRATIC_COEF,
        humidity_threshold=Config.GEN_HUMIDITY_THRESHOLD,
        humidity_coef=Config.GEN_HUMIDITY_COEF,
        wind_temp_threshold=Config.GEN_WIND_TEMP_THRESHOLD,
        wind_coef=Config.GEN_WIND_COEF,
        early_bird_frac=Config.GEN_EARLY_BIRD_FRAC,
        night_owl_frac=Config.GEN_NIGHT_OWL_FRAC,
        ar_phi=Config.GEN_AR_PHI,
        ar_sigma=Config.GEN_AR_SIGMA,
        seasonal_winter_boost=Config.GEN_SEASONAL_WINTER_BOOST,
        seasonal_summer_dip=Config.GEN_SEASONAL_SUMMER_DIP,
    )
    validate_generated_data(df)
    Config.print_summary()

    # ── 3. EDA ────────────────────────────────────────────────────────────────
    logger.info("\n[2/10] Исследовательский анализ данных...")
    run_eda(df, plots_dir=Config.PLOTS_DIR, save=True)

    # ── 4. Подготовка данных ──────────────────────────────────────────────────
    logger.info("\n[3/10] Подготовка данных (scaler fit только на train)...")
    data = prepare_data(
        df,
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
    )

    # ── ЧЕКЛИСТ: проверка формы тензоров ─────────────────────────────────────
    x_shape = data["X_train"].shape
    assert x_shape[2] == Config.N_FEATURES, (
        f"❌ n_features mismatch: X_train={x_shape[2]}, ожидается {Config.N_FEATURES}"
    )
    logger.info("✅ ЧЕКЛИСТ X_train.shape=%s | %d ковариат подтверждено",
                x_shape, x_shape[2])
    validate_data_integrity(data)

    # ── 5. Инициализация моделей ──────────────────────────────────────────────
    logger.info("\n[4/10] Инициализация моделей...")

    # LSTM — базовая линия
    # AttentionLSTM v4: LSTM + Temporal Self-Attention + LayerNorm + GELU + CosineDecay
    # total_steps ≈ EPOCHS × (n_train // batch): используется для расчёта CosineDecay.
    # Приближение — EarlyStopping всё равно остановит раньше, это безопасно.
    _n_train_approx = int(Config.DAYS * 0.70 * 24 - Config.HISTORY_LENGTH - Config.FORECAST_HORIZON)
    _total_steps = Config.EPOCHS * max(_n_train_approx // Config.BATCH_SIZE, 1)
    lstm = build_lstm_model(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        lstm_units_1=Config.LSTM_UNITS_1,
        lstm_units_2=Config.LSTM_UNITS_2,
        lstm_units_3=Config.LSTM_UNITS_3,
        dropout_rate=Config.DROPOUT_RATE,
        learning_rate=Config.LSTM_LEARNING_RATE,
        attn_heads=Config.LSTM_ATTN_HEADS,               # v4: Temporal Attention
        use_cosine_decay=Config.LSTM_USE_COSINE_DECAY,   # v4: CosineDecay LR
        total_steps=_total_steps,
    )

    # VanillaTransformer v4: Pre-LN + StochasticDepth + LearnedQueryPooling
    vanilla_tr = build_vanilla_transformer(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        d_model=Config.TRANSFORMER_D_MODEL,
        num_heads=Config.TRANSFORMER_N_HEADS,
        num_layers=Config.TRANSFORMER_N_LAYERS,
        dff=Config.TRANSFORMER_DFF,
        dropout=Config.TRANSFORMER_DROPOUT,
        learning_rate=Config.TRANSFORMER_LEARNING_RATE,
        pe_type="sinusoidal",
        stochastic_depth_rate=Config.TRANSFORMER_STOCHASTIC_DEPTH,  # v4: DropPath
    )

    # PatchTST v4: RevIN + StochasticDepth + LearnedQueryPooling
    patch_len = 8 if Config.HISTORY_LENGTH >= 48 else 6
    stride = patch_len // 2
    patchtst = build_patchtst(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        patch_len=patch_len,
        stride=stride,
        n_features=Config.N_FEATURES,
        d_model=Config.TRANSFORMER_D_MODEL,
        num_heads=Config.TRANSFORMER_N_HEADS,
        num_layers=Config.TRANSFORMER_N_LAYERS,
        dff=Config.TRANSFORMER_DFF,
        dropout=Config.TRANSFORMER_DROPOUT,
        learning_rate=Config.TRANSFORMER_LEARNING_RATE,
        stochastic_depth_rate=Config.TRANSFORMER_STOCHASTIC_DEPTH,  # v4: DropPath
        use_revin=Config.PATCHTST_USE_REVIN,                        # v4: RevIN
    )

    # Базовые sklearn-модели
    lr_model = build_linear_regression()
    xgb_model = build_xgboost(
        n_estimators=Config.XGB_N_ESTIMATORS,
        max_depth=Config.XGB_MAX_DEPTH,
        learning_rate=Config.XGB_LR,
        subsample=Config.XGB_SUBSAMPLE,
        colsample_bytree=Config.XGB_COLSAMPLE,
        seed=Config.SEED,
    )

    # Вывод параметрического паритета
    logger.info(
        "Параметры: LSTM=%d | VanillaTrans=%d | PatchTST=%d",
        count_parameters(lstm),
        count_parameters(vanilla_tr),
        count_parameters(patchtst),
    )

    # ── 6. Обучение через единый интерфейс ────────────────────────────────────
    logger.info("\n[5/10] Обучение моделей...")

    models_to_train = [
        (lstm,       "LSTM"),
        (vanilla_tr, "VanillaTransformer"),
        (patchtst,   "PatchTST"),
        (lr_model,   "LinearRegression"),
        (xgb_model,  "XGBoost"),
    ]

    trainers = []
    for model, name in models_to_train:
        trainer = ModelTrainer(model, name, Config.MODELS_DIR, Config.PLOTS_DIR)
        trainer.train(
            data,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            patience=Config.PATIENCE,
            lr_patience=Config.LR_PATIENCE,
            lr_factor=Config.LR_FACTOR,
            min_delta=Config.MIN_DELTA,
        )
        trainers.append(trainer)

        if trainer.history is not None:
            plot_training_history(
                trainer.history, model_name=name, plots_dir=Config.PLOTS_DIR
            )

    # ── 7. Сравнение моделей ──────────────────────────────────────────────────
    logger.info("\n[6/10] Сравнение моделей на тестовой выборке...")
    all_metrics = compare_trainers(trainers, data, split="test")
    plot_metrics_comparison(all_metrics, plots_dir=Config.PLOTS_DIR)

    scaler = data["scaler"]
    predictions = {
        t.model_name: inverse_scale(scaler, t.predict(data["X_test"]))
        for t in trainers
    }
    y_true = inverse_scale(scaler, data["Y_test"])
    plot_predictions_comparison(y_true, predictions, plots_dir=Config.PLOTS_DIR)

    best_name = min(all_metrics, key=lambda k: all_metrics[k]["MAE"])
    best_trainer = next(t for t in trainers if t.model_name == best_name)
    logger.info("🏆 Лучшая модель: %s (MAE=%.2f)", best_name, all_metrics[best_name]["MAE"])

    # ── 8. Визуализация внимания (только для Transformer-моделей) ─────────────
    logger.info("\n[7/10] Визуализация весов внимания...")
    try:
        from utils.attention_visualization import (
            visualize_attention_weights,
            visualize_attention_summary,
            compare_head_specialization,
        )
        # Берём один пример из теста
        sample_x = data["X_test"][:1]

        for t in trainers:
            if t.model_name in ("VanillaTransformer", "PatchTST"):
                visualize_attention_weights(
                    t.model, sample_x,
                    history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name,
                    plots_dir=Config.PLOTS_DIR,
                )
                visualize_attention_summary(
                    t.model, sample_x,
                    history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name,
                    plots_dir=Config.PLOTS_DIR,
                )
                compare_head_specialization(
                    t.model, sample_x,
                    model_name=t.model_name,
                    plots_dir=Config.PLOTS_DIR,
                )
    except Exception as exc:
        logger.warning("Визуализация внимания пропущена: %s", exc)

    # ── 9. Анализ остатков ────────────────────────────────────────────────────
    logger.info("\n[8/10] Анализ остатков лучшей модели: %s", best_name)
    best_pred = inverse_scale(scaler, best_trainer.predict(data["X_test"]))
    analyze_residuals(
        y_true, best_pred, model_name=best_name, plots_dir=Config.PLOTS_DIR
    )

    # ── 10. Бэктестинг ────────────────────────────────────────────────────────
    logger.info("\n[9/10] Скользящий бэктестинг...")
    run_backtesting(
        best_trainer.model,
        data,
        n_windows=8,
        plots_dir=Config.PLOTS_DIR,
        model_name=best_name,
    )

    # ── 11. Оптимизация накопителя ────────────────────────────────────────────
    logger.info("\n[10/10] Оптимизация накопителя энергии...")
    # raw_test — непрерывный реальный ряд потребления.
    # predictions[].flatten() — это склеенные 24-часовые окна (артефакт нарезки).
    sample_forecast = data["raw_test"][:Config.STORAGE_HORIZON]

    storage_result = simulate_storage(
        forecast=sample_forecast,
        capacity=Config.BATTERY_CAPACITY,
        max_power=Config.BATTERY_MAX_POWER,
        round_trip_efficiency=Config.BATTERY_EFFICIENCY,
        cycle_cost_per_kwh=Config.BATTERY_CYCLE_COST,
        min_soc=Config.BATTERY_MIN_SOC,
        max_soc=Config.BATTERY_MAX_SOC,
        tariff_night=Config.TARIFF_NIGHT,
        tariff_half_peak=Config.TARIFF_HALF_PEAK,
        tariff_peak=Config.TARIFF_PEAK,
        demand_charge_rub_per_kw_month=Config.DEMAND_CHARGE_RUB_PER_KW_MONTH,
        annual_om_share=Config.BATTERY_OM_SHARE,
    )
    plot_storage_result(storage_result, sample_forecast, plots_dir=Config.PLOTS_DIR)
    compare_strategies(
        sample_forecast,
        capacity=Config.BATTERY_CAPACITY,
        max_power=Config.BATTERY_MAX_POWER,
        round_trip_efficiency=Config.BATTERY_EFFICIENCY,
        cycle_cost_per_kwh=Config.BATTERY_CYCLE_COST,
        battery_cost_rub=Config.BATTERY_COST_RUB,   # ← ИСПРАВЛЕНО: было дефолт 3M вместо 45M
        tariff_night=Config.TARIFF_NIGHT,
        tariff_half_peak=Config.TARIFF_HALF_PEAK,
        tariff_peak=Config.TARIFF_PEAK,
        demand_charge_rub_per_kw_month=Config.DEMAND_CHARGE_RUB_PER_KW_MONTH,
        annual_om_share=Config.BATTERY_OM_SHARE,
    )

    # ── 12. Экспорт лучшей Keras-модели ──────────────────────────────────────
    if isinstance(best_trainer.model, tf.keras.Model):
        logger.info("Экспорт модели '%s'...", best_name)
        export_model_bundle(
            best_trainer.model,
            scaler,
            {
                "HISTORY_LENGTH": Config.HISTORY_LENGTH,
                "FORECAST_HORIZON": Config.FORECAST_HORIZON,
                "model_name": best_name,
            },
            export_dir=Config.MODELS_DIR,
            model_name=best_name,
        )

    logger.info("\n" + "=" * 70)
    logger.info("✅ Пайплайн завершён успешно!")
    logger.info("📁 Результаты в: %s", Config.OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()