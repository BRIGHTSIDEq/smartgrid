# -*- coding: utf-8 -*-
"""
main.py — Smart Grid v11.

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v11: seasonal_diff=True в prepare_data().

  Все модели обучаются предсказывать Y_diff = Y - Y_naive вместо Y_abs.
  Реконструкция в trainer.py: pred_abs = pred_diff + Y_naive.
  Ожидаемый результат: ACF(24) остатков < 0.15 (с 0.70), R² → 0.88–0.91.

  Дополнительно:
    - huber_delta=0.02 в LSTM (масштаб Y_diff меньше, kurtosis>5)
    - use_seasonal_skip=False в LSTM (больше не нужен при seasonal_diff=True)
    - WeightedEnsemble.optimize_weights работает в абсолютном пространстве
    - predict_absolute() для визуализации вместо predict()
    - TFT-Lite: исправлен вызов (убран неверный аргумент n_features)
"""

import logging
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")

from config import Config

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
    build_vanilla_transformer, build_patchtst, build_tft_lite, count_parameters,
    prepare_tft_covariates,
)
from models.baseline import build_linear_regression, build_xgboost
from models.trainer import ModelTrainer, WeightedEnsemble, compare_trainers

from optimization.storage import simulate_storage, compare_strategies
from utils.visualization import (
    plot_training_history, plot_predictions_comparison,
    plot_metrics_comparison, plot_storage_result,
)
from utils.deployment import export_model_bundle

logger = Config.setup_logging()


def main():
    logger.info("=" * 70)
    logger.info("  SMART GRID v11 — Seasonal Differencing Target")
    logger.info("=" * 70)

    # Config.set_fast_mode()
    # Config.set_optimal_mode()
    Config.set_full_mode()
    Config.create_dirs()

    # [1/10] Генерация данных
    logger.info("\n[1/10] Генерация данных...")
    df = generate_smartgrid_data(
        days=Config.DAYS, households=Config.HOUSEHOLDS,
        start_date=Config.START_DATE, seed=Config.SEED,
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
        ev_penetration=Config.GEN_EV_PENETRATION,
        solar_penetration=Config.GEN_SOLAR_PENETRATION,
        industrial_loads=Config.GEN_INDUSTRIAL_LOADS,
        city_districts=Config.GEN_CITY_DISTRICTS,
        coefficients=Config.get_generator_coefficients(),
    )
    validate_generated_data(df)
    Config.print_summary()

    # [2/10] EDA
    logger.info("\n[2/10] EDA...")
    run_eda(df, plots_dir=Config.PLOTS_DIR, save=True)

    # [3/10] Подготовка данных — seasonal_diff=True (главное изменение v11)
    logger.info("\n[3/10] Подготовка данных (seasonal_diff=True)...")
    data = prepare_data(
        df,
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        seasonal_diff=True,   # ← КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v11
    )
    x_shape = data["X_train"].shape
    assert x_shape[2] == Config.N_FEATURES, (
        f"n_features mismatch: {x_shape[2]} != {Config.N_FEATURES}"
    )
    logger.info(
        "✅ X_train=%s | seasonal_diff=%s | Y_diff std=%.4f",
        x_shape, data["seasonal_diff"], float(data["Y_train"].std()),
    )
    validate_data_integrity(data)
    lag_idx = data["lag_feature_start_idx"]

    # [4/10] Инициализация моделей
    logger.info("\n[4/10] Инициализация моделей...")

    _n_train = int(Config.DAYS * 0.70 * 24 - Config.HISTORY_LENGTH - Config.FORECAST_HORIZON)
    _total_steps = Config.EPOCHS * max(_n_train // Config.BATCH_SIZE, 1)

    # LSTM v11: use_seasonal_skip=False, huber_delta=0.02 (масштаб Y_diff)
    lstm = build_lstm_model(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        lstm_units_1=Config.LSTM_UNITS_1,
        lstm_units_2=Config.LSTM_UNITS_2,
        lstm_units_3=Config.LSTM_UNITS_3,
        dropout_rate=Config.DROPOUT_RATE,
        learning_rate=Config.LSTM_LEARNING_RATE,
        attn_heads=Config.LSTM_ATTN_HEADS,
        use_cosine_decay=Config.LSTM_USE_COSINE_DECAY,
        total_steps=_total_steps,
        tcn_filters=Config.LSTM_TCN_FILTERS,
        huber_delta=0.02,           # v11: меньше delta для Y_diff (kurtosis>5)
        seasonal_blend_init=Config.LSTM_SEASONAL_BLEND_INIT,
        use_seasonal_skip=False,    # v11: seasonal_diff на уровне данных
        lag_feature_start_idx=lag_idx,
    )

    # VanillaTransformer
    vanilla_tr = build_vanilla_transformer(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        d_model=Config.TRANSFORMER_D_MODEL,
        num_heads=Config.TRANSFORMER_N_HEADS,
        num_layers=Config.TRANSFORMER_N_LAYERS,
        dff=Config.TRANSFORMER_DFF,
        dropout=Config.TRANSFORMER_DROPOUT,
        learning_rate=Config.VANILLA_TRANSFORMER_LR,
        pe_type="sinusoidal",
        stochastic_depth_rate=Config.TRANSFORMER_STOCHASTIC_DEPTH,
        use_seasonal_residual=False,   # v11: seasonal_diff на уровне данных
        seasonal_blend_init=Config.VANILLA_SEASONAL_BLEND_INIT,
        huber_delta=0.02,              # v11: меньше delta для Y_diff
    )

    # PatchTST
    if Config.HISTORY_LENGTH >= 192:  patch_len = 16
    elif Config.HISTORY_LENGTH >= 96: patch_len = 12
    elif Config.HISTORY_LENGTH >= 48: patch_len = 8
    else:                             patch_len = 6
    stride   = patch_len // 2
    patchtst = build_patchtst(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        patch_len=patch_len, stride=stride,
        n_features=Config.N_FEATURES,
        d_model=Config.TRANSFORMER_D_MODEL,
        num_heads=Config.TRANSFORMER_N_HEADS,
        num_layers=Config.TRANSFORMER_N_LAYERS,
        dff=Config.TRANSFORMER_DFF,
        dropout=Config.TRANSFORMER_DROPOUT,
        learning_rate=Config.TRANSFORMER_LEARNING_RATE,
        stochastic_depth_rate=Config.TRANSFORMER_STOCHASTIC_DEPTH,
        use_revin=Config.PATCHTST_USE_REVIN,
        huber_delta=0.02,              # v11
    )

    # TFT-Lite (исправленный вызов — без n_features как отдельного аргумента)
    use_tft = False
    tft_lite = None
    try:
        tft_lite = build_tft_lite(
            history_length=Config.HISTORY_LENGTH,
            forecast_horizon=Config.FORECAST_HORIZON,
            d_model=min(Config.TRANSFORMER_D_MODEL, 128),
            num_heads=min(Config.TRANSFORMER_N_HEADS, 4),
            num_layers=min(Config.TRANSFORMER_N_LAYERS, 3),
            dff=min(Config.TRANSFORMER_DFF, 256),
            dropout=Config.TRANSFORMER_DROPOUT,
            learning_rate=Config.TRANSFORMER_LEARNING_RATE,
        )
        n_tr = len(data["X_train"]); n_vl = len(data["X_val"]); n_te = len(data["X_test"])
        tft_n_cov = 4
        zero_cov = lambda n: np.zeros((n, Config.HISTORY_LENGTH, tft_n_cov), np.float32)
        data["X_tft_train"] = [data["X_train"], zero_cov(n_tr)]
        data["X_tft_val"]   = [data["X_val"],   zero_cov(n_vl)]
        data["X_tft_test"]  = [data["X_test"],  zero_cov(n_te)]
        use_tft = True
        logger.info("TFT-Lite: %d params", count_parameters(tft_lite))
    except Exception as exc:
        logger.warning("TFT-Lite пропущен: %s", exc)

    lr_model  = build_linear_regression()
    xgb_model = build_xgboost(
        n_estimators=Config.XGB_N_ESTIMATORS,
        max_depth=Config.XGB_MAX_DEPTH,
        learning_rate=Config.XGB_LR,
        subsample=Config.XGB_SUBSAMPLE,
        colsample_bytree=Config.XGB_COLSAMPLE,
        seed=Config.SEED,
    )

    logger.info("Параметры: LSTM=%d | VanillaTr=%d | PatchTST=%d",
                count_parameters(lstm), count_parameters(vanilla_tr), count_parameters(patchtst))

    # [5/10] Обучение
    logger.info("\n[5/10] Обучение моделей (на Y_diff)...")
    models_to_train = [
        (lstm,       "LSTM"),
        (vanilla_tr, "VanillaTransformer"),
        (patchtst,   "PatchTST"),
        (lr_model,   "LinearRegression"),
        (xgb_model,  "XGBoost"),
    ]
    if use_tft and tft_lite is not None:
        models_to_train.append((tft_lite, "TFT-Lite"))

    trainers = []
    for model, name in models_to_train:
        trainer = ModelTrainer(model, name, Config.MODELS_DIR, Config.PLOTS_DIR)
        if name == "TFT-Lite":
            train_data = {**data, "X_train": data["X_tft_train"], "X_val": data["X_tft_val"]}
        else:
            train_data = data
        trainer.train(train_data, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
                      patience=Config.PATIENCE, lr_patience=Config.LR_PATIENCE,
                      lr_factor=Config.LR_FACTOR, min_delta=Config.MIN_DELTA)
        trainers.append(trainer)
        if trainer.history is not None:
            plot_training_history(trainer.history, model_name=name, plots_dir=Config.PLOTS_DIR)

    # [5b/10] WeightedEnsemble — оптимизация в абсолютном пространстве (кВт·ч)
    logger.info("\n[5b/10] WeightedEnsemble (оптимизация в абсолютном пространстве)...")
    single_trainers = [t for t in trainers if t.model_name != "TFT-Lite"]
    ensemble = WeightedEnsemble(single_trainers, model_name="WeightedEnsemble")
    try:
        ensemble.optimize_weights(data, split="val")
        trainers_all = trainers + [ensemble]
    except Exception as exc:
        logger.warning("Ensemble оптимизация не удалась: %s", exc)
        trainers_all = trainers + [ensemble]

    # [6/10] Сравнение
    logger.info("\n[6/10] Сравнение (метрики в кВт·ч, seasonal_diff реконструкция)...")
    all_metrics, best_name = compare_trainers(trainers_all, data, split="test", select_by="composite")
    plot_metrics_comparison(all_metrics, plots_dir=Config.PLOTS_DIR)

    # Визуализация — используем predict_absolute для всех тренеров
    scaler = data["scaler"]
    predictions = {}
    for t in trainers_all:
        try:
            if t.model_name == "TFT-Lite":
                raw = t.predict(data["X_tft_test"])
                from data.preprocessing import inverse_scale as _inv
                if data["seasonal_diff"]:
                    naive = data["Y_seasonal_naive_test"]
                    predictions[t.model_name] = _inv(scaler, (raw + naive).clip(0))
                else:
                    predictions[t.model_name] = _inv(scaler, raw)
            else:
                predictions[t.model_name] = t.predict_absolute(data, split="test") if hasattr(t, "predict_absolute") else t.evaluate(data)["MAE"]
                # fallback для WeightedEnsemble у которого нет predict_absolute
                if isinstance(t, WeightedEnsemble):
                    raw = t.predict(data["X_test"])
                    from data.preprocessing import inverse_scale as _inv
                    if data["seasonal_diff"]:
                        naive = data["Y_seasonal_naive_test"]
                        predictions[t.model_name] = _inv(scaler, (raw + naive).clip(0))
                    else:
                        predictions[t.model_name] = _inv(scaler, raw)
        except Exception as exc:
            logger.warning("predict_absolute %s: %s", t.model_name, exc)

    # Истинные абсолютные значения
    if data["seasonal_diff"]:
        from data.preprocessing import inverse_scale as _inv
        y_true = _inv(scaler, (data["Y_test"] + data["Y_seasonal_naive_test"]).clip(0))
    else:
        y_true = inverse_scale(scaler, data["Y_test"])

    if predictions:
        plot_predictions_comparison(y_true, predictions, plots_dir=Config.PLOTS_DIR)

    best_trainer = next((t for t in trainers_all if t.model_name == best_name), trainers[0])
    logger.info("🏆 Лучшая: %s | MAE=%.2f | composite=%.3f",
                best_name, all_metrics[best_name]["MAE"],
                all_metrics[best_name].get("composite_score", float("nan")))

    # [7/10] Attention
    logger.info("\n[7/10] Визуализация внимания...")
    try:
        from utils.attention_visualization import (
            visualize_attention_weights, visualize_attention_summary, compare_head_specialization)
        sample_x = data["X_test"][:1]
        for t in trainers:
            if t.model_name in ("VanillaTransformer", "PatchTST"):
                visualize_attention_weights(t.model, sample_x, history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
                visualize_attention_summary(t.model, sample_x, history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
                compare_head_specialization(t.model, sample_x, model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
    except Exception as exc:
        logger.warning("Attention пропущена: %s", exc)

    # [8/10] Остатки
    logger.info("\n[8/10] Анализ остатков: %s", best_name)
    if best_name in predictions:
        best_pred = predictions[best_name]
    else:
        best_pred = best_trainer.predict_absolute(data, "test") if hasattr(best_trainer, "predict_absolute") else y_true
    analyze_residuals(y_true, best_pred, model_name=best_name, plots_dir=Config.PLOTS_DIR)

    # [9/10] Бэктестинг (на LSTM)
    logger.info("\n[9/10] Бэктестинг (LSTM)...")
    lstm_trainer = next((t for t in trainers if t.model_name == "LSTM"), trainers[0])
    run_backtesting(lstm_trainer.model, data, n_windows=8,
                    plots_dir=Config.PLOTS_DIR, model_name="LSTM")

    # [9b/10] MC Dropout
    logger.info("\n[9b/10] MC Dropout (LSTM)...")
    try:
        mc_mean, mc_std = lstm_trainer.mc_predict(data["X_test"][:500], n_samples=30)
        # MC std — в пространстве diff, масштабируем приблизительно
        mc_std_kwh = float(mc_std.mean()) * float(scaler.scale_[0]) if hasattr(scaler, "scale_") else float(mc_std.mean())
        logger.info("MC Dropout | mean diff uncertainty: ±%.4f (scaled)", float(mc_std.mean()))
    except Exception as exc:
        logger.warning("MC Dropout пропущен: %s", exc)

    # [10/10] Батарея
    logger.info("\n[10/10] Оптимизация накопителя...")
    sample_forecast = data["raw_test"][:Config.STORAGE_HORIZON]
    storage_result = simulate_storage(
        forecast=sample_forecast,
        capacity=Config.BATTERY_CAPACITY, max_power=Config.BATTERY_MAX_POWER,
        round_trip_efficiency=Config.BATTERY_EFFICIENCY,
        cycle_cost_per_kwh=Config.BATTERY_CYCLE_COST,
        min_soc=Config.BATTERY_MIN_SOC, max_soc=Config.BATTERY_MAX_SOC,
        tariff_night=Config.TARIFF_NIGHT, tariff_half_peak=Config.TARIFF_HALF_PEAK,
        tariff_peak=Config.TARIFF_PEAK,
        demand_charge_rub_per_kw_month=Config.DEMAND_CHARGE_RUB_PER_KW_MONTH,
        annual_om_share=Config.BATTERY_OM_SHARE,
        battery_cost_rub=Config.BATTERY_COST_RUB,
    )
    plot_storage_result(storage_result, sample_forecast, plots_dir=Config.PLOTS_DIR)
    compare_strategies(
        sample_forecast,
        capacity=Config.BATTERY_CAPACITY, max_power=Config.BATTERY_MAX_POWER,
        round_trip_efficiency=Config.BATTERY_EFFICIENCY,
        cycle_cost_per_kwh=Config.BATTERY_CYCLE_COST,
        battery_cost_rub=Config.BATTERY_COST_RUB,
        tariff_night=Config.TARIFF_NIGHT, tariff_half_peak=Config.TARIFF_HALF_PEAK,
        tariff_peak=Config.TARIFF_PEAK,
        demand_charge_rub_per_kw_month=Config.DEMAND_CHARGE_RUB_PER_KW_MONTH,
        annual_om_share=Config.BATTERY_OM_SHARE,
    )

    # Экспорт
    best_keras = next((t for t in trainers_all
                       if t.model_name == best_name and isinstance(getattr(t,"model",None), tf.keras.Model)), None)
    if best_keras is None:
        best_keras = next((t for t in trainers if isinstance(getattr(t,"model",None), tf.keras.Model)), None)
    if best_keras is not None:
        export_model_bundle(
            best_keras.model, scaler,
            {"HISTORY_LENGTH": Config.HISTORY_LENGTH, "FORECAST_HORIZON": Config.FORECAST_HORIZON,
             "N_FEATURES": Config.N_FEATURES, "model_name": best_keras.model_name,
             "lag_feature_start_idx": lag_idx, "seasonal_diff": True},
            export_dir=Config.MODELS_DIR, model_name=best_keras.model_name,
        )

    logger.info("\n" + "=" * 70)
    logger.info("✅ Пайплайн завершён!")
    logger.info("Итог (sorted by composite_score):")
    for name, m in sorted(all_metrics.items(), key=lambda kv: kv[1].get("composite_score", 999)):
        logger.info("  %-24s MAE=%7.2f R²=%.4f ACF24=%.3f Composite=%.3f",
                    name, m["MAE"], m["R2"], m.get("ACF_24",float("nan")), m.get("composite_score",float("nan")))
    logger.info("📁 %s", Config.OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()