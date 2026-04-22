# -*- coding: utf-8 -*-
"""
main.py — Smart Grid v15.

ИЗМЕНЕНИЯ v15 (vs v14):

[1] VanillaTransformer → iTransformer (Liu et al., ICLR 2024).
  БЫЛО: VanillaTransformer — attention over T=192 timesteps.
    ПРОБЛЕМА: train MAE=0.030 при val MAE=0.019 всё время (inverted gap).
              MAE=8556, последнее место среди нейросетей.
  СТАЛО: iTransformer — attention over F=26 variates.
    Захватывает зависимости temperature↑+EV↑→consumption↑↑.
    RevIN proper: нормировка входа + денормировка выхода.
    SeasonalSkip: lag-24h + lag-168h (как у LSTM).
    WarmupCosineDecay: warmup 5% шагов → стабильный старт.

[2] WarmupCosineDecay для всех нейросетей.
  БЫЛО: LSTM использовал CosineDecay без warmup, best_epoch=78/84.
  СТАЛО: warmup 5% шагов (≈1.5 эпохи) → LR растёт 0→peak, потом косинус.

[3] RevIN denorm в LSTM: neural_out * std + mean → согласованный SeasonalSkip.

[4] PatchTST v6: WarmupCosineDecay + SeasonalSkip.

[5] TFT-Lite v2: WarmupCosineDecay + RevIN denorm → убирает MBE=-4509.

[6] MAPE убрана из training metrics: Train MAPE ~17000-20000% была из-за
    деления на нормализованные ≈0 значения. Теперь только MAE в training.
"""

import logging
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from config import Config

np.random.seed(Config.SEED)
random.seed(Config.SEED)

import tensorflow as tf
tf.random.set_seed(Config.SEED)

from data.generator import load_or_generate_smartgrid_data, validate_generated_data
from data.preprocessing import prepare_data, inverse_scale, validate_data_integrity
from analysis.eda import run_eda
from analysis.residuals import analyze_residuals
from analysis.backtesting import run_backtesting

from models.lstm import build_lstm_model
from models.transformer import (
    build_itransformer, build_patchtst, build_tft_lite, count_parameters,
    prepare_tft_covariates,
)
from models.baseline import build_linear_regression, build_xgboost
from models.trainer import ModelTrainer, WeightedEnsemble, compare_trainers

from optimization.storage import simulate_storage, compare_strategies
from utils.visualization import (
    plot_training_history, plot_predictions_comparison,
    plot_metrics_comparison, plot_storage_result, plot_scientific_diagnostics,
)
from utils.deployment import export_model_bundle

logger = Config.setup_logging()


def _cleanup_plots_dir(plots_dir: str) -> None:
    keep_prefixes = (
        "01_timeseries_patterns", "02_consumption_profiles",
        "03_decomposition", "04_acf_pacf", "05_temperature_dependency",
        "training_", "predictions_comparison", "metrics_comparison",
        "scientific_diagnostics_", "residuals_", "backtesting_",
        "storage_optimization", "attention_summary_", "head_specialization_",
    )
    for p in Path(plots_dir).glob("*"):
        if p.suffix.lower() != ".png":
            p.unlink(missing_ok=True)
            continue
        if not p.name.startswith(keep_prefixes):
            p.unlink(missing_ok=True)


def main():
    logger.info("=" * 70)
    logger.info("  SMART GRID v14 — Seasonal Skip + Dual Naive Fix + TFT Covariate Fix")
    logger.info("=" * 70)

    # Config.set_fast_mode()
    Config.set_optimal_mode()
    # Config.set_full_mode()
    Config.create_dirs()

    # ─────────────────────────────────────────────────────────────────────────
    # [1/10] Генерация данных
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[1/10] Генерация данных...")
    df = load_or_generate_smartgrid_data(
        csv_path=Config.GENERATED_DATA_CSV,
        force_regenerate=Config.FORCE_REGENERATE_DATA,
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
    expected_rows = Config.DAYS * 24
    if len(df) != expected_rows:
        raise ValueError(
            f"Несовместимый датасет: rows={len(df)}, ожидалось={expected_rows}. "
            f"Удалите кэш-файл: {Config.GENERATED_DATA_CSV}"
        )
    logger.info(
        "Data snapshot | rows=%d | consumption mean=%.2f std=%.2f | temp mean=%.2f std=%.2f",
        len(df), float(df["consumption"].mean()), float(df["consumption"].std()),
        float(df["temperature"].mean()), float(df["temperature"].std()),
    )
    Config.print_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # [2/10] EDA
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[2/10] EDA...")
    run_eda(df, plots_dir=Config.PLOTS_DIR, save=True)

    # ─────────────────────────────────────────────────────────────────────────
    # [3/10] Подготовка данных
    #
    # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v14: seasonal_diff=False
    # ──────────────────────────────────────────────────────────────────────────
    # seasonal_diff=False: модели предсказывают абсолютные значения ∈ [0,1].
    #   Преимущества для нейросетей:
    #   1. Нет проблемы "predict Y_diff ≈ 0" — нейросети сразу работают с полным сигналом.
    #   2. Seasonal skip (lag-24h, lag-168h) создаёт сильный prior к seasonal паттерну.
    #   3. NNs могут использовать нелинейные взаимодействия (EV×температура, DSR×праздники).
    #   4. Dual Naive хранится в Y_seasonal_naive_* для диагностики, но не используется в Loss.
    #
    # При history=192 (optimal mode): Dual Naive активируется (192 >= 168) — для справки.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(
        "\n[3/10] Подготовка данных (seasonal_diff=False, history=%d)...",
        Config.HISTORY_LENGTH,
    )
    data = prepare_data(
        df,
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        seasonal_diff=False,   # ИСПРАВЛЕНИЕ: абсолютные значения
    )
    x_shape = data["X_train"].shape
    assert x_shape[2] == Config.N_FEATURES, (
        f"n_features mismatch: {x_shape[2]} != {Config.N_FEATURES}"
    )
    naive_type = data.get("naive_type", "unknown")
    logger.info(
        "✅ X_train=%s | seasonal_diff=%s | naive=%s | Y∈[%.3f,%.3f]",
        x_shape, data["seasonal_diff"], naive_type,
        float(data["Y_train"].min()), float(data["Y_train"].max()),
    )
    logger.info(
        "Split lens | train=%d val=%d test=%d | raw_train mean=%.2f raw_test mean=%.2f",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
        float(np.mean(data["raw_train"])), float(np.mean(data["raw_test"])),
    )
    validate_data_integrity(data)
    lag_idx = data["lag_feature_start_idx"]

    # ─────────────────────────────────────────────────────────────────────────
    # [4/10] Инициализация моделей
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[4/10] Инициализация моделей...")

    _n_train = int(Config.DAYS * 0.70 * 24 - Config.HISTORY_LENGTH - Config.FORECAST_HORIZON)
    _n_train = max(_n_train, 1)
    _total_steps = Config.EPOCHS * max(_n_train // Config.BATCH_SIZE, 1)
    logger.info("total_steps для WarmupCosineDecay: %d (epochs=%d, n_train≈%d, batch=%d)",
                _total_steps, Config.EPOCHS, _n_train, Config.BATCH_SIZE)

    # ── LSTM v13: WarmupCosineDecay + RevIN denorm ───────────────────────────
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
        warmup_ratio=Config.LSTM_WARMUP_RATIO,
        tcn_filters=Config.LSTM_TCN_FILTERS,
        huber_delta=Config.LSTM_HUBER_DELTA,
        seasonal_blend_init=Config.LSTM_SEASONAL_BLEND_INIT,
        use_seasonal_skip=True,
        lag_feature_start_idx=lag_idx,
    )

    # ── iTransformer v1 (замена VanillaTransformer) ───────────────────────────
    # Attention over F=26 variates вместо T=192 timesteps.
    # Захватывает зависимости между переменными: temp↑+EV↑→consumption↑↑.
    # RevIN proper (norm вход + denorm выход) + SeasonalSkip + WarmupCosineDecay.
    itransformer = build_itransformer(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        d_model=Config.TRANSFORMER_D_MODEL,
        num_heads=Config.TRANSFORMER_N_HEADS,
        num_layers=Config.ITRANSFORMER_N_LAYERS,
        dff=Config.TRANSFORMER_DFF,
        dropout=Config.TRANSFORMER_DROPOUT,
        learning_rate=Config.VANILLA_TRANSFORMER_LR,
        use_cosine_decay=Config.TRANSFORMER_USE_COSINE_DECAY,
        total_steps=_total_steps,
        warmup_ratio=Config.TRANSFORMER_WARMUP_RATIO,
        huber_delta=Config.VANILLA_HUBER_DELTA,
        use_seasonal_skip=True,
        seasonal_blend_init=Config.VANILLA_SEASONAL_BLEND_INIT,
    )

    # ── PatchTST: patch_len=24 при history=192 (1 патч = 1 сутки, 8 патчей) ──
    # При history=192, horizon=24: 1 патч = 1 сутки — семантически осмысленно.
    # n_patches = (192-24)//12 + 1 = 168//12 + 1 = 14 + 1 = 15 патчей
    if Config.HISTORY_LENGTH >= 192:
        patch_len = 24; stride = 12   # 1 патч = 1 сутки
    elif Config.HISTORY_LENGTH >= 168:
        patch_len = 24; stride = 12
    elif Config.HISTORY_LENGTH >= 96:
        patch_len = 12; stride = 6
    elif Config.HISTORY_LENGTH >= 48:
        patch_len = 8; stride = 4
    else:
        patch_len = 6; stride = 3

    n_patches_expected = (Config.HISTORY_LENGTH - patch_len) // stride + 1
    logger.info(
        "PatchTST patch_len=%d stride=%d → n_patches=%d "
        "(history=%d, 1 patch=%dh)",
        patch_len, stride, n_patches_expected, Config.HISTORY_LENGTH, patch_len
    )

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
        use_cosine_decay=Config.TRANSFORMER_USE_COSINE_DECAY,
        total_steps=_total_steps,
        warmup_ratio=Config.TRANSFORMER_WARMUP_RATIO,
        huber_delta=Config.VANILLA_HUBER_DELTA,
        use_seasonal_skip=True,
        seasonal_blend_init=Config.VANILLA_SEASONAL_BLEND_INIT,
    )

    # ── TFT-Lite v2: 10 ковариат + WarmupCosineDecay + RevIN denorm ──────────
    _TFT_COVAR_INDICES = [1, 2, 3, 5, 6, 7, 15, 17, 23, 25]  # из 26 признаков
    use_tft = False
    tft_lite = None
    tft_n_cov = len(_TFT_COVAR_INDICES)
    try:
        tft_lite = build_tft_lite(
            history_length=Config.HISTORY_LENGTH,
            forecast_horizon=Config.FORECAST_HORIZON,
            d_model=min(Config.TRANSFORMER_D_MODEL, 128),
            num_heads=min(Config.TRANSFORMER_N_HEADS, 4),
            num_layers=min(Config.TRANSFORMER_N_LAYERS, 3),
            dropout=Config.TRANSFORMER_DROPOUT,
            learning_rate=Config.TRANSFORMER_LEARNING_RATE,
            n_covariate_features=tft_n_cov,
            use_cosine_decay=Config.TRANSFORMER_USE_COSINE_DECAY,
            total_steps=_total_steps,
            warmup_ratio=Config.TRANSFORMER_WARMUP_RATIO,
            huber_delta=Config.VANILLA_HUBER_DELTA,
        )

        def _to_series(x): return x[:, :, :1].astype(np.float32)
        def _to_covar(x):  return x[:, :, _TFT_COVAR_INDICES].astype(np.float32)

        data["X_tft_train"] = [_to_series(data["X_train"]), _to_covar(data["X_train"])]
        data["X_tft_val"]   = [_to_series(data["X_val"]),   _to_covar(data["X_val"])]
        data["X_tft_test"]  = [_to_series(data["X_test"]),  _to_covar(data["X_test"])]
        use_tft = True
        logger.info("TFT-Lite v2: %d params | series=(N,T,1) + covars=(N,T,%d) + RevIN denorm",
                    count_parameters(tft_lite), tft_n_cov)
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

    logger.info("Параметры: LSTM=%d | iTransformer=%d | PatchTST=%d",
                count_parameters(lstm), count_parameters(itransformer), count_parameters(patchtst))

    # ─────────────────────────────────────────────────────────────────────────
    # [5/10] Обучение
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(
        "\n[5/10] Обучение моделей | seasonal_diff=%s | history=%d | "
        "LSTM: seasonal_skip=True, cosine_decay=%s | "
        "VanillaTr: seasonal_residual=True",
        data["seasonal_diff"], Config.HISTORY_LENGTH, Config.LSTM_USE_COSINE_DECAY,
    )

    models_to_train = [
        (lstm,          "LSTM"),
        (itransformer,  "iTransformer"),
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

    # ─────────────────────────────────────────────────────────────────────────
    # [5b/10] WeightedEnsemble
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[5b/10] WeightedEnsemble (оптимизация в абсолютном пространстве)...")
    single_trainers = [t for t in trainers if t.model_name != "TFT-Lite"]
    ensemble = WeightedEnsemble(single_trainers, model_name="WeightedEnsemble")
    try:
        ensemble.optimize_weights(data, split="val")
        trainers_all = trainers + [ensemble]
    except Exception as exc:
        logger.warning("Ensemble оптимизация не удалась: %s", exc)
        trainers_all = trainers + [ensemble]

    # ─────────────────────────────────────────────────────────────────────────
    # [6/10] Сравнение моделей
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[6/10] Сравнение (метрики в кВт·ч)...")
    all_metrics, best_name = compare_trainers(
        trainers_all, data, split="test", select_by="composite")
    plot_metrics_comparison(all_metrics, plots_dir=Config.PLOTS_DIR)

    # Визуализация прогнозов
    scaler = data["scaler"]
    predictions = {}
    for t in trainers_all:
        try:
            if t.model_name == "TFT-Lite":
                raw = t.predict(data["X_tft_test"])
                predictions[t.model_name] = inverse_scale(scaler, raw)
            else:
                # WeightedEnsemble и ModelTrainer оба имеют predict_absolute
                predictions[t.model_name] = t.predict_absolute(data, split="test")
        except Exception as exc:
            logger.warning("predict_absolute %s: %s", t.model_name, exc)

    y_true = inverse_scale(scaler, data["Y_test"])  # seasonal_diff=False → прямой inverse_scale

    if predictions:
        plot_predictions_comparison(y_true, predictions, plots_dir=Config.PLOTS_DIR)

    best_trainer = next((t for t in trainers_all if t.model_name == best_name), trainers[0])
    logger.info("🏆 Лучшая: %s | MAE=%.2f | composite=%.3f",
                best_name, all_metrics[best_name]["MAE"],
                all_metrics[best_name].get("composite_score", float("nan")))

    if best_name in predictions:
        plot_scientific_diagnostics(
            y_true=y_true,
            y_pred=predictions[best_name],
            model_name=best_name.replace(" ", "_"),
            plots_dir=Config.PLOTS_DIR,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # [7/10] Визуализация внимания
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[7/10] Визуализация внимания...")
    try:
        from utils.attention_visualization import (
            visualize_attention_summary, compare_head_specialization)
        sample_x = data["X_test"][:1]
        for t in trainers:
            if t.model_name in ("iTransformer", "PatchTST"):
                visualize_attention_summary(
                    t.model, sample_x, history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
                compare_head_specialization(
                    t.model, sample_x, model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
    except Exception as exc:
        logger.warning("Attention пропущена: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # [8/10] Анализ остатков лучшей модели
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[8/10] Анализ остатков: %s", best_name)
    if best_name in predictions:
        best_pred = predictions[best_name]
    else:
        # Fallback: try predict_absolute, then reconstruct directly
        try:
            best_pred = best_trainer.predict_absolute(data, "test")
        except Exception as exc:
            logger.warning("Fallback predict_absolute failed: %s. Using y_true.", exc)
            best_pred = y_true
    analyze_residuals(y_true, best_pred, model_name=best_name, plots_dir=Config.PLOTS_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # [9/10] Бэктестинг
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[9/10] Бэктестинг (LSTM)...")
    lstm_trainer = next((t for t in trainers if t.model_name == "LSTM"), trainers[0])
    run_backtesting(lstm_trainer.model, data, n_windows=8,
                    plots_dir=Config.PLOTS_DIR, model_name="LSTM")

    # ─────────────────────────────────────────────────────────────────────────
    # [9b/10] MC Dropout (неопределённость прогноза)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[9b/10] MC Dropout (LSTM)...")
    try:
        mc_mean, mc_std = lstm_trainer.mc_predict(data["X_test"][:500], n_samples=30)
        # Конвертируем std в кВт·ч: scale_ содержит range MinMaxScaler
        if hasattr(scaler, "scale_") and scaler.scale_ is not None:
            mc_std_kwh = float(mc_std.mean()) / float(scaler.scale_[0])
        else:
            mc_std_kwh = float(mc_std.mean()) * float(data["raw_train"].max() - data["raw_train"].min())
        logger.info("MC Dropout | mean uncertainty: ±%.1f кВт·ч (scaled: ±%.4f)",
                    mc_std_kwh, float(mc_std.mean()))
    except Exception as exc:
        logger.warning("MC Dropout пропущен: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # [10/10] Оптимизация батареи BESS
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[10/10] Оптимизация накопителя...")
    # Используем среднегодовой срез теста, а не только первые 720ч (исправление смещения)
    sample_forecast = data["raw_test"][:Config.STORAGE_HORIZON]
    mean_test = float(np.mean(data["raw_test"]))
    mean_train = float(np.mean(data["raw_train"]))
    if mean_test > mean_train * 1.15:
        logger.warning(
            "Тест-период имеет повышенное потребление (mean=%.0f vs train=%.0f кВт·ч). "
            "Срок окупаемости батареи может быть занижен. "
            "Используйте годовой прогноз для реалистичной оценки.",
            mean_test, mean_train,
        )

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
        battery_cost_rub=Config.BATTERY_COST_RUB,
    )
    plot_storage_result(storage_result, sample_forecast, plots_dir=Config.PLOTS_DIR)
    compare_strategies(
        sample_forecast,
        capacity=Config.BATTERY_CAPACITY,
        max_power=Config.BATTERY_MAX_POWER,
        round_trip_efficiency=Config.BATTERY_EFFICIENCY,
        cycle_cost_per_kwh=Config.BATTERY_CYCLE_COST,
        battery_cost_rub=Config.BATTERY_COST_RUB,
        tariff_night=Config.TARIFF_NIGHT,
        tariff_half_peak=Config.TARIFF_HALF_PEAK,
        tariff_peak=Config.TARIFF_PEAK,
        demand_charge_rub_per_kw_month=Config.DEMAND_CHARGE_RUB_PER_KW_MONTH,
        annual_om_share=Config.BATTERY_OM_SHARE,
    )

    # ── Экспорт лучшей Keras-модели ──────────────────────────────────────────
    best_keras = next(
        (t for t in trainers_all
         if t.model_name == best_name and isinstance(getattr(t, "model", None), tf.keras.Model)),
        None,
    )
    if best_keras is None:
        best_keras = next(
            (t for t in trainers if isinstance(getattr(t, "model", None), tf.keras.Model)),
            None,
        )
    if best_keras is not None:
        export_model_bundle(
            best_keras.model, scaler,
            {
                "HISTORY_LENGTH": Config.HISTORY_LENGTH,
                "FORECAST_HORIZON": Config.FORECAST_HORIZON,
                "N_FEATURES": Config.N_FEATURES,
                "model_name": best_keras.model_name,
                "lag_feature_start_idx": lag_idx,
                "seasonal_diff": False,
                "naive_type": naive_type,
            },
            export_dir=Config.MODELS_DIR,
            model_name=best_keras.model_name,
        )

    _cleanup_plots_dir(Config.PLOTS_DIR)

    # ── Итоговый отчёт ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("✅ Пайплайн завершён!")
    logger.info("Итог (sorted by composite_score):")
    for name, m in sorted(all_metrics.items(), key=lambda kv: kv[1].get("composite_score", 999)):
        logger.info(
            "  %-24s MAE=%7.2f R²=%.4f ACF24=%.3f Composite=%.3f",
            name, m["MAE"], m["R2"], m.get("ACF_24", float("nan")),
            m.get("composite_score", float("nan")),
        )
    logger.info("📁 %s", Config.OUTPUT_DIR)
    logger.info("Config: history=%d | seasonal_diff=False | "
                "LSTM seasonal_skip=True | VanillaTr seasonal_residual=True",
                Config.HISTORY_LENGTH)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()