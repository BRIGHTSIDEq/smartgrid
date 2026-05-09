# -*- coding: utf-8 -*-
"""
main.py — Smart Grid v18.

ИЗМЕНЕНИЯ v18 (академические дополнения для курсовой):
  + [5a] SARIMA(1,1,1)(1,1,1,24) как статистическая базовая линия
  + [6a] Анализ важности признаков XGBoost (feature_importances_)
  + [6b] График сравнения HPO до/после
  + [6c] График SARIMA vs нейросети
  + [9b] MC Dropout с графиком доверительных интервалов (не только лог)
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
)
from models.baseline import (
    build_linear_regression, build_xgboost, build_sarima,
)
from models.trainer import ModelTrainer, WeightedEnsemble, compare_trainers

from optimization.storage import simulate_storage, compare_strategies
from utils.visualization import (
    plot_training_history, plot_predictions_comparison,
    plot_metrics_comparison, plot_storage_result, plot_scientific_diagnostics,
)
from utils.academic_plots import (
    plot_mc_dropout_uncertainty,
    compute_mc_predictions,
    plot_feature_importance_xgboost,
    plot_hpo_comparison,
    plot_sarima_vs_neural,
)
from utils.deployment import export_model_bundle

logger = Config.setup_logging()


def _cleanup_plots_dir(plots_dir: str) -> None:
    keep_prefixes = (
        "01_", "02_", "03_", "04_", "05_",
        "training_", "predictions_comparison", "metrics_comparison",
        "scientific_diagnostics_", "residuals_", "backtesting_",
        "storage_optimization", "attention_summary_", "head_specialization_",
        # v18 новые графики
        "mc_dropout_uncertainty_",
        "feature_importance_xgboost",
        "hpo_comparison",
        "sarima_vs_neural",
    )
    for p in Path(plots_dir).glob("*"):
        if p.suffix.lower() != ".png":
            p.unlink(missing_ok=True)
            continue
        if not any(p.name.startswith(pfx) for pfx in keep_prefixes):
            p.unlink(missing_ok=True)


def main():
    logger.info("=" * 70)
    logger.info("  SMART GRID v18 — + SARIMA + FeatureImportance + MCDropout + HPO")
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
            f"Удалите кэш: {Config.GENERATED_DATA_CSV}"
        )
    logger.info(
        "Data snapshot | rows=%d | consumption mean=%.2f std=%.2f",
        len(df), float(df["consumption"].mean()), float(df["consumption"].std()),
    )
    Config.print_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # [2/10] EDA
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[2/10] EDA...")
    run_eda(df, plots_dir=Config.PLOTS_DIR, save=True)

    # ─────────────────────────────────────────────────────────────────────────
    # [3/10] Подготовка данных
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[3/10] Подготовка данных (history=%d)...", Config.HISTORY_LENGTH)
    data = prepare_data(
        df,
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        seasonal_diff=False,
    )
    x_shape = data["X_train"].shape
    assert x_shape[2] == Config.N_FEATURES
    validate_data_integrity(data)
    lag_idx    = data["lag_feature_start_idx"]
    naive_type = data.get("naive_type", "unknown")
    scaler     = data["scaler"]

    logger.info(
        "Split | train=%d val=%d test=%d | raw_train=%.0f raw_test=%.0f кВт·ч",
        len(data["X_train"]), len(data["X_val"]), len(data["X_test"]),
        float(np.mean(data["raw_train"])), float(np.mean(data["raw_test"])),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # [4/10] Инициализация моделей
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[4/10] Инициализация моделей...")

    _n_train    = max(len(data["X_train"]) // Config.BATCH_SIZE, 1)
    _total_steps = Config.EPOCHS * _n_train

    # LSTM
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

    # iTransformer
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

    # PatchTST
    if Config.HISTORY_LENGTH >= 192:
        patch_len, stride = 24, 12
    elif Config.HISTORY_LENGTH >= 96:
        patch_len, stride = 12, 6
    else:
        patch_len, stride = 8, 4

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
        patchtst_learning_rate=Config.PATCHTST_LEARNING_RATE,
        patchtst_dropout=Config.PATCHTST_DROPOUT,
    )

    # TFT-Lite
    _TFT_COVAR_INDICES = [1, 2, 3, 5, 6, 7, 15, 17, 23, 25]
    use_tft = False
    tft_lite = None
    try:
        tft_lite = build_tft_lite(
            history_length=Config.HISTORY_LENGTH,
            forecast_horizon=Config.FORECAST_HORIZON,
            d_model=min(Config.TRANSFORMER_D_MODEL, 128),
            num_heads=min(Config.TRANSFORMER_N_HEADS, 4),
            num_layers=min(Config.TRANSFORMER_N_LAYERS, 3),
            dropout=Config.TRANSFORMER_DROPOUT,
            learning_rate=Config.TRANSFORMER_LEARNING_RATE,
            n_covariate_features=len(_TFT_COVAR_INDICES),
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

    # ── [NEW v18] SARIMA ───────────────────────────────────────────────────────
    sarima_model = build_sarima(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24),
        refit=False,
    )

    logger.info(
        "Параметры: LSTM=%d | iTransformer=%d | PatchTST=%d",
        count_parameters(lstm), count_parameters(itransformer), count_parameters(patchtst),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # [5/10] Обучение нейросетей и ML
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[5/10] Обучение моделей...")

    models_to_train = [
        (lstm,         "LSTM",             Config.PATIENCE),
        (itransformer, "iTransformer",      Config.PATIENCE),
        (patchtst,     "PatchTST",          Config.PATCHTST_PATIENCE),
        (lr_model,     "LinearRegression",  Config.PATIENCE),
        (xgb_model,    "XGBoost",           Config.PATIENCE),
    ]
    if use_tft and tft_lite is not None:
        models_to_train.append((tft_lite, "TFT-Lite", Config.PATIENCE))

    trainers = []
    for model, name, patience in models_to_train:
        trainer = ModelTrainer(model, name, Config.MODELS_DIR, Config.PLOTS_DIR)
        train_data = (
            {**data, "X_train": data["X_tft_train"], "X_val": data["X_tft_val"]}
            if name == "TFT-Lite" else data
        )
        trainer.train(
            train_data,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            patience=patience,
            lr_patience=Config.LR_PATIENCE,
            lr_factor=Config.LR_FACTOR,
            min_delta=Config.MIN_DELTA,
        )
        trainers.append(trainer)
        if trainer.history is not None:
            plot_training_history(trainer.history, model_name=name,
                                  plots_dir=Config.PLOTS_DIR)

    # ── [NEW v18] Обучение SARIMA ─────────────────────────────────────────────
    logger.info("\n[5a/10] Обучение SARIMA(1,1,1)(1,1,1,24)...")
    sarima_trainer = ModelTrainer(sarima_model, "SARIMA",
                                  Config.MODELS_DIR, Config.PLOTS_DIR)
    try:
        # SARIMA обучается на сыром ряду, не на нормализованных окнах
        sarima_model.fit(
            data["X_train"], data["Y_train"],
            raw_series=data["raw_train"],
        )
        trainers.append(sarima_trainer)
        sarima_trained = True
        logger.info("SARIMA обучена успешно")
    except Exception as exc:
        logger.warning("SARIMA пропущена: %s", exc)
        sarima_trained = False

    # ─────────────────────────────────────────────────────────────────────────
    # [5b/10] WeightedEnsemble
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[5b/10] WeightedEnsemble...")
    # SARIMA в ансамбль не включаем — разные пространства предсказаний
    ensemble_trainers = [t for t in trainers
                         if t.model_name not in ("TFT-Lite", "SARIMA")]
    ensemble = WeightedEnsemble(ensemble_trainers, model_name="WeightedEnsemble")
    try:
        ensemble.optimize_weights(data, split="val")
    except Exception as exc:
        logger.warning("Ensemble оптимизация: %s", exc)
    trainers_all = trainers + [ensemble]

    # ─────────────────────────────────────────────────────────────────────────
    # [5c/10] Bias correction
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[5c/10] Bias correction...")
    y_val_true = inverse_scale(scaler, data["Y_val"])
    for t in trainers_all:
        if t.model_name in ("SARIMA", "WeightedEnsemble"):
            continue
        try:
            y_val_pred = t.predict_absolute(data, split="val")
            bias = float(np.mean(y_val_pred - y_val_true))
            t._bias_correction = bias
            logger.info("  %-24s bias=%+.1f кВт·ч", t.model_name, bias)
        except Exception as exc:
            logger.warning("  bias %s: %s", t.model_name, exc)

    # ─────────────────────────────────────────────────────────────────────────
    # [6/10] Сравнение моделей
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[6/10] Сравнение (метрики в кВт·ч)...")
    all_metrics, best_name = compare_trainers(
        trainers_all, data, split="test", select_by="composite"
    )
    plot_metrics_comparison(all_metrics, plots_dir=Config.PLOTS_DIR)

    # Собираем предсказания всех моделей
    predictions = {}
    y_true = inverse_scale(scaler, data["Y_test"])

    for t in trainers_all:
        try:
            if t.model_name == "TFT-Lite":
                raw = t.predict(data["X_tft_test"])
                predictions[t.model_name] = inverse_scale(scaler, raw)
            elif t.model_name == "SARIMA" and sarima_trained:
                # SARIMA предсказывает в raw кВт·ч напрямую
                raw_pred = sarima_model.predict(data["X_test"])
                predictions["SARIMA"] = raw_pred
            else:
                predictions[t.model_name] = t.predict_absolute(data, split="test")
        except Exception as exc:
            logger.warning("predict %s: %s", t.model_name, exc)

    if predictions:
        plot_predictions_comparison(
            y_true, predictions,
            n_steps=168,
            mode="single_origin",
            plots_dir=Config.PLOTS_DIR,
        )

    best_trainer = next(
        (t for t in trainers_all if t.model_name == best_name), trainers[0]
    )
    logger.info("🏆 Лучшая: %s | MAE=%.2f | composite=%.3f",
                best_name,
                all_metrics[best_name]["MAE"],
                all_metrics[best_name].get("composite_score", float("nan")))

    if best_name in predictions:
        plot_scientific_diagnostics(
            y_true=y_true,
            y_pred=predictions[best_name],
            model_name=best_name.replace(" ", "_"),
            plots_dir=Config.PLOTS_DIR,
        )

    # ── [NEW v18] SARIMA vs нейросети ─────────────────────────────────────────
    logger.info("\n[6a/10] График SARIMA vs нейросети...")
    if predictions:
        try:
            plot_sarima_vs_neural(
                y_true=y_true,
                predictions=predictions,
                n_steps=96,
                plots_dir=Config.PLOTS_DIR,
            )
        except Exception as exc:
            logger.warning("sarima_vs_neural: %s", exc)

    # ── [NEW v18] Feature importance XGBoost ──────────────────────────────────
    logger.info("\n[6b/10] Feature importance XGBoost...")
    xgb_trainer = next(
        (t for t in trainers if t.model_name == "XGBoost"), None
    )
    if xgb_trainer is not None:
        try:
            imp_dict = xgb_trainer.model.get_feature_importance(top_n=25)
            if imp_dict is not None:
                plot_feature_importance_xgboost(
                    imp_dict,
                    top_n=25,
                    plots_dir=Config.PLOTS_DIR,
                )
        except Exception as exc:
            logger.warning("Feature importance: %s", exc)

    # ── [NEW v18] HPO comparison ───────────────────────────────────────────────
    logger.info("\n[6c/10] HPO comparison до/после...")
    try:
        # all_metrics содержит результаты ПОСЛЕ HPO (текущий запуск)
        results_after_hpo = {
            name: {
                "MAE":  m["MAE"],
                "R2":   m["R2"],
                "MAPE": m.get("MAPE", m.get("sMAPE", 0.0)),
            }
            for name, m in all_metrics.items()
        }
        plot_hpo_comparison(
            results_after=results_after_hpo,
            plots_dir=Config.PLOTS_DIR,
        )
    except Exception as exc:
        logger.warning("HPO comparison: %s", exc)

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
        logger.warning("Attention пропущена: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # [8/10] Анализ остатков
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[8/10] Анализ остатков: %s", best_name)
    if best_name in predictions:
        best_pred = predictions[best_name]
    else:
        try:
            best_pred = best_trainer.predict_absolute(data, "test")
        except Exception:
            best_pred = y_true
    analyze_residuals(
        y_true, best_pred,
        model_name=best_name,
        plots_dir=Config.PLOTS_DIR,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # [9/10] Бэктестинг
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[9/10] Бэктестинг (LSTM)...")
    lstm_trainer = next(
        (t for t in trainers if t.model_name == "LSTM"), trainers[0]
    )
    run_backtesting(
        lstm_trainer.model, data,
        n_windows=8,
        plots_dir=Config.PLOTS_DIR,
        model_name="LSTM",
    )

    # ── [NEW v18] MC Dropout с ГРАФИКОМ ───────────────────────────────────────
    logger.info("\n[9b/10] MC Dropout (LSTM) — с графиком доверительных интервалов...")
    try:
        n_mc_samples = 500
        mc_mean_kwh, mc_std_kwh = compute_mc_predictions(
            model=lstm_trainer.model,
            X=data["X_test"][:n_mc_samples],
            scaler=scaler,
            n_samples=30,
        )
        y_true_mc = inverse_scale(scaler, data["Y_test"][:n_mc_samples])

        # График доверительных интервалов
        plot_mc_dropout_uncertainty(
            y_true=y_true_mc,
            mc_mean=mc_mean_kwh,
            mc_std=mc_std_kwh,
            model_name="LSTM",
            n_steps=72,
            plots_dir=Config.PLOTS_DIR,
        )

        avg_std = float(mc_std_kwh.mean())
        logger.info(
            "MC Dropout | mean uncertainty: ±%.1f кВт·ч (%.2f%% от ср.потребления)",
            avg_std,
            avg_std / float(np.mean(data["raw_test"])) * 100,
        )
    except Exception as exc:
        logger.warning("MC Dropout: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # [10/10] Оптимизация батареи BESS
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[10/10] Оптимизация накопителя...")
    sample_forecast = data["raw_test"][:Config.STORAGE_HORIZON]
    mean_test  = float(np.mean(data["raw_test"]))
    mean_train = float(np.mean(data["raw_train"]))
    if mean_test > mean_train * 1.15:
        logger.warning(
            "Тест-период: повышенное потребление (mean=%.0f vs train=%.0f кВт·ч). "
            "Срок окупаемости занижен — для реалистичной оценки используйте годовой прогноз.",
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

    # Экспорт лучшей модели
    best_keras = next(
        (t for t in trainers_all
         if t.model_name == best_name
         and isinstance(getattr(t, "model", None), tf.keras.Model)),
        None,
    )
    if best_keras is None:
        best_keras = next(
            (t for t in trainers
             if isinstance(getattr(t, "model", None), tf.keras.Model)),
            None,
        )
    if best_keras is not None:
        export_model_bundle(
            best_keras.model, scaler,
            {
                "HISTORY_LENGTH":    Config.HISTORY_LENGTH,
                "FORECAST_HORIZON":  Config.FORECAST_HORIZON,
                "N_FEATURES":        Config.N_FEATURES,
                "model_name":        best_keras.model_name,
                "lag_feature_start_idx": lag_idx,
                "seasonal_diff":     False,
                "naive_type":        naive_type,
                "bias_correction":   getattr(best_keras, "_bias_correction", None),
            },
            export_dir=Config.MODELS_DIR,
            model_name=best_keras.model_name,
        )

    _cleanup_plots_dir(Config.PLOTS_DIR)

    # ── Итоговый отчёт ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(" Пайплайн v18 завершён!")
    logger.info("Итог (sorted by composite_score):")
    for name, m in sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("composite_score", 999)
    ):
        logger.info(
            "  %-24s MAE=%7.2f R²=%.4f ACF24=%.3f Composite=%.3f",
            name, m["MAE"], m["R2"],
            m.get("ACF_24", float("nan")),
            m.get("composite_score", float("nan")),
        )

    logger.info("\nНовые графики v18:")
    logger.info("  results/plots/sarima_vs_neural.png")
    logger.info("  results/plots/feature_importance_xgboost.png")
    logger.info("  results/plots/hpo_comparison.png")
    logger.info("  results/plots/mc_dropout_uncertainty_LSTM.png")
    logger.info(" %s", Config.OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()