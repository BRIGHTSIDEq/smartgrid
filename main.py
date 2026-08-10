# -*- coding: utf-8 -*-
"""
main.py — Точка входа пайплайна Smart Grid.

Полный цикл: генерация данных → EDA → подготовка → обучение моделей →
сравнение → диагностика → бэктестинг → оптимизация накопителя → экспорт
результатов.

Запуск:
    python main.py --mode optimal
    python main.py --mode smoke                    # проверка работоспособности
    python main.py --mode fast --seed 0            # другой сид
    python main.py --mode fast --models naive,ridge,xgboost   # подмножество
    python main.py --mode optimal --rolling-origin 4          # walk-forward

МЕТОДИЧЕСКИЕ ПРИНЦИПЫ, заложенные в порядок шагов:

  * Лучшая модель выбирается по ВАЛИДАЦИОННОЙ выборке. Тест используется один
    раз — для итоговой таблицы. Выбор победителя по тесту превратил бы тестовую
    оценку в оптимистично смещённую.
  * В сравнение всегда включаются наивные базлайны, а ключевой метрикой служит
    MASE. Модель, не превзошедшая «повторить прошлые сутки», прогностической
    ценности не имеет независимо от абсолютного MAE.
  * Накопитель управляется по ПРОГНОЗУ модели, а счёт выставляется по ФАКТУ.
    Дополнительно считается идеальный прогноз — разница показывает денежную
    цену ошибки прогнозирования.
"""

import argparse
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from config import Config

# ── Каталог доступных моделей: ключ CLI → человекочитаемое имя ───────────────
MODEL_REGISTRY = {
    "naive24":     "Naive24 (сутки)",
    "naive168":    "Naive168 (неделя)",
    "profile":     "HourlyProfile",
    "ridge":       "LinearRegression",
    "xgboost":     "XGBoost",
    "lstm":        "LSTM",
    "transformer": "VanillaTransformer",
    "patchtst":    "PatchTST",
}
NAIVE_KEYS = ("naive24", "naive168", "profile")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогнозирование энергопотребления Smart Grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "fast", "optimal", "full",
                 "panel-smoke", "panel-fast", "panel-optimal"],
        default="optimal",
        help="Режим прогона. Режимы panel-* обучают одну глобальную модель "
             "на множестве рядов (города и фидеры), остальные — на одном "
             "агрегатном ряде города (см. config.py)")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Сид генератора случайных чисел")
    parser.add_argument("--dataset", choices=["synthetic", "uci"], default="synthetic",
                        help="Источник данных для panel-режимов: synthetic — "
                             "собственный генератор, uci — реальные измерения "
                             "ElectricityLoadDiagrams20112014 (370 клиентов "
                             "Португалии). Внешний набор проверяет выводы на "
                             "данных, которых проект не создавал")
    parser.add_argument("--uci-path", type=str, default="data/raw/LD2011_2014.txt",
                        metavar="ПУТЬ",
                        help="Файл набора UCI. Не скачивается автоматически")
    parser.add_argument("--models", type=str, default="all",
                        help="Список моделей через запятую: "
                             + ",".join(MODEL_REGISTRY) + " либо all")
    parser.add_argument("--scenario", choices=["current", "forward"], default="current",
                        help="Сценарий проникновения технологий Smart Grid: "
                             "current — фактическое состояние на 2025 год, "
                             "forward — перспективный сценарий с заметной долей "
                             "электротранспорта и микрогенерации")
    parser.add_argument("--probabilistic", action="store_true",
                        help="Добавить вероятностные модели (квантили P10/P50/P90, "
                             "pinball, покрытие, калибровка) в panel-режимах")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Пропустить исследовательский анализ данных")
    parser.add_argument("--skip-storage", action="store_true",
                        help="Пропустить блок оптимизации накопителя")
    parser.add_argument("--skip-attention", action="store_true",
                        help="Пропустить визуализацию весов внимания")
    parser.add_argument("--rolling-origin", type=int, default=0, metavar="N",
                        help="Число origin-ов walk-forward бэктестинга "
                             "(0 = не запускать; процедура переобучает модель N раз)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Включить детерминированные операции TensorFlow "
                             "(медленнее, но результат воспроизводим побитово)")
    return parser.parse_args(argv)


def setup_environment(args: argparse.Namespace) -> Any:
    """Применяет режим, фиксирует сиды и настраивает логирование."""
    logger = Config.setup_logging()

    mode_setters = {
        "smoke": Config.set_smoke_mode,
        "fast": Config.set_fast_mode,
        "optimal": Config.set_optimal_mode,
        "full": Config.set_full_mode,
        "panel-smoke": Config.set_panel_smoke_mode,
        "panel-fast": Config.set_panel_fast_mode,
        "panel-optimal": Config.set_panel_optimal_mode,
    }
    mode_setters[args.mode]()
    # Сценарий и производные экономические параметры применяются ПОСЛЕ режима:
    # режим задаёт объём данных и ёмкость моделей, сценарий — состав нагрузки.
    Config.finalize(scenario=args.scenario)
    Config.SEED = args.seed
    Config.create_dirs()

    # Сиды фиксируются до создания любых моделей.
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    import tensorflow as tf
    tf.random.set_seed(args.seed)
    if args.deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            logger.info("Детерминированный режим TensorFlow включён.")
        except Exception as exc:
            logger.warning("Не удалось включить детерминизм TF: %s", exc)

    return logger


def resolve_models(spec: str) -> List[str]:
    """Разбирает значение --models в список ключей реестра."""
    if spec.strip().lower() == "all":
        return list(MODEL_REGISTRY)
    keys = [k.strip().lower() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODEL_REGISTRY]
    if unknown:
        raise SystemExit(
            f"Неизвестные модели: {', '.join(unknown)}. "
            f"Доступны: {', '.join(MODEL_REGISTRY)}"
        )
    return keys


def build_models(keys: List[str], logger, n_train_windows: int = 0) -> List[tuple]:
    """
    Создаёт экземпляры выбранных моделей. Возвращает [(модель, имя), ...].

    n_train_windows нужен для расписания learning rate: число шагов
    рассчитывается по фактическому размеру обучающей выборки, иначе косинусное
    затухание не попадёт в конец обучения.
    """
    from models.lstm import build_lstm_model
    from models.transformer import (
        build_vanilla_transformer, build_patchtst, count_parameters,
    )
    from models.baseline import (
        build_linear_regression, build_xgboost,
        build_persistence_24, build_seasonal_naive_168, build_hourly_profile,
    )

    # Полное число шагов оптимизатора: эпохи × батчей в эпохе.
    steps_per_epoch = max(int(np.ceil(n_train_windows / max(Config.BATCH_SIZE, 1))), 1)
    total_steps = Config.EPOCHS * steps_per_epoch if n_train_windows else 0
    lr_steps = total_steps if Config.TRANSFORMER_USE_WARMUP_COSINE else 0

    built: List[tuple] = []

    def add(key, factory):
        if key in keys:
            built.append((factory(), MODEL_REGISTRY[key]))

    # Наивные базлайны идут первыми: они задают точку отсчёта для всех остальных.
    add("naive24", build_persistence_24)
    # Недельный базлайн требует, чтобы окно истории покрывало целую неделю,
    # иначе он вырождается в суточный и дублирует предыдущую строку таблицы.
    if "naive168" in keys:
        if Config.HISTORY_LENGTH >= 168:
            add("naive168", build_seasonal_naive_168)
        else:
            logger.info(
                "Naive168 пропущен: HISTORY_LENGTH=%d < 168 ч, недельный лаг "
                "не попадает в окно истории (совпал бы с Naive24).",
                Config.HISTORY_LENGTH,
            )
    add("profile", build_hourly_profile)

    add("ridge", build_linear_regression)
    add("xgboost", lambda: build_xgboost(
        n_estimators=Config.XGB_N_ESTIMATORS,
        max_depth=Config.XGB_MAX_DEPTH,
        learning_rate=Config.XGB_LR,
        subsample=Config.XGB_SUBSAMPLE,
        colsample_bytree=Config.XGB_COLSAMPLE,
        seed=Config.SEED,
    ))

    add("lstm", lambda: build_lstm_model(
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
        tcn_filters=Config.LSTM_TCN_FILTERS,
        huber_delta=Config.LSTM_HUBER_DELTA,
        seasonal_blend_init=Config.LSTM_SEASONAL_BLEND_INIT,
    ))

    add("transformer", lambda: build_vanilla_transformer(
        history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        d_model=Config.TRANSFORMER_D_MODEL,
        num_heads=Config.TRANSFORMER_N_HEADS,
        num_layers=Config.TRANSFORMER_N_LAYERS,
        dff=Config.TRANSFORMER_DFF,
        dropout=Config.TRANSFORMER_DROPOUT,
        # Отдельный, пониженный LR: при общем 3e-4 обучение сваливается
        # в плохой локальный минимум за несколько эпох.
        learning_rate=Config.VANILLA_TRANSFORMER_LR,
        pe_type="sinusoidal",
        stochastic_depth_rate=Config.TRANSFORMER_STOCHASTIC_DEPTH,
        lr_schedule_total_steps=lr_steps,
        lr_warmup_fraction=Config.TRANSFORMER_WARMUP_FRACTION,
        use_seasonal_residual=Config.VANILLA_USE_SEASONAL_RESIDUAL,
        seasonal_blend_init=Config.VANILLA_SEASONAL_BLEND_INIT,
        huber_delta=Config.VANILLA_HUBER_DELTA,
    ))

    if "patchtst" in keys:
        patch_len = _select_patch_len(Config.HISTORY_LENGTH)
        built.append((build_patchtst(
            history_length=Config.HISTORY_LENGTH,
            forecast_horizon=Config.FORECAST_HORIZON,
            patch_len=patch_len, stride=max(patch_len // 2, 1),
            n_features=Config.N_FEATURES,
            d_model=Config.TRANSFORMER_D_MODEL,
            num_heads=Config.TRANSFORMER_N_HEADS,
            num_layers=Config.TRANSFORMER_N_LAYERS,
            dff=Config.TRANSFORMER_DFF,
            dropout=Config.TRANSFORMER_DROPOUT,
            learning_rate=Config.TRANSFORMER_LEARNING_RATE,
            stochastic_depth_rate=Config.TRANSFORMER_STOCHASTIC_DEPTH,
            lr_schedule_total_steps=lr_steps,
            lr_warmup_fraction=Config.TRANSFORMER_WARMUP_FRACTION,
            use_revin=Config.PATCHTST_USE_REVIN,
        ), MODEL_REGISTRY["patchtst"]))

    import tensorflow as tf
    for model, name in built:
        if isinstance(model, tf.keras.Model):
            logger.info("Модель %-20s | параметров: %d", name, count_parameters(model))
    return built


def _select_patch_len(history_length: int) -> int:
    """Длина патча PatchTST подбирается под длину истории."""
    if history_length >= 192:
        return 16
    if history_length >= 96:
        return 12
    if history_length >= 48:
        return 8
    return 6


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_environment(args)

    # Многорядный конвейер отличается структурой данных, составом моделей и
    # набором метрик, поэтому вынесен отдельным модулем, а не ветвлением
    # внутри агрегатного пути.
    if args.mode.startswith("panel-"):
        from panel_pipeline import run_panel_pipeline
        logger.info("=" * 78)
        logger.info("  МНОГОРЯДНОЕ ПРОГНОЗИРОВАНИЕ — SMART GRID PANEL")
        logger.info("  режим=%s | сценарий=%s | сид=%d",
                    args.mode, args.scenario, args.seed)
        logger.info("=" * 78)
        return run_panel_pipeline(args, logger)

    t_start = time.time()

    import tensorflow as tf
    from data.generator import (
        generate_smartgrid_data, validate_generated_data, validate_against_reference,
    )
    from data.preprocessing import (
        prepare_data, inverse_scale, validate_data_integrity,
        reconstruct_day_ahead_series, actual_series_for_forecast,
    )
    from analysis.eda import run_eda
    from analysis.residuals import analyze_residuals
    from analysis.backtesting import analyze_stability_over_windows, run_rolling_origin_backtest
    from models.trainer import ModelTrainer, compare_trainers
    from optimization.storage import compare_strategies, compare_forecast_sources
    from utils.metrics import metrics_by_horizon, pairwise_dm_table
    from utils.visualization import (
        plot_training_history, plot_predictions_comparison, plot_metrics_comparison,
        plot_storage_result, plot_metrics_by_horizon, plot_accuracy_vs_cost,
        plot_forecast_value,
    )
    from utils.deployment import export_model_bundle
    from utils import reporting

    model_keys = resolve_models(args.models)

    # Каталог прогона создаётся ДО первого артефакта, и все выходные пути
    # перенаправляются внутрь него. Копирование общего каталога графиков после
    # прогона недопустимо: туда попадали изображения посторонних запусков —
    # в частности, графики моделей, которые в этом прогоне не обучались.
    run_dir = reporting.make_run_dir(Config.OUTPUT_DIR, args.mode, args.scenario, args.seed)
    Config.PLOTS_DIR = os.path.join(run_dir, "plots")
    Config.MODELS_DIR = os.path.join(run_dir, "models")
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    logger.info("=" * 78)
    logger.info("  СИСТЕМА ПРОГНОЗИРОВАНИЯ ЭНЕРГОПОТРЕБЛЕНИЯ — SMART GRID")
    logger.info("  режим=%s | сценарий=%s | сид=%d | модели: %s",
                args.mode, args.scenario, args.seed, ", ".join(model_keys))
    logger.info("=" * 78)

    # ── [1/10] Генерация данных ──────────────────────────────────────────────
    logger.info("\n[1/10] Генерация данных...")
    from config import RealWorldReference
    df = generate_smartgrid_data(
        days=Config.DAYS, households=Config.HOUSEHOLDS,
        start_date=Config.START_DATE, seed=Config.SEED,
        temp_setpoint=Config.GEN_TEMP_SETPOINT,
        cooling_setpoint=Config.GEN_COOLING_SETPOINT,
        heating_coef=Config.GEN_HEATING_COEF,
        cooling_coef=Config.GEN_COOLING_COEF,
        temp_annual_mean=Config.GEN_TEMP_ANNUAL_MEAN,
        temp_annual_amplitude=Config.GEN_TEMP_ANNUAL_AMPLITUDE,
        temp_min=Config.GEN_TEMP_MIN, temp_max=Config.GEN_TEMP_MAX,
        humidity_threshold=Config.GEN_HUMIDITY_THRESHOLD,
        humidity_coef=Config.GEN_HUMIDITY_COEF,
        wind_temp_threshold=Config.GEN_WIND_TEMP_THRESHOLD,
        wind_coef=Config.GEN_WIND_COEF,
        early_bird_frac=Config.GEN_EARLY_BIRD_FRAC,
        night_owl_frac=Config.GEN_NIGHT_OWL_FRAC,
        ar_phi=Config.GEN_AR_PHI, ar_sigma=Config.GEN_AR_SIGMA,
        seasonal_winter_boost=Config.GEN_SEASONAL_WINTER_BOOST,
        seasonal_summer_dip=Config.GEN_SEASONAL_SUMMER_DIP,
        kwh_per_household_month=Config.GEN_KWH_PER_HOUSEHOLD_MONTH,
        nonresidential_share=Config.GEN_NONRESIDENTIAL_SHARE,
        annual_trend=Config.GEN_ANNUAL_TREND,
        ev_penetration=Config.GEN_EV_PENETRATION,
        ev_home_power_kw=RealWorldReference.EV_HOME_POWER_KW,
        ev_public_power_kw=RealWorldReference.EV_PUBLIC_POWER_KW,
        ev_fleet_power_kw=RealWorldReference.EV_FLEET_POWER_KW,
        solar_penetration=Config.GEN_SOLAR_PENETRATION,
        solar_panel_peak_kw=RealWorldReference.SOLAR_PANEL_PEAK_KW,
        dsr_events_per_year=Config.GEN_DSR_EVENTS_PER_YEAR,
        dsr_strength_range=Config.GEN_DSR_STRENGTH,
        industrial_loads=Config.GEN_INDUSTRIAL_LOADS,
        city_districts=Config.GEN_CITY_DISTRICTS,
    )
    validate_generated_data(df)
    # Сверка с фактическими показателями России: результат попадает в лог
    # и в метаданные прогона, чтобы расхождения нельзя было не заметить.
    reference_ok, reference_rows = validate_against_reference(
        df, households=Config.HOUSEHOLDS,
        kwh_per_household_month=Config.GEN_KWH_PER_HOUSEHOLD_MONTH,
    )
    Config.print_summary()

    # ── [2/10] EDA ───────────────────────────────────────────────────────────
    if args.skip_eda:
        logger.info("\n[2/10] EDA пропущен (--skip-eda)")
    else:
        logger.info("\n[2/10] Исследовательский анализ данных...")
        run_eda(df, plots_dir=Config.PLOTS_DIR, save=True)

    # ── [3/10] Подготовка данных ─────────────────────────────────────────────
    logger.info("\n[3/10] Подготовка данных (скалеры обучаются только на train)...")
    data = prepare_data(
        df, history_length=Config.HISTORY_LENGTH,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO, val_ratio=Config.VAL_RATIO,
    )
    x_shape = data["X_train"].shape
    assert x_shape[2] == Config.N_FEATURES, (
        f"Несовпадение числа признаков: X_train={x_shape[2]}, ожидается {Config.N_FEATURES}")
    validate_data_integrity(data)

    # ── [4/10] Инициализация моделей ─────────────────────────────────────────
    logger.info("\n[4/10] Инициализация моделей...")
    models_to_train = build_models(model_keys, logger,
                                   n_train_windows=len(data["X_train"]))

    # ── [5/10] Обучение ──────────────────────────────────────────────────────
    logger.info("\n[5/10] Обучение моделей...")
    # Состав моделей фиксируется явно: пайплайн обязан отчитаться, какие из
    # запрошенных моделей действительно обучились. Молчаливое исключение
    # упавшей модели с последующим успешным завершением скрывает отказ.
    requested_models = [name for _, name in models_to_train]
    trained_models: List[str] = []
    failed_models: List[Dict[str, str]] = []

    trainers: List[ModelTrainer] = []
    for model, name in models_to_train:
        trainer = ModelTrainer(model, name, Config.MODELS_DIR, Config.PLOTS_DIR)
        try:
            trainer.train(
                data, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
                patience=Config.PATIENCE, lr_patience=Config.LR_PATIENCE,
                lr_factor=Config.LR_FACTOR, min_delta=Config.MIN_DELTA,
            )
        except Exception as exc:
            logger.error("Модель %s не обучена: %s", name, exc)
            failed_models.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        trainers.append(trainer)
        trained_models.append(name)
        if trainer.history is not None:
            plot_training_history(trainer.history, model_name=name,
                                  plots_dir=Config.PLOTS_DIR)

    if not trainers:
        logger.error("Ни одна модель не обучена — дальнейшие шаги невозможны.")
        return 1

    # ── [6/10] Отбор лучшей модели ПО ВАЛИДАЦИИ ──────────────────────────────
    # Тест на этом шаге не используется: иначе итоговая оценка победителя
    # окажется смещённой (мы бы выбрали модель, случайно удачную на тесте).
    logger.info("\n[6/10] Отбор лучшей модели по ВАЛИДАЦИОННОЙ выборке...")
    val_metrics = compare_trainers(trainers, data, split="val")
    trainable = {n: m for n, m in val_metrics.items()
                 if n not in (MODEL_REGISTRY[k] for k in NAIVE_KEYS)}
    pool = trainable or val_metrics
    best_name = min(pool, key=lambda k: pool[k]["MAE"])
    best_trainer = next(t for t in trainers if t.model_name == best_name)
    logger.info("Лучшая модель по валидации: %s (MAE_val=%.2f)",
                best_name, pool[best_name]["MAE"])

    # ── [7/10] Итоговая оценка на тесте ──────────────────────────────────────
    logger.info("\n[7/10] Итоговая оценка на ТЕСТОВОЙ выборке...")
    test_metrics = compare_trainers(trainers, data, split="test")

    scaler = data["scaler"]
    y_true = inverse_scale(scaler, data["Y_test"])
    predictions = {t.model_name: t.predict_original_scale(data, "test") for t in trainers}

    plot_metrics_comparison(test_metrics, plots_dir=Config.PLOTS_DIR)
    plot_predictions_comparison(y_true, predictions, plots_dir=Config.PLOTS_DIR)
    plot_accuracy_vs_cost(test_metrics, plots_dir=Config.PLOTS_DIR)

    # Деградация ошибки по шагам горизонта
    per_horizon = {
        name: metrics_by_horizon(y_true, pred, mase_scale=data.get("mase_scale"))
        for name, pred in predictions.items()
    }
    plot_metrics_by_horizon(per_horizon, plots_dir=Config.PLOTS_DIR)
    _log_horizon_summary(per_horizon, logger)

    # Статистическая значимость различий между моделями
    dm_rows = pairwise_dm_table(y_true, predictions, h=Config.FORECAST_HORIZON)
    _log_dm_summary(dm_rows, best_name, logger)

    # ── [8/10] Диагностика: внимание и остатки ───────────────────────────────
    logger.info("\n[8/10] Диагностика лучшей модели: %s", best_name)
    if not args.skip_attention:
        _visualize_attention(trainers, data, logger)

    analyze_residuals(y_true, predictions[best_name],
                      model_name=best_name, plots_dir=Config.PLOTS_DIR)

    # ── [9/10] Устойчивость и бэктестинг ─────────────────────────────────────
    logger.info("\n[9/10] Устойчивость прогноза во времени...")
    analyze_stability_over_windows(
        best_trainer.model, data, n_windows=8,
        plots_dir=Config.PLOTS_DIR, model_name=best_name,
    )

    if args.rolling_origin > 0:
        best_key = next((k for k, v in MODEL_REGISTRY.items() if v == best_name), None)
        if best_key:
            logger.info("Запуск walk-forward бэктестинга (%d origin-ов)...",
                        args.rolling_origin)
            run_rolling_origin_backtest(
                build_fn=lambda: build_models([best_key], logger,
                                              n_train_windows=len(data["X_train"]))[0][0],
                df=df, model_name=best_name, n_origins=args.rolling_origin,
                history_length=Config.HISTORY_LENGTH,
                forecast_horizon=Config.FORECAST_HORIZON,
                epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE,
                patience=Config.PATIENCE, plots_dir=Config.PLOTS_DIR,
            )

    # ── [10/10] Оптимизация накопителя ───────────────────────────────────────
    storage_results: Dict[str, Any] = {}
    storage_forecasts: Dict[str, np.ndarray] = {}
    if args.skip_storage:
        logger.info("\n[10/10] Блок накопителя пропущен (--skip-storage)")
    else:
        logger.info("\n[10/10] Оптимизация накопителя энергии...")
        storage_results, storage_forecasts = _run_storage_block(
            data, predictions, best_name, logger,
            compare_strategies, compare_forecast_sources,
            plot_storage_result, plot_forecast_value,
            reconstruct_day_ahead_series, actual_series_for_forecast,
            best_trainer=best_trainer,
        )

    # ── Экспорт результатов ──────────────────────────────────────────────────
    logger.info("\nЭкспорт результатов...")
    run_meta = {
        "mode": args.mode, "scenario": args.scenario, "seed": args.seed,
        "reference_check_passed": bool(reference_ok),
        "reference_check": reference_rows,
        "kwh_per_household_month": Config.GEN_KWH_PER_HOUSEHOLD_MONTH,
        "ev_penetration": Config.GEN_EV_PENETRATION,
        "solar_penetration": Config.GEN_SOLAR_PENETRATION,
        "tariffs_rub_per_kwh": {"peak": Config.TARIFF_PEAK,
                                "half_peak": Config.TARIFF_HALF_PEAK,
                                "night": Config.TARIFF_NIGHT},
        "battery_cost_rub": Config.BATTERY_COST_RUB,
        "battery_cycle_cost_rub_per_kwh": round(Config.BATTERY_CYCLE_COST, 3),
        "days": Config.DAYS, "households": Config.HOUSEHOLDS,
        "history_length": Config.HISTORY_LENGTH,
        "forecast_horizon": Config.FORECAST_HORIZON,
        "epochs": Config.EPOCHS, "batch_size": Config.BATCH_SIZE,
        "best_model_by_val": best_name,
        "requested_models": requested_models,
        "trained_models": trained_models,
        "failed_models": failed_models,
        "partial_failure": bool(failed_models),
        "mase_scale": data.get("mase_scale"),
        "n_features": int(data["n_features"]),
        "split_sizes": {"train": int(len(data["X_train"])),
                        "val": int(len(data["X_val"])),
                        "test": int(len(data["X_test"]))},
        "n_train_windows": int(len(data["X_train"])),
        "runtime_sec": round(time.time() - t_start, 1),
    }
    reporting.write_run_metadata(run_dir, run_meta)
    actual_for_export = (storage_forecasts.pop("__actual__", None)
                         if storage_results else None)

    # Каталог прогона — основной носитель результатов; сводка в корне results
    # накапливает строки по сидам для агрегации mean ± std.
    # append=True обязателен для обоих каталогов: две записи подряд (test и val)
    # с перезаписью стирали бы строки первого сплита. Каталог прогона создаётся
    # пустым, поэтому дозапись там равносильна созданию файла.
    for target in (run_dir, Config.OUTPUT_DIR):
        reporting.export_metrics(test_metrics, target, seed=args.seed,
                                 split="test", run_meta=run_meta, append=True)
        reporting.export_metrics(val_metrics, target, seed=args.seed,
                                 split="val", run_meta=run_meta, append=True)
    reporting.export_horizon_metrics(per_horizon, run_dir, seed=args.seed)
    reporting.export_dm_tests(dm_rows, run_dir)
    if storage_results:
        reporting.export_storage_results(
            storage_results, run_dir,
            actual=actual_for_export, forecasts=storage_forecasts,
        )
    reporting.export_markdown_tables(test_metrics, run_dir,
                                     dm_rows=dm_rows,
                                     storage_results=storage_results or None)
    reporting.aggregate_seeds(Config.OUTPUT_DIR)

    # Экспорт бандла лучшей модели (только для Keras — sklearn-обёртки
    # сериализуются иначе и в инференс-контур не входят).
    # Бандл собирается для победителя независимо от его типа. Лучшей по
    # валидации регулярно оказывается не нейросеть, а градиентный бустинг:
    # прежнее ограничение на tf.keras.Model означало, что заявленная лучшая
    # модель оставалась без работающего инференса.
    bundle_dir = export_model_bundle(
        best_trainer.model, data,
        {"HISTORY_LENGTH": Config.HISTORY_LENGTH,
         "FORECAST_HORIZON": Config.FORECAST_HORIZON,
         "N_FEATURES": Config.N_FEATURES,
         "model_name": best_name, "mode": args.mode, "seed": args.seed},
        export_dir=Config.MODELS_DIR, model_name=best_name,
    )
    run_meta["deployable_model"] = best_name
    run_meta["bundle_dir"] = bundle_dir
    reporting.write_run_metadata(run_dir, run_meta)

    logger.info("\n" + "=" * 78)
    elapsed_min = (time.time() - t_start) / 60
    if failed_models:
        # Частичный отказ не выдаётся за успех: и в логе, и в коде возврата.
        # Иначе упавшие нейросети остаются незамеченными, а отчёт выглядит
        # полным, хотя построен по неполному составу моделей.
        logger.error(
            "Пайплайн завершён ЧАСТИЧНО за %.1f мин: не обучено моделей %d из %d.",
            elapsed_min, len(failed_models), len(requested_models),
        )
        for item in failed_models:
            logger.error("  не обучена: %s — %s", item["model"], item["error"])
        logger.error("  Результаты построены только по обученным моделям: %s",
                     ", ".join(trained_models))
        logger.info("  Результаты: %s", run_dir)
        logger.info("=" * 78)
        return 2

    logger.info("Пайплайн завершён за %.1f мин. Результаты: %s",
                elapsed_min, run_dir)
    logger.info("  Таблицы для записки: %s",
                os.path.join(run_dir, "markdown_tables.md"))
    logger.info("=" * 78)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ БЛОКИ
# ══════════════════════════════════════════════════════════════════════════════

def _log_horizon_summary(per_horizon: Dict[str, Dict[str, list]], logger) -> None:
    """Печатает ошибку на первом, среднем и последнем шаге горизонта."""
    logger.info("─" * 70)
    logger.info("ДЕГРАДАЦИЯ ПРОГНОЗА ПО ГОРИЗОНТУ (MAE, кВт·ч)")
    logger.info("%-22s %10s %10s %10s %10s", "Модель", "h=1", "h=12", "h=24", "рост, %")
    logger.info("─" * 70)
    for name, m in sorted(per_horizon.items(), key=lambda kv: np.mean(kv[1]["MAE"])):
        mae = m["MAE"]
        h_last = len(mae) - 1
        h_mid = min(11, h_last)
        growth = 100 * (mae[h_last] / mae[0] - 1) if mae[0] > 0 else float("nan")
        logger.info("%-22s %10.1f %10.1f %10.1f %10.1f",
                    name, mae[0], mae[h_mid], mae[h_last], growth)
    logger.info("─" * 70)


def _log_dm_summary(dm_rows: List[Dict[str, Any]], best_name: str, logger) -> None:
    """Печатает результаты теста Диболда–Мариано для лучшей модели."""
    relevant = [r for r in dm_rows
                if best_name in (r["model_a"], r["model_b"])]
    if not relevant:
        return
    logger.info("─" * 70)
    logger.info("ЗНАЧИМОСТЬ РАЗЛИЧИЙ С ЛУЧШЕЙ МОДЕЛЬЮ (тест Диболда–Мариано)")
    logger.info("─" * 70)
    for r in relevant:
        other = r["model_b"] if r["model_a"] == best_name else r["model_a"]
        logger.info("  %-22s vs %-22s p=%.4f → %s",
                    best_name, other, r["p_value"], r["better"])
    logger.info("─" * 70)
    logger.info("p >= 0.05 означает, что разница в точности статистически "
                "не подтверждена и на неё нельзя опираться в выводах.")


def _visualize_attention(trainers, data, logger) -> None:
    """Тепловые карты внимания для трансформерных моделей."""
    try:
        from utils.attention_visualization import (
            visualize_attention_weights, visualize_attention_summary,
            compare_head_specialization,
        )
        sample_x = data["X_test"][:1]
        for t in trainers:
            if t.model_name in ("VanillaTransformer", "PatchTST"):
                visualize_attention_weights(
                    t.model, sample_x, history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
                visualize_attention_summary(
                    t.model, sample_x, history_length=Config.HISTORY_LENGTH,
                    model_name=t.model_name, plots_dir=Config.PLOTS_DIR)
                compare_head_specialization(
                    t.model, sample_x, model_name=t.model_name,
                    plots_dir=Config.PLOTS_DIR)
    except Exception as exc:
        logger.warning("Визуализация внимания пропущена: %s", exc)


def _run_storage_block(
    data, predictions, best_name, logger,
    compare_strategies, compare_forecast_sources,
    plot_storage_result, plot_forecast_value,
    reconstruct_day_ahead_series, actual_series_for_forecast,
    best_trainer=None,
):
    """
    Оптимизация накопителя на РЕАЛЬНОМ прогнозе модели.

    Два принципиальных момента, без которых расчёт теряет смысл:
      1. Тарифные зоны привязываются к календарю тестового периода. Значения по
         умолчанию (понедельник 00:00) сдвинули бы «ночь» на середину дня.
      2. Решения принимаются по прогнозу, а стоимость считается по факту.
    """
    horizon = int(Config.FORECAST_HORIZON)

    # Прогнозный ряд «день вперёд»: стыкуем непересекающиеся горизонты.
    best_series = reconstruct_day_ahead_series(predictions[best_name], horizon)
    n_hours = min(int(Config.STORAGE_HORIZON), len(best_series))
    n_hours -= n_hours % horizon                     # только целые сутки
    if n_hours < horizon:
        logger.warning("Недостаточно данных для симуляции накопителя "
                       "(доступно %d ч) — блок пропущен.", n_hours)
        return {}, {}

    actual = actual_series_for_forecast(data, n_hours)
    if len(actual) < n_hours:
        n_hours = len(actual) - len(actual) % horizon
        actual = actual[:n_hours]

    # Календарная привязка первой прогнозируемой точки.
    ts0 = pd.Timestamp(data["timestamps"][data["test_start_idx"]])
    start_hour, start_weekday = int(ts0.hour), int(ts0.dayofweek)
    logger.info("Горизонт планирования: %d ч, начало %s (час=%d, день недели=%d)",
                n_hours, ts0, start_hour, start_weekday)

    common = dict(
        capacity=Config.BATTERY_CAPACITY, max_power=Config.BATTERY_MAX_POWER,
        round_trip_efficiency=Config.BATTERY_EFFICIENCY,
        cycle_cost_per_kwh=Config.BATTERY_CYCLE_COST,
        battery_cost_rub=Config.BATTERY_COST_RUB,
        tariff_night=Config.TARIFF_NIGHT, tariff_half_peak=Config.TARIFF_HALF_PEAK,
        tariff_peak=Config.TARIFF_PEAK,
        demand_charge_rub_per_kw_month=Config.DEMAND_CHARGE_RUB_PER_KW_MONTH,
        annual_om_share=Config.BATTERY_OM_SHARE,
        start_hour=start_hour, start_weekday=start_weekday,
    )

    # Сравнение стратегий по глубине разряда (календарное управление).
    strategy_results = compare_strategies(
        forecast=actual, actual=actual, policy="tariff", **common,
    )
    plot_storage_result(strategy_results["Умеренная"], actual,
                        plots_dir=Config.PLOTS_DIR)

    # Главное: во сколько обходится ошибка прогноза.
    forecasts = {best_name: best_series[:n_hours]}
    naive_name = MODEL_REGISTRY["naive24"]
    if naive_name in predictions:
        forecasts[naive_name] = reconstruct_day_ahead_series(
            predictions[naive_name], horizon)[:n_hours]

    # Верхний квантиль как вход управления. Недооценка пика оставляет накопитель
    # незаряженным и обесценивает месяц работы, переоценка стоит лишь небольшого
    # износа — значит, оптимален не средний прогноз, а сдвинутый вверх. Смещение
    # берётся с ВАЛИДАЦИИ: построенное по тесту, оно знало бы ответ.
    if best_trainer is not None:
        from utils.conformal import conformal_offsets, apply_offset
        from data.preprocessing import inverse_scale

        pred_val = best_trainer.predict_original_scale(data, "val")
        y_val = inverse_scale(data["scaler"], data["Y_val"])
        offsets = conformal_offsets(y_val - pred_val, quantiles=(0.9,))

        shifted = apply_offset(predictions[best_name], offsets[0.9])
        forecasts[f"{best_name} P90"] = reconstruct_day_ahead_series(
            shifted, horizon)[:n_hours]
        logger.info("Добавлен верхний квантиль P90 поверх %s: средний сдвиг %.1f кВт·ч",
                    best_name, float(np.mean(offsets[0.9])))

    value_results = compare_forecast_sources(
        actual=actual, forecasts=forecasts,
        min_soc=Config.BATTERY_MIN_SOC, max_soc=Config.BATTERY_MAX_SOC,
        **common,
    )
    plot_forecast_value(value_results, plots_dir=Config.PLOTS_DIR)

    # Несимметрична не оценка нагрузки, а стоимость ошибки, поэтому настраивать
    # надо решающее правило. Порог подбирается на ВАЛИДАЦИИ и проверяется на
    # тесте: подбор по тесту дал бы оптимистично смещённую оценку — ту же
    # ошибку, что и выбор модели по тесту.
    from optimization.storage import (
        select_shaving_threshold, sweep_shaving_threshold, simulate_storage,
    )
    sweep_kwargs = dict(min_soc=Config.BATTERY_MIN_SOC,
                        max_soc=Config.BATTERY_MAX_SOC, **common)

    if best_trainer is not None:
        pred_val_series = reconstruct_day_ahead_series(
            best_trainer.predict_original_scale(data, "val"), horizon)
        hist = int(data["history_length"])
        actual_val = np.asarray(data["raw_val"][hist:hist + len(pred_val_series)],
                                dtype=np.float64)
        n_val = min(len(pred_val_series), len(actual_val))
        n_val -= n_val % horizon

        if n_val >= horizon:
            # Валидационный период начинается в свой момент календаря, и
            # тарифные зоны обязаны считаться от него. Привязка тестового
            # периода, подставленная сюда, сдвигает «ночь» и «пик» относительно
            # данных: на этих данных та же кривая перебора менялась на 167 тыс.
            # рублей — больше, чем весь измеряемый эффект.
            ts_val = pd.Timestamp(
                data["timestamps"][data["train_end_idx"] + hist])
            val_kwargs = dict(sweep_kwargs)
            val_kwargs["start_hour"] = int(ts_val.hour)
            val_kwargs["start_weekday"] = int(ts_val.dayofweek)
            logger.info("Валидационный период для подбора порога: начало %s "
                        "(час=%d, день недели=%d)", ts_val, ts_val.hour,
                        ts_val.dayofweek)

            chosen_q = select_shaving_threshold(
                pred_val_series[:n_val], actual_val[:n_val], **val_kwargs)
            tuned = simulate_storage(
                forecast=forecasts[best_name], actual=actual, policy="peak_shaving",
                shave_quantile=chosen_q,
                forecast_source=f"{best_name}, порог q={chosen_q:.2f} по валидации",
                **sweep_kwargs)
            value_results[f"{best_name} (порог по валидации)"] = tuned
            logger.info("Порог q=%.2f, выбранный по валидации, на тесте даёт %.0f руб",
                        chosen_q, tuned.net_savings)

    # Полный перебор на тесте сохраняется отдельно как ВЕРХНЯЯ ГРАНИЦА: он
    # показывает, сколько теряется на неоптимальном пороге, но сам по себе
    # недостижим — значение выбрано по той же выборке, на которой измеряется.
    sweep = sweep_shaving_threshold(
        forecast=forecasts[best_name], actual=actual, label="тест (верхняя граница)",
        **sweep_kwargs)
    pd.DataFrame(sweep).to_csv(
        os.path.join(Config.OUTPUT_DIR, "storage_threshold_sweep.csv"),
        index=False, encoding="utf-8-sig")

    forecasts["__actual__"] = actual

    # Прогнозные ряды сохраняются рядом с результатами. Без них любой опыт с
    # политикой управления требует переобучения всех моделей — два с лишним
    # часа ради перебора порога, который считается за секунды.
    saved = {k: v[:n_hours] for k, v in forecasts.items() if len(v) >= n_hours}
    if best_trainer is not None:
        try:
            val_series = reconstruct_day_ahead_series(
                best_trainer.predict_original_scale(data, "val"), horizon)
            hist = int(data["history_length"])
            val_actual = np.asarray(data["raw_val"][hist:hist + len(val_series)],
                                    dtype=np.float64)
            n_v = min(len(val_series), len(val_actual))
            # Отметки времени обязательны: без них тарифные зоны при офлайн-
            # разборе привязываются к понедельнику 00:00 и съезжают относительно
            # данных, а расчёт экономики теряет смысл, оставаясь правдоподобным.
            ts_val = pd.to_datetime(data["timestamps"])[
                data["train_end_idx"] + hist: data["train_end_idx"] + hist + n_v]
            pd.DataFrame({"timestamp": ts_val,
                          "forecast": val_series[:n_v], "actual": val_actual[:n_v]}).to_csv(
                os.path.join(Config.OUTPUT_DIR, "forecast_series_val.csv"),
                index=False, encoding="utf-8-sig")
        except Exception as exc:
            logger.warning("Валидационные ряды не сохранены: %s", exc)

    frame = pd.DataFrame(saved)
    frame.insert(0, "timestamp",
                 pd.to_datetime(data["timestamps"])[
                     data["test_start_idx"]: data["test_start_idx"] + len(frame)])
    frame.to_csv(os.path.join(Config.OUTPUT_DIR, "forecast_series_test.csv"),
                 index=False, encoding="utf-8-sig")

    return value_results, forecasts


if __name__ == "__main__":
    sys.exit(main())
