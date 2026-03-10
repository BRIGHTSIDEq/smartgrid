# -*- coding: utf-8 -*-
"""config.py — Централизованная конфигурация Smart Grid. Версия 4."""

import os
import logging


class Config:

    SEED: int = 42

    # ── Данные ────────────────────────────────────────────────────────────────
    DAYS: int = 365
    HOUSEHOLDS: int = 500
    START_DATE: str = "2024-01-01"

    # ── Признаки ─────────────────────────────────────────────────────────────
    N_FEATURES: int = 9

    # ── Временные окна ────────────────────────────────────────────────────────
    HISTORY_LENGTH: int = 48
    FORECAST_HORIZON: int = 24
    STORAGE_HORIZON: int = 168

    # ── Разделение ────────────────────────────────────────────────────────────
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # ── Обучение (общее) ──────────────────────────────────────────────────────
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    PATIENCE: int = 25
    LR_PATIENCE: int = 10
    LR_FACTOR: float = 0.5
    MIN_DELTA: float = 1e-5      # мин. улучшение для EarlyStopping

    # ── LSTM ──────────────────────────────────────────────────────────────────
    LSTM_UNITS_1: int = 256
    LSTM_UNITS_2: int = 128
    LSTM_UNITS_3: int = 64
    DROPOUT_RATE: float = 0.20          # снижено с 0.30
    LSTM_LEARNING_RATE: float = 1e-4    # отдельный LR для LSTM (было 5e-4 — вызывало расхождение)

    # ── Transformer ───────────────────────────────────────────────────────────
    TRANSFORMER_D_MODEL: int = 128
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 4
    TRANSFORMER_DFF: int = 256
    TRANSFORMER_DROPOUT: float = 0.20
    TRANSFORMER_LEARNING_RATE: float = 1e-4

    # ── XGBoost ───────────────────────────────────────────────────────────────
    XGB_N_ESTIMATORS: int = 500          # увеличено с 200 (432 фичи требуют больше деревьев)
    XGB_MAX_DEPTH: int = 5               # снижено с 6 (меньше переобучения на 432 фичах)
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.80
    XGB_COLSAMPLE: float = 0.40          # снижено: 432×0.4=173 фичи/дерево — эффективнее

    # ── Батарея ───────────────────────────────────────────────────────────────
    BATTERY_CAPACITY: float = 300.0
    BATTERY_MAX_POWER: float = 150.0
    BATTERY_EFFICIENCY: float = 0.95
    BATTERY_CYCLE_COST: float = 0.06
    # Стратегия по умолчанию: умеренная
    BATTERY_MIN_SOC: float = 0.25
    BATTERY_MAX_SOC: float = 0.75

    # ── Тарифы ────────────────────────────────────────────────────────────────
    TARIFF_PEAK: float = 6.50
    TARIFF_HALF_PEAK: float = 4.20
    TARIFF_NIGHT: float = 1.80

    # ── Пути ──────────────────────────────────────────────────────────────────
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "results")
    MODELS_DIR: str = os.path.join(BASE_DIR, "results", "models")
    PLOTS_DIR: str = os.path.join(BASE_DIR, "results", "plots")
    LOGS_DIR: str = os.path.join(BASE_DIR, "results", "logs")

    # ── Логирование ───────────────────────────────────────────────────────────
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def create_dirs(cls) -> None:
        for d in (cls.OUTPUT_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR):
            os.makedirs(d, exist_ok=True)

    @classmethod
    def setup_logging(cls) -> logging.Logger:
        cls.create_dirs()
        logging.basicConfig(
            level=cls.LOG_LEVEL,
            format=cls.LOG_FORMAT,
            datefmt=cls.LOG_DATE_FMT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(cls.LOGS_DIR, "run.log"), encoding="utf-8"
                ),
            ],
        )
        return logging.getLogger("smart_grid")

    @classmethod
    def set_fast_mode(cls) -> None:
        cls.DAYS = 180
        cls.HOUSEHOLDS = 250
        cls.EPOCHS = 50
        cls.PATIENCE = 10
        cls.HISTORY_LENGTH = 24
        cls.STORAGE_HORIZON = 48
        cls.TRANSFORMER_D_MODEL = 64
        cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 2
        cls.TRANSFORMER_DFF = 128
        cls.XGB_N_ESTIMATORS = 200
        logging.getLogger("smart_grid").info(
            "⚡ Fast mode: DAYS=%d, EPOCHS=%d, HISTORY=%d",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
        )

    @classmethod
    def set_optimal_mode(cls) -> None:
        cls.DAYS = 365
        cls.HOUSEHOLDS = 500
        cls.EPOCHS = 200
        cls.PATIENCE = 25
        cls.HISTORY_LENGTH = 48
        cls.STORAGE_HORIZON = 168
        cls.TRANSFORMER_D_MODEL = 128
        cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4
        cls.TRANSFORMER_DFF = 256
        cls.XGB_N_ESTIMATORS = 500
        cls.XGB_COLSAMPLE = 0.40
        logging.getLogger("smart_grid").info(
            "🎯 Optimal mode: DAYS=%d, EPOCHS=%d, HISTORY=%d, Transformer d=%d h=%d L=%d",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
        )

    @classmethod
    def set_full_mode(cls) -> None:
        cls.DAYS = 730
        cls.HOUSEHOLDS = 500
        cls.EPOCHS = 300
        cls.PATIENCE = 35
        cls.HISTORY_LENGTH = 72
        cls.STORAGE_HORIZON = 168
        cls.TRANSFORMER_D_MODEL = 128
        cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4
        cls.TRANSFORMER_DFF = 256
        cls.XGB_N_ESTIMATORS = 800
        cls.XGB_COLSAMPLE = 0.35
        logging.getLogger("smart_grid").info(
            "🚀 Full mode: DAYS=%d, EPOCHS=%d, HISTORY=%d",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
        )

    @classmethod
    def print_summary(cls) -> None:
        log = logging.getLogger("smart_grid")
        log.info("─" * 50)
        log.info("КОНФИГУРАЦИЯ:")
        log.info("  Данные:      %d дней, %d домохозяйств", cls.DAYS, cls.HOUSEHOLDS)
        log.info("  Признаки:    %d ковариат на шаг (мультивариантный вход)", cls.N_FEATURES)
        log.info("  История:     %d ч → прогноз %d ч", cls.HISTORY_LENGTH, cls.FORECAST_HORIZON)
        log.info("  Обучение:    %d эпох, batch=%d", cls.EPOCHS, cls.BATCH_SIZE)
        log.info("  LSTM:        units=%d/%d/%d, drop=%.2f, lr=%g, input=(%d,%d)",
                 cls.LSTM_UNITS_1, cls.LSTM_UNITS_2, cls.LSTM_UNITS_3,
                 cls.DROPOUT_RATE, cls.LSTM_LEARNING_RATE,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  Transformer: d=%d h=%d L=%d dff=%d drop=%.2f, lr=%g, input=(%d,%d)",
                 cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS,
                 cls.TRANSFORMER_N_LAYERS, cls.TRANSFORMER_DFF,
                 cls.TRANSFORMER_DROPOUT, cls.TRANSFORMER_LEARNING_RATE,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  XGBoost:     n_est=%d, depth=%d, col=%.2f",
                 cls.XGB_N_ESTIMATORS, cls.XGB_MAX_DEPTH, cls.XGB_COLSAMPLE)
        log.info("  Тарифы:      пик=%.2f день=%.2f ночь=%.2f руб/кВт·ч",
                 cls.TARIFF_PEAK, cls.TARIFF_HALF_PEAK, cls.TARIFF_NIGHT)
        log.info("  Батарея:     %.0f кВт·ч, SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч)",
                 cls.BATTERY_CAPACITY,
                 cls.BATTERY_MIN_SOC * 100, cls.BATTERY_MAX_SOC * 100,
                 (cls.BATTERY_MAX_SOC - cls.BATTERY_MIN_SOC) * cls.BATTERY_CAPACITY)
        log.info("─" * 50)
