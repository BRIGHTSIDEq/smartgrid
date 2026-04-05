# -*- coding: utf-8 -*-
"""config.py — Централизованная конфигурация Smart Grid. Версия 9.

ИЗМЕНЕНИЯ v9:
  N_FEATURES: 22 → 26 (добавлены cloud_cover, ev_load, solar_gen, dsr_active)

  Отдельный LR для VanillaTransformer:
    БЫЛО: TRANSFORMER_LEARNING_RATE=3e-4 для всех трансформеров
    СТАЛО: VANILLA_TRANSFORMER_LR=5e-5 (10× ниже)
    ПРИЧИНА: VanillaTransformer останавливался на эпохе 27 (best=7).
    LR=3e-4 слишком высокий — модель сходилась в плохой локальный минимум
    за 7 эпох, затем 20 эпох не улучшалась. LR=5e-5 → стабильное обучение.

  LSTM параметры для новой архитектуры v8 (TCN+BiLSTM):
    LSTM_LEARNING_RATE: 3e-4 → 2e-4
"""

import os
import logging


class Config:

    SEED: int = 42

    DAYS: int = 365
    HOUSEHOLDS: int = 500
    START_DATE: str = "2024-01-01"

    # v9: 22 → 26 (cloud_cover, ev_load_norm, solar_gen_norm, dsr_active)
    N_FEATURES: int = 26

    HISTORY_LENGTH: int = 48
    FORECAST_HORIZON: int = 24
    STORAGE_HORIZON: int = 720

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    PATIENCE: int = 25          # v9: 20→25 (больше терпения для TCN+BiLSTM)
    LR_PATIENCE: int = 10       # v9: 8→10
    LR_FACTOR: float = 0.5
    MIN_DELTA: float = 0.0

    # LSTM (TCN+BiLSTM+Attention v8)
    LSTM_UNITS_1: int = 128          # BiLSTM units per direction → 256 total
    LSTM_UNITS_2: int = 64
    LSTM_UNITS_3: int = 64
    DROPOUT_RATE: float = 0.20
    LSTM_LEARNING_RATE: float = 2e-4  # v9: 3e-4→2e-4 для TCN+BiLSTM
    LSTM_ATTN_HEADS: int = 8          # v9: 4→8
    LSTM_USE_COSINE_DECAY: bool = False
    LSTM_TCN_FILTERS: int = 64        # v9: TCN filters per branch
    LSTM_HUBER_DELTA: float = 0.05    # v9: 0.10→0.05 (строже на пики)
    LSTM_SEASONAL_BLEND_INIT: float = 0.35

    # Transformer (PatchTST и VanillaTransformer)
    TRANSFORMER_D_MODEL: int = 128
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 4
    TRANSFORMER_DFF: int = 256
    TRANSFORMER_DROPOUT: float = 0.20
    TRANSFORMER_LEARNING_RATE: float = 3e-4    # PatchTST LR (работает хорошо)
    VANILLA_TRANSFORMER_LR: float = 5e-5       # v9: отдельный LR для VanillaTransformer
    TRANSFORMER_STOCHASTIC_DEPTH: float = 0.10
    PATCHTST_USE_REVIN: bool = True
    VANILLA_USE_SEASONAL_RESIDUAL: bool = True
    VANILLA_SEASONAL_BLEND_INIT: float = 0.40
    VANILLA_HUBER_DELTA: float = 0.05

    # XGBoost
    XGB_N_ESTIMATORS: int = 500
    XGB_MAX_DEPTH: int = 5
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.80
    XGB_COLSAMPLE: float = 0.40

    # Generator v5 params
    GEN_TEMP_SETPOINT: float = 18.0
    GEN_TEMP_QUADRATIC_COEF: float = 2.5e-4
    GEN_HUMIDITY_THRESHOLD: float = 60.0
    GEN_HUMIDITY_COEF: float = 0.30
    GEN_WIND_TEMP_THRESHOLD: float = 10.0
    GEN_WIND_COEF: float = 0.15
    GEN_EARLY_BIRD_FRAC: float = 0.28
    GEN_NIGHT_OWL_FRAC:  float = 0.20
    GEN_AR_PHI: float   = 0.65   # v9: усилен
    GEN_AR_SIGMA: float = 0.060  # v9: усилен
    GEN_SEASONAL_WINTER_BOOST: float = 0.15
    GEN_SEASONAL_SUMMER_DIP:   float = 0.10
    GEN_EV_PENETRATION: float  = 0.28   # v9: 28% EV
    GEN_SOLAR_PENETRATION: float = 0.22  # v9: 22% Solar
    GEN_INDUSTRIAL_LOADS: int  = 4       # v9: 4 пром.потребителя

    # Батарея
    BATTERY_CAPACITY: float = 4_500.0
    BATTERY_MAX_POWER: float = 2_250.0
    BATTERY_EFFICIENCY: float = 0.95
    BATTERY_CYCLE_COST: float = 0.06
    BATTERY_COST_RUB: float = 45_000_000.0
    BATTERY_OM_SHARE: float = 0.015
    DEMAND_CHARGE_RUB_PER_KW_MONTH: float = 950.0
    BATTERY_MIN_SOC: float = 0.25
    BATTERY_MAX_SOC: float = 0.75

    # Тарифы
    TARIFF_PEAK: float = 6.50
    TARIFF_HALF_PEAK: float = 4.20
    TARIFF_NIGHT: float = 1.80

    # Пути
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR: str  = os.path.join(BASE_DIR, "results")
    MODELS_DIR: str  = os.path.join(BASE_DIR, "results", "models")
    PLOTS_DIR: str   = os.path.join(BASE_DIR, "results", "plots")
    LOGS_DIR: str    = os.path.join(BASE_DIR, "results", "logs")

    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def create_dirs(cls):
        for d in (cls.OUTPUT_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR):
            os.makedirs(d, exist_ok=True)

    @classmethod
    def setup_logging(cls):
        cls.create_dirs()
        logging.basicConfig(
            level=cls.LOG_LEVEL, format=cls.LOG_FORMAT, datefmt=cls.LOG_DATE_FMT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(cls.LOGS_DIR, "run.log"), encoding="utf-8"),
            ],
        )
        return logging.getLogger("smart_grid")

    @classmethod
    def set_fast_mode(cls):
        cls.DAYS = 365; cls.HOUSEHOLDS = 250; cls.EPOCHS = 120
        cls.PATIENCE = 20; cls.LR_PATIENCE = 8
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720
        cls.N_FEATURES = 26
        cls.LSTM_UNITS_1 = 96; cls.LSTM_UNITS_2 = 48; cls.LSTM_UNITS_3 = 48
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 48
        cls.DROPOUT_RATE = 0.25; cls.LSTM_LEARNING_RATE = 2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.30
        cls.TRANSFORMER_D_MODEL = 64; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 2; cls.TRANSFORMER_DFF = 128
        cls.TRANSFORMER_DROPOUT = 0.20; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 8e-5
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.05; cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.35
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 300
        logging.getLogger("smart_grid").info(
            "Fast mode v9: DAYS=%d EPOCHS=%d N_FEATURES=%d | LSTM BiLSTM=%d TCN=%d | VanTr LR=%.0e",
            cls.DAYS, cls.EPOCHS, cls.N_FEATURES, cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS,
            cls.VANILLA_TRANSFORMER_LR)

    @classmethod
    def set_optimal_mode(cls):
        cls.DAYS = 365; cls.HOUSEHOLDS = 500; cls.EPOCHS = 200
        cls.PATIENCE = 25; cls.LR_PATIENCE = 10
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720
        cls.N_FEATURES = 26
        cls.LSTM_UNITS_1 = 128; cls.LSTM_UNITS_2 = 64; cls.LSTM_UNITS_3 = 64
        cls.LSTM_ATTN_HEADS = 8; cls.LSTM_TCN_FILTERS = 64
        cls.DROPOUT_RATE = 0.20; cls.LSTM_LEARNING_RATE = 2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.35
        cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 128; cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4; cls.TRANSFORMER_DFF = 256
        cls.TRANSFORMER_DROPOUT = 0.20; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 5e-5   # ключевое исправление
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.10; cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.40
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 500; cls.XGB_COLSAMPLE = 0.40
        logging.getLogger("smart_grid").info(
            "Optimal mode v9: DAYS=%d EPOCHS=%d N_FEATURES=%d | "
            "LSTM TCN-BiLSTM heads=%d tcn=%d lr=%.0e huber=%.2f | "
            "Trans d=%d h=%d L=%d lr=%.0e | VanTr LR=%.0e | "
            "STORAGE=%dh BAT=%.0f кВт·ч",
            cls.DAYS, cls.EPOCHS, cls.N_FEATURES,
            cls.LSTM_ATTN_HEADS, cls.LSTM_TCN_FILTERS, cls.LSTM_LEARNING_RATE, cls.LSTM_HUBER_DELTA,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
            cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR,
            cls.STORAGE_HORIZON, cls.BATTERY_CAPACITY)

    @classmethod
    def set_full_mode(cls):
        cls.DAYS = 730; cls.HOUSEHOLDS = 500; cls.EPOCHS = 300
        cls.PATIENCE = 30; cls.LR_PATIENCE = 10
        cls.HISTORY_LENGTH = 72; cls.STORAGE_HORIZON = 720
        cls.N_FEATURES = 26
        cls.LSTM_UNITS_1 = 192; cls.LSTM_UNITS_2 = 96; cls.LSTM_UNITS_3 = 96
        cls.LSTM_ATTN_HEADS = 8; cls.LSTM_TCN_FILTERS = 96
        cls.DROPOUT_RATE = 0.20; cls.LSTM_LEARNING_RATE = 2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.35
        cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 128; cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4; cls.TRANSFORMER_DFF = 512
        cls.TRANSFORMER_DROPOUT = 0.20; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 8e-5
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.10; cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.40
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 800; cls.XGB_COLSAMPLE = 0.35
        logging.getLogger("smart_grid").info(
            "Full mode v9: DAYS=%d EPOCHS=%d N_FEATURES=%d | BiLSTM=%d TCN=%d",
            cls.DAYS, cls.EPOCHS, cls.N_FEATURES, cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS)

    @classmethod
    def print_summary(cls):
        log = logging.getLogger("smart_grid")
        log.info("─" * 50)
        log.info("КОНФИГУРАЦИЯ:")
        log.info("  Данные:      %d дней, %d домохозяйств", cls.DAYS, cls.HOUSEHOLDS)
        log.info("  Признаки:    %d ковариат на шаг (мультивариантный вход)", cls.N_FEATURES)
        log.info("  История:     %d ч → прогноз %d ч", cls.HISTORY_LENGTH, cls.FORECAST_HORIZON)
        log.info("  Обучение:    %d эпох, batch=%d, patience=%d", cls.EPOCHS, cls.BATCH_SIZE, cls.PATIENCE)
        log.info("  LSTM v8:     BiLSTM=%d TCN_filters=%d attn=%dh, drop=%.2f, lr=%g, huber=%.2f, input=(%d,%d)",
                 cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS,
                 cls.DROPOUT_RATE, cls.LSTM_LEARNING_RATE, cls.LSTM_HUBER_DELTA,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  Transformer: d=%d h=%d L=%d dff=%d drop=%.2f, lr=%g (Vanilla lr=%g), input=(%d,%d)",
                 cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS,
                 cls.TRANSFORMER_N_LAYERS, cls.TRANSFORMER_DFF,
                 cls.TRANSFORMER_DROPOUT, cls.TRANSFORMER_LEARNING_RATE,
                 cls.VANILLA_TRANSFORMER_LR, cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  XGBoost:     n_est=%d, depth=%d, col=%.2f",
                 cls.XGB_N_ESTIMATORS, cls.XGB_MAX_DEPTH, cls.XGB_COLSAMPLE)
        log.info("  Тарифы:      пик=%.2f день=%.2f ночь=%.2f руб/кВт·ч",
                 cls.TARIFF_PEAK, cls.TARIFF_HALF_PEAK, cls.TARIFF_NIGHT)
        log.info("  Батарея:     %.0f кВт·ч, SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч)",
                 cls.BATTERY_CAPACITY,
                 cls.BATTERY_MIN_SOC*100, cls.BATTERY_MAX_SOC*100,
                 (cls.BATTERY_MAX_SOC-cls.BATTERY_MIN_SOC)*cls.BATTERY_CAPACITY)
        log.info("─" * 50)
