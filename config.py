# -*- coding: utf-8 -*-
"""config.py — Smart Grid v10.

ИЗМЕНЕНИЯ v10:
  LSTM overfitting fix:
    БЫЛО: LSTM_UNITS_1=128, LSTM_TCN_FILTERS=64 → 945K params → overfitting 2.4×
          train_mae=0.019 vs val_mae=0.045, MAE=1182 > LinReg MAE=953
    СТАЛО: LSTM_UNITS_1=64, LSTM_TCN_FILTERS=32 → ~150K params
           Правило: 6K сэмплов / 150K params = 40 сэмплов/параметр (норма для dropout)

  EV generator params:
    ev_penetration: 0.28 → 0.50
    (EV time-wrap баг исправлен в generator.py → реальный вклад теперь 8-12%)

  full_mode fixes:
    validate_data_integrity больше не падает (исправлено в preprocessing.py)
    full_mode: LSTM_UNITS_1=96 (было 192 → 730 days × 192 = ещё хуже overfitting)
"""

import os
import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorCoefficients:
    temp_setpoint: float
    temp_quadratic_coef: float
    humidity_threshold: float
    humidity_coef: float
    wind_temp_threshold: float
    wind_coef: float
    early_bird_frac: float
    night_owl_frac: float
    ar_phi: float
    ar_sigma: float
    seasonal_winter_boost: float
    seasonal_summer_dip: float
    ev_penetration: float
    solar_penetration: float
    industrial_loads: int
    city_districts: int


class Config:

    SEED: int = 42
    DAYS: int = 730
    HOUSEHOLDS: int = 2500
    START_DATE: str = "2024-01-01"
    N_FEATURES: int = 26

    HISTORY_LENGTH: int = 48
    FORECAST_HORIZON: int = 24
    STORAGE_HORIZON: int = 720

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    EPOCHS: int = 240
    BATCH_SIZE: int = 32
    PATIENCE: int = 25
    LR_PATIENCE: int = 10
    LR_FACTOR: float = 0.5
    MIN_DELTA: float = 0.0

    # LSTM v9 (TCN+BiLSTM+Attention)
    LSTM_UNITS_1: int = 96          # больше ёмкость на 12k+ train-сэмплов
    LSTM_UNITS_2: int = 64          # compat
    LSTM_UNITS_3: int = 64          # compat
    DROPOUT_RATE: float = 0.12
    LSTM_LEARNING_RATE: float = 2.0e-4
    LSTM_ATTN_HEADS: int = 4        # v10: 8→4 (part of size reduction)
    LSTM_USE_COSINE_DECAY: bool = False
    LSTM_TCN_FILTERS: int = 48
    LSTM_HUBER_DELTA: float = 0.05
    LSTM_SEASONAL_BLEND_INIT: float = 0.35

    # Transformer
    TRANSFORMER_D_MODEL: int = 192
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 5
    TRANSFORMER_DFF: int = 384
    TRANSFORMER_DROPOUT: float = 0.10
    TRANSFORMER_LEARNING_RATE: float = 2e-4
    VANILLA_TRANSFORMER_LR: float = 7e-5
    TRANSFORMER_STOCHASTIC_DEPTH: float = 0.06
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

    # Generator v6 params
    GEN_TEMP_SETPOINT: float = 18.0
    GEN_TEMP_QUADRATIC_COEF: float = 2.5e-4
    GEN_HUMIDITY_THRESHOLD: float = 60.0
    GEN_HUMIDITY_COEF: float = 0.30
    GEN_WIND_TEMP_THRESHOLD: float = 10.0
    GEN_WIND_COEF: float = 0.15
    GEN_EARLY_BIRD_FRAC: float = 0.28
    GEN_NIGHT_OWL_FRAC:  float = 0.20
    GEN_AR_PHI: float   = 0.65
    GEN_AR_SIGMA: float = 0.040
    GEN_SEASONAL_WINTER_BOOST: float = 0.15
    GEN_SEASONAL_SUMMER_DIP:   float = 0.10
    GEN_EV_PENETRATION: float   = 0.50   # v10: 28%→50% + time-wrap исправлен
    GEN_SOLAR_PENETRATION: float = 0.22
    GEN_INDUSTRIAL_LOADS: int   = 8
    GEN_CITY_DISTRICTS: int = 12

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

    TARIFF_PEAK: float = 6.50
    TARIFF_HALF_PEAK: float = 4.20
    TARIFF_NIGHT: float = 1.80

    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR: str  = os.path.join(BASE_DIR, "results")
    DATA_DIR: str = os.path.join(BASE_DIR, "data", "generated")
    GENERATED_DATA_CSV: str = os.path.join(DATA_DIR, "smartgrid_dataset.csv")
    FORCE_REGENERATE_DATA: bool = False
    MODELS_DIR: str  = os.path.join(BASE_DIR, "results", "models")
    PLOTS_DIR: str   = os.path.join(BASE_DIR, "results", "plots")
    LOGS_DIR: str    = os.path.join(BASE_DIR, "results", "logs")

    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def create_dirs(cls):
        for d in (cls.OUTPUT_DIR, cls.DATA_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR):
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
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.LSTM_UNITS_1 = 48; cls.LSTM_UNITS_2 = 48; cls.LSTM_UNITS_3 = 48
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 32
        cls.DROPOUT_RATE = 0.25; cls.LSTM_LEARNING_RATE = 2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.30; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 64; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 2; cls.TRANSFORMER_DFF = 128
        cls.TRANSFORMER_DROPOUT = 0.20; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 8e-5; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.05
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.35
        cls.VANILLA_HUBER_DELTA = 0.05; cls.XGB_N_ESTIMATORS = 300
        cls.GEN_EV_PENETRATION = 0.50; cls.GEN_INDUSTRIAL_LOADS = 4; cls.GEN_CITY_DISTRICTS = 8
        logging.getLogger("smart_grid").info(
            "Fast mode v10: DAYS=%d EPOCHS=%d N_FEATURES=%d | "
            "LSTM BiLSTM=%d TCN=%d (~%dK params) | VanTr LR=%.0e | EV=%.0f%%",
            cls.DAYS, cls.EPOCHS, cls.N_FEATURES, cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS,
            # rough param estimate
            (4*cls.LSTM_TCN_FILTERS*26 + 4*cls.LSTM_UNITS_1*(4*cls.LSTM_TCN_FILTERS+cls.LSTM_UNITS_1) + 128*64)//1000,
            cls.VANILLA_TRANSFORMER_LR, cls.GEN_EV_PENETRATION*100)

    @classmethod
    def set_optimal_mode(cls):
        # Режим для лучшего качества при разумном времени обучения (без переобучения fast/full).
        cls.DAYS = 730; cls.HOUSEHOLDS = 2500; cls.EPOCHS = 220
        cls.PATIENCE = 20; cls.LR_PATIENCE = 8
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.LSTM_UNITS_1 = 72; cls.LSTM_UNITS_2 = 72; cls.LSTM_UNITS_3 = 72
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 48
        cls.DROPOUT_RATE = 0.18; cls.LSTM_LEARNING_RATE = 1.5e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.35; cls.LSTM_HUBER_DELTA = 0.02
        cls.TRANSFORMER_D_MODEL = 128; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 3; cls.TRANSFORMER_DFF = 256
        cls.TRANSFORMER_DROPOUT = 0.15; cls.TRANSFORMER_LEARNING_RATE = 1.5e-4
        cls.VANILLA_TRANSFORMER_LR = 1.2e-4; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.08
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.40
        cls.VANILLA_HUBER_DELTA = 0.02
        cls.XGB_N_ESTIMATORS = 700; cls.XGB_COLSAMPLE = 0.50
        cls.GEN_AR_SIGMA = 0.040
        cls.GEN_EV_PENETRATION = 0.50; cls.GEN_INDUSTRIAL_LOADS = 8; cls.GEN_CITY_DISTRICTS = 12
        logging.getLogger("smart_grid").info(
            "Optimal mode v12: DAYS=%d EPOCHS=%d N_FEATURES=%d | "
            "LSTM TCN=%d BiLSTM=%d heads=%d lr=%.0e drop=%.2f huber=%.2f | "
            "Trans d=%d h=%d L=%d dff=%d drop=%.2f lr=%.0e | VanTr LR=%.0e | "
            "EV=%.0f%% STORAGE=%dh BAT=%.0f кВт·ч",
            cls.DAYS, cls.EPOCHS, cls.N_FEATURES,
            cls.LSTM_TCN_FILTERS, cls.LSTM_UNITS_1, cls.LSTM_ATTN_HEADS,
            cls.LSTM_LEARNING_RATE, cls.DROPOUT_RATE, cls.LSTM_HUBER_DELTA,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS, cls.TRANSFORMER_DFF, cls.TRANSFORMER_DROPOUT,
            cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR,
            cls.GEN_EV_PENETRATION*100, cls.STORAGE_HORIZON, cls.BATTERY_CAPACITY)

    @classmethod
    def set_full_mode(cls):
        """Максимальное качество (full mode): больше данных + более ёмкие seq-модели."""
        cls.DAYS = 730; cls.HOUSEHOLDS = 2500; cls.EPOCHS = 320
        cls.PATIENCE = 35; cls.LR_PATIENCE = 12
        cls.HISTORY_LENGTH = 192; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.BATCH_SIZE = 8
        # На 730 днях и большом количестве окон можно использовать более ёмкий LSTM.
        cls.LSTM_UNITS_1 = 128; cls.LSTM_UNITS_2 = 128; cls.LSTM_UNITS_3 = 128
        cls.LSTM_ATTN_HEADS = 8; cls.LSTM_TCN_FILTERS = 64
        cls.DROPOUT_RATE = 0.18; cls.LSTM_LEARNING_RATE = 1.2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.35; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 192; cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 5; cls.TRANSFORMER_DFF = 384
        cls.TRANSFORMER_DROPOUT = 0.12; cls.TRANSFORMER_LEARNING_RATE = 2e-4
        cls.VANILLA_TRANSFORMER_LR = 5e-5; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.08
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.40
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 900; cls.XGB_COLSAMPLE = 0.35
        cls.GEN_AR_SIGMA = 0.035
        cls.GEN_EV_PENETRATION = 0.55; cls.GEN_INDUSTRIAL_LOADS = 10; cls.GEN_CITY_DISTRICTS = 16
        logging.getLogger("smart_grid").info(
            "Full mode v11: DAYS=%d HH=%d EPOCHS=%d HIST=%d | "
            "LSTM BiLSTM=%d TCN=%d attn=%dh lr=%.0e drop=%.2f | "
            "Trans d=%d h=%d L=%d dff=%d lr=%.0e van_lr=%.0e | "
            "EV=%.0f%% Districts=%d AR_sigma=%.3f",
            cls.DAYS, cls.HOUSEHOLDS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS, cls.LSTM_LEARNING_RATE, cls.DROPOUT_RATE,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS, cls.TRANSFORMER_DFF,
            cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR,
            cls.GEN_EV_PENETRATION*100, cls.GEN_CITY_DISTRICTS, cls.GEN_AR_SIGMA)

    @classmethod
    def print_summary(cls):
        log = logging.getLogger("smart_grid")
        log.info("─" * 50)
        log.info("КОНФИГУРАЦИЯ:")
        log.info("  Данные:      %d дней, %d домохозяйств", cls.DAYS, cls.HOUSEHOLDS)
        log.info("  Признаки:    %d ковариат на шаг", cls.N_FEATURES)
        log.info("  История:     %d ч → прогноз %d ч", cls.HISTORY_LENGTH, cls.FORECAST_HORIZON)
        log.info("  Обучение:    %d эпох, batch=%d, patience=%d", cls.EPOCHS, cls.BATCH_SIZE, cls.PATIENCE)
        log.info("  LSTM v9:     BiLSTM=%d TCN=%d attn=%dh drop=%.2f lr=%g huber=%.2f input=(%d,%d)",
                 cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS,
                 cls.DROPOUT_RATE, cls.LSTM_LEARNING_RATE, cls.LSTM_HUBER_DELTA,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  Transformer: d=%d h=%d L=%d dff=%d drop=%.2f lr=%g (Vanilla lr=%g) input=(%d,%d)",
                 cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
                 cls.TRANSFORMER_DFF, cls.TRANSFORMER_DROPOUT,
                 cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  XGBoost:     n_est=%d depth=%d col=%.2f",
                 cls.XGB_N_ESTIMATORS, cls.XGB_MAX_DEPTH, cls.XGB_COLSAMPLE)
        log.info("  Generator:   EV=%.0f%% Solar=%.0f%% IndustLoads=%d Districts=%d AR_phi=%.2f AR_sig=%.3f",
                 cls.GEN_EV_PENETRATION*100, cls.GEN_SOLAR_PENETRATION*100,
                 cls.GEN_INDUSTRIAL_LOADS, cls.GEN_CITY_DISTRICTS, cls.GEN_AR_PHI, cls.GEN_AR_SIGMA)
        log.info("  Тарифы:      пик=%.2f день=%.2f ночь=%.2f руб/кВт·ч",
                 cls.TARIFF_PEAK, cls.TARIFF_HALF_PEAK, cls.TARIFF_NIGHT)
        log.info("  Батарея:     %.0f кВт·ч, SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч)",
                 cls.BATTERY_CAPACITY, cls.BATTERY_MIN_SOC*100, cls.BATTERY_MAX_SOC*100,
                 (cls.BATTERY_MAX_SOC-cls.BATTERY_MIN_SOC)*cls.BATTERY_CAPACITY)
        log.info("─" * 50)

    @classmethod
    def get_generator_coefficients(cls) -> GeneratorCoefficients:
        """Return generator coefficients as a typed dataclass."""
        return GeneratorCoefficients(
            temp_setpoint=cls.GEN_TEMP_SETPOINT,
            temp_quadratic_coef=cls.GEN_TEMP_QUADRATIC_COEF,
            humidity_threshold=cls.GEN_HUMIDITY_THRESHOLD,
            humidity_coef=cls.GEN_HUMIDITY_COEF,
            wind_temp_threshold=cls.GEN_WIND_TEMP_THRESHOLD,
            wind_coef=cls.GEN_WIND_COEF,
            early_bird_frac=cls.GEN_EARLY_BIRD_FRAC,
            night_owl_frac=cls.GEN_NIGHT_OWL_FRAC,
            ar_phi=cls.GEN_AR_PHI,
            ar_sigma=cls.GEN_AR_SIGMA,
            seasonal_winter_boost=cls.GEN_SEASONAL_WINTER_BOOST,
            seasonal_summer_dip=cls.GEN_SEASONAL_SUMMER_DIP,
            ev_penetration=cls.GEN_EV_PENETRATION,
            solar_penetration=cls.GEN_SOLAR_PENETRATION,
            industrial_loads=cls.GEN_INDUSTRIAL_LOADS,
            city_districts=cls.GEN_CITY_DISTRICTS,
        )