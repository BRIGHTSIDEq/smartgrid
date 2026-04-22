# -*- coding: utf-8 -*-

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

    # LSTM v13 (TCN+BiLSTM+Attention)
    LSTM_UNITS_1: int = 96
    LSTM_UNITS_2: int = 64
    LSTM_UNITS_3: int = 64
    DROPOUT_RATE: float = 0.12
    LSTM_LEARNING_RATE: float = 2.0e-4
    LSTM_ATTN_HEADS: int = 4
    LSTM_USE_COSINE_DECAY: bool = False
    LSTM_WARMUP_RATIO: float = 0.05
    LSTM_TCN_FILTERS: int = 48
    LSTM_HUBER_DELTA: float = 0.05
    LSTM_SEASONAL_BLEND_INIT: float = 0.35

    # Transformer / iTransformer (общие)
    TRANSFORMER_D_MODEL: int = 192
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 5
    ITRANSFORMER_N_LAYERS: int = 4
    TRANSFORMER_DFF: int = 384
    TRANSFORMER_DROPOUT: float = 0.10
    TRANSFORMER_LEARNING_RATE: float = 2e-4
    TRANSFORMER_USE_COSINE_DECAY: bool = False
    TRANSFORMER_WARMUP_RATIO: float = 0.05
    VANILLA_TRANSFORMER_LR: float = 7e-5
    TRANSFORMER_STOCHASTIC_DEPTH: float = 0.06
    PATCHTST_USE_REVIN: bool = True
    VANILLA_USE_SEASONAL_RESIDUAL: bool = False
    VANILLA_SEASONAL_BLEND_INIT: float = 0.40
    VANILLA_HUBER_DELTA: float = 0.05

    # PatchTST — НОВЫЕ отдельные параметры
    PATCHTST_LEARNING_RATE: float = 3e-4    
    PATCHTST_DROPOUT: float = 0.05         
    PATCHTST_PATIENCE: int = 30             

    # XGBoost
    XGB_N_ESTIMATORS: int = 500
    XGB_MAX_DEPTH: int = 5
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.80
    XGB_COLSAMPLE: float = 0.40

    # Generator v6
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
    GEN_EV_PENETRATION: float   = 0.50
    GEN_SOLAR_PENETRATION: float = 0.22
    GEN_INDUSTRIAL_LOADS: int   = 8
    GEN_CITY_DISTRICTS: int = 12

    # BESS
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
        cls.PATIENCE = 20; cls.LR_PATIENCE = 8; cls.MIN_DELTA = 1e-5
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.BATCH_SIZE = 32
        cls.LSTM_UNITS_1 = 48; cls.LSTM_UNITS_2 = 48; cls.LSTM_UNITS_3 = 48
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 32
        cls.DROPOUT_RATE = 0.25; cls.LSTM_LEARNING_RATE = 2e-4
        cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_WARMUP_RATIO = 0.05
        cls.LSTM_SEASONAL_BLEND_INIT = 0.60; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 64; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 2; cls.ITRANSFORMER_N_LAYERS = 3
        cls.TRANSFORMER_DFF = 128
        cls.TRANSFORMER_DROPOUT = 0.20; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.TRANSFORMER_USE_COSINE_DECAY = False; cls.TRANSFORMER_WARMUP_RATIO = 0.05
        cls.VANILLA_TRANSFORMER_LR = 1e-4; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.05
        cls.PATCHTST_USE_REVIN = True
        cls.PATCHTST_LEARNING_RATE = 3e-4
        cls.PATCHTST_DROPOUT = 0.05
        cls.PATCHTST_PATIENCE = 20
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True
        cls.VANILLA_SEASONAL_BLEND_INIT = 0.60
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 300
        cls.GEN_EV_PENETRATION = 0.50; cls.GEN_INDUSTRIAL_LOADS = 4; cls.GEN_CITY_DISTRICTS = 8
        logging.getLogger("smart_grid").info(
            "Fast mode v17: DAYS=%d EPOCHS=%d HISTORY=%d | "
            "LSTM BiLSTM=%d TCN=%d seasonal_skip=True | "
            "PatchTST lr=%.0e drop=%.2f patience=%d",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS,
            cls.PATCHTST_LEARNING_RATE, cls.PATCHTST_DROPOUT, cls.PATCHTST_PATIENCE)

    @classmethod
    def set_optimal_mode(cls):
        """
        v17 изменения vs v16:
          PATCHTST_LEARNING_RATE = 3e-4  (было наследование из TRANSFORMER_LEARNING_RATE=1e-4)
            Причина: best_epoch=17–18 из 61–67 → PatchTST не успевал обучиться.
          PATCHTST_DROPOUT = 0.05  (было 0.08 из TRANSFORMER_DROPOUT)
            Причина: меньше регуляризации → быстрее сходимость на patch-токенах.
          PATCHTST_PATIENCE = 30  (было 50 из общего PATIENCE)
            Причина: PatchTST сходится быстро, ждать 50 эпох нет смысла.
        """
        cls.DAYS = 730; cls.HOUSEHOLDS = 2500; cls.EPOCHS = 300
        cls.PATIENCE = 50; cls.LR_PATIENCE = 12; cls.MIN_DELTA = 1e-5
        cls.HISTORY_LENGTH = 192; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.FORECAST_HORIZON = 24; cls.BATCH_SIZE = 16

        # LSTM v13
        cls.LSTM_UNITS_1 = 128; cls.LSTM_UNITS_2 = 128; cls.LSTM_UNITS_3 = 128
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 64
        cls.DROPOUT_RATE = 0.15
        cls.LSTM_LEARNING_RATE = 1e-4
        cls.LSTM_USE_COSINE_DECAY = True
        cls.LSTM_WARMUP_RATIO = 0.05
        cls.LSTM_SEASONAL_BLEND_INIT = 0.50
        cls.LSTM_HUBER_DELTA = 0.05

        # iTransformer v2 (общие для transformer)
        cls.TRANSFORMER_D_MODEL = 128; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 3
        cls.ITRANSFORMER_N_LAYERS = 4
        cls.TRANSFORMER_DFF = 256
        cls.TRANSFORMER_DROPOUT = 0.08
        cls.TRANSFORMER_LEARNING_RATE = 1.0e-4
        cls.VANILLA_TRANSFORMER_LR = 1e-4
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.08
        cls.TRANSFORMER_USE_COSINE_DECAY = True
        cls.TRANSFORMER_WARMUP_RATIO = 0.05
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True
        cls.VANILLA_SEASONAL_BLEND_INIT = 0.60
        cls.VANILLA_HUBER_DELTA = 0.05

        # PatchTST 
        cls.PATCHTST_LEARNING_RATE = 3e-4    
        cls.PATCHTST_DROPOUT = 0.05          
        cls.PATCHTST_PATIENCE = 30           

        # XGBoost
        cls.XGB_N_ESTIMATORS = 700; cls.XGB_COLSAMPLE = 0.50
        cls.GEN_AR_SIGMA = 0.040
        cls.GEN_EV_PENETRATION = 0.50; cls.GEN_INDUSTRIAL_LOADS = 8; cls.GEN_CITY_DISTRICTS = 12

        logging.getLogger("smart_grid").info(
            "Optimal mode v17: DAYS=%d HISTORY=%d EPOCHS=%d PATIENCE=%d BATCH=%d | "
            "LSTM TCN=%d BiLSTM=%d heads=%d lr=%.0e drop=%.2f blend=%.2f | "
            "iTransformer d=%d h=%d L=%d drop=%.2f lr=%.0e | "
            "PatchTST lr=%.0e drop=%.2f patience=%d [FIX v17: was best_epoch=17] | "
            "EV=%.0f%% BAT=%.0f кВт·ч",
            cls.DAYS, cls.HISTORY_LENGTH, cls.EPOCHS, cls.PATIENCE, cls.BATCH_SIZE,
            cls.LSTM_TCN_FILTERS, cls.LSTM_UNITS_1, cls.LSTM_ATTN_HEADS,
            cls.LSTM_LEARNING_RATE, cls.DROPOUT_RATE, cls.LSTM_SEASONAL_BLEND_INIT,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.ITRANSFORMER_N_LAYERS,
            cls.TRANSFORMER_DROPOUT, cls.VANILLA_TRANSFORMER_LR,
            cls.PATCHTST_LEARNING_RATE, cls.PATCHTST_DROPOUT, cls.PATCHTST_PATIENCE,
            cls.GEN_EV_PENETRATION * 100, cls.BATTERY_CAPACITY)

    @classmethod
    def set_full_mode(cls):
        """Максимальное качество (full mode, v17)."""
        cls.DAYS = 730; cls.HOUSEHOLDS = 2500; cls.EPOCHS = 400
        cls.PATIENCE = 50; cls.LR_PATIENCE = 14; cls.MIN_DELTA = 1e-5
        cls.HISTORY_LENGTH = 192; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.BATCH_SIZE = 8
        cls.LSTM_UNITS_1 = 128; cls.LSTM_UNITS_2 = 128; cls.LSTM_UNITS_3 = 128
        cls.LSTM_ATTN_HEADS = 8; cls.LSTM_TCN_FILTERS = 64
        cls.DROPOUT_RATE = 0.15
        cls.LSTM_LEARNING_RATE = 8e-5
        cls.LSTM_USE_COSINE_DECAY = True
        cls.LSTM_WARMUP_RATIO = 0.05
        cls.LSTM_SEASONAL_BLEND_INIT = 0.65; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 192; cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4; cls.ITRANSFORMER_N_LAYERS = 5
        cls.TRANSFORMER_DFF = 512
        cls.TRANSFORMER_DROPOUT = 0.10; cls.TRANSFORMER_LEARNING_RATE = 8e-5
        cls.VANILLA_TRANSFORMER_LR = 8e-5; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.08
        cls.TRANSFORMER_USE_COSINE_DECAY = True; cls.TRANSFORMER_WARMUP_RATIO = 0.05
        cls.PATCHTST_USE_REVIN = True
        cls.PATCHTST_LEARNING_RATE = 5e-4
        cls.PATCHTST_DROPOUT = 0.05
        cls.PATCHTST_PATIENCE = 35
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True
        cls.VANILLA_SEASONAL_BLEND_INIT = 0.65
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 900; cls.XGB_COLSAMPLE = 0.35
        cls.GEN_AR_SIGMA = 0.035
        cls.GEN_EV_PENETRATION = 0.55; cls.GEN_INDUSTRIAL_LOADS = 10; cls.GEN_CITY_DISTRICTS = 16
        logging.getLogger("smart_grid").info(
            "Full mode v17: DAYS=%d HH=%d EPOCHS=%d HIST=%d | "
            "LSTM BiLSTM=%d TCN=%d attn=%dh lr=%.0e blend=%.2f | "
            "PatchTST lr=%.0e drop=%.2f patience=%d | "
            "iTransformer d=%d h=%d L=%d lr=%.0e",
            cls.DAYS, cls.HOUSEHOLDS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS,
            cls.LSTM_LEARNING_RATE, cls.LSTM_SEASONAL_BLEND_INIT,
            cls.PATCHTST_LEARNING_RATE, cls.PATCHTST_DROPOUT, cls.PATCHTST_PATIENCE,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.ITRANSFORMER_N_LAYERS,
            cls.VANILLA_TRANSFORMER_LR)

    @classmethod
    def print_summary(cls):
        log = logging.getLogger("smart_grid")
        log.info("─" * 60)
        log.info("КОНФИГУРАЦИЯ:")
        log.info("  Данные:      %d дней, %d домохозяйств", cls.DAYS, cls.HOUSEHOLDS)
        log.info("  Признаки:    %d ковариат на шаг", cls.N_FEATURES)
        log.info("  История:     %d ч → прогноз %d ч", cls.HISTORY_LENGTH, cls.FORECAST_HORIZON)
        log.info("  Обучение:    %d эпох, batch=%d, patience=%d, min_delta=%.0e",
                 cls.EPOCHS, cls.BATCH_SIZE, cls.PATIENCE, cls.MIN_DELTA)
        log.info("  LSTM:        BiLSTM=%d TCN=%d attn=%dh drop=%.2f lr=%g huber=%.2f "
                 "blend=%.2f cosine=%s input=(%d,%d)",
                 cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS,
                 cls.DROPOUT_RATE, cls.LSTM_LEARNING_RATE, cls.LSTM_HUBER_DELTA,
                 cls.LSTM_SEASONAL_BLEND_INIT, cls.LSTM_USE_COSINE_DECAY,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  iTransformer: d=%d h=%d L=%d dff=%d drop=%.2f lr=%g "
                 "seasonal_residual=%s blend=%.2f input=(%d,%d)",
                 cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.ITRANSFORMER_N_LAYERS,
                 cls.TRANSFORMER_DFF, cls.TRANSFORMER_DROPOUT,
                 cls.VANILLA_TRANSFORMER_LR,
                 cls.VANILLA_USE_SEASONAL_RESIDUAL, cls.VANILLA_SEASONAL_BLEND_INIT,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  PatchTST:    lr=%g drop=%.2f patience=%d [v17 individual params]",
                 cls.PATCHTST_LEARNING_RATE, cls.PATCHTST_DROPOUT, cls.PATCHTST_PATIENCE)
        log.info("  XGBoost:     n_est=%d depth=%d col=%.2f",
                 cls.XGB_N_ESTIMATORS, cls.XGB_MAX_DEPTH, cls.XGB_COLSAMPLE)
        log.info("  Generator:   EV=%.0f%% Solar=%.0f%% IndustLoads=%d Districts=%d AR_phi=%.2f AR_sig=%.3f",
                 cls.GEN_EV_PENETRATION * 100, cls.GEN_SOLAR_PENETRATION * 100,
                 cls.GEN_INDUSTRIAL_LOADS, cls.GEN_CITY_DISTRICTS,
                 cls.GEN_AR_PHI, cls.GEN_AR_SIGMA)
        log.info("  Тарифы:      пик=%.2f день=%.2f ночь=%.2f руб/кВт·ч",
                 cls.TARIFF_PEAK, cls.TARIFF_HALF_PEAK, cls.TARIFF_NIGHT)
        log.info("  Батарея:     %.0f кВт·ч, SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч)",
                 cls.BATTERY_CAPACITY, cls.BATTERY_MIN_SOC * 100, cls.BATTERY_MAX_SOC * 100,
                 (cls.BATTERY_MAX_SOC - cls.BATTERY_MIN_SOC) * cls.BATTERY_CAPACITY)
        log.info("─" * 60)

    @classmethod
    def get_generator_coefficients(cls) -> GeneratorCoefficients:
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