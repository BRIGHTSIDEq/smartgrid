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
    N_FEATURES: int = 19           # v4.2: +load_lag_24h/+48h/+168h + load_diff_24h

    # ── Временные окна ────────────────────────────────────────────────────────
    HISTORY_LENGTH: int = 48
    FORECAST_HORIZON: int = 24
    STORAGE_HORIZON: int = 720          # 30 дней

    # ── Разделение ────────────────────────────────────────────────────────────
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    # ── Обучение (общее) ──────────────────────────────────────────────────────
    EPOCHS: int = 200
    BATCH_SIZE: int = 32
    # v6: стратегия EarlyStopping пересмотрена
    # БЫЛО: PATIENCE=40, MIN_DELTA=1e-4 — плохая пара.
    #   val_loss на MSE [0,1] естественно колеблется ±3e-4 между эпохами.
    #   min_delta=1e-4 < 3e-4 → каждое случайное колебание считается прогрессом
    #   → patience сбрасывается слишком часто → модель учится 200 эпох до overfitting.
    # СТАЛО: MIN_DELTA=0.0 + PATIENCE=20 — строгий и быстрый минимум.
    #   MIN_DELTA=0.0: принимаем любое улучшение, даже 1e-8 → не пропускаем настоящий минимум.
    #   PATIENCE=20: если 20 эпох подряд нет улучшения вообще → останавливаемся.
    #   Итог: чекпоинт = истинный минимум val_loss, не случайная флуктуация.
    PATIENCE: int = 20               # v6: было 40
    LR_PATIENCE: int = 8             # v6: было 10 — ReduceLR срабатывает раньше
    LR_FACTOR: float = 0.5
    MIN_DELTA: float = 0.0           # v6: было 1e-4 — принимаем любое улучшение

    # ── LSTM ──────────────────────────────────────────────────────────────────
    LSTM_UNITS_1: int = 160
    LSTM_UNITS_2: int = 96
    LSTM_UNITS_3: int = 64           # не используется в v7, оставлен для совместимости
    DROPOUT_RATE: float = 0.20
    LSTM_LEARNING_RATE: float = 3e-4         # v5: 3e-4 (1e-3 насыщает forget-gate в LSTM+Adam)
    LSTM_ATTN_HEADS: int = 4                 # Temporal Attention heads
    LSTM_USE_COSINE_DECAY: bool = False      # v5: ОТКЛЮЧЁН (несовместим с EarlyStopping)

    # ── Transformer ───────────────────────────────────────────────────────────
    TRANSFORMER_D_MODEL: int = 160
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 4
    TRANSFORMER_DFF: int = 320
    TRANSFORMER_DROPOUT: float = 0.20
    TRANSFORMER_LEARNING_RATE: float = 3e-4   # v4: 1e-4→3e-4
    TRANSFORMER_STOCHASTIC_DEPTH: float = 0.10  # v4: DropPath rate (последний блок)
    PATCHTST_USE_REVIN: bool = True          # v4: RevIN нормализация

    # ── XGBoost ───────────────────────────────────────────────────────────────
    XGB_N_ESTIMATORS: int = 500
    XGB_MAX_DEPTH: int = 5
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.80
    XGB_COLSAMPLE: float = 0.40

    # ── Генератор данных v4 ────────────────────────────────────────────────────
    # Температурный отклик (U-кривая)
    GEN_TEMP_SETPOINT: float = 18.0
    GEN_TEMP_QUADRATIC_COEF: float = 2.5e-4

    # Взаимодействие температура × влажность
    GEN_HUMIDITY_THRESHOLD: float = 60.0
    GEN_HUMIDITY_COEF: float = 0.30

    # Взаимодействие температура × ветер
    GEN_WIND_TEMP_THRESHOLD: float = 10.0
    GEN_WIND_COEF: float = 0.15

    # Типы домохозяйств
    GEN_EARLY_BIRD_FRAC: float = 0.28
    GEN_NIGHT_OWL_FRAC:  float = 0.20

    # Поведенческий AR(1) остаток
    GEN_AR_PHI: float   = 0.40
    GEN_AR_SIGMA: float = 0.022

    # Годовой seasonal drift
    GEN_SEASONAL_WINTER_BOOST: float = 0.15
    GEN_SEASONAL_SUMMER_DIP:   float = 0.10

    # ── Батарея ───────────────────────────────────────────────────────────────
    BATTERY_CAPACITY: float = 4_500.0
    BATTERY_MAX_POWER: float = 2_250.0
    BATTERY_EFFICIENCY: float = 0.95
    BATTERY_CYCLE_COST: float = 0.06
    BATTERY_COST_RUB: float = 45_000_000.0
    BATTERY_OM_SHARE: float = 0.015
    DEMAND_CHARGE_RUB_PER_KW_MONTH: float = 950.0
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
        """Быстрый режим: 365 дней, ~120 эпох. Для отладки и проверки пайплайна."""
        cls.DAYS = 365                   # ОБЯЗАТЕЛЬНО 365 — иначе seasonal shift (зима≠лето)
        cls.HOUSEHOLDS = 250
        cls.EPOCHS = 120
        cls.PATIENCE = 15                # v6: было 30
        cls.LR_PATIENCE = 6              # v6: было 10
        # ── ИСПРАВЛЕНИЕ v7 ──────────────────────────────────────────────────
        # БЫЛО: cls.LR_PATIENCE = 6  # v6: было 10        cls.HISTORY_LENGTH = 48
        # cls.HISTORY_LENGTH = 48 была поглощена комментарием предыдущей строки
        # и никогда не выполнялась. Разнесены на отдельные строки.
        cls.HISTORY_LENGTH = 48
        # ────────────────────────────────────────────────────────────────────
        cls.STORAGE_HORIZON = 720
        # LSTM v7: 2 слоя
        cls.LSTM_UNITS_1 = 128
        cls.LSTM_UNITS_2 = 64
        cls.LSTM_UNITS_3 = 48
        cls.LSTM_ATTN_HEADS = 4          # key_dim = 48//4 = 12
        cls.DROPOUT_RATE = 0.25
        cls.LSTM_LEARNING_RATE = 3e-4
        cls.LSTM_USE_COSINE_DECAY = False
        # Transformer
        cls.TRANSFORMER_D_MODEL = 96
        cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 2
        cls.TRANSFORMER_DFF = 192
        cls.TRANSFORMER_DROPOUT = 0.20
        cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.05
        cls.PATCHTST_USE_REVIN = True
        cls.XGB_N_ESTIMATORS = 300
        logging.getLogger("smart_grid").info(
            "Fast mode v4: DAYS=%d, EPOCHS=%d, HISTORY=%d | "
            "LSTM=%d/%d/%d attn=%dh | "
            "Trans d=%d h=%d L=%d sdrop=%.2f | RevIN=%s",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_UNITS_2, cls.LSTM_UNITS_3, cls.LSTM_ATTN_HEADS,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
            cls.TRANSFORMER_STOCHASTIC_DEPTH, cls.PATCHTST_USE_REVIN,
        )

    @classmethod
    def set_optimal_mode(cls) -> None:
        """Оптимальный режим: 365 дней, 200 эпох. Рекомендуется по умолчанию."""
        cls.DAYS = 365
        cls.HOUSEHOLDS = 500
        cls.EPOCHS = 200
        cls.PATIENCE = 20            # v6: было 40
        cls.LR_PATIENCE = 8          # v6: было 10
        cls.HISTORY_LENGTH = 48
        cls.STORAGE_HORIZON = 720
        # LSTM v7: 2 слоя 128/64
        cls.LSTM_UNITS_1 = 160
        cls.LSTM_UNITS_2 = 96
        cls.LSTM_UNITS_3 = 64
        cls.LSTM_ATTN_HEADS = 4          # key_dim = 64//4 = 16
        cls.DROPOUT_RATE = 0.20
        cls.LSTM_LEARNING_RATE = 3e-4
        cls.LSTM_USE_COSINE_DECAY = False
        # Transformer
        cls.TRANSFORMER_D_MODEL = 160
        cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4
        cls.TRANSFORMER_DFF = 320
        cls.TRANSFORMER_DROPOUT = 0.20
        cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.10
        cls.PATCHTST_USE_REVIN = True
        cls.XGB_N_ESTIMATORS = 500
        cls.XGB_COLSAMPLE = 0.40
        logging.getLogger("smart_grid").info(
            "Optimal mode v4: DAYS=%d, EPOCHS=%d, HISTORY=%d | "
            "LSTM=%d/%d/%d attn=%dh | "
            "Trans d=%d h=%d L=%d lr=%.0e sdrop=%.2f | RevIN=%s | "
            "STORAGE=%dh | BAT=%.0f кВт·ч",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_UNITS_2, cls.LSTM_UNITS_3, cls.LSTM_ATTN_HEADS,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
            cls.TRANSFORMER_LEARNING_RATE, cls.TRANSFORMER_STOCHASTIC_DEPTH,
            cls.PATCHTST_USE_REVIN, cls.STORAGE_HORIZON, cls.BATTERY_CAPACITY,
        )

    @classmethod
    def set_full_mode(cls) -> None:
        """Полный режим: 730 дней (2 года), 300 эпох. Финальный прогон для диплома."""
        cls.DAYS = 730
        cls.HOUSEHOLDS = 500
        cls.EPOCHS = 300
        cls.PATIENCE = 25            # v6: было 50
        cls.LR_PATIENCE = 8
        cls.HISTORY_LENGTH = 72          # 72ч = 3 суток контекста
        cls.STORAGE_HORIZON = 720
        # LSTM v7
        cls.LSTM_UNITS_1 = 192
        cls.LSTM_UNITS_2 = 96
        cls.LSTM_UNITS_3 = 96
        cls.LSTM_ATTN_HEADS = 4          # key_dim = 96//4 = 24
        cls.DROPOUT_RATE = 0.20
        cls.LSTM_LEARNING_RATE = 3e-4
        cls.LSTM_USE_COSINE_DECAY = False
        # Transformer
        cls.TRANSFORMER_D_MODEL = 128
        cls.TRANSFORMER_N_HEADS = 8
        cls.TRANSFORMER_N_LAYERS = 4
        cls.TRANSFORMER_DFF = 512
        cls.TRANSFORMER_DROPOUT = 0.20
        cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.10
        cls.PATCHTST_USE_REVIN = True
        cls.XGB_N_ESTIMATORS = 800
        cls.XGB_COLSAMPLE = 0.35
        logging.getLogger("smart_grid").info(
            "Full mode v4: DAYS=%d, EPOCHS=%d, HISTORY=%d | "
            "LSTM=%d/%d/%d | Trans d=%d h=%d L=%d dff=%d | RevIN=%s",
            cls.DAYS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_UNITS_2, cls.LSTM_UNITS_3,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS,
            cls.TRANSFORMER_N_LAYERS, cls.TRANSFORMER_DFF, cls.PATCHTST_USE_REVIN,
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
