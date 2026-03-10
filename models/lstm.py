# -*- coding: utf-8 -*-
"""
models/lstm.py — LSTM для мультивариантного прогнозирования.

ИСПРАВЛЕНИЯ v3:
  1. learning_rate: 5e-4 → 1e-4
     Причина: при мультивариантном входе (48,9) пространство потерь более
     сложное. lr=5e-4 вызывало расхождение уже в первых эпохах (R²=0.06).

  2. Huber delta: 1.0 → 0.1
     Причина: targets нормализованы в [0,1], типичная ошибка ≈ 0.05–0.15.
     При delta=1.0 Huber = MSE на всём диапазоне → не защищает от выбросов.
     При delta=0.1 граница L1/L2 находится в правильной точке.

  3. BatchNormalization между LSTM: убрана.
     Причина: BN между LSTM-слоями нарушает временну́ю структуру скрытых
     состояний. После BN hidden_state теряет масштаб → следующий LSTM
     получает нормализованный вход, теряя «память» о предыдущем контексте.
     Правильнее: LayerNormalization ВНУТРИ ячейки (что делает modern LSTM),
     либо отсутствие нормализации между слоями.

  4. dropout_rate: 0.30 → 0.20
     Причина: с 9 входными признаками регуляризация обеспечивается
     разнообразием фичей. Dropout 0.30 слишком агрессивен → underfitting.
"""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("smart_grid.models.lstm")


def build_lstm_model(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 9,
    lstm_units_1: int = 256,
    lstm_units_2: int = 128,
    lstm_units_3: int = 64,
    dropout_rate: float = 0.20,        # было 0.30 — исправлено
    learning_rate: float = 1e-4,       # было 5e-4 — исправлено
    mc_dropout: bool = False,
) -> tf.keras.Model:
    """
    Трёхслойный LSTM для мультивариантного прогноза.

    Архитектура:
        Input(T, n_features)
        → LSTM(256, return_sequences=True) → Dropout(0.20)
        → LSTM(128, return_sequences=True) → Dropout(0.20)
        → LSTM(64)                          → Dropout(0.20)
        → Dense(128, relu, L2=1e-4)        → Dropout(0.10)
        → Dense(64, relu)
        → Dense(forecast_horizon)

    Обоснование отсутствия BatchNorm между LSTM:
        BN нормализует hidden state между слоями, разрушая «память» контекста.
        Регуляризация достигается через Dropout и L2 в Dense-голове.

    Parameters
    ----------
    learning_rate : 1e-4 (оптимально для мультивариантного входа)
    dropout_rate  : 0.20 (снижено с 0.30 — n_features уже обеспечивает регуляризацию)
    """
    training_flag: Optional[bool] = True if mc_dropout else None

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # ── Блок 1 ────────────────────────────────────────────────────────────────
    x = tf.keras.layers.LSTM(
        lstm_units_1, return_sequences=True,
        recurrent_dropout=0.05,   # лёгкий dropout в рекуррентных связях
        name="lstm_1",
    )(inp)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_1")(x, training=training_flag)

    # ── Блок 2 ────────────────────────────────────────────────────────────────
    x = tf.keras.layers.LSTM(
        lstm_units_2, return_sequences=True,
        recurrent_dropout=0.05,
        name="lstm_2",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_2")(x, training=training_flag)

    # ── Блок 3 ────────────────────────────────────────────────────────────────
    x = tf.keras.layers.LSTM(lstm_units_3, name="lstm_3")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_3")(x, training=training_flag)

    # ── Dense-голова ──────────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = tf.keras.layers.Dropout(0.10, name="drop_dense")(x, training=training_flag)
    x = tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_2",
    )(x)
    out = tf.keras.layers.Dense(forecast_horizon, name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="LSTM_SmartGrid")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=0.1),   # delta=0.1 для нормализованного [0,1]
        metrics=["mae", "mape"],
    )

    n_params = model.count_params()
    logger.info("LSTM модель построена: %d параметров | input=(%d, %d) | lr=%.0e | Huber δ=0.1",
                n_params, history_length, n_features, learning_rate)
    return model
