# -*- coding: utf-8 -*-
"""
models/lstm.py — AttentionLSTM v6: устранение избыточной регуляризации.

═══════════════════════════════════════════════════════════════════════════════
ПРАВКА v6: train/val gap 14× → убираем источники подавления амплитуды
═══════════════════════════════════════════════════════════════════════════════

ДИАГНОЗ по логам (optimal mode, 72 эпохи):
  train_loss финишировал ~0.0005, val_loss=0.007 → разрыв 14×.
  EarlyStopping восстановил веса из эпохи 32 (val_loss=0.0039), когда модель
  ещё не обучила амплитуду. Следствие: синяя кривая «прилипает к среднему»,
  MAE=1574 против MAE=665 у LinearRegression.

ИСТОЧНИК ПРОБЛЕМЫ 1: Dropout(0.20) между LSTM-1 и LSTM-2
──────────────────────────────────────────────────────────
  Dropout на sequence (B, T, d) разрывает _временны́е зависимости_ внутри
  окна. В отличие от трансформера, у LSTM нет позиционного кодирования:
  скрытое состояние ячейки h_{t} несёт контекст от h_{t-1}, и случайное
  обнуление 20% нейронов в каждый момент эффективно «рвёт память» на
  коротких горизонтах. Это заставляет LSTM предсказывать среднее вместо пиков.

  БЫЛО: Dropout(0.20) между слоями  → 1/5 hidden dim зануляется за шаг
  СТАЛО: Dropout(0.05) между слоями → редкий шум, не разрывает память

ИСТОЧНИК ПРОБЛЕМЫ 2: L2=1e-4 на Dense-слоях
──────────────────────────────────────────────
  L2-регуляризация на весах head_d1 и head_d2 штрафует за большие значения
  весов. При нормализованных целях [0,1] крупные пики потребления → большие
  выходные активации → высокий L2 штраф → оптимизатор «сжимает» предсказания
  к среднему вместо обучения амплитуды.

  БЫЛО: L2=1e-4 на Dense(128) и Dense(64)  → амплитуда подавляется
  СТАЛО: L2=1e-5 (×10 слабее)              → почти свободный выход

ИСТОЧНИКИ ПРОБЛЕМЫ 3 (v5, оставлены без изменений):
  CosineDecay+EarlyStopping → плоский LR=3e-4 + ReduceLROnPlateau ✓
  Persistence prior bypass  → чистый Dense-output ✓
  3 LSTM-слоя (~600K params) → 2 слоя (128/64, ~120K params) ✓

═══════════════════════════════════════════════════════════════════════════════
АРХИТЕКТУРА v6 (изменения относительно v5 помечены ★)
═══════════════════════════════════════════════════════════════════════════════

Input(48, 15)
  │
  ├─ LSTM(128, return_sequences=True) → LayerNorm → Dropout(0.05) ★
  ├─ LSTM(64,  return_sequences=True) → LayerNorm
  │    └─ all_hidden_states: (B, 48, 64)
  │
  ├─ Temporal Attention: Q=x[:,-1,:], K=V=x → MHA(4 heads, key_dim=16)
  │    → squeeze → LN → Dropout(0.10) → context: (B, 64)
  │
  ├─ Concat([last_token(B,64), context(B,64)]) → (B, 128)
  │
  └─ Dense Head:
       Dense(128, gelu, L2=1e-5) ★ → Dropout(0.20)
       Dense(64,  gelu, L2=1e-5) ★ → Dropout(0.10)
       Dense(24)

Optimizer: Adam(lr=3e-4, clipnorm=1.0)
Loss:      MSE
Params:    ~120K (optimal) / ~50K (fast)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("smart_grid.models.lstm")

__all__ = ["TemporalAttentionBlock", "build_lstm_model"]


class TemporalAttentionBlock(tf.keras.layers.Layer):
    """
    Temporal Self-Attention поверх LSTM hidden states.
    Query = последний скрытый вектор, Key/Value = вся история.
    Сохраняет _attn_weights для визуализации.
    """

    def __init__(self, num_heads=4, key_dim=16, dropout=0.10, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self._attn_weights = None
        self.mha  = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.ln   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, hidden_states, training=None):
        query = hidden_states[:, -1:, :]   # (B, 1, d)
        attn_out, self._attn_weights = self.mha(
            query=query, key=hidden_states, value=hidden_states,
            return_attention_scores=True, training=training)
        context = tf.squeeze(attn_out, axis=1)              # (B, d)
        return self.drop(self.ln(context), training=training)

    def get_config(self):
        return {**super().get_config(),
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "dropout": self.dropout_rate}


def build_lstm_model(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 19,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,
    lstm_units_3: int = 64,        # принимается для совместимости с config, не используется
    dropout_rate: float = 0.20,
    learning_rate: float = 3e-4,   # v5: 3e-4 (1e-3 насыщает forget-gate в LSTM+Adam)
    attn_heads: int = 4,
    use_cosine_decay: bool = False, # v5: ОТКЛЮЧЁН — см. обоснование выше
    total_steps: int = 37_000,     # принимается для совместимости, не используется
    mc_dropout: bool = False,
) -> tf.keras.Model:
    """
    AttentionLSTM v5: 2 LSTM + Temporal Attention + чистый Dense-output.

    Отличия от v4:
      - 2 LSTM-слоя (128/64) вместо 3 (256/128/64)  → ~120K vs ~600K параметров
      - Нет CosineDecay                               → плоский LR, адаптивный ReduceLR
      - Нет persistence prior bypass                  → прямой Dense-выход
      - L2=1e-4 на Dense-слоях                        → сильнее регуляризация
    """
    training_flag: Optional[bool] = True if mc_dropout else None

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # ── LSTM 1 ─────────────────────────────────────────────────────────────────
    x = tf.keras.layers.LSTM(lstm_units_1, return_sequences=True,
                              recurrent_dropout=0.0, name="lstm_1")(inp)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x)
    # v6: Dropout 0.20→0.05 между LSTM-слоями.
    # Dropout на sequence (B,T,d) разрывает временные зависимости h_{t}→h_{t+1}.
    # У LSTM нет позиционного кодирования — скрытое состояние IS память.
    # 20% зануление каждые T шагов эффективно стирает контекст → предсказание среднего.
    # 5% — редкий шум для регуляризации, не разрушающий цепочку памяти.
    x = tf.keras.layers.Dropout(0.05, name="drop_1")(x, training=training_flag)

    # ── LSTM 2 ─────────────────────────────────────────────────────────────────
    x = tf.keras.layers.LSTM(lstm_units_2, return_sequences=True,
                              recurrent_dropout=0.0, name="lstm_2")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_2")(x)
    # x: (B, T, lstm_units_2)

    # ── Temporal Attention ─────────────────────────────────────────────────────
    key_dim = max(lstm_units_2 // attn_heads, 8)
    context = TemporalAttentionBlock(
        num_heads=attn_heads, key_dim=key_dim,
        dropout=dropout_rate * 0.5, name="temporal_attn",
    )(x, training=training_flag)   # (B, lstm_units_2)

    # ── Агрегация ──────────────────────────────────────────────────────────────
    last_token = tf.keras.layers.Dropout(
        dropout_rate * 0.5, name="drop_last"
    )(x[:, -1, :], training=training_flag)  # (B, lstm_units_2)

    agg = tf.keras.layers.Concatenate(name="agg")([last_token, context])
    # agg: (B, lstm_units_2 * 2)

    # ── Dense-голова ───────────────────────────────────────────────────────────
    # v6: L2=1e-4→1e-5.
    # L2=1e-4 штрафует за большие веса → выходные активации «сжимаются» к нулю
    # при нормализованных целях [0,1] → крупные пики подавляются.
    # L2=1e-5 — слабый сглаживающий эффект без подавления амплитуды.
    # Dropout 0.20/0.10 в голове оставлены: здесь dropout работает правильно
    # (на финальных представлениях, не на временных зависимостях).
    h = tf.keras.layers.Dense(
        128, activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),  # v6: 1e-4 → 1e-5
        name="head_d1")(agg)
    h = tf.keras.layers.Dropout(dropout_rate, name="head_drop1")(h, training=training_flag)

    h = tf.keras.layers.Dense(
        64, activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),  # v6: 1e-4 → 1e-5
        name="head_d2")(h)
    h = tf.keras.layers.Dropout(dropout_rate * 0.5, name="head_drop2")(h, training=training_flag)

    out = tf.keras.layers.Dense(forecast_horizon, name="output")(h)

    # Сезонный skip: последняя суточная траектория (t-24..t-1) как baseline.
    # Модель учит остаток (delta), что резко снижает автокорреляцию остатков.
    seasonal_base = inp[:, -forecast_horizon:, 0]
    out = tf.keras.layers.Add(name="seasonal_skip")([out, seasonal_base])

    model = tf.keras.Model(inputs=inp, outputs=out, name="AttentionLSTM_v7")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=["mae", "mape"],
    )

    logger.info(
        "AttentionLSTM v7 | %d параметров | input=(%d,%d) | "
        "LSTM=%d/%d seq_drop=0.05 | Attn heads=%d key_dim=%d | "
        "Dense L2=1e-5 drop=0.20/0.10 | SeasonalSkip=True | lr=%.0e | Huber(δ=0.05)",
        model.count_params(), history_length, n_features,
        lstm_units_1, lstm_units_2, attn_heads, key_dim, learning_rate,
    )
    return model
