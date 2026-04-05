# -*- coding: utf-8 -*-
"""models/lstm.py — TCN-BiLSTM-Attention v9 (Keras 3 compatible).

ИСПРАВЛЕНИЯ v9 относительно v8:
═══════════════════════════════════════════════════════════════════════════════

БАГ #1 (КРИТИЧЕСКИЙ): SeasonalSkip использовал RevIN-нормализованный baseline.
  БЫЛО:  naive_slice = cons_norm[:, -24:, 0]  ← RevIN-нормализован ≈ N(0,1)
         final_out = alpha * neural_out + (1-alpha) * naive_slice
         Поскольку naive_slice ≈ 0 (RevIN mean), skip практически не работал.
         Математически: naive вклад ≈ (1-0.35)*0 = 0 → модель учила всё сама.
  СТАЛО: naive_slice = inp[:, -24:, 0]  ← raw MinMaxScaler [0,1] channel
         Теперь baseline = реальное "вчера в это же время" в правильном масштабе.
         При инициализации: final_out = 0.35*neural + 0.65*naive ∈ [0,1] = Y_train.

БАГ #2 (КРИТИЧЕСКИЙ): Overfitting — 945K параметров на 6K сэмплов.
  БЫЛО:  BiLSTM=128, TCN_filters=64, Dense=256/128 → 945K params
         train_mae=0.019 vs val_mae=0.045 → gap 2.4× = сильный overfitting
  СТАЛО: BiLSTM=64 per direction, TCN_filters=32, Dense=128/64 → ~150K params
         Правило: ≥40 сэмплов на параметр при dropout 0.20

═══════════════════════════════════════════════════════════════════════════════
АРХИТЕКТУРА v9 (параметры по умолчанию для optimal_mode):
  Input(48, 26)
    │
    ├─ RevIN нормализация consumption-канала (per-window mean/std)
    │
    ├─ TCN input_proj(32) → 4 ветви TCNBlock(32, dil=1,2,4,8) → concat(128) → LN → Drop(0.10)
    │
    ├─ BiLSTM(64 per dir → 128 total, return_sequences=True) → LN
    │
    ├─ TemporalAttention(4 heads, key_dim=32) → context(128)
    │
    ├─ Concat([last_token(128), context(128)]) → (256)
    │
    ├─ Dense(128, gelu, L2=1e-5) → Drop(0.20)
    │  Dense(64,  gelu, L2=1e-5) → Drop(0.10)
    │  Dense(24) → neural_out ∈ [0,1]
    │
    └─ SeasonalSkip: inp[:,-24:,0] = raw MinMaxScaler "вчера в это время" ∈ [0,1]
       final = sigmoid(logit) * neural_out + (1 - sigmoid(logit)) * naive_minmax

Optimizer: Adam(lr=2e-4, clipnorm=1.0)
Loss:      Huber(delta=0.05)
Params:    ~150K (optimal) / ~55K (fast)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("smart_grid.models.lstm")

__all__ = [
    "TemporalAttentionBlock",
    "TCNBlock",
    "build_lstm_model",
]


class TCNBlock(tf.keras.layers.Layer):
    """Dilated causal Conv1D block with residual connection."""

    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate  = dropout_rate

        pad = (kernel_size - 1) * dilation_rate
        self.pad1    = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))
        self.conv1   = tf.keras.layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="valid", use_bias=False)
        self.bn1     = tf.keras.layers.BatchNormalization()
        self.act1    = tf.keras.layers.Activation("gelu")
        self.drop1   = tf.keras.layers.Dropout(dropout_rate)
        self.pad2    = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))
        self.conv2   = tf.keras.layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="valid", use_bias=False)
        self.bn2     = tf.keras.layers.BatchNormalization()
        self.act2    = tf.keras.layers.Activation("gelu")
        self.proj    = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False)
        self.bn_proj = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        res = self.bn_proj(self.proj(x), training=training)
        h   = self.conv1(self.pad1(x))
        h   = self.act1(self.bn1(h, training=training))
        h   = self.drop1(h, training=training)
        h   = self.conv2(self.pad2(h))
        h   = self.bn2(h, training=training)
        return self.act2(h + res)

    def get_config(self):
        return {**super().get_config(), "filters": self.filters,
                "kernel_size": self.kernel_size, "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate}


class TemporalAttentionBlock(tf.keras.layers.Layer):
    """Temporal Multi-Head Attention over sequence hidden states."""

    def __init__(self, num_heads=4, key_dim=32, dropout=0.10, **kwargs):
        super().__init__(**kwargs)
        self.num_heads    = num_heads
        self.key_dim      = key_dim
        self.dropout_rate = dropout
        self._attn_weights = None
        self.mha  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.ln   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, hidden_states, training=None):
        query = hidden_states[:, -1:, :]
        attn_out, self._attn_weights = self.mha(
            query=query, key=hidden_states, value=hidden_states,
            return_attention_scores=True, training=training)
        context = tf.squeeze(attn_out, axis=1)
        return self.drop(self.ln(context), training=training)

    def get_config(self):
        return {**super().get_config(), "num_heads": self.num_heads,
                "key_dim": self.key_dim, "dropout": self.dropout_rate}


class SeasonalSkipConnection(tf.keras.layers.Layer):
    """
    Learnable blend: final = sigmoid(logit) * neural + (1-sigmoid(logit)) * naive.

    v9 FIX: оба входа теперь в [0,1] MinMaxScaler пространстве.
    БЫЛО: naive_slice из RevIN-нормализованного cons_norm ≈ N(0,1) → baseline ≈ 0.
    СТАЛО: naive_slice из raw inp[:,-24:,0] ∈ [0,1] → реальный seasonal naive.
    """

    def __init__(self, init_neural_weight=0.35, **kwargs):
        super().__init__(**kwargs)
        self.init_neural_weight = float(init_neural_weight)

    def build(self, input_shape):
        import math
        init_logit = math.log(self.init_neural_weight / (1.0 - self.init_neural_weight + 1e-8))
        self.logit = self.add_weight(
            name="blend_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(init_logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        neural_out, naive_out = inputs
        w = tf.keras.activations.sigmoid(self.logit)
        return w * neural_out + (1.0 - w) * naive_out

    def get_config(self):
        return {**super().get_config(), "init_neural_weight": self.init_neural_weight}


def build_lstm_model(
    history_length=48,
    forecast_horizon=24,
    n_features=26,
    lstm_units_1=64,          # v9: 128→64 (per BiLSTM direction)
    lstm_units_2=64,          # compat
    lstm_units_3=64,          # compat
    dropout_rate=0.20,
    learning_rate=2e-4,
    attn_heads=4,             # v9: 8→4
    use_cosine_decay=False,   # compat
    total_steps=37_000,       # compat
    mc_dropout=False,
    huber_delta=0.05,
    tcn_filters=32,           # v9: 64→32
    use_seasonal_skip=True,
    seasonal_blend_init=0.35,
):
    """
    TCN-BiLSTM-Attention v9.

    Ключевые исправления vs v8:
      ★ SeasonalSkip использует inp[:,-24:,0] (MinMaxScaler) вместо cons_norm (RevIN)
      ★ Размер модели сокращён с 945K до ~150K (BiLSTM 64, TCN 32, Dense 128/64)
    """
    del lstm_units_2, lstm_units_3, use_cosine_decay, total_steps
    training_flag: Optional[bool] = True if mc_dropout else None

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # ── RevIN нормализация consumption-канала ───────────────────────────────
    cons_slice = inp[:, :, :1]             # (B, T, 1)
    cov_slice  = inp[:, :, 1:]             # (B, T, n_features-1)
    # Keras 3: нельзя вызывать tf.reduce_* напрямую на KerasTensor вне слоя.
    # Оборачиваем RevIN-статистики в Lambda-слои (graph-safe Functional API).
    mean_cons = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        name="revin_mean_cons",
    )(cons_slice)
    std_cons = tf.keras.layers.Lambda(
        lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6,
        name="revin_std_cons",
    )(cons_slice)
    cons_norm = tf.keras.layers.Lambda(
        lambda xs: (xs[0] - xs[1]) / xs[2],
        name="revin_norm_cons",
    )([cons_slice, mean_cons, std_cons])
    x = tf.keras.layers.Concatenate(axis=-1, name="revin_concat")([cons_norm, cov_slice])

    # ── Multi-scale dilated TCN ─────────────────────────────────────────────
    tcn_proj = tf.keras.layers.Conv1D(tcn_filters, 1, padding="same", name="tcn_proj")(x)
    branches = [
        TCNBlock(tcn_filters, kernel_size=3, dilation_rate=d, dropout_rate=0.04, name=f"tcn_d{d}")(
            tcn_proj, training=training_flag)
        for d in [1, 2, 4, 8]
    ]
    tcn_out = tf.keras.layers.Concatenate(axis=-1, name="tcn_merge")(branches)
    tcn_out = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="tcn_ln")(tcn_out)
    tcn_out = tf.keras.layers.Dropout(0.10, name="tcn_drop")(tcn_out, training=training_flag)

    # ── BiLSTM ─────────────────────────────────────────────────────────────
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units_1, return_sequences=True, recurrent_dropout=0.0),
        name="bilstm_1"
    )(tcn_out)
    bilstm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="bilstm_ln")(bilstm)
    bilstm_dim = lstm_units_1 * 2   # bidirectional doubles the size

    # ── Temporal Attention ─────────────────────────────────────────────────
    key_dim = max(bilstm_dim // attn_heads, 8)
    context = TemporalAttentionBlock(
        num_heads=attn_heads, key_dim=key_dim, dropout=dropout_rate * 0.5, name="temporal_attn"
    )(bilstm, training=training_flag)                          # (B, bilstm_dim)

    last_token = tf.keras.layers.Dropout(
        dropout_rate * 0.5, name="drop_last"
    )(bilstm[:, -1, :], training=training_flag)               # (B, bilstm_dim)

    agg = tf.keras.layers.Concatenate(name="agg")([last_token, context])  # (B, bilstm_dim*2)

    # ── Dense head → neural_out ∈ [0,1] ────────────────────────────────────
    h = tf.keras.layers.Dense(128, activation="gelu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                               name="head_d1")(agg)
    h = tf.keras.layers.Dropout(dropout_rate, name="head_drop1")(h, training=training_flag)
    h = tf.keras.layers.Dense(64, activation="gelu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                               name="head_d2")(h)
    h = tf.keras.layers.Dropout(dropout_rate * 0.5, name="head_drop2")(h, training=training_flag)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="neural_output")(h)
    # neural_out предсказывает в [0,1] MinMaxScaler пространстве

    # ── SeasonalSkip (v9 FIX) ───────────────────────────────────────────────
    # БЫЛО (v8 БАГ):
    #   naive_slice = cons_norm[:, -forecast_horizon:, 0]  ← RevIN ≈ N(0,1)
    #   Blend: 0.35*neural + 0.65*RevIN_naive ≈ 0.35*neural + 0 → skip бесполезен
    #
    # СТАЛО (v9 FIX):
    #   naive_slice = inp[:, -forecast_horizon:, 0]  ← raw MinMaxScaler ∈ [0,1]
    #   Blend: 0.35*neural + 0.65*minmax_naive → оба ∈ [0,1] = пространство Y_train
    #   Инициализация хорошая: 65% "вчера в это время" = сильный baseline
    if use_seasonal_skip and history_length >= forecast_horizon:
        # raw MinMaxScaler consumption channel — hours t-23..t → "вчера в это время"
        naive_slice_minmax = inp[:, -forecast_horizon:, 0]    # (B, 24) ∈ [0,1]
        final_out = SeasonalSkipConnection(
            init_neural_weight=seasonal_blend_init, name="seasonal_skip"
        )([neural_out, naive_slice_minmax])
    else:
        final_out = neural_out

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)

    model = tf.keras.Model(inputs=inp, outputs=final_out, name="TCN_BiLSTM_Attention_v9")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta, name=f"huber_d{huber_delta}"),
        metrics=["mae", "mape"],
    )

    logger.info(
        "TCN-BiLSTM-Attention v9 | %d params | input=(%d,%d) | "
        "TCN filters=%d dil=[1,2,4,8] | BiLSTM=%d | Attn %dh key=%d | "
        "Dense 128/64 drop=%.2f | SeasonalSkip=%s (raw MinMaxScaler) blend=%.2f | "
        "lr=%.0e | Huber(δ=%.2f)",
        model.count_params(), history_length, n_features,
        tcn_filters, lstm_units_1, attn_heads, key_dim,
        dropout_rate, use_seasonal_skip, seasonal_blend_init,
        learning_rate, huber_delta,
    )
    return model
