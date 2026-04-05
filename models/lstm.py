# -*- coding: utf-8 -*-
"""models/lstm.py — TCN-BiLSTM-Attention v8 (Keras 3 compatible)."""

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

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        dilation_rate: int,
        dropout_rate: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        pad = (kernel_size - 1) * dilation_rate
        self.pad1 = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))
        self.conv1 = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=False,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("gelu")
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.pad2 = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))
        self.conv2 = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            padding="valid",
            use_bias=False,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation("gelu")

        self.proj = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False)
        self.bn_proj = tf.keras.layers.BatchNormalization()

    def call(self, x, training=None):
        res = self.bn_proj(self.proj(x), training=training)
        h = self.conv1(self.pad1(x))
        h = self.act1(self.bn1(h, training=training))
        h = self.drop1(h, training=training)
        h = self.conv2(self.pad2(h))
        h = self.bn2(h, training=training)
        return self.act2(h + res)

    def get_config(self):
        return {
            **super().get_config(),
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout_rate": self.dropout_rate,
        }


class TemporalAttentionBlock(tf.keras.layers.Layer):
    """Temporal Multi-Head Attention over sequence hidden states."""

    def __init__(self, num_heads=8, key_dim=32, dropout=0.10, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self._attn_weights = None
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout,
        )
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, hidden_states, training=None):
        query = hidden_states[:, -1:, :]
        attn_out, self._attn_weights = self.mha(
            query=query,
            key=hidden_states,
            value=hidden_states,
            return_attention_scores=True,
            training=training,
        )
        context = tf.keras.ops.squeeze(attn_out, axis=1)
        return self.drop(self.ln(context), training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout": self.dropout_rate,
        }


class SeasonalSkipConnection(tf.keras.layers.Layer):
    """Convex blend of neural forecast and seasonal naive baseline."""

    def __init__(self, init_neural_weight: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.init_neural_weight = float(init_neural_weight)

    def build(self, input_shape):
        init_logit = float(tf.math.log(self.init_neural_weight / (1.0 - self.init_neural_weight)))
        self.logit = self.add_weight(
            name="blend_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(init_logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        neural_out, naive_out = inputs
        w_neural = tf.keras.activations.sigmoid(self.logit)
        return w_neural * neural_out + (1.0 - w_neural) * naive_out

    def get_config(self):
        return {**super().get_config(), "init_neural_weight": self.init_neural_weight}


def build_lstm_model(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 26,
    lstm_units_1: int = 128,
    lstm_units_2: int = 64,  # compat
    lstm_units_3: int = 64,  # compat
    dropout_rate: float = 0.20,
    learning_rate: float = 2e-4,
    attn_heads: int = 8,
    use_cosine_decay: bool = False,  # compat
    total_steps: int = 37_000,  # compat
    mc_dropout: bool = False,
    huber_delta: float = 0.05,
    tcn_filters: int = 64,
    use_seasonal_skip: bool = True,
    seasonal_blend_init: float = 0.35,
) -> tf.keras.Model:
    del lstm_units_2, lstm_units_3, use_cosine_decay, total_steps
    training_flag: Optional[bool] = True if mc_dropout else None
    kops = tf.keras.ops

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # RevIN-like per-window normalization for consumption channel.
    cons_slice = inp[:, :, :1]
    cov_slice = inp[:, :, 1:]

    mean_cons = kops.mean(cons_slice, axis=1, keepdims=True)
    std_cons = kops.std(cons_slice, axis=1, keepdims=True) + 1e-6
    cons_norm = (cons_slice - mean_cons) / std_cons
    x = tf.keras.layers.Concatenate(axis=-1, name="revin_concat")([cons_norm, cov_slice])

    # Multi-scale dilated TCN.
    tcn_proj = tf.keras.layers.Conv1D(
        tcn_filters,
        1,
        padding="same",
        name="tcn_input_proj",
    )(x)
    branches = []
    for d in [1, 2, 4, 8]:
        branch = TCNBlock(
            tcn_filters,
            kernel_size=3,
            dilation_rate=d,
            dropout_rate=0.04,
            name=f"tcn_d{d}",
        )(tcn_proj, training=training_flag)
        branches.append(branch)

    tcn_out = tf.keras.layers.Concatenate(axis=-1, name="tcn_merge")(branches)
    tcn_out = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="tcn_ln")(tcn_out)
    tcn_out = tf.keras.layers.Dropout(0.10, name="tcn_drop")(tcn_out, training=training_flag)

    # BiLSTM encoder.
    bilstm_out = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            lstm_units_1,
            return_sequences=True,
            recurrent_dropout=0.0,
        ),
        name="bilstm_1",
    )(tcn_out)
    bilstm_out = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="bilstm_ln")(bilstm_out)

    # Temporal attention.
    key_dim = max((lstm_units_1 * 2) // attn_heads, 8)
    context = TemporalAttentionBlock(
        num_heads=attn_heads,
        key_dim=key_dim,
        dropout=dropout_rate * 0.5,
        name="temporal_attn",
    )(bilstm_out, training=training_flag)

    last_token = tf.keras.layers.Dropout(dropout_rate * 0.5, name="drop_last")(
        bilstm_out[:, -1, :],
        training=training_flag,
    )
    agg = tf.keras.layers.Concatenate(name="agg")([last_token, context])

    # Dense head.
    h = tf.keras.layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-6),
        name="head_d1",
    )(agg)
    h = tf.keras.layers.Dropout(dropout_rate, name="head_drop1")(h, training=training_flag)

    h = tf.keras.layers.Dense(
        128,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-6),
        name="head_d2",
    )(h)
    h = tf.keras.layers.Dropout(dropout_rate * 0.5, name="head_drop2")(h, training=training_flag)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="neural_output")(h)

    if use_seasonal_skip and history_length >= forecast_horizon:
        naive_slice = cons_norm[:, -forecast_horizon:, 0]
        mean_vec = kops.squeeze(mean_cons, axis=(1, 2))
        std_vec = kops.squeeze(std_cons, axis=(1, 2))

        naive_denorm = naive_slice * kops.expand_dims(std_vec, axis=-1) + kops.expand_dims(mean_vec, axis=-1)
        neural_denorm = neural_out * kops.expand_dims(std_vec, axis=-1) + kops.expand_dims(mean_vec, axis=-1)

        mixed = SeasonalSkipConnection(init_neural_weight=seasonal_blend_init, name="seasonal_skip")([neural_denorm, naive_denorm])
        final_out = (mixed - kops.expand_dims(mean_vec, axis=-1)) / kops.expand_dims(std_vec, axis=-1)
        final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)
    else:
        final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(neural_out)

    model = tf.keras.Model(inputs=inp, outputs=final_out, name="TCN_BiLSTM_Attention_v8")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta, name=f"huber_d{huber_delta}"),
        metrics=["mae", "mape"],
    )

    logger.info(
        "TCN-BiLSTM-Attention v8 | %d params | input=(%d,%d) | "
        "TCN filters=%d | BiLSTM=%d | Attn heads=%d key_dim=%d | "
        "Dense 256/128 drop=%.2f | SeasonalSkip=%s blend_init=%.2f | lr=%.0e | Huber(δ=%.2f)",
        model.count_params(),
        history_length,
        n_features,
        tcn_filters,
        lstm_units_1,
        attn_heads,
        key_dim,
        dropout_rate,
        use_seasonal_skip,
        seasonal_blend_init,
        learning_rate,
        huber_delta,
    )
    return model
