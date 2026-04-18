# -*- coding: utf-8 -*-
"""
models/transformer.py — Сравнительное исследование архитектур Transformer
для прогнозирования временных рядов энергопотребления.

═══════════════════════════════════════════════════════════════════════════════
РЕАЛИЗОВАННЫЕ АРХИТЕКТУРЫ
═══════════════════════════════════════════════════════════════════════════════

1. VanillaTransformer  — Encoder-only с Pre-LN и Sinusoidal/Time2Vec PE
2. TFTLite             — Temporal Fusion Transformer (упрощённый) с ковариатами
3. PatchTST            — State-of-the-art (Nie et al., 2023): патчи вместо точек

ИСПРАВЛЕНИЕ v5 в build_patchtst():
─────────────────────────────────────────────────────────────────────────────
RevIN применяется ТОЛЬКО к каналу consumption (channel 0), а не ко всем 26.

БЫЛА ПРОБЛЕМА:
  RevINNorm(eps=1e-5) применялась ко всем 26 каналам, включая бинарные:
    is_weekend (0/1), is_holiday (0/1), is_peak (0/1), is_night (0/1).
  Если весь window = одно значение (напр. весь выходной: is_weekend=1 везде),
  instance-std → 0, деление на eps=1e-5 → значения ×100 000 → взрыв градиентов.
  Результат: PatchTST best_epoch=23, val_loss=0.000442 (в 1.63× хуже LSTM).

ИСПРАВЛЕНО:
  cons_channel = inp[:, :, :1]      # только потребление
  other_channels = inp[:, :, 1:]    # ковариаты без нормализации
  revin_norm применяется только к cons_channel.
  Затем нормализованный cons конкатенируется с исходными ковариатами.
  RevINDenorm также работает только с channel 0 (consumption).

НОВЫЕ КОМПОНЕНТЫ v4
─────────────────────────────────────────────────────────────────────────────
• RevIN (Reversible Instance Normalization) — Kim et al., 2022
• StochasticDepth (DropPath) — Huang et al., 2016
• LearnedQueryPooling — вместо GlobalAveragePooling в голове
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger("smart_grid.models.transformer")


# ══════════════════════════════════════════════════════════════════════════════
# УТИЛИТЫ
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: tf.keras.Model) -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


# ══════════════════════════════════════════════════════════════════════════════
# НОВЫЕ КОМПОНЕНТЫ v4
# ══════════════════════════════════════════════════════════════════════════════

class RevINNorm(tf.keras.layers.Layer):
    """
    RevIN Normalization [Kim et al., ICLR 2022] — НОРМАЛИЗАЦИЯ.

    Graph-compatible версия для Keras Functional API.
    Возвращает (x_norm, mean, std) как три отдельных тензора.

    ВАЖНО: применять ТОЛЬКО к непрерывным каналам (consumption).
    Бинарные каналы (is_weekend, is_holiday) нельзя нормализовать —
    если весь window = константа, instance-std → 0 → взрыв активаций.
    """

    def __init__(self, eps: float = 1e-5, affine: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shape: tuple) -> None:
        n_features = input_shape[-1]
        if self.affine:
            self.gamma = self.add_weight(
                shape=(n_features,), initializer="ones",  name="revin_gamma"
            )
            self.beta = self.add_weight(
                shape=(n_features,), initializer="zeros", name="revin_beta"
            )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tuple:
        mean = tf.reduce_mean(x, axis=1, keepdims=True)            # (B, 1, C)
        std  = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps
        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm, mean, std

    def get_config(self) -> dict:
        return {**super().get_config(), "eps": self.eps, "affine": self.affine}


class RevINDenorm(tf.keras.layers.Layer):
    """
    RevIN Denormalization — ДЕНОРМАЛИЗАЦИЯ.

    Принимает (pred_norm, mean, std) — все как граф-тензоры из RevINNorm —
    и возвращает прогноз в исходном масштабе.

    Работает только с channel 0 (consumption): m = mean[:, 0, 0], s = std[:, 0, 0].
    """

    def __init__(self, eps: float = 1e-5, affine: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shapes: list) -> None:
        super().build(input_shapes)

    def call(self, inputs: list) -> tf.Tensor:
        pred, mean, std = inputs
        m = mean[:, 0, 0]              # (B,) — mean consumption этого окна
        s = std[:, 0, 0]               # (B,) — std consumption этого окна
        m = tf.expand_dims(m, axis=1)  # (B, 1)
        s = tf.expand_dims(s, axis=1)  # (B, 1)
        return pred * s + m            # broadcast (B, 1) × (B, 24) → (B, 24)

    def get_config(self) -> dict:
        return {**super().get_config(), "eps": self.eps, "affine": self.affine}


# Backward-compatibility alias
RevIN = RevINNorm


class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth / DropPath [Huang et al., ECCV 2016]."""

    def __init__(self, drop_rate: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if not training or self.drop_rate == 0.0:
            return x
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        keep_prob = 1.0 - self.drop_rate
        random_tensor = tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.math.floor(random_tensor + keep_prob)
        return x * binary_tensor / keep_prob

    def get_config(self) -> dict:
        return {**super().get_config(), "drop_rate": self.drop_rate}


class LearnedQueryPooling(tf.keras.layers.Layer):
    """Обучаемый query-pooling вместо GlobalAveragePooling."""

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape: tuple) -> None:
        self.query = self.add_weight(
            shape=(self.d_model,), initializer="glorot_uniform", name="pooling_query"
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        scores = tf.einsum("btd,d->bt", x, self.query) / (self.d_model ** 0.5)
        weights = tf.nn.softmax(scores, axis=-1)
        context = tf.einsum("bt,btd->bd", weights, x)
        return context

    def get_config(self) -> dict:
        return {**super().get_config(), "d_model": self.d_model}


# ══════════════════════════════════════════════════════════════════════════════
# ПОЗИЦИОННЫЕ КОДИРОВКИ
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(tf.keras.layers.Layer):
    """Синусоидальное позиционное кодирование [Vaswani et al., 2017]."""

    def __init__(self, d_model: int, max_len: int = 512, **kwargs) -> None:
        super().__init__(trainable=False, **kwargs)
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self._pe = tf.constant(angles[np.newaxis], dtype=tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self._pe[:, :seq_len, :]

    def get_config(self) -> dict:
        return {**super().get_config(), "d_model": int(self._pe.shape[-1])}


class LearnableRelativePE(tf.keras.layers.Layer):
    """Обучаемые смещения по относительному расстоянию (T5-style)."""

    def __init__(self, num_heads: int, max_len: int = 512, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.max_len = max_len
        self.rel_bias = self.add_weight(
            name="rel_bias",
            shape=(2 * max_len - 1, num_heads),
            initializer="zeros",
            trainable=True,
        )

    def call(self, seq_len: int) -> tf.Tensor:
        range_vec = tf.range(seq_len)
        dist = range_vec[:, None] - range_vec[None, :]
        dist = tf.clip_by_value(dist + (self.max_len - 1), 0, 2 * self.max_len - 2)
        biases = tf.gather(self.rel_bias, dist)
        return tf.transpose(biases, [2, 0, 1])

    def get_config(self) -> dict:
        return {**super().get_config(), "num_heads": self.num_heads, "max_len": self.max_len}


class Time2Vec(tf.keras.layers.Layer):
    """Векторное представление времени [Kazemi et al., 2019]."""

    def __init__(self, output_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape) -> None:
        self.W = self.add_weight("W", (1, self.output_dim), "glorot_uniform")
        self.b = self.add_weight("b", (1, self.output_dim), "zeros")
        super().build(input_shape)

    def call(self, tau: tf.Tensor) -> tf.Tensor:
        x = tau * self.W + self.b
        return tf.concat([x[:, :, :1], tf.sin(x[:, :, 1:])], axis=-1)

    def get_config(self) -> dict:
        return {**super().get_config(), "output_dim": self.output_dim}


# ══════════════════════════════════════════════════════════════════════════════
# МЕХАНИЗМ ВНИМАНИЯ: ProbSparse (Informer)
# ══════════════════════════════════════════════════════════════════════════════

class ProbSparseAttention(tf.keras.layers.Layer):
    """ProbSparse Self-Attention [Zhou et al., 2021 — Informer]."""

    def __init__(
        self, d_model: int, num_heads: int, factor: int = 5, dropout: float = 0.1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.factor = factor
        self.head_dim = d_model // num_heads

        self.Wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.Wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.Wv = tf.keras.layers.Dense(d_model, use_bias=False)
        self.out_proj = tf.keras.layers.Dense(d_model)
        self.drop = tf.keras.layers.Dropout(dropout)

    def _split(self, x: tf.Tensor, B: int) -> tf.Tensor:
        T = tf.shape(x)[1]
        return tf.transpose(
            tf.reshape(x, (B, T, self.num_heads, self.head_dim)), [0, 2, 1, 3]
        )

    def call(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, training=None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        B = tf.shape(q)[0]
        Q = self._split(self.Wq(q), B)
        K = self._split(self.Wk(k), B)
        V = self._split(self.Wv(v), B)
        scale = tf.sqrt(tf.cast(self.head_dim, tf.float32))
        scores = tf.matmul(Q, K, transpose_b=True) / scale

        importance = tf.reduce_max(scores, -1) - tf.reduce_mean(scores, -1)
        L_Q = tf.shape(q)[1]
        u = tf.minimum(
            tf.cast(self.factor * tf.math.ceil(tf.math.log(tf.cast(L_Q, tf.float32))), tf.int32),
            L_Q,
        )
        _, top_idx = tf.math.top_k(importance, k=u)
        top_oh = tf.cast(
            tf.reduce_sum(tf.one_hot(top_idx, depth=L_Q, axis=-1), axis=-2) > 0,
            tf.float32,
        )
        big_neg = scores * 0.0 + (-1e9)
        masked_scores = tf.where(top_oh[:, :, :, tf.newaxis] > 0, scores, big_neg)
        attn = tf.nn.softmax(masked_scores, axis=-1)
        attn = self.drop(attn, training=training)
        context = tf.reshape(
            tf.transpose(tf.matmul(attn, V), [0, 2, 1, 3]),
            (B, L_Q, self.d_model),
        )
        return self.out_proj(context), attn

    def get_config(self) -> dict:
        return {**super().get_config(), "d_model": self.d_model, "num_heads": self.num_heads}


# ══════════════════════════════════════════════════════════════════════════════
# PRE-LN ENCODER BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class PreLNEncoderBlock(tf.keras.layers.Layer):
    """Encoder-блок с Pre-Layer Normalization + StochasticDepth v4."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        stochastic_depth_rate: float = 0.0,
        use_prob_sparse: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_prob_sparse = use_prob_sparse

        if use_prob_sparse:
            self.attn = ProbSparseAttention(d_model, num_heads, dropout=dropout)
        else:
            self.attn = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
            )

        self.ffn1 = tf.keras.layers.Dense(dff, activation="gelu")
        self.ffn_drop = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(d_model)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.sdrop1 = StochasticDepth(stochastic_depth_rate)
        self.sdrop2 = StochasticDepth(stochastic_depth_rate)

        self._attn_weights: Optional[tf.Tensor] = None

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        residual = x
        xn = self.ln1(x)
        if self.use_prob_sparse:
            attn_out, self._attn_weights = self.attn(xn, xn, xn, training=training)
        else:
            attn_out, self._attn_weights = self.attn(
                xn, xn, return_attention_scores=True, training=training
            )
        x = residual + self.sdrop1(attn_out, training=training)

        residual = x
        xn = self.ln2(x)
        ffn_out = self.ffn2(self.ffn_drop(self.ffn1(xn), training=training))
        x = residual + self.sdrop2(ffn_out, training=training)
        return x

    def get_attention_weights(self) -> Optional[tf.Tensor]:
        return self._attn_weights

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.dropout_rate,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "use_prob_sparse": self.use_prob_sparse,
        }


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА 1: VANILLA TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

def build_vanilla_transformer(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 11,
    d_model: int = 128,
    num_heads: int = 8,
    num_layers: int = 4,
    dff: int = 256,
    dropout: float = 0.10,
    learning_rate: float = 3e-4,
    pe_type: str = "sinusoidal",
    use_prob_sparse: bool = False,
    stochastic_depth_rate: float = 0.10,
    use_seasonal_residual: bool = False,
    seasonal_blend_init: float = 0.40,
    huber_delta: float = 0.05,
    use_autoregressive_shortcut: bool = True,
    use_local_conv: bool = True,
) -> tf.keras.Model:
    """
    Encoder-only Transformer v4 с StochasticDepth + LearnedQueryPooling.
    use_seasonal_residual=False по умолчанию (seasonal_diff на уровне данных).
    """
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} не делится на num_heads={num_heads}")

    inp_series = tf.keras.Input(shape=(history_length, n_features), name="series_input")
    inputs = [inp_series]

    x = tf.keras.layers.Dense(d_model, name="input_proj")(inp_series)
    if use_local_conv:
        local_conv = tf.keras.layers.Conv1D(
            d_model, kernel_size=3, padding="causal", activation="gelu", name="local_conv"
        )(inp_series)
        x = tf.keras.layers.Add(name="input_plus_local")([x, local_conv])

    if pe_type == "sinusoidal":
        x = SinusoidalPE(d_model, max_len=max(history_length + 4, 256), name="sin_pe")(x)
    elif pe_type == "time2vec":
        tau = tf.keras.Input(shape=(history_length, 1), name="time_index")
        inputs.append(tau)
        t2v = Time2Vec(d_model, name="time2vec")(tau)
        x = x + t2v

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x)
    x = tf.keras.layers.Dropout(dropout, name="embed_drop")(x)

    enc_blocks = []
    for i in range(num_layers):
        sdrop_rate = (i / max(num_layers - 1, 1)) * stochastic_depth_rate
        blk = PreLNEncoderBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout=dropout,
            stochastic_depth_rate=sdrop_rate,
            use_prob_sparse=use_prob_sparse,
            name=f"enc_{i}",
        )
        x = blk(x)
        enc_blocks.append(blk)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    last = x[:, -1, :]
    learned_ctx = LearnedQueryPooling(d_model, name="lqp")(x)
    agg = tf.keras.layers.Concatenate(name="agg")([last, learned_ctx])

    h = tf.keras.layers.Dense(dff // 2, activation="gelu", name="head_d1")(agg)
    skip = tf.keras.layers.Dense(dff // 2, use_bias=False, name="head_skip")(agg)
    h = tf.keras.layers.Add(name="head_res")([h, skip])
    h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="head_ln")(h)
    h = tf.keras.layers.Dropout(dropout, name="head_drop")(h)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="neural_output")(h)

    if use_autoregressive_shortcut:
        last_features = inp_series[:, -1, :]
        ar_shortcut = tf.keras.layers.Dense(
            forecast_horizon, use_bias=False, name="ar_shortcut"
        )(last_features)
        neural_out = tf.keras.layers.Add(name="neural_plus_ar")([neural_out, ar_shortcut])

    if use_seasonal_residual and history_length >= forecast_horizon:
        naive = inp_series[:, -forecast_horizon:, 0]
        blend_logit = tf.keras.layers.Dense(1, name="seasonal_blend_logit",
                                            bias_initializer=tf.keras.initializers.Constant(
                                                np.log(seasonal_blend_init / (1 - seasonal_blend_init))
                                            ))(agg)
        blend = tf.keras.layers.Activation("sigmoid", name="seasonal_blend")(blend_logit)
        output = blend * neural_out + (1.0 - blend) * naive

        if history_length >= (168 + forecast_horizon):
            weekly = inp_series[:, -168:-168 + forecast_horizon, 0]
            w2_logit = tf.keras.layers.Dense(1, name="weekly_blend_logit",
                                             bias_initializer=tf.keras.initializers.Constant(
                                                 np.log(0.80 / 0.20)))(agg)
            w2 = tf.keras.layers.Activation("sigmoid", name="weekly_blend")(w2_logit)
            output = w2 * output + (1.0 - w2) * weekly
    else:
        output = neural_out

    model = tf.keras.Model(inputs=inputs, outputs=output,
                           name=f"VanillaTransformer_v4_{pe_type}")
    model._enc_blocks = enc_blocks

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae", "mape"],
    )
    logger.info(
        "VanillaTransformer v4 [%s%s] d=%d h=%d L=%d sdrop=%.2f seasonal_residual=%s | %d params",
        pe_type, "+ProbSparse" if use_prob_sparse else "",
        d_model, num_heads, num_layers, stochastic_depth_rate, use_seasonal_residual,
        count_parameters(model),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА 2: TFT LITE
# ══════════════════════════════════════════════════════════════════════════════

class GatedResidualNetwork(tf.keras.layers.Layer):
    """Gated Residual Network (GRN) — ключевой блок TFT [Lim et al., 2021]."""

    def __init__(self, hidden: int, output_size: int, dropout: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(hidden, activation="elu")
        self.fc2 = tf.keras.layers.Dense(output_size)
        self.gate = tf.keras.layers.Dense(output_size, activation="sigmoid")
        self.proj = tf.keras.layers.Dense(output_size)
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        h = self.drop(self.fc1(x), training=training)
        h2 = self.fc2(h)
        g = self.gate(h2)
        return self.ln(self.proj(x) + g * h2)

    def get_config(self) -> dict:
        return {**super().get_config()}


def build_tft_lite(
    history_length: int = 48,
    forecast_horizon: int = 24,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.10,
    learning_rate: float = 1e-4,
    n_covariate_features: int = 4,
) -> tf.keras.Model:
    """Temporal Fusion Transformer Lite [Lim et al., 2021 — упрощённая версия]."""
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} не делится на num_heads={num_heads}")

    inp_series = tf.keras.Input(shape=(history_length, 1), name="series_input")
    inp_covars = tf.keras.Input(shape=(history_length, n_covariate_features), name="covariate_input")

    s_emb = tf.keras.layers.Dense(d_model)(inp_series)
    c_emb = tf.keras.layers.Dense(d_model)(inp_covars)

    s_enc = GatedResidualNetwork(d_model, d_model, dropout, name="grn_series")(s_emb)
    c_enc = GatedResidualNetwork(d_model, d_model, dropout, name="grn_covar")(c_emb)

    fused = tf.keras.layers.Add()([s_enc, c_enc])
    fused = tf.keras.layers.LayerNormalization(epsilon=1e-6)(fused)

    lstm_out = tf.keras.layers.LSTM(d_model, return_sequences=True)(fused)
    lstm_out = tf.keras.layers.Dropout(dropout)(lstm_out)

    gate = tf.keras.layers.Dense(d_model, activation="sigmoid")(lstm_out)
    gated = gate * lstm_out + (1.0 - gate) * fused

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(gated)
    x = SinusoidalPE(d_model, max_len=max(history_length + 4, 256))(x)

    enc_blocks = []
    for i in range(num_layers):
        blk = PreLNEncoderBlock(d_model, num_heads, d_model * 4, dropout, name=f"tft_enc_{i}")
        x = blk(x)
        enc_blocks.append(blk)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    last = x[:, -1, :]
    avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    agg = tf.keras.layers.Concatenate()([last, avg])

    agg = GatedResidualNetwork(d_model, d_model // 2, dropout, name="head_grn")(agg)
    output = tf.keras.layers.Dense(forecast_horizon, name="output")(agg)

    model = tf.keras.Model(inputs=[inp_series, inp_covars], outputs=output, name="TFTLite")
    model._enc_blocks = enc_blocks

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=0.1),
        metrics=["mae"],
    )
    logger.info("TFTLite d=%d h=%d L=%d | %d params", d_model, num_heads, num_layers, count_parameters(model))
    return model


def prepare_tft_covariates(timestamps: np.ndarray) -> np.ndarray:
    """Создаёт матрицу ковариат (N, 4) из временны́х меток."""
    import pandas as pd
    ts = pd.DatetimeIndex(timestamps)
    hour = ts.hour.values
    weekday = ts.weekday.values
    return np.stack([
        np.sin(2 * np.pi * hour / 24).astype(np.float32),
        np.cos(2 * np.pi * hour / 24).astype(np.float32),
        (weekday / 6.0).astype(np.float32),
        (weekday >= 5).astype(np.float32),
    ], axis=-1)


def make_tft_windows(
    scaled_series: np.ndarray,
    covariates: np.ndarray,
    history: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Нарезает ряд + ковариаты на окна для TFT."""
    Xs, Xc, Y = [], [], []
    for i in range(len(scaled_series) - history - horizon + 1):
        Xs.append(scaled_series[i: i + history])
        Xc.append(covariates[i: i + history])
        Y.append(scaled_series[i + history: i + history + horizon])
    return (
        np.array(Xs, dtype=np.float32)[:, :, np.newaxis],
        np.array(Xc, dtype=np.float32),
        np.array(Y, dtype=np.float32),
    )


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА 3: PATCHTST  — FIX v5: RevIN только для channel 0
# ══════════════════════════════════════════════════════════════════════════════

def build_patchtst(
    history_length: int = 48,
    forecast_horizon: int = 24,
    patch_len: int = 8,
    stride: int = 4,
    n_features: int = 11,
    d_model: int = 128,
    num_heads: int = 8,
    num_layers: int = 4,
    dff: int = 256,
    dropout: float = 0.10,
    learning_rate: float = 3e-4,
    stochastic_depth_rate: float = 0.10,
    use_revin: bool = True,
    huber_delta: float = 0.10,
) -> tf.keras.Model:
    """
    PatchTST v5 [Nie et al., ICLR 2023] + RevIN (channel 0 only) + StochasticDepth.

    ИСПРАВЛЕНИЕ v5:
    RevIN применяется ТОЛЬКО к каналу 0 (consumption).
    Ранее нормализовались все 26 каналов, включая бинарные (is_weekend, is_holiday).
    Бинарный канал с константным окном (весь weekend) имеет instance-std=0,
    деление → взрыв активаций → нестабильность → ранняя остановка на epoch=23.

    БЫЛО (v4):
      revin_norm = RevINNorm(...)
      x, revin_mean, revin_std = revin_norm(inp)  # inp = (B, T, 26) — ВСЕ каналы!

    СТАЛО (v5):
      cons_channel = inp[:, :, :1]      # (B, T, 1) — только потребление
      other_channels = inp[:, :, 1:]    # (B, T, 25) — ковариаты БЕЗ нормализации
      cons_norm, revin_mean, revin_std = revin_norm(cons_channel)
      x = Concatenate([cons_norm, other_channels])
    """
    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} не делится на num_heads={num_heads}")
    n_patches = (history_length - patch_len) // stride + 1
    logger.info(
        "PatchTST v5: history=%d patch=%d stride=%d → %d patches | RevIN=%s (channel 0 only)",
        history_length, patch_len, stride, n_patches, use_revin,
    )

    inp = tf.keras.Input(shape=(history_length, n_features), name="series_input")

    # ── RevIN: ТОЛЬКО канал 0 (consumption) ──────────────────────────────────
    # Нормализуем только непрерывный канал потребления.
    # Бинарные каналы (is_weekend, is_holiday, is_peak, is_night) не нормализуем:
    #   если весь window = 1 (например, неделя выходных), instance-std=0 →
    #   деление на eps=1e-5 → значения ×100 000 → взрыв градиентов.
    if use_revin:
        cons_channel   = inp[:, :, :1]          # (B, T, 1)
        other_channels = inp[:, :, 1:]          # (B, T, n_features-1)
        revin_norm = RevINNorm(eps=1e-5, affine=True, name="revin_norm")
        cons_norm, revin_mean, revin_std = revin_norm(cons_channel)
        # Конкатенируем нормализованный consumption с исходными ковариатами
        x = tf.keras.layers.Concatenate(axis=-1, name="revin_concat")(
            [cons_norm, other_channels]
        )
    else:
        x = inp
        revin_mean = revin_std = None

    # Патч-эмбеддинг через Conv1D → (B, n_patches, d_model)
    patches = tf.keras.layers.Conv1D(
        d_model, kernel_size=patch_len, strides=stride,
        padding="valid", name="patch_embed",
    )(x)

    patches = SinusoidalPE(d_model, max_len=max(n_patches + 4, 64), name="patch_pe")(patches)
    patches = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="embed_ln")(patches)
    patches = tf.keras.layers.Dropout(dropout, name="embed_drop")(patches)

    enc_x = patches
    enc_blocks = []
    for i in range(num_layers):
        sdrop_rate = (i / max(num_layers - 1, 1)) * stochastic_depth_rate
        blk = PreLNEncoderBlock(
            d_model, num_heads, dff, dropout,
            stochastic_depth_rate=sdrop_rate,
            name=f"patch_enc_{i}",
        )
        enc_x = blk(enc_x)
        enc_blocks.append(blk)

    enc_x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(enc_x)

    last = enc_x[:, -1, :]
    learned_ctx = LearnedQueryPooling(d_model, name="lqp")(enc_x)
    agg = tf.keras.layers.Concatenate(name="agg")([last, learned_ctx])

    h = tf.keras.layers.Dense(dff // 2, activation="gelu", name="head_d1")(agg)
    skip = tf.keras.layers.Dense(dff // 2, use_bias=False, name="head_skip")(agg)
    h = tf.keras.layers.Add(name="head_res")([h, skip])
    h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="head_ln")(h)
    h = tf.keras.layers.Dropout(dropout, name="head_drop")(h)
    pred = tf.keras.layers.Dense(forecast_horizon, name="pred_norm")(h)

    # RevINDenorm: реконструкция в исходный масштаб через channel 0 статистику.
    # revin_mean/revin_std имеют shape (B, 1, 1) — только для consumption.
    # RevINDenorm берёт mean[:, 0, 0] и std[:, 0, 0] — скаляры на батч-элемент.
    if use_revin:
        output = RevINDenorm(eps=1e-5, affine=True, name="revin_denorm")(
            [pred, revin_mean, revin_std]
        )
    else:
        output = pred

    model = tf.keras.Model(inputs=inp, outputs=output, name="PatchTST_v5")
    model._enc_blocks = enc_blocks

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae", "mape"],
    )
    logger.info(
        "PatchTST v5 patch=%d stride=%d n_p=%d d=%d h=%d L=%d sdrop=%.2f "
        "RevIN=%s (ch0 only, FIX v5) | %d params",
        patch_len, stride, n_patches, d_model, num_heads, num_layers,
        stochastic_depth_rate, use_revin, count_parameters(model),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def transformer_grid_search(
    arch: str,
    data: Dict[str, Any],
    history_length: int = 48,
    forecast_horizon: int = 24,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    target_params: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[tf.keras.Model]]:
    """Grid Search по ключевым гиперпараметрам Transformer."""
    if param_grid is None:
        param_grid = {
            "d_model": [64, 128],
            "num_heads": [4, 8],
            "dropout": [0.05, 0.10],
        }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    best_val, best_cfg, best_model = float("inf"), {}, None

    logger.info("Grid Search [%s]: %d комбинаций", arch, len(combos))

    for combo in combos:
        cfg = dict(zip(keys, combo))
        dm = cfg.get("d_model", 64)
        nh = cfg.get("num_heads", 4)
        if dm % nh != 0:
            continue

        try:
            builders = {
                "vanilla": lambda c: build_vanilla_transformer(
                    history_length=history_length, forecast_horizon=forecast_horizon, **c),
                "patchtst": lambda c: build_patchtst(
                    history_length=history_length, forecast_horizon=forecast_horizon, **c),
                "tft": lambda c: build_tft_lite(
                    history_length=history_length, forecast_horizon=forecast_horizon, **c),
            }
            model = builders[arch](cfg)
            n_p = count_parameters(model)

            if target_params and abs(n_p - target_params) / target_params > 0.5:
                logger.debug("Пропуск %s: %d params vs target %d", cfg, n_p, target_params)
                continue

            cb = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                min_delta=1e-5,
                restore_best_weights=True
            )]

            X_tr, Y_tr = data["X_train"], data["Y_train"]
            X_v, Y_v = data["X_val"], data["Y_val"]

            if arch == "tft":
                cov_tr = data.get("X_covars_train", np.zeros((len(X_tr), history_length, 4)))
                cov_v = data.get("X_covars_val", np.zeros((len(X_v), history_length, 4)))
                X_tr, X_v = [X_tr, cov_tr], [X_v, cov_v]

            hist = model.fit(
                X_tr, Y_tr, validation_data=(X_v, Y_v),
                epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0,
            )
            vl = min(hist.history["val_loss"])
            logger.info("  cfg=%s | val_loss=%.5f | params=%d", cfg, vl, n_p)

            if vl < best_val:
                best_val, best_cfg, best_model = vl, {**cfg, "n_params": n_p}, model

        except Exception as exc:
            logger.warning("Ошибка cfg=%s: %s", cfg, exc)

    logger.info("✅ Лучший [%s]: %s | val_loss=%.5f", arch, best_cfg, best_val)
    return best_cfg, best_model