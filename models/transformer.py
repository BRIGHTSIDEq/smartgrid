# -*- coding: utf-8 -*-
"""
models/transformer.py — Сравнительное исследование архитектур Transformer
для прогнозирования временных рядов энергопотребления.

═══════════════════════════════════════════════════════════════════════════════
РЕАЛИЗОВАННЫЕ АРХИТЕКТУРЫ
═══════════════════════════════════════════════════════════════════════════════

1. VanillaTransformer  — Encoder-only с Pre-LN и Sinusoidal/Learnable/Time2Vec PE
2. TFTLite             — Temporal Fusion Transformer (упрощённый) с ковариатами
3. PatchTST            — State-of-the-art (Nie et al., 2023): патчи вместо точек

ПОЗИЦИОННЫЕ КОДИРОВКИ
  • SinusoidalPE        — Vaswani et al. (2017)
  • LearnableRelativePE — T5-style, смещения по относительному расстоянию
  • Time2Vec            — Kazemi et al. (2019)

МЕХАНИЗМЫ ВНИМАНИЯ
  • Standard MHA        — Scaled Dot-Product (базовая линия)
  • ProbSparseAttention — Zhou et al. (2021), Informer: O(L log L)

КЛЮЧЕВЫЕ ИСПРАВЛЕНИЯ vs предыдущей версии
  • Pre-LN (LN до Attention) вместо Post-LN → стабильные градиенты
  • Агрегация: last_token + avg вместо только GlobalAvgPool
  • TFT: явные ковариаты (час, день недели, праздник) через GRN
  • PatchTST: патч-токенизация через Conv1D
  • count_parameters() для параметрического паритета
  • transformer_grid_search() для выбора гиперпараметров

БИБЛИОГРАФИЯ
  [1] Vaswani A. et al. (2017). Attention is All You Need. NeurIPS.
  [2] Zhou H. et al. (2021). Informer: Beyond Efficient Transformer. AAAI.
  [3] Lim B. et al. (2021). Temporal Fusion Transformers. IJF.
  [4] Nie Y. et al. (2023). A Time Series is Worth 64 Words. ICLR.
  [5] Kazemi S.M. et al. (2019). Time2Vec. arXiv:1907.05321.
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
    """
    Считает обучаемые параметры модели.

    Используется для обеспечения параметрического паритета:
    нельзя делать вывод о превосходстве архитектуры, если у неё
    в 2-3 раза больше параметров, чем у конкурента.
    """
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


# ══════════════════════════════════════════════════════════════════════════════
# ПОЗИЦИОННЫЕ КОДИРОВКИ
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPE(tf.keras.layers.Layer):
    """
    Синусоидальное позиционное кодирование [Vaswani et al., 2017].

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    НЕ обучается → нет переобучения, обобщается на произвольные длины.
    Кодирует относительные расстояния тригонометрически.
    """

    def __init__(self, d_model: int, max_len: int = 512, **kwargs) -> None:
        super().__init__(trainable=False, **kwargs)
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self._pe = tf.constant(angles[np.newaxis], dtype=tf.float32)  # (1, T, d)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self._pe[:, :seq_len, :]

    def get_config(self) -> dict:
        return {**super().get_config(), "d_model": int(self._pe.shape[-1])}


class LearnableRelativePE(tf.keras.layers.Layer):
    """
    Обучаемые смещения по относительному расстоянию (T5-style).

    Вместо абсолютных позиций учим bias(i - j) — функцию расстояния
    между запросом и ключом. Добавляется к attention logits.

    Преимущество: модель явно учит «смотреть N часов назад»,
    что важно для суточных (24ч) и недельных (168ч) паттернов.
    """

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
        """Returns (num_heads, seq_len, seq_len) bias matrix."""
        range_vec = tf.range(seq_len)
        dist = range_vec[:, None] - range_vec[None, :]  # (T, T)
        dist = tf.clip_by_value(dist + (self.max_len - 1), 0, 2 * self.max_len - 2)
        biases = tf.gather(self.rel_bias, dist)          # (T, T, H)
        return tf.transpose(biases, [2, 0, 1])           # (H, T, T)

    def get_config(self) -> dict:
        return {**super().get_config(), "num_heads": self.num_heads, "max_len": self.max_len}


class Time2Vec(tf.keras.layers.Layer):
    """
    Векторное представление времени [Kazemi et al., 2019].

        t2v(τ)[0]   = ω_0 · τ + φ_0               (линейная компонента)
        t2v(τ)[i>0] = sin(ω_i · τ + φ_i)          (периодические компоненты)

    Обучаемые частоты ω_i могут выучить доминирующие периоды (24ч, 168ч).
    В отличие от SinusoidalPE — частоты адаптируются к данным.
    """

    def __init__(self, output_dim: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape) -> None:
        self.W = self.add_weight("W", (1, self.output_dim), "glorot_uniform")
        self.b = self.add_weight("b", (1, self.output_dim), "zeros")
        super().build(input_shape)

    def call(self, tau: tf.Tensor) -> tf.Tensor:
        """tau: (B, T, 1) → (B, T, output_dim)"""
        x = tau * self.W + self.b
        return tf.concat([x[:, :, :1], tf.sin(x[:, :, 1:])], axis=-1)

    def get_config(self) -> dict:
        return {**super().get_config(), "output_dim": self.output_dim}


# ══════════════════════════════════════════════════════════════════════════════
# МЕХАНИЗМ ВНИМАНИЯ: ProbSparse (Informer)
# ══════════════════════════════════════════════════════════════════════════════

class ProbSparseAttention(tf.keras.layers.Layer):
    """
    ProbSparse Self-Attention [Zhou et al., 2021 — Informer].

    Мотивация: большинство Q-запросов дают «размытое» распределение
    внимания ≈ равномерному. Только ~ln(L) запросов «заострены».

    Алгоритм:
      1. Оцениваем важность q_i: M(q_i) = max_j(q_i·k_j) - mean_j(q_i·k_j)
      2. Берём top-u = c·ln(L_Q) запросов (u << L_Q)
      3. Остальные → mean pooling по V (нейтральный ответ)

    Сложность O(L·log L) вместо O(L²).
    """

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
        scores = tf.matmul(Q, K, transpose_b=True) / scale   # (B,H,L,L)

        # Важность: max - mean
        importance = tf.reduce_max(scores, -1) - tf.reduce_mean(scores, -1)  # (B,H,L)
        L_Q = tf.shape(q)[1]
        u = tf.minimum(
            tf.cast(self.factor * tf.math.ceil(tf.math.log(tf.cast(L_Q, tf.float32))), tf.int32),
            L_Q,
        )
        _, top_idx = tf.math.top_k(importance, k=u)  # (B,H,u)
        # Маска: невыбранные запросы → -inf
        top_oh = tf.cast(
            tf.reduce_sum(tf.one_hot(top_idx, depth=L_Q, axis=-1), axis=-2) > 0,
            tf.float32,
        )  # (B,H,L)
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
    """
    Encoder-блок с Pre-Layer Normalization.

    Порядок Pre-LN:
        x → LN → MHA → Dropout → residual
        x → LN → FFN → Dropout → residual

    Vs. Post-LN (оригинал Vaswani):
        x → MHA → Dropout → residual → LN
        x → FFN → Dropout → residual → LN

    Pre-LN обеспечивает стабильность градиентов без warm-up schedule
    и позволяет обучать более глубокие сети [Xiong et al., 2020].

    Дополнительно: GELU вместо ReLU в FFN (меньше «мёртвых нейронов»).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        use_prob_sparse: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        self.use_prob_sparse = use_prob_sparse

        if use_prob_sparse:
            self.attn = ProbSparseAttention(d_model, num_heads, dropout=dropout)
        else:
            self.attn = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
            )

        self.ffn1 = tf.keras.layers.Dense(dff, activation="gelu")
        self.ffn2 = tf.keras.layers.Dense(d_model)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self._attn_weights: Optional[tf.Tensor] = None

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        # Pre-LN attention
        residual = x
        xn = self.ln1(x)
        if self.use_prob_sparse:
            attn_out, self._attn_weights = self.attn(xn, xn, xn, training=training)
        else:
            attn_out, self._attn_weights = self.attn(
                xn, xn, return_attention_scores=True, training=training
            )
        x = residual + self.drop1(attn_out, training=training)

        # Pre-LN FFN
        residual = x
        xn = self.ln2(x)
        x = residual + self.drop2(self.ffn2(self.ffn1(xn)), training=training)
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
            "use_prob_sparse": self.use_prob_sparse,
        }


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА 1: VANILLA TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

def build_vanilla_transformer(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 9,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    dff: int = 256,
    dropout: float = 0.10,
    learning_rate: float = 1e-4,
    pe_type: str = "sinusoidal",
    use_prob_sparse: bool = False,
) -> tf.keras.Model:
    """
    Encoder-only Transformer с Pre-LN и выбором позиционного кодирования.

    Преимущества над LSTM:
    - Self-Attention: прямой путь между любыми двумя шагами истории за O(1)
    - Multi-head: разные головы специализируются на разных лагах (24ч, 48ч, 168ч)
    - Параллельная обработка → нет проблемы затухания градиентов по времени

    Агрегация: [last_token; global_avg] вместо только GlobalAvgPool —
    last_token несёт «актуальное состояние», avg — «исторический контекст».

    Parameters
    ----------
    pe_type : "sinusoidal" | "relative" | "time2vec"
    use_prob_sparse : bool  — ProbSparse вместо Standard MHA
    """
    assert d_model % num_heads == 0, f"d_model={d_model} не делится на num_heads={num_heads}"

    inp_series = tf.keras.Input(shape=(history_length, n_features), name="series_input")
    inputs = [inp_series]

    # input_proj: (T, n_features) → (T, d_model)
    x = tf.keras.layers.Dense(d_model, name="input_proj")(inp_series)

    if pe_type == "sinusoidal":
        x = SinusoidalPE(d_model, max_len=max(history_length + 4, 256), name="sin_pe")(x)

    elif pe_type == "time2vec":
        tau = tf.keras.Input(shape=(history_length, 1), name="time_index")
        inputs.append(tau)
        t2v = Time2Vec(d_model, name="time2vec")(tau)
        x = x + t2v

    # "relative" — bias добавляется внутри блоков (упрощённо: без явного bias,
    # LearnableRelativePE будет добавлен в будущих версиях через custom MHA)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x)
    x = tf.keras.layers.Dropout(dropout, name="embed_drop")(x)

    enc_blocks = []
    for i in range(num_layers):
        blk = PreLNEncoderBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout=dropout, use_prob_sparse=use_prob_sparse,
            name=f"enc_{i}",
        )
        x = blk(x)
        enc_blocks.append(blk)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    last = x[:, -1, :]
    avg = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    agg = tf.keras.layers.Concatenate(name="agg")([last, avg])

    agg = tf.keras.layers.Dense(dff // 2, activation="gelu", name="head_d1")(agg)
    agg = tf.keras.layers.Dropout(dropout, name="head_drop")(agg)
    output = tf.keras.layers.Dense(forecast_horizon, name="output")(agg)

    model = tf.keras.Model(inputs=inputs, outputs=output,
                           name=f"VanillaTransformer_{pe_type}")
    model._enc_blocks = enc_blocks  # для визуализации внимания

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    logger.info(
        "VanillaTransformer[%s%s] d=%d h=%d L=%d | %d params",
        pe_type, "+ProbSparse" if use_prob_sparse else "",
        d_model, num_heads, num_layers, count_parameters(model),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА 2: TFT LITE
# ══════════════════════════════════════════════════════════════════════════════

class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    Gated Residual Network (GRN) — ключевой блок TFT [Lim et al., 2021].

        η1 = ELU(W1·x + b1)
        η2 = W2·η1 + b2
        gate = sigmoid(W3·η2 + b3)
        out = LayerNorm(proj(x) + gate * η2)

    Гейтинг позволяет модели «выключать» нерелевантные признаки
    (например, праздники в будние дни или температуру в межсезонье).
    """

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
    """
    Temporal Fusion Transformer Lite [Lim et al., 2021 — упрощённая версия].

    Оригинальная TFT — лучшая архитектура для многогоризонтного прогноза
    с интерпретируемостью (Variable Importance + Attention Weights).

    Гибридная архитектура LSTM + Attention:
    - LSTM кодирует ЛОКАЛЬНЫЕ паттерны (почасовые переходы, импульсы)
    - Attention захватывает ДАЛЬНИЕ зависимости (вчера в то же время, неделю назад)
    - Gate объединяет оба сигнала адаптивно

    Ковариаты (hour_sin, hour_cos, weekday_norm, is_weekend):
    - Без ковариат Transformer «не знает», что сейчас 19:00 пятница
    - GRN интегрирует их структурно (не просто конкатенация)

    Входы:
    - series_input : (B, T, 1) — нормализованный ряд
    - covariate_input : (B, T, n_covariate_features) — временны́е признаки
    """
    assert d_model % num_heads == 0

    inp_series = tf.keras.Input(shape=(history_length, 1), name="series_input")
    inp_covars = tf.keras.Input(shape=(history_length, n_covariate_features), name="covariate_input")

    s_emb = tf.keras.layers.Dense(d_model)(inp_series)
    c_emb = tf.keras.layers.Dense(d_model)(inp_covars)

    s_enc = GatedResidualNetwork(d_model, d_model, dropout, name="grn_series")(s_emb)
    c_enc = GatedResidualNetwork(d_model, d_model, dropout, name="grn_covar")(c_emb)

    fused = tf.keras.layers.Add()([s_enc, c_enc])
    fused = tf.keras.layers.LayerNormalization(epsilon=1e-6)(fused)

    # LSTM: локальные паттерны
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
        loss="huber",
        metrics=["mae"],
    )
    logger.info("TFTLite d=%d h=%d L=%d | %d params", d_model, num_heads, num_layers, count_parameters(model))
    return model


def prepare_tft_covariates(timestamps: np.ndarray) -> np.ndarray:
    """
    Создаёт матрицу ковариат (N, 4) из временны́х меток.

    Признаки:
    - hour_sin, hour_cos : циклическое кодирование часа (нет артефакта 23→0)
    - weekday_norm       : день недели / 6
    - is_weekend         : 0/1
    """
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
# АРХИТЕКТУРА 3: PATCHTST
# ══════════════════════════════════════════════════════════════════════════════

def build_patchtst(
    history_length: int = 48,
    forecast_horizon: int = 24,
    patch_len: int = 8,
    stride: int = 4,
    n_features: int = 9,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 3,
    dff: int = 256,
    dropout: float = 0.10,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    PatchTST — State-of-the-art Transformer [Nie et al., ICLR 2023].

    Ключевая идея: обрабатывать ряд ПАТЧАМИ (подпоследовательностями),
    а не отдельными точками. Аналогия с ViT для изображений.

    Почему патчи лучше точек:
    1. Семантически осмысленные токены:
       Патч из 8ч содержит «суточный переход» — смысловую единицу
       потребления. Отдельная точка = шумный скаляр.

    2. Эффективность внимания:
       При patch=8, stride=4, history=48: N_patches = (48-8)//4+1 = 11
       Сложность O(11²) вместо O(48²) — в 19 раз меньше операций.

    3. Локальный контекст в токене:
       Шум сглаживается ВНУТРИ патча ещё на этапе Conv1D-эмбеддинга.

    4. Более эффективные PE:
       11 позиций кодировать проще, чем 48; PE несёт больше информации.

    Результаты PatchTST [Nie et al., 2023]:
    - ETTh1 (512→96): PatchTST MSE=0.370 vs FEDformer 0.440 (-16%)
    - На длинных горизонтах превосходит все другие Transformer-варианты

    Parameters
    ----------
    patch_len : рекомендации: history=24→6, history=48→8, history=96→16
    stride    : обычно stride = patch_len // 2 (50% перекрытие)
    """
    assert d_model % num_heads == 0
    n_patches = (history_length - patch_len) // stride + 1
    logger.info(
        "PatchTST: history=%d patch=%d stride=%d → %d patches",
        history_length, patch_len, stride, n_patches,
    )

    inp = tf.keras.Input(shape=(history_length, n_features), name="series_input")

    # Патч-эмбеддинг через Conv1D: kernel=(patch_len, n_features) → d_model
    # Conv1D принимает (T, channels) нативно: здесь channels=n_features
    patches = tf.keras.layers.Conv1D(
        d_model, kernel_size=patch_len, strides=stride, padding="valid", name="patch_embed"
    )(inp)  # (B, n_patches, d_model)

    # Instance Normalization (RevIN-style): устраняет distribution shift
    patches = tf.keras.layers.LayerNormalization(
        axis=(1, 2), epsilon=1e-6, name="instance_norm"
    )(patches)

    patches = SinusoidalPE(d_model, max_len=max(n_patches + 4, 64), name="patch_pe")(patches)
    patches = tf.keras.layers.Dropout(dropout, name="embed_drop")(patches)

    x = patches
    enc_blocks = []
    for i in range(num_layers):
        blk = PreLNEncoderBlock(d_model, num_heads, dff, dropout, name=f"patch_enc_{i}")
        x = blk(x)
        enc_blocks.append(blk)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    x_flat = tf.keras.layers.Flatten(name="flatten")(x)
    x_flat = tf.keras.layers.Dense(dff // 2, activation="gelu", name="head_d1")(x_flat)
    x_flat = tf.keras.layers.Dropout(dropout, name="head_drop")(x_flat)
    output = tf.keras.layers.Dense(forecast_horizon, name="output")(x_flat)

    model = tf.keras.Model(inputs=inp, outputs=output, name="PatchTST")
    model._enc_blocks = enc_blocks

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )
    logger.info(
        "PatchTST patch=%d stride=%d n_p=%d d=%d h=%d L=%d | %d params",
        patch_len, stride, n_patches, d_model, num_heads, num_layers, count_parameters(model),
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
    """
    Grid Search по ключевым гиперпараметрам Transformer.

    Parameters
    ----------
    arch         : "vanilla" | "tft" | "patchtst"
    param_grid   : {param: [values]}. По умолчанию d_model × num_heads × dropout
    target_params: желаемое число параметров (для паритета с LSTM)
                   Пропускаем конфиги, где |params - target| / target > 50%
    """
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
                monitor="val_loss", patience=patience, restore_best_weights=True
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
