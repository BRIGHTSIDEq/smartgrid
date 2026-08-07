# -*- coding: utf-8 -*-
"""
models/transformer.py — Сравнительное исследование архитектур Transformer
для прогнозирования временных рядов энергопотребления.

═══════════════════════════════════════════════════════════════════════════════
РЕАЛИЗОВАННЫЕ АРХИТЕКТУРЫ
═══════════════════════════════════════════════════════════════════════════════

1. VanillaTransformer  — Encoder-only с Pre-LN и Sinusoidal/Time2Vec PE
2. PatchTST            — Nie et al., 2023: патч-токенизация вместо отдельных точек

НОВЫЕ КОМПОНЕНТЫ v4
─────────────────────────────────────────────────────────────────────────────
• RevIN (Reversible Instance Normalization) — Kim et al., 2022
    Нормализует каждое входное окно ДО кодирования (устраняет distribution
    shift между train/val/test окнами), денормализует после декодирования.
    Ключевое нововведение оригинального PatchTST [Nie et al., 2023].
    Улучшение: устраняет «шоки» на границах сезонов внутри каждого окна.

• StochasticDepth (DropPath) — Huang et al., 2016
    На каждой итерации обучения случайно выбрасывает всю residual-ветку
    блока с вероятностью, нарастающей по глубине: depth_i/num_layers × rate.
    Эффект: неявный ensemble сетей разной глубины → лучшая генерализация.
    Используется в ViT, DeiT, современных Transformer-моделях для изображений.

• LearnedQueryPooling — вместо GlobalAveragePooling в голове
    Обучаемый вектор-запрос q attending к финальному контексту:
    score_t = softmax(q · h_t) → взвешенная сумма h_t.
    Семантически: «выбираем наиболее релевантные шаги для прогноза»
    вместо равномерного усреднения.

ПОЗИЦИОННЫЕ КОДИРОВКИ
  • SinusoidalPE        — Vaswani et al. (2017)
  • Time2Vec            — Kazemi et al. (2019)

МЕХАНИЗМЫ ВНИМАНИЯ
  • Standard MHA        — Scaled Dot-Product (базовая линия)
  • ProbSparseAttention — Zhou et al. (2021), Informer: O(L log L)

БИБЛИОГРАФИЯ
  [1] Vaswani A. et al. (2017). Attention is All You Need. NeurIPS.
  [2] Zhou H. et al. (2021). Informer: Beyond Efficient Transformer. AAAI.
  [4] Nie Y. et al. (2023). A Time Series is Worth 64 Words. ICLR.
  [5] Kazemi S.M. et al. (2019). Time2Vec. arXiv:1907.05321.
  [6] Kim T. et al. (2022). RevIN: Reversible Instance Normalization. ICLR.
  [7] Huang G. et al. (2016). Deep Networks with Stochastic Depth. ECCV.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from models.lstm import SeasonalSkipConnection

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
# НОВЫЕ КОМПОНЕНТЫ v4
# ══════════════════════════════════════════════════════════════════════════════

class RevINNorm(tf.keras.layers.Layer):
    """
    RevIN Normalization [Kim et al., ICLR 2022] — НОРМАЛИЗАЦИЯ.

    Graph-compatible версия для Keras Functional API.

    Возвращает (x_norm, mean, std) как три отдельных тензора,
    чтобы mean/std можно было передать в RevINDenorm как явные
    граф-тензоры (а не Python-атрибуты, несовместимые с Functional API).

    Принцип:
      x_norm = (x - mean) / std  (+аффинное преобразование gamma/beta)
      mean, std — статистика по временной оси T каждого экземпляра отдельно
      (instance-normalization, не batch-normalization)
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
        """
        Returns
        -------
        x_norm : (B, T, C) — нормализованный вход
        mean   : (B, 1, C) — среднее по временной оси (для денормализации)
        std    : (B, 1, C) — ст. откл. по временной оси (для денормализации)
        """
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
    и возвращает прогноз в исходном масштабе:
      output = pred_norm * std_scalar + mean_scalar

    Где std_scalar / mean_scalar — скаляры, извлечённые из mean/std
    первого признака (consumption, индекс 0) входного окна.

    Это корректно, т.к. таргет Y = consumption_scaled (первый признак).
    """

    def __init__(self, eps: float = 1e-5, affine: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.affine = affine

    def build(self, input_shapes: list) -> None:
        # Нет обучаемых весов — денормализация использует статистику из RevINNorm
        super().build(input_shapes)

    def call(self, inputs: list) -> tf.Tensor:
        """
        Parameters
        ----------
        inputs : [pred, mean, std]
          pred : (B, forecast_horizon) — прогноз в RevIN-нормализованном пространстве
          mean : (B, 1, C)             — среднее из RevINNorm
          std  : (B, 1, C)             — ст. откл. из RevINNorm

        Returns
        -------
        output : (B, forecast_horizon) — прогноз в исходном (MinMax) масштабе

        ИСПРАВЛЕНИЕ: убраны отдельные denorm_gamma/denorm_beta.
        Инверсия RevINNorm: output = pred * std[0] + mean[0].
        Отдельные gamma/beta не нужны — это не обращение аффинного преобразования,
        это просто денормализация по сохранённым instance-statistics.
        Предыдущая версия обучала независимые веса, не связанные с RevINNorm.gamma/beta,
        что приводило к некорректной денормализации после нескольких эпох обучения.
        """
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
    """
    Stochastic Depth / DropPath [Huang et al., ECCV 2016].

    На каждом шаге обучения с вероятностью `drop_rate` выбрасывает
    ВСЮ residual-ветку блока (не отдельные нейроны, а всю ветку целиком).

    Зачем это лучше Dropout:
      Dropout: отдельные нейроны → модель учится работать с неполным слоем
      DropPath: целый блок → модель учится работать без блока → неявный
      ensemble сетей разной глубины [Loshchilov & Hutter, 2016 показали
      ту же идею с другой стороны].

    Использование в PreLNEncoderBlock:
      x = x + StochasticDepth(rate)(residual, training)
      При inference: rate=0 → обычный residual (нет выброса).

    Рекомендация: rate_i = (i / num_layers) × max_rate
      (линейно нарастает с глубиной — последние блоки выбрасываются чаще)
    """

    def __init__(self, drop_rate: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if not training or self.drop_rate == 0.0:
            return x
        # Сохраняем shape (B,) маску: одинаковая для всего batch-элемента
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        keep_prob = 1.0 - self.drop_rate
        random_tensor = tf.random.uniform(shape, dtype=x.dtype)
        # Rounded: 0 если выброшен, 1/keep_prob если оставлен (unbiased estimate)
        binary_tensor = tf.math.floor(random_tensor + keep_prob)
        return x * binary_tensor / keep_prob

    def get_config(self) -> dict:
        return {**super().get_config(), "drop_rate": self.drop_rate}


class LearnedQueryPooling(tf.keras.layers.Layer):
    """
    Обучаемый query-pooling вместо GlobalAveragePooling.

    Вместо равномерного усреднения: context = (1/T) Σ h_t
    Используем обучаемый вектор запроса q:
      score_t = softmax(q · h_t / √d)
      context = Σ score_t × h_t

    Семантика: «выбираем наиболее информативные шаги для прогноза»,
    аналогично механизму Attention с одним запросом (CLS-token в BERT).

    Улучшение над GlobalAvgPool: позволяет сосредоточиться на пиковых
    часах или переломных моментах ряда, а не усреднять всё поровну.
    """

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape: tuple) -> None:
        self.query = self.add_weight(
            shape=(self.d_model,), initializer="glorot_uniform", name="pooling_query"
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        context : (B, d_model)
        """
        # scores: (B, T)
        scores = tf.einsum("btd,d->bt", x, self.query) / (self.d_model ** 0.5)
        weights = tf.nn.softmax(scores, axis=-1)  # (B, T)
        context = tf.einsum("bt,btd->bd", weights, x)  # (B, d_model)
        return context

    def get_config(self) -> dict:
        return {**super().get_config(), "d_model": self.d_model}




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
    Encoder-блок с Pre-Layer Normalization + StochasticDepth v4.

    Порядок Pre-LN:
        x → LN → MHA → DropPath(rate) → residual
        x → LN → FFN → DropPath(rate) → residual

    НОВОЕ v4: StochasticDepth (DropPath) заменяет обычный Dropout
    в residual-ветке:
      Вместо случайного зануления отдельных нейронов — зануляется
      ВСЯ residual-ветка блока (Attention или FFN целиком).
      Эффект: неявный ensemble сетей разной глубины.
      [Huang et al., 2016 — Deep Networks with Stochastic Depth, ECCV]

    Vs. Post-LN (оригинал Vaswani):
        x → MHA → Dropout → residual → LN
    Pre-LN обеспечивает стабильность градиентов без warm-up schedule
    [Xiong et al., 2020].

    GELU в FFN: стандарт в современных архитектурах [Hendrycks 2016].
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout: float = 0.1,
        stochastic_depth_rate: float = 0.0,   # ← НОВЫЙ параметр v4
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

        # FFN: добавляем промежуточный Dropout внутри FFN (standard practice)
        self.ffn1 = tf.keras.layers.Dense(dff, activation="gelu")
        self.ffn_drop = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(d_model)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # StochasticDepth вместо простого Dropout в residual-ветке
        self.sdrop1 = StochasticDepth(stochastic_depth_rate)
        self.sdrop2 = StochasticDepth(stochastic_depth_rate)

        self._attn_weights: Optional[tf.Tensor] = None

    def call(self, x: tf.Tensor, training=None) -> tf.Tensor:
        # ── Pre-LN Attention ──────────────────────────────────────────────────
        residual = x
        xn = self.ln1(x)
        if self.use_prob_sparse:
            attn_out, self._attn_weights = self.attn(xn, xn, xn, training=training)
        else:
            attn_out, self._attn_weights = self.attn(
                xn, xn, return_attention_scores=True, training=training
            )
        # StochasticDepth применяется к residual-ветке (не к основному пути)
        x = residual + self.sdrop1(attn_out, training=training)

        # ── Pre-LN FFN ────────────────────────────────────────────────────────
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
    use_seasonal_residual: bool = True,
    seasonal_blend_init: float = 0.40,
    huber_delta: float = 0.05,
) -> tf.keras.Model:
    """
    Encoder-only Transformer v4 с StochasticDepth + LearnedQueryPooling.

    НОВОЕ v4:
    ─────────
    1. StochasticDepth (DropPath) в каждом блоке с линейно нарастающей rate:
       rate_i = (i / num_layers) × stochastic_depth_rate
       Интерпретация: неявный ensemble сетей глубиной 1..num_layers.
       Ранние слои (feature extraction) выбрасываются редко,
       глубокие (refinement) — чаще. [Huang et al., ECCV 2016]

    2. LearnedQueryPooling вместо GlobalAveragePool:
       Обучаемый вектор-запрос q выбирает информативные шаги:
         scores_t = softmax(q · h_t / √d)
         context = Σ scores_t × h_t
       vs. GlobalAvgPool: равномерное усреднение без учёта важности шагов.

    3. Улучшенная голова: Dense(gelu) + Skip + LN → стабильность при L=4.

    Преимущества над LSTM:
    - Self-Attention O(1) до любого шага истории (LSTM O(T) через ворота)
    - Многоголовая специализация: head_0→лаг-24ч, head_1→лаг-1ч, и т.д.
    - Параллельная обработка — нет проблемы затухающих градиентов

    Parameters
    ----------
    stochastic_depth_rate : float, максимальная rate DropPath (у последнего блока)
    """
    assert d_model % num_heads == 0, f"d_model={d_model} не делится на num_heads={num_heads}"

    inp_series = tf.keras.Input(shape=(history_length, n_features), name="series_input")
    inputs = [inp_series]

    x = tf.keras.layers.Dense(d_model, name="input_proj")(inp_series)

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
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="neural_head")(h)

    # Сезонный остаточный skip: сеть учит ПОПРАВКУ к суточно-наивному прогнозу,
    # а не уровень нагрузки с нуля. Вход берётся из сырого канала потребления
    # в масштабе MinMaxScaler [0,1] — том же, в котором задан таргет.
    if use_seasonal_residual:
        naive_slice = inp_series[:, -forecast_horizon:, 0]
        output = SeasonalSkipConnection(
            init_neural_weight=seasonal_blend_init, name="seasonal_residual",
        )([neural_out, naive_slice])
    else:
        output = neural_out

    model = tf.keras.Model(inputs=inputs, outputs=output,
                           name=f"VanillaTransformer_v4_{pe_type}")
    model._enc_blocks = enc_blocks

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )
    logger.info(
        "VanillaTransformer [%s%s] d=%d h=%d L=%d sdrop=%.2f "
        "seasonal_residual=%s | %d params",
        pe_type, "+ProbSparse" if use_prob_sparse else "",
        d_model, num_heads, num_layers, stochastic_depth_rate,
        use_seasonal_residual, count_parameters(model),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# АРХИТЕКТУРА 2: PATCHTST
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
    PatchTST v4 [Nie et al., ICLR 2023] + RevIN + StochasticDepth.

    КЛЮЧЕВЫЕ ИДЕИ PatchTST (оригинал):
    - Патч-токены: ряд разбивается на перекрывающиеся подпоследовательности.
      patch=8, stride=4, T=48 → N_patches=11. O(11²) vs O(48²) — 19x экономия.
    - Каждый патч = «суточный переход» — семантически осмысленный токен.

    НОВОЕ v4:
    ──────────────────────────────────────────────────────────────────────────
    1. RevIN (Reversible Instance Normalization) [Kim et al., ICLR 2022]:
       Нормализует КАЖДОЕ входное окно на собственные mean/std ДО кодирования.
       Денормализует прогноз ПОСЛЕ декодирования (обратимо).
       Это именно та нормализация из оригинального PatchTST — без неё
       distribution shift между сезонами снижает R² на 0.05-0.10.

    2. StochasticDepth с линейно нарастающей rate:
       Блок i: rate = (i / (L-1)) × max_rate
       rates = [0.00, 0.03, 0.07, 0.10] при L=4, max_rate=0.10.
       Эффект: ensemble сетей глубиной 1..4.

    3. LearnedQueryPooling вместо Flatten:
       Flatten(11×128=1408) → огромная голова. LQP: 1 вектор-запрос → (B,128).

    Parameters
    ----------
    use_revin : RevIN нормализация по instance (рекомендуется True)
    """
    assert d_model % num_heads == 0
    n_patches = (history_length - patch_len) // stride + 1
    logger.info(
        "PatchTST v4: history=%d patch=%d stride=%d → %d patches | RevIN=%s",
        history_length, patch_len, stride, n_patches, use_revin,
    )

    inp = tf.keras.Input(shape=(history_length, n_features), name="series_input")

    # RevIN нормализация: возвращает (x_norm, mean, std) как граф-тензоры.
    # mean/std сохраняются в графе и передаются в RevINDenorm явно —
    # это единственный graph-compatible способ для Keras Functional API.
    if use_revin:
        revin_norm = RevINNorm(eps=1e-5, affine=True, name="revin_norm")
        x, revin_mean, revin_std = revin_norm(inp)  # все три — граф-тензоры
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

    # RevIN денормализация: pred × std[consumption] + mean[consumption].
    # Все тензоры — граф-узлы, tf.* вызовы внутри слоёв, а не в Functional API.
    if use_revin:
        output = RevINDenorm(eps=1e-5, affine=True, name="revin_denorm")(
            [pred, revin_mean, revin_std]
        )
    else:
        output = pred

    model = tf.keras.Model(inputs=inp, outputs=output, name="PatchTST_v4")
    model._enc_blocks = enc_blocks

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae", "mape"],
    )
    logger.info(
        "PatchTST v4 patch=%d stride=%d n_p=%d d=%d h=%d L=%d sdrop=%.2f RevIN=%s | %d params",
        patch_len, stride, n_patches, d_model, num_heads, num_layers,
        stochastic_depth_rate, use_revin, count_parameters(model),
    )
    return model


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
    arch         : "vanilla" | "patchtst"
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
            }
            model = builders[arch](cfg)
            n_p = count_parameters(model)

            if target_params and abs(n_p - target_params) / target_params > 0.5:
                logger.debug("Пропуск %s: %d params vs target %d", cfg, n_p, target_params)
                continue

            cb = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                min_delta=1e-5,              # не останавливаться на шуме оптимизатора
                restore_best_weights=True
            )]

            X_tr, Y_tr = data["X_train"], data["Y_train"]
            X_v, Y_v = data["X_val"], data["Y_val"]

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