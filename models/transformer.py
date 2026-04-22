# -*- coding: utf-8 -*-
"""models/transformer.py — v16.

ИЗМЕНЕНИЯ v16 vs v15 (audit 2026-04-22):

[1] build_itransformer(): выставляет model._enc_blocks.
  БЫЛО: _enc_blocks не выставлялся → attention_visualization.py падал с
        WARNING "Encoder-блоки не найдены в модели iTransformer_v1".
        В main.py [7/10] вся визуализация внимания пропускалась.
  СТАЛО: model._enc_blocks = [layer for layer in model.layers
                               if isinstance(layer, iTransformerBlock)]
         Позволяет extract_attention_weights() корректно итерировать блоки.

[2] build_patchtst(): принимает отдельные patchtst_lr и patchtst_dropout.
  БЫЛО: learning_rate и dropout жёстко брались из вызывающего кода
        (TRANSFORMER_LEARNING_RATE=1e-4, TRANSFORMER_DROPOUT=0.08).
        Результат: best_epoch=17 из 61 — PatchTST не обучился.
  СТАЛО: отдельные параметры patchtst_learning_rate и patchtst_dropout
         с дефолтами 3e-4 / 0.05, соответствующими Config.PATCHTST_*.
         main.py передаёт Config.PATCHTST_LEARNING_RATE явно.

[3] Сохранена обратная совместимость: PreLNEncoderBlock, SinusoidalPE,
    Time2Vec, GatedResidualNetwork — stubs для load_keras().
"""

import logging
import math
from typing import Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger("smart_grid.models.transformer")

__all__ = [
    "iTransformerBlock",
    "PreLNEncoderBlock",
    "SinusoidalPE",
    "Time2Vec",
    "GatedResidualNetwork",
    "WarmupCosineDecay",
    "build_itransformer",
    "build_patchtst",
    "build_tft_lite",
    "count_parameters",
    "prepare_tft_covariates",
]


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Линейный warmup + CosineDecay."""

    def __init__(self, peak_lr: float, total_steps: int, warmup_steps: int = 0,
                 alpha: float = 0.02, name: str = "WarmupCosineDecay"):
        super().__init__()
        self.peak_lr      = float(peak_lr)
        self.total_steps  = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha        = float(alpha)
        self._name        = name

    def __call__(self, step):
        step_f  = tf.cast(step, tf.float32)
        ws      = float(max(self.warmup_steps, 1))
        ts      = float(self.total_steps)
        peak    = self.peak_lr
        alpha   = self.alpha
        warmup_lr = peak * (step_f / ws)
        cosine_steps = tf.maximum(step_f - ws, 0.0)
        cosine_total = tf.maximum(ts - ws, 1.0)
        progress     = tf.minimum(cosine_steps / cosine_total, 1.0)
        cosine_lr    = peak * (alpha + (1.0 - alpha) * 0.5 *
                               (1.0 + tf.cos(np.pi * progress)))
        return tf.where(step_f < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr":      self.peak_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha":        self.alpha,
            "name":         self._name,
        }


# ══════════════════════════════════════════════════════════════════════════════
# iTransformer BLOCK
# ══════════════════════════════════════════════════════════════════════════════

class iTransformerBlock(tf.keras.layers.Layer):
    """Pre-LN Transformer block для iTransformer (attention over variates)."""

    def __init__(self, d_model: int, num_heads: int, dff: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.dff       = dff
        self.dropout_rate = dropout

        key_dim    = max(d_model // num_heads, 8)
        self.ln1   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha   = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.ln2   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1   = tf.keras.layers.Dense(dff, activation="gelu",
                                           kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.ff2   = tf.keras.layers.Dense(d_model,
                                           kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.drop3 = tf.keras.layers.Dropout(dropout)
        # Сохраняем веса внимания для визуализации
        self._attn_weights = None

    def call(self, x, training=None):
        h = self.ln1(x)
        h, self._attn_weights = self.mha(
            h, h, h, training=training, return_attention_scores=True)
        h = self.drop1(h, training=training)
        x = x + h
        h = self.ln2(x)
        h = self.ff1(h)
        h = self.drop2(h, training=training)
        h = self.ff2(h)
        h = self.drop3(h, training=training)
        return x + h

    def get_attention_weights(self) -> Optional[tf.Tensor]:
        """Возвращает последние веса внимания (для визуализации)."""
        return self._attn_weights

    def get_config(self):
        return {
            **super().get_config(),
            "d_model":   self.d_model,
            "num_heads": self.num_heads,
            "dff":       self.dff,
            "dropout":   self.dropout_rate,
        }


# ══════════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPAT STUBS (для load_keras)
# ══════════════════════════════════════════════════════════════════════════════

class PreLNEncoderBlock(tf.keras.layers.Layer):
    """Stub: сохранён для load_keras совместимости со старыми .keras файлами."""
    def __init__(self, d_model=128, num_heads=4, dff=256, dropout=0.1,
                 stochastic_depth_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads
        self.dff = dff; self.dropout_rate = dropout
        self.sd_rate = stochastic_depth_rate
        key_dim = max(d_model // num_heads, 8)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ff1 = tf.keras.layers.Dense(dff, activation="gelu")
        self.drop2 = tf.keras.layers.Dropout(dropout)
        self.ff2 = tf.keras.layers.Dense(d_model)
        self.drop3 = tf.keras.layers.Dropout(dropout)
        self._attn_weights = None

    def call(self, x, training=None):
        h = self.ln1(x)
        h, self._attn_weights = self.mha(h, h, h, training=training, return_attention_scores=True)
        h = self.drop1(h, training=training); x = x + h
        h = self.ln2(x); h = self.ff1(h); h = self.drop2(h, training=training)
        h = self.ff2(h); h = self.drop3(h, training=training)
        return x + h

    def get_attention_weights(self) -> Optional[tf.Tensor]:
        return self._attn_weights

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model,
                "num_heads": self.num_heads, "dff": self.dff,
                "dropout": self.dropout_rate, "stochastic_depth_rate": self.sd_rate}


class SinusoidalPE(tf.keras.layers.Layer):
    """Stub: сохранён для load_keras совместимости."""
    def __init__(self, max_len=512, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len; self.d_model = d_model
    def call(self, x): return x
    def get_config(self):
        return {**super().get_config(), "max_len": self.max_len, "d_model": self.d_model}


class Time2Vec(tf.keras.layers.Layer):
    """Stub: сохранён для load_keras совместимости."""
    def __init__(self, output_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
    def call(self, x): return x
    def get_config(self):
        return {**super().get_config(), "output_dim": self.output_dim}


class GatedResidualNetwork(tf.keras.layers.Layer):
    """GRN из TFT. Используется в build_tft_lite."""
    def __init__(self, units: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units   = units
        self.dropout_rate = dropout
        self.dense1  = tf.keras.layers.Dense(units, activation="elu")
        self.dense2  = tf.keras.layers.Dense(units)
        self.dense_g = tf.keras.layers.Dense(units, activation="sigmoid")
        self.ln      = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop    = tf.keras.layers.Dropout(dropout)
        self.proj    = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.proj = tf.keras.layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=None):
        residual = self.proj(x) if self.proj else x
        h = self.dense1(x)
        h = self.drop(h, training=training)
        h = self.dense2(h)
        g = self.dense_g(h)
        return self.ln(residual + g * h)

    def get_config(self):
        return {**super().get_config(), "units": self.units, "dropout": self.dropout_rate}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model) -> int:
    if isinstance(model, tf.keras.Model):
        return int(model.count_params())
    return 0


def prepare_tft_covariates(X: np.ndarray, covar_indices) -> list:
    """Вспомогательная функция для подготовки входов TFT-Lite."""
    series = X[:, :, :1].astype(np.float32)
    covars = X[:, :, covar_indices].astype(np.float32)
    return [series, covars]


def _build_lr_schedule(
    learning_rate: float,
    use_cosine_decay: bool,
    total_steps: int,
    warmup_ratio: float = 0.05,
    alpha: float = 0.02,
    model_name: str = "",
):
    """Строит WarmupCosineDecay или фиксированный LR."""
    if not (use_cosine_decay and total_steps > 100):
        return learning_rate

    warmup_steps = int(warmup_ratio * total_steps) if warmup_ratio > 0 else 0
    schedule = WarmupCosineDecay(
        peak_lr=learning_rate, total_steps=total_steps,
        warmup_steps=warmup_steps, alpha=alpha,
    )
    if warmup_steps > 0:
        logger.info(
            "%s: WarmupCosineDecay lr=0→%.0e warmup=%d steps, cosine→%.0e total=%d",
            model_name, learning_rate, warmup_steps, learning_rate * alpha, total_steps,
        )
    else:
        logger.info(
            "%s: CosineDecay lr=%.0e→%.0e total=%d",
            model_name, learning_rate, learning_rate * alpha, total_steps,
        )
    return schedule


# ══════════════════════════════════════════════════════════════════════════════
# iTransformer v1
# ══════════════════════════════════════════════════════════════════════════════

def build_itransformer(
    history_length: int   = 192,
    forecast_horizon: int = 24,
    n_features: int       = 26,
    d_model: int          = 128,
    num_heads: int        = 4,
    num_layers: int       = 4,
    dff: int              = 256,
    dropout: float        = 0.12,
    learning_rate: float  = 1e-4,
    use_cosine_decay: bool  = True,
    total_steps: int        = 0,
    warmup_ratio: float     = 0.05,
    alpha: float            = 0.02,
    huber_delta: float      = 0.05,
    use_seasonal_skip: bool = True,
    seasonal_blend_init: float = 0.60,
) -> tf.keras.Model:
    """
    iTransformer v1 (Liu et al., ICLR 2024).

    v16 FIX: выставляет model._enc_blocks для attention_visualization.py.
    """
    from models.lstm import _SeasonalSkipLayer

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_seq")

    # RevIN нормировка потребления (channel 0)
    cons_in = tf.keras.layers.Lambda(lambda t: t[:, :, :1], name="cons_in")(inp)
    covs_in = tf.keras.layers.Lambda(lambda t: t[:, :, 1:], name="covs_in")(inp)
    mean_c = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True), name="revin_mean")(cons_in)
    std_c  = tf.keras.layers.Lambda(
        lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6,
        name="revin_std")(cons_in)
    cons_n = tf.keras.layers.Lambda(
        lambda xs: (xs[0] - xs[1]) / xs[2],
        name="revin_norm")([cons_in, mean_c, std_c])
    x = tf.keras.layers.Concatenate(axis=-1, name="revin_concat")([cons_n, covs_in])

    # Transpose → variate-first
    x = tf.keras.layers.Permute((2, 1), name="to_variate_first")(x)  # (N,F,T)

    # Per-variate temporal embedding → d_model
    x = tf.keras.layers.Dense(d_model, use_bias=False, name="variate_embed")(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="embed_ln")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5, name="embed_drop")(x)

    # iTransformer blocks
    enc_blocks = []
    for i in range(num_layers):
        blk = iTransformerBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout=dropout, name=f"itrans_blk_{i}")
        x = blk(x)
        enc_blocks.append(blk)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="enc_ln")(x)

    # Per-variate forecast projection
    per_var = tf.keras.layers.Dense(forecast_horizon, name="per_var_proj")(x)  # (N,F,H)

    # Только токен потребления (variate[0])
    neural_out = tf.keras.layers.Lambda(
        lambda t: t[:, 0, :], name="variate_pool")(per_var)  # (N,H)

    # RevIN denorm
    mean_2d = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="mean_2d")(mean_c)
    std_2d  = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="std_2d")(std_c)
    neural_out = tf.keras.layers.Lambda(
        lambda xs: xs[0] * xs[2] + xs[1],
        name="revin_denorm")([neural_out, mean_2d, std_2d])

    # SeasonalSkip
    if use_seasonal_skip and history_length >= forecast_horizon:
        naive_24h = tf.keras.layers.Lambda(
            lambda t: t[:, -forecast_horizon:, 0], name="naive_24h")(inp)
        final_out = _SeasonalSkipLayer(
            init_w=seasonal_blend_init, name="seasonal_skip_24h")([neural_out, naive_24h])

        if history_length >= (168 + forecast_horizon):
            weekly_naive = tf.keras.layers.Lambda(
                lambda t: t[:, -168:-168 + forecast_horizon, 0],
                name="weekly_naive")(inp)
            final_out = _SeasonalSkipLayer(
                init_w=0.80, name="seasonal_skip_168h")([final_out, weekly_naive])
        logger.info(
            "iTransformer: SeasonalSkip 24h (init=%.2f)%s",
            seasonal_blend_init,
            " + 168h" if history_length >= (168 + forecast_horizon) else "",
        )
    else:
        final_out = neural_out

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)
    model = tf.keras.Model(inputs=inp, outputs=final_out, name="iTransformer_v1")

    # v16 FIX: выставляем _enc_blocks для attention_visualization.py
    model._enc_blocks = enc_blocks

    # Optimizer
    lr_sched = _build_lr_schedule(learning_rate, use_cosine_decay, total_steps,
                                  warmup_ratio, alpha, "iTransformer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )

    skip_str = ""
    if use_seasonal_skip and history_length >= forecast_horizon:
        skip_str = " 24h"
        if history_length >= (168 + forecast_horizon):
            skip_str += "+168h"
    logger.info(
        "iTransformer v1 | %d params | input=(%d,%d) | F=%d variate tokens | "
        "d=%d h=%d L=%d dff=%d drop=%.2f | RevIN=True | "
        "SeasonalSkip=%s | WarmupCosine=%s warmup=%.0f%% | lr=%.0e | Huber(δ=%.2f) | "
        "_enc_blocks=%d [v16 FIX]",
        model.count_params(), history_length, n_features, n_features,
        d_model, num_heads, num_layers, dff, dropout,
        skip_str or "off",
        use_cosine_decay and total_steps > 100,
        warmup_ratio * 100,
        learning_rate, huber_delta,
        len(enc_blocks),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PatchTST v6 (+ отдельные patchtst_lr / patchtst_dropout)
# ══════════════════════════════════════════════════════════════════════════════

class _PatchEmbedding(tf.keras.layers.Layer):
    """Patch embedding: разбивает временной ряд на патчи и проецирует в d_model."""

    def __init__(self, patch_len: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.d_model   = d_model
        self.proj  = tf.keras.layers.Dense(d_model, use_bias=False)
        self.ln    = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop  = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None):
        x = self.proj(x)
        x = self.ln(x)
        return self.drop(x, training=training)

    def get_config(self):
        return {**super().get_config(), "patch_len": self.patch_len, "d_model": self.d_model}


def build_patchtst(
    history_length: int   = 192,
    forecast_horizon: int = 24,
    patch_len: int        = 24,
    stride: int           = 12,
    n_features: int       = 26,
    d_model: int          = 128,
    num_heads: int        = 4,
    num_layers: int       = 3,
    dff: int              = 256,
    dropout: float        = 0.12,          # используется как fallback если patchtst_dropout не задан
    learning_rate: float  = 1e-4,          # fallback
    stochastic_depth_rate: float = 0.08,
    use_revin: bool        = True,
    use_cosine_decay: bool = True,
    total_steps: int       = 0,
    warmup_ratio: float    = 0.05,
    alpha: float           = 0.02,
    huber_delta: float     = 0.05,
    use_seasonal_skip: bool = True,
    seasonal_blend_init: float = 0.60,
    # v16 FIX: отдельные параметры PatchTST (приоритет над общими)
    patchtst_learning_rate: Optional[float] = None,  # если None → используется learning_rate
    patchtst_dropout: Optional[float]       = None,  # если None → используется dropout
) -> tf.keras.Model:
    """
    PatchTST v6 (Nie et al., 2023).

    v16 FIX: добавлены patchtst_learning_rate и patchtst_dropout.
      БЫЛО: learning_rate=1e-4, dropout=0.08 → best_epoch=17 из 61.
      СТАЛО: patchtst_learning_rate=3e-4 (из Config.PATCHTST_LEARNING_RATE),
             patchtst_dropout=0.05 (из Config.PATCHTST_DROPOUT).
             Передаются из main.py явно.
    """
    from models.lstm import _SeasonalSkipLayer

    # Применяем patchtst-специфичные параметры если заданы
    eff_lr      = patchtst_learning_rate if patchtst_learning_rate is not None else learning_rate
    eff_dropout = patchtst_dropout       if patchtst_dropout       is not None else dropout

    n_patches = (history_length - patch_len) // stride + 1

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_seq")

    # RevIN (consumption channel 0 only)
    if use_revin:
        cons_in = tf.keras.layers.Lambda(lambda t: t[:, :, :1], name="cons_in")(inp)
        covs_in = tf.keras.layers.Lambda(lambda t: t[:, :, 1:], name="covs_in")(inp)
        mean_c  = tf.keras.layers.Lambda(
            lambda t: tf.reduce_mean(t, axis=1, keepdims=True), name="revin_mean")(cons_in)
        std_c   = tf.keras.layers.Lambda(
            lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6,
            name="revin_std")(cons_in)
        cons_n  = tf.keras.layers.Lambda(
            lambda xs: (xs[0] - xs[1]) / xs[2],
            name="revin_norm")([cons_in, mean_c, std_c])
        x_seq   = tf.keras.layers.Concatenate(
            axis=-1, name="revin_concat")([cons_n, covs_in])
    else:
        x_seq = inp
        mean_c = std_c = None

    x_seq = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x_seq)

    # Patch extraction (strided slicing)
    patches_list = []
    for p_start in range(0, history_length - patch_len + 1, stride):
        patch = tf.keras.layers.Lambda(
            lambda t, s=p_start: t[:, s:s + patch_len, :],
            name=f"patch_{p_start}")(x_seq)
        flat_p = tf.keras.layers.Reshape(
            (patch_len * n_features,),
            name=f"flat_patch_{p_start}")(patch)
        patches_list.append(flat_p)

    patches = tf.keras.layers.Lambda(
        lambda ts: tf.stack(ts, axis=1),
        name="stack_patches")(patches_list)  # (N, n_patches, patch_len*F)

    # Patch embedding
    x = _PatchEmbedding(
        patch_len=patch_len, d_model=d_model,
        dropout=eff_dropout * 0.5, name="patch_embed")(patches)

    # Learned positional embeddings
    pos_idx = tf.keras.layers.Lambda(
        lambda t: tf.tile(tf.expand_dims(tf.range(n_patches), 0), [tf.shape(t)[0], 1]),
        name="pos_range")(x)
    pos_emb = tf.keras.layers.Embedding(n_patches, d_model, name="pos_embed")(pos_idx)
    x = tf.keras.layers.Add(name="add_pos")([x, pos_emb])

    # Transformer encoder
    enc_blocks = []
    for i in range(num_layers):
        blk = PreLNEncoderBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout=eff_dropout, stochastic_depth_rate=stochastic_depth_rate,
            name=f"enc_blk_{i}")
        x = blk(x)
        enc_blocks.append(blk)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="enc_ln")(x)

    # Output head
    flat = tf.keras.layers.Flatten(name="flatten")(x)
    flat = tf.keras.layers.Dropout(eff_dropout, name="head_drop")(flat)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="forecast_head")(flat)

    # RevIN denorm
    if use_revin and mean_c is not None:
        mean_2d = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="mean_2d")(mean_c)
        std_2d  = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="std_2d")(std_c)
        neural_out = tf.keras.layers.Lambda(
            lambda xs: xs[0] * xs[2] + xs[1],
            name="revin_denorm")([neural_out, mean_2d, std_2d])

    # SeasonalSkip
    if use_seasonal_skip and history_length >= forecast_horizon:
        naive_24h = tf.keras.layers.Lambda(
            lambda t: t[:, -forecast_horizon:, 0], name="naive_24h")(inp)
        final_out = _SeasonalSkipLayer(
            init_w=seasonal_blend_init, name="seasonal_skip_24h")([neural_out, naive_24h])

        if history_length >= (168 + forecast_horizon):
            weekly_naive = tf.keras.layers.Lambda(
                lambda t: t[:, -168:-168 + forecast_horizon, 0],
                name="weekly_naive")(inp)
            final_out = _SeasonalSkipLayer(
                init_w=0.80, name="seasonal_skip_168h")([final_out, weekly_naive])
    else:
        final_out = neural_out

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)
    model = tf.keras.Model(inputs=inp, outputs=final_out, name="PatchTST_v6")
    model._enc_blocks = enc_blocks

    # Optimizer с patchtst-специфичным LR
    lr_sched = _build_lr_schedule(eff_lr, use_cosine_decay, total_steps,
                                  warmup_ratio, alpha, "PatchTST")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )

    logger.info(
        "PatchTST v6 patch=%d stride=%d n_p=%d d=%d h=%d L=%d sdrop=%.2f "
        "RevIN=%s SeasonalSkip=%s | lr=%.0e drop=%.2f [v16: individual params] "
        "WarmupCosine=%s warmup=%.0f%% | %d params",
        patch_len, stride, n_patches, d_model, num_heads, num_layers,
        stochastic_depth_rate, use_revin,
        "24h+168h" if (use_seasonal_skip and history_length >= 168 + forecast_horizon)
        else ("24h" if use_seasonal_skip else "off"),
        eff_lr, eff_dropout,
        use_cosine_decay and total_steps > 100,
        warmup_ratio * 100,
        model.count_params(),
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# TFT-Lite v2 (+ WarmupCosineDecay + RevIN denorm)
# ══════════════════════════════════════════════════════════════════════════════

def build_tft_lite(
    history_length: int   = 192,
    forecast_horizon: int = 24,
    d_model: int          = 128,
    num_heads: int        = 4,
    num_layers: int       = 3,
    dropout: float        = 0.12,
    learning_rate: float  = 1e-4,
    n_covariate_features: int = 10,
    use_cosine_decay: bool  = True,
    total_steps: int        = 0,
    warmup_ratio: float     = 0.05,
    alpha: float            = 0.02,
    huber_delta: float      = 0.05,
) -> tf.keras.Model:
    """TFT-Lite v2 с WarmupCosineDecay и RevIN denorm."""
    key_dim = max(d_model // num_heads, 8)

    inp_series = tf.keras.Input(shape=(history_length, 1), name="series_input")
    inp_covars = tf.keras.Input(shape=(history_length, n_covariate_features),
                                name="covars_input")

    # RevIN на серии потребления
    mean_s = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True), name="revin_mean")(inp_series)
    std_s  = tf.keras.layers.Lambda(
        lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6,
        name="revin_std")(inp_series)
    series_n = tf.keras.layers.Lambda(
        lambda xs: (xs[0] - xs[1]) / xs[2],
        name="revin_norm")([inp_series, mean_s, std_s])

    combined = tf.keras.layers.Concatenate(axis=-1, name="combined")([series_n, inp_covars])

    x = GatedResidualNetwork(d_model, dropout=dropout, name="input_grn")(combined)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x)

    for i in range(num_layers):
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        h = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=dropout, name=f"mha_{i}")(h, h, h)
        h = tf.keras.layers.Dropout(dropout, name=f"drop_mha_{i}")(h)
        x = tf.keras.layers.Add(name=f"add_mha_{i}")([x, h])
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        h = GatedResidualNetwork(d_model, dropout=dropout, name=f"grn_{i}")(h)
        x = tf.keras.layers.Add(name=f"add_grn_{i}")([x, h])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="enc_ln")(x)

    last = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_tok")(x)
    last = tf.keras.layers.Dropout(dropout, name="head_drop")(last)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="forecast")(last)

    # RevIN denorm
    mean_2d = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="mean_2d")(mean_s)
    std_2d  = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="std_2d")(std_s)
    neural_out = tf.keras.layers.Lambda(
        lambda xs: xs[0] * xs[2] + xs[1],
        name="revin_denorm")([neural_out, mean_2d, std_2d])

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(neural_out)

    model = tf.keras.Model(
        inputs=[inp_series, inp_covars], outputs=final_out, name="TFTLite_v2")

    lr_sched = _build_lr_schedule(learning_rate, use_cosine_decay, total_steps,
                                  warmup_ratio, alpha, "TFT-Lite")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )

    logger.info(
        "TFTLite v2 d=%d h=%d L=%d | %d params | RevIN=True denorm=True | "
        "WarmupCosine=%s warmup=%.0f%% | Huber(δ=%.2f)",
        d_model, num_heads, num_layers, model.count_params(),
        use_cosine_decay and total_steps > 100,
        warmup_ratio * 100, huber_delta,
    )
    return model