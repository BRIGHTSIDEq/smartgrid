# -*- coding: utf-8 -*-
"""models/transformer.py — v15.

ИЗМЕНЕНИЯ v15:
═══════════════════════════════════════════════════════════════════════════════

[1] VanillaTransformer → iTransformer (Liu et al., 2024 — ICLR 2024).

  БЫЛО: VanillaTransformer v4 — attention over T=192 timesteps (temporal tokens).
    ПРОБЛЕМА: Train MAE=0.030 при Val MAE=0.019 всё обучение (inverted gap).
              Модель хуже справляется с тренировочными данными чем с валидационными.
              Синусоидальное PE неинформативно для циклических данных с seasonal shift.
              MAE=8556, худшая нейросеть.

  СТАЛО: iTransformer v1 — attention over F=26 variates (variate tokens).
    КЛЮЧЕВАЯ ИДЕЯ: транспонировать вход (N,T,F) → (N,F,T).
    Каждая переменная (потребление, температура, EV, DSR...) = отдельный токен,
    который обрабатывает свой временной ряд через FFN.
    Self-attention захватывает зависимости МЕЖДУ переменными.
    Никакого PE: порядок variates не меняется, PE не нужен.

  ПРЕИМУЩЕСТВА iTransformer для многомерных временных рядов:
    1. Attention over F=26 variates → понимает "температура ↑ + EV ↑ → потребление ↑↑"
    2. Нет проблем с временным PE (T=192 — слишком длинно для обычного attention)
    3. RevIN нормирует каждый экземпляр → robust к distribution shift
       (test mean=70605 >> train mean=57015 — зимний период)
    4. Proper RevIN: нормируем вход И денормируем выход → согласованный SeasonalSkip

[2] WarmupCosineDecay — новый LR schedule для всех нейросетей.

  БЫЛО: CosineDecay без warmup.
    ПРОБЛЕМА: LSTM best_epoch=78 из 84 — всё ещё учился, LR уже почти 0.
              Первые шаги с большим LR разрушают инициализацию SeasonalSkip.

  СТАЛО: WarmupCosineDecay.
    Фаза 1 (warmup, 5% шагов): LR линейно растёт 0 → peak_lr
    Фаза 2 (cosine, 95% шагов): LR плавно падает peak_lr → alpha*peak_lr
    Warmup защищает learnable blend weights от ранних больших обновлений.

[3] PatchTST: WarmupCosineDecay + SeasonalSkip (был только ReduceLROnPlateau).

  БЫЛО: PatchTST без CosineDecay, без SeasonalSkip.
    ПРОБЛЕМА: Train LOSS продолжал падать до epoch 78 при Val плато с epoch 20.
              best_epoch=36 — ранняя остановка неоптимальна.

  СТАЛО: WarmupCosineDecay + SeasonalSkip (lag-24h + lag-168h при history>=192).
    WarmupCosineDecay даёт плавную траекторию LR, совместимую с PATIENCE=50.
    SeasonalSkip добавляет тот же inductive bias что у LSTM.

[4] Убрана MAPE из training metrics.

  БЫЛО: metrics=["mae", tf.keras.metrics.MeanAbsolutePercentageError()]
    ПРОБЛЕМА: Train MAPE ~17000-20000% (деление на нормализованные значения ≈0).
              VanillaTransformer/PatchTST MAPE-графики были бесполезны.

  СТАЛО: metrics=["mae"] только. MAPE вычисляется при evaluate() на кВт·ч.

[5] Обратная совместимость: классы PreLNEncoderBlock, SinusoidalPE, Time2Vec,
    GatedResidualNetwork сохранены для load_keras().
═══════════════════════════════════════════════════════════════════════════════
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
    """
    Линейный warmup + CosineDecay.

    Фаза 1 (шаги 0 → warmup_steps): LR линейно растёт 0 → peak_lr.
      Защищает начальные веса (особенно SeasonalSkip blend) от больших градиентов.
    Фаза 2 (шаги warmup_steps → total_steps): CosineDecay peak_lr → alpha*peak_lr.

    При warmup_steps=0 работает как обычный CosineDecay.
    """

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

        # Фаза 1: линейный warmup 0 → peak
        warmup_lr = peak * (step_f / ws)

        # Фаза 2: cosine decay peak → alpha*peak
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
    """
    Pre-LayerNorm Transformer block для iTransformer.

    Работает на variate-dimension: input (N, F, d_model), attention over F.
    Стандартный Pre-LN MHA + FFN с residual connections.
    """

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

    def call(self, x, training=None):
        # Pre-LN MHA (attention over variate dimension)
        h = self.ln1(x)
        h = self.mha(h, h, h, training=training)
        h = self.drop1(h, training=training)
        x = x + h

        # Pre-LN FFN
        h = self.ln2(x)
        h = self.ff1(h)
        h = self.drop2(h, training=training)
        h = self.ff2(h)
        h = self.drop3(h, training=training)
        return x + h

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
    def call(self, x, training=None):
        h = self.ln1(x); h = self.mha(h, h, h, training=training)
        h = self.drop1(h, training=training); x = x + h
        h = self.ln2(x); h = self.ff1(h); h = self.drop2(h, training=training)
        h = self.ff2(h); h = self.drop3(h, training=training)
        return x + h
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
        self.proj    = None  # built in build()

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
    """Строит WarmupCosineDecay или фиксированный LR. Возвращает schedule или float."""
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
    num_layers: int       = 4,       # 4 layers: iTransformer выигрывает от глубины
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

    Ключевая идея: транспонировать вход (N,T,F) → (N,F,T).
    Каждая переменная = отдельный токен, attention over variates (не time).

    Архитектура:
      Input(T, F)
        │
        ├─ RevIN(cons ch.0): нормировка по экземпляру
        ├─ Permute: (N,T,F) → (N,F,T)
        ├─ Dense(T→d): (N,F,T) → (N,F,d_model) [per-variate embedding]
        ├─ LayerNorm
        ├─ iTransformerBlock × num_layers  [attention over F variates]
        ├─ LayerNorm + Dense(d→H): (N,F,H)
        ├─ Mean over F: (N,H)
        ├─ RevIN denorm: → MinMaxScaler space (согласовано с SeasonalSkip)
        └─ SeasonalSkip: lag-24h + lag-168h (при history>=192)

    Преимущества над VanillaTransformer:
      1. Захватывает зависимости между переменными (temperature↑+EV↑→consumption↑↑)
      2. Нет проблем с PE при длинных последовательностях (T=192)
      3. Proper RevIN устраняет влияние distribution shift (зима vs лето)
      4. WarmupCosineDecay: стабильный старт + глубокое схождение

    Параметры:
      num_layers=4: iTransformer эффективнее при большей глубине (не ширине).
    """
    from models.lstm import _SeasonalSkipLayer

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_seq")

    # ── Шаг 1: RevIN нормировка потребления (channel 0) ──────────────────────
    # Нормируем по экземпляру: каждое окно имеет своё среднее/std.
    # Это устраняет distribution shift между зимой (test) и летом (train).
    cons_in = tf.keras.layers.Lambda(
        lambda t: t[:, :, :1], name="cons_in")(inp)
    covs_in = tf.keras.layers.Lambda(
        lambda t: t[:, :, 1:], name="covs_in")(inp)
    mean_c = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        name="revin_mean")(cons_in)                              # (N,1,1)
    std_c  = tf.keras.layers.Lambda(
        lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6,
        name="revin_std")(cons_in)                               # (N,1,1)
    cons_n = tf.keras.layers.Lambda(
        lambda xs: (xs[0] - xs[1]) / xs[2],
        name="revin_norm")([cons_in, mean_c, std_c])             # (N,T,1) ~ [-3,3]

    x = tf.keras.layers.Concatenate(
        axis=-1, name="revin_concat")([cons_n, covs_in])         # (N,T,F)

    # ── Шаг 2: Transpose → variate-first (ключевая операция iTransformer) ────
    # (N,T,F) → (N,F,T): теперь F переменных = F токенов, каждый длины T
    x = tf.keras.layers.Permute((2, 1), name="to_variate_first")(x)  # (N,F,T)

    # ── Шаг 3: Per-variate temporal embedding → d_model ──────────────────────
    # Dense применяется по последней оси: T → d_model, независимо для каждой переменной
    x = tf.keras.layers.Dense(d_model, use_bias=False,
                              name="variate_embed")(x)           # (N,F,d_model)
    x = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="embed_ln")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5, name="embed_drop")(x)

    # ── Шаг 4: iTransformer blocks (attention over F variates) ───────────────
    for i in range(num_layers):
        x = iTransformerBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout=dropout, name=f"itrans_blk_{i}")(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="enc_ln")(x)

    # ── Шаг 5: Per-variate forecast projection ───────────────────────────────
    per_var = tf.keras.layers.Dense(
        forecast_horizon, name="per_var_proj")(x)               # (N,F,H)

    # ── Шаг 6: Используем ТОЛЬКО токен потребления (variate 0) ───────────────
    # ИСПРАВЛЕНИЕ v2: В v1 использовался tf.reduce_mean по всем F=26 variates.
    # Проблема: усреднение прогнозов температуры, EV, DSR и других переменных
    # с прогнозом потребления давало «размытый» сигнал. Модель показывала
    # inverted gap (train_mae > val_mae) — тренировочные данные хуже предсказывались
    # чем валидационные, что нетипично и указывало на неверную агрегацию.
    #
    # После транспонирования (N,T,F)→(N,F,T) variate[0] = потребление (channel 0).
    # Attention разрешает токену потребления ПОСЕЩАТЬ другие variates и собирать
    # кросс-переменную информацию (temp↑+EV↑→consumption↑↑), но финальный прогноз
    # берётся только из токена потребления — как в оригинальной статье iTransformer.
    #
    # Это соответствует оригинальному iTransformer (Liu et al., ICLR 2024):
    #   "the representation of the target variate is extracted for forecasting"
    #
    neural_out = tf.keras.layers.Lambda(
        lambda t: t[:, 0, :],
        name="variate_pool")(per_var)                            # (N,H) — только потребление

    # ── Шаг 7: RevIN denorm → MinMaxScaler space ─────────────────────────────
    # Согласует neural_out с naive values (которые в MinMaxScaler space)
    # чтобы SeasonalSkip корректно смешивал оба сигнала.
    mean_2d = tf.keras.layers.Lambda(
        lambda t: t[:, 0, :], name="mean_2d")(mean_c)           # (N,1)
    std_2d  = tf.keras.layers.Lambda(
        lambda t: t[:, 0, :], name="std_2d")(std_c)             # (N,1)
    neural_out = tf.keras.layers.Lambda(
        lambda xs: xs[0] * xs[2] + xs[1],
        name="revin_denorm")([neural_out, mean_2d, std_2d])      # (N,H) ~ [0,1]

    # ── Шаг 8: SeasonalSkip (lag-24h + lag-168h) ─────────────────────────────
    if use_seasonal_skip and history_length >= forecast_horizon:
        naive_24h = tf.keras.layers.Lambda(
            lambda t: t[:, -forecast_horizon:, 0],
            name="naive_24h")(inp)                               # (N,H) MinMaxScaler
        final_out = _SeasonalSkipLayer(
            init_w=seasonal_blend_init,
            name="seasonal_skip_24h")([neural_out, naive_24h])

        if history_length >= (168 + forecast_horizon):
            weekly_naive = tf.keras.layers.Lambda(
                lambda t: t[:, -168:-168 + forecast_horizon, 0],
                name="weekly_naive")(inp)
            final_out = _SeasonalSkipLayer(
                init_w=0.80,
                name="seasonal_skip_168h")([final_out, weekly_naive])
        logger.info(
            "iTransformer: SeasonalSkip 24h (init=%.2f)%s",
            seasonal_blend_init,
            " + 168h" if history_length >= (168 + forecast_horizon) else "",
        )
    else:
        final_out = neural_out

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)
    model = tf.keras.Model(inputs=inp, outputs=final_out, name="iTransformer_v1")

    # ── Optimizer: WarmupCosineDecay ─────────────────────────────────────────
    lr_sched = _build_lr_schedule(learning_rate, use_cosine_decay, total_steps,
                                  warmup_ratio, alpha, "iTransformer")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],  # MAPE убрана: делилась бы на нормализованные ~0 значения
    )

    skip_str = ""
    if use_seasonal_skip and history_length >= forecast_horizon:
        skip_str = " 24h"
        if history_length >= (168 + forecast_horizon):
            skip_str += "+168h"
    logger.info(
        "iTransformer v1 | %d params | input=(%d,%d) | F=%d variate tokens | "
        "d=%d h=%d L=%d dff=%d drop=%.2f | RevIN=True | "
        "SeasonalSkip=%s | WarmupCosine=%s warmup=%.0f%% | lr=%.0e | Huber(δ=%.2f)",
        model.count_params(), history_length, n_features, n_features,
        d_model, num_heads, num_layers, dff, dropout,
        skip_str or "off",
        use_cosine_decay and total_steps > 100,
        warmup_ratio * 100,
        learning_rate, huber_delta,
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# PatchTST v6 (+ WarmupCosineDecay + SeasonalSkip)
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
        # x: (N, n_patches, patch_len)
        x = self.proj(x)    # (N, n_patches, d_model)
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
    dropout: float        = 0.12,
    learning_rate: float  = 1e-4,
    stochastic_depth_rate: float = 0.08,
    use_revin: bool        = True,
    use_cosine_decay: bool = True,   # НОВОЕ: WarmupCosineDecay вместо ReduceLROnPlateau
    total_steps: int       = 0,
    warmup_ratio: float    = 0.05,
    alpha: float           = 0.02,
    huber_delta: float     = 0.05,
    use_seasonal_skip: bool = True,  # НОВОЕ: SeasonalSkip (lag-24h + lag-168h)
    seasonal_blend_init: float = 0.60,
) -> tf.keras.Model:
    """
    PatchTST v6 (Nie et al., 2023) с улучшениями:
      - WarmupCosineDecay вместо ReduceLROnPlateau
        (Train LOSS продолжал падать 78 эпох при Val плато с epoch 20 → нестабильный LR)
      - SeasonalSkip (lag-24h + lag-168h при history>=192)
        (добавляет тот же сезонный prior что у LSTM и iTransformer)
      - MAPE убрана из training metrics

    Архитектура:
      Input(T, F)
        │
        ├─ RevIN(ch.0) + LayerNorm
        ├─ Patch extraction: (N, n_patches, patch_len*n_channels)
        ├─ PatchEmbedding: → (N, n_patches, d_model)
        ├─ Learned positional embedding
        ├─ PreLNEncoderBlock × num_layers
        ├─ Flatten → Dense(horizon) → (N, H)
        ├─ RevIN denorm
        └─ SeasonalSkip
    """
    from models.lstm import _SeasonalSkipLayer

    n_patches = (history_length - patch_len) // stride + 1

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_seq")

    # ── RevIN (consumption channel 0) ────────────────────────────────────────
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

    # ── Patch extraction ─────────────────────────────────────────────────────
    # Извлекаем патчи через strided slicing
    patches_list = []
    for p_start in range(0, history_length - patch_len + 1, stride):
        patch = tf.keras.layers.Lambda(
            lambda t, s=p_start: t[:, s:s + patch_len, :],
            name=f"patch_{p_start}")(x_seq)              # (N, patch_len, F)
        flat_p = tf.keras.layers.Reshape(
            (patch_len * n_features,),
            name=f"flat_patch_{p_start}")(patch)         # (N, patch_len*F)
        patches_list.append(flat_p)

    # Stack: (N, n_patches, patch_len*F)
    patches = tf.keras.layers.Lambda(
        lambda ts: tf.stack(ts, axis=1),
        name="stack_patches")(patches_list)

    # ── Patch embedding ───────────────────────────────────────────────────────
    x = _PatchEmbedding(
        patch_len=patch_len, d_model=d_model,
        dropout=dropout * 0.5, name="patch_embed")(patches)  # (N, n_patches, d_model)

    # Learned positional embeddings — фиксированные индексы, broadcast по batch
    pos_idx = tf.keras.layers.Lambda(
        lambda t: tf.tile(
            tf.expand_dims(tf.range(n_patches), 0),
            [tf.shape(t)[0], 1]
        ),
        name="pos_range")(x)                                   # (N, n_patches)
    pos_emb = tf.keras.layers.Embedding(
        n_patches, d_model, name="pos_embed")(pos_idx)         # (N, n_patches, d_model)
    x = tf.keras.layers.Add(name="add_pos")([x, pos_emb])

    # ── Transformer encoder ──────────────────────────────────────────────────
    for i in range(num_layers):
        x = PreLNEncoderBlock(
            d_model=d_model, num_heads=num_heads, dff=dff,
            dropout=dropout, stochastic_depth_rate=stochastic_depth_rate,
            name=f"enc_blk_{i}")(x)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="enc_ln")(x)

    # ── Output head ───────────────────────────────────────────────────────────
    flat = tf.keras.layers.Flatten(name="flatten")(x)       # (N, n_patches*d_model)
    flat = tf.keras.layers.Dropout(dropout, name="head_drop")(flat)
    neural_out = tf.keras.layers.Dense(
        forecast_horizon, name="forecast_head")(flat)       # (N, H)

    # ── RevIN denorm ──────────────────────────────────────────────────────────
    if use_revin and mean_c is not None:
        mean_2d = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="mean_2d")(mean_c)
        std_2d  = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="std_2d")(std_c)
        neural_out = tf.keras.layers.Lambda(
            lambda xs: xs[0] * xs[2] + xs[1],
            name="revin_denorm")([neural_out, mean_2d, std_2d])

    # ── SeasonalSkip (NEW in v6) ───────────────────────────────────────────────
    if use_seasonal_skip and history_length >= forecast_horizon:
        naive_24h = tf.keras.layers.Lambda(
            lambda t: t[:, -forecast_horizon:, 0],
            name="naive_24h")(inp)
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_sched = _build_lr_schedule(learning_rate, use_cosine_decay, total_steps,
                                  warmup_ratio, alpha, "PatchTST")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_sched, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )

    logger.info(
        "PatchTST v6 patch=%d stride=%d n_p=%d d=%d h=%d L=%d sdrop=%.2f "
        "RevIN=%s(ch0 only, denorm) SeasonalSkip=%s | WarmupCosine=%s warmup=%.0f%% | "
        "%d params",
        patch_len, stride, n_patches, d_model, num_heads, num_layers,
        stochastic_depth_rate, use_revin,
        "24h+168h" if (use_seasonal_skip and history_length >= 168 + forecast_horizon)
        else ("24h" if use_seasonal_skip else "off"),
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
    use_cosine_decay: bool  = True,   # НОВОЕ: WarmupCosineDecay
    total_steps: int        = 0,
    warmup_ratio: float     = 0.05,
    alpha: float            = 0.02,
    huber_delta: float      = 0.05,
) -> tf.keras.Model:
    """
    TFT-Lite v2 с WarmupCosineDecay и RevIN denorm.

    Принимает два входа: [series (N,T,1), covariates (N,T,n_cov)].
    series нормируется через RevIN, covariates используются как есть.

    Исправления v2:
      - WarmupCosineDecay: best_epoch=17 был слишком ранним из-за резкого LR
      - RevIN denorm output: согласует с SeasonalSkip scale
      - MBE=-4509 → исправляется за счёт правильной нормировки
    """
    key_dim = max(d_model // num_heads, 8)

    # Входы: series + covariates
    inp_series = tf.keras.Input(shape=(history_length, 1), name="series_input")
    inp_covars = tf.keras.Input(shape=(history_length, n_covariate_features),
                                name="covars_input")

    # ── RevIN на серии потребления ────────────────────────────────────────────
    mean_s = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True),
        name="revin_mean")(inp_series)
    std_s  = tf.keras.layers.Lambda(
        lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6,
        name="revin_std")(inp_series)
    series_n = tf.keras.layers.Lambda(
        lambda xs: (xs[0] - xs[1]) / xs[2],
        name="revin_norm")([inp_series, mean_s, std_s])

    # Concat series + covariates → (N, T, 1+n_cov)
    combined = tf.keras.layers.Concatenate(
        axis=-1, name="combined")([series_n, inp_covars])

    # ── GRN embedding ─────────────────────────────────────────────────────────
    x = GatedResidualNetwork(d_model, dropout=dropout, name="input_grn")(combined)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x)

    # ── Transformer encoder ───────────────────────────────────────────────────
    for i in range(num_layers):
        # Pre-LN MHA
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        h = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim,
            dropout=dropout, name=f"mha_{i}")(h, h, h)
        h = tf.keras.layers.Dropout(dropout, name=f"drop_mha_{i}")(h)
        x = tf.keras.layers.Add(name=f"add_mha_{i}")([x, h])
        # Pre-LN GRN
        h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        h = GatedResidualNetwork(d_model, dropout=dropout, name=f"grn_{i}")(h)
        x = tf.keras.layers.Add(name=f"add_grn_{i}")([x, h])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="enc_ln")(x)

    # ── Output: последний токен → forecast ────────────────────────────────────
    last = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_tok")(x)
    last = tf.keras.layers.Dropout(dropout, name="head_drop")(last)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="forecast")(last)

    # ── RevIN denorm ──────────────────────────────────────────────────────────
    mean_2d = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="mean_2d")(mean_s)
    std_2d  = tf.keras.layers.Lambda(lambda t: t[:, 0, :], name="std_2d")(std_s)
    neural_out = tf.keras.layers.Lambda(
        lambda xs: xs[0] * xs[2] + xs[1],
        name="revin_denorm")([neural_out, mean_2d, std_2d])

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(neural_out)

    model = tf.keras.Model(
        inputs=[inp_series, inp_covars], outputs=final_out, name="TFTLite_v2")

    # ── Optimizer ─────────────────────────────────────────────────────────────
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