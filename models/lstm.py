# -*- coding: utf-8 -*-
"""models/lstm.py — TCN-BiLSTM-Attention v12.

ИЗМЕНЕНИЯ v12 vs v11:
═══════════════════════════════════════════════════════════════════════════════

ИСПРАВЛЕНИЕ #1: use_cosine_decay фактически не работал.
  БЫЛО: `del use_cosine_decay, total_steps` — параметр удалялся, расписание не строилось.
  СТАЛО: при use_cosine_decay=True строится CosineDecay schedule.
         При total_steps <= 0 — fallback на фиксированный LR.

ИСПРАВЛЕНИЕ #2: use_seasonal_skip=True (по умолчанию при seasonal_diff=False).
  БЫЛО: use_seasonal_skip=False по умолчанию — seasonal skip никогда не включался.
  СТАЛО: seasonal skip включается в main.py явно.
         При history_length >= 168+forecast_horizon активируется weekly skip (lag-168h).
         Это ключевое изменение для победы нейросетей над LinReg.

АРХИТЕКТУРА v12:
  Input(192, 26)
    │
    ├─ RevIN(cons) + Concat + LayerNorm("input_ln")
    ├─ TCN(filters, dil=[1,2,4,8], LayerNorm×each)
    ├─ BiLSTM(units, return_seq) → LayerNorm
    ├─ TemporalAttention(heads) → context
    ├─ Concat([last, context]) → Dense(256/128) → Dense(horizon)
    ├─ AR-shortcut(lag_features)
    └─ SeasonalSkip:
         final = blend_24 * neural + (1-blend_24) * lag_24h  [всегда при seasonal_skip]
         final = blend_168 * final + (1-blend_168) * lag_168h  [при history >= 168+horizon]

  Inductive bias: модель стартует близко к seasonal naive и обучается
  корректировать его нелинейными взаимодействиями признаков.
  Это снижает ACF(24) остатков с первых эпох.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import math
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("smart_grid.models.lstm")

__all__ = ["TemporalAttentionBlock", "TCNBlock", "_SeasonalSkipLayer",
           "WarmupCosineDecay", "build_lstm_model"]


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE: WarmupCosineDecay
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Линейный warmup (0→peak_lr) + CosineDecay (peak_lr→alpha*peak_lr).

    МОТИВАЦИЯ для LSTM:
      В v12 использовался CosineDecay без warmup. best_epoch=78 из 84 означал,
      что к моменту остановки LR уже упал до ~3e-6 (почти ноль), но модель
      всё ещё улучшалась. WarmupCosineDecay решает двойную проблему:
        1) Warmup защищает _SeasonalSkipLayer.blend_logit от ранних больших
           обновлений (blend стартует с 0.60, первые шаги могут его сбить).
        2) Более плавное падение LR → больше «активного» обучения в середине.

    При warmup_steps=0 работает как чистый CosineDecay (совместимость).
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

        warmup_lr = peak * (step_f / ws)
        cosine_steps = tf.maximum(step_f - ws, 0.0)
        cosine_total = tf.maximum(ts - ws, 1.0)
        progress     = tf.minimum(cosine_steps / cosine_total, 1.0)
        cosine_lr    = peak * (alpha + (1.0 - alpha) * 0.5 *
                               (1.0 + tf.cos(math.pi * progress)))
        return tf.where(step_f < ws, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr":      self.peak_lr,
            "total_steps":  self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha":        self.alpha,
            "name":         self._name,
        }


class TCNBlock(tf.keras.layers.Layer):
    """Dilated causal Conv1D block with residual + LayerNorm."""

    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate  = dropout_rate

        pad = (kernel_size - 1) * dilation_rate
        self.pad1    = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))
        self.conv1   = tf.keras.layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="valid", use_bias=False)
        self.ln1     = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.act1    = tf.keras.layers.Activation("gelu")
        self.drop1   = tf.keras.layers.Dropout(dropout_rate)
        self.pad2    = tf.keras.layers.ZeroPadding1D(padding=(pad, 0))
        self.conv2   = tf.keras.layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="valid", use_bias=False)
        self.ln2     = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.act2    = tf.keras.layers.Activation("gelu")
        self.proj    = tf.keras.layers.Conv1D(filters, 1, padding="same", use_bias=False)
        self.ln_proj = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        res = self.ln_proj(self.proj(x))
        h   = self.conv1(self.pad1(x))
        h   = self.act1(self.ln1(h))
        h   = self.drop1(h, training=training)
        h   = self.conv2(self.pad2(h))
        h   = self.ln2(h)
        return self.act2(h + res)

    def get_config(self):
        return {**super().get_config(), "filters": self.filters,
                "kernel_size": self.kernel_size, "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate}


class TemporalAttentionBlock(tf.keras.layers.Layer):
    """Temporal MHA: последний токен как query, вся последовательность как key/value."""

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


class _SeasonalSkipLayer(tf.keras.layers.Layer):
    """
    Learnable blend между neural output и seasonal naive.
    output = sigmoid(logit) * neural + (1 - sigmoid(logit)) * naive

    При init_w=0.60: logit_init = log(0.60/0.40) ≈ 0.405
    Модель стартует с 60% neural + 40% naive (при seasonal_diff=False это означает
    сильный prior к seasonal pattern), постепенно обучаясь увеличивать долю neural.
    """

    def __init__(self, init_w: float = 0.60, **kwargs):
        super().__init__(**kwargs)
        self.init_w = init_w

    def build(self, input_shape):
        logit_init = math.log(max(self.init_w, 1e-4) / max(1.0 - self.init_w, 1e-4))
        self.logit = self.add_weight(
            name="blend_logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(logit_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        neural, naive = inputs
        w = tf.keras.activations.sigmoid(self.logit)
        return w * neural + (1.0 - w) * naive

    def get_config(self):
        return {**super().get_config(), "init_w": self.init_w}


def build_lstm_model(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 26,
    lstm_units_1: int = 128,
    lstm_units_2: int = 128,      # compat
    lstm_units_3: int = 128,      # compat
    dropout_rate: float = 0.12,
    learning_rate: float = 1e-4,
    attn_heads: int = 4,
    use_cosine_decay: bool = True,
    total_steps: int = 0,
    warmup_ratio: float = 0.05,   # НОВОЕ: 5% шагов — линейный warmup 0→peak_lr
    mc_dropout: bool = False,
    huber_delta: float = 0.05,
    tcn_filters: int = 64,
    use_seasonal_skip: bool = True,
    seasonal_blend_init: float = 0.60,
    use_autoregressive_shortcut: bool = True,
    lag_feature_start_idx: int = 15,
) -> tf.keras.Model:
    """
    TCN-BiLSTM-Attention v12.

    Ключевые изменения vs v11:
      1. use_cosine_decay фактически работает (строит CosineDecay schedule).
      2. use_seasonal_skip=True по умолчанию.
      3. Две ветки seasonal skip: lag-24h (всегда) + lag-168h (при history >= 168+horizon).
      4. huber_delta берётся из Config.LSTM_HUBER_DELTA через main.py — не хардкод.

    Seasonal skip strategy:
      При use_seasonal_skip=True:
        final_out = blend_24 * neural + (1-blend_24) * lag_24h
        Если history >= 168+horizon:
          final_out = blend_168 * final_out + (1-blend_168) * lag_168h

      Начальные веса: blend_24_init=seasonal_blend_init (0.60), blend_168_init=0.80
      (lag-168h менее информативен чем lag-24h, поэтому выше вес neural component).

    Args:
        use_cosine_decay: Если True и total_steps > 0, строит CosineDecay schedule.
                         Полезно с большим числом эпох (>=300) в optimal/full mode.
        total_steps:      Общее число шагов = EPOCHS * (N_train // BATCH_SIZE).
                         Вычисляется в main.py. Если 0 — fallback на фиксированный LR.
    """
    del lstm_units_2, lstm_units_3  # compat params, not used
    training_flag: Optional[bool] = True if mc_dropout else None

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # ── RevIN нормализация consumption-канала ─────────────────────────────────
    cons_slice = inp[:, :, :1]
    cov_slice  = inp[:, :, 1:]
    mean_cons  = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=1, keepdims=True), name="revin_mean")(cons_slice)
    std_cons   = tf.keras.layers.Lambda(
        lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6, name="revin_std")(cons_slice)
    cons_norm  = tf.keras.layers.Lambda(
        lambda xs: (xs[0]-xs[1])/xs[2], name="revin_norm")([cons_slice, mean_cons, std_cons])

    x = tf.keras.layers.Concatenate(axis=-1, name="revin_concat")([cons_norm, cov_slice])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x)

    # ── TCN (LayerNorm, dilations [1,2,4,8]) ─────────────────────────────────
    tcn_proj = tf.keras.layers.Conv1D(tcn_filters, 1, padding="same", name="tcn_proj")(x)
    branches = [
        TCNBlock(tcn_filters, kernel_size=3, dilation_rate=d, dropout_rate=0.04, name=f"tcn_d{d}")(
            tcn_proj, training=training_flag)
        for d in [1, 2, 4, 8]
    ]
    tcn_out = tf.keras.layers.Concatenate(axis=-1, name="tcn_merge")(branches)
    tcn_out = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="tcn_ln")(tcn_out)
    tcn_out = tf.keras.layers.Dropout(0.10, name="tcn_drop")(tcn_out, training=training_flag)

    # ── BiLSTM ────────────────────────────────────────────────────────────────
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units_1, return_sequences=True, recurrent_dropout=0.0),
        name="bilstm_1",
    )(tcn_out)
    bilstm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="bilstm_ln")(bilstm)
    bilstm_dim = lstm_units_1 * 2

    # ── Temporal Attention ────────────────────────────────────────────────────
    key_dim = max(bilstm_dim // attn_heads, 8)
    context = TemporalAttentionBlock(
        num_heads=attn_heads, key_dim=key_dim, dropout=dropout_rate * 0.5, name="temporal_attn",
    )(bilstm, training=training_flag)
    last_token = tf.keras.layers.Dropout(
        dropout_rate * 0.5, name="drop_last")(bilstm[:, -1, :], training=training_flag)
    agg = tf.keras.layers.Concatenate(name="agg")([last_token, context])

    # ── Dense head (256/128 для улучшения ёмкости модели) ─────────────────────
    h = tf.keras.layers.Dense(
        256, activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-6), name="head_d1")(agg)
    h = tf.keras.layers.Dropout(dropout_rate, name="head_drop1")(h, training=training_flag)
    h = tf.keras.layers.Dense(
        128, activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-6), name="head_d2")(h)
    h = tf.keras.layers.Dropout(dropout_rate * 0.5, name="head_drop2")(h, training=training_flag)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="neural_output")(h)

    # ── RevIN denorm → MinMaxScaler space (ПЕРЕД AR shortcut) ────────────────
    # ИСПРАВЛЕНИЕ v14: В v13 RevIN denorm применялся ПОСЛЕ neural_plus_ar.
    #
    # БАГ v13: neural_plus_ar смешивал:
    #   - neural_output   — в RevIN пространстве (~[-3,3])
    #   - ar_shortcut     — в MinMaxScaler пространстве ([0,1])
    #   - lag_shortcut    — в MinMaxScaler пространстве ([0,1])
    # Затем denorm (×std + mean) умножал MinMaxScaler-значения AR shortcut на
    # std_cons≈0.05-0.15 → сжимал их почти в ноль. Это разрушало весь AR-вклад.
    #
    # ИСПРАВЛЕНИЕ: denorm только чистого neural_output (RevIN → MinMaxScaler),
    # ПОТОМ добавляем AR shortcut (оба уже в MinMaxScaler [0,1]).
    # SeasonalSkip получает naive_24h тоже в MinMaxScaler [0,1] → всё согласовано.
    #
    mean_2d = tf.keras.layers.Lambda(
        lambda t: t[:, 0, :], name="revin_mean_2d")(mean_cons)  # (N,1)
    std_2d  = tf.keras.layers.Lambda(
        lambda t: t[:, 0, :], name="revin_std_2d")(std_cons)    # (N,1)
    neural_out = tf.keras.layers.Lambda(
        lambda xs: xs[0] * xs[2] + xs[1],
        name="revin_denorm")([neural_out, mean_2d, std_2d])     # (N,H) в MinMaxScaler [0,1]

    # ── AR-shortcut (lag-признаки из входа, тоже в MinMaxScaler) ──────────────
    # ПОРЯДОК ИСПРАВЛЕН: AR shortcut добавляется ПОСЛЕ revin_denorm,
    # так что и neural_out, и ar_shortcut находятся в одном MinMaxScaler [0,1] пространстве.
    if use_autoregressive_shortcut:
        last_features = inp[:, -1, :]                           # MinMaxScaler space
        ar_shortcut   = tf.keras.layers.Dense(
            forecast_horizon, use_bias=False, name="ar_shortcut")(last_features)
        add_terms = [neural_out, ar_shortcut]

        # Явный lag-shortcut по lag_24h, lag_48h, lag_168h
        lag_end = lag_feature_start_idx + 3
        if n_features >= lag_end:
            lag_feats = inp[:, -1, lag_feature_start_idx:lag_end]  # MinMaxScaler space
            lag_sc    = tf.keras.layers.Dense(
                forecast_horizon, use_bias=False, name="lag_shortcut")(lag_feats)
            add_terms.append(lag_sc)

        neural_out = tf.keras.layers.Add(name="neural_plus_ar")(add_terms)  # все в [0,1] ✓

    # ── Seasonal Skip (v12: две ветки — 24h и 168h) ───────────────────────────
    #
    # Мотивация: при seasonal_diff=False и use_seasonal_skip=True модель
    # явно включает lag-24h (и lag-168h) в выход через обучаемый blend.
    # Это снижает ACF(24) остатков и даёт сильный inductive bias к временным паттернам.
    #
    if use_seasonal_skip and history_length >= forecast_horizon:
        # Ветка 1: суточный skip (lag-24h)
        # inp[:, -24:, 0] = последние 24ч consumption = lag-24h относительно прогноза
        naive_24h = inp[:, -forecast_horizon:, 0]   # (batch, forecast_horizon)
        final_out = _SeasonalSkipLayer(
            init_w=seasonal_blend_init, name="seasonal_skip_24h"
        )([neural_out, naive_24h])

        # Ветка 2: недельный skip (lag-168h) — активируется при history >= 168+horizon
        if history_length >= (168 + forecast_horizon):
            # inp[:, -168:-168+24, 0] = потребление 168ч назад = lag-168h
            weekly_naive = inp[:, -(168):-168 + forecast_horizon, 0]
            # Начальный вес 0.80: lag-24h уже учтён, lag-168h — дополнительный prior
            final_out = _SeasonalSkipLayer(
                init_w=0.80, name="seasonal_skip_168h"
            )([final_out, weekly_naive])
        logger.info(
            "SeasonalSkip активирован: lag-24h (init_w=%.2f)%s",
            seasonal_blend_init,
            f" + lag-168h (init_w=0.80)" if history_length >= (168 + forecast_horizon) else ""
        )
    else:
        final_out = neural_out

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)
    model = tf.keras.Model(inputs=inp, outputs=final_out, name="TCN_BiLSTM_Attention_v14")

    # ── Optimizer: WarmupCosineDecay или фиксированный LR ────────────────────
    # УЛУЧШЕНИЕ v13: WarmupCosineDecay вместо чистого CosineDecay.
    #
    # В v12: CosineDecay без warmup. best_epoch=78 из 84 означал, что модель
    #   ещё улучшалась при LR~3e-6 (почти ноль). Первые шаги с полным LR=1e-4
    #   могли сбивать _SeasonalSkipLayer.blend_logit с инициализации 0.60.
    #
    # В v13: WarmupCosineDecay — warmup_steps=5% total_steps (≈1.5 эпохи):
    #   LR растёт 0→1e-4 (защищает blend init), потом косинус 1e-4→2e-6.
    #
    if use_cosine_decay and total_steps > 100:
        warmup_steps = int(warmup_ratio * total_steps) if warmup_ratio > 0 else 0
        lr_schedule = WarmupCosineDecay(
            peak_lr=learning_rate, total_steps=total_steps,
            warmup_steps=warmup_steps, alpha=0.02,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        if warmup_steps > 0:
            logger.info("LSTM: WarmupCosineDecay lr=0→%.0e warmup=%d steps → %.0e",
                        learning_rate, warmup_steps, learning_rate * 0.02)
        else:
            logger.info("LSTM: CosineDecay lr=%.0e→%.0e за %d шагов",
                        learning_rate, learning_rate * 0.02, total_steps)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=huber_delta, name=f"huber_d{huber_delta}"),
        metrics=["mae"],
    )

    weekly_info = " + lag-168h skip" if (use_seasonal_skip and history_length >= (168 + forecast_horizon)) else ""
    logger.info(
        "TCN-BiLSTM-Attention v13 | %d params | input=(%d,%d) | "
        "TCN=%d(LN) dil=[1,2,4,8] | BiLSTM=%d×2 | Attn %dh | Dense 256/128 drop=%.2f | "
        "RevIN=True(denorm) | SeasonalSkip=%s%s blend=%.2f | "
        "WarmupCosine=%s warmup=%.0f%% | lr=%.0e | Huber(δ=%.2f)",
        model.count_params(), history_length, n_features,
        tcn_filters, lstm_units_1, attn_heads, dropout_rate,
        use_seasonal_skip, weekly_info, seasonal_blend_init,
        use_cosine_decay and total_steps > 100,
        warmup_ratio * 100,
        learning_rate, huber_delta,
    )
    return model