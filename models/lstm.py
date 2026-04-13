# -*- coding: utf-8 -*-
"""models/lstm.py — TCN-BiLSTM-Attention v11.

ИЗМЕНЕНИЯ v11 (seasonal differencing):
═══════════════════════════════════════════════════════════════════════════════

КОНТЕКСТ:
  preprocessing.py v9 теперь вычитает seasonal naive из Y_target.
  Модель обучается предсказывать ОТКЛОНЕНИЕ от "вчера в это время".
  При инференсе: pred_abs = model_diff_output + Y_naive
  (выполняется в trainer.py через reconstruct_from_diff).

ИЗМЕНЕНИЕ #1: SeasonalSkip УДАЛЁН.
  БЫЛО:  SeasonalSkipConnection blend naive из inp[-24:, 0] с обучаемым logit.
         Проблема: optimizer сдвигал logit к "чисто нейронному" за 20-30 эпох,
         потому что нейросеть снижает loss быстрее чем naive на ранних эпохах.
         Результат: ACF(24)=0.70 в остатках — seasonal component не устраняется.
  СТАЛО: seasonal компонент вычтен на уровне данных (preprocessing.py v9).
         Модель предсказывает только деviации → SeasonalSkip не нужен.
         AR-shortcut на lag-каналах сохранён для тонкой коррекции.

ИЗМЕНЕНИЕ #2: Huber delta адаптируется к масштабу Y_diff.
  Y_diff имеет std ≈ 0.05–0.08 (vs Y_abs std ≈ 0.20).
  Huber(delta=0.05) → delta=0.02 (более агрессивный против выбросов).
  Kurtosis=8-10 в логах → тяжёлые хвосты → меньший delta лучше.

ИЗМЕНЕНИЕ #3: input_ln (LayerNorm после Concat) сохранён.
  BatchNorm → LayerNorm в TCNBlock — из v10, без изменений.

АРХИТЕКТУРА v11:
  Input(48, 26)  — X в MinMaxScaler пространстве
    │
    ├─ RevIN(cons) + Concat + LayerNorm("input_ln")
    ├─ TCN(filters, dil=[1,2,4,8], LayerNorm×each)
    ├─ BiLSTM(units, return_seq) → LayerNorm
    ├─ TemporalAttention(heads) → context
    ├─ Concat([last, context]) → Dense(128/64) → Dense(horizon)
    └─ AR-shortcut(lag_features) — добавляется к neural_out
         [NO SeasonalSkip — seasonal component removed at data level]

  Output: Y_diff ≈ N(0, σ_small)  — отклонение от seasonal naive
  При evaluate: pred_abs = Y_diff + Y_naive (в trainer.py)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("smart_grid.models.lstm")

__all__ = ["TemporalAttentionBlock", "TCNBlock", "build_lstm_model"]


class TCNBlock(tf.keras.layers.Layer):
    """
    Dilated causal Conv1D block with residual + LayerNorm.
    v10: BatchNorm → LayerNorm (стабильно при любом batch_size).
    """

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


def build_lstm_model(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 26,
    lstm_units_1: int = 96,
    lstm_units_2: int = 96,      # compat
    lstm_units_3: int = 96,      # compat
    dropout_rate: float = 0.12,
    learning_rate: float = 2e-4,
    attn_heads: int = 4,
    use_cosine_decay: bool = False,
    total_steps: int = 37_000,
    mc_dropout: bool = False,
    huber_delta: float = 0.02,   # v11: 0.05→0.02 (масштаб Y_diff меньше; kurtosis>5)
    tcn_filters: int = 48,
    use_seasonal_skip: bool = False,    # v11: по умолчанию False (seasonal diff на уровне данных)
    seasonal_blend_init: float = 0.35,  # compat, игнорируется если use_seasonal_skip=False
    use_autoregressive_shortcut: bool = True,
    lag_feature_start_idx: int = 15,
) -> tf.keras.Model:
    """
    TCN-BiLSTM-Attention v11.

    Ключевое изменение vs v10:
      use_seasonal_skip=False (default): seasonal компонент вычтен в preprocessing.
      Модель предсказывает Y_diff ≈ N(0, σ_small).
      huber_delta=0.02 (масштаб Y_diff значительно меньше Y_abs).

    Если use_seasonal_skip=True — поведение v10 (SeasonalSkip включён).
    Используй True только если seasonal_diff=False в prepare_data().
    """
    del lstm_units_2, lstm_units_3, use_cosine_decay, total_steps
    training_flag: Optional[bool] = True if mc_dropout else None

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # ── RevIN нормализация consumption-канала ─────────────────────────────────
    cons_slice = inp[:, :, :1]
    cov_slice  = inp[:, :, 1:]
    mean_cons  = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True), name="revin_mean")(cons_slice)
    std_cons   = tf.keras.layers.Lambda(lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-6, name="revin_std")(cons_slice)
    cons_norm  = tf.keras.layers.Lambda(lambda xs: (xs[0]-xs[1])/xs[2], name="revin_norm")([cons_slice, mean_cons, std_cons])

    # v10: LayerNorm выравнивает масштаб cons_norm≈N(0,1) и cov≈[0,1]
    x = tf.keras.layers.Concatenate(axis=-1, name="revin_concat")([cons_norm, cov_slice])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="input_ln")(x)

    # ── TCN (LayerNorm, v10 fix) ───────────────────────────────────────────────
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
    last_token = tf.keras.layers.Dropout(dropout_rate * 0.5, name="drop_last")(bilstm[:, -1, :], training=training_flag)
    agg = tf.keras.layers.Concatenate(name="agg")([last_token, context])

    # ── Dense head ────────────────────────────────────────────────────────────
    h = tf.keras.layers.Dense(128, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(1e-5), name="head_d1")(agg)
    h = tf.keras.layers.Dropout(dropout_rate, name="head_drop1")(h, training=training_flag)
    h = tf.keras.layers.Dense(64, activation="gelu",  kernel_regularizer=tf.keras.regularizers.l2(1e-5), name="head_d2")(h)
    h = tf.keras.layers.Dropout(dropout_rate * 0.5, name="head_drop2")(h, training=training_flag)
    neural_out = tf.keras.layers.Dense(forecast_horizon, name="neural_output")(h)

    # ── AR-shortcut (lag-признаки) ─────────────────────────────────────────────
    if use_autoregressive_shortcut:
        last_features = inp[:, -1, :]
        ar_shortcut   = tf.keras.layers.Dense(forecast_horizon, use_bias=False, name="ar_shortcut")(last_features)
        add_terms = [neural_out, ar_shortcut]
        lag_end = lag_feature_start_idx + 3
        if n_features >= lag_end:
            lag_feats = inp[:, -1, lag_feature_start_idx:lag_end]
            lag_sc    = tf.keras.layers.Dense(forecast_horizon, use_bias=False, name="lag_shortcut")(lag_feats)
            add_terms.append(lag_sc)
        neural_out = tf.keras.layers.Add(name="neural_plus_ar")(add_terms)

    # ── SeasonalSkip (опциональный, v10 compat) ───────────────────────────────
    # Включай ТОЛЬКО если seasonal_diff=False в prepare_data().
    if use_seasonal_skip and history_length >= forecast_horizon:
        import math
        class _SeasonalSkip(tf.keras.layers.Layer):
            def __init__(self, init_w=0.35, **kw):
                super().__init__(**kw)
                self.init_w = init_w
            def build(self, _):
                logit_init = math.log(self.init_w / (1.0 - self.init_w + 1e-8))
                self.logit = self.add_weight("blend_logit", shape=(), initializer=tf.keras.initializers.Constant(logit_init), trainable=True)
                super().build(_)
            def call(self, inputs):
                n, v = inputs; w = tf.keras.activations.sigmoid(self.logit)
                return w * n + (1.0 - w) * v
            def get_config(self): return {**super().get_config(), "init_w": self.init_w}

        naive_slice = inp[:, -forecast_horizon:, 0]
        final_out = _SeasonalSkip(init_w=seasonal_blend_init, name="seasonal_skip")([neural_out, naive_slice])
        if history_length >= (168 + forecast_horizon):
            weekly_slice = inp[:, -168:-168+forecast_horizon, 0]
            final_out = _SeasonalSkip(init_w=0.80, name="seasonal_weekly_skip")([final_out, weekly_slice])
    else:
        final_out = neural_out

    final_out = tf.keras.layers.Lambda(lambda t: t, name="output")(final_out)
    model = tf.keras.Model(inputs=inp, outputs=final_out, name="TCN_BiLSTM_Attention_v11")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta, name=f"huber_d{huber_delta}"),
        metrics=["mae"],
    )
    logger.info(
        "TCN-BiLSTM-Attention v11 | %d params | input=(%d,%d) | "
        "TCN=%d(LN) dil=[1,2,4,8] | BiLSTM=%d×2 | Attn %dh | Dense 128/64 drop=%.2f | "
        "SeasonalSkip=%s | SeasonalDiff=True | lr=%.0e | Huber(δ=%.2f)",
        model.count_params(), history_length, n_features,
        tcn_filters, lstm_units_1, attn_heads, dropout_rate,
        use_seasonal_skip, learning_rate, huber_delta,
    )
    return model