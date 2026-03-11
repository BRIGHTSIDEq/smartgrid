# -*- coding: utf-8 -*-
"""
models/lstm.py — AttentionLSTM v4: гибрид LSTM + Temporal Self-Attention.

═══════════════════════════════════════════════════════════════════════════════
АРХИТЕКТУРНЫЕ УЛУЧШЕНИЯ v4
═══════════════════════════════════════════════════════════════════════════════

УЛУЧШЕНИЕ 1: Temporal Multi-Head Self-Attention поверх LSTM
────────────────────────────────────────────────────────────
Проблема v3: LSTM сжимает всю 48-часовую историю в один вектор (hidden state).
  Паттерны «вчера в это же время» (лаг-24) и «позавчера» (лаг-48) теряются
  при «просачивании» через рекуррентные ворота — LSTM им не уделяет
  явного внимания, фокусируясь на ближайших шагах.

Решение: Self-Attention ПОСЛЕ LSTM-стека на полных скрытых состояниях.
  LSTM(return_sequences=True) → все hidden states (B, 48, d)
  → MHA(Q=last_token, K=V=all_hiddens)  ← «запрос к прошлому»
  → взвешенный контекст + skip-connection
  → Concat([last_token, attn_context])

  Интерпретация весов Attention:
  - Высокий вес на шаге t-24: «вчера в это же время» определяет прогноз
  - Высокий вес на шаге t-1..t-3: «инерционный» режим
  - Разные головы специализируются на разных лагах (24ч, 48ч, первые часы)

Обоснование:
  [Bahdanau et al., 2014] — Attention for seq2seq
  [Luong et al., 2015]    — dot-product / general / concat variants
  [Hao & Liu, 2019]       — Temporal Attention LSTM для прогноза нагрузки
  Результат: прямой путь к любому шагу истории за O(T) вместо
  экспоненциального затухания градиента в рекуррентных связях.

УЛУЧШЕНИЕ 2: LayerNormalization после каждого LSTM-блока
─────────────────────────────────────────────────────────
Проблема: без нормализации масштаб выходов LSTM нарастает/уменьшается
  между слоями, особенно при глубоких сетях.

Решение: LayerNorm (не BatchNorm!) после каждого LSTM + Dropout.
  BatchNorm нормализует по batch-размерности → нарушает временну́ю структуру.
  LayerNorm нормализует по feature-размерности → безопасно для RNN.
  [Ba et al., 2016 — Layer Normalization]

УЛУЧШЕНИЕ 3: GELU + Residual в Dense-голове
────────────────────────────────────────────
GELU вместо ReLU:
  - ReLU: «мёртвые нейроны» при отрицательных входах (градиент=0 навсегда)
  - GELU: мягкий переход, сохраняет информацию при небольших отрицательных x
  - Стандарт в GPT-2, BERT, всех современных LLM [Hendrycks & Gimpel, 2016]

Residual в голове: h = Dense(256, gelu) + proj(256) → Add → LN
  Устраняет деградацию при глубокой Dense-голове, ускоряет сходимость.

УЛУЧШЕНИЕ 4: CosineDecay Learning Rate Schedule
─────────────────────────────────────────────────
Flat LR = постоянный компромисс exploration/exploitation всё обучение.
CosineDecay: высокий LR в начале (широкое исследование) → плавный спуск
  к концу (точная настройка финального минимума).
  [Loshchilov & Hutter, 2016 — SGDR: Stochastic Gradient Descent]
  Улучшает финальное качество без ручного подбора lr_factor/patience.

═══════════════════════════════════════════════════════════════════════════════
АРХИТЕКТУРА — граф (optimal mode: 500 хоз-в, units=256/128/64)
═══════════════════════════════════════════════════════════════════════════════

Input(48, 11)
  │
  ├─ Dense(256) ← input projection: n_features → lstm_units_1
  │
  ├─ LSTM(256, return_sequences=True) → LayerNorm → Dropout(0.20)
  ├─ LSTM(128, return_sequences=True) → LayerNorm → Dropout(0.20)
  ├─ LSTM(64,  return_sequences=True) → LayerNorm
  │    └─ all_hidden_states: shape (B, 48, 64)
  │
  ├─ [TemporalAttentionBlock]
  │    Q = last_token = hidden[:, -1, :]              ← актуальное состояние
  │    K = V = all_hidden_states                       ← вся история
  │    → MHA(num_heads=4, key_dim=16) → (B, 48, 64)
  │    → GlobalAvgPool → LayerNorm → Dropout           context: (B, 64)
  │
  ├─ Concat([last_token(B,64), attn_context(B,64)])   → (B, 128)
  │
  └─ Dense Head (GELU + Residual):
       Dense(256, gelu) + skip_proj(256) → Add → LayerNorm → Dropout(0.15)
       Dense(128, gelu) → Dropout(0.10)
       Dense(24)  ← прямой multi-step прогноз

Параметры (optimal): ~600K  |  (fast): ~190K
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from typing import Optional

import tensorflow as tf

logger = logging.getLogger("smart_grid.models.lstm")


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЙ БЛОК: TEMPORAL ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAttentionBlock(tf.keras.layers.Layer):
    """
    Temporal Self-Attention поверх LSTM hidden states.

    Query  = последний скрытый вектор LSTM (актуальное состояние t)
    Key/Value = все T скрытых векторов (память о шагах t-T .. t-1)

    Механизм: «на какие шаги прошлого нужно обратить внимание,
    чтобы предсказать следующие 24ч исходя из текущего состояния?»

    Атрибут `_attn_weights` сохраняется для последующей визуализации.
    """

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int = 16,
        dropout: float = 0.10,
        **kwargs,
    ) -> None:
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
        self.ln   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(
        self,
        hidden_states: tf.Tensor,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """
        Parameters
        ----------
        hidden_states : (B, T, d) — выходы LSTM для всех T шагов

        Returns
        -------
        context : (B, d) — взвешенный контекст истории
        """
        query = hidden_states[:, -1:, :]  # (B, 1, d)

        attn_out, self._attn_weights = self.mha(
            query=query,
            key=hidden_states,
            value=hidden_states,
            return_attention_scores=True,
            training=training,
        )  # (B, 1, d)

        # Важно: query имеет длину 1, поэтому attn_out.shape == (B, 1, d).
        # Ранее тут использовалось сложение с hidden_states (B, T, d), что
        # приводило к неявному broadcasting и «размыванию» внимания по всем T.
        # Для прогноза по последнему состоянию берём именно целевой контекст.
        context = tf.squeeze(attn_out, axis=1)  # (B, d)
        context = self.drop(self.ln(context), training=training)
        return context

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "dropout":   self.dropout_rate,
        }


# ══════════════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ МОДЕЛЬ
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(
    history_length: int = 48,
    forecast_horizon: int = 24,
    n_features: int = 11,
    lstm_units_1: int = 256,
    lstm_units_2: int = 128,
    lstm_units_3: int = 64,
    dropout_rate: float = 0.20,
    learning_rate: float = 3e-4,
    attn_heads: int = 4,
    use_cosine_decay: bool = True,
    total_steps: int = 37_000,
    mc_dropout: bool = False,
) -> tf.keras.Model:
    """
    AttentionLSTM v4: LSTM + Temporal Self-Attention.

    ОТЛИЧИЕ от v3:
      v3: LSTM(last_hidden) → Dense → forecast
      v4: LSTM(all_hiddens) → MHA(Q=last, K=V=all) → [last+context] → Dense

    Parameters
    ----------
    lstm_units_1/2/3 : размеры LSTM-слоёв.
                       optimal: 256/128/64  |  fast: 128/64/32
    attn_heads       : число голов Attention. key_dim = lstm_units_3 // attn_heads
    use_cosine_decay : CosineDecay LR schedule [Loshchilov & Hutter, 2016]
    total_steps      : суммарных шагов обучения ≈ EPOCHS × (n_train // batch).
                       Пример: 200 × (6061 // 32) ≈ 37 900. EarlyStopping
                       остановит раньше — schedule безопасно обрывается.
    """
    training_flag: Optional[bool] = True if mc_dropout else None

    # ── Learning Rate Schedule ─────────────────────────────────────────────────
    if use_cosine_decay:
        warmup = max(total_steps // 20, 200)
        lr_schedule: object = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=total_steps,
            alpha=5e-7,
            warmup_target=learning_rate,
            warmup_steps=warmup,
        )
    else:
        lr_schedule = learning_rate

    inp = tf.keras.Input(shape=(history_length, n_features), name="input_sequence")

    # ── Input Projection ───────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(
        lstm_units_1,
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="input_proj",
    )(inp)

    # ── LSTM Блок 1 ────────────────────────────────────────────────────────────
    effective_dropout = min(dropout_rate, 0.15)

    x = tf.keras.layers.LSTM(
        lstm_units_1, return_sequences=True,
        recurrent_dropout=0.0,
        name="lstm_1",
    )(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x)
    x = tf.keras.layers.Dropout(effective_dropout, name="drop_1")(x, training=training_flag)

    # ── LSTM Блок 2 ────────────────────────────────────────────────────────────
    x = tf.keras.layers.LSTM(
        lstm_units_2, return_sequences=True,
        recurrent_dropout=0.0,
        name="lstm_2",
    )(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_2")(x)
    x = tf.keras.layers.Dropout(effective_dropout, name="drop_2")(x, training=training_flag)

    # ── LSTM Блок 3 (return_sequences=True — нужны ВСЕ hidden для Attention) ──
    x = tf.keras.layers.LSTM(
        lstm_units_3, return_sequences=True,   # ← КЛЮЧЕВОЕ ОТЛИЧИЕ от v3
        recurrent_dropout=0.0,
        name="lstm_3",
    )(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="ln_3")(x)
    # x: (B, history_length, lstm_units_3)

    # ── Temporal Attention Block ───────────────────────────────────────────────
    key_dim = max(lstm_units_3 // attn_heads, 8)
    attn_block = TemporalAttentionBlock(
        num_heads=attn_heads,
        key_dim=key_dim,
        dropout=dropout_rate * 0.5,
        name="temporal_attn",
    )
    context = attn_block(x, training=training_flag)   # (B, lstm_units_3)

    # Persistence-бейзлайн: последний и предыдущий 24-часовые профили.
    cons_hist = inp[:, :, 0]
    recent_profile = cons_hist[:, -forecast_horizon:]
    if history_length >= 2 * forecast_horizon:
        prev_day_profile = cons_hist[:, -2 * forecast_horizon:-forecast_horizon]
    else:
        prev_day_profile = recent_profile

    recent_expanded = tf.keras.layers.Reshape((forecast_horizon, 1), name="recent_expand")(recent_profile)
    prev_expanded = tf.keras.layers.Reshape((forecast_horizon, 1), name="prev_expand")(prev_day_profile)
    pers_stack = tf.keras.layers.Concatenate(axis=-1, name="pers_stack")([
        recent_expanded,
        prev_expanded,
    ])
    persistence = tf.keras.layers.Dense(1, use_bias=True, name="persistence_blend")(pers_stack)
    persistence = tf.keras.layers.Flatten(name="persistence_flat")(persistence)

    last_token = x[:, -1, :]   # (B, lstm_units_3)
    global_token = tf.keras.layers.GlobalAveragePooling1D(name="global_avg")(x)
    last_token = tf.keras.layers.Dropout(
        effective_dropout, name="drop_last"
    )(last_token, training=training_flag)

    # Конкатенация: актуальное состояние + контекст внимания
    agg = tf.keras.layers.Concatenate(name="agg")([last_token, context, global_token])
    # agg: (B, lstm_units_3 * 3)

    # ── Dense-голова с GELU + Residual ─────────────────────────────────────────
    head_dim = lstm_units_3 * 5

    h = tf.keras.layers.Dense(
        head_dim, activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="head_d1",
    )(agg)
    skip = tf.keras.layers.Dense(head_dim, use_bias=False, name="head_skip")(agg)

    h = tf.keras.layers.Add(name="head_res")([h, skip])
    h = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="head_ln")(h)
    h = tf.keras.layers.Dropout(0.15, name="head_drop1")(h, training=training_flag)

    h = tf.keras.layers.Dense(
        head_dim // 2, activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="head_d2",
    )(h)
    h = tf.keras.layers.Dropout(0.10, name="head_drop2")(h, training=training_flag)

    residual = tf.keras.layers.Dense(forecast_horizon, name="residual_head")(h)
    out = tf.keras.layers.Add(name="output")([persistence, residual])

    model = tf.keras.Model(inputs=inp, outputs=out, name="AttentionLSTM_v4")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.Huber(delta=0.1),
        metrics=["mae", "mape"],
    )

    n_params = model.count_params()
    logger.info(
        "AttentionLSTM v4 | %d параметров | input=(%d,%d) | "
        "LSTM=%d/%d/%d | Attn heads=%d key_dim=%d | "
        "lr=%.0e%s | Huber δ=0.1",
        n_params, history_length, n_features,
        lstm_units_1, lstm_units_2, lstm_units_3,
        attn_heads, key_dim,
        learning_rate, " +CosineDecay" if use_cosine_decay else "",
    )
    return model
