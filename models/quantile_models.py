# -*- coding: utf-8 -*-
"""
models/quantile_models.py — вероятностный прогноз квантилями.

ПОЧЕМУ ЭТО НЕ «ЕЩЁ ОДНА МОДЕЛЬ»
───────────────────────────────
Точечный прогноз отвечает, сколько будет; решение о разряде накопителя требует
ответа, насколько велик риск, что окажется больше. Плата за мощность —
статистика максимума, и недооценка пика стоит несопоставимо дороже переоценки.
В этом проекте измерено: ошибка точечного прогноза уничтожает 85% ценности
сглаживания пика, а с наивным прогнозом накопитель приносит убыток.

Средний прогноз оптимален для квадратичной ошибки, медианный — для абсолютной.
Для несимметричной стоимости оптимален соответствующий КВАНТИЛЬ, и получить
его усреднением нельзя: нужна другая функция потерь при обучении.

ДВА СПОСОБА, И ОБА НУЖНЫ
────────────────────────
Градиентный бустинг обучает уровни родной функцией `reg:quantileerror`, одна
модель на все уровни сразу. DLinear получает выход формы (горизонт, уровни) и
общую pinball-функцию потерь: уровни делят все веса, кроме последней проекции,
поэтому согласованы по построению и пересекаются реже независимых моделей.

Оба сравниваются с тривиальным вероятностным базлайном — квантилями остатков
сезонно-наивного прогноза. Без него нельзя утверждать, что вероятностная
модель вообще что-то добавляет: интервал, построенный по историческому
разбросу, уже даёт разумное покрытие.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf

logger = logging.getLogger("smart_grid.models.quantile_models")

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ПОТЕРЬ
# ══════════════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="smartgrid")
class PinballLoss(tf.keras.losses.Loss):
    """
    Средний pinball по набору уровней.

    Ожидает y_true формы (B, H) и y_pred формы (B, H, Q). Разворот целевой
    переменной по последней оси обязателен: без него сравнение молча пошло бы
    по правилам broadcast и дало бы бессмысленную величину, не падая.
    """

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES, **kwargs):
        super().__init__(**kwargs)
        self.quantiles = [float(q) for q in quantiles]

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if len(y_true.shape) == len(y_pred.shape) - 1:
            y_true = tf.expand_dims(y_true, axis=-1)

        levels = tf.constant(self.quantiles, dtype=y_pred.dtype)
        delta = y_true - y_pred
        loss = tf.maximum(levels * delta, (levels - 1.0) * delta)
        return tf.reduce_mean(loss, axis=[-2, -1])

    def get_config(self):
        config = super().get_config()
        config["quantiles"] = self.quantiles
        return config


# ══════════════════════════════════════════════════════════════════════════════
# ВЕРОЯТНОСТНЫЙ БАЗЛАЙН
# ══════════════════════════════════════════════════════════════════════════════

class QuantileNaive:
    """
    Сезонно-наивный прогноз плюс квантили его собственных остатков на обучении.

    Тривиальный, но обязательный базлайн: интервал, построенный по
    историческому разбросу ошибок, уже даёт разумное покрытие, и без него
    нельзя утверждать, что обучаемая вероятностная модель что-то добавляет.

    Остатки считаются отдельно по шагам горизонта. Измерено, что для этого
    базлайна разброс по шагам практически постоянен — отношение крайних ширин
    1.02: сезонно-наивный прогноз повторяет сутки назад, поэтому его ошибка не
    накапливается с удалением горизонта, в отличие от ошибки обучаемой модели.
    Разделение по шагам сохранено как более общее: оно ничего не стоит и станет
    существенным, если базой станет прогноз с растущей ошибкой.
    """

    name = "QuantileNaive"

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES):
        self.quantiles = [float(q) for q in quantiles]
        self.offsets: Optional[np.ndarray] = None      # (horizon, n_quantiles)
        self.horizon: Optional[int] = None

    def fit(self, data: Dict[str, Any], **_) -> "QuantileNaive":
        from models.panel_models import consumption_index

        channel = consumption_index(data["feature_names_hist"])
        horizon = int(data["Y_train"].shape[1])
        self.horizon = horizon

        base = data["X_hist_train"][:, -horizon:, channel]
        residuals = data["Y_train"] - base

        self.offsets = np.stack(
            [np.quantile(residuals, q, axis=0) for q in self.quantiles], axis=-1
        ).astype(np.float32)
        logger.info("%s: квантили остатков построены по %d окнам обучения",
                    self.name, len(residuals))
        return self

    def predict(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        from models.panel_models import consumption_index

        assert self.offsets is not None, "модель не обучена"
        channel = consumption_index(batch.get("feature_names_hist"))
        base = batch["hist"][:, -self.horizon:, channel]
        return (base[:, :, None] + self.offsets[None, :, :]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# ГРАДИЕНТНЫЙ БУСТИНГ
# ══════════════════════════════════════════════════════════════════════════════

class PanelQuantileXGBoost:
    """
    Бустинг с родной квантильной функцией потерь, по модели на шаг горизонта.

    Все уровни обучаются одной моделью через `quantile_alpha`: раздельные
    модели дали бы втрое больше обучения и чаще пересекающиеся квантили.
    """

    name = "QuantileXGBoost"

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES,
                 n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.06, seed: int = 42):
        self.quantiles = [float(q) for q in quantiles]
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           learning_rate=learning_rate, seed=seed)
        self.models: List[Any] = []

    def _flat(self, data: Dict[str, Any], split: str) -> np.ndarray:
        from models.panel_models import flatten_panel_inputs
        batch = {"hist": data[f"X_hist_{split}"], "future": data[f"X_future_{split}"],
                 "static": data[f"X_static_{split}"],
                 "feature_names_hist": data["feature_names_hist"],
                 "feature_names_future": data["feature_names_future"]}
        return flatten_panel_inputs(batch, aggregate_history=True)

    def fit(self, data: Dict[str, Any], **_) -> "PanelQuantileXGBoost":
        import xgboost as xgb

        X, Y = self._flat(data, "train"), data["Y_train"]
        X_val, Y_val = self._flat(data, "val"), data["Y_val"]

        self.models = []
        for h in range(Y.shape[1]):
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=np.array(self.quantiles),
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
                random_state=self.params["seed"] + h, n_jobs=4, verbosity=0,
                tree_method="hist", early_stopping_rounds=30,
            )
            model.fit(X, Y[:, h], eval_set=[(X_val, Y_val[:, h])], verbose=False)
            self.models.append(model)

        logger.info("%s обучен | признаков=%d | шагов=%d | уровней=%d",
                    self.name, X.shape[1], len(self.models), len(self.quantiles))
        return self

    def predict(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        from models.panel_models import flatten_panel_inputs

        X = flatten_panel_inputs(batch, aggregate_history=True)
        steps = [np.asarray(m.predict(X), dtype=np.float32) for m in self.models]
        # (H, B, Q) → (B, H, Q)
        return np.stack(steps, axis=0).transpose(1, 0, 2)


# ══════════════════════════════════════════════════════════════════════════════
# DLINEAR С КВАНТИЛЬНЫМ ВЫХОДОМ
# ══════════════════════════════════════════════════════════════════════════════

def build_quantile_dlinear(
    history_length: int,
    forecast_horizon: int,
    n_hist_features: int,
    n_future_features: int = 0,
    n_static_features: int = 0,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    kernel_size: int = 25,
    consumption_channel: int = 0,
    covariate_units: int = 64,
    dropout: float = 0.05,
    learning_rate: float = 1e-3,
    lr_schedule_total_steps: Optional[int] = None,
    lr_warmup_fraction: float = 0.05,
) -> tf.keras.Model:
    """
    DLinear, предсказывающий несколько квантилей одним проходом.

    Архитектура повторяет точечную версию, меняется только последняя проекция:
    выход (горизонт × уровни) вместо (горизонт), и pinball вместо Huber.
    Уровни делят все веса, кроме этой проекции, поэтому согласованы по
    построению и пересекаются заметно реже, чем независимо обученные модели.
    """
    from models.dlinear import SeriesDecomposition
    from models.transformer import _resolve_lr

    levels = [float(q) for q in quantiles]
    n_q = len(levels)

    hist_in = tf.keras.Input(shape=(history_length, n_hist_features), name="hist_input")
    inputs = [hist_in]

    target = hist_in[:, :, consumption_channel:consumption_channel + 1]
    trend, seasonal = SeriesDecomposition(kernel_size, name="decomposition")(target)

    width = forecast_horizon * n_q
    trend_proj = tf.keras.layers.Dense(width, name="linear_trend")(
        tf.keras.layers.Flatten(name="trend_flat")(trend))
    seasonal_proj = tf.keras.layers.Dense(width, name="linear_seasonal")(
        tf.keras.layers.Flatten(name="seasonal_flat")(seasonal))
    output = tf.keras.layers.Add(name="dlinear_core")([trend_proj, seasonal_proj])

    if n_hist_features > 1:
        hidden = tf.keras.layers.Dense(covariate_units, activation="gelu",
                                       name="hist_cov_hidden")(
            tf.keras.layers.Flatten(name="hist_flat")(hist_in))
        hidden = tf.keras.layers.Dropout(dropout, name="hist_cov_drop")(hidden)
        output = tf.keras.layers.Add(name="add_hist_cov")([
            output, tf.keras.layers.Dense(width, name="hist_cov_proj")(hidden)])

    if n_future_features > 0:
        fut_in = tf.keras.Input(shape=(forecast_horizon, n_future_features),
                                name="future_input")
        inputs.append(fut_in)
        hidden = tf.keras.layers.Dense(covariate_units, activation="gelu",
                                       name="future_hidden")(
            tf.keras.layers.Flatten(name="future_flat")(fut_in))
        hidden = tf.keras.layers.Dropout(dropout, name="future_drop")(hidden)
        output = tf.keras.layers.Add(name="add_future")([
            output, tf.keras.layers.Dense(width, name="future_proj")(hidden)])

    if n_static_features > 0:
        static_in = tf.keras.Input(shape=(n_static_features,), name="static_input")
        inputs.append(static_in)
        hidden = tf.keras.layers.Dense(covariate_units // 2, activation="gelu",
                                       name="static_hidden")(static_in)
        scale = tf.keras.layers.Dense(width, activation="tanh",
                                      name="static_scale")(hidden)
        output = tf.keras.layers.Add(name="add_static")([
            output, tf.keras.layers.Multiply(name="static_gate")([output, scale])])

    output = tf.keras.layers.Reshape((forecast_horizon, n_q),
                                     name="quantile_output")(output)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="QuantileDLinear")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=_resolve_lr(learning_rate, lr_schedule_total_steps,
                                      lr_warmup_fraction),
            clipnorm=1.0),
        loss=PinballLoss(levels),
    )

    logger.info("QuantileDLinear | история=%d горизонт=%d уровней=%d | параметров=%d",
                history_length, forecast_horizon, n_q, model.count_params())
    return model


# ══════════════════════════════════════════════════════════════════════════════
# ОЦЕНКА НА ПАНЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_panel_quantiles(model: Any, data: Dict[str, Any], split: str = "test",
                             quantiles: Sequence[float] = DEFAULT_QUANTILES,
                             ) -> Dict[str, float]:
    """
    Оценивает вероятностный прогноз на панели в ИСХОДНОМ масштабе каждого ряда.

    Обратное преобразование применяется к каждому уровню отдельно: нормировка
    своя у каждого ряда, и оценивать pinball в нормированном пространстве
    бессмысленно — вклад мелкого фидера оказался бы равен вкладу крупного при
    несопоставимых абсолютных ошибках.

    Порядок уровней восстанавливается перед оценкой, а доля исходных пересечений
    сохраняется в отчёте: сортировка не ухудшает pinball ни на одном уровне, но
    частые пересечения означают, что уровни обучены несогласованно, и это надо
    видеть.
    """
    import tensorflow as tf

    from data.panel_preprocessing import inverse_scale_series
    from models.panel_trainer import make_batch
    from utils.quantile_metrics import enforce_monotone, evaluate_quantiles

    batch = make_batch(data, split)
    series = data[f"series_{split}"]

    if isinstance(model, tf.keras.Model):
        names = [inp.name.split(":")[0] for inp in model.inputs]
        mapping = {"hist_input": batch["hist"], "future_input": batch["future"],
                   "static_input": batch["static"]}
        inputs = [mapping[n] for n in names if n in mapping] or [batch["hist"]]
        raw = model.predict(inputs, verbose=0)
    else:
        raw = model.predict(batch)

    levels = [float(q) for q in quantiles]
    scaled = {q: raw[:, :, i] for i, q in enumerate(levels)}
    crossing_before = None

    from utils.quantile_metrics import crossing_rate
    crossing_before = crossing_rate(scaled)
    scaled = enforce_monotone(scaled)

    original = {q: inverse_scale_series(data, v, series) for q, v in scaled.items()}
    y_true = inverse_scale_series(data, data[f"Y_{split}"], series)

    result = evaluate_quantiles(y_true, original, interval=(min(levels), max(levels)))
    result["crossing_rate_before_sort"] = float(crossing_before)
    result["model"] = getattr(model, "name", type(model).__name__)
    return result
