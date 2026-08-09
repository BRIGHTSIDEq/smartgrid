# -*- coding: utf-8 -*-
"""
models/dlinear.py — DLinear: декомпозиция плюс линейные проекции.

ЗАЧЕМ ЭТА МОДЕЛЬ В РАБОТЕ
─────────────────────────
Zeng et al., «Are Transformers Effective for Time Series Forecasting?»
(AAAI 2023) показали, что простая линейная модель с декомпозицией на тренд и
остаток обходит трансформерные архитектуры на стандартных бенчмарках
долгосрочного прогнозирования. Наши собственные измерения дали тот же
результат независимо: Ridge и градиентный бустинг обошли PatchTST и LSTM.
DLinear превращает это совпадение из курьёза в воспроизведение известного
результата и даёт корректную точку отсчёта, против которой любая сложная
архитектура обязана себя оправдывать.

УСТРОЙСТВО
──────────
    trend    = скользящее среднее по окну истории
    seasonal = история − trend
    прогноз  = Linear_s(seasonal) + Linear_t(trend) + вклад ковариат

Обе проекции идут напрямую из длины истории в длину горизонта: ни рекуррентности,
ни внимания. Модель расширена признаками, которых нет в оригинальной статье, но
которые есть в задаче прогнозирования нагрузки:

  future  — точно известный календарь целевого окна и прогноз погоды;
  static  — постоянные характеристики фидера, нужные глобальной модели,
            обучаемой сразу на множестве рядов.

Без статических признаков глобальная модель усредняла бы разнородные объекты,
без будущего календаря — вынуждена была бы угадывать час и день недели
целевого окна по истории.
"""

import logging
from typing import Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger("smart_grid.models.dlinear")


@tf.keras.utils.register_keras_serializable(package="smartgrid")
class SeriesDecomposition(tf.keras.layers.Layer):
    """
    Разделяет ряд на тренд и остаток скользящим средним.

    Края окна дополняются краевыми значениями, а не нулями: дополнение нулями
    создало бы искусственный провал тренда на границах, который модель приняла
    бы за реальный сигнал.
    """

    def __init__(self, kernel_size: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)

    def call(self, x: tf.Tensor):
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left

        front = tf.repeat(x[:, :1, :], pad_left, axis=1)
        back = tf.repeat(x[:, -1:, :], pad_right, axis=1)
        padded = tf.concat([front, x, back], axis=1)

        trend = tf.nn.avg_pool1d(padded, ksize=self.kernel_size, strides=1,
                                 padding="VALID")
        return trend, x - trend

    def compute_output_shape(self, input_shape):
        return input_shape, input_shape

    def get_config(self) -> dict:
        return {**super().get_config(), "kernel_size": self.kernel_size}


def build_dlinear(
    history_length: int,
    forecast_horizon: int,
    n_hist_features: int,
    n_future_features: int = 0,
    n_static_features: int = 0,
    kernel_size: int = 25,
    consumption_channel: int = 0,
    covariate_units: int = 64,
    dropout: float = 0.05,
    learning_rate: float = 1e-3,
    huber_delta: float = 0.05,
    lr_schedule_total_steps: Optional[int] = None,
    lr_warmup_fraction: float = 0.05,
) -> tf.keras.Model:
    """
    Собирает DLinear с тремя входами.

    Parameters
    ----------
    consumption_channel : int
        Индекс канала потребления в историческом входе. Декомпозиция
        применяется именно к нему: тренд и сезонность прочих ковариат для
        прогноза целевой переменной значения не имеют.
    covariate_units : int
        Ширина скрытого слоя для ковариат. Держится небольшой намеренно:
        смысл модели в том, что основную работу делают линейные проекции ряда,
        а ковариаты вносят поправку.
    """
    from models.transformer import _resolve_lr

    hist_in = tf.keras.Input(shape=(history_length, n_hist_features), name="hist_input")
    inputs = [hist_in]

    # ── Ядро DLinear: декомпозиция канала потребления ───────────────────────
    target_series = hist_in[:, :, consumption_channel:consumption_channel + 1]
    trend, seasonal = SeriesDecomposition(kernel_size, name="decomposition")(target_series)

    trend_flat = tf.keras.layers.Flatten(name="trend_flat")(trend)
    seasonal_flat = tf.keras.layers.Flatten(name="seasonal_flat")(seasonal)

    trend_proj = tf.keras.layers.Dense(forecast_horizon, name="linear_trend")(trend_flat)
    seasonal_proj = tf.keras.layers.Dense(forecast_horizon,
                                          name="linear_seasonal")(seasonal_flat)
    output = tf.keras.layers.Add(name="dlinear_core")([trend_proj, seasonal_proj])

    # ── Поправка по остальным каналам истории ───────────────────────────────
    if n_hist_features > 1:
        hist_flat = tf.keras.layers.Flatten(name="hist_flat")(hist_in)
        hist_hidden = tf.keras.layers.Dense(covariate_units, activation="gelu",
                                            name="hist_cov_hidden")(hist_flat)
        hist_hidden = tf.keras.layers.Dropout(dropout, name="hist_cov_drop")(hist_hidden)
        output = tf.keras.layers.Add(name="add_hist_cov")([
            output,
            tf.keras.layers.Dense(forecast_horizon, name="hist_cov_proj")(hist_hidden),
        ])

    # ── Известные заранее признаки целевого окна ────────────────────────────
    if n_future_features > 0:
        fut_in = tf.keras.Input(shape=(forecast_horizon, n_future_features),
                                name="future_input")
        inputs.append(fut_in)
        fut_flat = tf.keras.layers.Flatten(name="future_flat")(fut_in)
        fut_hidden = tf.keras.layers.Dense(covariate_units, activation="gelu",
                                           name="future_hidden")(fut_flat)
        fut_hidden = tf.keras.layers.Dropout(dropout, name="future_drop")(fut_hidden)
        output = tf.keras.layers.Add(name="add_future")([
            output,
            tf.keras.layers.Dense(forecast_horizon, name="future_proj")(fut_hidden),
        ])

    # ── Статические характеристики ряда ─────────────────────────────────────
    if n_static_features > 0:
        static_in = tf.keras.Input(shape=(n_static_features,), name="static_input")
        inputs.append(static_in)
        static_hidden = tf.keras.layers.Dense(covariate_units // 2, activation="gelu",
                                              name="static_hidden")(static_in)
        # Статика задаёт масштабирующую поправку: у разных фидеров одинаковая
        # форма графика может соответствовать разному уровню нагрузки.
        scale = tf.keras.layers.Dense(forecast_horizon, activation="tanh",
                                      name="static_scale")(static_hidden)
        output = tf.keras.layers.Add(name="add_static")([
            output,
            tf.keras.layers.Multiply(name="static_gate")([output, scale]),
        ])

    model = tf.keras.Model(inputs=inputs, outputs=output, name="DLinear")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=_resolve_lr(learning_rate, lr_schedule_total_steps,
                                      lr_warmup_fraction),
            clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=huber_delta),
        metrics=["mae"],
    )

    logger.info(
        "DLinear | история=%d горизонт=%d | каналов: hist=%d future=%d static=%d | "
        "ядро свёртки=%d | параметров=%d",
        history_length, forecast_horizon, n_hist_features, n_future_features,
        n_static_features, kernel_size, model.count_params(),
    )
    return model
