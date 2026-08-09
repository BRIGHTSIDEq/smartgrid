# -*- coding: utf-8 -*-
"""
Тесты panel-моделей: контракт схемы признаков, единообразие входов,
сериализация DLinear и согласованность иерархии.

Отдельный акцент на канал потребления. Модели обращаются к целевому ряду по
индексу, и подмена канала не нарушает ни одной проверки форм: обнаружить её
можно только по неправдоподобным метрикам. В ранней версии список признаков
сортировался по алфавиту, нулевым каналом оказывалась облачность, и наивный
прогноз повторял погоду вместо нагрузки — MAE выходил в пятнадцать раз хуже
климатологии.
"""

import numpy as np
import pytest
import tensorflow as tf

from data.panel import generate_panel_data
from data.panel_preprocessing import prepare_panel_data
from models.dlinear import build_dlinear, SeriesDecomposition
from models.panel_models import (
    PanelNaive24, PanelHourlyProfile, build_panel_ridge,
    consumption_index, flatten_panel_inputs,
)
from models.panel_trainer import PanelTrainer, make_batch, bottom_up_city_forecast


@pytest.fixture(scope="module")
def panel_data():
    df, specs = generate_panel_data(days=70, n_cities=1, feeders_per_city=3, seed=4)
    return prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=4)


# ══════════════════════════════════════════════════════════════════════════════
# КОНТРАКТ СХЕМЫ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════

def test_consumption_is_channel_zero(panel_data):
    """Потребление занимает нулевой канал — на это опираются модели."""
    assert panel_data["feature_names_hist"][0] == "consumption"
    assert panel_data["consumption_channel"] == 0


def test_consumption_index_resolves_by_name():
    """Индекс канала разрешается по имени, а не берётся вслепую."""
    names = ["cloud_cover", "consumption", "temperature"]
    assert consumption_index(names) == 1
    assert consumption_index(["consumption", "x"]) == 0
    assert consumption_index(None) == 0        # запасной путь


def test_naive_predicts_consumption_not_another_channel(panel_data):
    """
    Наивный прогноз повторяет именно потребление.

    Проверка построена на сравнении с фактическим срезом канала: если модель
    возьмёт соседний признак, значения разойдутся, хотя форма выхода останется
    правильной.
    """
    model = PanelNaive24().fit(panel_data)
    batch = make_batch(panel_data, "test")
    pred = model.predict(batch)

    ch = consumption_index(panel_data["feature_names_hist"])
    expected = batch["hist"][:, -24:, ch]

    assert pred.shape == panel_data["Y_test"].shape
    assert np.allclose(pred, expected)


def test_naive_is_a_reasonable_baseline(panel_data):
    """
    Наивный прогноз должен быть осмысленным, а не абсурдным.

    Регрессионная проверка на тот самый дефект: при работе не с тем каналом
    R² уходил в минус пять.
    """
    from utils.metrics import r2_score

    model = PanelNaive24().fit(panel_data)
    pred = model.predict(make_batch(panel_data, "test"))
    assert r2_score(panel_data["Y_test"], pred) > 0.5, (
        "наивный прогноз неправдоподобно плох — вероятно, используется не тот канал"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ЕДИНООБРАЗИЕ ВХОДОВ
# ══════════════════════════════════════════════════════════════════════════════

def test_flatten_includes_future_and_static(panel_data):
    """
    Классические модели получают те же признаки, что и нейросети.

    Иначе разрыв в метриках отражал бы разницу в объёме входной информации,
    а не в способностях моделей.
    """
    batch = make_batch(panel_data, "train")
    flat_full = flatten_panel_inputs(batch, aggregate_history=False)

    n_hist = batch["hist"].shape[1] * batch["hist"].shape[2]
    n_future = batch["future"].shape[1] * batch["future"].shape[2]
    n_static = batch["static"].shape[1]

    assert flat_full.shape == (len(batch["hist"]), n_hist + n_future + n_static)
    assert n_future > 0 and n_static > 0

    flat_agg = flatten_panel_inputs(batch, aggregate_history=True)
    assert flat_agg.shape[1] < flat_full.shape[1], "агрегация должна сжимать историю"
    assert flat_agg.shape[1] > n_future + n_static, "история не должна исчезнуть"


def test_panel_ridge_trains_and_predicts(panel_data):
    model = build_panel_ridge(alphas=[0.1, 1.0, 10.0])
    trainer = PanelTrainer(model, "Ridge").train(panel_data)
    pred = trainer.predict(panel_data, "test")

    assert pred.shape == panel_data["Y_test"].shape
    assert np.isfinite(pred).all()


def test_hourly_profile_is_per_series(panel_data):
    """Профиль строится отдельно для каждого ряда, а не общий на панель."""
    model = PanelHourlyProfile().fit(panel_data)

    assert model.profiles.shape == (len(panel_data["series_index"]), 7, 24)
    # Профили разных рядов не должны совпадать: фидеры имеют разную форму.
    assert not np.allclose(model.profiles[0], model.profiles[1])


# ══════════════════════════════════════════════════════════════════════════════
# DLINEAR
# ══════════════════════════════════════════════════════════════════════════════

def test_series_decomposition_splits_trend_and_residual():
    """Тренд плюс остаток восстанавливают исходный ряд в точности."""
    layer = SeriesDecomposition(kernel_size=25)
    x = tf.constant(np.random.RandomState(0).normal(size=(4, 96, 1)).astype("float32"))
    trend, seasonal = layer(x)

    assert trend.shape == x.shape
    assert np.allclose((trend + seasonal).numpy(), x.numpy(), atol=1e-5)

    # Тренд обязан быть глаже исходного ряда: скользящее среднее подавляет
    # высокочастотную составляющую, которая целиком уходит в остаток.
    roughness_trend = float(np.std(np.diff(trend.numpy(), axis=1)))
    roughness_input = float(np.std(np.diff(x.numpy(), axis=1)))
    assert roughness_trend < roughness_input


def test_dlinear_core_is_small():
    """
    Без ковариат DLinear остаётся крошечной моделью.

    В этом и смысл базлайна: если тысячи параметров хватает, сложная
    архитектура обязана объяснить свою стоимость.
    """
    model = build_dlinear(history_length=48, forecast_horizon=24, n_hist_features=1)
    assert model.count_params() < 5_000


def test_dlinear_accepts_three_inputs(panel_data):
    nH = len(panel_data["feature_names_hist"])
    nF = len(panel_data["feature_names_future"])
    nS = len(panel_data["static_names"])
    model = build_dlinear(48, 24, nH, nF, nS)

    batch = make_batch(panel_data, "test")
    out = model.predict([batch["hist"][:16], batch["future"][:16], batch["static"][:16]],
                        verbose=0)
    assert out.shape == (16, 24)
    assert np.isfinite(out).all()


def test_dlinear_roundtrip(tmp_path, panel_data):
    """Модель сохраняется и восстанавливается с тем же прогнозом."""
    nH = len(panel_data["feature_names_hist"])
    nF = len(panel_data["feature_names_future"])
    nS = len(panel_data["static_names"])
    model = build_dlinear(48, 24, nH, nF, nS)

    batch = make_batch(panel_data, "test")
    inputs = [batch["hist"][:8], batch["future"][:8], batch["static"][:8]]
    before = model.predict(inputs, verbose=0)

    path = str(tmp_path / "dlinear.keras")
    model.save(path)
    restored = tf.keras.models.load_model(
        path, custom_objects={"SeriesDecomposition": SeriesDecomposition},
        compile=False)
    after = restored.predict(inputs, verbose=0)

    assert np.allclose(before, after, rtol=1e-5, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# ИЕРАРХИЯ
# ══════════════════════════════════════════════════════════════════════════════

def test_bottom_up_city_forecast_is_coherent(panel_data):
    """
    Городской прогноз получается суммированием фидеров.

    Относительная ошибка на уровне города должна быть НЕ ХУЖЕ, чем на уровне
    фидера: при суммировании независимые составляющие ошибки частично гасят
    друг друга. Обратное означало бы ошибку агрегации.
    """
    from utils.metrics import mean_absolute_percentage_error
    from data.panel_preprocessing import inverse_scale_series

    model = PanelNaive24().fit(panel_data)
    pred = model.predict(make_batch(panel_data, "test"))

    result = bottom_up_city_forecast(panel_data, pred, "test")
    assert result["per_city"], "не построен ни один городской прогноз"

    series = panel_data["series_test"]
    feeder_mape = mean_absolute_percentage_error(
        inverse_scale_series(panel_data, panel_data["Y_test"], series),
        inverse_scale_series(panel_data, pred, series),
    )
    assert result["MAPE_mean"] <= feeder_mape + 1e-6, (
        "агрегация не может ухудшать относительную ошибку"
    )


def test_panel_trainer_reports_macro_and_worst(panel_data):
    """Оценка даёт micro, macro и худший ряд — усреднённое MAE их скрывает."""
    trainer = PanelTrainer(PanelNaive24(), "Naive24").train(panel_data)
    m = trainer.evaluate(panel_data, "test")

    for key in ("MAE", "MAE_macro", "MAE_worst_series", "worst_series", "per_series_MAE"):
        assert key in m
    assert m["MAE_worst_series"] >= m["MAE_macro"], "худший ряд не может быть лучше среднего"
    assert len(m["per_series_MAE"]) == len(panel_data["series_index"])
