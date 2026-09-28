# -*- coding: utf-8 -*-
"""Протокол ECL: разбиение, нормировка по train, линейные модели в замкнутой форме."""

import numpy as np
import pandas as pd
import pytest

from experiments.ecl_protocol import (
    _features, _windows, evaluate, fit_linear, moving_average, split_bounds,
)


def test_split_follows_the_autoformer_convention():
    b = split_bounds(1000, lookback=96)
    assert b["train"] == (0, 700)
    assert b["val"] == (700 - 96, 800)
    assert b["test"] == (800 - 96, 1000)


def test_moving_average_of_a_constant_is_the_constant():
    X = np.full((3, 50), 7.0)
    np.testing.assert_allclose(moving_average(X), X)


def test_moving_average_is_centred():
    X = np.arange(60, dtype=float)[None, :]
    np.testing.assert_allclose(moving_average(X, kernel=5)[0, 10:50], X[0, 10:50])


def test_linear_model_recovers_an_exact_linear_rule():
    """Если будущее — линейная функция прошлого, МНК находит её точно."""
    rng = np.random.RandomState(0)
    series = []
    for _ in range(3):
        x = np.zeros(400)
        x[:24] = rng.normal(size=24)
        for t in range(24, 400):
            x[t] = 0.6 * x[t - 24] + 0.3 * x[t - 1] + 0.01 * rng.normal()
        series.append(x)
    W = fit_linear(series, lookback=48, horizon=1, kind="linear", ridge=1e-9)
    X, Y = _windows(series[0], 48, 1)
    pred = _features(X, "linear") @ W
    assert np.mean((pred - Y) ** 2) < 1e-3


def _toy_frame(days=120, clients=3, seed=0):
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2012-01-01", periods=days * 24, freq="h")
    h = ts.hour.to_numpy()
    cols = {}
    for c in range(clients):
        daily = 1 + 0.5 * np.sin(2 * np.pi * (h - 6 - c) / 24)
        cols[f"MT_{c:03d}"] = (100 * (c + 1) * daily
                               * (1 + 0.05 * rng.normal(size=len(ts))))
    return pd.DataFrame(cols, index=ts)


def test_linear_models_beat_repeating_the_last_value():
    table = evaluate(_toy_frame(), lookback=96, horizons=(24, 48))
    ours = table[table["lookback"].notna()].pivot_table(index="model", columns="horizon",
                                                        values="MSE")
    for h in (24, 48):
        assert ours.loc["DLinear", h] < ours.loc["Повтор последнего значения", h]
        assert ours.loc["Linear (канально-независимая)", h] < \
            ours.loc["Повтор последнего значения", h]


def test_normalisation_uses_the_training_part_only():
    """Всплеск в тесте не меняет статистики нормировки."""
    from experiments.ecl_protocol import normalise

    values = _toy_frame().to_numpy()
    bounds = split_bounds(len(values), 96)
    spiked = values.copy()
    spiked[-100:] *= 50
    _, m1, s1 = normalise(values, bounds)
    _, m2, s2 = normalise(spiked, bounds)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(s1, s2)


def test_published_rows_are_marked_as_external():
    table = evaluate(_toy_frame(days=60), lookback=48, horizons=(96,))
    published = table[table["lookback"].isna()]
    assert set(published["horizon"]) == {96}
    assert (published["clients"] == 321).all()
