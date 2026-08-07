# -*- coding: utf-8 -*-
"""
Тесты метрик качества прогноза.

Значения подобраны так, чтобы результат можно было проверить вручную —
метрики не должны «проверяться» тем же кодом, который их считает.
"""

import numpy as np
import pytest

from utils.metrics import (
    mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error, symmetric_mape, r2_score,
    seasonal_naive_scale, mase, compute_all_metrics,
    metrics_by_horizon, diebold_mariano, pairwise_dm_table,
)


def test_mae_known_value():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # Ошибки: 2, 2, 3 → среднее 7/3
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(7.0 / 3.0)


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 4.0, 0.0])
    # Квадраты: 9, 16, 0 → среднее 25/3 → корень
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(25.0 / 3.0))


def test_mape_known_value():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    # Ошибки 10%, 10% → 10%
    assert mean_absolute_percentage_error(y_true, y_pred) == pytest.approx(10.0, abs=1e-4)


def test_smape_is_bounded():
    """sMAPE не превышает 200% даже при прогнозе противоположного знака."""
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([-1.0, -1.0])
    assert symmetric_mape(y_true, y_pred) <= 200.0 + 1e-6


def test_r2_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0, abs=1e-6)
    # Прогноз средним даёт R² = 0
    assert r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-6)


# ── MASE ──────────────────────────────────────────────────────────────────────

def test_seasonal_naive_scale_on_periodic_series():
    """Для идеально периодического ряда знаменатель MASE равен нулю."""
    series = np.tile(np.arange(24, dtype=float), 10)
    assert seasonal_naive_scale(series, season_length=24) == pytest.approx(0.0)


def test_mase_equals_one_for_naive_forecast():
    """
    Модель, воспроизводящая сезонно-наивный прогноз, должна давать MASE ≈ 1.

    Это смысловой якорь метрики: единица — граница практической ценности.
    """
    rng = np.random.default_rng(0)
    train = rng.normal(1000, 100, size=24 * 30)
    scale = seasonal_naive_scale(train, season_length=24)

    test_true = rng.normal(1000, 100, size=24 * 10)
    test_naive = np.concatenate([train[-24:], test_true[:-24]])

    value = mase(test_true, test_naive, scale)
    assert 0.7 < value < 1.4, f"MASE наивного прогноза = {value}, ожидалось около 1"


def test_mase_below_one_for_better_model():
    rng = np.random.default_rng(11)
    # Ряд с суточной сезонностью и шумом: знаменатель MASE строго положителен.
    train = np.tile(np.arange(24, dtype=float) * 10 + 500, 20) + rng.normal(0, 50, 480)
    scale = seasonal_naive_scale(train, season_length=24)
    assert scale > 0

    y_true = np.array([100.0, 200.0, 300.0])
    almost_perfect = y_true + 0.01
    assert mase(y_true, almost_perfect, scale) < 1.0


def test_mase_is_nan_for_degenerate_scale():
    """Идеально периодический ряд даёт нулевой знаменатель — MASE не определён."""
    train = np.tile(np.arange(24, dtype=float), 10)
    scale = seasonal_naive_scale(train, season_length=24)
    assert scale == pytest.approx(0.0)
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), scale))


def test_compute_all_metrics_includes_mase_only_when_scale_given():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert "MASE" not in compute_all_metrics(y, y)
    assert "MASE" in compute_all_metrics(y, y, mase_scale=10.0)


# ── Метрики по горизонтам ─────────────────────────────────────────────────────

def test_metrics_by_horizon_detects_degradation():
    """Ошибка, растущая с шагом горизонта, должна отражаться в кривой MAE."""
    n, horizon = 50, 4
    y_true = np.zeros((n, horizon))
    y_pred = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (n, 1))

    res = metrics_by_horizon(y_true, y_pred)

    assert res["h"] == [1, 2, 3, 4]
    assert res["MAE"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert res["MAE"][-1] > res["MAE"][0]


# ── Тест Диболда–Мариано ──────────────────────────────────────────────────────

def test_dm_detects_clearly_better_model():
    """При явном превосходстве модели A статистика отрицательна и значима."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(0, 1, size=2000)
    good = y_true + rng.normal(0, 0.1, size=2000)
    bad = y_true + rng.normal(0, 1.0, size=2000)

    dm, p = diebold_mariano(y_true, good, bad, h=1)
    assert dm < 0, "DM должна быть отрицательной, когда точнее модель A"
    assert p < 0.01


def test_dm_no_difference_for_identical_forecasts():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    dm, p = diebold_mariano(y_true, pred, pred, h=1)
    assert np.isnan(dm) or abs(dm) < 1e-6


def test_pairwise_dm_table_structure():
    rng = np.random.default_rng(2)
    y_true = rng.normal(0, 1, size=500)
    preds = {
        "A": y_true + rng.normal(0, 0.1, size=500),
        "B": y_true + rng.normal(0, 0.5, size=500),
        "C": y_true + rng.normal(0, 1.0, size=500),
    }
    rows = pairwise_dm_table(y_true, preds, h=1)
    assert len(rows) == 3            # C(3,2) пар
    for r in rows:
        assert set(r) == {"model_a", "model_b", "DM", "p_value", "better"}
