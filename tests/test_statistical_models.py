# -*- coding: utf-8 -*-
"""ETS(A,N,A) и ансамбль с весами по валидации."""

import numpy as np
import pytest

from models.statistical import (
    HoltWintersAdditive, ValidationWeightedEnsemble, bates_granger_weights,
    holt_winters_forecast,
)


def _periodic(n_windows=5, T=48, level=10.0):
    pattern = np.sin(2 * np.pi * np.arange(24) / 24)
    series = level + np.tile(pattern, (T + 24) // 24 + 2)
    X = np.stack([series[i:i + T] for i in range(n_windows)])
    Y = np.stack([series[i + T:i + T + 24] for i in range(n_windows)])
    return X, Y


@pytest.mark.parametrize("alpha, gamma", [(0.1, 0.1), (0.5, 0.9), (0.9, 0.02)])
def test_pure_seasonal_series_is_forecast_exactly(alpha, gamma):
    """На чисто сезонном ряде без шума прогноз совпадает с продолжением ряда."""
    X, Y = _periodic()
    np.testing.assert_allclose(holt_winters_forecast(X, 24, alpha, gamma), Y, atol=1e-5)


def test_level_follows_a_step_with_alpha_one():
    """При α = 1 уровень целиком переносится на последнее наблюдение."""
    X, _ = _periodic(n_windows=1)
    X = X.copy()
    X[0, 24:] += 5.0            # скачок уровня во втором сезоне
    pred = holt_winters_forecast(X, 24, alpha=1.0, gamma=0.0)
    pattern = X[0, :24] - X[0, :24].mean()
    np.testing.assert_allclose(pred[0], X[0, :24].mean() + 5.0 + pattern, atol=1e-5)


def test_window_shorter_than_season_is_rejected():
    with pytest.raises(ValueError, match="короче сезона"):
        holt_winters_forecast(np.zeros((1, 12)), 24, 0.1, 0.1)


def test_parameters_are_chosen_on_validation():
    rng = np.random.RandomState(0)
    X, Y = _periodic(n_windows=40)
    Xn = X + rng.normal(0, 0.3, X.shape)
    model = HoltWintersAdditive().fit(Xn[:, :, None], Y, Xn[:, :, None], Y)
    assert model.alpha_ in HoltWintersAdditive.GRID and model.gamma_ in HoltWintersAdditive.GRID
    grid_maes = [np.mean(np.abs(holt_winters_forecast(Xn, 24, a, g) - Y))
                 for a in HoltWintersAdditive.GRID for g in HoltWintersAdditive.GRID]
    assert model.val_mae_ == pytest.approx(min(grid_maes))
    assert model.predict(Xn[:, :, None]).shape == Y.shape


def test_ets_refuses_to_fit_without_validation():
    X, Y = _periodic()
    with pytest.raises(ValueError, match="валидации"):
        HoltWintersAdditive().fit(X[:, :, None], Y)


def test_bates_granger_weights_are_inverse_mse():
    w = bates_granger_weights({"a": 1.0, "b": 4.0, "bad": float("nan")})
    assert w == pytest.approx({"a": 0.8, "b": 0.2})


class _Fixed:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full((len(X), 2), self.value)


def test_ensemble_is_the_weighted_average_of_members():
    X = np.zeros((3, 4, 1))
    Y = np.ones((3, 2))
    ens = ValidationWeightedEnsemble([("a", _Fixed(2.0)), ("b", _Fixed(3.0))])
    ens.fit(X, Y, X, Y)
    # MSE: a — 1, b — 4 → веса 0.8 и 0.2.
    np.testing.assert_allclose(ens.predict(X), 0.8 * 2.0 + 0.2 * 3.0)


def test_ensemble_is_not_eligible_for_selection():
    """Веса подобраны по валидации, поэтому в отбор лучшей модели ансамбль не входит."""
    import main as main_module
    assert "ensemble" in main_module.NOT_SELECTABLE_KEYS
    assert "ets" not in main_module.NOT_SELECTABLE_KEYS


def test_ensemble_runs_through_the_trainer():
    """Ансамбль из обученных ModelTrainer оценивается штатным путём."""
    from data.generator import generate_smartgrid_data
    from data.preprocessing import prepare_data
    from models.baseline import build_linear_regression
    from models.statistical import build_ets
    from models.trainer import ModelTrainer

    df = generate_smartgrid_data(days=25, households=30, seed=4,
                                 industrial_loads=2, city_districts=2)
    data = prepare_data(df, history_length=48, forecast_horizon=24)
    members = []
    for model, name in ((build_linear_regression(), "LinearRegression"),
                        (build_ets(), "ETS")):
        t = ModelTrainer(model, name)
        t.train(data)
        members.append((name, t))

    ens = ModelTrainer(ValidationWeightedEnsemble(members), "Ensemble")
    ens.train(data)
    metrics = ens.evaluate(data, split="test", run_residual_diagnostics=False)
    assert np.isfinite(metrics["MAE"])
    assert set(ens.model.weights_) == {"LinearRegression", "ETS"}
