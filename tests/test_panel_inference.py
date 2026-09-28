# -*- coding: utf-8 -*-
"""
Пакетный прогноз по сохранённой многорядной модели.

Эталон — прогноз той же модели по окну, собранному при обучении. На вход
инференса идут только сырые наблюдения и прогноз погоды; если признаки
построены иначе, чем при обучении, прогнозы разойдутся.
"""

import numpy as np
import pandas as pd
import pytest

from data.panel import generate_panel_data
from data.panel_preprocessing import prepare_panel_data
from models.panel_models import PanelNaive24, build_panel_ridge
from models.panel_trainer import PanelTrainer
from panel_pipeline import save_panel_models
from utils.panel_inference import check_issue_time, forecast_panel, load_panel_models

H = 48


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    df, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=6)
    data = prepare_panel_data(df, specs, history_length=H, forecast_horizon=24, seed=6)
    trainers = [PanelTrainer(PanelNaive24(), "Naive24").train(data),
                PanelTrainer(build_panel_ridge(alphas=[0.1, 1.0]), "Ridge").train(data)]
    out = tmp_path_factory.mktemp("models")
    save_panel_models(trainers, data, str(out), best_name="Ridge")
    df = df.assign(series=df["city_id"].astype(str) + "/" + df["feeder_id"].astype(str))
    return df, data, trainers, str(out)


def _window_inputs(df, data, row):
    """Сырые входы инференса для окна row тестового сплита."""
    sid = int(data["series_test"][row])
    key = data["series_index"][sid]
    stamps = pd.DatetimeIndex(pd.to_datetime(data["timestamps"]))
    start = int(data["anchor_test"][row]) + H
    target = stamps[start:start + 24]
    series = df[df["series"] == key].sort_values("timestamp")
    history = series[series["timestamp"] < target[0]].tail(H + 200)

    # Прогноз погоды, на котором обучалась модель: обратная нормировка каналов.
    names = data["feature_names_future"]
    weather = pd.DataFrame({"series": key, "timestamp": target})
    for col, sc in data["weather_scalers"].items():
        ch = f"fut_{col}_forecast"
        if ch in names:
            scaled = data["X_future_test"][row, :, names.index(ch)]
            weather[col] = sc.inverse_transform(scaled.reshape(-1, 1)).ravel()
    return key, history, weather


def test_saved_model_reproduces_the_training_window(trained):
    df, data, trainers, models_dir = trained
    bundle = load_panel_models(models_dir)
    assert bundle["name"] == "Ridge"

    row = int(np.where(data["series_test"] == 1)[0][30])
    key, history, weather = _window_inputs(df, data, row)
    result = forecast_panel(bundle, history, weather)

    from data.panel_preprocessing import inverse_scale_series
    ridge = next(t for t in trainers if t.name == "Ridge")
    expected = inverse_scale_series(data, ridge.predict(data, "test")[row:row + 1],
                                    data["series_test"][row:row + 1])[0]
    assert list(result["series"].unique()) == [key]
    np.testing.assert_allclose(result["forecast"].to_numpy(), expected, rtol=1e-4, atol=1e-3)


def test_unknown_series_is_rejected(trained):
    df, data, _, models_dir = trained
    bundle = load_panel_models(models_dir)
    row = int(np.where(data["series_test"] == 0)[0][5])
    _, history, weather = _window_inputs(df, data, row)
    with pytest.raises(ValueError, match="не встречался"):
        forecast_panel(bundle, history.assign(series="чужой/1"), weather.assign(series="чужой/1"))


def test_short_history_and_missing_weather_are_rejected(trained):
    df, data, _, models_dir = trained
    bundle = load_panel_models(models_dir)
    row = int(np.where(data["series_test"] == 0)[0][5])
    _, history, weather = _window_inputs(df, data, row)
    with pytest.raises(ValueError, match="истории"):
        forecast_panel(bundle, history.tail(100), weather)
    with pytest.raises(ValueError, match="прогноз погоды"):
        forecast_panel(bundle, history, weather.iloc[:10])


def test_issue_time_coverage():
    ts = pd.date_range("2025-04-10 11:00", periods=24, freq="h")
    fc = pd.DataFrame({"timestamp": ts, "forecast": 1.0})
    r = check_issue_time(fc, "2025-04-11")
    assert r["covered_share"] == pytest.approx(11 / 24)
    assert r["first_missing"] == "2025-04-11 11:00:00"


def test_quantile_model_is_saved_and_forecasts_ordered_levels(tmp_path):
    """
    Квантильный бустинг сохраняется вместе с точечными моделями, а прогноз по
    нему отдаёт упорядоченные P10 ≤ P50 ≤ P90 в масштабе каждого ряда.
    """
    from models.quantile_models import PanelQuantileXGBoost

    df, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=6)
    data = prepare_panel_data(df, specs, history_length=H, forecast_horizon=24, seed=6)
    point = [PanelTrainer(PanelNaive24(), "Naive24").train(data)]
    q = PanelQuantileXGBoost(n_estimators=5, seed=0)
    q.fit(data)
    manifest = save_panel_models(point, data, str(tmp_path), best_name="Naive24",
                                 quantile_models={"QuantileXGBoost": q})
    assert manifest["quantile_models"] == {"QuantileXGBoost": "quantile_QuantileXGBoost.pkl"}

    bundle = load_panel_models(str(tmp_path), quantile=True)
    df = df.assign(series=df["city_id"].astype(str) + "/" + df["feeder_id"].astype(str))
    row = int(np.where(data["series_test"] == 0)[0][10])
    _, history, weather = _window_inputs(df, data, row)
    result = forecast_panel(bundle, history, weather)
    assert {"p10", "p50", "p90"} <= set(result.columns)
    assert (result["p10"] <= result["p50"]).all() and (result["p50"] <= result["p90"]).all()
    assert result["p50"].between(0.3 * history["consumption"].min(),
                                 3 * history["consumption"].max()).all()


def test_missing_quantile_model_is_reported(trained):
    _, _, _, models_dir = trained
    with pytest.raises(ValueError, match="вероятностная модель"):
        load_panel_models(models_dir, quantile=True)
