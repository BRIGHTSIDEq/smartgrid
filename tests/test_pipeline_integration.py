# -*- coding: utf-8 -*-
"""
Интеграционные тесты: инференс из бандла и диагностика остатков.

Оба блока раньше содержали дефекты, которые не проявлялись в метриках:
инференс падал на несовпадении формы входа, а диагностика измеряла
автокорреляцию не на тех лагах.
"""

import numpy as np
import pytest

from data.generator import generate_smartgrid_data
from data.preprocessing import prepare_data
from models.trainer import diagnose_residuals
from models.baseline import build_persistence_24, build_hourly_profile


@pytest.fixture(scope="module")
def tiny_data():
    df = generate_smartgrid_data(days=25, households=30, seed=3,
                                 industrial_loads=2, city_districts=2)
    data = prepare_data(df, history_length=48, forecast_horizon=24)
    return df, data


# ── Диагностика остатков ──────────────────────────────────────────────────────

def test_diagnose_residuals_detects_true_24h_autocorrelation():
    """
    ACF(24) должна реагировать на суточную периодичность остатков.

    Ключевой момент: настоящий почасовой ряд — это срез residuals[:, h] по
    номеру окна, а не развёрнутая матрица. Здесь остатки заданы синусоидой
    с периодом 24 именно по номеру окна, поэтому корректная реализация обязана
    увидеть сильную корреляцию на лаге 24.
    """
    n_windows, horizon = 600, 24
    idx = np.arange(n_windows)
    resid_column = np.sin(2 * np.pi * idx / 24.0)
    resid = np.tile(resid_column[:, None], (1, horizon))

    y_true = np.zeros((n_windows, horizon))
    diag = diagnose_residuals(y_true, -resid, model_name="test")

    assert diag["ACF_24"] > 0.8, "Суточная автокорреляция не обнаружена"
    assert diag["ACF_24_mean_over_h"] > 0.8


def test_diagnose_residuals_clean_on_white_noise():
    """На белом шуме автокорреляция должна быть близка к нулю, а DW — к 2."""
    rng = np.random.default_rng(0)
    n_windows, horizon = 800, 24
    resid = rng.normal(0, 1, size=(n_windows, horizon))

    diag = diagnose_residuals(np.zeros_like(resid), -resid, model_name="noise")

    assert abs(diag["ACF_24_mean_over_h"]) < 0.15
    assert 1.7 < diag["DW_mean_over_h"] < 2.3


def test_diagnose_residuals_reports_shape():
    resid = np.zeros((100, 24))
    diag = diagnose_residuals(resid, resid, model_name="zeros")
    assert diag["n_windows"] == 100
    assert diag["horizon"] == 24


# ── Наивные базлайны ──────────────────────────────────────────────────────────

def test_persistence_repeats_previous_day(tiny_data):
    """
    Суточный наивный прогноз должен буквально повторять последние 24 часа окна.
    """
    _, data = tiny_data
    model = build_persistence_24()
    model.fit(data["X_train"], data["Y_train"])

    X = data["X_test"][:5]
    pred = model.predict(X)

    assert pred.shape == (5, 24)
    # Канал 0 — потребление; последние 24 шага окна и есть прогноз.
    assert np.allclose(pred, X[:, -24:, 0])


def test_hourly_profile_predicts_within_observed_range(tiny_data):
    """Климатологический прогноз не должен выходить за диапазон обучающих данных."""
    _, data = tiny_data
    model = build_hourly_profile()
    model.fit(data["X_train"], data["Y_train"])

    n = min(20, len(data["X_test"]))
    pred = model.predict(data["X_test"][:n])
    train_cons = data["X_train"][:, :, 0]

    assert pred.shape == (n, 24)
    assert pred.min() >= train_cons.min() - 1e-6
    assert pred.max() <= train_cons.max() + 1e-6


# ── Инференс из бандла ────────────────────────────────────────────────────────

def test_model_bundle_roundtrip(tiny_data, tmp_path):
    """
    Полный цикл: обучили → экспортировали → загрузили → получили прогноз.

    Раньше инференс строил вход формы (1, T, 1), тогда как модель ожидает
    (1, T, N_FEATURES), и пример из документации падал при первом же вызове.
    """
    from models.lstm import build_lstm_model
    from utils.deployment import (
        export_model_bundle, load_model_bundle, predict_from_bundle,
    )

    df, data = tiny_data
    model = build_lstm_model(
        history_length=48, forecast_horizon=24, n_features=data["n_features"],
        lstm_units_1=8, tcn_filters=4, attn_heads=1,
    )
    model.fit(data["X_train"][:32], data["Y_train"][:32], epochs=1, verbose=0)

    bundle_dir = export_model_bundle(
        model, data,
        {"HISTORY_LENGTH": 48, "FORECAST_HORIZON": 24,
         "N_FEATURES": data["n_features"], "model_name": "TestLSTM"},
        export_dir=str(tmp_path), model_name="TestLSTM",
    )

    bundle = load_model_bundle(bundle_dir)
    assert bundle["scalers"]["scaler"] is not None
    assert bundle["config"]["N_FEATURES"] == data["n_features"]

    recent = df.tail(200)
    forecast = predict_from_bundle(bundle, recent)

    assert forecast.shape == (24,)
    assert np.isfinite(forecast).all()

    # Загруженная модель должна давать в точности тот же прогноз, что исходная:
    # это проверяет и сохранность весов, и идентичность сборки признаков.
    from data.preprocessing import (
        _add_lag_columns, _build_feature_matrix, inverse_scale,
    )
    features = _build_feature_matrix(
        _add_lag_columns(recent.copy()).iloc[-48:],
        cons_scaler=data["scaler"], temp_scaler=data["temp_scaler"],
        humidity_scaler=data["humidity_scaler"], wind_scaler=data["wind_scaler"],
        rolling_std_scaler=data["rolling_std_scaler"],
        cloud_scaler=data["cloud_scaler"],
        ev_scaler=data["ev_scaler"], solar_scaler=data["solar_scaler"],
        temp_sq_max=data["temp_sq_max"],
    )
    expected = inverse_scale(
        data["scaler"], model.predict(features[np.newaxis], verbose=0)
    )[0]
    assert np.allclose(forecast, expected, rtol=1e-4, atol=1e-2)


def test_predict_from_bundle_rejects_short_history(tiny_data, tmp_path):
    """Слишком короткая история должна давать понятную ошибку, а не падение."""
    from models.lstm import build_lstm_model
    from utils.deployment import (
        export_model_bundle, load_model_bundle, predict_from_bundle,
    )

    df, data = tiny_data
    model = build_lstm_model(
        history_length=48, forecast_horizon=24, n_features=data["n_features"],
        lstm_units_1=8, tcn_filters=4, attn_heads=1,
    )
    bundle_dir = export_model_bundle(
        model, data,
        {"HISTORY_LENGTH": 48, "FORECAST_HORIZON": 24,
         "N_FEATURES": data["n_features"], "model_name": "TestShort"},
        export_dir=str(tmp_path), model_name="TestShort",
    )
    bundle = load_model_bundle(bundle_dir)

    with pytest.raises(ValueError, match="не менее 48"):
        predict_from_bundle(bundle, df.head(10))
