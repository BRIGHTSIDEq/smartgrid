import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import Config, GeneratorCoefficients
from data.components.ev import simulate_ev_load
from data.components.household import build_household_aggregate
from data.components.industrial import simulate_industrial_load
from data.generator import load_or_generate_smartgrid_data
from data.components.nonlinear_states import (
    compute_cascade_factor,
    compute_dsr_rebound,
    compute_grid_stress_nonlinear,
)
from data.components.weather import compute_weather
from utils.deployment import predict_multifeature_from_bundle
from utils.plot_style import apply_publication_style, save_figure
from models.trainer import ModelTrainer


def test_config_generator_coefficients_dataclass():
    coeffs = Config.get_generator_coefficients()
    assert isinstance(coeffs, GeneratorCoefficients)
    assert coeffs.city_districts >= 1


def test_plot_style_apply_and_save(tmp_path):
    apply_publication_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png, pdf, svg = save_figure(fig, os.path.join(tmp_path, "smoke_plot.png"), save=True)
    plt.close(fig)
    assert os.path.exists(png)
    assert pdf is None
    assert svg is None


def test_component_smoke_paths():
    rng = np.random.default_rng(42)
    days = 14
    hours = days * 24
    t = np.arange(hours, dtype=np.float32)
    dates = pd.date_range("2024-01-01", periods=hours, freq="h")
    weather = compute_weather(rng=rng, t=t, dates=dates, days=days)
    assert weather.temperature.shape == (hours,)

    weekday = ((t // 24).astype(int) % 7).astype(np.int8)
    is_weekend = (weekday >= 5).astype(np.float32)
    holiday_mask = np.zeros(hours, dtype=np.float32)
    agg = build_household_aggregate(
        rng=rng,
        hour_arr=t % 24,
        is_weekend=is_weekend,
        holiday_mask=holiday_mask,
        households=500,
        early_bird_frac=0.28,
        night_owl_frac=0.20,
        days=days,
    )
    assert agg.shape == (hours,)

    holiday_daily = np.zeros(days, dtype=np.float32)
    ev = simulate_ev_load(
        rng=rng,
        days=days,
        hours=hours,
        households=500,
        ev_penetration=0.3,
        weekday=weekday,
        holiday_mask_daily=holiday_daily,
        temperature=weather.temperature,
    )
    ind = simulate_industrial_load(rng=rng, hours=hours, industrial_loads=3, weekday=weekday)
    assert ev.shape == (hours,)
    assert ind.shape == (hours,)

    base = np.maximum(100.0, agg * 1000.0).astype(np.float32)
    cascade = compute_cascade_factor(rng=rng, base_consumption=base)
    stress = compute_grid_stress_nonlinear(
        rng=rng,
        temperature=weather.temperature,
        temp_setpoint=18.0,
        hour_of_day=(t % 24).astype(np.int8),
        ev_load_raw=ev,
        cloud_cover=weather.cloud_cover,
    )
    rebound = compute_dsr_rebound(rng=rng, dsr_active=np.zeros(hours, dtype=np.float32))
    assert cascade.shape == (hours,)
    assert stress.shape == (hours,)
    assert rebound.shape == (hours,)


def test_deployment_multivariate_inference():
    history = 12
    features = 3
    horizon = 4
    x_in = tf.keras.Input(shape=(history, features))
    x = tf.keras.layers.Flatten()(x_in)
    out = tf.keras.layers.Dense(horizon)(x)
    model = tf.keras.Model(x_in, out)
    model.compile(optimizer="adam", loss="mse")

    scaler = MinMaxScaler()
    scaler.fit(np.random.rand(256, features))
    bundle = {
        "model": model,
        "scaler": scaler,
        "config": {"HISTORY_LENGTH": history, "N_FEATURES": features},
    }
    recent_window = np.random.rand(history, features).astype(np.float32)
    pred = predict_multifeature_from_bundle(bundle, recent_window)
    assert pred.shape == (horizon,)



def test_load_or_generate_smartgrid_data_uses_csv_cache(tmp_path):
    csv_path = tmp_path / "cached_data.csv"
    df_first = load_or_generate_smartgrid_data(
        csv_path=str(csv_path),
        days=10,
        households=100,
        start_date="2024-01-01",
        seed=42,
    )
    assert csv_path.exists()
    assert "timestamp" in df_first.columns
    assert len(df_first) == 240

    df_second = load_or_generate_smartgrid_data(
        csv_path=str(csv_path),
        days=10,
        households=100,
        start_date="2024-01-01",
        seed=42,
    )
    pd.testing.assert_frame_equal(df_first, df_second, check_dtype=False)


def test_load_or_generate_smartgrid_data_invalidates_incompatible_cache(tmp_path):
    csv_path = tmp_path / "cached_data.csv"
    df_10_days = load_or_generate_smartgrid_data(
        csv_path=str(csv_path),
        days=10,
        households=100,
        start_date="2024-01-01",
        seed=42,
    )
    assert len(df_10_days) == 240

    # Должна произойти регенерация, т.к. ожидаем уже 12 дней
    df_12_days = load_or_generate_smartgrid_data(
        csv_path=str(csv_path),
        days=12,
        households=100,
        start_date="2024-01-01",
        seed=42,
    )
    assert len(df_12_days) == 288


def test_model_trainer_uses_tft_split_inputs():
    history = 12
    horizon = 4
    # Модель с двумя входами, как у TFT-Lite
    x_series = tf.keras.Input(shape=(history, 1))
    x_cov = tf.keras.Input(shape=(history, 4))
    xs = tf.keras.layers.Flatten()(x_series)
    xc = tf.keras.layers.Flatten()(x_cov)
    out = tf.keras.layers.Dense(horizon)(tf.keras.layers.Concatenate()([xs, xc]))
    model = tf.keras.Model([x_series, x_cov], out)
    model.compile(optimizer="adam", loss="mse")

    trainer = ModelTrainer(model, model_name="TFT-Lite")
    n = 8
    data = {
        "X_test": np.random.rand(n, history, 26).astype(np.float32),  # неверный формат для TFT
        "X_tft_test": [
            np.random.rand(n, history, 1).astype(np.float32),
            np.random.rand(n, history, 4).astype(np.float32),
        ],
        "Y_test": np.random.rand(n, horizon).astype(np.float32),
        "Y_seasonal_naive_test": np.zeros((n, horizon), dtype=np.float32),
        "seasonal_diff": True,
        "scaler": MinMaxScaler().fit(np.random.rand(256, 1)),
    }
    metrics = trainer.evaluate(data, split="test", run_residual_diagnostics=False)
    assert "MAE" in metrics and np.isfinite(metrics["MAE"])