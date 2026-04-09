import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import Config, GeneratorCoefficients
from data.components.ev import simulate_ev_load
from data.components.household import build_household_aggregate
from data.components.industrial import simulate_industrial_load
from data.components.nonlinear_states import (
    compute_cascade_factor,
    compute_dsr_rebound,
    compute_grid_stress_nonlinear,
)
from data.components.weather import compute_weather
from utils.deployment import predict_multifeature_from_bundle
from utils.plot_style import apply_publication_style, save_figure


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
    assert os.path.exists(pdf)
    assert os.path.exists(svg)


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
