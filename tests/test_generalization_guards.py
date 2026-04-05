import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.lstm import build_lstm_model
from models.transformer import build_vanilla_transformer


def _make_synthetic_windows(n_samples=160, history=24, horizon=6, n_features=19, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples + history + horizon + 200, dtype=np.float32)

    daily = np.sin(2 * np.pi * t / 24.0)
    weekly = 0.5 * np.sin(2 * np.pi * t / 168.0)
    trend = 0.002 * t
    signal = 0.55 + 0.22 * daily + 0.12 * weekly + trend
    signal += rng.normal(0.0, 0.015, size=signal.shape)
    signal = np.clip(signal, 0.0, 1.0)

    X = np.zeros((n_samples, history, n_features), dtype=np.float32)
    Y = np.zeros((n_samples, horizon), dtype=np.float32)

    for i in range(n_samples):
        start = i
        hist = signal[start : start + history]
        fut = signal[start + history : start + history + horizon]

        hour = (np.arange(history) + start) % 24
        X[i, :, 0] = hist
        X[i, :, 1] = np.sin(2 * np.pi * hour / 24.0)
        X[i, :, 2] = np.cos(2 * np.pi * hour / 24.0)
        X[i, :, 3] = ((hour >= 8) & (hour <= 22)).astype(np.float32)
        X[i, :, 4] = ((hour < 6) | (hour >= 23)).astype(np.float32)

        for f in range(5, n_features):
            X[i, :, f] = np.clip(hist + rng.normal(0.0, 0.01, size=history), 0.0, 1.0)

        Y[i] = fut

    n_train = int(n_samples * 0.75)
    return (X[:n_train], Y[:n_train]), (X[n_train:], Y[n_train:])


def test_lstm_not_underfit_on_structured_synthetic_data():
    (x_train, y_train), (x_val, y_val) = _make_synthetic_windows()

    model = build_lstm_model(
        history_length=x_train.shape[1],
        forecast_horizon=y_train.shape[1],
        n_features=x_train.shape[2],
        lstm_units_1=64,
        lstm_units_2=32,
        dropout_rate=0.1,
        learning_rate=5e-4,
    )

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=18,
        batch_size=16,
        verbose=0,
    )

    train_mae = float(hist.history["mae"][-1])
    val_mae = float(hist.history["val_mae"][-1])

    assert train_mae < 0.09, f"LSTM underfit: train_mae={train_mae:.4f}"
    assert val_mae < 0.12, f"LSTM val degraded: val_mae={val_mae:.4f}"
    assert val_mae / max(train_mae, 1e-8) < 1.8, (
        f"LSTM overfit risk: val/train={val_mae / max(train_mae, 1e-8):.2f}"
    )


def test_vanilla_transformer_overfit_capacity_smoke():
    (x_train, y_train), _ = _make_synthetic_windows(n_samples=96)

    model = build_vanilla_transformer(
        history_length=x_train.shape[1],
        forecast_horizon=y_train.shape[1],
        n_features=x_train.shape[2],
        d_model=48,
        num_heads=4,
        num_layers=2,
        dff=96,
        dropout=0.05,
        learning_rate=8e-4,
        stochastic_depth_rate=0.0,
    )

    # Намеренно маленький срез: проверяем способность модели запомнить сложный шаблон
    x_small = x_train[:24]
    y_small = y_train[:24]

    hist = model.fit(x_small, y_small, epochs=30, batch_size=8, verbose=0)
    final_loss = float(hist.history["loss"][-1])

    assert final_loss < 0.01, f"Transformer cannot overfit tiny sample, loss={final_loss:.5f}"
