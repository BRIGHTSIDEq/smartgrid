import numpy as np

from models.lstm import build_lstm_model
from models.transformer import build_vanilla_transformer
from models.trainer import diagnose_training_regime


def test_diagnose_training_regime_overfitting():
    hist = {"mae": [0.12, 0.08, 0.05], "val_mae": [0.13, 0.10, 0.09]}
    diag = diagnose_training_regime(hist, overfit_gap=0.02, underfit_floor=0.08)
    assert diag["status"] == "overfitting"


def test_diagnose_training_regime_underfitting():
    hist = {"mae": [0.18, 0.16, 0.14], "val_mae": [0.19, 0.18, 0.15]}
    diag = diagnose_training_regime(hist, overfit_gap=0.02, underfit_floor=0.08)
    assert diag["status"] == "underfitting"


def test_lstm_seasonal_blend_forward_pass():
    model = build_lstm_model(
        history_length=48,
        forecast_horizon=24,
        n_features=26,
        lstm_units_1=32,
        tcn_filters=16,
        attn_heads=4,
        seasonal_blend_init=0.30,
    )
    x = np.random.rand(2, 48, 26).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 24)
    assert np.isfinite(y).all()


def test_vanilla_transformer_seasonal_residual_forward_pass():
    model = build_vanilla_transformer(
        history_length=48,
        forecast_horizon=24,
        n_features=26,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dff=64,
        use_seasonal_residual=True,
        seasonal_blend_init=0.4,
        huber_delta=0.05,
    )
    x = np.random.rand(2, 48, 26).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 24)
    assert np.isfinite(y).all()
