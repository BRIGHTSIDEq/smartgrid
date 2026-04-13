import numpy as np
import pytest

from utils.metrics import compute_all_metrics, mean_absolute_error


def test_mean_absolute_error_rejects_mismatched_lengths():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="same number of elements"):
        mean_absolute_error(y_true, y_pred)


def test_compute_all_metrics_rejects_empty_arrays():
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError, match="non-empty arrays"):
        compute_all_metrics(y_true, y_pred, model_name="dummy")


def test_compute_all_metrics_contains_extended_metrics():
    y_true = np.array([100.0, 120.0, 130.0, 110.0])
    y_pred = np.array([98.0, 125.0, 128.0, 109.0])
    metrics = compute_all_metrics(y_true, y_pred)
    for key in ("MAE", "RMSE", "MAPE", "sMAPE", "WAPE", "MBE", "R2"):
        assert key in metrics
