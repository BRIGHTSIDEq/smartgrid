# -*- coding: utf-8 -*-
"""
utils/deployment.py — Экспорт модели, сериализация, инференс.
"""

import json
import logging
import os
import pickle
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

logger = logging.getLogger("smart_grid.utils.deployment")


def export_model_bundle(
    model: tf.keras.Model,
    scaler: Any,
    config_dict: Dict[str, Any],
    export_dir: str = "results/models",
    model_name: str = "best_model",
) -> str:
    """
    Сохраняет Keras-модель + scaler + конфиг в одну директорию.

    Parameters
    ----------
    model       : tf.keras.Model
    scaler      : MinMaxScaler
    config_dict : dict с гиперпараметрами
    export_dir  : str
    model_name  : str

    Returns
    -------
    str — путь к директории бандла
    """
    bundle_dir = os.path.join(export_dir, model_name)
    os.makedirs(bundle_dir, exist_ok=True)

    try:
        # Модель
        model_path = os.path.join(bundle_dir, "model.keras")
        model.save(model_path)
        logger.info("Модель сохранена: %s", model_path)

        # Scaler
        scaler_path = os.path.join(bundle_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Scaler сохранён: %s", scaler_path)

        # Конфиг
        config_path = os.path.join(bundle_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        logger.info("Конфиг сохранён: %s", config_path)

    except Exception as exc:
        logger.error("Ошибка экспорта модели: %s", exc)
        raise

    return bundle_dir


def load_model_bundle(bundle_dir: str) -> Dict[str, Any]:
    """
    Загружает бандл: модель + scaler + конфиг.

    Returns
    -------
    {"model": ..., "scaler": ..., "config": ...}
    """
    from models.transformer import (
        PreLNEncoderBlock, SinusoidalPE, Time2Vec, GatedResidualNetwork, ProbSparseAttention
    )

    try:
        custom_objects = {
            "PreLNEncoderBlock": PreLNEncoderBlock,
            "SinusoidalPE": SinusoidalPE,
            "Time2Vec": Time2Vec,
            "GatedResidualNetwork": GatedResidualNetwork,
            "ProbSparseAttention": ProbSparseAttention,
        }
        model = tf.keras.models.load_model(
            os.path.join(bundle_dir, "model.keras"),
            custom_objects=custom_objects,
        )
        with open(os.path.join(bundle_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(bundle_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Бандл загружен из: %s", bundle_dir)
        return {"model": model, "scaler": scaler, "config": config}
    except Exception as exc:
        logger.error("Ошибка загрузки бандла: %s", exc)
        raise


def predict_from_bundle(
    bundle: Dict[str, Any],
    recent_values: np.ndarray,
) -> np.ndarray:
    """
    Deprecated wrapper для однопризнакового инференса.

    Для multivariate-инференса используйте `predict_multifeature_from_bundle`.

    Parameters
    ----------
    bundle        : dict из load_model_bundle()
    recent_values : np.ndarray shape=(history_length,) — последние N часов

    Returns
    -------
    np.ndarray shape=(forecast_horizon,) — прогноз в оригинальном масштабе
    """
    warnings.warn(
        "predict_from_bundle is deprecated and will be removed in a future release. "
        "Use predict_multifeature_from_bundle(bundle, recent_window).",
        DeprecationWarning,
        stacklevel=2,
    )
    history = int(bundle.get("config", {}).get("HISTORY_LENGTH", 48))
    recent_arr = np.asarray(recent_values, dtype=np.float32).reshape(-1)
    if recent_arr.size < history:
        raise ValueError(
            f"recent_values должен содержать не менее {history} значений, "
            f"получено {recent_arr.size}"
        )
    window = recent_arr[-history:].reshape(history, 1)
    return predict_multifeature_from_bundle(bundle=bundle, recent_window=window)


def _validate_inference_window_shape(
    recent_window: np.ndarray,
    history_length: int,
    n_features: int,
) -> np.ndarray:
    """
    Validate and normalize inference window to shape (history_length, n_features).
    """
    arr = np.asarray(recent_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"recent_window должен иметь ndim=2 и форму (T, F), получено ndim={arr.ndim}"
        )
    if arr.shape[0] < history_length:
        raise ValueError(
            f"Недостаточная длина истории: требуется >= {history_length}, получено {arr.shape[0]}"
        )
    if arr.shape[1] != n_features:
        raise ValueError(
            f"Число признаков не совпадает с конфигом: ожидается {n_features}, "
            f"получено {arr.shape[1]}"
        )
    return arr[-history_length:, :]


def _transform_window_with_scaler(window: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Scale multivariate window with scikit-like scaler.
    """
    transformed = scaler.transform(window)
    if transformed.shape != window.shape:
        raise ValueError(
            f"Scaler вернул некорректную форму: ожидалось {window.shape}, получено {transformed.shape}"
        )
    return transformed.astype(np.float32)


def predict_multifeature_from_bundle(
    bundle: Dict[str, Any],
    recent_window: np.ndarray,
) -> np.ndarray:
    """
    Run multivariate inference from an exported model bundle.

    Args:
        bundle: Output from `load_model_bundle`, containing `model`, `scaler`, `config`.
        recent_window: Last observed multivariate window with shape (T, F), where
            T >= config["HISTORY_LENGTH"] and F == config["N_FEATURES"].

    Returns:
        Forecast array in original target units with shape (forecast_horizon,).

    Raises:
        KeyError: If required keys are missing in bundle.
        ValueError: If input schema or scaler output shape is invalid.

    Example:
        >>> bundle = load_model_bundle("results/models/PatchTST")
        >>> x = np.random.rand(bundle["config"]["HISTORY_LENGTH"], bundle["config"]["N_FEATURES"])
        >>> y = predict_multifeature_from_bundle(bundle, x)
        >>> y.shape
        (24,)
    """
    if "model" not in bundle or "scaler" not in bundle or "config" not in bundle:
        raise KeyError("bundle должен содержать ключи: model, scaler, config")

    model: tf.keras.Model = bundle["model"]
    scaler = bundle["scaler"]
    config = bundle["config"]
    history = int(config.get("HISTORY_LENGTH", 48))
    n_features = int(config.get("N_FEATURES", 1))

    window = _validate_inference_window_shape(recent_window, history, n_features)
    scaled_window = _transform_window_with_scaler(window, scaler)
    X = scaled_window[np.newaxis, :, :]  # (1, T, F)

    pred_scaled = model.predict(X, verbose=0)[0]
    pred_scaled_2d = pred_scaled.reshape(-1, 1)

    if hasattr(scaler, "n_features_in_") and int(scaler.n_features_in_) > 1:
        template = np.zeros((pred_scaled_2d.shape[0], int(scaler.n_features_in_)), dtype=np.float32)
        template[:, 0] = pred_scaled_2d[:, 0]
        pred = scaler.inverse_transform(template)[:, 0]
    else:
        pred = scaler.inverse_transform(pred_scaled_2d).flatten()

    return pred.astype(np.float32)
