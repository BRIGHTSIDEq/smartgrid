# -*- coding: utf-8 -*-
"""
utils/deployment.py — Экспорт модели, сериализация, инференс.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

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
    Инференс: принимает сырые значения потребления, возвращает прогноз.

    Parameters
    ----------
    bundle        : dict из load_model_bundle()
    recent_values : np.ndarray shape=(history_length,) — последние N часов

    Returns
    -------
    np.ndarray shape=(forecast_horizon,) — прогноз в оригинальном масштабе
    """
    model: tf.keras.Model = bundle["model"]
    scaler = bundle["scaler"]
    history = bundle["config"].get("HISTORY_LENGTH", 48)

    if len(recent_values) < history:
        raise ValueError(
            f"recent_values должен содержать не менее {history} значений, "
            f"получено {len(recent_values)}"
        )

    window = recent_values[-history:].reshape(-1, 1)
    scaled = scaler.transform(window).flatten()
    X = scaled[np.newaxis, :, np.newaxis].astype(np.float32)  # (1, T, 1)

    pred_scaled = model.predict(X, verbose=0)[0]              # (H,)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    return pred
