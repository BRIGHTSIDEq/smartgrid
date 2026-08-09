# -*- coding: utf-8 -*-
"""
utils/deployment.py — Экспорт обученной модели и инференс.

Бандл модели самодостаточен: помимо весов он содержит ВСЕ скалеры, которыми
были нормированы признаки при обучении, и конфигурацию окна. Без этого
воспроизвести вход модели невозможно — она принимает не сырой ряд потребления,
а матрицу из N_FEATURES ковариат на каждый час, построенную теми же скалерами,
что и на обучении.

Состав бандла:
    model.keras     веса и архитектура
    scalers.pkl     cons/temp/humidity/wind/rolling_std/cloud/ev/solar + temp_sq_max
    config.json     HISTORY_LENGTH, FORECAST_HORIZON, N_FEATURES, имя модели
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

logger = logging.getLogger("smart_grid.utils.deployment")

# Ключи скалеров, которые нужны для восстановления матрицы признаков.
_SCALER_KEYS = (
    "scaler", "temp_scaler", "humidity_scaler", "wind_scaler",
    "rolling_std_scaler", "cloud_scaler", "ev_scaler", "solar_scaler",
)


def export_model_bundle(
    model: tf.keras.Model,
    data: Dict[str, Any],
    config_dict: Dict[str, Any],
    export_dir: str = "results/models",
    model_name: str = "best_model",
) -> str:
    """
    Сохраняет модель, все скалеры и конфиг в одну директорию.

    Parameters
    ----------
    model : tf.keras.Model
    data : dict
        Словарь из prepare_data() — из него берутся скалеры.
    config_dict : dict
        Гиперпараметры окна (HISTORY_LENGTH, FORECAST_HORIZON, N_FEATURES).
    """
    bundle_dir = os.path.join(export_dir, model_name)
    os.makedirs(bundle_dir, exist_ok=True)

    try:
        model_path = os.path.join(bundle_dir, "model.keras")
        model.save(model_path)

        scalers = {k: data.get(k) for k in _SCALER_KEYS}
        scalers["temp_sq_max"] = data.get("temp_sq_max")
        with open(os.path.join(bundle_dir, "scalers.pkl"), "wb") as f:
            pickle.dump(scalers, f)

        with open(os.path.join(bundle_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info("Бандл модели сохранён: %s (модель + %d скалеров + конфиг)",
                    bundle_dir, sum(1 for v in scalers.values() if v is not None))
    except Exception as exc:
        logger.error("Ошибка экспорта модели: %s", exc)
        raise

    return bundle_dir


def load_model_bundle(bundle_dir: str) -> Dict[str, Any]:
    """
    Загружает бандл: модель + скалеры + конфиг.

    Returns
    -------
    {"model": ..., "scalers": {...}, "config": {...}}
    """
    from models.transformer import (
        PreLNEncoderBlock, SinusoidalPE, Time2Vec,
        ProbSparseAttention, RevINNorm, RevINDenorm, StochasticDepth,
        LearnedQueryPooling, LearnableRelativePE,
    )
    from models.lstm import (
        TemporalAttentionBlock, TCNBlock, SeasonalSkipConnection, ConsumptionRevIN,
    )

    custom_objects = {
        "PreLNEncoderBlock": PreLNEncoderBlock,
        "SinusoidalPE": SinusoidalPE,
        "Time2Vec": Time2Vec,
        "ProbSparseAttention": ProbSparseAttention,
        "RevINNorm": RevINNorm,
        "RevINDenorm": RevINDenorm,
        "StochasticDepth": StochasticDepth,
        "LearnedQueryPooling": LearnedQueryPooling,
        "LearnableRelativePE": LearnableRelativePE,
        "TemporalAttentionBlock": TemporalAttentionBlock,
        "TCNBlock": TCNBlock,
        "SeasonalSkipConnection": SeasonalSkipConnection,
        "ConsumptionRevIN": ConsumptionRevIN,
    }

    try:
        model = tf.keras.models.load_model(
            os.path.join(bundle_dir, "model.keras"), custom_objects=custom_objects,
        )
        scalers_path = os.path.join(bundle_dir, "scalers.pkl")
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(
                f"В бандле нет scalers.pkl: {bundle_dir}. Бандл собран старой "
                "версией export_model_bundle и не пригоден для инференса — "
                "переэкспортируйте модель."
            )
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        with open(os.path.join(bundle_dir, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        logger.info("Бандл загружен из: %s", bundle_dir)
        return {"model": model, "scalers": scalers, "config": config}
    except Exception as exc:
        logger.error("Ошибка загрузки бандла: %s", exc)
        raise


def assert_input_matches_model(
    model: tf.keras.Model,
    X: np.ndarray,
    expected_features: Optional[int] = None,
) -> None:
    """
    Проверяет, что тензор входа соответствует модели по длине окна и числу
    признаков.

    Keras не защищает от подачи входа неверной ширины: тензор (1, 48, 1)
    успешно проходит через модель, обученную на (None, 48, 26), — последняя
    размерность просто транслируется. Прогноз при этом получается численно
    правдоподобным, но бессмысленным, и заметить это по метрикам невозможно.
    Поэтому ширина входа проверяется явно.

    Raises
    ------
    ValueError — если длина окна или число признаков не совпадают.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(
            f"Ожидается трёхмерный вход (batch, history, features), получено {X.shape}"
        )

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):          # модели с несколькими входами
        input_shape = input_shape[0]

    if input_shape is not None and len(input_shape) == 3:
        exp_history, exp_features = input_shape[1], input_shape[2]
        if exp_history is not None and X.shape[1] != exp_history:
            raise ValueError(
                f"Длина окна {X.shape[1]} не совпадает с ожидаемой моделью {exp_history}"
            )
        if exp_features is not None and X.shape[2] != exp_features:
            raise ValueError(
                f"Число признаков {X.shape[2]} не совпадает с ожидаемым моделью "
                f"{exp_features}. Вход должен содержать полную матрицу ковариат, "
                "построенную тем же препроцессингом, что и при обучении."
            )

    if expected_features is not None and X.shape[2] != expected_features:
        raise ValueError(
            f"Число признаков {X.shape[2]} не совпадает с конфигурацией бандла "
            f"({expected_features})"
        )


def predict_from_bundle(
    bundle: Dict[str, Any],
    recent_df: pd.DataFrame,
) -> np.ndarray:
    """
    Инференс на новых данных.

    Parameters
    ----------
    bundle : dict
        Результат load_model_bundle().
    recent_df : pd.DataFrame
        Последние HISTORY_LENGTH часов наблюдений. Набор колонок — тот же, что
        отдаёт data.generator.generate_smartgrid_data: consumption, temperature,
        humidity, wind_speed, cloud_cover, ev_load_kw, solar_gen_kw, dsr_active,
        hour, weekday, is_weekend, is_holiday, is_peak_hour, is_night_hour,
        tariff_zone, day_of_year, timestamp.
        Отсутствующие ковариаты заменяются нулями с предупреждением в лог.

    Returns
    -------
    np.ndarray shape=(FORECAST_HORIZON,) — прогноз в кВт·ч.
    """
    from data.preprocessing import _add_lag_columns, _build_feature_matrix, inverse_scale

    model: tf.keras.Model = bundle["model"]
    scalers: Dict[str, Any] = bundle["scalers"]
    config: Dict[str, Any] = bundle["config"]

    history = int(config.get("HISTORY_LENGTH", 48))
    n_features = int(config.get("N_FEATURES", 26))

    if len(recent_df) < history:
        raise ValueError(
            f"recent_df должен содержать не менее {history} строк "
            f"(HISTORY_LENGTH), получено {len(recent_df)}"
        )

    df = _add_lag_columns(recent_df.copy()).iloc[-history:]

    features = _build_feature_matrix(
        df,
        cons_scaler=scalers["scaler"],
        temp_scaler=scalers["temp_scaler"],
        humidity_scaler=scalers.get("humidity_scaler"),
        wind_scaler=scalers.get("wind_scaler"),
        rolling_std_scaler=scalers.get("rolling_std_scaler"),
        cloud_scaler=scalers.get("cloud_scaler"),
        ev_scaler=scalers.get("ev_scaler"),
        solar_scaler=scalers.get("solar_scaler"),
        temp_sq_max=scalers.get("temp_sq_max"),
    )

    if features.shape[1] != n_features:
        raise ValueError(
            f"Построено {features.shape[1]} признаков, модель ожидает {n_features}. "
            "Проверьте, что состав колонок recent_df совпадает с обучающим."
        )

    X = features[np.newaxis, :, :].astype(np.float32)   # (1, T, F)
    assert_input_matches_model(model, X, expected_features=n_features)
    pred_scaled = model.predict(X, verbose=0)
    return inverse_scale(scalers["scaler"], pred_scaled)[0]
