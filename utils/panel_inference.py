# -*- coding: utf-8 -*-
"""
Пакетный прогноз по сохранённой многорядной модели.

Каталог models/ панельного прогона содержит обученные модели, нормировщики и
manifest.json (panel_pipeline.save_panel_models). Здесь по нему строится
прогноз на следующие horizon часов для каждого ряда:

  история  — длинная таблица series, timestamp, consumption и измеряемые
             ковариаты, не меньше history + 168 часов подряд на ряд;
  погода   — прогноз погоды на целевые часы: series (или общий для всех),
             timestamp и колонки temperature, humidity, cloud_cover.

Признаки строятся теми же функциями, что при обучении
(data.panel_preprocessing.history_channels и future_channels_from_forecast).

ВРЕМЯ ВЫДАЧИ. Модель прогнозирует ровно horizon часов после последнего
наблюдения. Заявка на рынок на сутки D подаётся днём D−1, и тогда между
последним измерением и началом суток D лежит разрыв. check_issue_time
сообщает, покрывает ли прогноз целевые сутки; модель с горизонтом 24 ч
покрывает их, только если история заканчивается в 23:00 D−1.
"""

import json
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

MAX_LAG_HOURS = 168


# Модель по умолчанию для продуктового прогноза с интервалами: на реальных
# данных квантильный бустинг дал лучший pinball и почти номинальное покрытие.
DEFAULT_QUANTILE_MODEL = "QuantileXGBoost"


def load_panel_models(models_dir: str, model_name: Optional[str] = None,
                      quantile: bool = False) -> Dict[str, Any]:
    """
    Загружает модель и предобработку.

    По умолчанию — лучшую точечную модель по валидации; с quantile=True —
    вероятностную (по умолчанию квантильный XGBoost).
    """
    with open(os.path.join(models_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    pool = manifest.get("quantile_models", {}) if quantile else manifest["models"]
    name = model_name or (DEFAULT_QUANTILE_MODEL if quantile else manifest["best_model_by_val"])
    if name not in pool:
        kind = "вероятностная модель" if quantile else "модель"
        raise ValueError(f"{kind} {name} не сохранена; есть: {', '.join(pool) or 'ничего'}")
    path = os.path.join(models_dir, pool[name])
    if path.endswith(".keras"):
        import tensorflow as tf
        import models.dlinear  # noqa: F401  регистрирует нестандартные слои
        model = tf.keras.models.load_model(path, compile=False)
    else:
        with open(path, "rb") as f:
            model = pickle.load(f)
    with open(os.path.join(models_dir, manifest["preprocessing"]), "rb") as f:
        prep = pickle.load(f)
    return {"name": name, "model": model, "prep": prep, "manifest": manifest,
            "quantiles": manifest.get("quantiles") if quantile else None}


def _calendar(frame: pd.DataFrame) -> pd.DataFrame:
    from data.generator import holiday_flags

    ts = pd.to_datetime(frame["timestamp"])
    out = frame.copy()
    out["hour"] = ts.dt.hour.to_numpy()
    out["weekday"] = ts.dt.dayofweek.to_numpy()
    out["day_of_year"] = ts.dt.dayofyear.to_numpy()
    out["is_holiday"] = holiday_flags(ts).astype(np.float32)
    return out


def build_inference_batch(history: pd.DataFrame, weather: pd.DataFrame,
                          prep: Dict[str, Any]) -> Dict[str, Any]:
    """
    Собирает по одному окну на ряд в том же виде, что make_batch при обучении.

    Отказы вместо догадок: неизвестный ряд (модель не знает его нормировки),
    короткая или прерывистая история, отсутствие прогноза погоды на целевые
    часы — всё это ошибки с понятным текстом, а не нули во входе модели.
    """
    from data.panel_preprocessing import future_channels_from_forecast, history_channels

    H = int(prep["history_length"])
    horizon = int(prep["forecast_horizon"])
    index = list(prep["series_index"])
    hist_numeric = list(prep.get("hist_numeric", [c for c in prep["weather_scalers"]]))
    static_names = list(prep["static_names"])

    need = ["series", "timestamp", "consumption"] + hist_numeric + static_names
    missing = [c for c in need if c not in history.columns]
    if missing:
        raise ValueError(f"В истории нет колонок: {', '.join(missing)}")

    hist_rows, fut_rows, stat_rows, sids, targets = [], [], [], [], []
    for key, grp in history.groupby("series", sort=True):
        key = str(key)
        if key not in index:
            raise ValueError(f"Ряд {key} не встречался при обучении: его нормировка неизвестна")
        grp = _calendar(grp.sort_values("timestamp").reset_index(drop=True))
        if len(grp) < H + MAX_LAG_HOURS:
            raise ValueError(f"Ряд {key}: нужно не меньше {H + MAX_LAG_HOURS} ч истории, "
                             f"получено {len(grp)}")
        steps = pd.to_datetime(grp["timestamp"]).diff().dropna()
        if not (steps == pd.Timedelta(hours=1)).all():
            raise ValueError(f"Ряд {key}: отметки времени должны идти подряд с шагом 1 ч")
        if grp[["consumption"] + hist_numeric].isna().any().any():
            raise ValueError(f"Ряд {key}: в истории есть пропуски")

        scaler = prep["series_scalers"][key]
        cons = scaler.transform(grp["consumption"].to_numpy(np.float64).reshape(-1, 1))
        channels = history_channels(grp, cons.flatten().astype(np.float32),
                                    prep["weather_scalers"], hist_numeric)
        matrix = np.stack([channels[k] for k in prep["feature_names_hist"]], axis=-1)
        hist_rows.append(matrix[-H:])

        last = pd.Timestamp(grp["timestamp"].iloc[-1])
        target = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon, freq="h")
        w = weather[weather["series"].astype(str) == key] if "series" in weather.columns \
            else weather
        w = w.assign(timestamp=pd.to_datetime(w["timestamp"])).set_index("timestamp")
        w = w.reindex(target)
        fut = _calendar(pd.DataFrame({"timestamp": target}))
        for col in w.columns:
            if col != "series":
                fut[col] = w[col].to_numpy()
        f_channels = future_channels_from_forecast(fut, prep["weather_scalers"])
        absent = [n for n in prep["feature_names_future"] if n not in f_channels]
        if absent:
            raise ValueError(f"Ряд {key}: нет прогноза погоды для {', '.join(absent)}")
        future = np.stack([f_channels[k][0] for k in prep["feature_names_future"]], axis=-1)
        if np.isnan(future).any():
            raise ValueError(f"Ряд {key}: прогноз погоды не покрывает все целевые часы")
        fut_rows.append(future)

        static = (grp.iloc[0][static_names].to_numpy(np.float64) - prep["static_mean"]) \
            / prep["static_std"] if static_names else np.zeros(0)
        stat_rows.append(static.astype(np.float32))
        sids.append(index.index(key))
        targets.append(target)

    n = len(sids)
    return {
        "X_hist_inference": np.stack(hist_rows).astype(np.float32),
        "X_future_inference": np.stack(fut_rows).astype(np.float32),
        "X_static_inference": np.stack(stat_rows).astype(np.float32),
        "series_inference": np.asarray(sids, dtype=np.int32),
        "anchor_inference": np.zeros(n, dtype=np.int64),
        "feature_names_hist": prep["feature_names_hist"],
        "feature_names_future": prep["feature_names_future"],
        "series_index": index, "series_scalers": prep["series_scalers"],
        "targets": targets,
    }


def forecast_panel(bundle: Dict[str, Any], history: pd.DataFrame,
                   weather: pd.DataFrame) -> pd.DataFrame:
    """Прогноз на следующие horizon часов: series, timestamp, forecast (кВт·ч)."""
    from data.panel_preprocessing import inverse_scale_series
    from models.panel_trainer import PanelTrainer

    batch = build_inference_batch(history, weather, bundle["prep"])
    trainer = PanelTrainer(bundle["model"], bundle["name"])
    scaled = np.asarray(trainer.predict(batch, "inference"))
    sids = batch["series_inference"]

    if scaled.ndim == 3:
        # Уровни упорядочиваются по возрастанию: пересечение квантилей
        # (P10 выше P50) бессмысленно для заявки и устраняется сортировкой,
        # как при оценке модели.
        scaled = np.sort(scaled, axis=-1)
        levels = bundle.get("quantiles") or [0.1, 0.5, 0.9]
        columns = {f"p{int(round(100 * q))}": inverse_scale_series(batch, scaled[:, :, i], sids)
                   for i, q in enumerate(levels)}
    else:
        columns = {"forecast": inverse_scale_series(batch, scaled, sids)}

    rows = []
    for i, (sid, target) in enumerate(zip(sids, batch["targets"])):
        part = {"series": batch["series_index"][int(sid)], "timestamp": target}
        part.update({k: v[i] for k, v in columns.items()})
        part["model"] = bundle["name"]
        rows.append(pd.DataFrame(part))
    return pd.concat(rows, ignore_index=True)


def check_issue_time(forecast: pd.DataFrame, target_day) -> Dict[str, Any]:
    """
    Покрывает ли прогноз все часы целевых суток.

    Возвращает долю покрытых часов и первый непокрытый час: по ним видно,
    хватает ли горизонта модели для заявки, поданной в данный момент.
    """
    day = pd.Timestamp(target_day).normalize()
    hours = pd.date_range(day, periods=24, freq="h")
    have = set(pd.to_datetime(forecast["timestamp"]))
    missing = [h for h in hours if h not in have]
    return {"target_day": str(day.date()), "covered_share": 1 - len(missing) / 24,
            "first_missing": str(missing[0]) if missing else None}
