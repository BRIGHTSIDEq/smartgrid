# -*- coding: utf-8 -*-
"""
data/panel_preprocessing.py — подготовка окон для многорядного обучения.

ЧТО ЗДЕСЬ ПРИНЦИПИАЛЬНО ИНАЧЕ, ЧЕМ В ОДНОРЯДНОМ СЛУЧАЕ
──────────────────────────────────────────────────────
1. Окна нарезаются ВНУТРИ каждого ряда. Если нарезать общий массив насквозь,
   на стыке фидеров окно склеит хвост одного объекта с началом другого, и
   модель будет обучаться на несуществующем переходе.

2. Масштаб нормируется ПО КАЖДОМУ РЯДУ отдельно. Фидеры различаются по
   мощности в десятки раз; общая нормировка сжала бы мелкие ряды почти в ноль,
   и глобальная модель училась бы только на крупных.

3. Разделение хронологическое и одинаковое для всех рядов: границы train/val/
   test задаются по времени, а не по номеру окна. Иначе тестовый период одного
   фидера попал бы в обучающий период другого, а через общую погоду это прямая
   утечка.

ТРИ ГРУППЫ ПРИЗНАКОВ
────────────────────
historical  — что известно об окне истории: потребление, погода фактическая,
              электротранспорт, генерация, календарь.
future      — что ТОЧНО известно о целевом окне заранее: календарь, тарифная
              зона, праздники, а также ПРОГНОЗ погоды.
static      — постоянные характеристики фидера: размер, состав потребителей,
              чувствительности.

Ключевое ограничение: в future-признаки нельзя класть фактическую будущую
погоду. В реальной эксплуатации доступен только прогноз, и его ошибка растёт
с горизонтом. Использование факта дало бы модели информацию, которой у неё
никогда не будет, и завысило бы качество.

История изменений — в CHANGELOG.md.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("smart_grid.data.panel_preprocessing")

# Погодные величины, для которых строится прогноз с ошибкой.
FORECASTABLE_WEATHER = ("temperature", "humidity", "cloud_cover")


def make_weather_forecast(
    actual: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    base_sigma: float,
    growth_per_hour: float,
) -> np.ndarray:
    """
    Строит прогноз погоды с ошибкой, растущей по мере удаления горизонта.

    Возвращает матрицу (T, horizon): для каждого момента t — прогноз на h часов
    вперёд, каким он был бы сделан в момент t.

    Ошибка моделируется случайным блужданием по горизонту: сосед­ние lead time
    коррелированы, а дисперсия накапливается — так ведёт себя реальная ошибка
    численного прогноза погоды. Независимый шум на каждом шаге дал бы
    нереалистично «рваный» прогноз, который модель легко усреднила бы.

    Прогноз строится ТОЛЬКО по фактической погоде и не использует нагрузку:
    иначе в известные заранее признаки просочилась бы целевая переменная.
    """
    n = len(actual)
    # Матрица факта: future_actual[t, h] = actual[t + h + 1]
    idx = np.arange(n)[:, None] + np.arange(1, horizon + 1)[None, :]
    idx = np.clip(idx, 0, n - 1)
    future_actual = actual[idx]

    # Накопленная ошибка: шаг блуждания одинаков, дисперсия растёт как sqrt(h).
    steps = rng.normal(0.0, 1.0, size=(n, horizon))
    walk = np.cumsum(steps, axis=1)
    lead = np.arange(1, horizon + 1)[None, :]
    sigma = base_sigma + growth_per_hour * lead
    error = walk / np.sqrt(lead) * sigma

    return (future_actual + error).astype(np.float32)


def _cyclical(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    angle = 2 * np.pi * values / period
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def _tariff_zone_code(hour: np.ndarray, weekday: np.ndarray,
                      holiday: np.ndarray) -> np.ndarray:
    """
    Кодирует тарифную зону: 0 ночная, 0.5 полупиковая, 1 пиковая.

    Порядок соответствует действующему в России: пик 07–10 и 17–21,
    полупик 10–17 и 21–23, ночь 23–07. В выходные и праздники пика нет.
    """
    # np.full по shape, а не по длине: функция вызывается и для одномерного
    # ряда истории, и для двумерной матрицы (момент × шаг горизонта).
    code = np.full(np.shape(hour), 0.5, dtype=np.float32)
    code[(hour < 7) | (hour >= 23)] = 0.0
    is_peak = (((hour >= 7) & (hour < 10)) | ((hour >= 17) & (hour < 21)))
    code[is_peak & (weekday < 5) & (holiday < 0.5)] = 1.0
    return code


def build_future_known_frame(
    frame: pd.DataFrame,
    horizon: int,
    rng: np.random.Generator,
    weather_forecast_sigma: Dict[str, Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """
    Формирует известные заранее признаки целевого окна для одного ряда.

    Календарь известен точно: час, день недели, праздник и тарифную зону можно
    вычислить на любую дату вперёд. Погода известна лишь как прогноз.
    """
    n = len(frame)
    idx = np.clip(np.arange(n)[:, None] + np.arange(1, horizon + 1)[None, :], 0, n - 1)

    hour = frame["hour"].values.astype(np.float32)[idx]
    weekday = frame["weekday"].values.astype(np.float32)[idx]
    holiday = frame["is_holiday"].values.astype(np.float32)[idx]
    doy = frame["day_of_year"].values.astype(np.float32)[idx]

    hour_sin, hour_cos = _cyclical(hour, 24.0)
    dow_sin, dow_cos = _cyclical(weekday, 7.0)
    month_sin, month_cos = _cyclical(doy, 365.0)
    is_weekend = (weekday >= 5).astype(np.float32)
    tariff = _tariff_zone_code(hour, weekday, holiday)

    channels: Dict[str, np.ndarray] = {
        "fut_hour_sin": hour_sin, "fut_hour_cos": hour_cos,
        "fut_dow_sin": dow_sin, "fut_dow_cos": dow_cos,
        "fut_month_sin": month_sin, "fut_month_cos": month_cos,
        "fut_is_weekend": is_weekend, "fut_is_holiday": holiday,
        "fut_tariff_zone": tariff,
        "fut_is_peak": (tariff >= 0.99).astype(np.float32),
        "fut_is_night": (tariff <= 0.01).astype(np.float32),
    }

    # Прогноз погоды: только он, никогда не факт.
    for name in FORECASTABLE_WEATHER:
        if name not in frame.columns:
            continue
        base_sigma, growth = weather_forecast_sigma.get(name, (0.5, 0.05))
        channels[f"fut_{name}_forecast"] = make_weather_forecast(
            frame[name].values.astype(np.float64), horizon, rng, base_sigma, growth)
    return channels


def _make_series_windows(
    hist_matrix: np.ndarray,
    future_channels: Dict[str, np.ndarray],
    target: np.ndarray,
    history: int,
    horizon: int,
    start: int,
    end: int,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Нарезает окна одного ряда в пределах [start, end).

    Окно допустимо, только если и история, и горизонт целиком помещаются в
    заданный интервал: иначе обучающее окно заглянуло бы в валидацию.
    """
    first = start
    last = end - history - horizon
    if last < first:
        return None

    anchors = np.arange(first, last + 1)
    hist_idx = anchors[:, None] + np.arange(history)[None, :]
    tgt_idx = anchors[:, None] + history + np.arange(horizon)[None, :]

    X_hist = hist_matrix[hist_idx]
    Y = target[tgt_idx]

    # Признаки будущего берутся в точке, где заканчивается история: именно
    # оттуда делается прогноз.
    origin = anchors + history - 1
    X_future = np.stack([future_channels[k][origin] for k in sorted(future_channels)],
                        axis=-1)

    return {"X_hist": X_hist.astype(np.float32),
            "X_future": X_future.astype(np.float32),
            "Y": Y.astype(np.float32),
            "anchor": anchors}


def prepare_panel_data(
    df: pd.DataFrame,
    specs: List[Any],
    history_length: int = 48,
    forecast_horizon: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    weather_forecast_sigma: Optional[Dict[str, Tuple[float, float]]] = None,
    max_windows_warn: int = 2_000_000,
) -> Dict[str, Any]:
    """
    Готовит окна для глобальной модели по всем рядам panel-датасета.

    Returns
    -------
    dict с ключами X_hist_/X_future_/X_static_/Y_ для train, val, test,
    метаданными рядов, скалерами и схемой признаков.
    """
    weather_forecast_sigma = weather_forecast_sigma or {
        # Базовая ошибка и её рост на час горизонта. Значения соответствуют
        # порядку точности численного прогноза погоды на сутки вперёд.
        "temperature": (0.6, 0.06),
        "humidity": (3.0, 0.25),
        "cloud_cover": (0.06, 0.006),
    }

    rng = np.random.default_rng(seed)
    df = df.sort_values(["city_id", "feeder_id", "timestamp"]).reset_index(drop=True)

    timestamps = np.sort(df["timestamp"].unique())
    n_time = len(timestamps)
    train_end = int(n_time * train_ratio)
    val_end = int(n_time * (train_ratio + val_ratio))

    # Границы одинаковы для всех рядов: иначе тест одного фидера пересёкся бы
    # по времени с обучением другого, а через общую погоду это утечка.
    logger.info(
        "Panel split по времени: train %s … %s | val … %s | test … %s",
        timestamps[0], timestamps[train_end - 1],
        timestamps[val_end - 1], timestamps[-1],
    )

    static_cols = sorted([c for c in df.columns if c.startswith("static_")])
    hist_numeric = ["temperature", "humidity", "wind_speed", "cloud_cover",
                    "ev_load_kw", "solar_gen_kw", "dsr_active"]
    hist_numeric = [c for c in hist_numeric if c in df.columns]

    # ── Скалеры погоды: общие, обучены на обучающем ОТРЕЗКЕ ВРЕМЕНИ ─────────
    train_mask_global = df["timestamp"].isin(timestamps[:train_end])
    weather_scalers: Dict[str, MinMaxScaler] = {}
    for col in hist_numeric:
        sc = MinMaxScaler((0, 1))
        sc.fit(df.loc[train_mask_global, col].values.reshape(-1, 1))
        weather_scalers[col] = sc

    parts: Dict[str, Dict[str, List[np.ndarray]]] = {
        split: {"X_hist": [], "X_future": [], "X_static": [], "Y": [],
                "series": [], "anchor": []}
        for split in ("train", "val", "test")
    }
    series_scalers: Dict[str, MinMaxScaler] = {}
    series_index: List[str] = []
    feature_names_hist: List[str] = []
    feature_names_future: List[str] = []

    for series_no, ((city, feeder), grp) in enumerate(
            df.groupby(["city_id", "feeder_id"], sort=True)):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        series_key = f"{city}/{feeder}"
        series_index.append(series_key)

        # ── Нормировка потребления ПО ЭТОМУ РЯДУ, только по train-отрезку ───
        cons = grp["consumption"].values.astype(np.float64).reshape(-1, 1)
        sc = MinMaxScaler((0, 1))
        sc.fit(cons[:train_end])
        series_scalers[series_key] = sc
        cons_scaled = sc.transform(cons).flatten().astype(np.float32)

        hour = grp["hour"].values.astype(np.float32)
        weekday = grp["weekday"].values.astype(np.float32)
        holiday = grp["is_holiday"].values.astype(np.float32)
        doy = grp["day_of_year"].values.astype(np.float32)

        hs, hc = _cyclical(hour, 24.0)
        ds, dc = _cyclical(weekday, 7.0)
        ms, mc = _cyclical(doy, 365.0)

        hist_channels: Dict[str, np.ndarray] = {
            "consumption": cons_scaled,
            "hour_sin": hs, "hour_cos": hc,
            "dow_sin": ds, "dow_cos": dc,
            "month_sin": ms, "month_cos": mc,
            "is_weekend": (weekday >= 5).astype(np.float32),
            "is_holiday": holiday,
            "tariff_zone": _tariff_zone_code(hour, weekday, holiday),
        }
        for col in hist_numeric:
            hist_channels[col] = weather_scalers[col].transform(
                grp[col].values.reshape(-1, 1)).flatten().astype(np.float32)
        # Лаги потребления в масштабе самого ряда.
        for lag in (24, 168):
            lagged = np.concatenate([np.full(lag, cons_scaled[0], dtype=np.float32),
                                     cons_scaled[:-lag]])
            hist_channels[f"lag_{lag}h"] = lagged

        if not feature_names_hist:
            feature_names_hist = sorted(hist_channels)
        hist_matrix = np.stack([hist_channels[k] for k in feature_names_hist], axis=-1)

        future_channels = build_future_known_frame(
            grp, forecast_horizon, rng, weather_forecast_sigma)
        if not feature_names_future:
            feature_names_future = sorted(future_channels)

        static_vec = grp.iloc[0][static_cols].values.astype(np.float32) if static_cols \
            else np.zeros(0, dtype=np.float32)

        bounds = {"train": (0, train_end),
                  "val": (train_end, val_end),
                  "test": (val_end, len(grp))}
        for split, (lo, hi) in bounds.items():
            win = _make_series_windows(hist_matrix, future_channels, cons_scaled,
                                       history_length, forecast_horizon, lo, hi)
            if win is None:
                continue
            n_win = len(win["Y"])
            parts[split]["X_hist"].append(win["X_hist"])
            parts[split]["X_future"].append(win["X_future"])
            parts[split]["Y"].append(win["Y"])
            parts[split]["X_static"].append(np.repeat(static_vec[None, :], n_win, axis=0))
            parts[split]["series"].append(np.full(n_win, series_no, dtype=np.int32))
            parts[split]["anchor"].append(win["anchor"])

    data: Dict[str, Any] = {}
    total_windows = 0
    for split in ("train", "val", "test"):
        if not parts[split]["Y"]:
            raise ValueError(
                f"Сплит {split} пуст: длина ряда меньше history+horizon. "
                "Увеличьте число дней или уменьшите окно."
            )
        for key in ("X_hist", "X_future", "X_static", "Y", "series", "anchor"):
            data[f"{key}_{split}"] = np.concatenate(parts[split][key], axis=0)
        total_windows += len(data[f"Y_{split}"])

    if total_windows > max_windows_warn:
        logger.warning(
            "Сформировано %d окон — материализация в памяти может быть тяжёлой. "
            "Для больших режимов предусмотрите потоковую подачу (tf.data).",
            total_windows,
        )

    data.update({
        "series_index": series_index,
        "series_scalers": series_scalers,
        "weather_scalers": weather_scalers,
        "feature_names_hist": feature_names_hist,
        "feature_names_future": feature_names_future,
        "static_names": static_cols,
        "history_length": history_length,
        "forecast_horizon": forecast_horizon,
        "timestamps": timestamps,
        "split_time_bounds": {"train_end": train_end, "val_end": val_end},
        "weather_forecast_sigma": weather_forecast_sigma,
    })

    logger.info(
        "Panel-окна: train=%d val=%d test=%d | рядов=%d | "
        "hist=(%d, %d) future=(%d, %d) static=%d",
        len(data["Y_train"]), len(data["Y_val"]), len(data["Y_test"]),
        len(series_index), history_length, len(feature_names_hist),
        forecast_horizon, len(feature_names_future), len(static_cols),
    )
    return data


def inverse_scale_series(data: Dict[str, Any], values: np.ndarray,
                         series_ids: np.ndarray) -> np.ndarray:
    """
    Возвращает прогноз в исходный масштаб КАЖДОГО ряда отдельно.

    Общая обратная нормировка здесь невозможна: у фидеров разные диапазоны, и
    применение чужого скалера исказило бы величину в разы.
    """
    values = np.atleast_2d(values)
    out = np.empty_like(values, dtype=np.float64)
    index = data["series_index"]
    for sid in np.unique(series_ids):
        mask = series_ids == sid
        scaler = data["series_scalers"][index[int(sid)]]
        flat = values[mask].reshape(-1, 1)
        out[mask] = scaler.inverse_transform(flat).reshape(values[mask].shape)
    return out
