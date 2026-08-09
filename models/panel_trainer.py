# -*- coding: utf-8 -*-
"""
models/panel_trainer.py — обучение и оценка глобальных моделей на панели рядов.

ЧЕМ ОЦЕНКА ПАНЕЛИ ОТЛИЧАЕТСЯ ОТ ОДНОРЯДНОЙ
──────────────────────────────────────────
Одно усреднённое MAE по всем рядам вводит в заблуждение: крупные фидеры
доминируют в абсолютной ошибке и полностью скрывают качество на мелких.
Поэтому считаются три величины:

  micro  — по всем окнам сразу, взвешено размером ряда;
  macro  — среднее по рядам, каждый ряд весит одинаково;
  worst  — худший ряд, который и определяет пригодность модели в эксплуатации.

Все они вычисляются в ИСХОДНОМ масштабе каждого ряда: сравнивать
нормированные величины между фидерами разной мощности бессмысленно.

ИЕРАРХИЯ
────────
Городской прогноз получается суммированием прогнозов фидеров (bottom-up).
Поскольку фактический городской ряд по построению равен сумме фидеров,
согласованность проверяется напрямую, а не постулируется.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data.panel_preprocessing import inverse_scale_series
from utils.metrics import (
    mean_absolute_error, root_mean_squared_error,
    mean_absolute_percentage_error, r2_score,
)

logger = logging.getLogger("smart_grid.models.panel_trainer")


def seasonal_naive_scale(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Знаменатель MASE по рядам: ошибка суточного наивного прогноза на ОБУЧЕНИИ.

    Масштаб берётся с train, а не с оцениваемого отрезка. Иначе он менялся бы
    вместе с тестовой выборкой, и MASE перестала бы быть сопоставимой между
    прогонами и режимами — а сопоставимость и есть её единственный смысл.

    Абсолютные метрики на панели несравнимы между фидерами: MAE крупного
    промышленного ряда на порядок больше MAE мелкого жилого при одинаковом
    качестве прогноза. MASE безразмерна, поэтому усреднение по разнородным
    рядам осмысленно, а порог 1.0 отделяет модель от повтора прошлых суток.

    Результат кэшируется в самом словаре данных: величина зависит только от
    обучающей выборки и одинакова для всех моделей.
    """
    cached = data.get("_mase_scale")
    if cached is not None:
        return cached

    from models.panel_models import consumption_index

    series = data["series_train"]
    horizon = int(data["Y_train"].shape[1])
    channel = consumption_index(data["feature_names_hist"])

    y_true = inverse_scale_series(data, data["Y_train"], series)
    y_naive = inverse_scale_series(
        data, data["X_hist_train"][:, -horizon:, channel], series)

    scale: Dict[str, float] = {}
    for sid in np.unique(series):
        mask = series == sid
        key = data["series_index"][int(sid)]
        scale[key] = float(np.mean(np.abs(y_true[mask] - y_naive[mask])))

    degenerate = [k for k, v in scale.items() if not np.isfinite(v) or v <= 0]
    if degenerate:
        logger.warning(
            "MASE неопределена для %d рядов с нулевым суточным изменением: %s",
            len(degenerate), ", ".join(degenerate[:5]))

    data["_mase_scale"] = scale
    return scale


def make_batch(data: Dict[str, Any], split: str) -> Dict[str, np.ndarray]:
    """Собирает единообразный набор входов для любого типа модели."""
    return {
        "hist": data[f"X_hist_{split}"],
        "future": data[f"X_future_{split}"],
        "static": data[f"X_static_{split}"],
        "series": data[f"series_{split}"],
        "anchor": data[f"anchor_{split}"],
        "feature_names_hist": data["feature_names_hist"],
        "feature_names_future": data["feature_names_future"],
    }


class PanelTrainer:
    """Единый интерфейс обучения для Keras- и sklearn-подобных panel-моделей."""

    def __init__(self, model: Any, name: str):
        self.model = model
        self.name = name
        self.train_time: float = 0.0
        self.history = None
        self.n_params: int = 0

    def _is_keras(self) -> bool:
        import tensorflow as tf
        return isinstance(self.model, tf.keras.Model)

    def _keras_inputs(self, batch: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Порядок входов соответствует порядку, заданному при сборке модели."""
        names = [inp.name.split(":")[0] for inp in self.model.inputs]
        mapping = {"hist_input": batch["hist"], "future_input": batch["future"],
                   "static_input": batch["static"]}
        return [mapping[n] for n in names if n in mapping] or [batch["hist"]]

    def train(self, data: Dict[str, Any], epochs: int = 50, batch_size: int = 128,
              patience: int = 10) -> "PanelTrainer":
        import tensorflow as tf

        logger.info("=" * 70)
        logger.info("Обучение panel-модели: %s", self.name)
        t0 = time.time()

        if self._is_keras():
            self.n_params = int(self.model.count_params())
            cb = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1)]
            # ReduceLROnPlateau не добавляется: у panel-моделей задано
            # собственное расписание learning rate.
            self.history = self.model.fit(
                self._keras_inputs(make_batch(data, "train")), data["Y_train"],
                validation_data=(self._keras_inputs(make_batch(data, "val")),
                                 data["Y_val"]),
                epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=2,
            )
        else:
            self.model.fit(data)

        self.train_time = time.time() - t0
        logger.info("%s обучена за %.1f с", self.name, self.train_time)
        return self

    def predict(self, data: Dict[str, Any], split: str) -> np.ndarray:
        batch = make_batch(data, split)
        if self._is_keras():
            return self.model.predict(self._keras_inputs(batch), verbose=0)
        return self.model.predict(batch)

    def evaluate(self, data: Dict[str, Any], split: str = "test") -> Dict[str, Any]:
        """Метрики micro, macro и по худшему ряду в исходном масштабе."""
        pred_scaled = self.predict(data, split)
        series = data[f"series_{split}"]

        y_true = inverse_scale_series(data, data[f"Y_{split}"], series)
        y_pred = inverse_scale_series(data, pred_scaled, series)

        micro = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

        scale = seasonal_naive_scale(data)

        per_series: Dict[str, float] = {}
        per_series_mase: Dict[str, float] = {}
        maes, mapes, mases = [], [], []
        for sid in np.unique(series):
            mask = series == sid
            key = data["series_index"][int(sid)]
            mae = mean_absolute_error(y_true[mask], y_pred[mask])
            per_series[key] = mae
            maes.append(mae)
            mapes.append(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))

            denom = scale.get(key, 0.0)
            if np.isfinite(denom) and denom > 0:
                per_series_mase[key] = mae / denom
                mases.append(mae / denom)

        worst_key = max(per_series, key=per_series.get)
        # MASE усредняется по рядам, а не по окнам: она уже безразмерна, и
        # взвешивание размером ряда вернуло бы доминирование крупных фидеров,
        # ради устранения которого метрика и вводится.
        mase_macro = float(np.mean(mases)) if mases else float("nan")
        mase_worst = float(max(per_series_mase.values())) if per_series_mase else float("nan")

        result = {
            "model": self.name,
            "MAE": micro["MAE"], "RMSE": micro["RMSE"],
            "MAPE": micro["MAPE"], "R2": micro["R2"],
            "MAE_macro": float(np.mean(maes)),
            "MAPE_macro": float(np.mean(mapes)),
            "MAE_worst_series": float(per_series[worst_key]),
            "worst_series": worst_key,
            "MASE": mase_macro,
            "MASE_worst_series": mase_worst,
            "n_params": self.n_params,
            "train_time_sec": round(self.train_time, 1),
            "per_series_MAE": per_series,
        }
        logger.info(
            "%-14s micro MAE=%9.2f | macro MAE=%9.2f | MASE=%6.3f | MAPE=%6.2f%% | "
            "R²=%7.4f | худший ряд %s (MAE=%0.1f)",
            self.name, result["MAE"], result["MAE_macro"], result["MASE"],
            result["MAPE"], result["R2"], worst_key, result["MAE_worst_series"],
        )
        return result


def bottom_up_city_forecast(
    data: Dict[str, Any], y_pred_scaled: np.ndarray, split: str = "test",
) -> Dict[str, Any]:
    """
    Собирает городской прогноз суммированием прогнозов фидеров.

    Суммируются только окна с одинаковой временной привязкой: складывать
    прогнозы, сделанные из разных моментов, нельзя — это разные величины.
    Фактический городской ряд равен сумме фидеров по построению генератора,
    поэтому ошибка согласованности здесь измеряется, а не предполагается.
    """
    series = data[f"series_{split}"]
    anchors = data[f"anchor_{split}"]
    index = data["series_index"]

    y_true = inverse_scale_series(data, data[f"Y_{split}"], series)
    y_pred = inverse_scale_series(data, y_pred_scaled, series)

    cities = np.array([index[int(s)].split("/")[0] for s in series])

    rows: List[Dict[str, Any]] = []
    for city in np.unique(cities):
        mask = cities == city
        city_anchors = anchors[mask]
        true_c, pred_c = y_true[mask], y_pred[mask]

        # Группировка по моменту прогноза: суммируем одновременные окна.
        order = np.argsort(city_anchors, kind="stable")
        a_sorted = city_anchors[order]
        uniq, starts = np.unique(a_sorted, return_index=True)
        splits = np.split(order, starts[1:])

        agg_true = np.stack([true_c[g].sum(axis=0) for g in splits])
        agg_pred = np.stack([pred_c[g].sum(axis=0) for g in splits])

        rows.append({
            "city_id": city,
            "n_origins": len(uniq),
            "MAE": mean_absolute_error(agg_true, agg_pred),
            "MAPE": mean_absolute_percentage_error(agg_true, agg_pred),
            "R2": r2_score(agg_true, agg_pred),
            "mean_city_load": float(agg_true.mean()),
        })

    logger.info("─" * 78)
    logger.info("ГОРОДСКОЙ ПРОГНОЗ СУММИРОВАНИЕМ ФИДЕРОВ (bottom-up)")
    logger.info("%-10s %10s %12s %9s %12s", "Город", "окон", "MAE", "MAPE%", "средняя")
    logger.info("─" * 78)
    for r in rows:
        logger.info("%-10s %10d %12.1f %8.2f%% %12.1f",
                    r["city_id"], r["n_origins"], r["MAE"], r["MAPE"],
                    r["mean_city_load"])
    logger.info("─" * 78)
    return {"per_city": rows,
            "MAE_mean": float(np.mean([r["MAE"] for r in rows])),
            "MAPE_mean": float(np.mean([r["MAPE"] for r in rows]))}


def compare_panel_models(trainers: List[PanelTrainer], data: Dict[str, Any],
                         split: str = "test") -> Dict[str, Dict[str, Any]]:
    """Сводная таблица по всем моделям панели."""
    results: Dict[str, Dict[str, Any]] = {}
    for tr in trainers:
        try:
            results[tr.name] = tr.evaluate(data, split)
        except Exception as exc:
            logger.error("Ошибка оценки %s: %s", tr.name, exc)

    if not results:
        return results

    logger.info("\n%s", "─" * 92)
    logger.info("СРАВНЕНИЕ PANEL-МОДЕЛЕЙ (split=%s, %d рядов)",
                split, len(data["series_index"]))
    logger.info("%-14s %11s %11s %8s %9s %11s %9s", "Модель", "MAE micro",
                "MAE macro", "MAPE%", "R2", "худший ряд", "время,с")
    logger.info("%s", "─" * 92)
    for name in sorted(results, key=lambda k: results[k]["MAE"]):
        m = results[name]
        logger.info("%-14s %11.2f %11.2f %7.2f%% %9.4f %11.1f %9.1f",
                    name, m["MAE"], m["MAE_macro"], m["MAPE"], m["R2"],
                    m["MAE_worst_series"], m["train_time_sec"])
    logger.info("%s", "─" * 92)
    return results
