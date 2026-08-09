# -*- coding: utf-8 -*-
"""
models/panel_models.py — модели для многорядного (panel) обучения.

Все модели здесь получают ОДИН И ТОТ ЖЕ набор входов:

    hist   (B, history, n_hist)     история ряда
    future (B, horizon, n_future)   точно известный календарь и прогноз погоды
    static (B, n_static)            постоянные характеристики фидера

Единообразие принципиально для честности сравнения. Если бы будущие признаки
получали только нейросети, разрыв в метриках отражал бы разницу в объёме
входной информации, а не в способности моделей. Поэтому Ridge и градиентный
бустинг получают те же признаки в развёрнутом виде.

Обучается ОДНА глобальная модель на всех рядах сразу, а не отдельная модель на
каждый фидер: при десятках рядов это на порядок больше обучающих примеров, и
именно так устроены современные системы прогнозирования нагрузки.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("smart_grid.models.panel_models")

# Индекс канала потребления. Препроцессинг гарантирует нулевую позицию, но
# модели дополнительно разрешают его по имени: молчаливая работа не с тем
# каналом не проявляется ни в одной проверке форм и обнаруживается только по
# неправдоподобным метрикам.
CONSUMPTION_CHANNEL = 0


def consumption_index(feature_names_hist) -> int:
    """Возвращает индекс канала потребления, разрешая его по имени."""
    if feature_names_hist and "consumption" in feature_names_hist:
        return list(feature_names_hist).index("consumption")
    return CONSUMPTION_CHANNEL


# ══════════════════════════════════════════════════════════════════════════════
# НАИВНЫЕ БАЗЛАЙНЫ
# ══════════════════════════════════════════════════════════════════════════════

class PanelNaive24:
    """
    Суточный наивный прогноз: повтор последних 24 часов истории.

    Работает в нормированном пространстве каждого ряда, поэтому пригоден для
    панели с разномасштабными фидерами без дополнительных преобразований.
    """

    name = "Naive24"

    def __init__(self, consumption_channel: int = CONSUMPTION_CHANNEL):
        self.channel = consumption_channel
        self.horizon: Optional[int] = None

    def fit(self, data: Dict[str, Any], **_) -> "PanelNaive24":
        self.horizon = int(data["Y_train"].shape[1])
        logger.info("%s: обучение не требуется", self.name)
        return self

    def predict(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        hist = batch["hist"]
        h = self.horizon or hist.shape[1]
        ch = consumption_index(batch.get("feature_names_hist"))
        return hist[:, -h:, ch].astype(np.float32)


class PanelHourlyProfile:
    """
    Климатологический прогноз: среднее по (ряд, час, день недели) на обучении.

    Профиль считается ОТДЕЛЬНО для каждого ряда: у фидеров разные формы
    суточного графика, и общий профиль усреднил бы жилой и промышленный
    объекты в одну бессмысленную кривую.
    """

    name = "HourlyProfile"

    def __init__(self):
        self.profiles: Optional[np.ndarray] = None    # (n_series, 7, 24)
        self.global_mean: float = 0.0
        self.horizon: Optional[int] = None

    def fit(self, data: Dict[str, Any], **_) -> "PanelHourlyProfile":
        hist = data["X_hist_train"]
        series = data["series_train"]
        names = data["feature_names_hist"]
        self.horizon = int(data["Y_train"].shape[1])

        i_sin, i_cos = names.index("hour_sin"), names.index("hour_cos")
        d_sin, d_cos = names.index("dow_sin"), names.index("dow_cos")
        ch = consumption_index(names)
        n_series = len(data["series_index"])

        sums = np.zeros((n_series, 7, 24))
        counts = np.zeros((n_series, 7, 24))

        for step in range(hist.shape[1]):
            hour = _decode_cycle(hist[:, step, i_sin], hist[:, step, i_cos], 24)
            dow = _decode_cycle(hist[:, step, d_sin], hist[:, step, d_cos], 7)
            np.add.at(sums, (series, dow, hour), hist[:, step, ch])
            np.add.at(counts, (series, dow, hour), 1.0)

        self.global_mean = float(hist[:, :, ch].mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            self.profiles = np.where(counts > 0, sums / np.maximum(counts, 1e-9),
                                     self.global_mean).astype(np.float32)
        logger.info("%s: профили построены для %d рядов", self.name, n_series)
        return self

    def predict(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        assert self.profiles is not None, "модель не обучена"
        future, series = batch["future"], batch["series"]
        names = batch["feature_names_future"]

        i_sin, i_cos = names.index("fut_hour_sin"), names.index("fut_hour_cos")
        d_sin, d_cos = names.index("fut_dow_sin"), names.index("fut_dow_cos")

        h = future.shape[1]
        out = np.empty((len(series), h), dtype=np.float32)
        for j in range(h):
            hour = _decode_cycle(future[:, j, i_sin], future[:, j, i_cos], 24)
            dow = _decode_cycle(future[:, j, d_sin], future[:, j, d_cos], 7)
            out[:, j] = self.profiles[series, dow, hour]
        return out


def _decode_cycle(sin_v: np.ndarray, cos_v: np.ndarray, period: int) -> np.ndarray:
    """Восстанавливает целочисленное значение из циклического кодирования."""
    ang = np.arctan2(sin_v, cos_v)
    return np.mod(np.round(ang / (2 * np.pi) * period), period).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# КЛАССИЧЕСКИЕ МОДЕЛИ НА РАЗВЁРНУТЫХ ПРИЗНАКАХ
# ══════════════════════════════════════════════════════════════════════════════

def flatten_panel_inputs(batch: Dict[str, np.ndarray],
                         aggregate_history: bool = False) -> np.ndarray:
    """
    Сводит три входа в одну матрицу признаков.

    aggregate_history=True сжимает историю до скользящих агрегатов: при полном
    развороте окна признаков становится больше тысячи, и обучение отдельной
    модели на каждый шаг горизонта перестаёт быть практичным.
    """
    hist, future, static = batch["hist"], batch["future"], batch["static"]

    if aggregate_history:
        cons = hist[:, :, consumption_index(batch.get("feature_names_hist"))]
        pieces = [cons[:, -24:]]                       # последние сутки целиком
        for w in (6, 12, 24, 48):
            w = min(w, cons.shape[1])
            pieces.append(cons[:, -w:].mean(axis=1, keepdims=True))
            pieces.append(cons[:, -w:].std(axis=1, keepdims=True))
        pieces.append(cons.min(axis=1, keepdims=True))
        pieces.append(cons.max(axis=1, keepdims=True))
        # Среднее по каждому каналу истории за последние сутки.
        pieces.append(hist[:, -24:, :].mean(axis=1))
        hist_part = np.concatenate(pieces, axis=1)
    else:
        hist_part = hist.reshape(len(hist), -1)

    return np.concatenate([hist_part, future.reshape(len(future), -1), static],
                          axis=1).astype(np.float32)


class PanelSklearnModel:
    """
    Обёртка для sklearn-совместимых моделей на panel-данных.

    Отдельная модель на каждый шаг горизонта: прогноз на час вперёд и на сутки
    опирается на разные признаки, и одна общая регрессия усредняет эти режимы.
    """

    def __init__(self, name: str, factory, aggregate_history: bool = False,
                 per_horizon: bool = True):
        self.name = name
        self.factory = factory
        self.aggregate_history = aggregate_history
        self.per_horizon = per_horizon
        self.models: List[Any] = []

    def _batch(self, data: Dict[str, Any], split: str) -> Dict[str, np.ndarray]:
        return {"hist": data[f"X_hist_{split}"], "future": data[f"X_future_{split}"],
                "static": data[f"X_static_{split}"], "series": data[f"series_{split}"],
                "feature_names_hist": data["feature_names_hist"],
                "feature_names_future": data["feature_names_future"]}

    def fit(self, data: Dict[str, Any], **_) -> "PanelSklearnModel":
        X = flatten_panel_inputs(self._batch(data, "train"), self.aggregate_history)
        Y = data["Y_train"]
        X_val = flatten_panel_inputs(self._batch(data, "val"), self.aggregate_history)
        Y_val = data["Y_val"]

        self.models = []
        if self.per_horizon:
            for h in range(Y.shape[1]):
                m = self.factory(h)
                _fit_one(m, X, Y[:, h], X_val, Y_val[:, h])
                self.models.append(m)
        else:
            m = self.factory(0)
            _fit_one(m, X, Y, X_val, Y_val)
            self.models.append(m)

        logger.info("%s обучен | признаков=%d | моделей=%d",
                    self.name, X.shape[1], len(self.models))
        return self

    def predict(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
        X = flatten_panel_inputs(batch, self.aggregate_history)
        if self.per_horizon:
            return np.stack([m.predict(X) for m in self.models], axis=1).astype(np.float32)
        return np.asarray(self.models[0].predict(X), dtype=np.float32)


def _fit_one(model, X, y, X_val, y_val) -> None:
    """
    Обучает модель, передавая валидацию тем, кто её принимает.

    Диспетчеризация идёт по сигнатуре, а не по перехвату TypeError. Перехват
    неотличим от TypeError, возникшего ВНУТРИ уже начавшегося обучения, и тогда
    модель молча переобучается без валидации. Именно так Ridge оставался с
    первой alpha из сетки: его fit не принимает eval_set, вызов падал, запасной
    путь обучал без валидации, и подбор регуляризации не происходил вовсе.
    """
    params = inspect.signature(model.fit).parameters

    if "eval_set" in params:                       # xgboost, lightgbm
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    elif "X_val" in params:                        # собственные обёртки с отбором
        model.fit(X, y, X_val=X_val, Y_val=y_val)
    else:                                          # обычный sklearn-регрессор
        model.fit(X, y)


def build_panel_ridge(alphas=None):
    """Ridge с подбором регуляризации по валидации, один на весь горизонт."""
    from sklearn.linear_model import Ridge

    grid = list(alphas) if alphas is not None else list(np.logspace(-3, 5, 17))

    class _RidgeCV:
        """
        Перебор alpha по отложенной валидации.

        Готовый RidgeCV из sklearn здесь не подходит: он отбирает по
        перекрёстной проверке со случайным разбиением, а на временном ряде это
        обучение на будущем. Нужен именно хронологический отложенный отрезок.
        """

        def __init__(self):
            self.model = None
            self.alpha_ = None
            self.val_mae_ = None

        def fit(self, X, Y, X_val=None, Y_val=None):
            if X_val is None or Y_val is None:
                raise ValueError(
                    "_RidgeCV требует валидационную выборку: без неё отбор alpha "
                    "не выполняется и модель молча остаётся с первым значением сетки."
                )

            best = (np.inf, None, None)
            for a in grid:
                m = Ridge(alpha=a).fit(X, Y)
                score = float(np.mean(np.abs(Y_val - m.predict(X_val))))
                if score < best[0]:
                    best = (score, a, m)
            self.val_mae_, self.alpha_, self.model = best

            # Оптимум на краю сетки означает, что она не покрывает нужный
            # диапазон, и выбранное значение упирается в границу перебора.
            if self.alpha_ in (grid[0], grid[-1]) and len(grid) > 1:
                logger.warning(
                    "Panel Ridge: alpha=%.4g на границе сетки [%.4g, %.4g] — "
                    "оптимум может лежать за её пределами",
                    self.alpha_, grid[0], grid[-1])

            logger.info("Panel Ridge: alpha=%.4g (MAE_val=%.5f, перебрано %d значений)",
                        self.alpha_, self.val_mae_, len(grid))
            return self

        def predict(self, X):
            return self.model.predict(X)

    return PanelSklearnModel("Ridge", lambda h: _RidgeCV(),
                             aggregate_history=False, per_horizon=False)


def build_panel_xgboost(n_estimators: int = 200, max_depth: int = 6,
                        learning_rate: float = 0.06, seed: int = 42):
    """Градиентный бустинг на агрегированных признаках, по модели на шаг."""
    import xgboost as xgb

    def factory(h: int):
        return xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.6,
            min_child_weight=10, reg_alpha=0.1, reg_lambda=2.0,
            random_state=seed + h, n_jobs=4, verbosity=0,
            tree_method="hist", early_stopping_rounds=30,
        )

    return PanelSklearnModel("XGBoost", factory,
                             aggregate_history=True, per_horizon=True)
