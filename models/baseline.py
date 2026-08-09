# -*- coding: utf-8 -*-
"""
models/baseline.py — Базовые модели сравнения.

Три уровня базлайнов, от простого к сложному:

  1. НАИВНЫЕ (без обучения) — обязательная точка отсчёта для временных рядов.
     Persistence24  : прогноз = потребление тех же часов предыдущих суток.
     SeasonalNaive168: прогноз = потребление тех же часов неделю назад.
     HourlyProfile  : среднее по (час × день недели), посчитанное на train.
     Если сложная модель не бьёт эти базлайны — она не имеет прогностической
     ценности, каким бы низким ни был её MAE в абсолютных единицах.

  2. ЛИНЕЙНАЯ  — Ridge на полном развёрнутом окне (T×F признаков).
  3. АНСАМБЛЕВАЯ — XGBoost на агрегированных лаг-признаках, отдельная модель
     на каждый шаг горизонта.

О составе лаг-признаков XGBoost (~165 при T=48, F=26): rolling mean/std
считаются по ВСЕМ каналам, а не только по потреблению. Это уравнивает
информационный контекст с Ridge, который видит сырое окно целиком, — иначе
сравнение алгоритмов подменяется сравнением объёма входных данных.
История изменений — в CHANGELOG.md.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import xgboost as xgb

logger = logging.getLogger("smart_grid.models.baseline")

# ── Индексы каналов в матрице признаков (см. data/preprocessing.py) ───────────
CH_CONSUMPTION = 0
CH_HOUR_SIN = 1
CH_HOUR_COS = 2
CH_DOW_SIN = 18
CH_DOW_COS = 19


# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ПОСТРОЕНИЯ ЛАГ-ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════

def build_lag_features(X: np.ndarray) -> np.ndarray:
    """
    Преобразует 3D окно (N, T, F) в 2D матрицу признаков (N, ~165).

    v5 vs v4: добавлены rolling_mean(6,12,24,48) и rolling_std(6,24)
    ПО ВСЕМ F КАНАЛАМ — выравнивает информационный контекст с LinearRegression.

    LinearRegression получает 48×15=720 сырых значений (весь тензор).
    XGBoost v5 получает 165 агрегированных — сопоставимая информационная
    ёмкость при лучшей инвариантности к шуму.

    Состав (при T=48, F=15):
      ЧАСТЬ 1 — потребление (канал 0), всё окно:
        48 лагов + 4 rolling mean(3,6,12,24) + 2 rolling std(6,24)
        + 3 (min/max/range 24ч) + 4 тренда = 61 признак

      ЧАСТЬ 2 — ковариаты последнего шага:
        F-1 = 14 признаков (каналы 1..F-1)

      ЧАСТЬ 3 ★ NEW — rolling_mean(6,12,24,48) × F каналов = 4F признаков
      ЧАСТЬ 4 ★ NEW — rolling_std(6,24) × F каналов = 2F признаков

    Итого: 61 + 14 + 60 + 30 = 165 признаков (при F=15).

    Примечание: rolling по каналу 0 в частях 3–4 дублирует часть 1,
    но XGBoost через feature importance самостоятельно занулит дубликаты.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, F)

    Returns
    -------
    features : np.ndarray, shape (N, 61 + (F-1) + 6*F)
    """
    N, T, F = X.shape
    cons = X[:, :, 0]   # (N, T) — нормализованное потребление

    feature_list = []

    # ── ЧАСТЬ 1: Потребление (канал 0) ────────────────────────────────────────

    # 1a. Все T лагов потребления
    for i in range(T):
        feature_list.append(cons[:, i])

    # 1b. Скользящие агрегаты потребления
    for window in [3, 6, 12, 24]:
        w = min(window, T)
        feature_list.append(cons[:, -w:].mean(axis=1))
    for window in [6, 24]:
        w = min(window, T)
        feature_list.append(cons[:, -w:].std(axis=1))

    # 1c. Min / max / range за 24ч
    w24 = min(24, T)
    cons_24 = cons[:, -w24:]
    feature_list.append(cons_24.min(axis=1))
    feature_list.append(cons_24.max(axis=1))
    feature_list.append(cons_24.max(axis=1) - cons_24.min(axis=1))

    # 1d. Трендовые дельты
    feature_list.append(cons[:, -1] - cons[:, -2]  if T >= 2  else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] - cons[:, -4]  if T >= 4  else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] - cons[:, -25] if T >= 25 else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] / (cons[:, -25] + 1e-8) if T >= 25 else np.ones(N, np.float32))

    # ── ЧАСТЬ 2: Ковариаты последнего шага (каналы 1..F-1) ────────────────────
    for ch in range(1, F):
        feature_list.append(X[:, -1, ch])

    # ── ЧАСТЬ 3: rolling_mean(6,12,24,48) по ВСЕМ F каналам ★ ────────────────
    # Физический смысл:
    #   channel=0 (cons):       среднее потребление за w часов
    #   channel=7 (temp):       средняя температура за w часов
    #   channel=11 (humidity):  средняя влажность за w часов
    #   channel=12 (wind):      средний ветер за w часов
    #   channel=3/4 (is_peak):  доля пиковых часов за w часов
    # LinearRegression видит эти значения напрямую в сыром окне.
    # Здесь мы даём XGBoost агрегированный эквивалент.
    for window in [6, 12, 24, 48]:
        w = min(window, T)
        for ch in range(F):
            feature_list.append(X[:, -w:, ch].mean(axis=1))

    # ── ЧАСТЬ 4: rolling_std(6,24) по ВСЕМ F каналам ★ ──────────────────────
    # Std температуры за 6ч = вариабельность погоды в ближайшие часы.
    # Std потребления за 24ч = амплитуда суточного профиля.
    for window in [6, 24]:
        w = min(window, T)
        for ch in range(F):
            feature_list.append(X[:, -w:, ch].std(axis=1))

    result = np.stack(feature_list, axis=1).astype(np.float32)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ОБЁРТКИ
# ══════════════════════════════════════════════════════════════════════════════

class LagFeaturesWrapper:
    """
    Обёртка для XGBoost: 3D (N,T,F) → лаг-признаки (N,~31) → fit/predict.
    Использует MultiOutputRegressor для независимого обучения по каждому шагу.
    """

    def __init__(self, estimator: BaseEstimator, name: str = "") -> None:
        self.estimator = estimator
        self.name = name or type(estimator).__name__

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "LagFeaturesWrapper":
        X2d = build_lag_features(X_train)
        if X_val is not None and Y_val is not None:
            X_val_2d = build_lag_features(X_val)
            self.estimator.fit(X2d, Y_train, X_val=X_val_2d, Y_val=Y_val)
        else:
            self.estimator.fit(X2d, Y_train)
        logger.info(
            "%s обучен | lag-features: %d | Y: %s",
            self.name, X2d.shape[1], str(Y_train.shape),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2d = build_lag_features(X)
        return self.estimator.predict(X2d)

    def __repr__(self) -> str:
        return f"LagFeaturesWrapper({self.name})"


class FlattenWrapper:
    """
    Разворачивает 3D тензоры (N, T, F) → 2D (N, T×F) для sklearn.
    Используется для LinearRegression (Ridge работает хорошо на полном окне).
    """

    def __init__(self, estimator: BaseEstimator, name: str = "") -> None:
        self.estimator = estimator
        self.name = name or type(estimator).__name__

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "FlattenWrapper":
        X2d = X_train.reshape(X_train.shape[0], -1)
        # Валидация пробрасывается в оценщик: она нужна для подбора
        # регуляризации так же, как ранняя остановка нужна остальным моделям.
        if X_val is not None and Y_val is not None:
            self.estimator.fit(X2d, Y_train,
                               X_val=X_val.reshape(X_val.shape[0], -1), Y_val=Y_val)
        else:
            self.estimator.fit(X2d, Y_train)
        logger.info("%s обучен | X_flat: %d фич", self.name, X2d.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X2d = X.reshape(X.shape[0], -1)
        return self.estimator.predict(X2d)

    def __repr__(self) -> str:
        return f"FlattenWrapper({self.name})"


# ══════════════════════════════════════════════════════════════════════════════
# ПУБЛИЧНЫЙ API
# ══════════════════════════════════════════════════════════════════════════════

class RidgeValidationAlpha:
    """
    Ridge с выбором коэффициента регуляризации по ВАЛИДАЦИОННОЙ выборке.

    Фиксированная alpha ставила линейную модель в неравные условия: у бустинга
    есть ранняя остановка и собственная регуляризация, у сетей — dropout и
    ранняя остановка, а Ridge обучался с произвольным значением. На развёрнутом
    окне признаков в разы больше, чем нужно, и при слабой регуляризации модель
    переобучается — это видно по немонотонной деградации ошибки с горизонтом.

    Перебор идёт по логарифмической сетке, alpha выбирается по MAE на
    валидации. Встроенный RidgeCV здесь не годится: он использует
    перекрёстную проверку со случайным разбиением, что для временного ряда
    означает подглядывание в будущее.
    """

    def __init__(self, alphas=None) -> None:
        # Сетка охватывает семь порядков: на больших выборках оптимум уходит к
        # слабой регуляризации, на малых — к сильной. Прежний нижний край 0.1
        # достигался при 12 тыс. обучающих окон, то есть выбор упирался в
        # границу и истинный оптимум мог лежать за ней.
        self.alphas = list(alphas) if alphas is not None else list(np.logspace(-3, 5, 17))
        self.alpha_: Optional[float] = None
        self.model_: Optional[Ridge] = None

    def fit(self, X2d, Y, X_val=None, Y_val=None) -> "RidgeValidationAlpha":
        if X_val is None or Y_val is None:
            self.alpha_ = float(np.median(self.alphas))
            self.model_ = Ridge(alpha=self.alpha_).fit(X2d, Y)
            logger.warning(
                "Ridge: валидация не передана, alpha взята медианной (%.3g)", self.alpha_)
            return self

        best = (np.inf, None, None)
        for a in self.alphas:
            m = Ridge(alpha=a).fit(X2d, Y)
            mae = float(np.mean(np.abs(Y_val - m.predict(X_val))))
            if mae < best[0]:
                best = (mae, a, m)
        _, self.alpha_, self.model_ = best
        logger.info("Ridge: alpha=%.4g выбрана по валидации из %d значений (MAE_val=%.4f)",
                    self.alpha_, len(self.alphas), best[0])
        return self

    def predict(self, X2d):
        if self.model_ is None:
            raise RuntimeError("Модель не обучена: вызовите fit() перед predict().")
        return self.model_.predict(X2d)


def build_linear_regression(alphas=None) -> FlattenWrapper:
    """
    Ridge на полном развёрнутом окне признаков с подбором alpha по валидации.

    Ridge нечувствителен к мультиколлинеарности, поэтому сырое окно ему
    подходит; критичен только уровень регуляризации.
    """
    return FlattenWrapper(RidgeValidationAlpha(alphas), name="LinearRegression")


class MultiHorizonXGB:
    """
    Отдельный XGBRegressor на каждый шаг горизонта прогноза.

    Класс объявлен на уровне модуля, а не внутри фабрики: локальные классы не
    поддаются сериализации через pickle, из-за чего обученную модель было
    невозможно сохранить в инференс-бандл. Гиперпараметры хранятся в атрибутах
    экземпляра, поэтому объект восстанавливается вместе с ними.

    Ранняя остановка по валидационной выборке ограничивает переобучение
    каждой из моделей горизонта независимо.
    """

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.75,
        colsample_bytree: float = 0.80,
        min_child_weight: int = 10,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.seed = seed
        self.models: List[xgb.XGBRegressor] = []
        self.horizon: Optional[int] = None

    def _make_estimator(self, random_state_offset: int = 0) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=self.seed + random_state_offset,
            n_jobs=1,
            verbosity=0,
            tree_method="hist",
            early_stopping_rounds=50,
        )

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "MultiHorizonXGB":
        self.horizon = Y_train.shape[1]
        self.models = []
        use_val = X_val is not None and Y_val is not None
        for h in range(self.horizon):
            model = self._make_estimator(random_state_offset=h)
            if use_val:
                model.fit(
                    X_train, Y_train[:, h],
                    eval_set=[(X_val, Y_val[:, h])],
                    verbose=False,
                )
            else:
                model.fit(X_train, Y_train[:, h], verbose=False)
            self.models.append(model)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Модель не обучена: вызовите fit() перед predict().")
        preds = [m.predict(X) for m in self.models]
        return np.stack(preds, axis=1).astype(np.float32)


def build_xgboost(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.75,
    colsample_bytree: float = 0.80,
    min_child_weight: int = 10,
    seed: int = 42,
) -> LagFeaturesWrapper:
    """
    XGBoost на расширенных лаг-признаках (~165 фич), отдельная модель на каждый
    шаг горизонта.

    Состав признаков включает rolling mean/std по всем каналам, а не только по
    потреблению: иначе сравнение с Ridge, который видит сырое окно целиком,
    подменялось бы сравнением объёма входных данных.
    """
    estimator = MultiHorizonXGB(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        seed=seed,
    )
    logger.info(
        "XGBoost (FullLags + RollingAllChannels + MultiHorizon) | "
        "~165 признаков | отдельная модель на шаг горизонта + val-early-stopping | "
        "n_est=%d depth=%d min_cw=%d",
        n_estimators, max_depth, min_child_weight,
    )
    return LagFeaturesWrapper(estimator, name="XGBoost")


# ══════════════════════════════════════════════════════════════════════════════
# НАИВНЫЕ БАЗЛАЙНЫ
# ══════════════════════════════════════════════════════════════════════════════
#
# Все три работают в НОРМИРОВАННОМ пространстве (как и остальные модели):
# на вход подаётся X формы (N, T, F), где канал 0 — масштабированное
# потребление, на выходе — (N, H) в том же масштабе. Это позволяет прогонять их
# через тот же ModelTrainer и сравнивать в одной таблице без оговорок.


class _SeasonalNaive:
    """
    Сезонно-наивный прогноз: ŷ[t+j] = y[t+j-m].

    Значение «m часов назад» для целевого шага j лежит в окне истории на
    позиции T + j - m. Условие доступности: T >= m (окно должно покрывать
    целый сезонный период). Иначе автоматический откат на m=24.

    m=24  — прогноз «как вчера в этот же час» (классический базлайн нагрузки);
    m=168 — «как неделю назад», учитывает различие будни/выходные.
    """

    def __init__(self, season_length: int = 24, name: str = "") -> None:
        self.season_length = int(season_length)
        self.effective_season = self.season_length
        self.horizon: Optional[int] = None
        self.name = name or f"Naive{season_length}"

    def fit(self, X_train, Y_train, X_val=None, Y_val=None) -> "_SeasonalNaive":
        _, T, _ = X_train.shape
        H = Y_train.shape[1]
        self.horizon = int(H)
        if T < self.season_length:
            fallback = 24 if T >= 24 else T
            logger.warning(
                "%s: длина истории T=%d < периода m=%d → откат на m=%d. "
                "Для корректного недельного базлайна нужен HISTORY_LENGTH >= 168.",
                self.name, T, self.season_length, fallback,
            )
            self.effective_season = fallback
        else:
            self.effective_season = self.season_length

        if H > self.effective_season:
            logger.warning(
                "%s: горизонт H=%d больше периода m=%d — часть шагов повторит "
                "начало того же периода.", self.name, H, self.effective_season,
            )
        logger.info("%s обучен (обучение не требуется) | m=%d ч",
                    self.name, self.effective_season)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.horizon is None:
            raise RuntimeError("Модель не обучена: вызовите fit() перед predict().")
        _, T, _ = X.shape
        m = self.effective_season
        start = T - m
        idx = [(start + j) % T for j in range(self.horizon)]
        return X[:, idx, CH_CONSUMPTION].astype(np.float32)


class _HourlyProfile:
    """
    Климатологический базлайн: среднее потребление по (час × день недели),
    вычисленное на обучающей выборке.

    Час и день недели восстанавливаются из циклических признаков окна
    (hour_sin/hour_cos, dow_sin/dow_cos) — отдельный календарь не нужен.
    Это «сезонный профиль без динамики»: он не знает текущего уровня нагрузки
    и потому показывает, какую долю дисперсии объясняет один лишь календарь.
    """

    def __init__(self, name: str = "HourlyProfile") -> None:
        self.name = name
        self.profile: Optional[np.ndarray] = None   # (7, 24)
        self.global_mean: float = 0.0
        self.horizon: Optional[int] = None

    @staticmethod
    def _decode_hour(x_step: np.ndarray) -> np.ndarray:
        """Восстанавливает час [0..23] из (hour_sin, hour_cos)."""
        ang = np.arctan2(x_step[:, CH_HOUR_SIN], x_step[:, CH_HOUR_COS])
        return np.mod(np.round(ang / (2 * np.pi) * 24), 24).astype(int)

    @staticmethod
    def _decode_weekday(x_step: np.ndarray) -> np.ndarray:
        """Восстанавливает день недели [0..6] из (dow_sin, dow_cos)."""
        ang = np.arctan2(x_step[:, CH_DOW_SIN], x_step[:, CH_DOW_COS])
        return np.mod(np.round(ang / (2 * np.pi) * 7), 7).astype(int)

    def fit(self, X_train, Y_train, X_val=None, Y_val=None) -> "_HourlyProfile":
        N, T, _ = X_train.shape
        self.horizon = int(Y_train.shape[1])
        # Разворачиваем окна в набор наблюдений (час, день недели, потребление).
        sums = np.zeros((7, 24), dtype=np.float64)
        counts = np.zeros((7, 24), dtype=np.float64)

        for step in range(T):
            xs = X_train[:, step, :]
            hours = self._decode_hour(xs)
            wdays = self._decode_weekday(xs)
            vals = X_train[:, step, CH_CONSUMPTION].astype(np.float64)
            np.add.at(sums, (wdays, hours), vals)
            np.add.at(counts, (wdays, hours), 1.0)

        self.global_mean = float(X_train[:, :, CH_CONSUMPTION].mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            profile = np.where(counts > 0, sums / np.maximum(counts, 1e-9), self.global_mean)
        self.profile = profile.astype(np.float32)

        empty = int((counts == 0).sum())
        logger.info(
            "%s обучен | профиль 7×24, пустых ячеек=%d, среднее=%.4f",
            self.name, empty, self.global_mean,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.profile is None or self.horizon is None:
            raise RuntimeError("Модель не обучена: вызовите fit() перед predict().")
        H = self.horizon
        last = X[:, -1, :]
        h_last = self._decode_hour(last)
        w_last = self._decode_weekday(last)

        preds = np.empty((X.shape[0], H), dtype=np.float32)
        for j in range(H):
            # Целевой шаг j отстоит от последнего шага окна на (j+1) часов.
            hour = np.mod(h_last + j + 1, 24)
            day_shift = (h_last + j + 1) // 24
            wday = np.mod(w_last + day_shift, 7)
            preds[:, j] = self.profile[wday, hour]
        return preds


def build_persistence_24() -> _SeasonalNaive:
    """Наивный суточный базлайн: «завтра как вчера в тот же час»."""
    return _SeasonalNaive(season_length=24, name="Naive24 (сутки)")


def build_seasonal_naive_168() -> _SeasonalNaive:
    """Наивный недельный базлайн: «как в тот же час неделю назад»."""
    return _SeasonalNaive(season_length=168, name="Naive168 (неделя)")


def build_hourly_profile() -> _HourlyProfile:
    """Климатологический базлайн: среднее по (час × день недели) на train."""
    return _HourlyProfile()