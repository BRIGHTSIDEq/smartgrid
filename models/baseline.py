# -*- coding: utf-8 -*-
"""
models/baseline.py — Базовые модели: XGBoost, LinearRegression, SARIMA.

ИЗМЕНЕНИЯ v6:
  + SARIMAWrapper — SARIMA(1,1,1)(1,1,1,24) как классическая статистическая
    базовая линия. Стандарт в академических работах по энергопотреблению.
  + get_feature_importance() в LagFeaturesWrapper — для анализа важности
    признаков XGBoost (feature_importances_ + визуализация).
  + XGBoost: выравнивание информационного контекста с LinearRegression (v5).
"""

import logging
import warnings
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
import xgboost as xgb

logger = logging.getLogger("smart_grid.models.baseline")


# ══════════════════════════════════════════════════════════════════════════════
# SARIMA WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class SARIMAWrapper:
    """
    SARIMA(1,1,1)(1,1,1,24) — классическая статистическая базовая линия.

    Используется как академический benchmark: если нейросети и XGBoost
    не превосходят SARIMA, это сигнал о проблемах с данными или моделью.
    В задачах почасового энергопотребления SARIMA(1,1,1)(1,1,1,24) —
    стандарт (Box & Jenkins, 1976; Taylor, 2003).

    Стратегия прогноза:
      - Обучается на train-выборке (raw кВт·ч, без нормализации).
      - Прогноз: скользящее окно — для каждого тестового окна переобучается
        на последних refit_window точках + делает predict(horizon).
      - При refit=False использует одну обученную модель для всех окон
        (быстрее, но хуже на нестационарных данных).

    Parameters
    ----------
    order           : tuple (p,d,q) — ARIMA порядок
    seasonal_order  : tuple (P,D,Q,m) — сезонный порядок
    refit           : bool — переобучаться на каждом тестовом окне
    refit_window    : int — сколько точек истории использовать при refit
    """

    def __init__(
        self,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 24),
        refit: bool = False,
        refit_window: int = 24 * 30,
    ):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.refit          = refit
        self.refit_window   = refit_window
        self._fitted_model  = None
        self._train_series: Optional[np.ndarray] = None
        self.name = "SARIMA"

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        raw_series: Optional[np.ndarray] = None,
    ) -> "SARIMAWrapper":
        """
        Обучает SARIMA на сырых значениях потребления.

        Parameters
        ----------
        X_train    : не используется (оставлен для совместимости интерфейса)
        Y_train    : не используется напрямую
        raw_series : np.ndarray — сырой ряд кВт·ч (train split).
                     Если None — извлекается из X_train[:, -1, 0] как прокси.
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels >= 0.14 требуется для SARIMA. "
                              "Установите: pip install statsmodels")

        if raw_series is not None:
            series = np.asarray(raw_series, dtype=np.float64)
        else:
            # Прокси: берём последний шаг окна как приблизительный ряд
            series = X_train[:, -1, 0].astype(np.float64)

        self._train_series = series

        logger.info(
            "SARIMA(%d,%d,%d)(%d,%d,%d,%d) обучается на %d точках...",
            *self.order, *self.seasonal_order, len(series)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._fitted_model = model.fit(disp=False, maxiter=200)

        aic = self._fitted_model.aic
        logger.info("SARIMA обучена: AIC=%.2f", aic)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Строит прогноз на горизонт=24ч для каждого входного окна.
 
        ИСПРАВЛЕНИЕ v2 (было: forecast(24) вызывался N раз с одной точки):
          Statsmodels forecast() каждый раз стартует с конца обучающего ряда,
          поэтому все N окон получали идентичный прогноз → MAPE=40%, R²=-2.35.
 
          Теперь: один длинный forecast(history+N+23), нарезается на окна:
            окно i = long_forecast[history+i : history+i+horizon]
 
        Требование к fit(): передавать raw_series = train+val, чтобы конец
        обучающего ряда совпадал с началом тестового периода.
        """
        if self._fitted_model is None:
            raise RuntimeError("SARIMAWrapper: вызовите fit() перед predict()")
 
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            raise ImportError("statsmodels >= 0.14 требуется для SARIMA.")
 
        N            = len(X)
        horizon      = 24
        history_len  = int(X.shape[1]) if X.ndim == 3 else 192
        predictions  = np.zeros((N, horizon), dtype=np.float32)
 
        if not self.refit:
            # ── Исправленный быстрый режим ────────────────────────────────────
            # Один длинный прогноз вместо N вызовов forecast(24).
            # Окно i соответствует временному отрезку:
            #   [fit_end + history_len + i .. fit_end + history_len + i + horizon]
            total_steps = history_len + N + horizon - 1
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    long_fc = self._fitted_model.forecast(steps=total_steps)
 
                for i in range(N):
                    start = history_len + i
                    predictions[i] = np.maximum(
                        long_fc[start : start + horizon], 0
                    ).astype(np.float32)
 
            except Exception as exc:
                logger.warning("SARIMA forecast(long) ошибка: %s", exc)
                if self._train_series is not None:
                    predictions[:] = float(self._train_series[-1])
 
        else:
            # ── Refit-режим (без изменений) ───────────────────────────────────
            if self._train_series is None:
                raise RuntimeError("raw_series не передана в fit()")
 
            total_train = len(self._train_series)
            for i in range(N):
                window_start = max(0, total_train + i - self.refit_window)
                history = self._train_series[window_start : total_train + i]
                if len(history) < 48:
                    history = self._train_series[-self.refit_window:]
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(
                            history,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False, maxiter=100)
                    predictions[i] = np.maximum(
                        m.forecast(steps=horizon), 0
                    ).astype(np.float32)
                except Exception as exc:
                    logger.debug("SARIMA refit ошибка окно %d: %s", i, exc)
                    predictions[i] = float(history[-1]) * np.ones(horizon)
 
                if i % 100 == 0:
                    logger.info("SARIMA predict: %d/%d окон", i, N)
 
        return predictions

    def __repr__(self) -> str:
        return (f"SARIMA({self.order})({self.seasonal_order}) "
                f"refit={self.refit}")


def build_sarima(
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 24),
    refit: bool = False,
) -> SARIMAWrapper:
    """
    Создаёт SARIMA(1,1,1)(1,1,1,24) — стандарт для почасового энергопотребления.

    Параметры обоснованы:
      p=1, d=1, q=1  — AR(1) + разность + MA(1) устраняют нестационарность
      P=1, D=1, Q=1  — сезонный аналог для периода m=24ч
      m=24           — суточная сезонность

    References
    ----------
    Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis.
    Taylor, J.W. (2003). Short-term electricity demand forecasting
      using double seasonal exponential smoothing. Journal of OR Society.
    """
    logger.info(
        "SARIMA(%d,%d,%d)(%d,%d,%d,%d) инициализирована | refit=%s",
        *order, *seasonal_order, refit
    )
    return SARIMAWrapper(order=order, seasonal_order=seasonal_order, refit=refit)


# ══════════════════════════════════════════════════════════════════════════════
# ФУНКЦИЯ ПОСТРОЕНИЯ ЛАГ-ПРИЗНАКОВ (v5, без изменений)
# ══════════════════════════════════════════════════════════════════════════════

def build_lag_features(X: np.ndarray) -> np.ndarray:
    """
    Преобразует 3D окно (N, T, F) в 2D матрицу признаков (N, ~165).
    v5: rolling_mean/std по всем F каналам для параметрического паритета с Ridge.
    """
    N, T, F = X.shape
    cons = X[:, :, 0]

    feature_list = []

    # Часть 1: потребление (канал 0)
    for i in range(T):
        feature_list.append(cons[:, i])
    for window in [3, 6, 12, 24]:
        w = min(window, T)
        feature_list.append(cons[:, -w:].mean(axis=1))
    for window in [6, 24]:
        w = min(window, T)
        feature_list.append(cons[:, -w:].std(axis=1))
    w24 = min(24, T)
    cons_24 = cons[:, -w24:]
    feature_list.append(cons_24.min(axis=1))
    feature_list.append(cons_24.max(axis=1))
    feature_list.append(cons_24.max(axis=1) - cons_24.min(axis=1))
    feature_list.append(cons[:, -1] - cons[:, -2]  if T >= 2  else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] - cons[:, -4]  if T >= 4  else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] - cons[:, -25] if T >= 25 else np.zeros(N, np.float32))
    feature_list.append(cons[:, -1] / (cons[:, -25] + 1e-8) if T >= 25 else np.ones(N, np.float32))

    # Часть 2: ковариаты последнего шага
    for ch in range(1, F):
        feature_list.append(X[:, -1, ch])

    # Части 3-4: rolling по всем каналам
    for window in [6, 12, 24, 48]:
        w = min(window, T)
        for ch in range(F):
            feature_list.append(X[:, -w:, ch].mean(axis=1))
    for window in [6, 24]:
        w = min(window, T)
        for ch in range(F):
            feature_list.append(X[:, -w:, ch].std(axis=1))

    result = np.stack(feature_list, axis=1).astype(np.float32)
    return result


def build_feature_names(n_features: int = 26, history_length: int = 48) -> List[str]:
    """
    Строит список имён признаков для feature importance анализа.
    Порядок должен совпадать с build_lag_features().
    """
    from data.preprocessing import FEATURE_NAMES
    names = []
    T = history_length
    F = n_features

    # Лаги потребления
    for i in range(T):
        names.append(f"cons_lag_{T-i}h")

    # Rolling consumption
    for w in [3, 6, 12, 24]:
        names.append(f"cons_mean_{min(w,T)}h")
    for w in [6, 24]:
        names.append(f"cons_std_{min(w,T)}h")

    # Min/max/range
    names += ["cons_min_24h", "cons_max_24h", "cons_range_24h"]

    # Тренды
    names += ["delta_1h", "delta_3h", "delta_24h", "ratio_24h"]

    # Ковариаты последнего шага
    feat_names = FEATURE_NAMES if len(FEATURE_NAMES) == F else [f"feat_{i}" for i in range(F)]
    for ch in range(1, F):
        names.append(f"last_{feat_names[ch]}")

    # Rolling по всем каналам
    for w in [6, 12, 24, 48]:
        for ch in range(F):
            fn = feat_names[ch] if ch < len(feat_names) else f"feat_{ch}"
            names.append(f"mean{min(w,T)}h_{fn}")

    for w in [6, 24]:
        for ch in range(F):
            fn = feat_names[ch] if ch < len(feat_names) else f"feat_{ch}"
            names.append(f"std{min(w,T)}h_{fn}")

    return names


# ══════════════════════════════════════════════════════════════════════════════
# ОБЁРТКИ
# ══════════════════════════════════════════════════════════════════════════════

class LagFeaturesWrapper:
    """
    Обёртка для XGBoost: 3D (N,T,F) → лаг-признаки → fit/predict.
    v6: добавлен get_feature_importance() для анализа важности признаков.
    """

    def __init__(self, estimator: BaseEstimator, name: str = "") -> None:
        self.estimator = estimator
        self.name = name or type(estimator).__name__
        self._feature_names: Optional[List[str]] = None
        self._n_features_in: int = 26
        self._history_length: int = 48

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
    ) -> "LagFeaturesWrapper":
        self._history_length = X_train.shape[1]
        self._n_features_in  = X_train.shape[2]
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

    def get_feature_importance(
        self,
        top_n: int = 30,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Возвращает важность признаков XGBoost усреднённую по 24 горизонтам.

        Returns
        -------
        dict {
            "names":       np.ndarray имён признаков (top_n),
            "importances": np.ndarray важностей (top_n),
            "all_names":   полный список имён,
            "all_importances": полный массив важностей,
        }
        или None если estimator не поддерживает feature_importances_.
        """
        inner = self.estimator
        # MultiHorizonXGB хранит список моделей
        models_list = getattr(inner, "models", None)
        if models_list is None or len(models_list) == 0:
            logger.warning("get_feature_importance: модели не обучены")
            return None

        # Усредняем важность по всем 24 горизонтам
        all_imp = []
        for m in models_list:
            fi = getattr(m, "feature_importances_", None)
            if fi is not None:
                all_imp.append(fi)

        if not all_imp:
            logger.warning("get_feature_importance: feature_importances_ не найден")
            return None

        mean_imp = np.mean(all_imp, axis=0)

        # Строим имена признаков
        try:
            feat_names = build_feature_names(
                n_features=self._n_features_in,
                history_length=self._history_length,
            )
        except Exception:
            feat_names = [f"feat_{i}" for i in range(len(mean_imp))]

        # Выравниваем длину
        n = min(len(mean_imp), len(feat_names))
        mean_imp  = mean_imp[:n]
        feat_names = feat_names[:n]

        # Сортируем по убыванию
        idx_sorted = np.argsort(mean_imp)[::-1]
        top_idx    = idx_sorted[:top_n]

        result = {
            "names":            np.array(feat_names)[top_idx],
            "importances":      mean_imp[top_idx],
            "all_names":        np.array(feat_names),
            "all_importances":  mean_imp,
        }
        logger.info(
            "Feature importance: топ-5: %s",
            ", ".join(f"{n}={v:.4f}"
                      for n, v in zip(result["names"][:5], result["importances"][:5]))
        )
        return result

    def __repr__(self) -> str:
        return f"LagFeaturesWrapper({self.name})"


class FlattenWrapper:
    """
    Разворачивает 3D тензоры (N, T, F) → 2D (N, T×F) для sklearn.
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

def build_linear_regression(alpha: float = 1.0) -> FlattenWrapper:
    return FlattenWrapper(Ridge(alpha=alpha), name="LinearRegression")


def build_xgboost(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.75,
    colsample_bytree: float = 0.80,
    min_child_weight: int = 10,
    seed: int = 42,
) -> LagFeaturesWrapper:
    """XGBoost на расширенных лаг-признаках (~165 фич) + MultiOutputRegressor."""

    class MultiHorizonXGB:
        def __init__(self) -> None:
            self.models: List[xgb.XGBRegressor] = []
            self.horizon: Optional[int] = None

        def _make_estimator(self, offset: int = 0) -> xgb.XGBRegressor:
            return xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=seed + offset,
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
            self.models  = []
            use_val = X_val is not None and Y_val is not None
            for h in range(self.horizon):
                m = self._make_estimator(offset=h)
                if use_val:
                    m.fit(X_train, Y_train[:, h],
                          eval_set=[(X_val, Y_val[:, h])], verbose=False)
                else:
                    m.fit(X_train, Y_train[:, h], verbose=False)
                self.models.append(m)
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            if not self.models:
                raise RuntimeError("Модель не обучена.")
            return np.stack([m.predict(X) for m in self.models],
                            axis=1).astype(np.float32)

    logger.info(
        "XGBoost v5 | ~165 признаков | 24 модели | "
        "n_est=%d depth=%d min_cw=%d",
        n_estimators, max_depth, min_child_weight,
    )
    return LagFeaturesWrapper(MultiHorizonXGB(), name="XGBoost")