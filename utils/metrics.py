# -*- coding: utf-8 -*-
"""
utils/metrics.py — Функции расчёта метрик качества прогноза.

Помимо базовых MAE/RMSE/MAPE/R² здесь реализованы:
  * MASE  — масштабно-независимая метрика относительно сезонно-наивного прогноза.
            MASE < 1 означает, что модель лучше наивного базлайна.
  * sMAPE — симметричная процентная ошибка (устойчивее MAPE к малым значениям).
  * Метрики по шагам горизонта (h = 1..H) — кривая деградации прогноза.
  * Тест Диболда–Мариано — статистическая значимость различия двух моделей.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("smart_grid.utils.metrics")


# ══════════════════════════════════════════════════════════════════════════════
# БАЗОВЫЕ МЕТРИКИ
# ══════════════════════════════════════════════════════════════════════════════

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.flatten() - y_pred.flatten())))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    t, p = y_true.flatten(), y_pred.flatten()
    return float(np.mean(np.abs((t - p) / (np.abs(t) + eps))) * 100)


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """sMAPE: 100 · mean( |t-p| / ((|t|+|p|)/2) ). Ограничена сверху 200%."""
    t, p = y_true.flatten(), y_pred.flatten()
    denom = (np.abs(t) + np.abs(p)) / 2.0 + eps
    return float(np.mean(np.abs(t - p) / denom) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    t, p = y_true.flatten(), y_pred.flatten()
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


# ══════════════════════════════════════════════════════════════════════════════
# MASE
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_naive_scale(y_train: np.ndarray, season_length: int = 24) -> float:
    """
    Знаменатель MASE: средняя абсолютная ошибка сезонно-наивного прогноза
    на ОБУЧАЮЩЕЙ выборке (Hyndman & Koehler, 2006).

    scale = mean( |y[t] - y[t-m]| ),  t = m..n

    Parameters
    ----------
    y_train : np.ndarray
        Непрерывный ряд потребления на train (в исходном масштабе).
    season_length : int
        Период сезонности m. Для почасовых данных — 24.
    """
    y = np.asarray(y_train, dtype=np.float64).flatten()
    if len(y) <= season_length:
        raise ValueError(
            f"Ряд длиной {len(y)} короче периода сезонности {season_length}"
        )
    return float(np.mean(np.abs(y[season_length:] - y[:-season_length])))


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """
    MASE = MAE(model) / scale, где scale — ошибка сезонно-наивного на train.

    Интерпретация:
      MASE < 1  — модель лучше сезонно-наивного прогноза;
      MASE = 1  — паритет;
      MASE > 1  — модель хуже, чем «повторить прошлые сутки».
    """
    if scale <= 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred) / scale)


# ══════════════════════════════════════════════════════════════════════════════
# СВОДНЫЙ РАСЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    mase_scale: Optional[float] = None,
) -> Dict[str, float]:
    """
    Вычисляет MAE, RMSE, MAPE, sMAPE, R² (и MASE, если передан mase_scale).

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Фактические и предсказанные значения в ИСХОДНОМ масштабе (кВт·ч).
    model_name : str
        Имя для логирования.
    mase_scale : float, optional
        Знаменатель MASE из `seasonal_naive_scale(raw_train)`.

    Returns
    -------
    dict {"MAE", "RMSE", "MAPE", "sMAPE", "R2"[, "MASE"]}
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "sMAPE": symmetric_mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
    if mase_scale is not None:
        metrics["MASE"] = mase(y_true, y_pred, mase_scale)

    if model_name:
        mase_str = f" | MASE={metrics['MASE']:6.4f}" if "MASE" in metrics else ""
        logger.info(
            "%-20s | MAE=%9.2f | RMSE=%9.2f | MAPE=%6.2f%% | R²=%7.4f%s",
            model_name,
            metrics["MAE"], metrics["RMSE"], metrics["MAPE"], metrics["R2"], mase_str,
        )
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# МЕТРИКИ ПО ШАГАМ ГОРИЗОНТА
# ══════════════════════════════════════════════════════════════════════════════

def metrics_by_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_scale: Optional[float] = None,
) -> Dict[str, List[float]]:
    """
    Метрики отдельно для каждого шага горизонта h = 1..H.

    Показывает, как деградирует прогноз с ростом горизонта — обязательный
    разрез для multi-horizon прогнозирования: усреднённый MAE скрывает то,
    что h=1 может быть отличным, а h=24 — бесполезным.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N, H)

    Returns
    -------
    dict {"h": [1..H], "MAE": [...], "RMSE": [...], "MAPE": [...], "R2": [...]}
    """
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    horizon = y_true.shape[1]

    out: Dict[str, List[float]] = {"h": [], "MAE": [], "RMSE": [], "MAPE": [], "R2": []}
    if mase_scale is not None:
        out["MASE"] = []

    for h in range(horizon):
        t, p = y_true[:, h], y_pred[:, h]
        out["h"].append(h + 1)
        out["MAE"].append(mean_absolute_error(t, p))
        out["RMSE"].append(root_mean_squared_error(t, p))
        out["MAPE"].append(mean_absolute_percentage_error(t, p))
        out["R2"].append(r2_score(t, p))
        if mase_scale is not None:
            out["MASE"].append(mase(t, p, mase_scale))
    return out


def metrics_by_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    """
    Метрики на подмножестве точек (например, только пиковые часы или только зима).

    Parameters
    ----------
    mask : np.ndarray of bool
        Та же форма, что y_true, либо плоский массив той же длины.
    """
    t = y_true.flatten()
    p = y_pred.flatten()
    m = np.asarray(mask).flatten().astype(bool)
    if m.sum() == 0:
        logger.warning("metrics_by_mask(%s): маска пуста, метрики не определены", label)
        return {"MAE": float("nan"), "RMSE": float("nan"),
                "MAPE": float("nan"), "R2": float("nan"), "n": 0}
    res = {
        "MAE": mean_absolute_error(t[m], p[m]),
        "RMSE": root_mean_squared_error(t[m], p[m]),
        "MAPE": mean_absolute_percentage_error(t[m], p[m]),
        "R2": r2_score(t[m], p[m]),
        "n": int(m.sum()),
    }
    return res


# ══════════════════════════════════════════════════════════════════════════════
# ТЕСТ ДИБОЛДА–МАРИАНО
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    h: int = 1,
    loss: str = "abs",
) -> Tuple[float, float]:
    """
    Тест Диболда–Мариано на равенство прогностической точности двух моделей.

    H0: модели A и B одинаково точны.
    DM < 0 → модель A точнее; DM > 0 → модель B точнее.

    Используется поправка Ньюи–Уэста на автокорреляцию ряда потерь до лага h-1,
    что обязательно для многошаговых прогнозов (остатки коррелированы).

    Parameters
    ----------
    y_true, y_pred_a, y_pred_b : np.ndarray
        Одинаковой формы. Сравнение идёт по плоскому ряду разностей потерь.
    h : int
        Горизонт прогноза (для окна Ньюи–Уэста). Для h=1 поправка не нужна.
    loss : {"abs", "sq"}
        Функция потерь: абсолютная (сопоставима с MAE) или квадратичная (RMSE).

    Returns
    -------
    (dm_stat, p_value) — двусторонний p-value по нормальному приближению.
    """
    from scipy import stats

    t = np.asarray(y_true, dtype=np.float64).flatten()
    a = np.asarray(y_pred_a, dtype=np.float64).flatten()
    b = np.asarray(y_pred_b, dtype=np.float64).flatten()

    if loss == "sq":
        e_a, e_b = (t - a) ** 2, (t - b) ** 2
    else:
        e_a, e_b = np.abs(t - a), np.abs(t - b)

    d = e_a - e_b
    n = len(d)
    d_mean = float(np.mean(d))

    # Долгосрочная дисперсия с поправкой Ньюи–Уэста на лаги 1..h-1
    gamma0 = float(np.mean((d - d_mean) ** 2))
    var_d = gamma0
    for lag in range(1, max(h, 1)):
        if lag >= n:
            break
        gamma = float(np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean)))
        var_d += 2.0 * (1.0 - lag / h) * gamma

    if var_d <= 0 or n == 0:
        return float("nan"), float("nan")

    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p_value


def to_non_overlapping(array: np.ndarray, horizon: int) -> np.ndarray:
    """
    Оставляет от матрицы окон (N, H) только непересекающиеся: каждое H-е.

    Окна нарезаны со сдвигом один час, поэтому в развёрнутом виде каждый
    физический час входит в ряд H раз, а соседние элементы относятся к РАЗНЫМ
    моментам времени. Тест Диболда–Мариано на таком векторе считает N·H
    независимых наблюдений вместо примерно N/H, и статистика завышается в
    несколько раз. Измерено на данных этого проекта: −32.4 против −6.85 на
    непересекающемся ряде, тогда как блочная оценка по суткам даёт −6.50.

    Тот же принцип уже применён в проекте к автокорреляции остатков: «лаг 24»
    по развёрнутому вектору равен одному часу реального времени, а не суткам.
    """
    values = np.asarray(array)
    if values.ndim != 2:
        return values.ravel()
    return values[::max(int(horizon), 1)].ravel()


def pairwise_dm_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    h: int = 24,
) -> List[Dict[str, object]]:
    """
    Попарный тест Диболда–Мариано для всех моделей.

    Матрицы окон прореживаются до непересекающихся: перекрытие завышало бы
    значимость в несколько раз (см. to_non_overlapping). Поправка
    Ньюи–Уэста на лаги 1..h−1 после прореживания снимает остаточную
    зависимость между соседними сутками.

    Returns
    -------
    Список словарей {"model_a", "model_b", "DM", "p_value", "better"}.
    """
    y_true = to_non_overlapping(y_true, h)
    predictions = {k: to_non_overlapping(v, h) for k, v in predictions.items()}
    names = list(predictions.keys())
    rows: List[Dict[str, object]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            dm, p = diebold_mariano(y_true, predictions[a], predictions[b], h=h)
            if np.isnan(dm):
                better = "н/д"
            elif p >= 0.05:
                better = "различие незначимо"
            else:
                better = a if dm < 0 else b
            rows.append({"model_a": a, "model_b": b,
                         "DM": round(dm, 4), "p_value": round(p, 6), "better": better})
    return rows
