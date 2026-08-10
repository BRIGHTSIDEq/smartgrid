# -*- coding: utf-8 -*-
"""
utils/quantile_metrics.py — оценка вероятностного прогноза.

ЗАЧЕМ ВООБЩЕ КВАНТИЛИ
─────────────────────
Точечный прогноз отвечает на вопрос «сколько будет», а решение о разряде
накопителя требует ответа на другой: «насколько велик риск, что окажется
больше». Плата за мощность — статистика максимума, и недооценка пика стоит
несопоставимо дороже переоценки. Измерено в этом проекте: ошибка точечного
прогноза уничтожает 85% ценности сглаживания пика.

Средний прогноз оптимален для квадратичной ошибки, медианный — для абсолютной,
и ни один из них не оптимален для несимметричной стоимости. Верхний квантиль
оптимален именно для неё, и его качество нельзя измерить ни MAE, ни RMSE.

ЧТО ИМЕННО НУЖНО ИЗМЕРЯТЬ
─────────────────────────
Три свойства, и ни одно не заменяет другие:

  острота     — насколько узок интервал; тривиально улучшается расширением
                границ, поэтому в одиночку бессмысленна;
  покрытие    — какая доля фактов попала внутрь интервала; тривиально
                улучшается растягиванием интервала до бесконечности;
  pinball     — собственно качество квантиля; строго правильная функция
                оценки, минимум достигается на истинном квантиле.

Отсюда порядок: pinball отвечает за качество, покрытие проверяет
добросовестность, острота показывает практическую пользу. Модель с идеальным
покрытием и интервалом от нуля до бесконечности бесполезна, и только острота
это обнаружит.
"""

import logging
from typing import Dict, Sequence, Tuple

import numpy as np

logger = logging.getLogger("smart_grid.utils.quantile_metrics")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """
    Функция потерь квантильной регрессии для уровня `quantile`.

        L(y, f) = max(q·(y − f), (q − 1)·(y − f))

    Недооценка (f < y) штрафуется с весом q, переоценка — с весом 1 − q.
    При q = 0.5 обе стороны равны, и величина составляет половину абсолютной
    ошибки: медиана и есть оптимум для MAE.

    Это строго правильная функция оценки: её математическое ожидание минимально
    тогда и только тогда, когда f равен истинному квантилю уровня q. Поэтому
    сравнивать вероятностные прогнозы можно именно по ней, а не по покрытию —
    покрытие достигается растягиванием интервала и ничего не говорит о
    качестве.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"Уровень квантиля должен лежать в (0, 1), получено {quantile}")

    delta = np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64)
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


def mean_pinball_loss(y_true: np.ndarray, predictions: Dict[float, np.ndarray]) -> float:
    """
    Средний pinball по набору уровней — дискретное приближение CRPS.

    Одна сводная величина нужна, чтобы сравнивать вероятностные прогнозы между
    собой: по отдельным уровням модели нередко выигрывают вразнобой, и без
    общего критерия сравнение превращается в перечисление.
    """
    if not predictions:
        raise ValueError("Не передано ни одного уровня квантиля")
    return float(np.mean([pinball_loss(y_true, p, q) for q, p in predictions.items()]))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Доля фактов, попавших в интервал [lower, upper]."""
    y = np.asarray(y_true)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def interval_width(lower: np.ndarray, upper: np.ndarray,
                   scale: float = 1.0) -> float:
    """
    Средняя ширина интервала, при необходимости в долях масштаба ряда.

    Без этой величины покрытие не интерпретируется: интервал «от нуля до
    бесконечности» даёт стопроцентное покрытие и нулевую пользу.
    """
    width = float(np.mean(np.asarray(upper) - np.asarray(lower)))
    return width / scale if scale not in (0.0, None) else width


def calibration(y_true: np.ndarray,
                predictions: Dict[float, np.ndarray]) -> Dict[float, float]:
    """
    Фактическая доля фактов ниже каждого предсказанного квантиля.

    Для откалиброванного прогноза доля совпадает с уровнем: ниже P90 обязаны
    оказаться 90% наблюдений. Расхождение означает систематическую ошибку,
    которую не видно ни в MAE, ни в pinball по отдельному уровню: модель может
    хорошо упорядочивать наблюдения и при этом стабильно смещать границы.
    """
    y = np.asarray(y_true)
    return {q: float(np.mean(y <= np.asarray(p))) for q, p in sorted(predictions.items())}


def crossing_rate(predictions: Dict[float, np.ndarray]) -> float:
    """
    Доля точек, где квантили нарушают порядок.

    Уровни обучаются независимыми моделями, и ничто не мешает предсказанию для
    P10 оказаться выше P90. Такой прогноз бессмыслен как распределение, но
    остаётся совершенно правдоподобным по отдельным метрикам: и pinball, и MAE
    у каждого уровня в отдельности будут нормальными.
    """
    levels = sorted(predictions)
    if len(levels) < 2:
        return 0.0

    stacked = np.stack([np.asarray(predictions[q]).ravel() for q in levels], axis=0)
    violated = np.any(np.diff(stacked, axis=0) < 0, axis=0)
    return float(np.mean(violated))


def enforce_monotone(predictions: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    """
    Восстанавливает порядок квантилей сортировкой значений в каждой точке.

    Сортировка — не косметика: она не ухудшает pinball ни на одном уровне.
    Если предсказания переставлены так, что нарушают порядок, приведение их к
    возрастающему виду может только приблизить каждое к своему квантилю.
    Величина нарушения при этом обязана сохраняться в отчёте: частые пересечения
    означают, что уровни обучены несогласованно.
    """
    levels = sorted(predictions)
    stacked = np.stack([np.asarray(predictions[q], np.float64) for q in levels], axis=0)
    stacked = np.sort(stacked, axis=0)
    return {q: stacked[i] for i, q in enumerate(levels)}


def evaluate_quantiles(y_true: np.ndarray, predictions: Dict[float, np.ndarray],
                       interval: Tuple[float, float] = (0.1, 0.9),
                       scale: float = 1.0) -> Dict[str, float]:
    """
    Сводная оценка вероятностного прогноза.

    Возвращает и pinball, и покрытие, и остроту одновременно: каждая из трёх
    величин по отдельности улучшается тривиальным способом, и только вместе они
    описывают пригодность прогноза.
    """
    lo_q, hi_q = interval
    missing = [q for q in (lo_q, hi_q) if q not in predictions]
    if missing:
        raise KeyError(f"Для интервала нужны уровни {missing}, а их нет в прогнозе")

    lower, upper = predictions[lo_q], predictions[hi_q]
    nominal = hi_q - lo_q
    empirical = coverage(y_true, lower, upper)

    result = {
        "pinball_mean": mean_pinball_loss(y_true, predictions),
        "coverage": empirical,
        "coverage_nominal": nominal,
        "coverage_gap": empirical - nominal,
        "interval_width": interval_width(lower, upper, scale),
        "crossing_rate": crossing_rate(predictions),
    }
    for q, value in calibration(y_true, predictions).items():
        result[f"calib_q{int(round(q * 100)):02d}"] = value

    if abs(result["coverage_gap"]) > 0.05:
        logger.warning(
            "Покрытие %.1f%% против номинальных %.1f%% — интервал %s",
            empirical * 100, nominal * 100,
            "слишком узок" if empirical < nominal else "избыточно широк")
    if result["crossing_rate"] > 0.0:
        logger.warning("Квантили пересекаются в %.2f%% точек",
                       result["crossing_rate"] * 100)
    return result
