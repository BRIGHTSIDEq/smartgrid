# -*- coding: utf-8 -*-
"""
utils/conformal.py — интервалы поверх готового точечного прогноза.

ЗАЧЕМ ЕЩЁ ОДИН СПОСОБ, ЕСЛИ ЕСТЬ КВАНТИЛЬНЫЕ МОДЕЛИ
───────────────────────────────────────────────────
Квантильная модель обучается отдельно и заменяет точечную. Здесь решается
другая задача: у нас уже есть лучшая по валидации точечная модель, и нужно
получить от НЕЁ верхнюю границу, не переобучая и не меняя выводов о точности.

Способ прямой: взять остатки на ОТЛОЖЕННОЙ выборке и сдвинуть прогноз на их
квантиль. Это split conformal prediction — при обмениваемости остатков
валидации и теста покрытие гарантировано конечной выборкой, без предположений
о виде распределения ошибки. Для нормального распределения ошибок результат
совпал бы с оценкой через сигму, но такого предположения здесь не требуется, а
ошибка прогноза нагрузки заметно асимметрична.

ЗАЧЕМ ЭТО НУЖНО ИМЕННО В ЭТОЙ РАБОТЕ
────────────────────────────────────
Измерено, что ошибка точечного прогноза уничтожает 85% ценности сглаживания
пика. Причина в несимметрии: недооценка пика оставляет накопитель незаряженным
и обесценивает месяц работы, переоценка стоит лишь небольшого износа. Значит,
на вход управлению следует подавать не средний прогноз, а верхний квантиль.
Настоящий модуль позволяет проверить это, не трогая сами модели: сравниваются
одни и те же прогнозы, сдвинутые и несдвинутые.

СМЕЩЕНИЕ СЧИТАЕТСЯ ПО ШАГАМ ГОРИЗОНТА
─────────────────────────────────────
Ошибка прогноза на час вперёд и на сутки различается в разы, поэтому единое
смещение дало бы слишком широкий запас в начале горизонта и слишком узкий в
конце. Для сезонно-наивного прогноза это не так — его ошибка по горизонту почти
постоянна, — но для обучаемых моделей рост существенный.
"""

import logging
from typing import Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger("smart_grid.utils.conformal")


def conformal_offsets(residuals: np.ndarray,
                      quantiles: Sequence[float] = (0.1, 0.5, 0.9),
                      per_step: bool = True) -> Dict[float, np.ndarray]:
    """
    Квантили остатков отложенной выборки.

    Parameters
    ----------
    residuals : np.ndarray формы (N, H)
        Разности «факт минус прогноз» на ВАЛИДАЦИИ. Остатки теста здесь
        недопустимы: интервал, построенный по оцениваемой выборке, знал бы
        ответ и показал бы покрытие, недостижимое в эксплуатации.
    per_step : bool
        Считать ли смещение отдельно для каждого шага горизонта.

    Returns
    -------
    dict {уровень: смещение формы (H,) либо скаляр}
    """
    residuals = np.asarray(residuals, np.float64)
    if residuals.ndim != 2:
        raise ValueError(f"Ожидались остатки формы (N, H), получено {residuals.shape}")
    if len(residuals) < 20:
        logger.warning("Всего %d остатков — квантильная оценка ненадёжна",
                       len(residuals))

    axis = 0 if per_step else None
    offsets = {float(q): np.quantile(residuals, q, axis=axis) for q in quantiles}

    for q, value in sorted(offsets.items()):
        logger.info("Конформное смещение q=%.2f: среднее %.3f", q, float(np.mean(value)))
    return offsets


def apply_offset(point_forecast: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """
    Сдвигает точечный прогноз на смещение, полученное по валидации.

    Прогноз имеет форму (N, H), смещение — (H,) либо скаляр: сдвиг применяется
    к соответствующему шагу горизонта, а не ко всему окну одинаково.
    """
    forecast = np.asarray(point_forecast, np.float64)
    shift = np.asarray(offset, np.float64)

    if shift.ndim == 1 and shift.shape[0] != forecast.shape[-1]:
        raise ValueError(
            f"Смещение задано на {shift.shape[0]} шагов горизонта, "
            f"а прогноз имеет {forecast.shape[-1]}"
        )
    return (forecast + shift).astype(np.float32)


def conformal_band(point_forecast: np.ndarray, residuals: np.ndarray,
                   quantiles: Sequence[float] = (0.1, 0.5, 0.9),
                   per_step: bool = True) -> Dict[float, np.ndarray]:
    """Готовый набор квантильных прогнозов из точечного и остатков валидации."""
    offsets = conformal_offsets(residuals, quantiles, per_step)
    return {q: apply_offset(point_forecast, offsets[q]) for q in sorted(offsets)}


def empirical_coverage_report(y_true: np.ndarray, band: Dict[float, np.ndarray],
                              interval: Optional[Sequence[float]] = None
                              ) -> Dict[str, float]:
    """
    Проверяет, выполняется ли обещанное покрытие на оцениваемой выборке.

    Конформная гарантия опирается на обмениваемость остатков валидации и теста.
    Хронологическое разбиение её нарушает: распределение ошибки со временем
    смещается. Поэтому фактическое покрытие обязано проверяться, а не
    предполагаться — расхождение с номиналом здесь ожидаемо и информативно.
    """
    from utils.quantile_metrics import coverage

    lo_q, hi_q = (interval if interval is not None
                  else (min(band), max(band)))
    empirical = coverage(y_true, band[lo_q], band[hi_q])
    nominal = float(hi_q) - float(lo_q)

    gap = empirical - nominal
    if abs(gap) > 0.05:
        logger.warning(
            "Конформное покрытие %.1f%% против номинальных %.1f%%: "
            "обмениваемость нарушена сдвигом распределения ошибки во времени",
            empirical * 100, nominal * 100)

    return {"coverage": empirical, "coverage_nominal": nominal, "coverage_gap": gap}
