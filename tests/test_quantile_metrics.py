# -*- coding: utf-8 -*-
"""
Тесты оценки вероятностного прогноза.

Проверяются свойства, ради которых квантильные метрики и вводятся: pinball
обязан достигать минимума на истинном квантиле, покрытие — ловить слишком
узкий интервал, а острота — не давать выдать бесконечно широкий интервал за
хороший результат. Каждая из трёх величин по отдельности улучшается
тривиальным способом, поэтому проверяется и то, что они друг друга не
подменяют.
"""

import numpy as np
import pytest

from utils.quantile_metrics import (
    calibration, coverage, crossing_rate, enforce_monotone, evaluate_quantiles,
    interval_width, mean_pinball_loss, pinball_loss,
)


# ══════════════════════════════════════════════════════════════════════════════
# PINBALL
# ══════════════════════════════════════════════════════════════════════════════

def test_pinball_at_median_is_half_the_absolute_error():
    """
    При q = 0.5 pinball равен половине MAE.

    Это связывает вероятностную оценку с привычной: медиана и есть оптимум для
    абсолютной ошибки.
    """
    y = np.array([10.0, 20.0, 30.0])
    f = np.array([12.0, 18.0, 33.0])

    assert pinball_loss(y, f, 0.5) == pytest.approx(np.mean(np.abs(y - f)) / 2)


def test_pinball_penalises_underestimation_more_at_high_quantile():
    """
    Верхний квантиль штрафует занижение сильнее завышения.

    Ради этой несимметрии квантили и нужны: недооценка пика стоит дорого,
    переоценка почти ничего.
    """
    y = np.array([100.0])
    under, over = np.array([90.0]), np.array([110.0])

    assert pinball_loss(y, under, 0.9) > pinball_loss(y, over, 0.9)
    assert pinball_loss(y, under, 0.1) < pinball_loss(y, over, 0.1)
    # На медиане обе стороны равноценны.
    assert pinball_loss(y, under, 0.5) == pytest.approx(pinball_loss(y, over, 0.5))


def test_pinball_is_minimised_at_the_true_quantile():
    """
    Главное свойство: минимум достигается ровно на истинном квантиле.

    Именно оно делает pinball пригодным для сравнения моделей. Проверка идёт по
    выборке с известным распределением: перебор кандидатов обязан привести к
    выборочному квантилю, а не к среднему или медиане.
    """
    rng = np.random.RandomState(0)
    sample = rng.normal(50.0, 10.0, size=20_000)

    for q in (0.1, 0.5, 0.9):
        true_q = float(np.quantile(sample, q))
        candidates = true_q + np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
        losses = [pinball_loss(sample, np.full_like(sample, c), q) for c in candidates]

        assert int(np.argmin(losses)) == 3, f"минимум не на истинном квантиле q={q}"


def test_pinball_rejects_levels_outside_the_open_interval():
    y = np.array([1.0])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="лежать в"):
            pinball_loss(y, y, bad)


def test_mean_pinball_requires_at_least_one_level():
    with pytest.raises(ValueError, match="ни одного уровня"):
        mean_pinball_loss(np.array([1.0]), {})


# ══════════════════════════════════════════════════════════════════════════════
# ПОКРЫТИЕ И ОСТРОТА
# ══════════════════════════════════════════════════════════════════════════════

def test_coverage_matches_the_nominal_level_for_a_correct_forecast():
    """Правильно построенный интервал покрывает заявленную долю наблюдений."""
    rng = np.random.RandomState(1)
    sample = rng.normal(0.0, 1.0, size=50_000)
    lo, hi = np.quantile(sample, 0.1), np.quantile(sample, 0.9)

    assert coverage(sample, np.full_like(sample, lo),
                    np.full_like(sample, hi)) == pytest.approx(0.8, abs=0.01)


def test_width_exposes_a_uselessly_wide_interval():
    """
    Покрытие само по себе ничего не значит — его выдаёт острота.

    Интервал, растянутый до бесконечности, покрывает всё и бесполезен. Без
    ширины такой прогноз выглядел бы безупречным.
    """
    y = np.random.RandomState(2).normal(size=1000)
    huge_lo, huge_hi = np.full_like(y, -1e6), np.full_like(y, 1e6)

    assert coverage(y, huge_lo, huge_hi) == 1.0, "широкий интервал покрывает всё"
    assert interval_width(huge_lo, huge_hi) > 1e6, "ширина обязана выдать подвох"


def test_calibration_detects_a_systematically_shifted_quantile():
    """
    Калибровка ловит смещение, которого не видно по одному уровню.

    Модель может верно упорядочивать наблюдения и при этом стабильно
    промахиваться по границам.
    """
    rng = np.random.RandomState(3)
    y = rng.normal(0.0, 1.0, size=20_000)

    honest = {q: np.full_like(y, np.quantile(y, q)) for q in (0.1, 0.5, 0.9)}
    shifted = {q: v + 1.0 for q, v in honest.items()}

    for q, share in calibration(y, honest).items():
        assert share == pytest.approx(q, abs=0.02)

    biased = calibration(y, shifted)
    assert biased[0.5] > 0.7, "смещение вверх обязано проявиться в калибровке"


# ══════════════════════════════════════════════════════════════════════════════
# ПЕРЕСЕЧЕНИЕ КВАНТИЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def test_crossing_is_detected_and_absent_when_order_is_correct():
    """
    Пересечение уровней не проявляется в метриках каждого уровня.

    Уровни обучаются независимо, и P10 может оказаться выше P90. Как
    распределение такой прогноз бессмыслен, но pinball и MAE у каждого уровня
    в отдельности остаются нормальными.
    """
    ordered = {0.1: np.array([1.0, 2.0]), 0.5: np.array([2.0, 3.0]),
               0.9: np.array([3.0, 4.0])}
    assert crossing_rate(ordered) == 0.0

    crossed = {0.1: np.array([1.0, 5.0]), 0.5: np.array([2.0, 3.0]),
               0.9: np.array([3.0, 4.0])}
    assert crossing_rate(crossed) == pytest.approx(0.5)


def test_enforcing_order_does_not_worsen_pinball():
    """
    Сортировка восстанавливает порядок и не ухудшает качество ни на одном уровне.

    Это и есть основание применять её: приведение переставленных значений к
    возрастающему виду может лишь приблизить каждое к своему квантилю.
    """
    rng = np.random.RandomState(4)
    y = rng.normal(0.0, 1.0, size=5_000)
    levels = (0.1, 0.5, 0.9)
    noisy = {q: np.quantile(y, q) + rng.normal(0, 0.8, size=y.shape) for q in levels}

    assert crossing_rate(noisy) > 0.0, "проверка требует наличия пересечений"
    fixed = enforce_monotone(noisy)
    assert crossing_rate(fixed) == 0.0

    for q in levels:
        assert pinball_loss(y, fixed[q], q) <= pinball_loss(y, noisy[q], q) + 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# СВОДНАЯ ОЦЕНКА
# ══════════════════════════════════════════════════════════════════════════════

def test_summary_reports_all_three_properties():
    """Сводка не позволяет отчитаться одной величиной вместо трёх."""
    rng = np.random.RandomState(5)
    y = rng.normal(100.0, 15.0, size=10_000)
    preds = {q: np.full_like(y, np.quantile(y, q)) for q in (0.1, 0.5, 0.9)}

    result = evaluate_quantiles(y, preds, interval=(0.1, 0.9))

    for key in ("pinball_mean", "coverage", "coverage_gap", "interval_width",
                "crossing_rate", "calib_q10", "calib_q50", "calib_q90"):
        assert key in result, f"в сводке нет {key}"

    assert result["coverage"] == pytest.approx(0.8, abs=0.02)
    assert result["coverage_nominal"] == pytest.approx(0.8)
    assert result["crossing_rate"] == 0.0


def test_summary_refuses_an_interval_it_cannot_build():
    """Отсутствие нужного уровня — отказ, а не молча посчитанная неверная сводка."""
    y = np.array([1.0, 2.0])
    preds = {0.5: np.array([1.0, 2.0])}

    with pytest.raises(KeyError, match="0.1"):
        evaluate_quantiles(y, preds, interval=(0.1, 0.9))
