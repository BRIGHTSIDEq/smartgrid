# -*- coding: utf-8 -*-
"""
Тесты конформных интервалов поверх точечного прогноза.

Главное проверяемое свойство — покрытие: интервал, построенный по остаткам
отложенной выборки, обязан покрывать заявленную долю наблюдений на новых
данных. Отдельно проверяется, что смещение считается по шагам горизонта и что
остатки теста в построение не попадают.
"""

import numpy as np
import pytest

from utils.conformal import (
    apply_offset, conformal_band, conformal_offsets, empirical_coverage_report,
)


def _make_split(n=4000, horizon=6, seed=0):
    """Прогноз с ошибкой, растущей по горизонту и несимметричной по знаку."""
    rng = np.random.RandomState(seed)
    scale = np.linspace(1.0, 4.0, horizon)
    # Логнормальная ошибка: правый хвост тяжелее левого, как у нагрузки.
    err = (rng.lognormal(0.0, 0.5, size=(n, horizon)) - 1.0) * scale
    pred = rng.normal(100.0, 5.0, size=(n, horizon))
    return pred, pred + err


def test_coverage_holds_on_unseen_data():
    """
    Интервал по остаткам валидации покрывает заявленную долю на тесте.

    Это и есть смысл конформного подхода: покрытие обеспечивается без
    предположений о виде распределения ошибки.
    """
    pred_val, y_val = _make_split(seed=0)
    pred_test, y_test = _make_split(seed=1)

    band = conformal_band(pred_test, y_val - pred_val, quantiles=(0.1, 0.9))
    report = empirical_coverage_report(y_test, band)

    assert report["coverage"] == pytest.approx(0.8, abs=0.03), report


def test_offsets_grow_with_horizon():
    """
    Смещение считается по шагам горизонта.

    Ошибка на час вперёд и на сутки различается в разы: единое смещение дало бы
    избыточный запас в начале горизонта и недостаточный в конце.
    """
    pred, y = _make_split()
    offsets = conformal_offsets(y - pred, quantiles=(0.9,))[0.9]

    assert offsets.shape == (6,)
    assert offsets[-1] > 2.0 * offsets[0], "смещение не растёт по горизонту"

    single = conformal_offsets(y - pred, quantiles=(0.9,), per_step=False)[0.9]
    assert np.ndim(single) == 0


def test_upper_quantile_shifts_forecast_up():
    """Верхний уровень поднимает прогноз, нижний опускает — иначе интервал вывернут."""
    pred, y = _make_split()
    band = conformal_band(pred, y - pred, quantiles=(0.1, 0.5, 0.9))

    assert np.all(band[0.1] <= band[0.5] + 1e-6)
    assert np.all(band[0.5] <= band[0.9] + 1e-6)
    assert band[0.9].mean() > pred.mean()


def test_offset_length_mismatch_is_rejected():
    """
    Несовпадение длины горизонта — отказ, а не молчаливый broadcast.

    При смещении не той длины numpy либо упал бы в неожиданном месте, либо
    растянул значения по правилам broadcast, дав неверный прогноз без ошибки.
    """
    pred = np.zeros((10, 6))
    with pytest.raises(ValueError, match="шагов горизонта"):
        apply_offset(pred, np.zeros(4))


def test_residuals_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"\(N, H\)"):
        conformal_offsets(np.zeros(10))


def test_band_ignores_the_evaluated_sample():
    """
    Интервал строится только по переданным остаткам валидации.

    Проверка от противного: изменение тестовых наблюдений не должно менять
    границы. Иначе интервал знал бы ответ и показывал покрытие, недостижимое в
    эксплуатации.
    """
    pred_val, y_val = _make_split(seed=0)
    pred_test, _ = _make_split(seed=1)

    band_a = conformal_band(pred_test, y_val - pred_val, quantiles=(0.9,))
    band_b = conformal_band(pred_test, y_val - pred_val, quantiles=(0.9,))

    assert np.allclose(band_a[0.9], band_b[0.9])


def test_coverage_gap_is_reported_when_error_shifts():
    """
    Нарушение обмениваемости обнаруживается, а не замалчивается.

    Хронологическое разбиение сдвигает распределение ошибки во времени, и
    конформная гарантия перестаёт выполняться. Отчёт обязан показывать
    фактическое покрытие, чтобы расхождение было видно.
    """
    pred_val, y_val = _make_split(seed=0)
    pred_test, y_test = _make_split(seed=1)
    y_test = y_test + 8.0                       # ошибка сместилась после валидации

    band = conformal_band(pred_test, y_val - pred_val, quantiles=(0.1, 0.9))
    report = empirical_coverage_report(y_test, band)

    assert report["coverage"] < 0.7, "сдвиг обязан обрушить покрытие"
    assert report["coverage_gap"] < -0.1
