# -*- coding: utf-8 -*-
"""
Поправки к тесту Диболда–Мариано: малая выборка и множественные сравнения.

После прореживания до непересекающихся суток независимых наблюдений около
сотни, а пар моделей — до 28. Без поправок нормальное приближение завышает
значимость, а среди 28 пар почти наверняка находится ложная «значимая».
"""

import numpy as np
import pytest
from scipy import stats

from utils.metrics import hln_correction, holm_adjust, pairwise_dm_table


def test_hln_shrinks_the_statistic_and_widens_p():
    dm, n, h = 2.5, 100, 24
    corrected, p = hln_correction(dm, n, h)
    factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert corrected == pytest.approx(dm * factor)
    assert abs(corrected) < abs(dm)
    assert p == pytest.approx(2 * stats.t.sf(abs(corrected), df=n - 1))
    assert p > 2 * stats.norm.sf(dm), "поправка обязана делать вывод осторожнее"


def test_hln_vanishes_on_large_samples():
    corrected, _ = hln_correction(3.0, 1_000_000, 1)
    assert corrected == pytest.approx(3.0, rel=1e-5)


def test_holm_matches_the_textbook_example():
    # Пример из Holm (1979): упорядоченные p умножаются на m, m−1, …
    # с сохранением монотонности.
    assert holm_adjust([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_skips_missing_values():
    out = holm_adjust([0.01, float("nan"), 0.02])
    assert out[0] == pytest.approx(0.02) and np.isnan(out[1]) and out[2] == pytest.approx(0.02)


def test_pairwise_table_keeps_raw_columns_and_adds_corrected_ones():
    """
    Шум без реального различия: сырой тест на части пар может ошибиться,
    а поправленный вывод обязан быть не смелее сырого.
    """
    rng = np.random.RandomState(0)
    truth = rng.normal(100, 10, size=(240, 24))
    preds = {f"M{i}": truth + rng.normal(0, 5, size=truth.shape) for i in range(6)}
    rows = pairwise_dm_table(truth, preds, h=24)

    assert len(rows) == 15
    for r in rows:
        assert {"DM", "p_value", "better", "DM_HLN", "p_HLN", "p_holm", "better_holm"} <= set(r)
        assert r["p_holm"] >= r["p_HLN"] - 1e-9 >= r["p_value"] - 1e-6
        if r["better_holm"] != "различие незначимо":
            assert r["better"] == r["better_holm"]
