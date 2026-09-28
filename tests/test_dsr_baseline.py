# -*- coding: utf-8 -*-
"""Базовая линия управления спросом: похожие дни, поправка, оценка точности."""

import numpy as np
import pandas as pd
import pytest

from analysis.dsr_baseline import evaluate_baselines, similar_days_baseline

TS = pd.date_range("2025-03-03", periods=35 * 24, freq="h")     # понедельник


def _load(weekend_drop=100.0):
    h = TS.hour.to_numpy()
    base = 300 + 100 * np.exp(-((h - 18) ** 2) / 6)
    base = base - weekend_drop * (TS.dayofweek.to_numpy() >= 5)
    return pd.Series(base, index=TS)


def test_workday_uses_only_workdays():
    s = _load()
    day = pd.Timestamp("2025-03-31")                   # понедельник
    base = similar_days_baseline(s, day, n_days=10)
    np.testing.assert_allclose(base.to_numpy(), s.loc[day:day + pd.Timedelta(hours=23)])


def test_weekend_uses_only_weekends():
    s = _load()
    day = pd.Timestamp("2025-03-30")                   # воскресенье
    base = similar_days_baseline(s, day, n_days=4)
    np.testing.assert_allclose(base.to_numpy(), s.loc[day:day + pd.Timedelta(hours=23)])


def test_holiday_counts_as_a_day_off():
    s = _load()
    day = pd.Timestamp("2025-03-31")
    base = similar_days_baseline(s, day, n_days=4, holidays=[day])
    assert base.iloc[18] == pytest.approx(s.loc["2025-03-30 18:00"])


def test_past_event_days_are_excluded():
    s = _load()
    s.loc["2025-03-28 17:00":"2025-03-28 20:00"] -= 150     # прошлое событие
    day = pd.Timestamp("2025-03-31")
    with_event = similar_days_baseline(s, day, n_days=5)
    without = similar_days_baseline(s, day, n_days=5, exclude_days=["2025-03-28"])
    assert without.iloc[18] > with_event.iloc[18]
    assert without.iloc[18] == pytest.approx(s.loc["2025-03-31 18:00"])


def test_same_day_adjustment_is_capped():
    s = _load()
    day = pd.Timestamp("2025-03-31")
    s.loc[day:day + pd.Timedelta(hours=23)] *= 1.5          # аномально высокий день
    adj = similar_days_baseline(s, day, n_days=10, adjust_hours=[12, 13, 14], adjust_cap=0.2)
    raw = similar_days_baseline(s, day, n_days=10)
    np.testing.assert_allclose(adj.to_numpy(), raw.to_numpy() * 1.2)


def test_high_mode_picks_the_largest_days():
    s = _load()
    s.loc["2025-03-27"] += 50                               # один «тяжёлый» четверг
    day = pd.Timestamp("2025-03-31")
    high = similar_days_baseline(s, day, n_days=1, pool_days=10, mode="high")
    assert high.iloc[0] == pytest.approx(s.loc["2025-03-27 00:00"])


def test_evaluation_ranks_an_accurate_forecast_first():
    rng = np.random.RandomState(0)
    s = _load() * (1 + rng.normal(0, 0.05, len(TS)))
    forecast = s * (1 + rng.normal(0, 0.005, len(TS)))
    days = pd.date_range("2025-03-24", "2025-04-04", freq="B")
    table = evaluate_baselines(s, days, event_hours=[17, 18, 19, 20], forecast=forecast)
    assert table.iloc[0]["method"] == "Прогноз модели"
    assert set(table["method"]) >= {"10 похожих дней", "5 максимальных из 10"}
