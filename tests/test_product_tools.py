# -*- coding: utf-8 -*-
"""Калькулятор ценности прогноза, мониторинг качества и конфигурация клиента."""

import numpy as np
import pandas as pd
import pytest

from analysis.value_calculator import PERFECT, forecast_value, simple_forecasts
from optimization.tariffs_ru import RuTariff
from utils.client_config import load_client_config
from utils.monitoring import alerts, daily_quality


def _history(days=50, seed=0):
    ts = pd.date_range("2025-02-03", periods=days * 24, freq="h")
    rng = np.random.RandomState(seed)
    h = ts.hour.to_numpy()
    load = (300 + 120 * np.exp(-((h - 18) ** 2) / 6) + 60 * np.exp(-((h - 9) ** 2) / 4)
            - 80 * (ts.dayofweek.to_numpy() >= 5) + rng.normal(0, 15, len(ts)))
    return pd.Series(load, index=ts)


def test_simple_forecasts_do_not_look_ahead():
    s = _history()
    fc = simple_forecasts(s, eval_days=14)
    start = fc[PERFECT].index[0]
    assert start == s.index[-1].normalize() - pd.Timedelta(days=13)
    np.testing.assert_allclose(fc["Как вчера"].to_numpy(),
                               s.shift(24).loc[start:].to_numpy())
    # Профиль часа недели не меняется, если испортить оцениваемый отрезок.
    spoiled = s.copy()
    spoiled.loc[start:] *= 10
    np.testing.assert_allclose(simple_forecasts(spoiled, 14)["Профиль часа недели"],
                               fc["Профиль часа недели"])


def test_perfect_forecast_is_worth_the_most():
    table = forecast_value(_history(), RuTariff(category=6), eval_days=14)
    assert table.iloc[0]["source"] == PERFECT
    assert table.loc[table["source"] == PERFECT, "deviation_cost"].iloc[0] == 0.0
    assert (table["deviation_cost"] >= 0).all()
    assert table.loc[table["source"] == "Как вчера", "annual_gain_vs_yesterday"].iloc[0] == 0.0


def test_model_forecast_can_be_added():
    s = _history()
    good = s + np.random.RandomState(3).normal(0, 5, len(s))
    table = forecast_value(s, RuTariff(category=6), eval_days=14, extra={"Модель": good})
    row = table.set_index("source")
    assert row.loc["Модель", "MAE"] < row.loc["Как вчера", "MAE"]
    assert row.loc["Модель", "annual_value"] > row.loc["Как вчера", "annual_value"]


def test_monitoring_flags_degradation_and_missing_data():
    ts = pd.date_range("2025-04-01", periods=10 * 24, freq="h")
    actual = pd.DataFrame({"series": "A", "timestamp": ts, "consumption": 100.0})
    err = np.where(ts >= pd.Timestamp("2025-04-06"), 30.0, 5.0)
    fc = pd.DataFrame({"series": "A", "timestamp": ts, "forecast": 100.0 + err})
    actual.loc[actual["timestamp"] >= pd.Timestamp("2025-04-10"), "consumption"] = np.nan

    table = daily_quality(fc, actual, naive_scale={"A": 10.0}, reference_mase={"A": 0.5},
                          window_days=3)
    by_day = table.set_index("day")["status"]
    assert by_day[pd.Timestamp("2025-04-03")] == "норма"
    assert by_day[pd.Timestamp("2025-04-08")] == "деградация"
    assert by_day[pd.Timestamp("2025-04-10")] == "нет данных"
    assert list(alerts(table)["status"]) == ["нет данных"]


def test_client_config_example_loads():
    cfg = load_client_config("clients/example.yaml")
    assert cfg.tariff.category == 6 and cfg.tariff.two_rate_network
    assert cfg.export.wide and cfg.export.unit == "kW"
    assert cfg.tariff.peak_windows[1] == (8, 21)


def test_client_config_rejects_typos(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: x\ndata_path: y\ntariff:\n  gen_capacity_rte: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gen_capacity_rte"):
        load_client_config(str(path))
