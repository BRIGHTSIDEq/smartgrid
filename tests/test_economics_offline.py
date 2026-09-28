# -*- coding: utf-8 -*-
"""
Офлайн-экономика по сохранённым прогнозам.

Ключевое свойство — точное разложение экономии по суткам: на нём держится
доверительный интервал. Если сумма суточных вкладов разойдётся со счётом,
интервал будет построен для другой величины.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.economics import (
    ACTUAL_COLUMN, ORACLE, UPPER_BOUND, battery_for_load, block_bootstrap_annual,
    daily_savings, evaluate_sources, load_forecast_series, main, sensitivity,
)
from optimization.controllers import evaluate_schedule, mpc_day_ahead
from optimization.tariffs_ru import RuTariff

TS = pd.date_range("2025-03-24", periods=21 * 24, freq="h")     # стык марта и апреля


def _frame(seed=0):
    rng = np.random.RandomState(seed)
    h = TS.hour.to_numpy()
    actual = (300 + 120 * np.exp(-((h - 18) ** 2) / 6) + 60 * np.exp(-((h - 9) ** 2) / 4)
              + rng.normal(0, 10, len(h)))
    return pd.DataFrame({"timestamp": TS,
                         "Хороший": actual + rng.normal(0, 8, len(h)),
                         "Плохой": np.roll(actual, 5),
                         ACTUAL_COLUMN: actual})


@pytest.mark.parametrize("category", [3, 4, 5, 6])
def test_daily_savings_add_up_to_the_bill(category):
    frame = _frame()
    actual = frame[ACTUAL_COLUMN].to_numpy()
    fc = frame["Хороший"].to_numpy()
    tariff = RuTariff(category=category)
    battery = battery_for_load(actual.max())
    sched = mpc_day_ahead(fc, actual, TS, tariff, battery)
    plan = fc + (sched["grid"] - actual) if tariff.with_plan else None

    total = evaluate_schedule(sched["grid"], sched["charged"], actual, TS, tariff, battery,
                              plan=plan)
    daily = daily_savings(actual, sched["grid"], sched["charged"], TS, tariff, battery,
                          plan=plan)
    assert len(daily) == 21
    assert daily.sum() == pytest.approx(total["net_savings"], rel=1e-6, abs=1e-3)


def test_bootstrap_interval_contains_the_mean():
    days = pd.Series(np.random.RandomState(0).normal(100, 30, 60))
    r = block_bootstrap_annual(days, n_boot=500)
    assert r["annual_lo"] < r["annual_mean"] < r["annual_hi"]
    assert r["annual_mean"] == pytest.approx(days.mean() * 365)


def test_constant_daily_savings_give_a_degenerate_interval():
    r = block_bootstrap_annual(pd.Series(np.full(30, 50.0)), n_boot=200)
    assert r["annual_lo"] == pytest.approx(r["annual_hi"]) == pytest.approx(50 * 365)


def test_sources_are_ranked_below_the_upper_bound():
    frame = _frame()
    tariff = RuTariff(category=4)
    battery = battery_for_load(frame[ACTUAL_COLUMN].max())
    res = evaluate_sources(frame, tariff, battery, n_boot=200)
    table = res["summary"].set_index("source")
    bound = table.loc[UPPER_BOUND, "net_savings"]
    assert (table["net_savings"] <= bound + 1e-6).all()
    assert table.loc[ORACLE, "net_savings"] >= table.loc["Плохой", "net_savings"]
    assert set(res["monthly"]["month"]) == {"2025-03", "2025-04"}


def test_sensitivity_grid_is_complete():
    frame = _frame()
    sens = sensitivity(frame, RuTariff(category=4), "Хороший",
                       capacity_scales=(0.5, 1.0), capex_per_kwh=(16_000.0,),
                       capacity_rate_scales=(1.0, 1.3))
    assert len(sens) == 2 * 1 * 2 * 2
    # Выше ставка мощности — выше верхняя граница экономии.
    ub = sens[sens["controller"] == UPPER_BOUND].set_index(
        ["capacity_scale", "capacity_rate_scale"])["annual_net_savings"]
    assert ub[(1.0, 1.3)] > ub[(1.0, 1.0)]


def test_command_line_writes_a_report(tmp_path):
    frame = _frame()
    frame.to_csv(tmp_path / "forecast_series_test.csv", index=False, encoding="utf-8-sig")
    assert main([str(tmp_path), "--category", "4"]) == 0
    out = tmp_path / "economics_ru_cat4"
    for name in ("summary.csv", "monthly.csv", "daily.csv", "report.md"):
        assert (out / name).exists()
    assert "Оптимум при известном будущем" in (out / "report.md").read_text(encoding="utf-8")


def test_missing_series_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="forecast_series_test.csv"):
        load_forecast_series(str(tmp_path))


def test_panel_series_export_and_per_client_economics(tmp_path):
    """
    Ряды «на сутки вперёд» по клиентам стыкуются без перекрытий и дают
    экономику по каждому клиенту.
    """
    from analysis.economics import write_panel_report
    from data.panel import generate_panel_data
    from data.panel_preprocessing import prepare_panel_data
    from models.panel_models import PanelNaive24
    from models.panel_trainer import PanelTrainer
    from panel_pipeline import export_panel_forecast_series

    df, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=2)
    data = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=2)
    trainer = PanelTrainer(PanelNaive24(), "Naive24").train(data)

    path = export_panel_forecast_series(trainer, data, str(tmp_path))
    long = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
    assert long["series"].nunique() == 2
    for _, g in long.groupby("series"):
        steps = g["timestamp"].sort_values().diff().dropna()
        assert (steps == pd.Timedelta(hours=1)).all(), "ряд клиента с дырами или повторами"
        assert g["timestamp"].min().hour == 0

    # Факт в файле совпадает с исходными данными генератора.
    key = long["series"].iloc[0]
    city, feeder = key.split("/")
    src = df[(df["city_id"].astype(str) == city) & (df["feeder_id"].astype(str) == feeder)]
    merged = long[long["series"] == key].merge(src[["timestamp", "consumption"]], on="timestamp")
    np.testing.assert_allclose(merged["actual"], merged["consumption"], rtol=1e-4)

    table = write_panel_report(str(tmp_path), category=4, n_boot=100)
    assert set(table["controller"]) == {"MPC по прогнозу", UPPER_BOUND}
    assert (tmp_path / "economics_ru_cat4_panel" / "report.md").exists()
    by = table.pivot(index="series", columns="controller", values="net_savings")
    assert (by[UPPER_BOUND] >= by["MPC по прогнозу"] - 1e-6).all()


def test_thinned_panel_windows_are_not_exported(tmp_path):
    from data.panel import generate_panel_data
    from data.panel_preprocessing import prepare_panel_data
    from models.panel_models import PanelNaive24
    from models.panel_trainer import PanelTrainer
    from panel_pipeline import export_panel_forecast_series

    df, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=2)
    data = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=2,
                              window_stride=11)
    trainer = PanelTrainer(PanelNaive24(), "Naive24").train(data)
    assert export_panel_forecast_series(trainer, data, str(tmp_path)) is None


def test_bootstrap_samples_every_day_equally():
    """
    Выброс в последние сутки не выталкивает среднее за пределы интервала.

    Нециклические блоки с началом не позже n − b брали последние сутки реже
    прочих; на 13 сутках с выбросом в конце среднее оказывалось вне
    собственного 90%-интервала (найдено на smoke-прогоне UCI).
    """
    days = pd.Series(np.r_[np.full(12, 10.0), 1000.0])
    r = block_bootstrap_annual(days, n_boot=4000, seed=1)
    assert r["annual_lo"] <= r["annual_mean"] <= r["annual_hi"]
    boot_centre = (r["annual_lo"] + r["annual_hi"]) / 2
    assert abs(boot_centre - r["annual_mean"]) < 0.5 * (r["annual_hi"] - r["annual_lo"])


def test_short_period_still_has_an_interval():
    """На четырёх сутках интервал не схлопывается в точку."""
    r = block_bootstrap_annual(pd.Series([10.0, 50.0, 20.0, 80.0]), n_boot=500)
    assert r["annual_lo"] < r["annual_hi"]


@pytest.mark.parametrize("category", [5, 6])
def test_perfect_forecast_saves_no_deviation_cost(category):
    """
    При идеальном прогнозе отклонений от плана нет ни с накопителем, ни без.

    Прежде счёт «без накопителя» брал план, в который уже было вшито
    расписание накопителя, и накопителю засчитывалась экономия на
    отклонениях, которых без него не было бы.
    """
    frame = _frame()
    tariff = RuTariff(category=category, deviation_up_rate=2.0, deviation_down_rate=2.0)
    battery = battery_for_load(frame[ACTUAL_COLUMN].max())
    table = evaluate_sources(frame, tariff, battery, n_boot=50)["summary"].set_index("source")
    assert table.loc[ORACLE, "saved_deviation_cost"] == pytest.approx(0.0, abs=1e-6)


def test_monthly_savings_add_up_to_the_total():
    frame = _frame()
    tariff = RuTariff(category=6)
    battery = battery_for_load(frame[ACTUAL_COLUMN].max())
    res = evaluate_sources(frame, tariff, battery, n_boot=50)
    table = res["summary"].set_index("source")
    monthly = res["monthly"].groupby("source")["gross_savings"].sum()
    for source in table.index:
        assert monthly[source] == pytest.approx(table.loc[source, "gross_savings"], rel=1e-6, abs=1e-3)
