# -*- coding: utf-8 -*-
"""
Счёт по ценовым категориям 3–6 и контроллеры накопителя под него.

Проверяется арифметика на рядах, где ответ известен заранее: постоянная
нагрузка, всплеск в окне и вне окна, всплеск в выходной, неполный месяц.
Для контроллеров — физичность расписания и порядок «нет накопителя ≤ MPC ≤
оптимум при известном будущем».
"""

import numpy as np
import pandas as pd
import pytest

from optimization.controllers import (
    Battery, evaluate_schedule, mpc_day_ahead, perfect_foresight_ru,
)
from optimization.tariffs_ru import (
    RuTariff, calendar_frame, daily_capacity_weights, monthly_bill,
)

# Апрель 2025: без федеральных праздников, 22 рабочих дня.
APRIL = pd.date_range("2025-04-01", periods=30 * 24, freq="h")


def _no_holidays(ts):
    return np.zeros(len(ts), dtype=bool)


def test_april_2025_has_22_working_days():
    cal = calendar_frame(APRIL, RuTariff(), _no_holidays(APRIL))
    assert cal.loc[cal["working"], "day"].nunique() == 22


def test_constant_load_bill_components():
    t = RuTariff(category=4, energy_price=3.0, gen_capacity_rate=1000.0,
                 net_capacity_rate=1200.0, net_energy_rate=0.4)
    bill = monthly_bill(np.full(len(APRIL), 100.0), APRIL, t, holidays=_no_holidays(APRIL))
    row = bill.iloc[0]
    energy = 100.0 * len(APRIL)
    assert row["energy_cost"] == pytest.approx(energy * 3.0)
    assert row["gen_capacity_kw"] == pytest.approx(100.0)
    assert row["gen_capacity_cost"] == pytest.approx(100.0 * 1000.0)
    assert row["net_capacity_cost"] == pytest.approx(100.0 * 1200.0)
    assert row["net_energy_cost"] == pytest.approx(energy * 0.4)


def _spiky(hour, weekday_only=True, value=300.0, base=100.0, ts=APRIL):
    load = np.full(len(ts), base)
    mask = ts.hour == hour
    if weekday_only:
        mask &= ts.dayofweek < 5
    load[mask] = value
    return load


def test_spike_inside_the_window_sets_both_capacities():
    """Суточный максимум в окне СО задаёт и сетевую, и генерирующую мощность."""
    t = RuTariff(category=4)
    row = monthly_bill(_spiky(18), APRIL, t, holidays=_no_holidays(APRIL)).iloc[0]
    assert row["gen_capacity_kw"] == pytest.approx(300.0)
    assert row["net_capacity_kw"] == pytest.approx(300.0)


def test_spike_outside_the_window_is_free():
    """Всплеск в 22 ч вне окна СО мощность не меняет."""
    t = RuTariff(category=4)
    row = monthly_bill(_spiky(22), APRIL, t, holidays=_no_holidays(APRIL)).iloc[0]
    assert row["gen_capacity_kw"] == pytest.approx(100.0)
    assert row["net_capacity_kw"] == pytest.approx(100.0)


def test_weekend_spike_is_free():
    load = np.full(len(APRIL), 100.0)
    load[(APRIL.hour == 18) & (APRIL.dayofweek == 5)] = 500.0
    row = monthly_bill(load, APRIL, RuTariff(category=4), holidays=_no_holidays(APRIL)).iloc[0]
    assert row["net_capacity_kw"] == pytest.approx(100.0)


def test_capacity_is_an_average_over_working_days():
    """Всплеск в один рабочий день из 22 даёт 1/22 прироста, а не весь прирост."""
    load = np.full(len(APRIL), 100.0)
    load[(APRIL.normalize() == pd.Timestamp("2025-04-01")) & (APRIL.hour == 18)] = 320.0
    row = monthly_bill(load, APRIL, RuTariff(category=4), holidays=_no_holidays(APRIL)).iloc[0]
    assert row["net_capacity_kw"] == pytest.approx(100.0 + 220.0 / 22)


def test_region_load_decides_the_peak_hour():
    """Час пика берётся по ряду субъекта, а не по нагрузке потребителя."""
    region = _spiky(9)
    own = _spiky(18)
    row = monthly_bill(own, APRIL, RuTariff(category=4), region_load=region,
                       holidays=_no_holidays(APRIL)).iloc[0]
    assert row["gen_capacity_kw"] == pytest.approx(100.0)
    assert row["net_capacity_kw"] == pytest.approx(300.0)


def test_partial_month_is_prorated():
    ts = APRIL[: 10 * 24]
    row = monthly_bill(np.full(len(ts), 100.0), ts, RuTariff(category=4, gen_capacity_rate=900.0),
                       holidays=_no_holidays(ts)).iloc[0]
    assert row["gen_capacity_cost"] == pytest.approx(100.0 * 900.0 * 10 / 30)


def test_category_three_has_no_network_capacity():
    t = RuTariff(category=3, net_single_rate=2.5)
    row = monthly_bill(_spiky(18), APRIL, t, holidays=_no_holidays(APRIL)).iloc[0]
    assert row["net_capacity_cost"] == 0.0
    assert row["net_energy_cost"] == pytest.approx(row["energy_kwh"] * 2.5)


def test_category_five_charges_deviations_both_ways():
    t = RuTariff(category=5, deviation_up_rate=0.3, deviation_down_rate=0.1)
    load = np.full(len(APRIL), 100.0)
    plan = load.copy()
    plan[0], plan[1] = 90.0, 110.0          # недобор плана и перебор
    row = monthly_bill(load, APRIL, t, plan=plan, holidays=_no_holidays(APRIL)).iloc[0]
    assert row["deviation_cost"] == pytest.approx(10 * 0.3 + 10 * 0.1)


def test_daily_weights_add_up_to_the_monthly_rate():
    t = RuTariff(category=4, gen_capacity_rate=1000.0, net_capacity_rate=500.0)
    w = daily_capacity_weights(APRIL, t, _no_holidays(APRIL))
    assert w.sum() == pytest.approx(1500.0)
    assert (w > 0).sum() == 22


def test_unsupported_category_is_rejected():
    with pytest.raises(ValueError):
        RuTariff(category=2)


# ══════════════════════════════════════════════════════════════════════════════
# КОНТРОЛЛЕРЫ
# ══════════════════════════════════════════════════════════════════════════════

TWO_WEEKS = pd.date_range("2025-04-07", periods=14 * 24, freq="h")
BATTERY = Battery(capacity=200.0, max_power=60.0, capex_rub=2_000_000.0)


def _load(seed=0):
    rng = np.random.RandomState(seed)
    h = TWO_WEEKS.hour.to_numpy()
    shape = 300 + 120 * np.exp(-((h - 18) ** 2) / 6) + 60 * np.exp(-((h - 9) ** 2) / 4)
    return shape + rng.normal(0, 10, len(h))


def _check_physical(schedule, actual, battery):
    assert (schedule["grid"] >= -1e-6).all(), "отдача в сеть запрещена"
    lo, hi = battery.min_soc * battery.capacity, battery.max_soc * battery.capacity
    assert (schedule["soc"] >= lo - 1e-6).all() and (schedule["soc"] <= hi + 1e-6).all()
    assert (schedule["charged"] <= battery.max_power + 1e-6).all()


def test_perfect_foresight_bounds_mpc_and_doing_nothing():
    t = RuTariff(category=4)
    actual = _load()
    hol = _no_holidays(TWO_WEEKS)
    noisy = actual + np.random.RandomState(1).normal(0, 25, len(actual))

    pf = perfect_foresight_ru(actual, TWO_WEEKS, t, BATTERY, hol)
    mpc_exact = mpc_day_ahead(actual, actual, TWO_WEEKS, t, BATTERY, hol)
    mpc_noisy = mpc_day_ahead(noisy, actual, TWO_WEEKS, t, BATTERY, hol)
    for s in (pf, mpc_exact, mpc_noisy):
        _check_physical(s, actual, BATTERY)

    def gross_minus_wear(s):
        r = evaluate_schedule(s["grid"], s["charged"], actual, TWO_WEEKS, t, BATTERY,
                              holidays=hol)
        return r["gross_savings"] - r["degradation"]

    v_pf, v_exact, v_noisy = map(gross_minus_wear, (pf, mpc_exact, mpc_noisy))
    assert v_pf >= v_exact - 1e-6 >= -1e-6
    assert v_pf >= v_noisy - 1e-6
    assert v_exact > 0, "при известной нагрузке срезать пики в окне выгодно"


def test_mpc_shaves_the_network_capacity():
    t = RuTariff(category=4)
    actual = _load()
    hol = _no_holidays(TWO_WEEKS)
    mpc = mpc_day_ahead(actual, actual, TWO_WEEKS, t, BATTERY, hol)
    r = evaluate_schedule(mpc["grid"], mpc["charged"], actual, TWO_WEEKS, t, BATTERY,
                          holidays=hol)
    assert r["saved_net_capacity_cost"] > 0


def test_mpc_does_nothing_without_an_incentive():
    """Плоская цена и нулевые ставки мощности: цикл только теряет на КПД и износе."""
    t = RuTariff(category=4, gen_capacity_rate=0.0, net_capacity_rate=0.0)
    actual = _load()
    mpc = mpc_day_ahead(actual, actual, TWO_WEEKS, t, BATTERY, _no_holidays(TWO_WEEKS))
    assert mpc["charged"].sum() == pytest.approx(0.0, abs=1e-6)


def test_mpc_returns_to_the_starting_charge_each_day():
    """Условие на конец суток не даёт «проесть» начальный заряд."""
    t = RuTariff(category=4)
    actual = _load()
    mpc = mpc_day_ahead(actual, actual, TWO_WEEKS, t, BATTERY, _no_holidays(TWO_WEEKS))
    end_of_day = mpc["soc"][23::24]
    assert (end_of_day >= BATTERY.initial_soc * BATTERY.capacity - 1e-6).all()
