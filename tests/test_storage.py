# -*- coding: utf-8 -*-
"""
Тесты экономической модели накопителя.

Ключевые инварианты: тарифный календарь привязан к реальному времени,
решения принимаются по прогнозу, а счёт выставляется по факту, и ошибка
прогноза не может быть выгоднее его отсутствия.
"""

import numpy as np
import pytest

from optimization.storage import (
    _get_zone, build_price_vector, build_zone_list,
    simulate_storage, compare_strategies, compare_forecast_sources,
)

BATTERY_COST = 45_000_000.0
COMMON = dict(
    capacity=4500.0, max_power=2250.0, round_trip_efficiency=0.95,
    cycle_cost_per_kwh=0.06, battery_cost_rub=BATTERY_COST,
    tariff_night=4.08, tariff_half_peak=7.87, tariff_peak=11.24,
    demand_charge_rub_per_kw_month=950.0, annual_om_share=0.015,
)


def _synthetic_load(n=240, seed=0):
    """Суточный профиль с вечерним пиком и шумом."""
    rng = np.random.default_rng(seed)
    hours = np.arange(n) % 24
    base = 5000 + 2500 * np.exp(-((hours - 19) ** 2) / 8.0)
    return base + rng.normal(0, 150, size=n)


# ── Тарифные зоны ─────────────────────────────────────────────────────────────

def test_zone_boundaries_weekday():
    """
    Зоны соответствуют порядку, действующему в России.

    Пиковая зона — это утренний и вечерний максимумы потребления
    (07–10 и 17–21), а не середина дня. Ранее в проекте пиковая и полупиковая
    зоны были переставлены местами, из-за чего накопитель разряжался в часы
    самого дешёвого из дневных тарифов.
    """
    assert _get_zone(3, 0) == "night"       # ночная 23–07
    assert _get_zone(23, 0) == "night"
    assert _get_zone(8, 0) == "peak"        # утренний пик 07–10
    assert _get_zone(18, 0) == "peak"       # вечерний пик 17–21
    assert _get_zone(12, 0) == "day"        # полупиковая 10–17
    assert _get_zone(22, 0) == "day"        # полупиковая 21–23
    # Границы включительно/исключительно
    assert _get_zone(7, 0) == "peak"
    assert _get_zone(10, 0) == "day"
    assert _get_zone(17, 0) == "peak"
    assert _get_zone(21, 0) == "day"


def test_weekend_has_no_peak_zone():
    """В выходные пиковая зона не применяется — только ночь и день."""
    for hour in range(24):
        assert _get_zone(hour, 5) in ("night", "day")
        assert _get_zone(hour, 6) in ("night", "day")


def test_price_vector_respects_calendar_offset():
    """
    Сдвиг начала ряда должен сдвигать тарифные зоны.

    Это защита от бага, при котором симуляция считала, что ряд начинается
    в понедельник в 00:00, тогда как тестовые данные начинались в пятницу
    в полдень: все зоны уезжали на 12 часов и «ночная зарядка» приходилась
    на дневной пик.
    """
    zones_midnight = build_zone_list(24, start_hour=0, start_weekday=0)
    zones_morning = build_zone_list(24, start_hour=8, start_weekday=0)

    assert zones_midnight[0] == "night"     # 00:00 — ночная зона
    assert zones_morning[0] == "peak"       # 08:00 — утренний пик
    assert zones_midnight != zones_morning

    prices_midnight = build_price_vector(24, start_hour=0, start_weekday=0)
    prices_morning = build_price_vector(24, start_hour=8, start_weekday=0)
    assert not np.allclose(prices_midnight, prices_morning)

    # Смещение по дню недели тоже должно учитываться: в субботу пиковой зоны нет.
    zones_saturday = build_zone_list(24, start_hour=8, start_weekday=5)
    assert "peak" not in zones_saturday


# ── Защита от ошибок вызова ───────────────────────────────────────────────────

def test_battery_cost_is_required():
    """Тихий дефолт стоимости занижал бы O&M и срок окупаемости в разы."""
    with pytest.raises(TypeError, match="battery_cost_rub"):
        simulate_storage(_synthetic_load(48), capacity=100, max_power=50)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError, match="policy"):
        simulate_storage(_synthetic_load(48), policy="magic", **COMMON)


def test_forecast_and_actual_length_mismatch_rejected():
    with pytest.raises(ValueError, match="не совпадают"):
        simulate_storage(_synthetic_load(48), actual=_synthetic_load(24), **COMMON)


# ── Физика и экономика ────────────────────────────────────────────────────────

def test_soc_stays_within_bounds():
    load = _synthetic_load(240)
    res = simulate_storage(load, min_soc=0.25, max_soc=0.75,
                           start_hour=0, start_weekday=0, **COMMON)
    levels = np.array(res.battery_levels)
    assert levels.min() >= -1e-6
    assert levels.max() <= COMMON["capacity"] + 1e-6


def test_zero_power_battery_yields_no_savings():
    """Батарея нулевой мощности не может дать положительную чистую экономию."""
    load = _synthetic_load(240)
    kwargs = dict(COMMON)
    kwargs["max_power"] = 0.0
    res = simulate_storage(load, start_hour=0, start_weekday=0, **kwargs)

    assert res.total_energy_cycled == pytest.approx(0.0)
    assert res.gross_savings == pytest.approx(0.0, abs=1e-6)
    # Остаются только расходы: O&M за горизонт.
    assert res.net_savings < 0


def test_efficiency_losses_are_accounted():
    """
    Из сети берётся больше, чем попадает в батарею: заряд делится на КПД.
    """
    load = np.full(240, 5000.0)
    res = simulate_storage(load, round_trip_efficiency=0.81,   # односторонний 0.9
                           start_hour=0, start_weekday=0,
                           **{k: v for k, v in COMMON.items()
                              if k != "round_trip_efficiency"})
    charge_hours = [i for i, a in enumerate(res.actions) if a == "charge"]
    assert charge_hours, "Ожидалась хотя бы одна зарядка в ночной зоне"
    i = charge_hours[0]
    # Потребление из сети в час зарядки выше базовой нагрузки.
    assert res.energy_from_grid[i] > load[i]


def test_night_charging_never_raises_billing_peak():
    """
    Ночной зарядный ток не должен ухудшать плату за мощность.

    Плата берётся по максимуму в биллинговые пиковые часы, поэтому рост
    потребления ночью на неё не влияет, а экономия не может стать отрицательной.
    """
    load = _synthetic_load(720)
    res = simulate_storage(load, min_soc=0.10, max_soc=0.90,
                           start_hour=0, start_weekday=0, **COMMON)
    assert res.peak_after_kw <= res.peak_before_kw + 1e-6
    assert res.demand_charge_savings >= 0


def test_peak_shaving_targets_maximum_better_than_calendar_policy():
    """
    Прогноз-зависимая срезка снижает плату за мощность лучше календарной.

    Календарная стратегия разряжает батарею в первые же часы пикового окна и
    к моменту реального максимума нагрузки оказывается разряженной. Срезка по
    прогнозу расходует заряд там, где превышение порога наибольшее, — то есть
    целится именно в тот час, по которому выставляется счёт за мощность.
    """
    load = _synthetic_load(720, seed=8)
    kwargs = dict(min_soc=0.10, max_soc=0.90, start_hour=0, start_weekday=0, **COMMON)

    res_tariff = simulate_storage(load, actual=load, policy="tariff", **kwargs)
    res_shave = simulate_storage(load, actual=load, policy="peak_shaving", **kwargs)

    assert res_shave.peak_after_kw <= res_tariff.peak_after_kw
    assert res_shave.demand_charge_savings >= res_tariff.demand_charge_savings


# ── Прогноз против факта ──────────────────────────────────────────────────────

def test_decisions_follow_forecast_costs_follow_actual():
    """
    Стоимость считается по факту, даже если прогноз сильно завышен.

    Базовая стоимость не должна зависеть от того, какой прогноз подан.
    """
    actual = _synthetic_load(240, seed=1)
    inflated = actual * 2.0

    res_good = simulate_storage(actual, actual=actual, policy="peak_shaving",
                                start_hour=0, start_weekday=0, **COMMON)
    res_bad = simulate_storage(inflated, actual=actual, policy="peak_shaving",
                               start_hour=0, start_weekday=0, **COMMON)

    assert res_good.baseline_cost == pytest.approx(res_bad.baseline_cost)


def test_perfect_forecast_is_not_worse_than_noisy_one():
    """
    Идеальный прогноз задаёт верхнюю границу эффекта.

    Если бы шумный прогноз систематически выигрывал, это означало бы ошибку
    в связке «прогноз → решение → деньги».
    """
    rng = np.random.default_rng(3)
    actual = _synthetic_load(480, seed=2)
    noisy = actual + rng.normal(0, 900, size=len(actual))

    res_oracle = simulate_storage(actual, actual=actual, policy="peak_shaving",
                                  min_soc=0.25, max_soc=0.75,
                                  start_hour=0, start_weekday=0, **COMMON)
    res_noisy = simulate_storage(noisy, actual=actual, policy="peak_shaving",
                                 min_soc=0.25, max_soc=0.75,
                                 start_hour=0, start_weekday=0, **COMMON)

    assert res_oracle.net_savings >= res_noisy.net_savings


def test_tariff_policy_ignores_forecast():
    """
    Календарная стратегия не использует прогноз — результат от него не зависит.

    Именно поэтому она не годится для оценки ценности прогнозной модели.
    """
    actual = _synthetic_load(240, seed=4)
    garbage = np.full_like(actual, 1.0)

    res_a = simulate_storage(actual, actual=actual, policy="tariff",
                             start_hour=0, start_weekday=0, **COMMON)
    res_b = simulate_storage(garbage, actual=actual, policy="tariff",
                             start_hour=0, start_weekday=0, **COMMON)

    assert res_a.net_savings == pytest.approx(res_b.net_savings)


def test_compare_forecast_sources_includes_oracle():
    actual = _synthetic_load(240, seed=5)
    rng = np.random.default_rng(6)
    forecasts = {"Модель": actual + rng.normal(0, 400, size=len(actual))}

    results = compare_forecast_sources(
        actual=actual, forecasts=forecasts,
        min_soc=0.25, max_soc=0.75, start_hour=0, start_weekday=0, **COMMON,
    )

    assert "Идеальный прогноз" in results
    assert "Модель" in results
    assert results["Идеальный прогноз"].forecast_source == "Идеальный прогноз"


def test_compare_strategies_orders_by_depth_of_discharge():
    """Более глубокий разряд прокачивает больше энергии."""
    load = _synthetic_load(480)
    res = compare_strategies(load, start_hour=0, start_weekday=0, **COMMON)

    assert res["Агрессивная"].total_energy_cycled > res["Консервативная"].total_energy_cycled
