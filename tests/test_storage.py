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


# ══════════════════════════════════════════════════════════════════════════════
# ПРАЗДНИКИ В ТАРИФНОМ КАЛЕНДАРЕ
# ══════════════════════════════════════════════════════════════════════════════

def test_holiday_has_no_peak_zone():
    """
    В праздник пиковой зоны нет, как и в выходной.

    Прежде тарифный модуль накопителя праздников не знал: 12 июня получало
    пиковую зону, и эти часы входили в плату за мощность, хотя генератор и
    признаки моделей считали день праздничным.
    """
    from optimization.storage import build_zone_list

    # Понедельник 00:00, первые сутки — праздник, вторые — обычный рабочий день.
    holidays = [True] * 24 + [False] * 24
    zones = build_zone_list(48, start_hour=0, start_weekday=0, holidays=holidays)

    assert "peak" not in zones[:24]
    assert zones[24 + 8] == "peak" and zones[24 + 18] == "peak"
    assert zones[:24] == build_zone_list(24, start_hour=0, start_weekday=5)


def test_holiday_mask_length_must_match_the_horizon():
    from optimization.storage import build_zone_list

    with pytest.raises(ValueError, match="Маска праздников"):
        build_zone_list(48, holidays=[False] * 24)


def test_holiday_flags_follow_the_actual_dates():
    """Признак праздника берётся по фактической дате, в том числе при старте в середине суток."""
    import pandas as pd

    from data.generator import holiday_flags

    stamps = pd.date_range("2025-06-11 12:00", periods=48, freq="h")
    flags = holiday_flags(stamps)

    assert not flags[:12].any(), "11 июня — рабочий день"
    assert flags[12:36].all(), "12 июня — праздник"
    assert not flags[36:].any()


def test_subperiod_keeps_its_own_calendar():
    """
    Подпериод получает свою календарную привязку, а не привязку начала ряда.

    Робастный отбор порога делит валидацию на подпериоды, которые начинаются в
    произвольный час и день недели. С привязкой начала всего ряда их тарифные
    зоны сдвигались относительно данных.
    """
    from optimization.storage import _calendar_slice, build_zone_list

    n = 24 * 20
    holidays = [False] * n
    holidays[200:224] = [True] * 24
    full = build_zone_list(n, start_hour=13, start_weekday=2, holidays=holidays)

    for lo, hi in ((0, 120), (117, 301), (301, n)):
        kw = _calendar_slice({"start_hour": 13, "start_weekday": 2, "holidays": holidays}, lo, hi)
        part = build_zone_list(hi - lo, kw["start_hour"], kw["start_weekday"], kw["holidays"])
        assert part == full[lo:hi], f"подпериод [{lo}:{hi}] сдвинут"


def test_robust_threshold_selection_uses_subperiod_calendars():
    """
    Робастный отбор порога оценивает каждый подпериод в его собственном календаре.

    Эталон считается вручную: перебор на каждом подпериоде с правильной
    привязкой и выбор порога с наилучшим наихудшим результатом. Отбор обязан
    дать тот же порог.
    """
    import numpy as np
    from optimization.storage import (
        _calendar_slice, select_shaving_threshold, sweep_shaving_threshold,
    )

    rng = np.random.RandomState(0)
    n = 24 * 28
    h = np.arange(n)
    actual = (1000 + 300 * np.sin(2 * np.pi * ((h + 13) % 24 - 6) / 24)
              + 400 * (((h + 13) % 24 >= 18) & ((h + 13) % 24 < 21)) + rng.normal(0, 60, n))
    forecast = actual + rng.normal(0, 120, n)
    holidays = [False] * n
    holidays[100:124] = [True] * 24

    kwargs = dict(capacity=800.0, max_power=300.0, battery_cost_rub=8_000_000.0,
                  start_hour=13, start_weekday=3, holidays=holidays)
    quantiles = (0.6, 0.7, 0.8, 0.9)

    bounds = np.linspace(0, n, 3).astype(int)
    worst = {q: [] for q in quantiles}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for row in sweep_shaving_threshold(forecast[lo:hi], actual[lo:hi], quantiles,
                                           **_calendar_slice(kwargs, lo, hi)):
            worst[row["shave_quantile"]].append(row["net_savings"])
    expected = max(quantiles, key=lambda q: min(worst[q]))

    chosen = select_shaving_threshold(forecast, actual, quantiles, n_subperiods=2, **kwargs)
    assert chosen == expected


# ══════════════════════════════════════════════════════════════════════════════
# ВЕРХНЯЯ ГРАНИЦА ЭКОНОМИИ
# ══════════════════════════════════════════════════════════════════════════════

def _lp_case(seed=0, n_days=21):
    import numpy as np
    rng = np.random.RandomState(seed)
    h = np.arange(n_days * 24)
    actual = (1000 + 300 * np.sin(2 * np.pi * (h % 24 - 6) / 24)
              + 400 * ((h % 24 >= 18) & (h % 24 < 21)) + rng.normal(0, 60, len(h)))
    kwargs = dict(capacity=900.0, max_power=300.0, battery_cost_rub=12_000_000.0,
                  round_trip_efficiency=0.88, cycle_cost_per_kwh=3.3,
                  min_soc=0.1, max_soc=0.9, start_hour=0, start_weekday=0)
    return actual, kwargs, rng


def test_perfect_foresight_bounds_every_policy():
    """
    Оптимум при известном будущем не хуже ни одного правила управления.

    Прежде границей считался идеальный прогноз с порогом по умолчанию, и строка
    с настроенным порогом его превосходила: недобор выходил отрицательным.
    Здесь граница сверяется с десятками вариантов — разные прогнозы, пороги и
    обе стратегии.
    """
    import numpy as np
    from optimization.storage import perfect_foresight_optimum, simulate_storage

    actual, kw, rng = _lp_case()
    bound = perfect_foresight_optimum(actual, **kw).net_savings

    for noise in (0.0, 40.0, 150.0):
        forecast = actual + rng.normal(0, noise, len(actual)) if noise else actual
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            res = simulate_storage(forecast=forecast, actual=actual, policy="peak_shaving",
                                   shave_quantile=q, **kw)
            assert res.net_savings <= bound + 1e-6, (noise, q, res.net_savings, bound)
        res = simulate_storage(forecast=forecast, actual=actual, policy="tariff", **kw)
        assert res.net_savings <= bound + 1e-6


def test_perfect_foresight_schedule_is_physical():
    """Расписание соблюдает ёмкость, мощность и запрет отдачи в сеть."""
    import numpy as np
    from optimization.storage import perfect_foresight_optimum

    actual, kw, _ = _lp_case(seed=1)
    res = perfect_foresight_optimum(actual, **kw)
    soc = np.asarray(res.battery_levels[1:])
    grid = np.asarray(res.energy_from_grid)

    assert soc.min() >= kw["min_soc"] * kw["capacity"] - 1e-6
    assert soc.max() <= kw["max_soc"] * kw["capacity"] + 1e-6
    assert grid.min() >= -1e-6, "накопитель отдал энергию в сеть"
    assert np.abs(np.diff(np.concatenate([[res.battery_levels[0]], soc]))).max() <= kw["max_power"] + 1e-6


def test_perfect_foresight_does_not_invent_money():
    """
    Если циклировать невыгодно, оптимум — бездействие, и экономия равна минус
    расходам на обслуживание.

    Начальный заряд ставится на нижнюю границу: иначе оптимум законно разрядит
    исходные 50% ёмкости и получит энергию без заряда. Симуляция может сделать
    то же самое, поэтому граница обязана это допускать.
    """
    from optimization.storage import perfect_foresight_optimum

    actual, kw, _ = _lp_case(seed=2)
    kw.update(cycle_cost_per_kwh=1_000.0)
    res = perfect_foresight_optimum(actual, demand_charge_rub_per_kw_month=0.0,
                                    initial_soc=kw["min_soc"], **kw)

    assert res.total_energy_cycled < 1e-6
    assert res.net_savings == pytest.approx(-res.om_cost, abs=1e-6)


def test_forecast_comparison_reports_non_negative_shortfall():
    """В сравнении источников прогноза ни одна строка не превосходит верхнюю границу."""
    import numpy as np
    from optimization.storage import UPPER_BOUND_KEY, compare_forecast_sources

    actual, kw, rng = _lp_case(seed=3)
    kw.pop("min_soc"); kw.pop("max_soc")
    results = compare_forecast_sources(
        actual=actual, forecasts={"шумный": actual + rng.normal(0, 120, len(actual))},
        min_soc=0.1, max_soc=0.9, **kw)

    bound = results[UPPER_BOUND_KEY].net_savings
    assert all(r.net_savings <= bound + 1e-6 for r in results.values())


def test_perfect_foresight_never_exports_even_when_it_would_pay():
    """
    Отдача в сеть запрещена и тогда, когда она была бы выгодна.

    Мощный накопитель с дешёвым циклом мог бы заряжаться ночью и продавать
    энергию в пиковые часы сверх собственной нагрузки. Инвертор этого не
    умеет, и симуляция этого не допускает — граница тоже не должна.
    """
    import numpy as np
    from optimization.storage import perfect_foresight_optimum

    actual, kw, _ = _lp_case(seed=4)
    kw.update(capacity=20_000.0, max_power=5_000.0, cycle_cost_per_kwh=0.01)
    res = perfect_foresight_optimum(actual, **kw)

    assert np.asarray(res.energy_from_grid).min() >= -1e-6
