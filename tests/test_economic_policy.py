# -*- coding: utf-8 -*-
"""
Тесты экономической состоятельности управления накопителем, подбора
регуляризации линейной модели и изоляции результатов между прогонами.

Все три проверки закрывают дефекты, обнаруженные на реальном fast-прогоне:
контроллер циклировал батарею в убыток, Ridge был единственной моделью без
подбора гиперпараметра, а результаты разных режимов затирали друг друга.
"""

import numpy as np
import pandas as pd
import pytest

from optimization.storage import (
    cycle_margin_rub_per_kwh, simulate_storage, build_zone_list,
)
from utils import reporting

BATTERY_COST = 42_600_000.0
COMMON = dict(
    capacity=1704.0, max_power=426.0, round_trip_efficiency=0.88,
    cycle_cost_per_kwh=5.21, battery_cost_rub=BATTERY_COST,
    tariff_night=4.08, tariff_half_peak=7.87, tariff_peak=11.24,
    demand_charge_rub_per_kw_month=950.0, annual_om_share=0.015,
    start_hour=0, start_weekday=0,
)


def _load(n=480, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.arange(n) % 24
    base = 900 + 500 * np.exp(-((hours - 19) ** 2) / 8.0)
    return base + rng.normal(0, 40, size=n)


# ══════════════════════════════════════════════════════════════════════════════
# ЭКОНОМИКА ЦИКЛА
# ══════════════════════════════════════════════════════════════════════════════

def test_cycle_margin_matches_manual_calculation():
    """Маржа считается как выручка минус стоимость заряда и износа."""
    eff = np.sqrt(0.88)
    margin = cycle_margin_rub_per_kwh(11.24, 4.08, eff, 5.21)
    expected = 11.24 * eff - 4.08 / eff - 5.21
    assert margin == pytest.approx(expected)


def test_half_peak_cycle_is_unprofitable_at_current_tariffs():
    """
    При действующих тарифах и стоимости LFP цикл «ночь → полупик» убыточен.

    Это не гипотеза, а арифметика: удельная деградация сопоставима с тарифным
    спредом. Тест фиксирует факт, на котором держится ограничение разряда.
    """
    eff = np.sqrt(0.88)
    peak = cycle_margin_rub_per_kwh(11.24, 4.08, eff, 5.21)
    half = cycle_margin_rub_per_kwh(7.87, 4.08, eff, 5.21)

    assert peak > 0, "цикл в пиковую зону должен быть прибыльным"
    assert half < 0, "цикл в полупиковую зону при этих параметрах убыточен"


def test_gate_prevents_discharging_in_half_peak():
    """
    С проверкой маржи батарея не разряжается в убыточной полупиковой зоне.

    Без неё контроллер циклирует ради снижения счёта за энергию, тратя на
    износ больше, чем экономит.
    """
    load = _load(480, seed=1)
    zones = build_zone_list(len(load), 0, 0)

    gated = simulate_storage(load, actual=load, policy="peak_shaving",
                             min_soc=0.25, max_soc=0.75,
                             require_positive_margin=True, **COMMON)
    ungated = simulate_storage(load, actual=load, policy="peak_shaving",
                               min_soc=0.25, max_soc=0.75,
                               require_positive_margin=False, **COMMON)

    half_peak_discharges = sum(
        1 for z, a in zip(zones, gated.actions) if z == "day" and a == "discharge")
    assert half_peak_discharges == 0, "разряд в полупиковой зоне должен быть запрещён"

    assert gated.degradation_cost < ungated.degradation_cost, (
        "проверка маржи обязана снизить износ"
    )
    assert gated.net_savings > ungated.net_savings, (
        "запрет убыточных циклов должен улучшать чистый результат"
    )


def test_gate_still_allows_peak_zone_discharge():
    """
    В пиковой зоне разряд разрешён: он снижает плату за мощность, а она
    многократно перекрывает потери одного цикла.
    """
    load = _load(720, seed=2)
    res = simulate_storage(load, actual=load, policy="peak_shaving",
                           min_soc=0.25, max_soc=0.75,
                           require_positive_margin=True, **COMMON)
    assert res.n_discharge_hours > 0, "разряд в пиковой зоне должен сохраняться"
    assert res.peak_after_kw <= res.peak_before_kw + 1e-6


# ══════════════════════════════════════════════════════════════════════════════
# ПОДБОР РЕГУЛЯРИЗАЦИИ RIDGE
# ══════════════════════════════════════════════════════════════════════════════

def test_ridge_selects_alpha_on_validation():
    """
    Alpha выбирается по валидации, а не берётся фиксированной.

    Иначе линейная модель — единственная в сравнении без настройки, и её
    проигрыш говорит о неудачном гиперпараметре, а не о свойствах метода.
    """
    from models.baseline import build_linear_regression

    rng = np.random.default_rng(3)
    n, T, F, H = 300, 12, 4, 3
    X = rng.normal(size=(n, T, F)).astype(np.float32)
    # Целевая переменная слабо связана со входом и сильно зашумлена: без
    # регуляризации модель переобучится, поэтому оптимум окажется не на краю.
    Y = (X[:, -1, :1] * 2.0 + rng.normal(0, 3.0, size=(n, 1))).repeat(H, axis=1)

    model = build_linear_regression()
    model.fit(X[:200], Y[:200], X_val=X[200:260], Y_val=Y[200:260])

    est = model.estimator
    assert est.alpha_ is not None, "alpha должна быть выбрана"
    assert est.alpha_ in est.alphas, "выбранное значение обязано быть из сетки"

    pred = model.predict(X[260:])
    assert pred.shape == (40, H)
    assert np.isfinite(pred).all()


def test_ridge_falls_back_without_validation():
    """Без валидации модель обучается, но предупреждает о произвольной alpha."""
    from models.baseline import build_linear_regression

    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 8, 3)).astype(np.float32)
    Y = rng.normal(size=(60, 2)).astype(np.float32)

    model = build_linear_regression()
    model.fit(X, Y)
    assert model.estimator.alpha_ is not None
    assert np.isfinite(model.predict(X)).all()


# ══════════════════════════════════════════════════════════════════════════════
# ИЗОЛЯЦИЯ РЕЗУЛЬТАТОВ МЕЖДУ ПРОГОНАМИ
# ══════════════════════════════════════════════════════════════════════════════

def test_metrics_csv_separates_modes(tmp_path):
    """
    Прогоны разных режимов с одним сидом не затирают друг друга.

    Без колонки режима строки smoke и optimal неразличимы, и последний
    прогон молча уничтожает предыдущий.
    """
    out = str(tmp_path)
    metrics = {"XGBoost": {"MAE": 10.0}, "LSTM": {"MAE": 20.0}}

    reporting.export_metrics(metrics, out, seed=42, split="test",
                             run_meta={"mode": "smoke", "scenario": "current"})
    reporting.export_metrics({"XGBoost": {"MAE": 5.0}, "LSTM": {"MAE": 6.0}},
                             out, seed=42, split="test",
                             run_meta={"mode": "optimal", "scenario": "current"})

    df = pd.read_csv(tmp_path / "metrics.csv")
    assert set(df["mode"]) == {"smoke", "optimal"}, "строки режимов обязаны сосуществовать"
    assert len(df) == 4

    # Повторный прогон того же режима перезаписывает свои строки, не плодя дубли.
    reporting.export_metrics({"XGBoost": {"MAE": 4.0}, "LSTM": {"MAE": 5.0}},
                             out, seed=42, split="test",
                             run_meta={"mode": "optimal", "scenario": "current"})
    df2 = pd.read_csv(tmp_path / "metrics.csv")
    assert len(df2) == 4
    assert df2[(df2["mode"] == "optimal") & (df2["model"] == "XGBoost")]["MAE"].iloc[0] == 4.0


def test_plots_are_written_directly_into_run_dir():
    """
    Графики пишутся в каталог прогона сразу, а не копируются туда после.

    Копирование общего каталога затягивало в отчёт изображения посторонних
    запусков, поэтому механизм заменён на перенаправление путей вывода.
    """
    from utils import reporting

    assert not hasattr(reporting, "copy_plots_to_run"), (
        "копирование общего каталога графиков должно быть удалено"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ПОДБОР ПОРОГА СРЕЗКИ
# ══════════════════════════════════════════════════════════════════════════════

def _peak_load(n_days=30, seed=0):
    """Ряд с выраженными вечерними пиками — на нём срезка имеет смысл."""
    rng = np.random.RandomState(seed)
    h = np.arange(n_days * 24)
    base = 1000.0 + 300.0 * np.sin(2 * np.pi * (h % 24 - 6) / 24.0)
    peaks = 400.0 * ((h % 24 >= 18) & (h % 24 < 21))
    return h, base + peaks + rng.normal(0, 60.0, size=len(h))


def _imperfect_forecast(h, seed=1):
    """Прогноз, ошибающийся и во времени пика, и в его величине."""
    rng = np.random.RandomState(seed)
    base = 1000.0 + 300.0 * np.sin(2 * np.pi * (h % 24 - 6) / 24.0)
    return base + 400.0 * ((h % 24 >= 17) & (h % 24 < 20)) + rng.normal(0, 120.0, len(h))


_BATTERY = dict(capacity=800.0, max_power=300.0, battery_cost_rub=8_000_000.0)


def test_optimum_is_interior_when_the_forecast_errs():
    """
    При ошибающемся прогнозе оптимальный порог лежит внутри сетки.

    Механизмы на концах разные: низкий порог заставляет разряжаться по неверно
    предсказанным пикам, расходуя ресурс впустую, высокий — почти не срезает
    пик. Если бы оптимум оказался на краю, перебор ничего бы не выбирал.
    """
    from optimization.storage import sweep_shaving_threshold

    h, actual = _peak_load()
    rows = sweep_shaving_threshold(forecast=_imperfect_forecast(h), actual=actual,
                                   label="тест", **_BATTERY)

    savings = [r["net_savings"] for r in rows]
    best = int(np.argmax(savings))
    assert 0 < best < len(savings) - 1, f"оптимум на краю сетки: {savings}"

    # Износ монотонно падает с ростом порога: разряжаемся реже.
    degradation = [r["degradation_cost"] for r in rows]
    assert degradation[0] > degradation[-1]


def test_threshold_depends_on_forecast_quality():
    """
    Оптимальный порог определяется качеством прогноза, а не только формой ряда.

    Направление сдвига не универсально и зависит от того, ограничена ли батарея
    по ёмкости. При точном прогнозе и малом накопителе выгоднее тратить заряд
    только на саму вершину — порог выше. При большом накопителе, наоборот,
    выгоден агрессивный разряд. Измерено: на батарее 800 кВт·ч оптимум при
    идеальном прогнозе 0.80, при ошибающемся 0.75; на батарее 2000 кВт·ч
    идеальный прогноз уводит оптимум к самому краю сетки.

    Поэтому проверяется не направление, а сам факт зависимости и цена
    неверного выбора: порог, подобранный под другое качество прогноза, стоит
    измеримых денег.
    """
    from optimization.storage import select_shaving_threshold, simulate_storage

    h, actual = _peak_load()
    forecast = _imperfect_forecast(h)

    q_perfect = select_shaving_threshold(actual, actual, **_BATTERY)
    q_noisy = select_shaving_threshold(forecast, actual, **_BATTERY)

    assert q_perfect != q_noisy, "выбор порога не реагирует на качество прогноза"

    matched = simulate_storage(forecast=forecast, actual=actual, policy="peak_shaving",
                               shave_quantile=q_noisy, **_BATTERY).net_savings
    mismatched = simulate_storage(forecast=forecast, actual=actual, policy="peak_shaving",
                                  shave_quantile=q_perfect, **_BATTERY).net_savings

    assert matched > mismatched, "порог под своё качество прогноза обязан быть не хуже"
    assert (matched - mismatched) / abs(matched) > 0.01, (
        f"разница {matched - mismatched:.0f} слишком мала, чтобы выбор порога имел смысл"
    )


def test_pipeline_tunes_the_threshold_on_validation():
    """
    Конвейер подбирает порог по валидации, а не по тесту.

    Та же ошибка для выбора модели в этой работе уже исправлялась. Проверка
    структурная: подбор обязан получать валидационный ряд, а перебор по тесту —
    оставаться отдельной верхней границей, а не источником решения.
    """
    import inspect
    import main as main_module

    src = inspect.getsource(main_module._run_storage_block)

    idx_select = src.index("select_shaving_threshold(")
    assert 'predict_original_scale(data, "val")' in src[:idx_select], (
        "порог выбирается раньше, чем построен валидационный прогноз"
    )
    assert 'raw_val' in src[:idx_select], "подбор не использует валидационный факт"
    assert 'label="тест (верхняя граница)"' in src, (
        "перебор по тесту должен быть помечен как верхняя граница"
    )
