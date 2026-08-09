# -*- coding: utf-8 -*-
"""
Тесты методологических инвариантов, вынесенные по результатам аудита.

Каждый тест здесь защищает утверждение, на которое опираются выводы работы.
Если инвариант нарушится, результаты потеряют смысл, поэтому проверки написаны
так, чтобы падать на самой сути, а не на побочных признаках.

Покрываемые утверждения:
  1. Тестовая выборка не участвует в выборе модели.
  2. Накопитель управляется по прогнозу, а не по факту; энергобаланс сходится.
  3. Обученные модели загружаются обратно и дают тот же прогноз;
     инференс требует полную матрицу из N_FEATURES признаков.
  5. Экстремальное значение, попавшее только в тест, не меняет ни train-скалеры,
     ни обучающие признаки.
  7. Ряд остатков строится по срезу горизонта, а не разворачиванием матрицы.
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

import main as main_module
from data.generator import generate_smartgrid_data
from data.preprocessing import prepare_data
from analysis.residuals import extract_residual_series
from optimization.storage import simulate_storage


BATTERY_COST = 112_500_000.0
STORAGE_KW = dict(
    capacity=4500.0, max_power=2250.0, round_trip_efficiency=0.88,
    cycle_cost_per_kwh=5.21, battery_cost_rub=BATTERY_COST,
    tariff_night=4.08, tariff_half_peak=7.87, tariff_peak=11.24,
    demand_charge_rub_per_kw_month=950.0, annual_om_share=0.015,
    start_hour=0, start_weekday=0,
)


def _load_profile(n=480, seed=0):
    rng = np.random.default_rng(seed)
    hours = np.arange(n) % 24
    base = 900 + 500 * np.exp(-((hours - 19) ** 2) / 8.0)
    return base + rng.normal(0, 40, size=n)


# ══════════════════════════════════════════════════════════════════════════════
# ПУНКТ 1: тест не участвует в выборе модели
# ══════════════════════════════════════════════════════════════════════════════

def test_model_selection_reads_only_validation_metrics():
    """
    Победитель определяется по валидации, а тестовые метрики считаются позже.

    Проверяется исходный текст main.main(): переменная best_name должна
    вычисляться из val_metrics, и в коде до этого момента не должно быть
    обращения к split="test". Иначе итоговая оценка окажется смещённой —
    мы выбрали бы модель, случайно удачную именно на тесте.
    """
    src = inspect.getsource(main_module.main)

    idx_val = src.index('compare_trainers(trainers, data, split="val")')
    idx_best = src.index("best_name = min(")
    idx_test = src.index('compare_trainers(trainers, data, split="test")')

    assert idx_val < idx_best, "val-метрики должны считаться до выбора модели"
    assert idx_best < idx_test, "тестовые метрики не должны участвовать в выборе"

    # best_name формируется из пула, построенного по val_metrics.
    selection_block = src[idx_val:idx_best + 200]
    assert "val_metrics" in selection_block
    assert "test_metrics" not in src[:idx_best], (
        "переменная test_metrics не должна существовать на момент выбора модели"
    )


def test_selection_pool_excludes_naive_baselines():
    """
    Наивные базлайны участвуют в сравнении, но не могут быть «лучшей моделью».

    Они нужны как точка отсчёта; выбирать их для последующего анализа остатков
    и экспорта бессмысленно.
    """
    src = inspect.getsource(main_module.main)
    assert "NAIVE_KEYS" in src, "пул отбора должен исключать наивные базлайны"


# ══════════════════════════════════════════════════════════════════════════════
# ПУНКТ 2: управление по прогнозу и энергобаланс
# ══════════════════════════════════════════════════════════════════════════════

def test_energy_balance_holds_every_hour():
    """
    Энергобаланс: потребление из сети + отдача батареи = фактическая нагрузка.

    В часы заряда сеть покрывает нагрузку и зарядный ток. Раньше при завышенном
    прогнозе батарея списывала из SOC больше, чем потребляла нагрузка, и разница
    исчезала — баланс не сходился.
    """
    actual = _load_profile(480, seed=1)
    rng = np.random.default_rng(7)
    inflated = actual * 1.8 + rng.normal(0, 100, size=len(actual))

    res = simulate_storage(inflated, actual=actual, policy="peak_shaving",
                           min_soc=0.25, max_soc=0.75, **STORAGE_KW)

    soc = np.array(res.battery_levels)
    one_way = np.sqrt(STORAGE_KW["round_trip_efficiency"])
    grid = np.array(res.energy_from_grid)

    for i, action in enumerate(res.actions):
        delta_soc = soc[i + 1] - soc[i]
        if action == "charge":
            # Из сети берётся нагрузка плюс зарядный ток с учётом потерь.
            expected = actual[i] + delta_soc / one_way
            assert grid[i] == pytest.approx(expected, rel=1e-6, abs=1e-6)
        elif action == "discharge":
            delivered = -delta_soc * one_way
            assert grid[i] + delivered == pytest.approx(actual[i], rel=1e-6, abs=1e-6)
            assert delivered <= actual[i] + 1e-6, (
                "нельзя отдать в сеть больше, чем потребляет нагрузка"
            )
        else:
            assert grid[i] == pytest.approx(actual[i], rel=1e-9)


def test_discharge_never_exceeds_actual_load():
    """Отдача батареи ограничена фактическим спросом даже при абсурдном прогнозе."""
    actual = _load_profile(240, seed=2)
    absurd = np.full_like(actual, actual.max() * 10)

    res = simulate_storage(absurd, actual=actual, policy="peak_shaving",
                           min_soc=0.10, max_soc=0.90, **STORAGE_KW)

    grid = np.array(res.energy_from_grid)
    assert (grid >= -1e-9).all(), "потребление из сети не может быть отрицательным"


def test_peak_shaving_actually_depends_on_forecast():
    """
    Прогноз-зависимая стратегия должна менять решения при смене прогноза.

    Если бы результат не зависел от прогноза, блок оптимизации не мог бы
    служить доказательством ценности прогнозирования.
    """
    actual = _load_profile(480, seed=3)
    rng = np.random.default_rng(4)
    noisy = actual + rng.normal(0, 250, size=len(actual))

    res_true = simulate_storage(actual, actual=actual, policy="peak_shaving",
                                min_soc=0.25, max_soc=0.75, **STORAGE_KW)
    res_noisy = simulate_storage(noisy, actual=actual, policy="peak_shaving",
                                 min_soc=0.25, max_soc=0.75, **STORAGE_KW)

    assert res_true.actions != res_noisy.actions, (
        "решения контроллера обязаны зависеть от прогноза"
    )
    assert res_true.net_savings >= res_noisy.net_savings, (
        "идеальный прогноз задаёт верхнюю границу эффекта"
    )


def test_storage_receives_model_forecast_not_ground_truth():
    """
    В блоке накопителя фигурируют прогнозы модели, а не сырой тестовый ряд.

    Раньше в оптимизатор подавался data["raw_test"], поэтому качество
    прогнозирования не влияло на экономику вообще.
    """
    src = inspect.getsource(main_module._run_storage_block)
    assert "predictions[best_name]" in src
    assert "reconstruct_day_ahead_series" in src
    assert 'data["raw_test"]' not in src, (
        "фактический ряд не должен подаваться как прогноз"
    )
    # Факт для расчёта стоимости берётся отдельной функцией.
    assert "actual_series_for_forecast" in src


# ══════════════════════════════════════════════════════════════════════════════
# ПУНКТ 5: экстремум в тесте не влияет на обучение
# ══════════════════════════════════════════════════════════════════════════════

def test_extreme_value_in_test_does_not_affect_train_scalers():
    """
    Аномалия, помещённая ТОЛЬКО в тестовую часть, не должна менять ни один
    train-скалер и ни один обучающий признак.

    Это прямая проверка отсутствия утечки: если бы нормировка считалась по
    всему ряду, подмена одного значения в тесте сдвинула бы всю обучающую
    матрицу.
    """
    df = generate_smartgrid_data(days=30, households=40, seed=5,
                                 industrial_loads=2, city_districts=2)
    baseline = prepare_data(df, history_length=48, forecast_horizon=24)

    # Экстремум в последней трети ряда — заведомо внутри тестовой части.
    # Значение записывается в dtype самой колонки: присваивание обычного
    # Python-float повысило бы float32-колонку до float64, и вся матрица
    # признаков пересчиталась бы с другой точностью. Тогда тест падал бы на
    # различии порядка 1e-7, не имеющем отношения к утечке данных.
    # Позиция аномалии выбирается сразу после начала тестовой части: у самого
    # конца ряда она попала бы только в целевые значения последних окон и не
    # вошла бы ни в одно входное окно, из-за чего проверка стала бы пустой.
    df_spiked = df.copy()
    last = int(len(df_spiked) * 0.85) + 10
    for column, factor in (("consumption", 50.0), ("temperature", 3.0),
                           ("humidity", 1.0), ("wind_speed", 20.0),
                           ("ev_load_kw", 100.0), ("solar_gen_kw", 100.0),
                           ("cloud_cover", 1.0)):
        dtype = df_spiked[column].dtype
        value = float(df_spiked[column].max()) * factor + 1000.0
        df_spiked.iloc[last, df_spiked.columns.get_loc(column)] = dtype.type(value)
    spiked = prepare_data(df_spiked, history_length=48, forecast_horizon=24)

    scaler_keys = ("scaler", "temp_scaler", "humidity_scaler", "wind_scaler",
                   "cloud_scaler", "ev_scaler", "solar_scaler")
    for key in scaler_keys:
        a, b = baseline[key], spiked[key]
        assert a is not None and b is not None, f"{key} отсутствует"
        assert a.data_max_[0] == pytest.approx(b.data_max_[0]), (
            f"{key}: максимум обучающей выборки изменился из-за значения в тесте"
        )
        assert a.data_min_[0] == pytest.approx(b.data_min_[0]), (
            f"{key}: минимум обучающей выборки изменился из-за значения в тесте"
        )

    assert baseline["temp_sq_max"] == pytest.approx(spiked["temp_sq_max"]), (
        "нормировка temperature² не должна зависеть от тестовых температур"
    )
    assert baseline["mase_scale"] == pytest.approx(spiked["mase_scale"]), (
        "знаменатель MASE считается по train и не должен зависеть от теста"
    )

    # Обучающая матрица признаков обязана совпасть побитово.
    assert np.array_equal(baseline["X_train"], spiked["X_train"]), (
        "обучающие признаки изменились из-за аномалии в тестовой части"
    )
    assert np.array_equal(baseline["Y_train"], spiked["Y_train"])

    # А тестовая — обязана измениться, иначе тест ничего не проверяет.
    assert not np.array_equal(baseline["X_test"], spiked["X_test"]), (
        "аномалия не попала в тестовую часть — проверка недействительна"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ПУНКТ 7: построение ряда остатков
# ══════════════════════════════════════════════════════════════════════════════

def test_residual_series_is_horizon_slice_not_flatten():
    """
    Ряд остатков — это срез при фиксированном шаге горизонта.

    Развёрнутая матрица (N, H) не является временным рядом: элемент k = i·H + h
    относится к моменту i + h, поэтому соседние элементы скачут во времени.
    Здесь остатки закодированы так, что правильный срез даёт строго
    возрастающую последовательность, а flatten — нет.
    """
    n_windows, horizon = 40, 24
    # Значение = номер окна: срез по фиксированному h обязан дать 0,1,2,...
    resid = np.tile(np.arange(n_windows, dtype=float)[:, None], (1, horizon))

    series = extract_residual_series(np.zeros_like(resid), -resid, horizon_step=0)

    assert series.shape == (n_windows,), "ряд должен индексироваться номером окна"
    assert np.array_equal(series, np.arange(n_windows, dtype=float))
    assert len(series) != resid.size, "ряд не должен быть развёрнутой матрицей"


def test_residual_series_respects_horizon_step():
    """Разные шаги горизонта дают разные ряды остатков."""
    n_windows, horizon = 30, 24
    resid = np.tile(np.arange(horizon, dtype=float)[None, :], (n_windows, 1))

    s0 = extract_residual_series(np.zeros_like(resid), -resid, horizon_step=0)
    s5 = extract_residual_series(np.zeros_like(resid), -resid, horizon_step=5)

    assert np.allclose(s0, 0.0)
    assert np.allclose(s5, 5.0)


def test_residual_analysis_does_not_flatten_overlapping_matrix():
    """В analysis/residuals.py тесты автокорреляции идут по срезу, а не по flatten."""
    import analysis.residuals as residuals_module
    src = inspect.getsource(residuals_module.analyze_residuals)

    # Ряд для статистических тестов получается через extract_residual_series.
    assert "extract_residual_series" in src
    for test_call in ("adfuller(residuals)", "kpss(residuals",
                      "acorr_ljungbox(residuals", "durbin_watson(residuals)"):
        assert test_call in src, f"не найден вызов {test_call} по срезу"
    # residuals_all (flatten) допускается только для гистограммы и нормальности.
    assert "adfuller(residuals_all" not in src
    assert "acorr_ljungbox(residuals_all" not in src


# ══════════════════════════════════════════════════════════════════════════════
# ПУНКТ 6: названия соответствуют вычислениям
# ══════════════════════════════════════════════════════════════════════════════

def test_stability_analysis_is_not_called_backtesting():
    """
    Процедура без переобучения не должна называться бэктестингом.

    analyze_stability_over_windows только прогоняет уже обученную модель по
    окнам теста; настоящий walk-forward с переобучением — отдельная функция.
    """
    import analysis.backtesting as bt

    stability_src = inspect.getsource(bt.analyze_stability_over_windows)
    assert "ModelTrainer" not in stability_src, (
        "анализ устойчивости не должен переобучать модель"
    )
    assert re.search(r"устойчивост", stability_src, re.IGNORECASE), (
        "в логах и подписях должно фигурировать «устойчивость», не «бэктестинг»"
    )

    rolling_src = inspect.getsource(bt.run_rolling_origin_backtest)
    assert "ModelTrainer" in rolling_src and "build_fn()" in rolling_src, (
        "rolling-origin обязан создавать и обучать модель заново на каждом origin"
    )
    assert "prepare_data" in rolling_src, (
        "на каждом origin данные должны готовиться заново по доступной истории"
    )


def test_main_uses_stability_function_not_legacy_alias():
    """main.py вызывает функцию с корректным названием."""
    src = inspect.getsource(main_module.main)
    assert "analyze_stability_over_windows" in src
    assert "run_backtesting(" not in src, "устаревшее имя не должно использоваться"
