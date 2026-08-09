# -*- coding: utf-8 -*-
"""
Тесты panel-конвейера: диспетчеризация режимов, защита от нехватки памяти,
состав моделей и отчётность о частичных отказах.
"""

import inspect

import pytest

import main as main_module
import panel_pipeline
from config import Config


# ══════════════════════════════════════════════════════════════════════════════
# ДИСПЕТЧЕРИЗАЦИЯ РЕЖИМОВ
# ══════════════════════════════════════════════════════════════════════════════

def test_panel_modes_are_available_in_cli():
    """Режимы panel-* доступны через ту же точку входа, что и агрегатные."""
    args = main_module.parse_args(["--mode", "panel-smoke"])
    assert args.mode == "panel-smoke"

    for mode in ("panel-fast", "panel-optimal"):
        assert main_module.parse_args(["--mode", mode]).mode == mode


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        main_module.parse_args(["--mode", "panel-huge"])


def test_main_dispatches_panel_modes_before_aggregate_path():
    """
    Panel-режимы уходят в отдельный конвейер до начала агрегатного пути.

    Смешивание двух конвейеров в одной функции означало бы ветвление на каждом
    шаге: структура данных, состав моделей и набор метрик у них разные.
    """
    src = inspect.getsource(main_module.main)

    idx_dispatch = src.index('args.mode.startswith("panel-")')
    idx_generate = src.index("generate_smartgrid_data(")
    assert idx_dispatch < idx_generate, (
        "диспетчеризация обязана происходить до генерации агрегатных данных"
    )
    assert "run_panel_pipeline" in src


def test_all_panel_modes_have_config_setters():
    for mode in ("panel-smoke", "panel-fast", "panel-optimal"):
        setter = "set_" + mode.replace("-", "_") + "_mode"
        assert hasattr(Config, setter), f"нет метода {setter}"


def test_panel_modes_scale_up_monotonically():
    """Каждый следующий режим крупнее предыдущего хотя бы по одному измерению."""
    sizes = []
    for mode in ("panel_smoke", "panel_fast", "panel_optimal"):
        getattr(Config, f"set_{mode}_mode")()
        sizes.append(Config.PANEL_CITIES * Config.PANEL_FEEDERS_PER_CITY
                     * Config.PANEL_DAYS)
    assert sizes[0] < sizes[1] < sizes[2]


# ══════════════════════════════════════════════════════════════════════════════
# ЗАЩИТА ОТ НЕХВАТКИ ПАМЯТИ
# ══════════════════════════════════════════════════════════════════════════════

def test_window_estimate_matches_manual_calculation():
    """Оценка числа окон считается до генерации данных."""
    n = panel_pipeline.estimate_windows(n_series=4, days=90, history=48,
                                        horizon=24, train_ratio=0.70)
    per_series = int(90 * 24 * 0.70) - 48 - 24
    assert n == 4 * per_series


def test_panel_optimal_exceeds_memory_budget():
    """
    Крупный режим действительно не помещается в память при материализации.

    Тест фиксирует основание для отказа: без потоковой подачи запуск
    panel-optimal исчерпал бы память посреди расчёта, потеряв часы работы.
    """
    Config.set_panel_optimal_mode()
    n_series = Config.PANEL_CITIES * Config.PANEL_FEEDERS_PER_CITY
    expected = panel_pipeline.estimate_windows(
        n_series, Config.PANEL_DAYS, Config.PANEL_HISTORY, Config.FORECAST_HORIZON)

    assert expected > panel_pipeline._MAX_WINDOWS_IN_MEMORY, (
        "порог отказа должен срабатывать для panel-optimal"
    )


def test_panel_smoke_fits_in_memory():
    Config.set_panel_smoke_mode()
    n_series = Config.PANEL_CITIES * Config.PANEL_FEEDERS_PER_CITY
    expected = panel_pipeline.estimate_windows(
        n_series, Config.PANEL_DAYS, Config.PANEL_HISTORY, Config.FORECAST_HORIZON)

    assert expected < panel_pipeline._MAX_WINDOWS_IN_MEMORY


def test_pipeline_refuses_oversized_run_before_generating(monkeypatch):
    """
    Отказ происходит ДО генерации данных.

    Проверять память после того, как данные уже созданы, поздно: генерация
    крупной панели сама по себе занимает значительный объём.
    """
    called = {"generated": False}

    def _fail_if_called(*args, **kwargs):
        called["generated"] = True
        raise AssertionError("генерация не должна запускаться при отказе")

    monkeypatch.setattr("data.panel.generate_panel_data", _fail_if_called)

    Config.set_panel_optimal_mode()

    class _Args:
        mode, scenario, seed = "panel-optimal", "current", 0

    import logging
    code = panel_pipeline.run_panel_pipeline(_Args(), logging.getLogger("test"))

    assert code == 1, "переполнение памяти должно давать ненулевой код"
    assert not called["generated"]


# ══════════════════════════════════════════════════════════════════════════════
# СОСТАВ МОДЕЛЕЙ И ОТЧЁТНОСТЬ
# ══════════════════════════════════════════════════════════════════════════════

def test_panel_model_set_includes_naive_baselines():
    """
    Наивные базлайны обязательны.

    Без точки отсчёта нельзя утверждать, что обучаемая модель вообще имеет
    прогностическую ценность.
    """
    src = inspect.getsource(panel_pipeline.build_panel_models)
    for name in ("PanelNaive24", "PanelHourlyProfile", "build_panel_ridge",
                 "build_panel_xgboost", "build_dlinear"):
        assert name in src, f"{name} отсутствует в составе моделей"


def test_selection_excludes_naive_from_best_model():
    """Наивный базлайн участвует в сравнении, но не объявляется лучшей моделью."""
    src = inspect.getsource(panel_pipeline.run_panel_pipeline)
    assert '"Naive24", "HourlyProfile"' in src
    # Отбор идёт по валидации, тест используется после.
    idx_val = src.index('compare_panel_models(trainers, data, "val")')
    idx_best = src.index("best_name = min(")
    idx_test = src.index('compare_panel_models(trainers, data, "test")')
    assert idx_val < idx_best < idx_test


def test_partial_failure_returns_nonzero_code():
    """Частичный отказ не выдаётся за успех."""
    src = inspect.getsource(panel_pipeline.run_panel_pipeline)
    assert "failed_models" in src and "partial_failure" in src
    tail = src[src.index("if failed:"):]
    assert "return 2" in tail


def test_run_metadata_records_panel_shape():
    """Метаданные фиксируют состав панели и размерности входов."""
    src = inspect.getsource(panel_pipeline.run_panel_pipeline)
    for key in ('"cities"', '"feeders_per_city"', '"n_series"',
                '"hist"', '"future"', '"static"', '"best_model_by_val"'):
        assert key in src, f"{key} отсутствует в метаданных прогона"
