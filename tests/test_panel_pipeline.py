# -*- coding: utf-8 -*-
"""
Тесты panel-конвейера: диспетчеризация режимов, защита от нехватки памяти,
состав моделей и отчётность о частичных отказах.
"""

import inspect
import json
import logging

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


def test_stride_is_what_makes_the_large_mode_feasible():
    """
    Именно прореживание переводит panel-optimal из неисполнимых в исполнимые.

    Без него 96 рядов за три года дают около 1.8 млн обучающих окон — десятки
    гигабайт при материализации. Тест фиксирует обе стороны: и то, что режим
    без прореживания действительно неисполним, и то, что с ним он проходит.
    """
    Config.set_panel_optimal_mode()
    n_series = Config.PANEL_CITIES * Config.PANEL_FEEDERS_PER_CITY
    args = (n_series, Config.PANEL_DAYS, Config.PANEL_HISTORY, Config.FORECAST_HORIZON)

    without = panel_pipeline.estimate_windows(*args, stride=1)
    with_stride = panel_pipeline.estimate_windows(
        *args, stride=Config.PANEL_WINDOW_STRIDE)

    assert Config.PANEL_WINDOW_STRIDE > 1, "крупный режим обязан прореживать окна"
    assert without > panel_pipeline._MAX_WINDOWS_IN_MEMORY
    assert with_stride < panel_pipeline._MAX_WINDOWS_IN_MEMORY


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

    # Прореживание отключается намеренно: с ним крупный режим помещается в
    # память, и проверять было бы нечего. Тест касается самого механизма
    # отказа, а не конкретной настройки режима.
    Config.set_panel_optimal_mode()
    monkeypatch.setattr(Config, "PANEL_WINDOW_STRIDE", 1)

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


# ══════════════════════════════════════════════════════════════════════════════
# ИСТОЧНИК ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def test_dataset_switch_is_available_in_cli():
    """Внешний набор выбирается тем же ключом запуска, что и синтетика."""
    assert main_module.parse_args([]).dataset == "synthetic"
    assert main_module.parse_args(["--dataset", "uci"]).dataset == "uci"
    with pytest.raises(SystemExit):
        main_module.parse_args(["--dataset", "kaggle"])


def test_both_sources_enter_the_pipeline_at_one_point():
    """
    Ветвление по источнику происходит ровно один раз — до конвейера.

    Если бы синтетика и реальные данные обрабатывались по-разному дальше по
    коду, их результаты стали бы несравнимыми, а сравнение и есть единственная
    причина подключать внешний набор.
    """
    src = inspect.getsource(panel_pipeline.run_panel_pipeline)
    assert src.count("load_panel_dataset(") == 1
    assert "generate_panel_data" not in src, (
        "конвейер не должен обращаться к генератору напрямую"
    )
    assert 'dataset == "uci"' not in src, "ветвление по источнику осталось в конвейере"


def test_uci_data_runs_through_the_whole_pipeline(tmp_path, monkeypatch, uci_file_factory):
    """
    Реальный формат проходит конвейер целиком, а не только адаптер.

    Проверка отдельно взятого адаптера не поймала бы расхождение контракта:
    отсутствие погодных колонок, другой набор статических признаков и иные
    имена рядов проявляются только при сквозном прогоне.
    """
    # Файл обязан доходить до конца 2014 года: окно отсчитывается назад от
    # конца наблюдений, и более короткий файл с ним просто не пересечётся.
    path = uci_file_factory(tmp_path / "LD2011_2014.txt",
                            start="2014-01-01 00:15:00", periods=96 * 365,
                            connect_date="2014-01-01")

    Config.set_panel_smoke_mode()
    monkeypatch.setattr(Config, "OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(Config, "PANEL_DAYS", 120)

    class _Args:
        mode, scenario, seed = "panel-smoke", "current", 0
        dataset, uci_path = "uci", str(path)

    code = panel_pipeline.run_panel_pipeline(_Args(), logging.getLogger("test"))
    assert code == 0, "конвейер не прошёл на реальном формате данных"

    runs = list((tmp_path / "results" / "runs").iterdir())
    assert len(runs) == 1
    meta = json.loads((runs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    run = meta.get("run", meta)
    assert run["dataset"] == "uci"
    assert run["dataset_report"]["рядов отобрано"] == run["panel"]["n_series"]
    assert run["failed_models"] == []


def test_missing_uci_file_fails_before_creating_a_run(tmp_path, monkeypatch):
    """
    Отсутствие внешнего набора — понятный отказ, а не стек вызовов.

    Проверка идёт до создания каталога прогона: иначе неудачный запуск оставлял
    бы пустой каталог, неотличимый от прерванного вручную.
    """
    Config.set_panel_smoke_mode()
    monkeypatch.setattr(Config, "OUTPUT_DIR", str(tmp_path / "results"))

    class _Args:
        mode, scenario, seed = "panel-smoke", "current", 0
        dataset, uci_path = "uci", str(tmp_path / "нет.txt")

    code = panel_pipeline.run_panel_pipeline(_Args(), logging.getLogger("test"))

    assert code == 1
    assert not (tmp_path / "results" / "runs").exists(), "создан каталог пустого прогона"


def test_selection_uses_scale_free_criterion():
    """
    Победитель выбирается по безразмерной метрике, а не по абсолютной ошибке.

    micro-MAE на панели определяется крупнейшим фидером: он один решал бы, какая
    модель лучше, независимо от качества на остальных рядах.
    """
    src = inspect.getsource(panel_pipeline.run_panel_pipeline)
    crit = src[src.index("def _selection_criterion"):src.index("best_name = min(")]
    assert '"MASE"' in crit, "критерий отбора обязан опираться на MASE"
    assert '"MAE_macro"' in crit, "запасной критерий тоже должен уравнивать ряды"
    assert '["MAE"]' not in crit, "micro-MAE не должен участвовать в отборе"


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
