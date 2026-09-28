# -*- coding: utf-8 -*-
"""
Ключи командной строки: неприменимое сочетание — отказ, а не молчаливое
игнорирование.

Прежде panel-smoke --models foo обучал стандартный набор, smoke --dataset uci
учился на синтетике, а naive168 при истории 48 ч молча выпадал из таблицы.
Запрошенное не выполнялось, а по результату этого не было видно.
"""

import pytest

import main as main_module
from config import Config


def _check(*argv):
    main_module.validate_cli(main_module.parse_args(list(argv)))


@pytest.mark.parametrize("argv, fragment", [
    (["--mode", "smoke", "--dataset", "uci"], "--dataset"),
    (["--mode", "optimal", "--probabilistic"], "--probabilistic"),
    (["--mode", "panel-smoke", "--models", "foo"], "Неизвестные модели"),
    (["--mode", "panel-smoke", "--models", "lstm"], "Неизвестные модели"),
    (["--mode", "panel-fast", "--skip-storage"], "--skip-storage"),
    (["--mode", "panel-fast", "--rolling-origin", "3"], "--rolling-origin"),
    (["--mode", "panel-fast", "--dataset", "uci", "--scenario", "forward"], "--scenario"),
    (["--mode", "panel-fast", "--uci-path", "x.txt"], "--uci-path"),
    (["--mode", "smoke", "--models", "dlinear"], "Неизвестные модели"),
])
def test_inapplicable_flags_are_rejected(argv, fragment):
    with pytest.raises(SystemExit) as exc:
        _check(*argv)
    assert fragment in str(exc.value)


@pytest.mark.parametrize("argv", [
    ["--mode", "smoke", "--models", "naive24,ridge,xgboost"],
    ["--mode", "optimal", "--rolling-origin", "4", "--skip-eda"],
    ["--mode", "panel-fast", "--models", "ridge,xgboost"],
    ["--mode", "panel-fast", "--dataset", "uci", "--probabilistic"],
    ["--mode", "panel-optimal", "--dataset", "uci", "--uci-path", "other.txt"],
])
def test_valid_combinations_pass(argv):
    _check(*argv)


def test_naive168_on_short_history_is_an_error_when_requested(monkeypatch):
    monkeypatch.setattr(Config, "HISTORY_LENGTH", 48)
    with pytest.raises(SystemExit, match="168"):
        main_module.check_models_fit_history(["naive24", "naive168"], explicit=True)


def test_naive168_on_short_history_is_dropped_from_all(monkeypatch):
    monkeypatch.setattr(Config, "HISTORY_LENGTH", 48)
    assert main_module.check_models_fit_history(
        ["naive24", "naive168", "ridge"], explicit=False) == ["naive24", "ridge"]


def test_naive168_is_kept_with_a_week_of_history(monkeypatch):
    monkeypatch.setattr(Config, "HISTORY_LENGTH", 168)
    assert main_module.check_models_fit_history(["naive168"], explicit=True) == ["naive168"]


def test_panel_models_follow_the_models_flag():
    """В panel-режиме --models отбирает модели, а не игнорируется."""
    from data.panel import generate_panel_data
    from data.panel_preprocessing import prepare_panel_data
    from panel_pipeline import build_panel_models

    df, specs = generate_panel_data(days=40, n_cities=1, feeders_per_city=2, seed=1)
    data = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=1)

    assert [n for _, n in build_panel_models(data, 0, ["ridge", "naive24"])] == \
        ["Naive24", "Ridge"]
    assert len(build_panel_models(data, 0)) == len(main_module.PANEL_MODEL_REGISTRY)
    assert sorted(n for _, n in build_panel_models(data, 0)) == \
        sorted(main_module.PANEL_MODEL_REGISTRY.values())


# ══════════════════════════════════════════════════════════════════════════════
# РЕЖИМЫ CONFIG НЕ НАСЛЕДУЮТ ДРУГ ДРУГА
# ══════════════════════════════════════════════════════════════════════════════

def test_mode_does_not_depend_on_the_previous_mode():
    """
    Режим задаёт одни и те же параметры независимо от того, что было до него.

    Режимы меняют только часть параметров: после smoke режим optimal получал
    размер батча 64 вместо 32, потому что сам его не задаёт.
    """
    Config.set_optimal_mode()
    Config.finalize("current")
    clean = Config.snapshot()

    Config.set_smoke_mode()
    Config.finalize("forward")
    Config.set_optimal_mode()
    Config.finalize("current")
    after_smoke = Config.snapshot()

    diff = {k for k in clean if clean[k] != after_smoke.get(k)}
    assert not diff, f"optimal унаследовал от smoke: {sorted(diff)}"


def test_scenario_does_not_stick_between_runs():
    Config.set_fast_mode()
    Config.finalize("forward")
    Config.set_fast_mode()
    Config.finalize()
    assert Config.GEN_SCENARIO == "current"


def test_reset_keeps_redirected_paths(tmp_path, monkeypatch):
    """Сброс режима не возвращает каталог результатов в дерево проекта."""
    monkeypatch.setattr(Config, "OUTPUT_DIR", str(tmp_path))
    Config.set_smoke_mode()
    assert Config.OUTPUT_DIR == str(tmp_path)


def test_full_mode_runs_walk_forward_by_default():
    """Полный режим включает walk-forward, если число точек не задано явно."""
    Config.set_full_mode()
    assert Config.ROLLING_ORIGINS == 4
    Config.set_optimal_mode()
    assert Config.ROLLING_ORIGINS == 0
    assert main_module.parse_args(["--mode", "full"]).rolling_origin is None


def test_run_metadata_records_the_whole_config(tmp_path):
    import json
    from utils import reporting

    Config.set_smoke_mode()
    reporting.write_run_metadata(str(tmp_path), {"mode": "smoke"})
    meta = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["config"]["HISTORY_LENGTH"] == Config.HISTORY_LENGTH
    assert meta["config"]["BATCH_SIZE"] == 64
    assert not any(k.endswith("_DIR") for k in meta["config"])


def test_uci_feature_set_matches_uci_columns(tmp_path):
    """
    Синтетика с набором признаков UCI даёт те же каналы, что реальные данные.

    Иначе сравнение «синтетика против UCI» смешано со сравнением «19 каналов
    против 12».
    """
    from conftest import write_uci_file
    from data.panel import generate_panel_data
    from data.panel_preprocessing import prepare_panel_data
    from data.uci import uci_to_panel
    from panel_pipeline import to_uci_feature_set

    path = tmp_path / "LD.txt"
    write_uci_file(path, periods=24 * 4 * 120)
    uci, uci_specs, _ = uci_to_panel(str(path), start="2013-01-01", end="2013-04-25",
                                     max_series=2, train_ratio=0.7)
    syn, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=1)
    syn = to_uci_feature_set(syn, 0.7)

    a = prepare_panel_data(syn, specs, history_length=48, forecast_horizon=24, seed=1)
    b = prepare_panel_data(uci, uci_specs, history_length=48, forecast_horizon=24, seed=1)
    assert a["feature_names_hist"] == b["feature_names_hist"]
    assert a["feature_names_future"] == b["feature_names_future"]
    assert a["static_names"] == b["static_names"]


def test_feature_set_flag_rules():
    with pytest.raises(SystemExit, match="--feature-set"):
        _check("--mode", "smoke", "--feature-set", "uci")
    with pytest.raises(SystemExit, match="--feature-set"):
        _check("--mode", "panel-fast", "--dataset", "uci", "--feature-set", "uci")
    _check("--mode", "panel-fast", "--feature-set", "uci")


def test_uci_feature_set_runs_are_a_separate_source():
    from panel_pipeline import dataset_label

    class A:
        dataset, feature_set = "synthetic", "uci"
    assert dataset_label(A()) == "synthetic-ucifeat"
    A.feature_set = "full"
    assert dataset_label(A()) == "synthetic"
