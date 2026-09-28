# -*- coding: utf-8 -*-
"""
Что прогон оставляет после себя: состав каталога, статус и поведение при отказах.

Тесты этого модуля запускают настоящий конвейер в отдельном процессе с
каталогом результатов во временной папке. Проверки по исходному тексту здесь
не годятся: дефекты этапа 1 проявлялись только в том, какие файлы и куда
реально записываются.
"""

import json
import logging
import os
import subprocess
import sys

import pandas as pd
import pytest

from utils import reporting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Запуск конвейера с перенаправленным каталогом результатов. sabotage
# подменяет блок накопителя падающим и ломает обучение наивной модели, чтобы
# проверить оба вида частичного отказа.
_RUNNER = r'''
import sys
sys.path.insert(0, {root!r})
from config import Config
Config.OUTPUT_DIR = {out!r}
import main
if {sabotage!r}:
    def _broken(*args, **kwargs):
        raise RuntimeError("накопитель сломан намеренно")
    main._run_storage_block = _broken

    import models.baseline as baseline

    class _BrokenModel:
        def fit(self, *args, **kwargs):
            raise RuntimeError("модель сломана намеренно")

    baseline.build_persistence_24 = lambda: _BrokenModel()
from utils.reporting import run_with_status
sys.exit(run_with_status(main.main, ["--mode", "smoke", "--models", "naive24,ridge",
                                     "--skip-eda", "--skip-attention", "--seed", "0"]))
'''


def _run_pipeline(out_dir: str, sabotage: bool = False):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="3")
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER.format(root=ROOT, out=out_dir, sabotage=sabotage)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=900)
    runs = os.listdir(os.path.join(out_dir, "runs"))
    assert len(runs) == 1, runs
    return proc, os.path.join(out_dir, "runs", runs[0])


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("results"))
    proc, run_dir = _run_pipeline(out)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return out, run_dir


@pytest.fixture(scope="module")
def sabotaged_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("results"))
    proc, run_dir = _run_pipeline(out, sabotage=True)
    return proc, out, run_dir


def _meta(run_dir):
    with open(os.path.join(run_dir, "run_metadata.json"), encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# СОСТАВ КАТАЛОГА ПРОГОНА
# ══════════════════════════════════════════════════════════════════════════════

def test_run_specific_files_live_in_the_run_dir(smoke_run):
    """
    Перебор порога и прогнозные ряды лежат в каталоге прогона, а не в корне.

    В корне results/ они перезаписывались каждым запуском, и smoke-прогон
    молча подменял ряды полного прогона, по которым строились выводы.
    """
    out, run_dir = smoke_run
    for name in ("storage_threshold_sweep.csv", "forecast_series_test.csv",
                 "forecast_series_val.csv", "storage_forecast_value.csv"):
        assert os.path.exists(os.path.join(run_dir, name)), f"нет {name} в каталоге прогона"
        assert not os.path.exists(os.path.join(out, name)), f"{name} записан в корень"


def test_run_metrics_json_has_both_splits(smoke_run):
    """В metrics.json прогона есть и тест, и валидация."""
    _, run_dir = smoke_run
    with open(os.path.join(run_dir, "metrics.json"), encoding="utf-8") as f:
        payload = json.load(f)
    assert {r["split"] for r in payload["metrics"]} == {"test", "val"}


def test_completed_run_is_marked_completed(smoke_run):
    _, run_dir = smoke_run
    meta = _meta(run_dir)
    assert meta["status"] == "completed"
    assert meta["dataset"] == "synthetic"
    assert meta["failed_blocks"] == []


# ══════════════════════════════════════════════════════════════════════════════
# ОТКАЗ НЕОБЯЗАТЕЛЬНОГО БЛОКА
# ══════════════════════════════════════════════════════════════════════════════

def test_failed_optional_block_keeps_metrics(sabotaged_run):
    """
    Отказ экономики накопителя не уничтожает метрики моделей.

    Прежде метрики выгружались в самом конце прогона, и исключение в любом
    блоке после оценки оставляло каталог без единой цифры.
    """
    _, out, run_dir = sabotaged_run
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"), encoding="utf-8-sig")
    assert set(df["split"]) == {"test", "val"}
    assert "LinearRegression" in set(df["model"])
    assert "Naive24 (сутки)" not in set(df["model"]), "упавшая модель не попадает в таблицу"
    assert os.path.exists(os.path.join(run_dir, "metrics_by_horizon.csv"))
    assert os.path.exists(os.path.join(run_dir, "diebold_mariano.csv"))


def test_failed_optional_block_makes_the_run_partial(sabotaged_run):
    """Отказ блока не выдаётся за успех: код 2 и статус partial."""
    proc, _, run_dir = sabotaged_run
    assert proc.returncode == 2, proc.stderr[-3000:]
    meta = _meta(run_dir)
    assert meta["status"] == "partial"
    assert [b["block"] for b in meta["failed_blocks"]] == ["экономика накопителя"]
    assert "накопитель сломан намеренно" in meta["failed_blocks"][0]["error"]


def test_failed_model_is_listed_and_not_reported_as_success(sabotaged_run):
    """
    Упавшая модель перечислена в метаданных, прогон частичный, в логе нет
    сообщения об успешном завершении.

    Прежде упавшая модель молча исключалась из сравнения, а программа
    возвращала 0. Поведение раньше проверялось по тексту исходника main().
    """
    proc, _, run_dir = sabotaged_run
    meta = _meta(run_dir)
    assert meta["requested_models"] == ["Naive24 (сутки)", "LinearRegression"]
    assert meta["trained_models"] == ["LinearRegression"]
    assert [m["model"] for m in meta["failed_models"]] == ["Naive24 (сутки)"]
    assert meta["partial_failure"] is True
    log = proc.stdout + proc.stderr
    assert "ЧАСТИЧНО" in log
    assert "Пайплайн завершён за" not in log


def test_successful_run_returns_zero_and_writes_plots_into_the_run(smoke_run):
    """
    Успешный прогон завершается кодом 0, а графики пишутся прямо в каталог
    прогона, не в общий results/plots.

    Прежде общий каталог копировался в отчёт целиком, вместе с изображениями
    чужих прогонов.
    """
    out, run_dir = smoke_run
    plots = os.listdir(os.path.join(run_dir, "plots"))
    assert any(p.endswith(".png") for p in plots)
    shared = os.path.join(out, "plots")
    assert not os.path.isdir(shared) or not os.listdir(shared)


def test_battery_is_driven_by_the_model_forecast(smoke_run):
    """
    Накопитель управляется прогнозом модели, а не фактом.

    Прежде в оптимизатор подавался сам тестовый ряд, и качество прогноза на
    экономику не влияло. Раньше проверялось по тексту исходника.
    """
    _, run_dir = smoke_run
    series = pd.read_csv(os.path.join(run_dir, "forecast_series_test.csv"), encoding="utf-8-sig")
    assert (series["LinearRegression"] - series["__actual__"]).abs().mean() > 0
    value = pd.read_csv(os.path.join(run_dir, "storage_forecast_value.csv"), encoding="utf-8-sig")
    row = value[value["forecast_source"] == "LinearRegression"].iloc[0]
    assert row["forecast_MAE"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# СТАТУС ПРИ ИСКЛЮЧЕНИИ И КОДЕ ВОЗВРАТА
# ══════════════════════════════════════════════════════════════════════════════

def test_crash_marks_the_run_failed(tmp_path):
    """
    Исключение после создания каталога даёт статус failed с текстом ошибки.

    Упавший прогон навсегда оставался в статусе running и был неотличим от
    прогона, который ещё идёт.
    """
    def entry(argv):
        reporting.make_run_dir(str(tmp_path), "smoke", "current", 0)
        raise MemoryError("не хватило памяти")

    with pytest.raises(MemoryError):
        reporting.run_with_status(entry)

    run_dir = os.path.join(tmp_path, "runs", os.listdir(tmp_path / "runs")[0])
    meta = _meta(run_dir)
    assert meta["status"] == "failed"
    assert "MemoryError" in meta["error"]
    assert meta["mode"] == "smoke", "прежние метаданные сохраняются"


@pytest.mark.parametrize("code, status", [(0, "completed"), (1, "failed"), (2, "partial")])
def test_exit_code_sets_the_status(tmp_path, code, status):
    def entry(argv):
        run_dir = reporting.make_run_dir(str(tmp_path), "smoke", "current", 0)
        reporting.write_run_metadata(run_dir, {"mode": "smoke"})
        return code

    assert reporting.run_with_status(entry) == code
    run_dir = os.path.join(tmp_path, "runs", os.listdir(tmp_path / "runs")[0])
    assert _meta(run_dir)["status"] == status


def test_status_of_a_previous_run_is_not_touched(tmp_path):
    """Отказ до создания каталога не меняет статус прогона из прошлого вызова."""
    first = reporting.make_run_dir(str(tmp_path), "smoke", "current", 0)
    reporting.write_run_metadata(first, {"mode": "smoke"})

    assert reporting.run_with_status(lambda argv: 1) == 1
    assert _meta(first)["status"] == "completed"


def test_optional_block_failure_is_recorded():
    failures = []
    assert reporting.run_optional_block("ok", lambda: 5, failures) == 5
    assert reporting.run_optional_block("плохой", lambda: 1 / 0, failures) is None
    assert failures == [{"block": "плохой",
                         "error": "ZeroDivisionError: division by zero"}]


# ══════════════════════════════════════════════════════════════════════════════
# ВЕРОЯТНОСТНЫЙ БЛОК
# ══════════════════════════════════════════════════════════════════════════════

def test_probabilistic_model_failure_is_reported(tmp_path, monkeypatch):
    """
    Отказ вероятностной модели попадает в список отказов.

    Прежде он оставался только в логе, прогон завершался с кодом 0, и таблица
    квантилей без нейросети выглядела полной.
    """
    from config import Config
    from data.panel import generate_panel_data
    from data.panel_preprocessing import prepare_panel_data
    import models.quantile_models as qm
    from panel_pipeline import run_probabilistic_block

    df, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=3)
    data = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=3)
    monkeypatch.setattr(Config, "PANEL_EPOCHS", 1)
    monkeypatch.setattr(Config, "PANEL_XGB_ESTIMATORS", 5)

    def _broken_fit(self, data):
        raise RuntimeError("квантили не сошлись")

    monkeypatch.setattr(qm.PanelQuantileXGBoost, "fit", _broken_fit)

    class _Args:
        seed = 0

    failures = []
    rows = run_probabilistic_block(data, _Args(), str(tmp_path),
                                   logging.getLogger("test"), failures=failures)

    assert [f["model"] for f in failures] == ["QuantileXGBoost"]
    assert "квантили не сошлись" in failures[0]["error"]
    assert "QuantileXGBoost" not in [r["model"] for r in rows]


def test_run_has_economics_for_price_category_four(smoke_run):
    """Экономика по категории 4 считается по сохранённым рядам прогона."""
    _, run_dir = smoke_run
    out = os.path.join(run_dir, "economics_ru_cat4")
    for name in ("summary.csv", "daily.csv", "sensitivity.csv", "report.md"):
        assert os.path.exists(os.path.join(out, name)), name
    summary = pd.read_csv(os.path.join(out, "summary.csv"), encoding="utf-8-sig")
    assert "Оптимум при известном будущем" in set(summary["source"])


def test_run_has_a_self_contained_dashboard(smoke_run):
    """Сводка открывается без сервера: изображения встроены, внешних ссылок нет."""
    _, run_dir = smoke_run
    page = open(os.path.join(run_dir, "dashboard.html"), encoding="utf-8").read()
    assert "Метрики на тесте" in page and "completed" in page
    assert "data:image/png;base64," in page
    assert "http://" not in page and "https://" not in page


def test_dashboard_shows_failures(tmp_path):
    from utils.dashboard import build_dashboard

    reporting.write_run_metadata(str(tmp_path), {
        "mode": "smoke", "status": "partial",
        "failed_blocks": [{"block": "экономика накопителя", "error": "RuntimeError: x"}]})
    page = open(build_dashboard(str(tmp_path)), encoding="utf-8").read()
    assert "Отказы" in page and "экономика накопителя" in page and "partial" in page


def test_dashboard_marks_a_run_with_failures_as_partial(tmp_path):
    from utils.dashboard import build_dashboard

    reporting.write_run_metadata(str(tmp_path), {
        "mode": "smoke", "status": "completed",
        "failed_models": [{"model": "LSTM", "error": "NaN"}]})
    page = open(build_dashboard(str(tmp_path)), encoding="utf-8").read()
    assert 'class="status partial"' in page
