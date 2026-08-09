# -*- coding: utf-8 -*-
"""
Тесты устойчивости запуска: кодировка вывода, частичные отказы, строгость
экспортируемого JSON и пригодность бандла лучшей модели к инференсу.

Каждый тест соответствует отказу, который уже наблюдался на практике и был бы
незаметен без проверки: программа завершалась «успешно» с упавшими моделями,
JSON содержал NaN и ломал строгие парсеры, а бандл собирался только для
нейросетей, хотя лучшей по валидации оказывался градиентный бустинг.
"""

import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

import main as main_module
from utils import reporting


# ══════════════════════════════════════════════════════════════════════════════
# КОДИРОВКА КОНСОЛЬНОГО ВЫВОДА
# ══════════════════════════════════════════════════════════════════════════════

def test_keras_progress_bar_disabled():
    """
    Обучение не должно использовать анимированный прогресс-бар Keras.

    Бар рисуется символами Unicode; на консоли с однобайтовой кодировкой
    (cp1251 в русской Windows) он вызывает UnicodeEncodeError и обрывает
    обучение нейросетей. verbose=2 печатает одну ASCII-строку на эпоху.
    """
    from models.trainer import ModelTrainer

    src = inspect.getsource(ModelTrainer._train_keras)
    assert "verbose=2" in src, "fit должен вызываться с verbose=2"
    assert "verbose=1," not in src.split("self.model.fit(")[1].split(")")[0], (
        "в вызове fit не должно остаться verbose=1"
    )


def test_console_streams_are_made_encoding_safe():
    """Config.setup_logging переводит вывод в UTF-8 с заменой символов."""
    from config import Config

    src = inspect.getsource(Config._make_console_encoding_safe)
    assert "utf-8" in src and "replace" in src
    assert "_make_console_encoding_safe" in inspect.getsource(Config.setup_logging)


def test_pipeline_survives_cp1251_console():
    """
    Логирование не падает, когда поток вывода не умеет кодировать псевдографику.

    Воспроизводится в отдельном процессе с cp1251: до исправления первая же
    строка с символом ─ роняла программу.
    """
    code = (
        "import io, sys\n"
        "sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='cp1251')\n"
        "sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='cp1251')\n"
        "from config import Config\n"
        "log = Config.setup_logging()\n"
        "log.info('%s', '─' * 20 + ' проверка → ✅ ' + '═' * 10)\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    assert proc.returncode == 0, f"падение на cp1251:\n{proc.stderr[-2000:]}"
    assert "UnicodeEncodeError" not in proc.stderr


# ══════════════════════════════════════════════════════════════════════════════
# ЧАСТИЧНЫЕ ОТКАЗЫ
# ══════════════════════════════════════════════════════════════════════════════

def test_partial_failure_is_not_reported_as_success():
    """
    Падение запрошенной модели обязано давать ненулевой код возврата.

    Раньше упавшая модель молча исключалась из сравнения, а программа
    возвращала 0: снаружи запуск выглядел полностью успешным, хотя все
    нейросети отвалились.
    """
    src = inspect.getsource(main_module.main)

    assert "failed_models" in src and "trained_models" in src
    assert "requested_models" in src

    # После цикла обучения обязателен возврат ненулевого кода при отказах.
    tail = src[src.index("elapsed_min ="):]
    assert "if failed_models:" in tail
    assert "return 2" in tail, "частичный отказ должен возвращать ненулевой код"

    # Сообщение об успешном завершении не должно печататься при отказе.
    success_idx = tail.rindex("Пайплайн завершён за")
    failure_idx = tail.index("if failed_models:")
    assert failure_idx < success_idx, (
        "ветка частичного отказа обязана прерывать выполнение до сообщения об успехе"
    )


def test_failure_lists_go_into_run_metadata():
    """Состав запрошенных, обученных и упавших моделей попадает в метаданные."""
    src = inspect.getsource(main_module.main)
    meta_block = src[src.index("run_meta = {"):src.index("reporting.write_run_metadata")]
    for key in ('"requested_models"', '"trained_models"',
                '"failed_models"', '"partial_failure"'):
        assert key in meta_block, f"{key} отсутствует в метаданных прогона"


def test_full_success_returns_zero():
    """При отсутствии отказов код возврата нулевой."""
    src = inspect.getsource(main_module.main)
    assert src.rstrip().endswith("return 0")


# ══════════════════════════════════════════════════════════════════════════════
# СТРОГОСТЬ JSON
# ══════════════════════════════════════════════════════════════════════════════

def test_json_safe_replaces_non_finite_with_null():
    """NaN и бесконечности заменяются на null."""
    payload = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"),
               "ok": 1.5, "nested": [float("nan"), 2.0]}
    safe = reporting.json_safe(payload)

    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["ninf"] is None
    assert safe["ok"] == 1.5
    assert safe["nested"] == [None, 2.0]


def test_json_safe_preserves_boolean_type():
    """
    Булевы значения остаются булевыми, а не превращаются в строки.

    Сериализация через default=str записывала numpy-флаги как "True",
    и потребитель получал непустую строку там, где ожидал флаг.
    """
    safe = reporting.json_safe({"py": True, "np": np.bool_(False), "int": np.int64(5)})

    assert safe["py"] is True
    assert safe["np"] is False
    assert isinstance(safe["np"], bool)
    assert safe["int"] == 5 and isinstance(safe["int"], int)
    assert not isinstance(safe["int"], bool)


def test_dump_strict_json_rejects_non_finite_on_read(tmp_path):
    """Записанный файл разбирается строгим парсером без NaN и Infinity."""
    path = str(tmp_path / "strict.json")
    reporting.dump_strict_json(
        {"a": float("nan"), "b": [float("inf"), 1], "flag": np.bool_(True)}, path
    )

    text = path and open(path, encoding="utf-8").read()
    assert "NaN" not in text and "Infinity" not in text

    def _fail(const):
        raise AssertionError(f"недопустимая константа {const}")

    with open(path, encoding="utf-8") as f:
        parsed = json.load(f, parse_constant=_fail)
    assert parsed["a"] is None
    assert parsed["b"] == [None, 1]
    assert parsed["flag"] is True


def test_exported_metrics_json_is_strict(tmp_path):
    """
    Реальный экспорт метрик не содержит NaN и Infinity.

    MASE и R² легко становятся нечисловыми на вырожденных выборках, поэтому
    проверяется именно штатный путь записи.
    """
    metrics = {
        "ModelA": {"MAE": 1.0, "R2": float("nan"), "MASE": float("inf")},
        "ModelB": {"MAE": 2.0, "R2": 0.5, "MASE": 0.9},
    }
    reporting.export_metrics(metrics, str(tmp_path), seed=0, split="test",
                             run_meta={"ok": np.bool_(True), "bad": float("nan")},
                             append=False)

    def _fail(const):
        raise AssertionError(f"в metrics.json найдена константа {const}")

    with open(tmp_path / "metrics.json", encoding="utf-8") as f:
        payload = json.load(f, parse_constant=_fail)

    assert payload["run"]["ok"] is True
    assert payload["run"]["bad"] is None


def test_run_metadata_json_is_strict(tmp_path):
    """Метаданные прогона тоже проходят строгий разбор."""
    reporting.write_run_metadata(str(tmp_path), {
        "mode": "smoke", "seed": 0,
        "mase_scale": float("nan"),
        "partial_failure": np.bool_(False),
        "failed_models": [],
    })

    def _fail(const):
        raise AssertionError(f"в run_metadata.json найдена константа {const}")

    with open(tmp_path / "run_metadata.json", encoding="utf-8") as f:
        payload = json.load(f, parse_constant=_fail)

    assert payload["mase_scale"] is None
    assert payload["partial_failure"] is False
    assert isinstance(payload["environment"]["git_dirty"], bool)


# ══════════════════════════════════════════════════════════════════════════════
# БАНДЛ НЕ-KERAS МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def tiny_data():
    from data.generator import generate_smartgrid_data
    from data.preprocessing import prepare_data

    df = generate_smartgrid_data(days=25, households=30, seed=17,
                                 industrial_loads=2, city_districts=2)
    return df, prepare_data(df, history_length=48, forecast_horizon=24)


@pytest.mark.parametrize("builder_name", ["build_xgboost", "build_linear_regression"])
def test_sklearn_bundle_roundtrip(builder_name, tiny_data, tmp_path):
    """
    Обучение → сохранение → загрузка → совпадающий прогноз для не-Keras моделей.

    Лучшей по валидации регулярно оказывается XGBoost или Ridge; без этого
    цикла заявленная лучшая модель осталась бы без работающего инференса.
    """
    from models import baseline
    from utils.deployment import (
        export_model_bundle, load_model_bundle, predict_from_bundle,
    )
    from data.preprocessing import (
        _add_lag_columns, _build_feature_matrix, inverse_scale,
    )

    df, data = tiny_data
    builder = getattr(baseline, builder_name)
    model = (builder(n_estimators=10, max_depth=3)
             if builder_name == "build_xgboost" else builder())
    model.fit(data["X_train"], data["Y_train"],
              X_val=data["X_val"], Y_val=data["Y_val"])

    before = model.predict(data["X_test"][:4])
    assert before.shape == (4, 24)

    bundle_dir = export_model_bundle(
        model, data,
        {"HISTORY_LENGTH": 48, "FORECAST_HORIZON": 24,
         "N_FEATURES": data["n_features"], "model_name": builder_name},
        export_dir=str(tmp_path), model_name=builder_name,
    )

    with open(f"{bundle_dir}/config.json", encoding="utf-8") as f:
        assert json.load(f)["model_kind"] == "sklearn"

    bundle = load_model_bundle(bundle_dir)
    after = bundle["model"].predict(data["X_test"][:4])

    assert after.shape == before.shape
    assert np.allclose(before, after, rtol=1e-5, atol=1e-5), (
        "прогноз после загрузки не совпал — состояние модели потеряно"
    )

    # Полный инференс-путь через штатный препроцессинг.
    recent = df.tail(200)
    forecast = predict_from_bundle(bundle, recent)
    assert forecast.shape == (24,)
    assert np.isfinite(forecast).all()

    features = _build_feature_matrix(
        _add_lag_columns(recent.copy()).iloc[-48:],
        cons_scaler=data["scaler"], temp_scaler=data["temp_scaler"],
        humidity_scaler=data["humidity_scaler"], wind_scaler=data["wind_scaler"],
        rolling_std_scaler=data["rolling_std_scaler"],
        cloud_scaler=data["cloud_scaler"],
        ev_scaler=data["ev_scaler"], solar_scaler=data["solar_scaler"],
        temp_sq_max=data["temp_sq_max"],
    )
    expected = inverse_scale(data["scaler"], model.predict(features[np.newaxis]))[0]
    assert np.allclose(forecast, expected, rtol=1e-4, atol=1e-2)


def test_bundle_export_is_not_limited_to_keras():
    """main.py собирает бандл для победителя любого типа."""
    src = inspect.getsource(main_module.main)
    assert "if isinstance(best_trainer.model, tf.keras.Model):" not in src, (
        "экспорт бандла не должен зависеть от типа модели"
    )
    assert '"deployable_model"' in src

def test_seed_aggregation_records_what_it_aggregated(tmp_path):
    """
    Сводная таблица по сидам называет режим и сиды.

    Без этого она неинтерпретируема: MAE агрегатного ряда города и MAE
    отдельного фидера различаются в разы, а по одним числам не понять, что
    именно усреднено.
    """
    import pandas as pd
    from utils import reporting

    rows = []
    for seed in (0, 1):
        for model in ("XGBoost", "Naive24"):
            rows.append({"model": model, "seed": seed, "split": "test", "mode": "panel-fast",
                         "MAE": 10.0 + seed, "RMSE": 12.0, "MAPE": 4.0, "R2": 0.99, "MASE": 0.6})
    pd.DataFrame(rows).to_csv(tmp_path / "metrics.csv", index=False, encoding="utf-8-sig")

    out = reporting.aggregate_seeds(str(tmp_path))
    assert out is not None

    agg = pd.read_csv(out, encoding="utf-8-sig")
    assert (agg["mode"] == "panel-fast").all()
    assert (agg["seeds"] == "0,1").all()
