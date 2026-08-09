# -*- coding: utf-8 -*-
"""
Тесты диагностики обучения, расписания learning rate и изоляции артефактов.

Все три закрывают дефекты, обнаруженные на реальном optimal-прогоне:
диагностика описывала не ту модель, которая оценивается; трансформеры уходили
в плохой минимум за несколько эпох из-за отсутствия прогрева; в каталог
прогона попадали графики посторонних запусков.
"""

import json
import os

import numpy as np
import pytest
import tensorflow as tf

from models.trainer import diagnose_training_regime
from models.transformer import WarmupCosineSchedule


# ══════════════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА ПО ЛУЧШЕЙ ЭПОХЕ
# ══════════════════════════════════════════════════════════════════════════════

def _history_overfitting_after_epoch5():
    """
    История, в которой валидация улучшается до пятой эпохи, затем ухудшается,
    а ошибка на обучении продолжает падать. Именно так вёл себя PatchTST.
    """
    val = [0.043, 0.040, 0.039, 0.0385, 0.0380, 0.0390, 0.0400, 0.0405, 0.0410, 0.0412]
    train = [0.065, 0.052, 0.046, 0.0436, 0.0410, 0.0360, 0.0310, 0.0270, 0.0245, 0.0236]
    return {"mae": train, "val_mae": val,
            "loss": [v * 0.05 for v in train], "val_loss": [v * 0.05 for v in val]}


def test_best_epoch_is_detected_not_last():
    """Лучшая эпоха определяется по минимуму валидации, а не берётся последней."""
    diag = diagnose_training_regime(_history_overfitting_after_epoch5())

    assert diag["best_epoch"] == 5, "минимум валидации приходится на пятую эпоху"
    assert diag["final_epoch"] == 10
    assert diag["epochs_after_best"] == 5


def test_metrics_reported_at_best_epoch():
    """
    В отчёт попадают метрики лучшей эпохи — той модели, что восстановит
    EarlyStopping, а не финального переобученного состояния.
    """
    diag = diagnose_training_regime(_history_overfitting_after_epoch5())

    assert diag["train_metric_at_best"] == pytest.approx(0.0410)
    assert diag["val_metric_at_best"] == pytest.approx(0.0380)
    # Финальные значения тоже доступны, но отдельными полями.
    assert diag["final_train_metric"] == pytest.approx(0.0236)
    assert diag["final_val_metric"] == pytest.approx(0.0412)
    assert diag["final_train_metric"] < diag["train_metric_at_best"]


def test_overfitting_after_best_is_flagged():
    """
    Расхождение кривых после лучшей эпохи распознаётся как переобучение.

    Прежняя реализация сравнивала train и val последней эпохи: разрыв
    0.0412 - 0.0236 = 0.0176 не дотягивал до порога 0.02, и модель, которая
    переобучилась на пятой эпохе, помечалась как сбалансированная.
    """
    diag = diagnose_training_regime(_history_overfitting_after_epoch5())

    assert diag["overfit_after_best"] is True
    assert diag["status"] == "overfitting"


def test_healthy_training_is_not_flagged():
    """Монотонно улучшающаяся валидация не должна помечаться переобучением."""
    train = [0.09, 0.07, 0.06, 0.055, 0.052]
    val = [0.095, 0.075, 0.065, 0.058, 0.054]
    diag = diagnose_training_regime({"mae": train, "val_mae": val,
                                     "val_loss": [v * 0.05 for v in val]})

    assert diag["best_epoch"] == 5 == diag["final_epoch"]
    assert diag["epochs_after_best"] == 0
    assert diag["overfit_after_best"] is False
    assert diag["status"] in ("balanced", "underfitting")


def test_missing_history_returns_unknown():
    assert diagnose_training_regime({})["status"] == "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# РАСПИСАНИЕ LEARNING RATE
# ══════════════════════════════════════════════════════════════════════════════

def test_warmup_cosine_shape():
    """
    Прогрев поднимает шаг с нуля до пикового, затем косинус опускает к минимуму.

    Без прогрева первые шаги делаются при неинициализированной статистике Adam,
    и модель уходит в плохой минимум за несколько эпох.
    """
    peak, total = 3e-4, 1000
    sched = WarmupCosineSchedule(peak_lr=peak, total_steps=total,
                                 warmup_fraction=0.1, min_lr=1e-6)

    at_start = float(sched(0))
    at_warmup_end = float(sched(100))
    at_mid = float(sched(550))
    at_end = float(sched(total))

    assert at_start < peak * 0.02, "в начале шаг должен быть близок к нулю"
    assert at_warmup_end == pytest.approx(peak, rel=1e-5), "после прогрева — пик"
    assert peak * 0.3 < at_mid < peak * 0.7, "в середине примерно половина пика"
    assert at_end < peak * 0.02, "к концу шаг затухает"


def test_warmup_cosine_is_monotonic_in_each_phase():
    sched = WarmupCosineSchedule(peak_lr=1e-3, total_steps=500, warmup_fraction=0.2)
    warm = [float(sched(s)) for s in range(0, 100, 10)]
    decay = [float(sched(s)) for s in range(100, 500, 40)]

    assert all(b >= a for a, b in zip(warm, warm[1:])), "прогрев должен расти"
    assert all(b <= a for a, b in zip(decay, decay[1:])), "затухание должно падать"


def test_warmup_cosine_is_serializable():
    """Расписание обязано переживать сохранение модели."""
    sched = WarmupCosineSchedule(peak_lr=2e-4, total_steps=800, warmup_fraction=0.05)
    cfg = sched.get_config()
    restored = WarmupCosineSchedule.from_config(cfg)

    for step in (0, 40, 400, 800):
        assert float(sched(step)) == pytest.approx(float(restored(step)))


def test_transformer_uses_schedule_when_steps_given():
    """Фабрика подключает расписание только при заданном числе шагов."""
    from models.transformer import build_patchtst

    with_sched = build_patchtst(history_length=48, forecast_horizon=24, n_features=26,
                                d_model=16, num_heads=2, num_layers=1, dff=32,
                                patch_len=8, stride=4, lr_schedule_total_steps=500)
    without = build_patchtst(history_length=48, forecast_horizon=24, n_features=26,
                             d_model=16, num_heads=2, num_layers=1, dff=32,
                             patch_len=8, stride=4, lr_schedule_total_steps=0)

    # Keras 3 отдаёт через optimizer.learning_rate текущее ЗНАЧЕНИЕ, а не объект
    # расписания. Наличие расписания определяется по конфигурации оптимизатора —
    # именно так его распознаёт ModelTrainer, чтобы отключить ReduceLROnPlateau.
    def has_schedule(model):
        lr_cfg = model.optimizer.get_config().get("learning_rate")
        return isinstance(lr_cfg, dict) and "class_name" in lr_cfg

    assert has_schedule(with_sched), "расписание должно быть подключено"
    assert not has_schedule(without), "без числа шагов расписания быть не должно"
    assert with_sched.optimizer.get_config()["learning_rate"]["class_name"] ==         "WarmupCosineSchedule"

    # Функциональная проверка: шаг обучения действительно меняется со временем.
    sched = getattr(with_sched.optimizer, "_learning_rate")
    assert float(sched(0)) < float(sched(25)), "прогрев должен повышать шаг"


# ══════════════════════════════════════════════════════════════════════════════
# ЁМКОСТЬ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def test_optimal_models_are_within_capacity_budget():
    """
    Ёмкость сетей в режиме optimal соразмерна объёму выборки.

    При 1.68 млн параметров на 12 193 окна PatchTST достигал минимума
    валидации на пятой эпохе и дальше только запоминал обучающие данные.
    """
    import logging
    import main as main_module
    from config import Config
    from models.transformer import count_parameters

    Config.set_optimal_mode()
    Config.finalize("current")
    log = logging.getLogger("test")

    for key in ("lstm", "patchtst", "transformer"):
        model, name = main_module.build_models([key], log, n_train_windows=12193)[0]
        n = count_parameters(model)
        assert n < 400_000, f"{name}: {n} параметров — избыточно для 12 тыс. окон"
        tf.keras.backend.clear_session()


# ══════════════════════════════════════════════════════════════════════════════
# ИЗОЛЯЦИЯ АРТЕФАКТОВ ПРОГОНА
# ══════════════════════════════════════════════════════════════════════════════

def test_run_dir_receives_artifacts_directly():
    """
    Пайплайн перенаправляет каталоги графиков и моделей внутрь каталога прогона.

    Прежняя схема копировала общий results/plots целиком, и в отчёт попадали
    изображения моделей, которые в этом прогоне не обучались, — в том числе
    четырёхмесячной давности.
    """
    import inspect
    import main as main_module

    src = inspect.getsource(main_module.main)
    assert "Config.PLOTS_DIR = os.path.join(run_dir" in src
    assert "Config.MODELS_DIR = os.path.join(run_dir" in src
    assert "copy_plots_to_run" not in src, (
        "копирование общего каталога графиков должно быть удалено"
    )

    # Каталог прогона создаётся до первого артефакта: раньше него в исходнике
    # не должно быть ни EDA, ни обучения.
    idx_run_dir = src.index("run_dir = reporting.make_run_dir")
    for later in ("run_eda(", "ModelTrainer(", "plot_training_history("):
        assert idx_run_dir < src.index(later), (
            f"{later} выполняется до создания каталога прогона"
        )


def test_two_runs_do_not_share_artifacts(tmp_path):
    """
    Два прогона получают независимые каталоги, и файлы одного не попадают в другой.
    """
    from utils import reporting

    first = reporting.make_run_dir(str(tmp_path), "smoke", "current", 0)
    os.makedirs(os.path.join(first, "plots"), exist_ok=True)
    with open(os.path.join(first, "plots", "model_A.png"), "wb") as f:
        f.write(b"first")

    second = reporting.make_run_dir(str(tmp_path), "smoke", "current", 1)
    os.makedirs(os.path.join(second, "plots"), exist_ok=True)
    with open(os.path.join(second, "plots", "model_B.png"), "wb") as f:
        f.write(b"second")

    assert first != second
    assert os.listdir(os.path.join(first, "plots")) == ["model_A.png"]
    assert os.listdir(os.path.join(second, "plots")) == ["model_B.png"]


def test_run_metadata_records_schema_version(tmp_path):
    """Метаданные содержат версию формата — старый прогон не выдаст себя за новый."""
    from utils import reporting

    run_dir = reporting.make_run_dir(str(tmp_path), "smoke", "current", 7)
    reporting.write_run_metadata(run_dir, {"mode": "smoke", "seed": 7})

    with open(os.path.join(run_dir, "run_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["schema_version"] == reporting.SCHEMA_VERSION
    assert "git_commit" in meta["environment"]
    assert isinstance(meta["environment"]["git_dirty"], bool)
