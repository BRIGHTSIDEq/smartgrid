# -*- coding: utf-8 -*-
"""
Отказ при оценке модели и метрики, устойчивые к нулевым фактам.

Ошибка оценки прежде выкидывала модель из таблицы молча: прогон завершался с
кодом 0, а сравнение строилось по неполному составу. MAPE_macro на UCI
достигал 2·10^8: в отличие от micro-MAPE, он делил на нулевые показания.
"""

import numpy as np
import pytest

import models.panel_trainer as pt
from models.panel_trainer import PanelTrainer, compare_panel_models
from models.trainer import compare_trainers


class _Broken:
    def __init__(self, name):
        self.model_name = self.name = name

    def evaluate(self, *args, **kwargs):
        raise ValueError("прогноз содержит NaN")


def test_aggregate_evaluation_error_is_a_model_failure():
    failures = []
    results = compare_trainers([_Broken("LSTM")], {}, split="val", failures=failures)
    assert results == {}
    assert failures == [{"model": "LSTM",
                         "error": "оценка (val): ValueError: прогноз содержит NaN"}]


def test_panel_evaluation_error_is_a_model_failure():
    failures = []
    compare_panel_models([_Broken("DLinear")], {"series_index": {}}, "test", failures=failures)
    assert [f["model"] for f in failures] == ["DLinear"]
    assert "оценка (test)" in failures[0]["error"]


class _Constant:
    """Модель, прогнозирующая единицу в масштабе, который тест подменяет."""

    def predict(self, batch):
        return np.ones((4, 2), dtype=np.float32)


def test_mape_macro_ignores_zero_actuals(monkeypatch):
    """
    Ряд с нулевыми показаниями не превращает MAPE_macro в бессмыслицу.

    Первый ряд: факт 2, прогноз 1, MAPE 50%. Второй ряд целиком нулевой — его
    MAPE не определён и в среднее по рядам не входит.
    """
    truth = np.array([[2.0, 2.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    monkeypatch.setattr(pt, "inverse_scale_series",
                        lambda data, values, series: truth if values is data["Y_test"]
                        else np.ones_like(truth))
    monkeypatch.setattr(pt, "seasonal_naive_scale", lambda data: {"a": 1.0, "b": 1.0})
    monkeypatch.setattr(pt, "make_batch", lambda data, split: {})

    data = {"Y_test": np.zeros((4, 2)), "series_test": np.array([0, 0, 1, 1]),
            "series_index": {0: "a", 1: "b"}}
    trainer = PanelTrainer(_Constant(), "Const")
    monkeypatch.setattr(trainer, "_is_keras", lambda: False)

    m = trainer.evaluate(data, "test")
    assert m["MAPE_macro"] == pytest.approx(50.0)
    assert m["MAPE"] == pytest.approx(50.0)
