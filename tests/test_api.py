# -*- coding: utf-8 -*-
"""REST API: те же функции, что у утилит, и понятные ошибки на неверный ввод."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from service.api import app


@pytest.fixture
def client():
    return TestClient(app)


def _series(days=21):
    ts = pd.date_range("2025-03-24", periods=days * 24, freq="h")
    h = ts.hour.to_numpy()
    actual = 300 + 120 * np.exp(-((h - 18) ** 2) / 6) + np.random.RandomState(0).normal(0, 8, len(h))
    return ts, actual


def test_health_without_models(client, monkeypatch):
    monkeypatch.delenv("SMARTGRID_MODELS_DIR", raising=False)
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_forecast_needs_a_models_dir(client, monkeypatch):
    monkeypatch.delenv("SMARTGRID_MODELS_DIR", raising=False)
    r = client.post("/forecast", json={"history": [], "weather": []})
    assert r.status_code == 503


def test_economics_endpoint(client):
    ts, actual = _series()
    body = {"timestamp": [t.isoformat() for t in ts], "actual": actual.tolist(),
            "forecast": (actual + 5).tolist(), "category": 4}
    r = client.post("/economics", json=body)
    assert r.status_code == 200
    sources = {row["source"] for row in r.json()["summary"]}
    assert {"Прогноз", "Идеальный прогноз", "Оптимум при известном будущем"} <= sources


def test_economics_rejects_bad_tariff(client):
    ts, actual = _series(2)
    body = {"timestamp": [t.isoformat() for t in ts], "actual": actual.tolist(),
            "forecast": actual.tolist(), "category": 4, "tariff": {"no_such_rate": 1}}
    assert client.post("/economics", json=body).status_code == 422


def test_quality_endpoint_reports_duplicates(client):
    ts, actual = _series(20)
    records = [{"series": "M", "timestamp": t.isoformat(), "consumption": float(v)}
               for t, v in zip(ts, actual)]
    records.append(dict(records[10]))
    r = client.post("/quality", json={"records": records})
    assert r.status_code == 200
    assert r.json()["report"][0]["duplicate_rows"] == 1


def test_forecast_endpoint_end_to_end(client, monkeypatch, tmp_path):
    from data.panel import generate_panel_data
    from data.panel_preprocessing import prepare_panel_data
    from models.panel_models import build_panel_ridge
    from models.panel_trainer import PanelTrainer
    from panel_pipeline import save_panel_models

    df, specs = generate_panel_data(days=60, n_cities=1, feeders_per_city=2, seed=6)
    data = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=6)
    trainer = PanelTrainer(build_panel_ridge(alphas=[1.0]), "Ridge").train(data)
    save_panel_models([trainer], data, str(tmp_path), best_name="Ridge")
    monkeypatch.setenv("SMARTGRID_MODELS_DIR", str(tmp_path))

    df = df.assign(series=df["city_id"].astype(str) + "/" + df["feeder_id"].astype(str))
    key = data["series_index"][0]
    series = df[df["series"] == key].sort_values("timestamp")
    history = series.iloc[-400:-24]
    target = series.iloc[-24:]
    weather = target[["series", "timestamp", "temperature", "humidity", "cloud_cover"]]

    def rec(frame):
        out = frame.copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return out.to_dict(orient="records")

    r = client.post("/forecast", json={"history": rec(history), "weather": rec(weather)})
    assert r.status_code == 200, r.text
    rows = r.json()["forecast"]
    assert len(rows) == 24 and rows[0]["series"] == key

    bad = client.post("/forecast", json={"history": rec(history.tail(50)), "weather": rec(weather)})
    assert bad.status_code == 422 and "истории" in bad.json()["detail"]
