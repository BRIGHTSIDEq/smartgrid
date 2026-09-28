# -*- coding: utf-8 -*-
"""
REST API для прогноза, экономики и проверки данных.

Тонкая обёртка над теми же функциями, что используют командные утилиты:
utils.panel_inference, analysis.economics, data.connector. Своей логики у
сервиса нет — поэтому и отдельных источников расхождений тоже.

    SMARTGRID_MODELS_DIR=results/runs/<прогон>/models \\
        uvicorn service.api:app --port 8000

Эндпоинты:
    GET  /health             — живость и какие модели загружены
    POST /forecast           — прогноз на следующие сутки по истории и погоде
    POST /economics          — экономика накопителя по ряду факта и прогноза
    POST /quality            — отчёт о качестве почасовых данных учёта

Ошибки входных данных возвращаются с кодом 422 и текстом причины из тех же
проверок, что у командных утилит.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Smart Grid: прогноз и экономика накопителя", version="1.0")


class ForecastRequest(BaseModel):
    history: List[Dict[str, Any]] = Field(..., description="series, timestamp, consumption, …")
    weather: List[Dict[str, Any]] = Field(..., description="прогноз погоды на целевые часы")
    model: Optional[str] = None
    quantiles: bool = False


class EconomicsRequest(BaseModel):
    timestamp: List[str]
    actual: List[float]
    forecast: List[float]
    category: int = 4
    tariff: Dict[str, Any] = Field(default_factory=dict)


class QualityRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="series, timestamp, consumption")


def _models_dir() -> str:
    path = os.environ.get("SMARTGRID_MODELS_DIR")
    if not path:
        raise HTTPException(503, "Не задан SMARTGRID_MODELS_DIR — каталог моделей прогона")
    return path


@lru_cache(maxsize=8)
def _bundle(models_dir: str, model: Optional[str], quantile: bool):
    from utils.panel_inference import load_panel_models
    return load_panel_models(models_dir, model, quantile=quantile)


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.replace({np.nan: None}).to_dict(orient="records")


@app.get("/health")
def health() -> Dict[str, Any]:
    path = os.environ.get("SMARTGRID_MODELS_DIR")
    info: Dict[str, Any] = {"status": "ok", "models_dir": path}
    if path and os.path.exists(os.path.join(path, "manifest.json")):
        import json
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        info["models"] = sorted(manifest["models"])
        info["quantile_models"] = sorted(manifest.get("quantile_models", {}))
        info["best_model_by_val"] = manifest["best_model_by_val"]
    return info


@app.post("/forecast")
def forecast(req: ForecastRequest) -> Dict[str, Any]:
    from utils.panel_inference import forecast_panel

    try:
        bundle = _bundle(_models_dir(), req.model, req.quantiles)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    history = pd.DataFrame(req.history)
    weather = pd.DataFrame(req.weather)
    for frame in (history, weather):
        if "timestamp" not in frame.columns:
            raise HTTPException(422, "Нет колонки timestamp")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    try:
        result = forecast_panel(bundle, history, weather)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"model": bundle["name"], "forecast": _records(result)}


@app.post("/economics")
def economics(req: EconomicsRequest) -> Dict[str, Any]:
    from analysis.economics import ACTUAL_COLUMN, battery_for_load, evaluate_sources
    from optimization.tariffs_ru import RuTariff

    if not (len(req.timestamp) == len(req.actual) == len(req.forecast)):
        raise HTTPException(422, "timestamp, actual и forecast разной длины")
    try:
        tariff = RuTariff(category=req.category, **req.tariff)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"Неверные параметры тарифа: {exc}")
    frame = pd.DataFrame({"timestamp": pd.to_datetime(req.timestamp),
                          "Прогноз": req.forecast, ACTUAL_COLUMN: req.actual})
    battery = battery_for_load(float(frame[ACTUAL_COLUMN].max()))
    result = evaluate_sources(frame, tariff, battery, n_boot=500)
    cols = ["source", "forecast_MAE", "net_savings", "annual_mean", "annual_lo",
            "annual_hi", "share_of_bound", "payback_years"]
    summary = result["summary"][cols].replace([np.inf, -np.inf], None)
    return {"battery_kwh": battery.capacity, "battery_kw": battery.max_power,
            "summary": _records(summary)}


@app.post("/quality")
def quality(req: QualityRequest) -> Dict[str, Any]:
    from data.connector import quality_report

    frame = pd.DataFrame(req.records)
    missing = {"series", "timestamp", "consumption"} - set(frame.columns)
    if missing:
        raise HTTPException(422, f"Нет колонок: {', '.join(sorted(missing))}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    # Повторы считаются до удаления: у оставшейся строки часа — число лишних копий.
    repeats = frame.groupby(["series", "timestamp"])["consumption"].transform("size") - 1
    frame = frame.assign(intervals=1, expected_intervals=1, duplicates=repeats)
    frame = frame.drop_duplicates(["series", "timestamp"])
    return {"report": _records(quality_report(frame))}
