# -*- coding: utf-8 -*-
"""
Веб-приложение: данные учёта → экономия накопителя → прогноз → модели.

Экраны и слова — docs/design/INTERFACE.md, вид — docs/design/VISUAL_SYSTEM.md.
Приложение только вызывает функции проекта (data.connector,
analysis.economics, optimization.*) и читает каталоги прогонов; своей
расчётной логики у него нет, поэтому цифры совпадают с командными утилитами.

    python -m webapp            # http://127.0.0.1:8050
"""

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp import formatting, texts
from webapp.services import econ
from webapp.services.errors import DataError, friendly
from webapp.services.ingest import build_format, load_hourly, sanity_summary, sniff_format
from webapp.services.jobs import JobRunner, params_key
from webapp.services.runs import Runs, display_model
from webapp.services.workspace import MAX_UPLOAD_BYTES, Workspace, default_root

logger = logging.getLogger("smart_grid.webapp")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


@dataclass
class Settings:
    workspace_dir: str = field(default_factory=default_root)
    runs_dir: str = os.path.join(ROOT, "results", "runs")
    executor: Any = None                 # None — фоновый поток; в тестах — синхронный
    n_boot: int = 2000
    demo_days: int = 365


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    ws = Workspace(settings.workspace_dir)
    ws.cleanup()
    runs = Runs(settings.runs_dir)
    jobs = JobRunner(settings.executor)

    app = FastAPI(title="Smart Grid", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
    templates = Jinja2Templates(directory=os.path.join(HERE, "templates"))
    formatting.register(templates.env)
    texts.register(templates.env)
    templates.env.globals["display_model"] = display_model
    # Версия статических файлов в адресе: после обновления браузер не возьмёт
    # из кэша старые стили и скрипты (на защите — чужой ноутбук, старый кэш).
    templates.env.globals["asset_v"] = _static_version(os.path.join(HERE, "static"))

    def page(request: Request, template: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, template, ctx)

    def fragment(name: str, **ctx) -> str:
        return templates.env.get_template(name).render(**ctx)

    def error_page(request: Request, err: DataError, status: int = 422, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, "error.html",
                                          {"error": err, **ctx}, status_code=status)

    def upload_meta(uid: Optional[str]) -> Optional[Dict[str, Any]]:
        if not uid:
            return None
        try:
            return ws.meta(uid)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Загрузка не найдена")

    # ── Данные ───────────────────────────────────────────────────────────────

    @app.get("/")
    def index():
        return RedirectResponse("/data", status_code=303)

    @app.get("/data", response_class=HTMLResponse)
    def data_page(request: Request, u: Optional[str] = None):
        meta = upload_meta(u)
        if meta is None:
            return page(request, "data.html", state="empty", nav="data", step=1)
        if not meta.get("read"):
            return page(request, "data.html", state="parse", nav="data", u=u, meta=meta, step=2,
                        sniff=meta["sniff"], units=_unit_check(ws, u, meta))
        quality = pd.read_csv(ws.path(u, "quality.csv"), encoding="utf-8")
        monthly = json.loads(meta.get("monthly_json", "[]"))
        return page(request, "data.html", state="report", nav="data", u=u, meta=meta, step=3,
                    quality=quality.to_dict(orient="records"), monthly=monthly,
                    summary=_quality_summary(quality))

    @app.post("/data/upload")
    def upload(request: Request, file: UploadFile = File(...)):
        try:
            uid = ws.create(file.file, file.filename or "выгрузка.csv")
        except OverflowError:
            return error_page(request, DataError(
                "Файл больше 50 МБ", "", "Выгрузите меньший период или меньше приборов."),
                status=413, nav="data")
        return _after_upload(request, uid)

    @app.post("/data/demo")
    def demo(request: Request):
        from webapp.demo.build_demo import write_demo_export
        path = os.path.join(ws.root, "demo", f"askue_demo_{settings.demo_days}.csv")
        if not os.path.exists(path):
            write_demo_export(path, days=settings.demo_days)
        with open(path, "rb") as f:
            uid = ws.create(f, "Демонстрационный объект.csv")
        ws.write_meta(uid, {"demo": True})
        return _after_upload(request, uid)

    def _after_upload(request: Request, uid: str):
        try:
            sniff = sniff_format(ws.original(uid))
        except Exception as exc:              # noqa: BLE001
            return error_page(request, friendly(exc), nav="data")
        ws.write_meta(uid, {"sniff": sniff})
        return RedirectResponse(f"/data?u={uid}", status_code=303)

    @app.post("/data/read")
    async def read_upload(request: Request, u: str):
        meta = upload_meta(u)
        form = dict(await request.form())
        form["wide"] = form.get("wide") == "1"
        form["exclude_columns"] = [c for c in (await request.form()).getlist("exclude_columns")]
        try:
            fmt = build_format(form)
            loaded = load_hourly(ws.original(u), fmt)
        except Exception as exc:              # noqa: BLE001
            return error_page(request, friendly(exc, form.get("encoding")), nav="data", u=u,
                              back=f"/data?u={u}")
        from data.connector import clean_hourly, quality_report
        hourly = loaded["hourly"]
        quality = quality_report(hourly)
        ws.save_frame(u, "hourly", hourly)
        ws.save_frame(u, "clean", clean_hourly(hourly))
        quality.to_csv(ws.path(u, "quality.csv"), index=False, encoding="utf-8")
        per_month = (hourly.groupby(hourly["timestamp"].dt.to_period("M"))["consumption"].sum())
        ws.write_meta(u, {"read": True, "format": asdict(loaded["format"]),
                          "skipped_rows": loaded["skipped_rows"],
                          "skipped_examples": loaded["skipped_examples"],
                          "sanity": sanity_summary(hourly),
                          "monthly_json": json.dumps([{"month": str(k), "kwh": float(v)}
                                                      for k, v in per_month.items()])})
        return RedirectResponse(f"/data?u={u}", status_code=303)

    @app.get("/data/meter.json")
    def meter_json(u: str, series: str):
        upload_meta(u)
        hourly = ws.load_frame(u, "hourly")
        part = hourly[hourly["series"] == series].sort_values("timestamp")
        if part.empty:
            raise HTTPException(404, "Прибор не найден")
        s = part.set_index("timestamp")
        full = pd.date_range(s.index.min(), s.index.max(), freq="h")
        s = s.reindex(full)
        bad = (s["consumption"].isna()
               | (s.get("negatives", 0).fillna(0) > 0)
               | ((s["intervals"] > 0) & (s["intervals"] < s["expected_intervals"])))
        # Застывшие показания — те же правила, что в отчёте о качестве: одно
        # ненулевое значение 6 часов подряд и дольше. Без этого отчёт называл
        # проблему, а график её не показывал.
        v = s["consumption"]
        same = v.eq(v.shift()) & v.ne(0) & v.notna()
        run_id = (~same).cumsum()
        run_len = same.groupby(run_id).transform("sum") + 1
        bad |= (run_len >= 6) & (same | same.shift(-1, fill_value=False))
        problems = s.index[bad]
        return JSONResponse({
            "t": (full.astype("int64") // 10**9).tolist(),
            "y": [None if pd.isna(x) else round(float(x), 2) for x in v],
            "problems": (problems.astype("int64") // 10**9).tolist(),
        })

    # ── Экономия ─────────────────────────────────────────────────────────────

    @app.get("/economics", response_class=HTMLResponse)
    def economics_page(request: Request, u: Optional[str] = None):
        meta = upload_meta(u)
        if meta is None or not meta.get("read"):
            return page(request, "economics.html", nav="economics", u=u, meta=meta, ready=False)
        quality = pd.read_csv(ws.path(u, "quality.csv"), encoding="utf-8")
        cached_html = None
        key = meta.get("last_economics")
        if key and os.path.exists(ws.path(u, f"econ_{key}.json")):
            with open(ws.path(u, f"econ_{key}.json"), encoding="utf-8") as f:
                cached = json.load(f)
            cached_html = fragment("_econ_result.html", r=cached["result"],
                                   passport=cached["passport"], u=u, cached=True)
        return page(request, "economics.html", nav="economics", u=u, meta=meta, ready=True, step=4,
                    meters=quality.to_dict(orient="records"),
                    params=meta.get("economics_params") or _default_params(quality),
                    cached_html=cached_html)

    @app.post("/economics/run")
    async def economics_run(request: Request, u: str):
        meta = upload_meta(u)
        form = await request.form()
        params = _econ_params(form)
        key = params_key({**params, "file": meta["original"], "created": meta["created"],
                          "code": econ.code_fingerprint(), "n_boot": settings.n_boot})

        def work(job):
            clean = ws.load_frame(u, "clean")
            picked = econ.series_for(clean, params["meters"])
            frame = econ.economics_frame(picked["series"])
            tariff = econ.tariff_from_form(params)
            result = econ.run_economics(frame, tariff, params["capex_per_kwh"],
                                        params["power_share"], n_boot=settings.n_boot, job=job)
            if params.get("rates_confirmed"):
                result["tariff"]["example"] = False
            # Выпавшие часы считаются только внутри периода оценки: часы до его
            # начала (например, прибор ещё не был подключён) на расчёт не влияют.
            in_period = picked["series"].loc[frame["timestamp"].iloc[0]:frame["timestamp"].iloc[-1]]
            passport = econ.passport(result, picked["meters"], int(in_period.isna().sum()))
            with open(ws.path(u, f"econ_{key}.json"), "w", encoding="utf-8") as f:
                json.dump({"result": result, "passport": passport}, f, ensure_ascii=False,
                          default=_json_default)
            ws.write_meta(u, {"last_economics": key, "economics_params": params})
            return fragment("_econ_result.html", r=result, passport=passport, u=u, cached=False)

        job = jobs.submit(key, "экономия", work)
        return JSONResponse({"job": job.id})

    @app.post("/economics/whatif")
    async def economics_whatif(request: Request, u: str):
        meta = upload_meta(u)
        params = meta.get("economics_params")
        key0 = meta.get("last_economics")
        if not params or not key0:
            raise HTTPException(409, "Сначала рассчитайте экономию")
        with open(ws.path(u, f"econ_{key0}.json"), encoding="utf-8") as f:
            base = json.load(f)["result"]
        key = params_key({"whatif": key0, "code": econ.code_fingerprint()})

        def work(job):
            clean = ws.load_frame(u, "clean")
            frame = econ.economics_frame(econ.series_for(clean, params["meters"])["series"])
            points = econ.run_what_if(frame, econ.tariff_from_form(params), base["best_source"],
                                      params["capex_per_kwh"], params["power_share"],
                                      n_boot=max(200, settings.n_boot // 4), job=job,
                                      stop=job.stop)
            return fragment("_whatif.html", points=points, base=base,
                            complete=len(points) == len(econ.what_if_points()))

        job = jobs.submit(key, "что если", work)
        return JSONResponse({"job": job.id})

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Задача не найдена")
        body: Dict[str, Any] = {"status": job.status, "progress": round(job.progress, 3),
                                "stage": job.stage, "elapsed": round(job.elapsed, 1)}
        if job.status == "done":
            body["html"] = job.result
        elif job.status == "error":
            err = friendly(job.error)
            if err.title == "Внутренняя ошибка":
                logger.error("Задача %s упала", job_id, exc_info=job.error)
            body["html"] = fragment("_error.html", error=err)
        return JSONResponse(body)

    @app.post("/jobs/{job_id}/stop")
    def job_stop(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Задача не найдена")
        job.stop.set()
        return JSONResponse({"stopping": True})

    # ── Прогноз и модели ─────────────────────────────────────────────────────

    @app.get("/forecast", response_class=HTMLResponse)
    def forecast_page(request: Request, run: Optional[str] = None, series: Optional[str] = None):
        candidates = [r for r in runs.scan(include_smoke=True)
                      if r["panel"] and r["has_panel_series"]]
        if not candidates:
            return page(request, "forecast.html", nav="forecast", runs=[], chosen=None)
        chosen = run or candidates[0]["name"]
        try:
            names = runs.panel_series_names(chosen)
        except KeyError:
            raise HTTPException(404, "Прогон не найден")
        series = series if series in names else (names[0] if names else None)
        check = runs.panel_forecast(chosen, series) if series else None
        meta = runs.meta(chosen) or {}
        manifest = {}
        mpath = runs.path(chosen, "models", "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                manifest = json.load(f)
        return page(request, "forecast.html", nav="forecast", runs=candidates, chosen=chosen,
                    series_names=names, series=series, check=check, meta=meta,
                    quantile=bool(manifest.get("quantile_models")))

    @app.get("/models", response_class=HTMLResponse)
    def models_page(request: Request, smoke: int = 0):
        return page(request, "models.html", nav="models", runs=runs.scan(include_smoke=bool(smoke)),
                    smoke=smoke)

    @app.get("/models/{name}", response_class=HTMLResponse)
    def run_page(request: Request, name: str):
        try:
            meta = runs.meta(name)
            table = runs.models_table(name)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Прогон не найден")
        best = (meta or {}).get("best_model_by_val")
        return page(request, "run.html", nav="models", name=name, meta=meta or {}, models=table,
                    series=runs.series_table(name, best) if best else [],
                    clients=runs.clients_economics(name),
                    has_dashboard=os.path.exists(runs.path(name, "dashboard.html")))

    @app.get("/models/{name}/dashboard")
    def run_dashboard(name: str):
        try:
            path = runs.path(name, "dashboard.html")
        except KeyError:
            raise HTTPException(404, "Прогон не найден")
        if not os.path.exists(path):
            raise HTTPException(404, "Сводки нет")
        return FileResponse(path, media_type="text/html")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.endswith(".json") or request.url.path.startswith("/jobs"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return templates.TemplateResponse(
            request, "error.html",
            {"error": DataError(str(exc.detail)), "nav": None}, status_code=exc.status_code)

    app.state.workspace = ws
    app.state.jobs = jobs
    return app


# ── Вспомогательное ──────────────────────────────────────────────────────────

def _static_version(folder: str) -> str:
    import hashlib
    h = hashlib.sha1()
    for dirpath, _, files in sorted(os.walk(folder)):
        for name in sorted(files):
            with open(os.path.join(dirpath, name), "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:8]


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _unit_check(ws: Workspace, uid: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Проверка обоих вариантов единиц до чтения: сутки с наибольшим потреблением.

    Ошибка «кВт вместо кВт·ч» на получасовых данных даёт расхождение ровно
    вдвое, и лучше увидеть это до расчёта, чем в итоговой цифре.
    """
    fmt = dict(meta["sniff"]["format"])
    if not fmt.get("interval_minutes"):
        return None
    try:
        fmt.update(unit="kW", stamp_at="end")
        as_kw = sanity_summary(load_hourly(ws.original(uid), build_format(fmt))["hourly"])
    except Exception:                         # noqa: BLE001
        return None
    if as_kw is None:
        return None
    factor = 60 / fmt["interval_minutes"]
    return {"day": as_kw["day"], "kw_day": as_kw["day_kwh"], "kw_peak": as_kw["peak_kwh"],
            "kwh_day": as_kw["day_kwh"] * factor, "kwh_peak": as_kw["peak_kwh"] * factor,
            "peak_hour": as_kw["peak_hour"]}


def _quality_summary(quality: pd.DataFrame) -> Dict[str, Any]:
    rows = quality.to_dict(orient="records")
    ok = [r for r in rows if not texts.human_problems(r)]
    return {"total": len(rows), "ok": len(ok)}


def _default_params(quality: pd.DataFrame) -> Dict[str, Any]:
    from optimization.tariffs_ru import RuTariff
    t = RuTariff(category=4)
    good = [r["series"] for r in quality.to_dict(orient="records") if not texts.human_problems(r)]
    meters = good or quality["series"].tolist()
    return {"meters": meters, "category": 4, "capex_per_kwh": econ.DEFAULT_CAPEX,
            "power_share": 0.20, "peak_hours": ",".join(str(h) for h in range(8, 21)),
            **{f: getattr(t, f) for f in econ.TARIFF_FIELDS}}


def _econ_params(form) -> Dict[str, Any]:
    params: Dict[str, Any] = {"meters": form.getlist("meters"),
                              "category": int(form.get("category", 4)),
                              "peak_hours": form.get("peak_hours", ""),
                              # «Все ставки верны»: пользователь подтвердил, что
                              # значения по умолчанию — его тариф.
                              "rates_confirmed": form.get("rates_confirmed") == "1"}
    try:
        for f in econ.TARIFF_FIELDS:
            value = form.get(f, "")
            params[f] = econ._number(value) if value not in ("", None) else ""
    except ValueError:
        raise HTTPException(422, "Ставки тарифа должны быть числами")
    try:
        params["capex_per_kwh"] = econ._number(form.get("capex_per_kwh") or econ.DEFAULT_CAPEX)
        params["power_share"] = econ._number(form.get("power_share") or 20) / 100.0
    except ValueError:
        raise HTTPException(422, "Неверные параметры накопителя")
    return params
