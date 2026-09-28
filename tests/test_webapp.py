# -*- coding: utf-8 -*-
"""
Веб-приложение: сквозной сценарий, совпадение с расчётным кодом, безопасность.

Приложение не должно считать само — только вызывать функции проекта. Поэтому
главная проверка — цифры страницы совпадают с прямым вызовом
analysis.economics.evaluate_sources на том же кадре.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from webapp.app import Settings, create_app
from webapp.services import econ
from webapp.services.jobs import ImmediateExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_FORM = {"timestamp_col": "Дата и время", "wide": "1", "interval_minutes": "30",
             "sep": ";", "decimal": ",", "encoding": "cp1251", "unit": "kW",
             "stamp_at": "end", "exclude_columns": "Итого"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    settings = Settings(workspace_dir=str(tmp_path_factory.mktemp("ws")), runs_dir=str(runs),
                        executor=ImmediateExecutor(), n_boot=200, demo_days=70)
    app = create_app(settings)
    return TestClient(app), app, runs


@pytest.fixture(scope="module")
def uploaded(client):
    c, app, _ = client
    r = c.post("/data/demo", follow_redirects=False)
    assert r.status_code == 303
    uid = re.search(r"u=([0-9a-f]{32})", r.headers["location"]).group(1)
    r = c.post(f"/data/read?u={uid}", data=DEMO_FORM, follow_redirects=False)
    assert r.status_code == 303, r.text[:500]
    return uid


# ══════════════════════════════════════════════════════════════════════════════
# ДАННЫЕ
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_page_requires_the_two_decisions(client):
    c, _, _ = client
    r = c.post("/data/demo")
    assert r.status_code == 200
    assert "Что означают числа в таблице?" in r.text and "Windows-1251" in r.text
    uid = re.search(r"/data/read\?u=([0-9a-f]{32})", r.text).group(1)
    form = {k: v for k, v in DEMO_FORM.items() if k != "unit"}
    bad = c.post(f"/data/read?u={uid}", data=form)
    assert bad.status_code == 422
    assert "единицы значений" in bad.text


def test_quality_report_finds_the_planted_defects(client, uploaded):
    c, _, _ = client
    page = c.get(f"/data?u={uploaded}").text
    assert "Можно считать 3 прибора из 6" in page
    for problem in ("перепутан знак", "нет данных 30 ч подряд", "завис счётчик"):
        assert problem in page.lower()


def test_meter_series_marks_problem_hours(client, uploaded):
    c, _, _ = client
    d = c.get(f"/data/meter.json?u={uploaded}&series=ТП-7 Ф-1").json()
    assert len(d["t"]) == len(d["y"])
    assert len(d["problems"]) >= 30, "30 часов без связи должны быть отмечены"


def test_total_row_and_encoding_are_handled(client, tmp_path):
    """Строка «Итого» в конце и файл в cp1251 при выбранной UTF-8 не мешают чтению."""
    c, _, _ = client
    text = "Дата и время;Счётчик 1;Итого\n" + "".join(
        f"{d.strftime('%d.%m.%Y %H:%M')};10,5;10,5\n"
        for d in pd.date_range("2025-03-01 01:00", periods=24 * 5, freq="h")) + "Итого;1260;1260\n"
    path = tmp_path / "export.csv"
    path.write_bytes(text.encode("cp1251"))
    with open(path, "rb") as f:
        r = c.post("/data/upload", files={"file": ("export.csv", f, "text/csv")}, follow_redirects=False)
    uid = re.search(r"u=([0-9a-f]{32})", r.headers["location"]).group(1)
    form = {**DEMO_FORM, "timestamp_col": "Дата и время", "interval_minutes": "60",
            "encoding": "utf-8-sig", "unit": "kWh"}
    r = c.post(f"/data/read?u={uid}", data=form, follow_redirects=True)
    assert r.status_code == 200
    assert "Пропущено строк без даты: 1" in r.text
    assert "Счётчик 1" in r.text


def test_oversized_upload_is_rejected(client, monkeypatch):
    import webapp.services.workspace as ws_module
    c, app, _ = client
    ws = app.state.workspace
    monkeypatch.setattr(ws, "create", lambda stream, name, max_bytes=None:
                        ws_module.Workspace.create(ws, stream, name, max_bytes=10))
    r = c.post("/data/upload", files={"file": ("big.csv", b"x" * 100, "text/csv")})
    assert r.status_code == 413


# ══════════════════════════════════════════════════════════════════════════════
# ЭКОНОМИЯ
# ══════════════════════════════════════════════════════════════════════════════

def _run_economics(c, uid, **extra):
    form = {"meters": ["ТП-3 Ф-1", "ТП-12 Ф-2"], "category": "4",
            "peak_hours": ",".join(str(h) for h in range(8, 21)),
            "capex_per_kwh": "16000", "power_share": "20", **extra}
    job = c.post(f"/economics/run?u={uid}", data=form).json()["job"]
    return c.get(f"/jobs/{job}").json()


def test_economics_matches_the_calculation_code(client, uploaded):
    """Цифра страницы = прямой вызов evaluate_sources на том же кадре."""
    from analysis.economics import battery_for_load, evaluate_sources

    c, app, _ = client
    status = _run_economics(c, uploaded)
    assert status["status"] == "done", status
    assert "Накопитель сэкономит в год" in status["html"]

    ws = app.state.workspace
    meta = ws.meta(uploaded)
    with open(ws.path(uploaded, f"econ_{meta['last_economics']}.json"), encoding="utf-8") as f:
        shown = json.load(f)["result"]

    clean = ws.load_frame(uploaded, "clean")
    frame = econ.economics_frame(econ.series_for(clean, ["ТП-3 Ф-1", "ТП-12 Ф-2"])["series"])
    tariff = econ.tariff_from_form({"category": 4, "peak_hours": ",".join(map(str, range(8, 21)))})
    battery = battery_for_load(float(frame["__actual__"].max()), 16000.0, 1.0)
    # Экономия считается на отрезке оценки — после суток, где выбирался прогноз.
    evaluated = econ.selection_split(frame)[1]
    direct = evaluate_sources(evaluated, tariff, battery, n_boot=200)["summary"].set_index("source")
    assert shown["best"]["net_savings"] == pytest.approx(direct.loc[shown["best_source"], "net_savings"])


def test_passport_names_the_example_rates_until_confirmed(client, uploaded):
    c, _, _ = client
    assert "Ставки и часы пика — примерные" in _run_economics(c, uploaded)["html"]
    confirmed = _run_economics(c, uploaded, rates_confirmed="1")["html"]
    assert "Ставки и часы пика — примерные" not in confirmed


def test_what_if_centre_equals_the_headline(client, uploaded):
    c, app, _ = client
    _run_economics(c, uploaded, category="6")
    job = c.post(f"/economics/whatif?u={uploaded}").json()["job"]
    html = c.get(f"/jobs/{job}").json()["html"]
    assert html.count("<tr") == 8                     # шапка + 7 вариантов

    ws = app.state.workspace
    meta = ws.meta(uploaded)
    with open(ws.path(uploaded, f"econ_{meta['last_economics']}.json"), encoding="utf-8") as f:
        base = json.load(f)["result"]
    clean = ws.load_frame(uploaded, "clean")
    frame = econ.economics_frame(econ.series_for(clean, meta["economics_params"]["meters"])["series"])
    centre = econ.run_what_if(frame, econ.tariff_from_form(meta["economics_params"]),
                              base["best_source"], points=econ.what_if_points()[:1], n_boot=50)[0]
    assert centre["annual"] == pytest.approx(base["best"]["annual_mean"])


def test_few_days_with_gaps_are_dropped_and_named(client, uploaded):
    """30 часов без связи у ТП-7 Ф-1 — несколько суток: они исключаются, а не валят расчёт."""
    c, _, _ = client
    status = _run_economics(c, uploaded, meters=["ТП-7 Ф-1"])
    assert status["status"] == "done", status
    assert "Исключены сутки с пропусками данных" in status["html"]


def test_bad_rate_is_rejected(client, uploaded):
    c, _, _ = client
    r = c.post(f"/economics/run?u={uploaded}", data={"meters": ["ТП-3 Ф-1"], "energy_price": "abc"})
    assert r.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# ПРОГОНЫ И БЕЗОПАСНОСТЬ
# ══════════════════════════════════════════════════════════════════════════════

def _fake_run(runs_dir, name, status="completed", mode="panel-fast"):
    d = runs_dir / name
    d.mkdir()
    (d / "run_metadata.json").write_text(json.dumps({
        "mode": mode, "seed": 0, "status": status, "dataset": "uci",
        "best_model_by_val": "XGBoost",
        "failed_models": [{"model": "DLinear", "error": "NaN"}] if status == "partial" else []}),
        encoding="utf-8")
    pd.DataFrame({"model": ["Naive24", "XGBoost"], "split": ["test", "test"],
                  "MAE": [10.0, 8.0], "MASE": [1.0, 0.8]}).to_csv(d / "metrics.csv", index=False)
    return d


def test_models_page_lists_runs_and_compares_with_yesterday(client):
    c, _, runs = client
    _fake_run(runs, "20260101_010101_panel-fast_uci_current_seed0", status="partial")
    page = c.get("/models").text
    assert "частично" in page and "UCI" in page
    run = c.get("/models/20260101_010101_panel-fast_uci_current_seed0").text
    assert "как вчера" in run
    assert "точнее на 20 %" in run, "XGBoost на 20 % лучше «как вчера» по MAE"
    assert "DLinear — NaN" in run


@pytest.mark.parametrize("name", ["..%5C..%5Cwindows", "C:%5Cwindows", "..%2F..%2Fsecrets",
                                  "20260101_010101_x%2F..%2F..%2Fetc"])
def test_run_name_cannot_escape_the_runs_folder(client, name):
    c, _, _ = client
    assert c.get(f"/models/{name}").status_code == 404


def test_upload_id_is_validated(client):
    c, _, _ = client
    assert c.get("/data?u=../../windows").status_code == 404
    assert c.get("/data/meter.json?u=zzz&series=x").status_code == 404


def test_forecast_page_without_panel_runs_explains_what_to_do(client):
    c, _, _ = client
    assert "Прогнозы появляются после многорядного прогона" in c.get("/forecast").text


def test_forecast_check_on_history(client):
    c, _, runs = client
    d = _fake_run(runs, "20260102_010101_panel-fast_current_seed0")
    ts = pd.date_range("2025-03-03", periods=24 * 4, freq="h")
    actual = 100 + 20 * np.sin(2 * np.pi * ts.hour.to_numpy() / 24)
    pd.DataFrame({"series": "C00/C00_F00", "timestamp": ts, "forecast": actual + 1.0,
                  "actual": actual}).to_csv(d / "forecast_series_panel_test.csv", index=False)
    page = c.get("/forecast?run=20260102_010101_panel-fast_current_seed0").text
    assert "Как модель угадала последние" in page
    assert "Коридора возможных значений нет" in page


# ══════════════════════════════════════════════════════════════════════════════
# ОФЛАЙН И ОФОРМЛЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def test_no_external_urls_in_templates_and_static():
    """Приложение работает без интернета: ни одного внешнего адреса."""
    base = os.path.join(ROOT, "webapp")
    for folder in ("templates", "static"):
        for dirpath, _, files in os.walk(os.path.join(base, folder)):
            for name in files:
                if name.endswith((".woff2", ".woff", ".png", ".ico")):
                    continue                 # шрифты и картинки — двоичные, адресов в них нет
                text = open(os.path.join(dirpath, name), encoding="utf-8").read()
                found = re.findall(r"https?://(?!www\.w3\.org/2000/svg)[^\s\"')]+", text)
                assert not found, f"{name}: {found}"


def test_numbers_follow_the_russian_rules():
    from webapp.formatting import num, rub, rub_range, years, years_range

    assert num(1234567.891, 2) == "1 234 567,89"
    assert num(-3.1, 1) == "−3,1"
    assert rub(1_520_000) == "1,52 млн ₽"
    assert rub_range(-2e5, 4e5) == "от −200 до 400 тыс. ₽"
    assert years(float("inf")) == "не окупается"
    assert years(40) == "больше 25 лет"
    assert years_range(3.1, 3.3) == "3,1–3,3 года"
    assert years_range(9.2, 14.1) == "9,2–14 лет"


def test_old_run_without_status_is_not_shown_as_running(client):
    """Прогоны до появления поля status не выдаются за идущие."""
    c, _, runs = client
    d = _fake_run(runs, "20250101_010101_optimal_current_seed42", mode="optimal")
    meta = json.loads((d / "run_metadata.json").read_text(encoding="utf-8"))
    meta.pop("status")
    (d / "run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    page = c.get("/models").text
    row = page[page.index("01.01.2025"):][:600]
    assert "статус не записан" in row and "идёт или прерван" not in row
