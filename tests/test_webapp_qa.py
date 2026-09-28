# -*- coding: utf-8 -*-
"""
Дефекты, найденные функциональной проверкой веб-приложения 29.09.2026.

Каждый тест воспроизводит сценарий из отчёта и падает на коде до исправления.
Номера (Б1, М4…) — из отчёта; сводка — в CHANGELOG.md, раздел «Веб-приложение».
"""

import io
import json
import re
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from data.connector import ExportFormat, parse_stamps, read_meter_export
from webapp.app import Settings, create_app
from webapp.services import econ
from webapp.services.ingest import build_format, load_hourly, sniff_format
from webapp.services.jobs import ImmediateExecutor, JobRunner

DEMO_FORM = {"timestamp_col": "Дата и время", "wide": "1", "interval_minutes": "30",
             "sep": ";", "decimal": ",", "encoding": "cp1251", "unit": "kW",
             "stamp_at": "end", "exclude_columns": "Итого"}
RUN_FORM = {"meters": ["ТП-3 Ф-1", "ТП-12 Ф-2"], "category": "4",
            "peak_hours": ",".join(str(h) for h in range(8, 21)),
            "capex_per_kwh": "16000", "power_share": "20"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(workspace_dir=str(tmp_path_factory.mktemp("ws")),
                        runs_dir=str(tmp_path_factory.mktemp("runs")),
                        executor=ImmediateExecutor(), n_boot=100, demo_days=70)
    app = create_app(settings)
    return TestClient(app), app


@pytest.fixture(scope="module")
def uploaded(client):
    c, _ = client
    r = c.post("/data/demo", follow_redirects=False)
    uid = re.search(r"u=([0-9a-f]{32})", r.headers["location"]).group(1)
    assert c.post(f"/data/read?u={uid}", data=DEMO_FORM, follow_redirects=False).status_code == 303
    return uid


def _upload(c, name, payload):
    r = c.post("/data/upload", files={"file": (name, payload, "text/csv")}, follow_redirects=False)
    return r, (re.search(r"u=([0-9a-f]{32})", r.headers.get("location", "")) or [None, None])[1]


def _run(c, uid, **extra):
    job = c.post(f"/economics/run?u={uid}", data={**RUN_FORM, **extra}).json()["job"]
    return c.get(f"/jobs/{job}").json()


# ── Б1: расчёт из кэша не должен оставлять на странице чужие параметры ──────

def test_cached_repeat_becomes_the_shown_calculation(client, uploaded):
    c, app = client
    ws = app.state.workspace
    _run(c, uploaded, power_share="20")
    a = ws.meta(uploaded)["last_economics"]
    _run(c, uploaded, power_share="40")
    _run(c, uploaded, power_share="20")               # из кэша: задача не запускается
    meta = ws.meta(uploaded)
    assert meta["last_economics"] == a
    assert meta["economics_params"]["power_share"] == pytest.approx(0.20)
    with open(ws.path(uploaded, f"econ_{a}.json"), encoding="utf-8") as f:
        shown = json.load(f)["result"]["best"]["annual_mean"]
    page = c.get(f"/economics?u={uploaded}").text
    assert re.search(r'id="readout" data-value="([^"]+)"', page).group(1) == str(shown)


# ── Б2, М9: даты ISO, «24:00», часовой пояс ──────────────────────────────────

def test_iso_dates_keep_day_and_month():
    s = pd.Series(["2025-03-01 00:00", "2025-03-13 01:00", "2025-05-02 00:00"])
    months = parse_stamps(s).dt.month.tolist()
    assert months == [3, 3, 5]


def test_offset_and_t_separator_are_local_wall_time():
    s = parse_stamps(pd.Series(["2025-01-13T00:00:00+03:00", "2025-01-13T01:00:00+03:00"]))
    assert s.iloc[0] == pd.Timestamp("2025-01-13 00:00")


def test_midnight_as_24_00_is_next_day():
    s = parse_stamps(pd.Series(["01.01.2025 23:00", "01.01.2025 24:00"]))
    assert s.iloc[1] == pd.Timestamp("2025-01-02 00:00")


def test_skipped_row_examples_come_from_the_time_column(tmp_path):
    path = tmp_path / "e.csv"
    body = "Прибор;Время;Значение\n" + "".join(
        f"А;{t:%d.%m.%Y %H:%M};1\n" for t in pd.date_range("2025-01-01 01:00", periods=48, freq="h"))
    path.write_text(body + "А;Итого;48\n", encoding="utf-8")
    fmt = build_format({"timestamp_col": "Время", "series_col": "Прибор", "value_col": "Значение",
                        "wide": False, "unit": "kWh", "stamp_at": "end", "sep": ";",
                        "decimal": ",", "encoding": "utf-8-sig"})
    assert load_hourly(str(path), fmt)["skipped_examples"] == ["Итого"]


# ── Б3: шаг меньше указанного не должен молча утраивать энергию ─────────────

def test_wrong_step_is_an_error_not_triple_energy():
    ts = pd.date_range("2025-01-01 00:05", periods=12 * 24, freq="5min")
    frame = pd.DataFrame({"series": "А", "timestamp": ts, "value": 1.0})
    with pytest.raises(ValueError, match="шаг"):
        read_meter_export(frame, ExportFormat(unit="kWh", interval_minutes=15))
    ok = read_meter_export(frame, ExportFormat(unit="kWh", interval_minutes=5))
    assert ok["consumption"].max() == pytest.approx(12.0)


# ── М7, М8: шапка отчёта, дата и время в разных столбцах, пустой столбец ───

def test_report_header_and_split_date_time(tmp_path):
    rows = "".join(f"{t:%d.%m.%Y};{t:%H:%M};{10 + t.hour};;\n"
                   for t in pd.date_range("2025-01-01 01:00", periods=24 * 20, freq="h"))
    path = tmp_path / "askue.csv"
    path.write_text("Отчёт АСКУЭ по точкам учёта\nПериод: январь 2025\n"
                    "Дата;Время;Ввод 1;Пустой;\n" + rows, encoding="utf-8")
    sniff = sniff_format(str(path))
    f = sniff["format"]
    assert f["skip_rows"] == 2 and f["timestamp_col"] == "Дата" and f["time_col"] == "Время"
    assert "Пустой" in f["exclude_columns"]
    fmt = build_format({**f, "unit": "kWh", "stamp_at": "end", "interval_minutes": 60,
                        "wide": True})
    hourly = load_hourly(str(path), fmt)["hourly"]
    assert set(hourly["series"]) == {"Ввод 1"}
    assert hourly["timestamp"].dt.hour.nunique() == 24, "час не должен падать на 00:00"


def test_thin_space_thousands_are_numbers(tmp_path):
    path = tmp_path / "t.csv"
    rows = "".join(f"{t:%d.%m.%Y %H:%M};1 234,5;2 000\n"
                   for t in pd.date_range("2025-01-01 01:00", periods=48, freq="h"))
    path.write_text("Время;А;Б\n" + rows, encoding="utf-8")
    f = sniff_format(str(path))["format"]
    assert f["wide"] is True
    fmt = build_format({**f, "unit": "kWh", "stamp_at": "end", "interval_minutes": 60})
    hourly = load_hourly(str(path), fmt)["hourly"]
    assert hourly.loc[hourly["series"] == "А", "consumption"].iloc[5] == pytest.approx(1234.5)


# ── М11: Excel читается, двоичный мусор отклоняется понятно ─────────────────

def test_xlsx_upload_is_converted(client, tmp_path):
    c, app = client
    ts = pd.date_range("2025-01-01 01:00", periods=24 * 3, freq="h")
    buf = io.BytesIO()
    pd.DataFrame({"Время": ts, "Ввод": np.arange(len(ts), dtype=float)}).to_excel(buf, index=False)
    r, uid = _upload(c, "выгрузка.xlsx", buf.getvalue())
    assert r.status_code == 303 and uid
    sniff = app.state.workspace.meta(uid)["sniff"]
    assert sniff["format"]["timestamp_col"] == "Время" and sniff["format"]["interval_minutes"] == 60


def test_binary_upload_is_rejected_with_reason(client):
    c, _ = client
    r = c.post("/data/upload", files={"file": ("x.csv", b"\x00\x01\x02garbage" * 50, "text/csv")})
    assert r.status_code == 422 and "двоичные" in r.text


# ── М1, М2, М4: ошибки формы — JSON для скрипта, диапазоны, пустые ставки ──

def test_fetch_errors_are_json(client, uploaded):
    c, _ = client
    r = c.post(f"/economics/run?u={uploaded}", data={**RUN_FORM, "energy_price": "abc"},
               headers={"X-Requested-With": "fetch"})
    assert r.status_code == 422 and "detail" in r.json()


@pytest.mark.parametrize("share", ["0", "-5", "300"])
def test_battery_power_out_of_range_is_rejected(client, uploaded, share):
    c, _ = client
    r = c.post(f"/economics/run?u={uploaded}", data={**RUN_FORM, "power_share": share},
               headers={"X-Requested-With": "fetch"})
    assert r.status_code == 422 and "Мощность накопителя" in r.json()["detail"]


def test_cleared_rate_is_an_error_not_the_example(client, uploaded):
    c, _ = client
    r = c.post(f"/economics/run?u={uploaded}", data={**RUN_FORM, "gen_capacity_rate": ""},
               headers={"X-Requested-With": "fetch"})
    assert r.status_code == 422 and "покупка мощности" in r.json()["detail"]
    r = c.post(f"/economics/run?u={uploaded}", data={**RUN_FORM, "peak_hours": ""},
               headers={"X-Requested-With": "fetch"})
    assert r.status_code == 422


# ── М5, М6: остановка ────────────────────────────────────────────────────────

def test_stopped_job_is_not_cached():
    runner = JobRunner(ImmediateExecutor())

    def work(job):
        job.stop.set()
        return "половина"

    first = runner.submit("k", "t", work)
    assert first.status == "stopped"
    second = runner.submit("k", "t", lambda job: "всё")
    assert second.id != first.id and second.result == "всё"


def test_economics_honours_stop(uploaded, client):
    _, app = client
    clean = app.state.workspace.load_frame(uploaded, "clean")
    frame = econ.economics_frame(econ.series_for(clean, ["ТП-3 Ф-1"])["series"])

    class Job:
        stop = threading.Event()
        stage, progress = "", 0.0
    Job.stop.set()
    with pytest.raises(econ.Stopped):
        econ.run_economics(frame, econ.tariff_from_form({"category": 4}), n_boot=10, job=Job())


# ── М10: общий период приборов, немногие сутки с пропуском ─────────────────

def _series(days=70, gaps=()):
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    s = pd.Series(100 + 30 * np.sin(np.arange(len(idx)) / 24 * 2 * np.pi), index=idx)
    for day in gaps:
        s.loc[idx[day * 24]:idx[day * 24 + 23]] = np.nan
    return s


def test_single_gap_day_is_dropped():
    frame = econ.economics_frame(_series(gaps=[50]))
    assert frame.attrs["dropped_days"] == ["20.02.2025"]
    # 70 суток, одни без данных: 69 с данными − 28 на историю − 1 исключённые.
    assert frame["timestamp"].dt.normalize().nunique() == 40


def test_many_gap_days_are_an_error_with_sorted_dates():
    with pytest.raises(econ.DataError) as err:
        econ.economics_frame(_series(gaps=[44, 46, 48, 50, 52, 54, 56, 58]))
    shown = re.findall(r"\d\d\.\d\d\.\d{4}", err.value.detail)
    assert shown == sorted(shown, key=lambda d: pd.to_datetime(d, dayfirst=True))
    assert shown[0] == "14.02.2025"


def test_meters_are_cut_to_their_common_period():
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h")
    clean = pd.concat([
        pd.DataFrame({"series": "А", "timestamp": idx, "consumption": 1.0}),
        pd.DataFrame({"series": "Б", "timestamp": idx[:24 * 50], "consumption": 1.0}),
    ])
    picked = econ.series_for(clean, ["А", "Б"])
    assert picked["series"].index[-1] == idx[24 * 50 - 1] and picked["dropped_hours"] == 0


# ── М12: примерные ставки — поимённо ────────────────────────────────────────

def test_example_rates_are_listed_field_by_field():
    t = econ.tariff_from_form({"category": 4, "gen_capacity_rate": "900",
                               "peak_hours": ",".join(map(str, range(9, 19)))})
    fields = econ.example_fields(t)
    assert "покупка мощности" not in fields and "часы пика" not in fields
    assert "электроэнергия" in fields and econ.is_example_tariff(t)


# ── М15: формат можно поменять после чтения ─────────────────────────────────

def test_reparse_shows_the_form_with_previous_answers(client, uploaded):
    c, _ = client
    page = c.get(f"/data?u={uploaded}&reparse=1").text
    assert "Что означают числа в таблице?" in page
    assert re.search(r'value="kW"[^>]*checked', page)


# ── Выбор прогноза не на том же отрезке, где экономия ───────────────────────

def test_selection_and_evaluation_do_not_overlap():
    frame = econ.economics_frame(_series(days=140))
    sel, rest = econ.selection_split(frame)
    assert sel["timestamp"].max() < rest["timestamp"].min()
    assert sel["timestamp"].dt.normalize().nunique() == 28


# ── Финансы и риск часа пика ────────────────────────────────────────────────

def test_npv_and_irr_match_the_annuity_formula():
    f = econ.cash_flows(100.0, 500.0, 0.10, 10, 0.0)
    annuity = (1 - 1.10 ** -10) / 0.10
    assert f["npv"] == pytest.approx(100 * annuity - 500)
    r = f["irr"]
    assert sum(100 / (1 + r) ** t for t in range(1, 11)) == pytest.approx(500, rel=1e-6)
    assert econ.cash_flows(10.0, 500.0, 0.1, 10, 0.0)["irr"] is None


def test_finance_is_recomputed_without_a_new_calculation(client, uploaded):
    c, _ = client
    _run(c, uploaded)
    r = c.post(f"/economics/finance?u={uploaded}", data={"discount": "5", "life": "15", "growth": "0"})
    assert r.status_code == 200 and "NPV" in r.text and 'value="15"' in r.text
    bad = c.post(f"/economics/finance?u={uploaded}", data={"discount": "150"})
    assert bad.status_code == 422


def test_peak_hour_risk_orders_the_scenarios(client, uploaded):
    c, app = client
    _run(c, uploaded)
    ws = app.state.workspace
    with open(ws.path(uploaded, f"econ_{ws.meta(uploaded)['last_economics']}.json"), encoding="utf-8") as f:
        risk = json.load(f)["result"]["peak_risk"]
    values = [h["annual"] for h in risk["by_hour"]]
    assert risk["worst"] == pytest.approx(min(values))
    assert min(values) <= risk["mean"] <= max(values)


# ── Отчёт и выгрузка ────────────────────────────────────────────────────────

def test_report_and_excel_export(client, uploaded):
    c, _ = client
    _run(c, uploaded)
    assert "Экономия от накопителя энергии" in c.get(f"/economics/report?u={uploaded}").text
    r = c.get(f"/economics/export.xlsx?u={uploaded}")
    assert r.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert any("sheet" in n for n in names)


def test_what_if_price_points_follow_the_users_price():
    values = [p["value"] for p in econ.what_if_points(12_000.0) if p["kind"] == "capex"]
    assert values == [9_000.0, 18_000.0]


# ── Мелочи форматирования ───────────────────────────────────────────────────

def test_payback_words_agree_between_helpers():
    from webapp.formatting import rub, years_range
    assert years_range(30, 40) == "больше 25 лет"
    assert years_range(5, float("inf")).startswith("от 5 лет")
    assert rub(11.07e9) == "11,1 млрд ₽"


def test_forecast_for_a_run_without_series_explains_and_falls_back(tmp_path):
    runs = tmp_path / "runs"
    good = runs / "20260102_010101_panel-fast_current_seed0"
    good.mkdir(parents=True)
    (good / "run_metadata.json").write_text(json.dumps({"mode": "panel-fast", "seed": 0,
                                                        "status": "completed"}), encoding="utf-8")
    pd.DataFrame({"model": ["Naive24"], "split": ["test"], "MAE": [1.0]}).to_csv(good / "metrics.csv", index=False)
    ts = pd.date_range("2025-03-03", periods=48, freq="h")
    pd.DataFrame({"series": "C00/C00_F00", "timestamp": ts, "forecast": 1.0, "actual": 1.0}).to_csv(
        good / "forecast_series_panel_test.csv", index=False)
    bare = runs / "20260929_012214_panel-fast_current_seed0"
    bare.mkdir()
    (bare / "run_metadata.json").write_text(json.dumps({"mode": "panel-fast", "seed": 0}), encoding="utf-8")
    app = create_app(Settings(workspace_dir=str(tmp_path / "ws"), runs_dir=str(runs),
                              executor=ImmediateExecutor()))
    page = TestClient(app).get(f"/forecast?run={bare.name}").text
    assert "по рядам не сохранены" in page
    assert f'value="{good.name}" selected' in page
