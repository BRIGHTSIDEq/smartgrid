# -*- coding: utf-8 -*-
"""
Выгрузки приборов учёта: форматы и отчёт о качестве.

Каждая ловушка формата проверяется на выгрузке, где правильный ответ известен:
метка конца интервала, мощность вместо энергии, получасовой шаг, широкий
формат, пропуски, повторы, застывшие показания.
"""

import numpy as np
import pandas as pd
import pytest

from data.connector import ExportFormat, clean_hourly, quality_report, read_meter_export


def _quarter_hours(days=3, value_kw=4.0):
    end = pd.date_range("2025-01-01 00:15", periods=days * 96, freq="15min")
    return pd.DataFrame({"timestamp": end, "series": "M1", "value": value_kw})


def test_interval_end_stamps_and_power_units():
    """Мощность 4 кВт в 15-минутных интервалах — это 4 кВт·ч за час, начиная с 00:00."""
    hourly = read_meter_export(_quarter_hours(), ExportFormat(unit="kW", stamp_at="end"))
    assert hourly["timestamp"].min() == pd.Timestamp("2025-01-01 00:00")
    assert hourly["consumption"].iloc[:-1].to_numpy() == pytest.approx(4.0)
    assert (hourly["intervals"].iloc[:-1] == 4).all()


def test_energy_units_are_summed_not_scaled():
    frame = _quarter_hours(value_kw=1.0)
    hourly = read_meter_export(frame, ExportFormat(unit="kWh"))
    assert hourly["consumption"].iloc[0] == pytest.approx(4.0)


def test_start_stamps_are_not_shifted():
    frame = _quarter_hours()
    frame["timestamp"] -= pd.Timedelta(minutes=15)
    hourly = read_meter_export(frame, ExportFormat(unit="kW", stamp_at="start"))
    assert hourly["timestamp"].min() == pd.Timestamp("2025-01-01 00:00")
    assert (hourly["intervals"] == 4).all()


def test_wide_half_hour_export():
    stamps = pd.date_range("2025-01-01 00:30", periods=48, freq="30min")
    wide = pd.DataFrame({"timestamp": stamps, "А-1": 2.0, "А-2": 3.0})
    hourly = read_meter_export(wide, ExportFormat(wide=True, unit="kWh"))
    assert set(hourly["series"]) == {"А-1", "А-2"}
    assert hourly.groupby("series")["consumption"].first().to_dict() == {"А-1": 4.0, "А-2": 6.0}


def test_quality_report_finds_the_planted_problems():
    ts = pd.date_range("2025-01-01", periods=24 * 20, freq="h")
    rng = np.random.RandomState(0)
    cons = np.asarray(100 + 20 * np.sin(2 * np.pi * ts.hour / 24)) + rng.normal(0, 2, len(ts))
    cons[:48] = 0.0                              # объект ещё не подключён
    cons[100:110] = 123.0                        # застывшие показания
    cons[200] = -5.0                             # ошибка знака
    frame = pd.DataFrame({"series": "M", "timestamp": ts, "consumption": cons,
                          "intervals": 1, "expected_intervals": 1, "duplicates": 0})
    frame = frame.drop(index=range(300, 340))    # 40 ч без связи
    frame.loc[5, "duplicates"] = 1

    r = quality_report(frame).iloc[0]
    assert r["leading_zero_hours"] == 48
    assert r["flat_runs_over_limit"] == 1
    assert r["negative_values"] == 1
    assert r["missing_hours"] == 40 and r["longest_gap_hours"] == 40
    assert r["duplicate_rows"] == 1
    assert r["verdict"].startswith("проверить")


def test_clean_series_is_accepted():
    ts = pd.date_range("2025-01-01", periods=24 * 20, freq="h")
    cons = 100 + 20 * np.sin(2 * np.pi * ts.hour / 24) + np.random.RandomState(1).normal(0, 2, len(ts))
    frame = pd.DataFrame({"series": "M", "timestamp": ts, "consumption": cons,
                          "intervals": 1, "expected_intervals": 1, "duplicates": 0})
    assert quality_report(frame).iloc[0]["verdict"] == "пригоден"


def test_cleaning_fills_short_gaps_only():
    ts = pd.date_range("2025-01-01", periods=48, freq="h")
    cons = np.linspace(10, 57, 48)
    cons[:3] = 0.0
    frame = pd.DataFrame({"series": "M", "timestamp": ts, "consumption": cons,
                          "intervals": 1, "expected_intervals": 1, "duplicates": 0})
    frame = frame.drop(index=[10, 11]).drop(index=range(20, 30))
    clean = clean_hourly(frame, max_fill_hours=3)
    s = clean.set_index("timestamp")["consumption"]
    assert s.index[0] == ts[3], "нули до подключения отрезаются"
    assert s.loc[ts[10]] == pytest.approx(cons[10]), "короткий пропуск интерполируется"
    assert s.loc[ts[20:30]].isna().all(), "длинный пропуск не выдумывается"
