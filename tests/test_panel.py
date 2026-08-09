# -*- coding: utf-8 -*-
"""
Тесты многорядного (panel) конвейера: генерация, нарезка окон, отсутствие
утечек между рядами и корректность известных заранее признаков.

Проверяются гарантии, нарушение которых не видно ни в одной метрике: окно,
склеившее два фидера, или прогноз погоды, подменённый фактом, дадут отличные
цифры и полностью неверные выводы.
"""

import numpy as np
import pandas as pd
import pytest

from data.panel import generate_panel_data, validate_panel, city_totals, FeederSpec
from data.panel_preprocessing import (
    prepare_panel_data, inverse_scale_series, make_weather_forecast,
    _tariff_zone_code,
)


@pytest.fixture(scope="module")
def panel():
    df, specs = generate_panel_data(days=90, n_cities=2, feeders_per_city=3, seed=5)
    return df, specs


@pytest.fixture(scope="module")
def prepared(panel):
    df, specs = panel
    return prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=5)


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР
# ══════════════════════════════════════════════════════════════════════════════

def test_panel_produces_multiple_series(panel):
    """Panel-режим даёт несколько рядов, а не один агрегат."""
    df, specs = panel
    n_series = df.groupby(["city_id", "feeder_id"]).ngroups

    assert n_series == 6, "2 города × 3 фидера"
    assert len(specs) == 6
    assert df["city_id"].nunique() == 2


def test_feeders_are_not_copies_of_each_other(panel):
    """
    Ряды должны различаться по существу.

    Если фидеры окажутся почти одинаковыми, panel-режим не добавит информации
    по сравнению с одним агрегатным рядом, и весь смысл перехода теряется.
    """
    df, _ = panel
    means = df.groupby("feeder_id")["consumption"].mean()

    assert means.std() / means.mean() > 0.15, "размеры фидеров слишком однородны"

    pivot = df.pivot_table(index="timestamp", columns="feeder_id", values="consumption")
    corr = pivot.corr().values
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert np.nanmax(off) < 0.995, "нашлись практически идентичные ряды"


def test_city_total_equals_sum_of_feeders(panel):
    """Иерархия согласована: городской ряд — точная сумма фидеров."""
    df, _ = panel
    totals = city_totals(df)

    manual = (df.groupby(["city_id", "timestamp"])["consumption"].sum()
                .reset_index(name="manual"))
    merged = totals.merge(manual, on=["city_id", "timestamp"])

    assert len(merged) == len(totals)
    assert np.allclose(merged["city_consumption"], merged["manual"], rtol=1e-6)


def test_panel_generation_is_reproducible():
    """Один и тот же seed даёт побитово одинаковый датасет."""
    a, specs_a = generate_panel_data(days=30, n_cities=1, feeders_per_city=3, seed=11)
    b, specs_b = generate_panel_data(days=30, n_cities=1, feeders_per_city=3, seed=11)

    assert np.allclose(a["consumption"].values, b["consumption"].values)
    assert [s.feeder_id for s in specs_a] == [s.feeder_id for s in specs_b]
    assert [s.households for s in specs_a] == [s.households for s in specs_b]


def test_different_seed_changes_data():
    a, _ = generate_panel_data(days=30, n_cities=1, feeders_per_city=3, seed=1)
    b, _ = generate_panel_data(days=30, n_cities=1, feeders_per_city=3, seed=2)
    assert not np.allclose(a["consumption"].values, b["consumption"].values)


def test_panel_validation_passes(panel):
    df, specs = panel
    ok, rows = validate_panel(df, specs)
    failed = [r["показатель"] for r in rows if not r["ok"]]
    assert ok, f"отклонения: {failed}"


def test_static_features_present_and_constant_per_feeder(panel):
    """Статические признаки постоянны внутри ряда и различаются между рядами."""
    df, _ = panel
    static_cols = [c for c in df.columns if c.startswith("static_")]
    assert static_cols, "статические признаки отсутствуют"

    for _, grp in df.groupby("feeder_id"):
        for col in static_cols:
            assert grp[col].nunique() == 1, f"{col} меняется внутри ряда"

    per_feeder = df.groupby("feeder_id")[static_cols].first()
    assert per_feeder.drop_duplicates().shape[0] > 1, "все фидеры описаны одинаково"


# ══════════════════════════════════════════════════════════════════════════════
# ОКНА НЕ ПЕРЕСЕКАЮТ ГРАНИЦЫ РЯДОВ
# ══════════════════════════════════════════════════════════════════════════════

def test_windows_never_cross_series_boundary(panel):
    """
    Окно целиком принадлежит одному фидеру.

    Проверка построена на подмене данных: каждому ряду присваивается
    собственное постоянное значение потребления. Если окно склеит два ряда,
    внутри него окажется больше одного уникального значения.
    """
    df, specs = panel
    marked = df.copy()
    codes = {fid: float(i + 1) * 1000.0
             for i, fid in enumerate(sorted(marked["feeder_id"].unique()))}
    marked["consumption"] = marked["feeder_id"].map(codes).astype(np.float32)

    data = prepare_panel_data(marked, specs, history_length=48,
                              forecast_horizon=24, seed=0)

    # Нормировка своя у каждого ряда, поэтому постоянный ряд даёт постоянное
    # значение внутри окна независимо от масштаба.
    for split in ("train", "val", "test"):
        X = data[f"X_hist_{split}"][:, :, data["feature_names_hist"].index("consumption")]
        spread = X.max(axis=1) - X.min(axis=1)
        assert np.allclose(spread, 0.0, atol=1e-5), (
            f"{split}: окно содержит данные более чем одного фидера"
        )


def test_series_ids_are_consistent_within_window(prepared):
    """Каждому окну соответствует ровно один идентификатор ряда."""
    for split in ("train", "val", "test"):
        ids = prepared[f"series_{split}"]
        assert ids.ndim == 1
        assert len(ids) == len(prepared[f"Y_{split}"])
        assert ids.min() >= 0
        assert ids.max() < len(prepared["series_index"])


def test_split_is_chronological(prepared):
    """
    Обучающие окна расположены во времени раньше валидационных и тестовых.

    Границы общие для всех рядов: иначе тестовый период одного фидера попал бы
    в обучающий период другого, а через общую погоду это прямая утечка.
    """
    train_end = prepared["split_time_bounds"]["train_end"]
    val_end = prepared["split_time_bounds"]["val_end"]
    history = prepared["history_length"]
    horizon = prepared["forecast_horizon"]

    # Последнее обучающее окно должно целиком помещаться до границы train.
    assert prepared["anchor_train"].max() + history + horizon <= train_end
    assert prepared["anchor_val"].min() >= train_end
    assert prepared["anchor_val"].max() + history + horizon <= val_end
    assert prepared["anchor_test"].min() >= val_end


# ══════════════════════════════════════════════════════════════════════════════
# НОРМИРОВКА И УТЕЧКИ
# ══════════════════════════════════════════════════════════════════════════════

def test_extreme_in_test_does_not_affect_train(panel):
    """
    Аномалия, помещённая только в тестовый период, не меняет ни скалеры,
    ни обучающие окна.
    """
    df, specs = panel
    base = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=0)

    spiked = df.copy()
    last_ts = spiked["timestamp"].max()
    mask = spiked["timestamp"] == last_ts
    spiked.loc[mask, "consumption"] = spiked["consumption"].max() * 100.0
    after = prepare_panel_data(spiked, specs, history_length=48,
                               forecast_horizon=24, seed=0)

    for key in base["series_index"]:
        a, b = base["series_scalers"][key], after["series_scalers"][key]
        assert a.data_max_[0] == pytest.approx(b.data_max_[0]), (
            f"{key}: скалер изменился из-за значения в тесте"
        )
    assert np.array_equal(base["X_hist_train"], after["X_hist_train"])
    assert np.array_equal(base["Y_train"], after["Y_train"])


def test_per_series_scaling_uses_own_range(prepared):
    """Каждый ряд нормируется своим скалером, а не общим на всю панель."""
    scalers = prepared["series_scalers"]
    maxima = [s.data_max_[0] for s in scalers.values()]

    assert len(scalers) == len(prepared["series_index"])
    assert max(maxima) / max(min(maxima), 1e-9) > 1.5, (
        "диапазоны рядов совпадают — вероятно, применена общая нормировка"
    )


def test_inverse_scale_is_per_feeder(prepared):
    """Обратная нормировка возвращает каждый ряд в его собственный масштаб."""
    ids = prepared["series_test"]
    y = prepared["Y_test"]
    restored = inverse_scale_series(prepared, y, ids)

    assert restored.shape == y.shape
    assert np.isfinite(restored).all()

    # Разные ряды должны восстанавливаться в разные диапазоны.
    uniq = np.unique(ids)
    if len(uniq) > 1:
        means = [restored[ids == s].mean() for s in uniq[:3]]
        assert max(means) / max(min(means), 1e-9) > 1.05


# ══════════════════════════════════════════════════════════════════════════════
# ПРИЗНАКИ БУДУЩЕГО
# ══════════════════════════════════════════════════════════════════════════════

def test_future_features_contain_no_target(prepared):
    """
    Среди признаков будущего нет потребления.

    Попадание целевой переменной в известные заранее признаки — самая грубая
    из возможных утечек, и обнаружить её по метрикам невозможно: они просто
    станут неправдоподобно хорошими.
    """
    names = prepared["feature_names_future"]
    for forbidden in ("consumption", "load", "target"):
        assert not any(forbidden in n.lower() for n in names), (
            f"признак с '{forbidden}' не может быть известен заранее"
        )


def test_future_calendar_matches_target_timestamps(panel):
    """
    Календарь будущего соответствует реальным меткам времени целевого окна.

    Смещение на шаг здесь дало бы модели «сдвинутое» время суток и тихо
    испортило бы прогноз пиковых часов.
    """
    df, specs = panel
    one = df[df["feeder_id"] == sorted(df["feeder_id"].unique())[0]].copy()
    one = one.sort_values("timestamp").reset_index(drop=True)

    from data.panel_preprocessing import build_future_known_frame
    rng = np.random.default_rng(0)
    horizon = 24
    channels = build_future_known_frame(one, horizon, rng, {})

    origin = 100
    expected_hours = one["hour"].values[origin + 1: origin + 1 + horizon]
    got_sin = channels["fut_hour_sin"][origin]
    got_cos = channels["fut_hour_cos"][origin]
    got_hours = np.mod(np.round(np.arctan2(got_sin, got_cos) / (2 * np.pi) * 24), 24)

    assert np.array_equal(got_hours.astype(int), expected_hours.astype(int)), (
        "календарь будущего не совпадает с метками времени целевого окна"
    )


def test_weather_forecast_differs_from_actual(prepared, panel):
    """Прогноз погоды не равен факту — иначе он не был бы прогнозом."""
    df, specs = panel
    one = df[df["feeder_id"] == sorted(df["feeder_id"].unique())[0]].copy()
    one = one.sort_values("timestamp").reset_index(drop=True)

    from data.panel_preprocessing import build_future_known_frame
    rng = np.random.default_rng(0)
    horizon = 24
    channels = build_future_known_frame(
        one, horizon, rng, {"temperature": (0.6, 0.06)})

    forecast = channels["fut_temperature_forecast"]
    actual = one["temperature"].values
    idx = np.clip(np.arange(len(one))[:, None] + np.arange(1, horizon + 1)[None, :],
                  0, len(one) - 1)
    future_actual = actual[idx]

    assert not np.allclose(forecast, future_actual), "прогноз совпал с фактом"
    # Но и не должен быть шумом: связь обязана сохраняться.
    assert np.corrcoef(forecast.ravel(), future_actual.ravel())[0, 1] > 0.8


def test_weather_forecast_error_grows_with_lead_time():
    """
    Ошибка прогноза погоды растёт с горизонтом.

    Постоянная по горизонту ошибка сделала бы прогноз на сутки вперёд таким же
    точным, как на час, чего в действительности не бывает.
    """
    rng = np.random.default_rng(3)
    n, horizon = 4000, 24
    actual = 10.0 + 8.0 * np.sin(np.arange(n) * 2 * np.pi / 24)

    forecast = make_weather_forecast(actual, horizon, rng,
                                     base_sigma=0.6, growth_per_hour=0.06)
    idx = np.clip(np.arange(n)[:, None] + np.arange(1, horizon + 1)[None, :], 0, n - 1)
    err = np.abs(forecast - actual[idx])
    mae_by_lead = err.mean(axis=0)

    assert mae_by_lead[0] < mae_by_lead[-1], "ошибка обязана расти с горизонтом"
    assert mae_by_lead[-1] / mae_by_lead[0] > 1.5, "рост слишком слабый"
    # Монотонность в среднем: сравниваем первую и последнюю трети.
    assert mae_by_lead[:8].mean() < mae_by_lead[16:].mean()


def test_weather_forecast_is_reproducible():
    actual = np.linspace(0, 20, 500)
    a = make_weather_forecast(actual, 12, np.random.default_rng(7), 0.5, 0.05)
    b = make_weather_forecast(actual, 12, np.random.default_rng(7), 0.5, 0.05)
    assert np.allclose(a, b)


def test_tariff_zone_encoding_matches_russian_order():
    """Кодировка тарифных зон совпадает с действующим в России порядком."""
    hours = np.arange(24)
    weekday = np.zeros(24)          # понедельник
    holiday = np.zeros(24)
    code = _tariff_zone_code(hours, weekday, holiday)

    assert code[3] == 0.0, "03:00 — ночная зона"
    assert code[8] == 1.0, "08:00 — утренний пик"
    assert code[18] == 1.0, "18:00 — вечерний пик"
    assert code[12] == 0.5, "12:00 — полупиковая"
    assert code[22] == 0.5, "22:00 — полупиковая"

    weekend = _tariff_zone_code(hours, np.full(24, 5.0), holiday)
    assert not np.any(weekend == 1.0), "в выходные пиковой зоны нет"
