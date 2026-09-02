# -*- coding: utf-8 -*-
"""
Тесты корректности подготовки данных.

Проверяется главное, что может незаметно испортить результаты работы:
утечка тестовой информации в обучающую выборку и пересечение входных окон
с целевыми значениями.
"""

import numpy as np
import pandas as pd
import pytest

from data.generator import generate_smartgrid_data
from data.preprocessing import (
    prepare_data, _make_multivariate_windows, inverse_scale,
    reconstruct_day_ahead_series, actual_series_for_forecast, N_FEATURES,
)


@pytest.fixture(scope="module")
def small_df():
    return generate_smartgrid_data(days=30, households=40, seed=7,
                                   industrial_loads=2, city_districts=3)


@pytest.fixture(scope="module")
def prepared(small_df):
    return prepare_data(small_df, history_length=48, forecast_horizon=24,
                        train_ratio=0.70, val_ratio=0.15)


# ── Утечка данных ─────────────────────────────────────────────────────────────

def test_consumption_scaler_fitted_on_train_only(small_df, prepared):
    """Скалер потребления не должен видеть максимум val/test."""
    total = len(small_df)
    train_end = int(total * 0.70)
    train_max = float(small_df["consumption"].values[:train_end].max())
    train_min = float(small_df["consumption"].values[:train_end].min())

    scaler = prepared["scaler"]
    assert scaler.data_max_[0] == pytest.approx(train_max, rel=1e-6)
    assert scaler.data_min_[0] == pytest.approx(train_min, rel=1e-6)


def test_covariate_scalers_fitted_on_train_only(small_df, prepared):
    """
    Все ковариатные скалеры обучены только на train.

    Раньше ev/solar/temperature² нормировались на максимум по всему ряду —
    это тихая утечка, которую не видно ни в одной метрике.
    """
    total = len(small_df)
    train_end = int(total * 0.70)
    checks = [
        ("temp_scaler", "temperature"),
        ("humidity_scaler", "humidity"),
        ("wind_scaler", "wind_speed"),
        ("cloud_scaler", "cloud_cover"),
        ("ev_scaler", "ev_load_kw"),
        ("solar_scaler", "solar_gen_kw"),
    ]
    for scaler_key, column in checks:
        scaler = prepared[scaler_key]
        assert scaler is not None, f"{scaler_key} не создан"
        expected = float(small_df[column].values[:train_end].max())
        assert scaler.data_max_[0] == pytest.approx(expected, rel=1e-6), (
            f"{scaler_key} обучен не только на train"
        )


def test_temp_squared_normalized_by_train_max(small_df, prepared):
    """temperature² нормируется по train-максимуму, а не по всему ряду."""
    total = len(small_df)
    train_end = int(total * 0.70)
    expected = float((small_df["temperature"].values[:train_end] ** 2).max())
    assert prepared["temp_sq_max"] == pytest.approx(expected, rel=1e-3)


def test_generator_emits_raw_not_normalized(small_df):
    """Генератор отдаёт сырые физические величины, а не поделённые на максимум."""
    assert "ev_load_kw" in small_df.columns
    assert "solar_gen_kw" in small_df.columns
    assert "ev_load_norm" not in small_df.columns
    assert "solar_gen_norm" not in small_df.columns
    assert "temperature_squared" not in small_df.columns
    # Сырое потребление измеряется в кВт и заведомо выходит за пределы [0, 1].
    assert small_df["consumption"].max() > 1.5

    # Электротранспорт: при фактическом проникновении в 0.5% на несколько
    # десятков домохозяйств не приходится ни одного автомобиля, поэтому для
    # проверки размерности сценарий задаётся явно.
    df_ev = generate_smartgrid_data(days=20, households=200, seed=11,
                                    ev_penetration=0.30,
                                    industrial_loads=1, city_districts=2)
    assert df_ev["ev_load_kw"].max() > 3.0, "Нагрузка ЭТ должна быть в кВт, не нормирована"


# ── Корректность окон ─────────────────────────────────────────────────────────

def test_windows_do_not_overlap_targets():
    """
    Целевые значения не должны попадать во входное окно.

    Это фундаментальная проверка: при ошибке индексации модель получила бы
    ответ на входе и показала бы фантастические метрики.
    """
    n, features = 200, 3
    series = np.arange(n * features, dtype=np.float32).reshape(n, features)
    history, horizon = 10, 5

    X, Y = _make_multivariate_windows(series, history, horizon)

    for i in (0, 7, len(X) - 1):
        window_values = set(X[i, :, 0].tolist())
        target_values = set(Y[i].tolist())
        assert not (window_values & target_values), (
            f"Окно {i}: вход и цель пересекаются"
        )
        # Цель идёт строго после окна.
        assert min(target_values) > max(window_values)


def test_window_shapes(prepared):
    assert prepared["X_train"].shape[1:] == (48, N_FEATURES)
    assert prepared["Y_train"].shape[1] == 24
    assert len(prepared["X_train"]) == len(prepared["Y_train"])
    assert len(prepared["X_val"]) > 0
    assert len(prepared["X_test"]) > 0


def test_splits_are_chronological(prepared):
    assert prepared["train_end_idx"] < prepared["val_end_idx"]
    assert prepared["val_end_idx"] < prepared["test_start_idx"]


def test_no_nan_or_inf(prepared):
    for split in ("train", "val", "test"):
        X, Y = prepared[f"X_{split}"], prepared[f"Y_{split}"]
        assert np.isfinite(X).all(), f"X_{split} содержит NaN/Inf"
        assert np.isfinite(Y).all(), f"Y_{split} содержит NaN/Inf"


def test_inverse_scale_roundtrip(prepared):
    scaled = prepared["Y_test"]
    restored = inverse_scale(prepared["scaler"], scaled)
    rescaled = prepared["scaler"].transform(restored.reshape(-1, 1)).reshape(scaled.shape)
    assert np.allclose(scaled, rescaled, atol=1e-4)


# ── Реконструкция прогнозного ряда ────────────────────────────────────────────

def test_reconstruct_day_ahead_series_is_continuous():
    """
    Ряд «день вперёд» собирается из непересекающихся горизонтов.

    Окно 0 даёт часы 0..23, окно 24 — часы 24..47 и т.д. Простое разворачивание
    матрицы (N, H) дало бы совсем другой порядок моментов времени.
    """
    horizon = 24
    n_windows = 100
    # Значение = абсолютный момент времени, которому соответствует прогноз.
    preds = np.array([[i + j for j in range(horizon)] for i in range(n_windows)],
                     dtype=np.float32)

    series = reconstruct_day_ahead_series(preds, forecast_horizon=horizon)

    expected = np.arange(len(series), dtype=np.float32)
    assert np.array_equal(series, expected), (
        "Ряд не непрерывен во времени: стыковка горизонтов нарушена"
    )


def test_actual_series_alignment(prepared):
    """Факт для сравнения с прогнозом сдвинут ровно на длину истории."""
    n_hours = 48
    actual = actual_series_for_forecast(prepared, n_hours)
    history = prepared["history_length"]
    expected = prepared["raw_test"][history:history + n_hours]
    assert np.allclose(actual, expected)


def test_tariff_zone_fallback_matches_the_column():
    """
    Обе реализации тарифной кодировки дают одно и то же.

    Запасная ветка срабатывает при инференсе на DataFrame без колонки
    tariff_zone. Прежняя её версия присваивала пик часам 10-17 и 21-23, то есть
    ровно полупиковым, — зоны были переставлены на 16 часах из 24, и прогноз
    ухудшался молча, без единого признака в метриках.
    """
    import numpy as np
    import pandas as pd

    from data.preprocessing import _encode_tariff_zone
    from data.panel_preprocessing import _tariff_zone_code

    hours = np.tile(np.arange(24), 7)
    weekday = np.repeat(np.arange(7), 24)
    frame = pd.DataFrame({"hour": hours, "weekday": weekday,
                          "is_holiday": np.zeros(len(hours), dtype=np.int8)})

    fallback = _encode_tariff_zone(frame)
    reference = _tariff_zone_code(hours.astype(np.float32),
                                  weekday.astype(np.float32),
                                  np.zeros(len(hours), np.float32))

    assert np.allclose(fallback, reference), (
        f"расходятся на часах {np.where(~np.isclose(fallback, reference))[0].tolist()[:10]}"
    )

    # Ветка с колонкой обязана совпадать с ними обеими.
    zones = np.where((hours >= 7) & (hours < 10) | (hours >= 17) & (hours < 21), "peak",
                     np.where((hours < 7) | (hours >= 23), "night", "day"))
    zones = np.where(weekday >= 5, np.where((hours < 7) | (hours >= 23), "night", "day"), zones)
    with_column = _encode_tariff_zone(frame.assign(tariff_zone=zones))
    assert np.allclose(with_column, reference)


def test_calendar_features_follow_the_actual_dates():
    """
    Календарные признаки согласованы с колонкой timestamp при любой стартовой дате.

    Прежде weekday считался как day_of_sim % 7 и был верен только потому, что
    START_DATE приходилась на понедельник. При другой дате день недели,
    выходные, пиковые часы и тарифная зона разошлись бы с метками времени, и
    модель училась бы на календаре, не соответствующем данным. Ошибка не видна
    ни в одной проверке форм.
    """
    import pandas as pd

    from data.generator import generate_smartgrid_data

    # Четверг, чтобы остаток от деления заведомо не совпал с истиной.
    df = generate_smartgrid_data(days=20, households=300, seed=3,
                                 start_date="2024-03-07")
    stamps = pd.DatetimeIndex(df["timestamp"])

    assert (df["weekday"].to_numpy() == stamps.dayofweek.to_numpy()).all()
    assert (df["hour"].to_numpy() == stamps.hour.to_numpy()).all()
    assert (df["is_weekend"].to_numpy().astype(bool)
            == np.asarray(stamps.dayofweek >= 5)).all()
    assert (df["day_of_year"].to_numpy() == stamps.dayofyear.to_numpy()).all()
