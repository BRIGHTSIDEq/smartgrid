# -*- coding: utf-8 -*-
"""
Единый календарь тарифных зон и согласие с ним всех модулей.

Часы зон были записаны вручную в семи местах, и копии расходились: запасная
ветка признаков переставляла пик и полупик на 16 часах из 24, накопитель не
знал праздников. Здесь проверяется сама таблица зон и то, что генератор,
признаки агрегатной и панельной моделей и накопитель дают одно и то же на
одних и тех же датах.
"""

import numpy as np
import pandas as pd
import pytest

from utils.tariffs import (
    ZONE_CODE, peak_clock_hours, peak_zone_mask, zone_codes, zone_names, zone_of,
)

_WORKDAY = (["night"] * 7 + ["peak"] * 3 + ["day"] * 7 + ["peak"] * 4
            + ["day"] * 2 + ["night"])
_DAY_OFF = ["night"] * 7 + ["day"] * 16 + ["night"]


@pytest.mark.parametrize("weekday, holiday, expected", [
    (0, False, _WORKDAY), (4, False, _WORKDAY),
    (5, False, _DAY_OFF), (6, False, _DAY_OFF),
    (2, True, _DAY_OFF),
])
def test_zone_table_follows_the_russian_three_zone_tariff(weekday, holiday, expected):
    assert [zone_of(h, weekday, holiday) for h in range(24)] == expected


def test_codes_and_names_agree():
    hour = np.tile(np.arange(24), 14)
    weekday = np.repeat(np.arange(14) % 7, 24)
    holiday = np.repeat((np.arange(14) == 9).astype(float), 24)
    names = zone_names(hour, weekday, holiday)
    np.testing.assert_array_equal(zone_codes(hour, weekday, holiday),
                                  np.array([ZONE_CODE[n] for n in names], np.float32))


def test_codes_keep_the_shape_of_a_horizon_matrix():
    hour = np.arange(48).reshape(2, 24) % 24
    assert zone_codes(hour, np.zeros_like(hour), np.zeros_like(hour)).shape == (2, 24)


def test_clock_peak_ignores_the_calendar():
    """Суточный максимум нагрузки бывает и в выходные — в отличие от тарифного пика."""
    hours = np.arange(24)
    assert peak_clock_hours(hours).sum() == 7
    assert not peak_zone_mask(hours, np.full(24, 6)).any()


# ══════════════════════════════════════════════════════════════════════════════
# СОГЛАСИЕ МОДУЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def generated():
    from data.generator import generate_smartgrid_data
    # Январь и май: новогодние и майские праздники попадают в окно.
    return generate_smartgrid_data(days=140, households=30, seed=2)


def test_battery_zones_match_the_generator(generated):
    """Накопитель тарифицирует те же часы, что генератор считает пиковыми."""
    from data.generator import holiday_flags
    from optimization.storage import build_zone_list

    ts = pd.to_datetime(generated["timestamp"])
    zones = build_zone_list(len(ts), int(ts[0].hour), int(ts[0].dayofweek),
                            holiday_flags(ts))
    assert zones == list(generated["tariff_zone"])


def test_feature_encoding_matches_the_generator(generated):
    """Признак зоны в агрегатной и панельной моделях совпадает с колонкой генератора."""
    from data.panel_preprocessing import _tariff_zone_code
    from data.preprocessing import _encode_tariff_zone

    expected = generated["tariff_zone"].map(ZONE_CODE).to_numpy(np.float32)
    np.testing.assert_array_equal(_encode_tariff_zone(generated), expected)
    np.testing.assert_array_equal(
        _encode_tariff_zone(generated.drop(columns="tariff_zone")), expected)

    ts = pd.to_datetime(generated["timestamp"])
    np.testing.assert_array_equal(
        _tariff_zone_code(ts.dt.hour.to_numpy(), ts.dt.dayofweek.to_numpy(),
                          generated["is_holiday"].to_numpy(np.float32)),
        expected)
