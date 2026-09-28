# -*- coding: utf-8 -*-
"""
Единый календарь тарифных зон.

Порядок, действующий в России для трёхзонного тарифа:

    пиковая     07:00–10:00 и 17:00–21:00
    полупиковая 10:00–17:00 и 21:00–23:00
    ночная      23:00–07:00

В выходные и праздничные дни пиковая зона не применяется: сутки делятся на
ночную и полупиковую.

Прежде эти часы были записаны вручную в семи местах: генератор, признаки
агрегатной и панельной моделей, накопитель, события управления спросом. Копии
расходились: запасная ветка признаков одно время переставляла пик и полупик,
накопитель не знал праздников. Теперь все они берут границы отсюда.
"""

from typing import Tuple

import numpy as np

NIGHT_START = 23
NIGHT_END = 7
PEAK_WINDOWS: Tuple[Tuple[int, int], ...] = ((7, 10), (17, 21))

# Числовой код зоны в признаках моделей. Порядок монотонен по цене.
ZONE_CODE = {"night": 0.0, "day": 0.5, "peak": 1.0}


def peak_clock_hours(hour) -> np.ndarray:
    """
    Часы пиковых окон без учёта дня недели.

    Нужны там, где важен суточный максимум нагрузки, а не тариф: шум
    потребления и напряжённость сети растут в эти часы и в выходные.
    """
    h = np.asarray(hour)
    mask = np.zeros(h.shape, dtype=bool)
    for lo, hi in PEAK_WINDOWS:
        mask |= (h >= lo) & (h < hi)
    return mask


def night_hours(hour) -> np.ndarray:
    h = np.asarray(hour)
    return (h < NIGHT_END) | (h >= NIGHT_START)


def peak_zone_mask(hour, weekday, holiday=None) -> np.ndarray:
    """Часы пиковой тарифной зоны: пиковое окно рабочего непраздничного дня."""
    workday = np.asarray(weekday) < 5
    if holiday is not None:
        workday = workday & ~(np.asarray(holiday) > 0.5)
    return peak_clock_hours(hour) & workday


def zone_codes(hour, weekday, holiday=None) -> np.ndarray:
    """
    Код зоны для каждого часа: 0 ночная, 0.5 полупиковая, 1 пиковая.

    Форма результата повторяет форму hour: функция вызывается и для ряда
    истории, и для матрицы «момент × шаг горизонта».
    """
    code = np.full(np.shape(hour), ZONE_CODE["day"], dtype=np.float32)
    code[night_hours(hour)] = ZONE_CODE["night"]
    code[peak_zone_mask(hour, weekday, holiday)] = ZONE_CODE["peak"]
    return code


def zone_names(hour, weekday, holiday=None) -> np.ndarray:
    """Названия зон ("night", "day", "peak") для каждого часа."""
    names = np.full(np.shape(hour), "day", dtype=object)
    names[night_hours(hour)] = "night"
    names[peak_zone_mask(hour, weekday, holiday)] = "peak"
    return names


def zone_of(hour: int, weekday: int, holiday: bool = False) -> str:
    """Зона одного часа."""
    return str(zone_names(np.array([hour]), np.array([weekday]),
                          np.array([float(holiday)]))[0])
