# -*- coding: utf-8 -*-
"""Household demand profile component."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _gaussian_peak(hour_arr: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-((hour_arr - center) ** 2) / (2 * width**2))


def _build_household_profiles(
    hour_arr: np.ndarray,
    is_weekend: np.ndarray,
    is_holiday: np.ndarray,
    weekend_scale: float = 0.88,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = hour_arr.astype(np.float32)
    b = 0.65
    early_wd = b + 0.90 * _gaussian_peak(h, 7.0, 1.2) + 0.35 * _gaussian_peak(h, 12.5, 1.5) + 0.45 * _gaussian_peak(h, 18.5, 2.0)
    standard_wd = b + 0.45 * _gaussian_peak(h, 8.0, 1.5) + 0.30 * _gaussian_peak(h, 13.0, 1.5) + 1.00 * _gaussian_peak(h, 19.0, 2.5)
    night_wd = b + 0.25 * _gaussian_peak(h, 8.5, 1.5) + 0.35 * _gaussian_peak(h, 13.5, 1.5) + 0.80 * _gaussian_peak(h, 21.5, 2.0)
    early_we = b + 0.75 * _gaussian_peak(h, 9.0, 2.0) + 0.40 * _gaussian_peak(h, 14.0, 2.0) + 0.50 * _gaussian_peak(h, 19.5, 3.0)
    standard_we = b + 0.35 * _gaussian_peak(h, 10.0, 2.0) + 0.40 * _gaussian_peak(h, 14.0, 2.0) + 0.85 * _gaussian_peak(h, 20.0, 3.5)
    night_we = b + 0.20 * _gaussian_peak(h, 11.0, 2.0) + 0.40 * _gaussian_peak(h, 15.0, 2.0) + 0.90 * _gaussian_peak(h, 22.0, 2.5)

    cooking = 0.30 * _gaussian_peak(h, 14.0, 1.5)
    is_we = is_weekend.astype(bool) | is_holiday.astype(bool)
    is_hol = is_holiday.astype(bool)
    early = np.where(is_we, early_we, early_wd)
    standard = np.where(is_we, standard_we, standard_wd)
    night = np.where(is_we, night_we, night_wd)
    early = np.where(is_hol, early + cooking, early)
    standard = np.where(is_hol, standard + cooking, standard)
    night = np.where(is_hol, night + cooking, night)
    early = np.where(is_we, early * weekend_scale, early)
    standard = np.where(is_we, standard * weekend_scale, standard)
    night = np.where(is_we, night * weekend_scale, night)
    return early.astype(np.float32), standard.astype(np.float32), night.astype(np.float32)


def build_household_aggregate(
    rng: np.random.Generator,
    hour_arr: np.ndarray,
    is_weekend: np.ndarray,
    holiday_mask: np.ndarray,
    households: int,
    early_bird_frac: float,
    night_owl_frac: float,
    days: int,
) -> np.ndarray:
    """Build aggregate household demand profile including regime shifts."""
    n_early = int(households * early_bird_frac)
    n_night = int(households * night_owl_frac)
    n_standard = households - n_early - n_night
    ep, sp, np_ = _build_household_profiles(hour_arr, is_weekend, holiday_mask)
    agg = (n_early * ep + n_standard * sp + n_night * np_) / households
    regime = np.ones_like(agg, dtype=np.float32)
    s = 0
    while s < days:
        se = min(s + int(rng.integers(55, 90)), days)
        regime[s * 24 : se * 24] = 1.0 + rng.uniform(-0.09, 0.09)
        s = se
    return (agg * regime).astype(np.float32)
