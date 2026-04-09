# -*- coding: utf-8 -*-
"""EV load simulation component."""

from __future__ import annotations

import numpy as np


def simulate_ev_load(
    rng: np.random.Generator,
    *,
    days: int,
    hours: int,
    households: int,
    ev_penetration: float,
    weekday: np.ndarray,
    holiday_mask_daily: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """Simulate residential and commercial EV charging load."""
    n_ev = int(households * ev_penetration)
    ev_load_raw = np.zeros(hours, np.float64)
    for day_idx in range(days):
        is_we_day = bool(weekday[day_idx * 24] >= 5)
        is_hol_day = bool(holiday_mask_daily[day_idx] > 0)
        is_cold = bool(temperature[day_idx * 24] < -15)
        charge_prob = 0.90 if is_cold else 0.85
        n_charging = int(rng.binomial(n_ev, charge_prob))
        home_frac = 0.72 if (is_we_day or is_hol_day) else 0.62

        for _ in range(n_charging):
            base_h = int(rng.choice([21, 22, 23, 0, 1, 2, 3, 20])) if rng.random() < home_frac else int(rng.integers(9, 18))
            duration = int(rng.integers(3, 9))
            power = rng.uniform(11.0, 22.0)
            for h_offset in range(duration):
                abs_h = day_idx * 24 + base_h + h_offset
                if abs_h >= hours:
                    break
                ev_load_raw[abs_h] += power

        if weekday[day_idx * 24] == 4 and not is_hol_day:
            peak_h = day_idx * 24 + 21
            if peak_h < hours:
                ev_load_raw[peak_h] += n_ev * 0.15 * rng.uniform(0.8, 1.2) * 16.0
        if is_cold:
            surge_h = day_idx * 24 + 19
            if surge_h < hours:
                ev_load_raw[surge_h] += n_ev * ev_penetration * rng.uniform(5.0, 10.0)

    n_commercial = max(1, households // 150)
    for _ in range(n_commercial):
        depot_start = int(rng.integers(21, 24))
        kw_truck = rng.uniform(30.0, 50.0)
        n_trucks = int(rng.integers(2, 5))
        for day_idx in range(days):
            next_wd = (day_idx + 1) % 7
            if next_wd < 5 and rng.random() < 0.85:
                duration = int(rng.integers(8, 11))
                for h_offset in range(duration):
                    abs_h = day_idx * 24 + depot_start + h_offset
                    if abs_h >= hours:
                        break
                    ev_load_raw[abs_h] += kw_truck * n_trucks * rng.uniform(0.90, 1.05)
    return ev_load_raw.astype(np.float32)
