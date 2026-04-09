# -*- coding: utf-8 -*-
"""Industrial load component."""

from __future__ import annotations

import numpy as np


def simulate_industrial_load(
    rng: np.random.Generator,
    *,
    hours: int,
    industrial_loads: int,
    weekday: np.ndarray,
) -> np.ndarray:
    """Simulate industrial demand from multiple semi-Markov factories."""
    industrial_load_raw = np.zeros(hours, np.float32)
    for _ in range(industrial_loads):
        power_kw = rng.uniform(500, 2500)
        p_wd = rng.uniform(0.65, 0.90)
        p_we = rng.uniform(0.15, 0.40)
        in_work = False
        till_ch = int(rng.integers(4, 12))
        for h in range(hours):
            p = p_wd if weekday[h] < 5 else p_we
            if till_ch <= 0:
                in_work = rng.random() < (p if not in_work else (1 - p * 0.3))
                till_ch = int(rng.integers(8, 17) if in_work else rng.integers(4, 9))
            if in_work:
                industrial_load_raw[h] += power_kw * rng.uniform(0.92, 1.08)
            till_ch -= 1
    return industrial_load_raw
