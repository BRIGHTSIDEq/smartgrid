# -*- coding: utf-8 -*-
"""Nonlinear latent state helpers for synthetic demand."""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def compute_cascade_factor(
    rng: np.random.Generator,
    base_consumption: np.ndarray,
    threshold_ratio: float = 1.35,
) -> np.ndarray:
    """Compute post-spike inertia factor."""
    hours = len(base_consumption)
    cascade_factor = np.ones(hours, np.float32)
    cascade_state = 0.0
    cascade_threshold = float(base_consumption.mean()) * threshold_ratio
    for h in range(hours):
        bc = float(base_consumption[h])
        if bc > cascade_threshold:
            cascade_state = min(cascade_state + rng.uniform(0.03, 0.08), 0.25)
        else:
            cascade_state = max(cascade_state - 0.04, 0.0)
        cascade_factor[h] = 1.0 + cascade_state
    return cascade_factor


def compute_grid_stress_nonlinear(
    rng: np.random.Generator,
    *,
    temperature: np.ndarray,
    temp_setpoint: float,
    hour_of_day: np.ndarray,
    ev_load_raw: np.ndarray,
    cloud_cover: np.ndarray,
) -> np.ndarray:
    """Compute nonlinear network-stress multiplier."""
    hours = len(temperature)
    grid_stress = np.zeros(hours, np.float32)
    ev_norm_proxy = ev_load_raw / (float(ev_load_raw.max()) + 1e-6)
    temp_extreme = np.clip(np.abs(temperature - temp_setpoint) / 22.0, 0.0, 1.4).astype(np.float32)
    peak_hours = (((hour_of_day >= 7) & (hour_of_day < 10)) | ((hour_of_day >= 18) & (hour_of_day < 22))).astype(np.float32)
    stress_state = 0.0
    for h in range(hours):
        trigger = (
            0.40 * float(temp_extreme[h])
            + 0.25 * float(peak_hours[h])
            + 0.25 * float(ev_norm_proxy[h])
            + 0.10 * float(cloud_cover[h])
        )
        stress_state = 0.90 * stress_state + 0.10 * trigger + float(rng.normal(0.0, 0.015))
        stress_state = float(np.clip(stress_state, 0.0, 1.4))
        grid_stress[h] = stress_state
    return (1.0 + 0.18 * _sigmoid(6.0 * (grid_stress - 0.55)) + 0.06 * (grid_stress**2)).astype(np.float32)


def compute_dsr_rebound(
    rng: np.random.Generator,
    dsr_active: np.ndarray,
) -> np.ndarray:
    """Compute rebound after demand-side-response events."""
    hours = len(dsr_active)
    dsr_rebound = np.zeros(hours, np.float32)
    for h in range(1, hours):
        if dsr_active[h - 1] > 0 and dsr_active[h] == 0:
            rebound_amp = rng.uniform(0.08, 0.16)
            rebound_len = int(rng.integers(2, 5))
            for k in range(rebound_len):
                idx = h + k
                if idx >= hours:
                    break
                dsr_rebound[idx] += rebound_amp * np.exp(-0.65 * k)
    return dsr_rebound
