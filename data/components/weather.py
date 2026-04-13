# -*- coding: utf-8 -*-
"""Weather generation component for smart-grid synthetic data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class WeatherOutputs:
    temperature: np.ndarray
    cloud_cover: np.ndarray
    humidity: np.ndarray
    wind_speed: np.ndarray
    heat_surge_factor: np.ndarray
    cold_wave_factor: np.ndarray


def _smooth_event_envelope(length: int, ramp: int = 12) -> np.ndarray:
    """Cosine ramp envelope to avoid abrupt jumps at event borders."""
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    ramp = int(max(1, min(ramp, length // 2)))
    env = np.ones(length, dtype=np.float32)
    up = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp, dtype=np.float32))
    env[:ramp] = up
    env[-ramp:] = up[::-1]
    return env


def compute_weather(
    rng: np.random.Generator,
    t: np.ndarray,
    dates: pd.DatetimeIndex,
    days: int,
) -> WeatherOutputs:
    """Generate weather traces with seasonality and stochastic waves."""
    hours = len(t)
    temp_annual = 5.0 + 20.0 * np.sin(2 * np.pi * t / (24 * 365.25) - np.pi / 2)
    temp_diurnal = 3.5 * np.sin(2 * np.pi * (t % 24) / 24 - np.pi / 4)
    tn = np.zeros(hours)
    tn[0] = rng.normal(0, 1.5)
    for i in range(1, hours):
        tn[i] = 0.97 * tn[i - 1] + rng.normal(0, 0.6)
    temperature = np.clip(temp_annual + temp_diurnal + tn, -38.0, 45.0).astype(np.float32)

    heat_surge_factor = np.ones(hours, np.float32)
    for idx in np.where((dates.month >= 6) & (dates.month <= 8))[0][: max(1, int(days / 365 * 3)) * 50 : 50]:
        e = min(idx + int(rng.integers(48, 120)), hours)
        event_len = e - idx
        envelope = _smooth_event_envelope(event_len, ramp=12)
        temp_boost = rng.uniform(8, 14) * envelope
        factor_boost = 1.0 + (rng.uniform(1.28, 1.40) - 1.0) * envelope
        temperature[idx:e] = np.clip(temperature[idx:e] + temp_boost, -38, 45)
        heat_surge_factor[idx:e] = np.maximum(heat_surge_factor[idx:e], factor_boost.astype(np.float32))

    cold_wave_factor = np.ones(hours, np.float32)
    for idx in np.where((dates.month == 12) | (dates.month <= 2))[0][: max(1, days // 120) * 50 : 50]:
        e = min(idx + int(rng.integers(72, 168)), hours)
        event_len = e - idx
        envelope = _smooth_event_envelope(event_len, ramp=16)
        temp_drop = rng.uniform(5, 12) * envelope
        factor_boost = 1.0 + (rng.uniform(1.12, 1.22) - 1.0) * envelope
        temperature[idx:e] = np.clip(temperature[idx:e] - temp_drop, -38, 45)
        cold_wave_factor[idx:e] = np.maximum(cold_wave_factor[idx:e], factor_boost.astype(np.float32))

    cloud_annual = 0.52 + 0.18 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    cn = np.zeros(hours)
    cn[0] = rng.normal(0, 0.08)
    for i in range(1, hours):
        cn[i] = 0.92 * cn[i - 1] + rng.normal(0, 0.06)
    cloud_cover = np.clip(cloud_annual + cn, 0.0, 1.0).astype(np.float32)

    hum = 60.0 + 8.0 * np.sin(2 * np.pi * t / (24 * 365.25))
    hn = np.zeros(hours)
    hn[0] = rng.normal(0, 5.0)
    for i in range(1, hours):
        hn[i] = 0.85 * hn[i - 1] + rng.normal(0, 3.0)
    humidity = np.clip(hum + hn, 20.0, 98.0).astype(np.float32)

    wind_base = 4.0 + 2.5 * np.cos(2 * np.pi * t / (24 * 365.25) + np.pi)
    wind_speed = np.clip(wind_base + rng.exponential(2.0, hours), 0.0, 30.0).astype(np.float32)

    return WeatherOutputs(
        temperature=temperature,
        cloud_cover=cloud_cover,
        humidity=humidity,
        wind_speed=wind_speed,
        heat_surge_factor=heat_surge_factor,
        cold_wave_factor=cold_wave_factor,
    )