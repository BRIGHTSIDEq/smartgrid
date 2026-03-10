# -*- coding: utf-8 -*-
"""
optimization/storage.py — Оптимизация BESS (Battery Energy Storage System).

═══════════════════════════════════════════════════════════════════════════════
ИСПРАВЛЕНИЕ v4 — стратегии по SOC, а не по C-rate

ДИАГНОЗ БАГА v3 (одинаковая экономия):
  При батарее 300 кВт·ч и SOC диапазоне 10–90% (270 кВт·ч):
    - Ночная зона: 8 ч/сутки × 7 дней = 56 ч
    - C=0.3 → 90 кВт → заряд до SOC_max за 270/90 = 3 ч → за 56 ч = ~18 полных циклов
    - C=1.0 → 300 кВт → заряд за 0.9 ч → за 56 ч = ~18 полных циклов
  Итого: все стратегии успевают зарядить батарею ПОЛНОСТЬЮ каждую ночь →
  суммарная прокачка одинакова → экономия одинакова.

КОРЕНЬ ОШИБКИ: C-rate только меняет скорость, не объём.
  Для разной экономии нужно менять ОБЪЁМ (целевой SOC).

ИСПРАВЛЕНИЕ: три стратегии по целевому SOC и глубине разряда:

  Консервативная: заряд до 60%, разряд до 40%  → ΔE = 60 кВт·ч
  Умеренная:      заряд до 75%, разряд до 25%  → ΔE = 150 кВт·ч
  Агрессивная:    заряд до 92%, разряд до 8%   → ΔE = 252 кВт·ч

  Экономия ∝ ΔE × (tariff_peak - tariff_night/√η):
    Консервативная: 60  × (6.50 - 1.85) ≈  279 руб/цикл
    Умеренная:      150 × (6.50 - 1.85) ≈  698 руб/цикл
    Агрессивная:    252 × (6.50 - 1.85) ≈ 1172 руб/цикл

  НО: деградация тоже растёт:
    Агрессивная: 252 × 0.06 = 15.1 руб/цикл (сильнее изнашивает батарею)
    → чистая экономия у умеренной может оказаться лучше агрессивной.

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ:
  Консервативная: ~1–2% экономии
  Умеренная:      ~4–7%
  Агрессивная:    ~7–12%  (но срок службы батареи короче)
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

logger = logging.getLogger("smart_grid.optimization.storage")


# ══════════════════════════════════════════════════════════════════════════════
# ТАРИФНЫЙ МОДУЛЬ
# ══════════════════════════════════════════════════════════════════════════════

def _get_zone(hour: int, weekday: int) -> str:
    if hour < 7 or hour >= 23:
        return "night"
    if weekday >= 5:
        return "day"
    if (10 <= hour < 17) or (21 <= hour < 23):
        return "peak"
    return "day"


def build_price_vector(
    n: int,
    tariff_night: float = 1.80,
    tariff_day: float = 4.20,
    tariff_peak: float = 6.50,
    start_hour: int = 0,
    start_weekday: int = 0,
) -> np.ndarray:
    prices = np.empty(n, dtype=np.float64)
    tariff = {"night": tariff_night, "day": tariff_day, "peak": tariff_peak}
    for i in range(n):
        h = (start_hour + i) % 24
        wd = (start_weekday + (start_hour + i) // 24) % 7
        prices[i] = tariff[_get_zone(h, wd)]
    return prices


def build_zone_list(n: int, start_hour: int = 0, start_weekday: int = 0) -> List[str]:
    return [
        _get_zone((start_hour + i) % 24,
                  (start_weekday + (start_hour + i) // 24) % 7)
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS РЕЗУЛЬТАТА
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StorageResult:
    strategy_name: str = ""
    baseline_cost: float = 0.0
    optimized_cost: float = 0.0
    degradation_cost: float = 0.0
    gross_savings: float = 0.0
    net_savings: float = 0.0
    net_savings_pct: float = 0.0
    battery_levels: List[float] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    energy_from_grid: List[float] = field(default_factory=list)
    prices: np.ndarray = field(default_factory=lambda: np.array([]))
    hourly_costs: List[float] = field(default_factory=list)
    total_energy_cycled: float = 0.0
    n_charge_hours: int = 0
    n_discharge_hours: int = 0
    annual_savings_est: float = 0.0
    payback_years: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ЯДРО СИМУЛЯЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def simulate_storage(
    forecast: np.ndarray,
    capacity: float = 300.0,
    max_power: float = 150.0,
    round_trip_efficiency: float = 0.95,
    cycle_cost_per_kwh: float = 0.06,
    min_soc: float = 0.10,          # нижняя граница SOC (доля от capacity)
    max_soc: float = 0.90,          # верхняя граница SOC
    initial_soc: float = 0.50,
    tariff_night: float = 1.80,
    tariff_half_peak: float = 4.20,
    tariff_peak: float = 6.50,
    start_hour: int = 0,
    start_weekday: int = 0,
    battery_cost_rub: float = 3_000_000.0,
    strategy_name: str = "",
) -> StorageResult:
    """
    Симулирует BESS с умной тарифной стратегией.

    ФИЗИКА (исправленная):
      Зарядка (ночь):
        charge_to_batt = min(max_power, soc_max_kwh - soc)   [→ в батарею]
        grid_draw      = charge_to_batt / one_way_eff         [← из сети]
        soc           += charge_to_batt

      Разрядка (пик):
        can_draw       = min(max_power, soc - soc_min_kwh)   [← из SOC]
        delivered      = can_draw * one_way_eff               [→ в нагрузку]
        soc           -= can_draw
        grid_energy    = max(0, demand - delivered)

    Ключевой момент: min_soc и max_soc контролируют ОБЪЁМ цикла.
    Именно они делают стратегии экономически различимыми.
    """
    n = len(forecast)
    one_way_eff = np.sqrt(round_trip_efficiency)

    prices = build_price_vector(
        n, tariff_night, tariff_half_peak, tariff_peak, start_hour, start_weekday
    )
    zones = build_zone_list(n, start_hour, start_weekday)

    soc = initial_soc * capacity
    soc_min_kwh = min_soc * capacity
    soc_max_kwh = max_soc * capacity

    battery_levels: List[float] = [soc]
    actions: List[str] = []
    energy_from_grid: List[float] = []
    hourly_costs: List[float] = []
    total_cycled = 0.0

    for i in range(n):
        demand = float(forecast[i])
        price = float(prices[i])
        zone = zones[i]

        if zone == "night" and soc < soc_max_kwh - 0.1:
            # ЗАРЯДКА
            charge_to_batt = min(max_power, soc_max_kwh - soc)
            grid_draw = charge_to_batt / one_way_eff
            soc += charge_to_batt
            grid_energy = demand + grid_draw
            action = "charge"
            total_cycled += charge_to_batt

        elif zone == "peak" and soc > soc_min_kwh + 0.1:
            # РАЗРЯДКА
            can_draw = min(max_power, soc - soc_min_kwh)
            delivered = can_draw * one_way_eff
            soc -= can_draw
            grid_energy = max(0.0, demand - delivered)
            action = "discharge"
            total_cycled += can_draw

        else:
            grid_energy = demand
            action = "idle"

        soc = float(np.clip(soc, 0.0, capacity))
        battery_levels.append(soc)
        actions.append(action)
        energy_from_grid.append(grid_energy)
        hourly_costs.append(grid_energy * price)

    # ── Экономика ─────────────────────────────────────────────────────────────
    baseline_cost = float(np.dot(forecast, prices))
    optimized_cost = float(sum(hourly_costs))
    gross_savings = baseline_cost - optimized_cost
    degradation_cost = total_cycled * cycle_cost_per_kwh
    net_savings = gross_savings - degradation_cost
    net_savings_pct = (net_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

    n_ch = actions.count("charge")
    n_dis = actions.count("discharge")
    annual_est = net_savings * (8760.0 / n) if n > 0 else 0.0
    payback = battery_cost_rub / annual_est if annual_est > 0 else float("inf")

    # ── Лог ───────────────────────────────────────────────────────────────────
    soc_range_pct = (max_soc - min_soc) * 100
    logger.info("─" * 50)
    logger.info("Горизонт: %d ч (%.1f сут) | SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч)",
                n, n / 24, min_soc * 100, max_soc * 100, (max_soc - min_soc) * capacity)
    logger.info("Базовая стоимость:         %10.2f руб", baseline_cost)
    logger.info("Оптимизированная (грязная):%10.2f руб", optimized_cost)
    logger.info("Валовая экономия:          %10.2f руб", gross_savings)
    logger.info("Стоимость деградации:      %10.2f руб", degradation_cost)
    logger.info("ЧИСТАЯ экономия:           %10.2f руб (%.2f%%)", net_savings, net_savings_pct)
    logger.info("Прокачано:                 %10.2f кВт·ч", total_cycled)
    logger.info("Часов заряд/разряд:        %d / %d из %d", n_ch, n_dis, n)
    logger.info("Экономия/год (оценка):     %10.2f руб/год", annual_est)
    logger.info("Срок окупаемости:          %10.1f лет", payback)
    if strategy_name:
        logger.info("Стратегия: %s", strategy_name)

    return StorageResult(
        strategy_name=strategy_name,
        baseline_cost=baseline_cost,
        optimized_cost=optimized_cost,
        degradation_cost=degradation_cost,
        gross_savings=gross_savings,
        net_savings=net_savings,
        net_savings_pct=net_savings_pct,
        battery_levels=battery_levels,
        actions=actions,
        energy_from_grid=energy_from_grid,
        prices=prices,
        hourly_costs=hourly_costs,
        total_energy_cycled=total_cycled,
        n_charge_hours=n_ch,
        n_discharge_hours=n_dis,
        annual_savings_est=annual_est,
        payback_years=payback,
    )


# ══════════════════════════════════════════════════════════════════════════════
# СРАВНЕНИЕ СТРАТЕГИЙ (по SOC, не по C-rate)
# ══════════════════════════════════════════════════════════════════════════════

def compare_strategies(
    forecast: np.ndarray,
    capacity: float = 300.0,
    max_power: float = 150.0,
    round_trip_efficiency: float = 0.95,
    cycle_cost_per_kwh: float = 0.06,
    battery_cost_rub: float = 3_000_000.0,
) -> Dict[str, StorageResult]:
    """
    Три стратегии с разным целевым SOC.

    Консервативная: SOC 40%→60%  ΔE= 60 кВт·ч — минимальный износ
    Умеренная:      SOC 25%→75%  ΔE=150 кВт·ч — оптимальный баланс
    Агрессивная:    SOC  8%→92%  ΔE=252 кВт·ч — максимальная экономия, быстрый износ

    ΔE определяет прокачанную энергию → разную экономию и деградацию.
    """
    strategies = [
        ("Консервативная",  0.40, 0.60),  # (name, min_soc, max_soc)
        ("Умеренная",       0.25, 0.75),
        ("Агрессивная",     0.08, 0.92),
    ]

    results: Dict[str, StorageResult] = {}
    for name, min_soc, max_soc in strategies:
        delta_e = (max_soc - min_soc) * capacity
        label = f"{name} (SOC {min_soc*100:.0f}%→{max_soc*100:.0f}%, ΔE={delta_e:.0f} кВт·ч)"
        results[name] = simulate_storage(
            forecast=forecast,
            capacity=capacity,
            max_power=max_power,
            round_trip_efficiency=round_trip_efficiency,
            cycle_cost_per_kwh=cycle_cost_per_kwh,
            min_soc=min_soc,
            max_soc=max_soc,
            battery_cost_rub=battery_cost_rub,
            strategy_name=label,
        )

    # ── Сводная таблица ───────────────────────────────────────────────────────
    logger.info("═" * 65)
    logger.info("%-20s %8s %14s %10s %11s",
                "Стратегия", "ΔE кВт·ч", "Чистая экон.", "Экон.%", "Окупаем.")
    logger.info("─" * 65)
    for (name, min_soc, max_soc), res in zip(strategies, results.values()):
        de = (max_soc - min_soc) * capacity
        payback_str = f"{res.payback_years:.1f} лет" if res.payback_years < 999 else "∞"
        logger.info("%-20s %8.0f %12.0f руб %8.2f%% %11s",
                    name, de, res.net_savings, res.net_savings_pct, payback_str)
    logger.info("═" * 65)

    best = max(results, key=lambda k: results[k].net_savings_pct)
    logger.info("🏆 Лучшая: %s | %.2f%% | окупаемость %.1f лет",
                best, results[best].net_savings_pct, results[best].payback_years)
    return results
