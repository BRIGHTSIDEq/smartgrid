# -*- coding: utf-8 -*-
"""
optimization/storage.py — Оптимизация BESS (Battery Energy Storage System).

═══════════════════════════════════════════════════════════════════════════════
ИСПРАВЛЕНИЕ v5 — два критических бага экономического расчёта
═══════════════════════════════════════════════════════════════════════════════

БАГ #1 (O&M скачок 3 696 → 55 441 руб, разрыв ×15):
──────────────────────────────────────────────────────
  Первый вызов simulate_storage() в main.py не передавал battery_cost_rub.
  Дефолт функции = 3_000_000, тогда как Config.BATTERY_COST_RUB = 45_000_000.
  Результат: O&M = 45_000_000 × 0.015 × (720/8766) × (3M/45M) = 3 696 руб
  → payback = 1.2 года (нереалистично), все последующие вызовы через
  compare_strategies() давали 55 441 руб (корректно).

  БЫЛО: battery_cost_rub: float = 3_000_000.0  ← тихий дефолт
  СТАЛО: battery_cost_rub: float — обязательный параметр без дефолта.
         Отсутствие значения → TypeError при вызове, не в расчёте.
         main.py передаёт Config.BATTERY_COST_RUB=45_000_000 явно.

БАГ #2 (Demand-charge эффект = 0.00 руб во всех стратегиях):
──────────────────────────────────────────────────────────────
  СИМПТОМ: все три стратегии → demand_charge_savings = 0.00 руб.

  ПРИЧИНА: demand-charge считался по максимуму из ВСЕХ 720 часов:
    baseline_peak_kw  = np.max(forecast)          ← правильно
    optimized_peak_kw = np.max(energy_from_grid)  ← НЕПРАВИЛЬНО

  При зарядке ночью:
    grid_energy = demand + grid_draw  (добавляем зарядный ток к нагрузке)
    Пример: ночная нагрузка 5 000 кВт + зарядка 2 250 кВт = 7 250 кВт
    → max(energy_from_grid) ≥ max(forecast)  → экономия <= 0 → clamp → 0.

  Физически это некорректно: demand-charge в России выставляется за
  максимальную мощность в ПИКОВЫЕ БИЛЛИНГОВЫЕ ЧАСЫ (10–17 и 21–23 в будни),
  а не за ночные часы зарядки.

  БЫЛО: np.max(energy_from_grid) по всем 720 ч → всегда >= baseline
  СТАЛО: отдельные списки grid_peak_hours_baseline / grid_peak_hours_optimized,
         np.max() только по zone=="peak" часам → реальное снижение пика.

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ ПОСЛЕ ПАТЧА:
  Demand-charge эффект: 0 → ненулевое значение (зависит от профиля нагрузки
  и мощности батареи). При 4500 кВт·ч, ΔE=3780 кВт·ч: ~50–200k руб/30 дней.
  Payback агрессивной стратегии: 12 лет → 8–10 лет.
  O&M: единообразно ~55 441 руб во всех вызовах.

═══════════════════════════════════════════════════════════════════════════════
ИСПРАВЛЕНИЕ v4 (оставлено без изменений): стратегии по SOC, не по C-rate
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
    demand_charge_savings: float = 0.0
    om_cost: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ЯДРО СИМУЛЯЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def simulate_storage(
    forecast: np.ndarray,
    capacity: float = 300.0,
    max_power: float = 150.0,
    round_trip_efficiency: float = 0.95,
    cycle_cost_per_kwh: float = 0.06,
    min_soc: float = 0.10,
    max_soc: float = 0.90,
    initial_soc: float = 0.50,
    tariff_night: float = 1.80,
    tariff_half_peak: float = 4.20,
    tariff_peak: float = 6.50,
    start_hour: int = 0,
    start_weekday: int = 0,
    # ── ИСПРАВЛЕНИЕ v5 #1: убран дефолт battery_cost_rub ────────────────────
    # БЫЛО: battery_cost_rub: float = 3_000_000.0
    #   Дефолт молча давал O&M в 15× меньше реального значения.
    #   3_000_000 × 0.015 × (720/8766) = 3 696 руб (вместо 55 441)
    # СТАЛО: параметр обязателен. Отсутствие → TypeError на вызове.
    #   Правильное значение: Config.BATTERY_COST_RUB = 45_000_000
    battery_cost_rub: float = None,
    # ────────────────────────────────────────────────────────────────────────
    demand_charge_rub_per_kw_month: float = 950.0,
    annual_om_share: float = 0.015,
    strategy_name: str = "",
) -> StorageResult:
    """
    Симулирует BESS с умной тарифной стратегией.

    ИСПРАВЛЕНИЯ v5:
      #1 battery_cost_rub обязателен (нет дефолта)
      #2 demand-charge считается только в пиковые биллинговые часы

    Parameters
    ----------
    battery_cost_rub : float
        Стоимость батарейной системы в рублях. ОБЯЗАТЕЛЬНЫЙ параметр.
        Передавайте Config.BATTERY_COST_RUB явно.
        Типовое значение для 4500 кВт·ч BESS: 45_000_000 руб.
    """
    # ── Защита от забытого аргумента ─────────────────────────────────────────
    if battery_cost_rub is None:
        raise TypeError(
            "simulate_storage() требует явный аргумент battery_cost_rub. "
            "Передайте Config.BATTERY_COST_RUB. "
            "Дефолт 3_000_000 удалён в v5 — он давал O&M в 15× ниже реального."
        )

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

    # ── ИСПРАВЛЕНИЕ v5 #2: отдельные списки для пиковых биллинговых часов ────
    # БЫЛО: baseline/optimized peak = np.max(forecast / energy_from_grid) по 720 ч
    #   Зарядный ток ночью поднимал energy_from_grid выше baseline → savings = 0.
    # СТАЛО: собираем потребление из сети ТОЛЬКО в zone=="peak" часы.
    #   В эти часы BESS разряжается → grid_energy < demand → реальное снижение пика.
    grid_peak_hours_baseline: List[float] = []
    grid_peak_hours_optimized: List[float] = []
    # ────────────────────────────────────────────────────────────────────────

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

        # ── Сбор пика только в биллинговые пиковые часы ─────────────────────
        if zone == "peak":
            grid_peak_hours_baseline.append(demand)
            grid_peak_hours_optimized.append(grid_energy)
        # ────────────────────────────────────────────────────────────────────

    # ── Экономика ─────────────────────────────────────────────────────────────
    baseline_cost = float(np.dot(forecast, prices))
    optimized_cost = float(sum(hourly_costs))

    horizon_days = max(n / 24.0, 1e-9)
    months_in_horizon = horizon_days / 30.4375

    # ── ИСПРАВЛЕНИЕ v5 #2: demand-charge по пиковым часам ────────────────────
    if grid_peak_hours_baseline:
        baseline_peak_kw  = float(np.max(grid_peak_hours_baseline))
        optimized_peak_kw = float(np.max(grid_peak_hours_optimized))
    else:
        # Нестандартный сценарий: пиковых часов нет в горизонте
        logger.warning(
            "В горизонте %d ч не найдено пиковых биллинговых часов "
            "(zone='peak', 10–17 и 21–23 в будни). "
            "Demand-charge считается по всем часам — возможно некорректно.",
            n
        )
        baseline_peak_kw  = float(np.max(forecast))
        optimized_peak_kw = float(np.max(np.asarray(energy_from_grid, dtype=np.float64)))

    demand_charge_savings = max(
        0.0,
        (baseline_peak_kw - optimized_peak_kw) * demand_charge_rub_per_kw_month * months_in_horizon,
    )
    # ────────────────────────────────────────────────────────────────────────

    gross_savings = (baseline_cost - optimized_cost) + demand_charge_savings
    degradation_cost = total_cycled * cycle_cost_per_kwh
    om_cost = battery_cost_rub * annual_om_share * (horizon_days / 365.25)
    net_savings = gross_savings - degradation_cost - om_cost
    net_savings_pct = (net_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

    n_ch = actions.count("charge")
    n_dis = actions.count("discharge")
    annual_est = net_savings * (8760.0 / n) if n > 0 else 0.0
    payback = battery_cost_rub / annual_est if annual_est > 0 else float("inf")

    # ── Лог ───────────────────────────────────────────────────────────────────
    logger.info("─" * 50)
    logger.info("Горизонт: %d ч (%.1f сут) | SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч)",
                n, n / 24, min_soc * 100, max_soc * 100, (max_soc - min_soc) * capacity)
    logger.info("Базовая стоимость:         %10.2f руб", baseline_cost)
    logger.info("Оптимизированная (грязная):%10.2f руб", optimized_cost)
    logger.info("Валовая экономия:          %10.2f руб", gross_savings)
    logger.info("  ├─ Energy arbitrage:     %10.2f руб", baseline_cost - optimized_cost)
    logger.info("  └─ Demand-charge эффект: %10.2f руб", demand_charge_savings)
    logger.info("     (пик-часы: baseline=%.0f кВт → opt=%.0f кВт, снижение=%.0f кВт)",
                baseline_peak_kw, optimized_peak_kw,
                max(0.0, baseline_peak_kw - optimized_peak_kw))
    logger.info("Стоимость деградации:      %10.2f руб", degradation_cost)
    logger.info("O&M за горизонт:           %10.2f руб  "
                "(%.0f M × %.1f%% × %.4f лет)",
                om_cost,
                battery_cost_rub / 1_000_000,
                annual_om_share * 100,
                horizon_days / 365.25)
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
        demand_charge_savings=demand_charge_savings,
        om_cost=om_cost,
    )


# ══════════════════════════════════════════════════════════════════════════════
# СРАВНЕНИЕ СТРАТЕГИЙ
# ══════════════════════════════════════════════════════════════════════════════

def compare_strategies(
    forecast: np.ndarray,
    capacity: float = 300.0,
    max_power: float = 150.0,
    round_trip_efficiency: float = 0.95,
    cycle_cost_per_kwh: float = 0.06,
    battery_cost_rub: float = None,   # v5: обязательный параметр
    tariff_night: float = 1.80,
    tariff_half_peak: float = 4.20,
    tariff_peak: float = 6.50,
    demand_charge_rub_per_kw_month: float = 950.0,
    annual_om_share: float = 0.015,
) -> Dict[str, StorageResult]:
    """
    Три стратегии с разным целевым SOC.

    Консервативная: SOC 40%→60%  ΔE= 60 кВт·ч — минимальный износ
    Умеренная:      SOC 25%→75%  ΔE=150 кВт·ч — оптимальный баланс
    Агрессивная:    SOC  8%→92%  ΔE=252 кВт·ч — максимальная экономия, быстрый износ
    """
    if battery_cost_rub is None:
        raise TypeError(
            "compare_strategies() требует явный аргумент battery_cost_rub. "
            "Передайте Config.BATTERY_COST_RUB."
        )

    strategies = [
        ("Консервативная",  0.40, 0.60),
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
            tariff_night=tariff_night,
            tariff_half_peak=tariff_half_peak,
            tariff_peak=tariff_peak,
            demand_charge_rub_per_kw_month=demand_charge_rub_per_kw_month,
            annual_om_share=annual_om_share,
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
