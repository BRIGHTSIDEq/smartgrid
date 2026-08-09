# -*- coding: utf-8 -*-
"""
optimization/storage.py — Оптимизация накопителя энергии (BESS).

Симулирует работу батарейной системы на горизонте планирования и считает
экономический эффект: тарифный арбитраж, снижение платы за мощность,
стоимость деградации и O&M, срок окупаемости.

РАЗДЕЛЕНИЕ «ПРОГНОЗ» И «ФАКТ» — ключевой момент методики.
Контроллер накопителя принимает решения по ПРОГНОЗУ (`forecast`), а физические
энергопотоки и счёт за электроэнергию определяются ФАКТОМ (`actual`).
Если передан только `forecast`, он же считается фактом — это идеальный случай
(oracle), полезный как верхняя граница эффекта. Разница между прогоном на
прогнозе модели и прогоном на факте и есть денежная цена ошибки прогноза —
именно она связывает задачу прогнозирования с задачей оптимизации.

ДВЕ СТРАТЕГИИ УПРАВЛЕНИЯ (`policy`):
  "tariff"       — календарная: заряд в ночной зоне, разряд в пиковой.
                   Прогноз не используется вообще, поэтому качество модели
                   на результат не влияет. Служит контрольной точкой.
  "peak_shaving" — прогноз-зависимая: порог срезки считается как квантиль
                   прогноза на ближайшие 24 ч, разряд идёт пропорционально
                   превышению прогноза над порогом. Ошибка прогноза напрямую
                   переходит в недобор экономии: заниженный прогноз оставляет
                   пик несрезанным, завышенный тратит заряд впустую.

ТАРИФНЫЙ КАЛЕНДАРЬ. Зоны определяются часом и днём недели, поэтому вызывающая
сторона ОБЯЗАНА передать `start_hour` и `start_weekday`, соответствующие первой
точке ряда. Значения по умолчанию (понедельник 00:00) верны лишь случайно и
при несовпадении сдвигают все тарифные зоны, обесценивая расчёт.

История изменений — в CHANGELOG.md.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("smart_grid.optimization.storage")


# ══════════════════════════════════════════════════════════════════════════════
# ТАРИФНЫЙ МОДУЛЬ
# ══════════════════════════════════════════════════════════════════════════════

def _get_zone(hour: int, weekday: int) -> str:
    """
    Тарифная зона по часу и дню недели (порядок, действующий в России).

        пиковая     07:00–10:00 и 17:00–21:00 — утренний и вечерний максимумы
        полупиковая 10:00–17:00 и 21:00–23:00
        ночная      23:00–07:00

    В выходные и праздничные дни пиковая зона не применяется: сутки делятся
    только на ночную и полупиковую («дневную») зоны.
    """
    if hour < 7 or hour >= 23:
        return "night"
    if weekday >= 5:
        return "day"
    if (7 <= hour < 10) or (17 <= hour < 21):
        return "peak"
    return "day"


def build_price_vector(
    n: int,
    tariff_night: float = 4.08,
    tariff_day: float = 7.87,
    tariff_peak: float = 11.24,
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
    policy: str = ""
    forecast_source: str = ""
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
    peak_before_kw: float = 0.0
    peak_after_kw: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ЯДРО СИМУЛЯЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def cycle_margin_rub_per_kwh(
    discharge_price: float,
    charge_price: float,
    one_way_eff: float,
    cycle_cost_per_kwh: float,
) -> float:
    """
    Маржа одного цикла в расчёте на 1 кВт·ч, снятый с батареи.

        выручка  = цена разряда × КПД разряда
        затраты  = цена заряда / КПД заряда  +  удельная деградация

    Величина знаковая: при современной стоимости накопителя деградация
    сопоставима с тарифным спредом, поэтому цикл «ночь → полупик» уходит в
    минус, а «ночь → пик» остаётся слабо прибыльным. Контроллер, который этого
    не проверяет, изнашивает батарею быстрее, чем зарабатывает.
    """
    revenue = discharge_price * one_way_eff
    cost = charge_price / one_way_eff + cycle_cost_per_kwh
    return revenue - cost


def _shaving_thresholds(
    forecast: np.ndarray,
    quantile: float = 0.85,
    window: int = 24,
) -> np.ndarray:
    """
    Порог срезки пика для каждого часа: квантиль ПРОГНОЗА на ближайшие
    `window` часов.

    Контроллер день-вперёд знает только прогноз, поэтому и порог считается по
    нему. Именно здесь ошибка прогноза превращается в потерю денег: завышенный
    прогноз поднимает порог и оставляет реальный пик несрезанным.
    """
    f = np.asarray(forecast, dtype=np.float64)
    n = len(f)
    thr = np.empty(n, dtype=np.float64)
    for i in range(n):
        end = min(i + window, n)
        thr[i] = np.quantile(f[i:end], quantile)
    return thr


def simulate_storage(
    forecast: np.ndarray,
    capacity: float = 300.0,
    max_power: float = 150.0,
    round_trip_efficiency: float = 0.95,
    cycle_cost_per_kwh: float = 0.06,
    min_soc: float = 0.10,
    max_soc: float = 0.90,
    initial_soc: float = 0.50,
    tariff_night: float = 4.08,
    tariff_half_peak: float = 7.87,
    tariff_peak: float = 11.24,
    start_hour: int = 0,
    start_weekday: int = 0,
    battery_cost_rub: Optional[float] = None,
    demand_charge_rub_per_kw_month: float = 950.0,
    annual_om_share: float = 0.015,
    strategy_name: str = "",
    actual: Optional[np.ndarray] = None,
    policy: str = "tariff",
    shave_quantile: float = 0.85,
    forecast_source: str = "",
    require_positive_margin: bool = True,
) -> StorageResult:
    """
    Симулирует работу накопителя и считает экономический эффект.

    Parameters
    ----------
    forecast : np.ndarray
        Прогноз нагрузки, по которому контроллер ПРИНИМАЕТ РЕШЕНИЯ.
    actual : np.ndarray, optional
        Фактическая нагрузка, определяющая энергопотоки и счёт. Если не задана,
        считается равной прогнозу (идеальный прогноз, верхняя граница эффекта).
    policy : {"tariff", "peak_shaving"}
        Стратегия управления. "tariff" не использует прогноз вообще,
        "peak_shaving" — использует (см. docstring модуля).
    shave_quantile : float
        Квантиль прогноза, задающий порог срезки для policy="peak_shaving".
    battery_cost_rub : float
        Стоимость батарейной системы, руб. ОБЯЗАТЕЛЬНЫЙ параметр: тихий дефолт
        занижал бы O&M и срок окупаемости в разы.
    start_hour, start_weekday : int
        Календарная привязка первой точки ряда. Без неё тарифные зоны
        сдвигаются относительно данных и весь расчёт теряет смысл.
    require_positive_margin : bool
        Запрещать разряд, если цикл убыточен: выручка от разряда не покрывает
        стоимость ночного заряда и износа. Исключение — пиковая биллинговая
        зона: там разряд снижает плату за мощность, а она на порядок превышает
        потери одного цикла. Без этой проверки контроллер циклирует батарею в
        полупиковой зоне себе в убыток.
    """
    if battery_cost_rub is None:
        raise TypeError(
            "simulate_storage() требует явный аргумент battery_cost_rub. "
            "Передайте Config.BATTERY_COST_RUB."
        )
    if policy not in ("tariff", "peak_shaving"):
        raise ValueError(f"Неизвестная стратегия policy={policy!r}")

    forecast = np.asarray(forecast, dtype=np.float64)
    actual_arr = forecast if actual is None else np.asarray(actual, dtype=np.float64)
    if len(actual_arr) != len(forecast):
        raise ValueError(
            f"Длины прогноза ({len(forecast)}) и факта ({len(actual_arr)}) не совпадают"
        )

    n = len(forecast)
    one_way_eff = np.sqrt(round_trip_efficiency)

    prices = build_price_vector(
        n, tariff_night, tariff_half_peak, tariff_peak, start_hour, start_weekday
    )
    zones = build_zone_list(n, start_hour, start_weekday)
    thresholds = (_shaving_thresholds(forecast, shave_quantile)
                  if policy == "peak_shaving" else np.zeros(n))

    soc = initial_soc * capacity
    soc_min_kwh = min_soc * capacity
    soc_max_kwh = max_soc * capacity

    battery_levels: List[float] = [soc]
    actions: List[str] = []
    energy_from_grid: List[float] = []
    hourly_costs: List[float] = []
    total_cycled = 0.0

    # Плата за мощность в РФ взимается по максимуму в пиковые биллинговые часы
    # (07–10 и 17–21 в будни), а не по максимуму за все сутки. Поэтому пик
    # собирается только по zone == "peak": иначе ночной зарядный ток задрал бы
    # «оптимизированный» максимум выше базового и экономия всегда была бы нулевой.
    grid_peak_hours_baseline: List[float] = []
    grid_peak_hours_optimized: List[float] = []

    def _discharge_is_worthwhile(discharge_price: float, zone: str) -> bool:
        """Оправдан ли разряд экономически в данный час."""
        if not require_positive_margin:
            return True
        # Снижение платы за мощность взимается по максимуму в пиковой зоне и
        # многократно перекрывает убыток от одного цикла, поэтому там разряд
        # оправдан независимо от арбитражной маржи.
        if zone == "peak" and demand_charge_rub_per_kw_month > 0:
            return True
        return cycle_margin_rub_per_kwh(
            discharge_price, tariff_night, one_way_eff, cycle_cost_per_kwh) > 0

    for i in range(n):
        planned = float(forecast[i])       # что видит контроллер
        real = float(actual_arr[i])        # что произошло на самом деле
        price = float(prices[i])
        zone = zones[i]

        charge_cmd = 0.0
        discharge_cmd = 0.0

        if policy == "tariff":
            if zone == "night" and soc < soc_max_kwh - 0.1:
                charge_cmd = min(max_power, soc_max_kwh - soc)
            elif (zone == "peak" and soc > soc_min_kwh + 0.1
                  and _discharge_is_worthwhile(price, zone)):
                discharge_cmd = min(max_power, soc - soc_min_kwh)
        else:
            # Прогноз-зависимая срезка пика: разряжаем ровно столько, сколько
            # нужно, чтобы опустить ПРОГНОЗНУЮ нагрузку до порога.
            if zone == "night" and soc < soc_max_kwh - 0.1:
                charge_cmd = min(max_power, soc_max_kwh - soc)
            elif zone in ("peak", "day") and soc > soc_min_kwh + 0.1:
                excess = planned - float(thresholds[i])
                if excess > 0 and _discharge_is_worthwhile(price, zone):
                    deliverable = min(max_power, soc - soc_min_kwh) * one_way_eff
                    delivered_target = min(excess, deliverable)
                    discharge_cmd = delivered_target / one_way_eff

        if charge_cmd > 0:
            grid_draw = charge_cmd / one_way_eff
            soc += charge_cmd
            grid_energy = real + grid_draw
            action = "charge"
            # Через батарею энергия проходит один раз за цикл «заряд-разряд»,
            # поэтому оборот считается только на заряде. Учёт и заряда, и
            # разряда удвоил бы расчётный износ.
            total_cycled += charge_cmd
        elif discharge_cmd > 0:
            can_draw = min(discharge_cmd, soc - soc_min_kwh)
            # Инвертор питает только собственную нагрузку: выдача в сеть не
            # предусмотрена, поэтому отдать больше фактического потребления
            # физически невозможно. Разряд ограничивается спросом, а не
            # списывается «в никуда»: неиспользованная энергия остаётся в
            # батарее. Без этого ограничения энергобаланс не сходится —
            # при завышенном прогнозе часть заряда просто исчезала.
            delivered = min(can_draw * one_way_eff, real)
            drawn_from_batt = delivered / one_way_eff
            soc -= drawn_from_batt
            grid_energy = real - delivered
            action = "discharge" if drawn_from_batt > 1e-9 else "idle"
        else:
            grid_energy = real
            action = "idle"

        soc = float(np.clip(soc, 0.0, capacity))
        battery_levels.append(soc)
        actions.append(action)
        energy_from_grid.append(grid_energy)
        hourly_costs.append(grid_energy * price)

        if zone == "peak":
            grid_peak_hours_baseline.append(real)
            grid_peak_hours_optimized.append(grid_energy)

    # ── Экономика ─────────────────────────────────────────────────────────────
    # База сравнения — фактическая нагрузка без накопителя.
    baseline_cost = float(np.dot(actual_arr, prices))
    optimized_cost = float(sum(hourly_costs))

    horizon_days = max(n / 24.0, 1e-9)
    months_in_horizon = horizon_days / 30.4375

    if grid_peak_hours_baseline:
        baseline_peak_kw  = float(np.max(grid_peak_hours_baseline))
        optimized_peak_kw = float(np.max(grid_peak_hours_optimized))
    else:
        # Нестандартный сценарий: пиковых часов нет в горизонте
        logger.warning(
            "В горизонте %d ч не найдено пиковых биллинговых часов "
            "(zone='peak', 07–10 и 17–21 в будни). "
            "Demand-charge считается по всем часам — возможно некорректно.",
            n
        )
        baseline_peak_kw  = float(np.max(actual_arr))
        optimized_peak_kw = float(np.max(np.asarray(energy_from_grid, dtype=np.float64)))

    demand_charge_savings = max(
        0.0,
        (baseline_peak_kw - optimized_peak_kw) * demand_charge_rub_per_kw_month * months_in_horizon,
    )

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
    logger.info("Стратегия управления: %s | источник прогноза: %s",
                policy, forecast_source or ("факт (oracle)" if actual is None else "модель"))
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
        policy=policy,
        forecast_source=forecast_source or ("oracle" if actual is None else "model"),
        peak_before_kw=baseline_peak_kw,
        peak_after_kw=optimized_peak_kw,
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
    battery_cost_rub: Optional[float] = None,
    tariff_night: float = 4.08,
    tariff_half_peak: float = 7.87,
    tariff_peak: float = 11.24,
    demand_charge_rub_per_kw_month: float = 950.0,
    annual_om_share: float = 0.015,
    start_hour: int = 0,
    start_weekday: int = 0,
    actual: Optional[np.ndarray] = None,
    policy: str = "tariff",
    shave_quantile: float = 0.85,
    forecast_source: str = "",
) -> Dict[str, StorageResult]:
    """
    Три стратегии с разной глубиной разряда (целевым диапазоном SOC).

    Консервативная: SOC 40%→60% — минимальный износ, минимальная экономия
    Умеренная:      SOC 25%→75% — баланс износа и эффекта
    Агрессивная:    SOC  8%→92% — максимальная экономия, ускоренная деградация
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
            actual=actual,
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
            start_hour=start_hour,
            start_weekday=start_weekday,
            policy=policy,
            shave_quantile=shave_quantile,
            forecast_source=forecast_source,
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


# ══════════════════════════════════════════════════════════════════════════════
# ЦЕНА ОШИБКИ ПРОГНОЗА
# ══════════════════════════════════════════════════════════════════════════════

def compare_forecast_sources(
    actual: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    battery_cost_rub: float,
    capacity: float,
    max_power: float,
    round_trip_efficiency: float = 0.95,
    cycle_cost_per_kwh: float = 0.06,
    min_soc: float = 0.25,
    max_soc: float = 0.75,
    tariff_night: float = 4.08,
    tariff_half_peak: float = 7.87,
    tariff_peak: float = 11.24,
    demand_charge_rub_per_kw_month: float = 950.0,
    annual_om_share: float = 0.015,
    start_hour: int = 0,
    start_weekday: int = 0,
    shave_quantile: float = 0.85,
) -> Dict[str, StorageResult]:
    """
    Прогоняет прогноз-зависимую стратегию на разных источниках прогноза
    и измеряет, во сколько обходится ошибка каждой модели.

    Идеальный прогноз (`actual`) задаёт верхнюю границу достижимого эффекта.
    Недобор экономии относительно него — это и есть денежная стоимость ошибки
    прогнозирования, то самое звено, которое связывает первую часть работы
    (прогнозные модели) со второй (оптимизация накопителя).

    Parameters
    ----------
    actual : np.ndarray
        Фактическая нагрузка на горизонте планирования.
    forecasts : dict {название модели: прогноз той же длины}
        Ключ "Идеальный прогноз" добавляется автоматически.

    Returns
    -------
    dict {источник прогноза: StorageResult}
    """
    common = dict(
        capacity=capacity, max_power=max_power,
        round_trip_efficiency=round_trip_efficiency,
        cycle_cost_per_kwh=cycle_cost_per_kwh,
        min_soc=min_soc, max_soc=max_soc,
        battery_cost_rub=battery_cost_rub,
        tariff_night=tariff_night, tariff_half_peak=tariff_half_peak,
        tariff_peak=tariff_peak,
        demand_charge_rub_per_kw_month=demand_charge_rub_per_kw_month,
        annual_om_share=annual_om_share,
        start_hour=start_hour, start_weekday=start_weekday,
        policy="peak_shaving", shave_quantile=shave_quantile,
    )

    sources: Dict[str, np.ndarray] = {"Идеальный прогноз": np.asarray(actual, dtype=np.float64)}
    for name, f in forecasts.items():
        sources[name] = np.asarray(f, dtype=np.float64)

    results: Dict[str, StorageResult] = {}
    for name, f in sources.items():
        results[name] = simulate_storage(
            forecast=f, actual=actual, forecast_source=name,
            strategy_name=f"Peak shaving по прогнозу: {name}", **common,
        )

    # ── Сводная таблица ───────────────────────────────────────────────────────
    oracle = results["Идеальный прогноз"].net_savings
    logger.info("═" * 86)
    logger.info("ЦЕНА ОШИБКИ ПРОГНОЗА (стратегия peak shaving, SOC %.0f%%→%.0f%%)",
                min_soc * 100, max_soc * 100)
    logger.info("─" * 86)
    logger.info("%-24s %10s %16s %14s %12s",
                "Источник прогноза", "MAE", "Чистая экон.", "Недобор", "Реализовано")
    logger.info("─" * 86)
    for name, res in sorted(results.items(), key=lambda kv: -kv[1].net_savings):
        if name == "Идеальный прогноз":
            mae = 0.0
        else:
            mae = float(np.mean(np.abs(np.asarray(actual, dtype=np.float64) - sources[name])))
        shortfall = oracle - res.net_savings
        realized = (res.net_savings / oracle * 100) if oracle > 0 else float("nan")
        logger.info("%-24s %10.1f %14.0f руб %10.0f руб %10.1f%%",
                    name, mae, res.net_savings, shortfall, realized)
    logger.info("═" * 86)
    logger.info(
        "Разница между строками — денежная стоимость ошибки прогноза: "
        "именно она показывает, окупается ли усложнение модели."
    )
    return results