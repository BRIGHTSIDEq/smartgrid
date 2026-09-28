# -*- coding: utf-8 -*-
"""
Управление накопителем при оплате мощности по ценовым категориям 3–6.

Пороговое правило из optimization/storage.py настроено на одно событие в
месяц — максимум месяца. В категориях 3–6 мощность оплачивается как среднее
по рабочим дням (см. optimization/tariffs_ru.py), то есть около двадцати
событий в месяц. Здесь два контроллера для этой постановки:

  mpc_day_ahead — каждые сутки решается линейная задача по ПРОГНОЗУ на эти
  сутки, а исполняется она на ФАКТЕ. Час пика субъекта заранее неизвестен,
  поэтому оптимизатор снижает суточный максимум в окне СО целиком: час пика
  всегда лежит внутри окна, и его потребление не больше суточного максимума.

  perfect_foresight_ru — верхняя граница: одна задача на весь период по
  фактической нагрузке. Час пика субъекта задним числом известен, поэтому
  целевая функция совпадает со счётом точно.

Физика накопителя та же, что в storage.simulate_storage и
storage.perfect_foresight_optimum: КПД √η в каждую сторону, отдача в сеть
запрещена, износ — по заряженной энергии.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from optimization.tariffs_ru import (
    RuTariff, calendar_frame, daily_capacity_weights, monthly_bill, peak_hours_of_region,
)


@dataclass(frozen=True)
class Battery:
    capacity: float                     # кВт·ч
    max_power: float                    # кВт
    capex_rub: float = 0.0
    round_trip_efficiency: float = 0.95
    min_soc: float = 0.10
    max_soc: float = 0.90
    initial_soc: float = 0.50
    cycle_cost_per_kwh: float = 0.06
    annual_om_share: float = 0.015

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.round_trip_efficiency))


def _marginal_energy_price(tariff: RuTariff, n: int) -> np.ndarray:
    """Цена кВт·ч из сети: энергия плюс энергетическая часть передачи."""
    energy = np.asarray(tariff.energy_price, dtype=np.float64)
    energy = np.full(n, float(energy)) if energy.ndim == 0 else energy.astype(np.float64)
    network = tariff.net_energy_rate if tariff.two_rate_network else tariff.net_single_rate
    return energy + network


def _solve_lp(load: np.ndarray, price: np.ndarray, battery: Battery, soc0: float,
              groups: Sequence[np.ndarray], group_weights: Sequence[float],
              point_weights: Optional[np.ndarray] = None,
              terminal_soc: Optional[float] = None):
    """
    Линейная задача расписания на отрезке.

    Переменные: заряд c (кВт·ч в батарею), отдача d (кВт·ч потребителю), запас
    s после часа и по одной переменной M_k на каждую группу часов — максимум
    сетевой нагрузки в этой группе (суточный максимум в окне СО).
    point_weights — цена сетевой нагрузки в отдельных часах (час пика субъекта).
    """
    from scipy import sparse
    from scipy.optimize import linprog

    n = len(load)
    eta = battery.eta
    k = len(groups)
    nc, nd, ns, nm = 0, n, 2 * n, 3 * n
    nvar = 3 * n + k
    point = np.zeros(n) if point_weights is None else np.asarray(point_weights, float)

    cost = np.zeros(nvar)
    cost[nc:nc + n] = (price + point) / eta + battery.cycle_cost_per_kwh
    cost[nd:nd + n] = -(price + point)
    cost[nm:nm + k] = np.asarray(group_weights, dtype=np.float64)

    rows, cols, vals = [], [], []
    for i in range(n):
        rows += [i, i, i]
        cols += [ns + i, nc + i, nd + i]
        vals += [1.0, -1.0, 1.0 / eta]
        if i > 0:
            rows.append(i)
            cols.append(ns + i - 1)
            vals.append(-1.0)
    a_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(n, nvar))
    b_eq = np.zeros(n)
    b_eq[0] = soc0

    rows, cols, vals, b_ub = [], [], [], []
    r = 0
    for g, hours in enumerate(groups):
        for i in hours:
            rows += [r, r, r]
            cols += [nc + i, nd + i, nm + g]
            vals += [1.0 / eta, -1.0, -1.0]
            b_ub.append(-load[i])
            r += 1
    if terminal_soc is not None:
        rows.append(r)
        cols.append(ns + n - 1)
        vals.append(-1.0)
        b_ub.append(-terminal_soc)
        r += 1
    a_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nvar)) if r else None

    cap = battery.capacity
    bounds = ([(0.0, battery.max_power)] * n
              + [(0.0, float(min(max(x, 0.0), battery.max_power * eta))) for x in load]
              + [(battery.min_soc * cap, battery.max_soc * cap)] * n
              + [(0.0, None)] * k)
    sol = linprog(cost, A_ub=a_ub, b_ub=np.asarray(b_ub) if r else None,
                  A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not sol.success:
        raise RuntimeError(f"Задача расписания не решена: {sol.message}")
    return sol.x[nc:nc + n], sol.x[nd:nd + n], sol.x[ns:ns + n]


def execute_on_actual(actual: np.ndarray, charge_plan: np.ndarray,
                      discharge_plan: np.ndarray, battery: Battery, soc0: float):
    """
    Исполняет расписание на фактической нагрузке.

    План построен по прогнозу, поэтому на факте его ограничивают физика и
    запрет отдачи в сеть: разряд не больше фактической нагрузки и доступного
    запаса, заряд не больше свободной ёмкости.
    """
    eta = battery.eta
    lo, hi = battery.min_soc * battery.capacity, battery.max_soc * battery.capacity
    soc = soc0
    grid = np.empty(len(actual))
    charged = np.empty(len(actual))
    socs = np.empty(len(actual))
    for i, a in enumerate(actual):
        d = min(discharge_plan[i], max(a, 0.0), (soc - lo) * eta, battery.max_power * eta)
        d = max(d, 0.0)
        soc -= d / eta
        c = max(min(charge_plan[i], hi - soc, battery.max_power), 0.0)
        soc += c
        grid[i] = a + c / eta - d
        charged[i] = c
        socs[i] = soc
    return grid, charged, socs


def mpc_day_ahead(forecast: np.ndarray, actual: np.ndarray, timestamps,
                  tariff: RuTariff, battery: Battery,
                  holidays: Optional[Sequence[bool]] = None) -> Dict[str, np.ndarray]:
    """
    Суточное управление по прогнозу.

    Каждые сутки: задача на 24 ч по прогнозу с ценой суточного максимума в окне
    СО и условием вернуть запас к началу суток (иначе контроллер «проедал» бы
    начальный заряд и выглядел лучше, чем есть). Затем план исполняется на
    факте, и фактический запас переходит в следующие сутки.
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    cal = calendar_frame(timestamps, tariff, holidays)
    weights = daily_capacity_weights(timestamps, tariff, holidays)
    price = _marginal_energy_price(tariff, len(actual))

    grid = np.empty_like(actual)
    charged = np.empty_like(actual)
    soc_path = np.empty_like(actual)
    soc = battery.initial_soc * battery.capacity
    for day, g in cal.groupby("day", sort=True):
        idx = g.index.to_numpy()
        window = np.where(g["in_window"].to_numpy())[0]
        w = float(weights.get(day, 0.0))
        groups = [window] if (w > 0 and len(window)) else []
        c, d, _ = _solve_lp(forecast[idx], price[idx], battery, soc, groups,
                            [w] if groups else [], terminal_soc=soc)
        grid[idx], charged[idx], soc_path[idx] = execute_on_actual(
            actual[idx], c, d, battery, soc)
        soc = soc_path[idx][-1]
    return {"grid": grid, "charged": charged, "soc": soc_path}


def perfect_foresight_ru(actual: np.ndarray, timestamps, tariff: RuTariff,
                         battery: Battery,
                         holidays: Optional[Sequence[bool]] = None) -> Dict[str, np.ndarray]:
    """
    Верхняя граница экономии: расписание на весь период по известному факту.

    Генерирующая мощность оплачивается по часам пика субъекта; при известном
    будущем они известны (субъект — нагрузка без накопителя), и в целевой
    функции стоит ровно та ставка, что и в счёте. Сетевая мощность — через
    суточные максимумы в окне СО.
    """
    actual = np.asarray(actual, dtype=np.float64)
    cal = calendar_frame(timestamps, tariff, holidays)
    price = _marginal_energy_price(tariff, len(actual))
    peaks = peak_hours_of_region(actual, cal)

    point = np.zeros(len(actual))
    groups, group_w = [], []
    for month, g in cal.groupby("month", sort=True):
        share = g["day"].nunique() / pd.Period(month).days_in_month
        work_days = g.loc[g["working"], "day"].unique()
        if not len(work_days):
            continue
        per_day_gen = tariff.gen_capacity_rate * share / len(work_days)
        per_day_net = tariff.net_capacity_rate * share / len(work_days)
        month_idx = g.index.to_numpy()
        point[month_idx[peaks[month_idx]]] = per_day_gen
        if tariff.two_rate_network:
            for day in work_days:
                hours = g.index[(g["day"] == day) & g["in_window"]].to_numpy()
                if len(hours):
                    groups.append(hours)
                    group_w.append(per_day_net)

    c, d, s = _solve_lp(actual, price, battery, battery.initial_soc * battery.capacity,
                        groups, group_w, point_weights=point)
    eta = battery.eta
    return {"grid": actual + c / eta - d, "charged": c, "soc": s}


def evaluate_schedule(grid: np.ndarray, charged: np.ndarray, baseline: np.ndarray,
                      timestamps, tariff: RuTariff, battery: Battery,
                      plan: Optional[np.ndarray] = None,
                      holidays: Optional[Sequence[bool]] = None) -> Dict[str, float]:
    """
    Экономия по счёту: без накопителя минус с накопителем, минус износ и O&M.

    Час пика субъекта в обоих счетах определяется по нагрузке без накопителя:
    один потребитель не сдвигает пик региона.
    """
    base_bill = monthly_bill(baseline, timestamps, tariff, plan=plan,
                             region_load=baseline, holidays=holidays)
    batt_bill = monthly_bill(grid, timestamps, tariff, plan=plan,
                             region_load=baseline, holidays=holidays)
    n_hours = len(grid)
    degradation = float(np.sum(charged)) * battery.cycle_cost_per_kwh
    om = battery.capex_rub * battery.annual_om_share * n_hours / 8760.0
    gross = float(base_bill["total"].sum() - batt_bill["total"].sum())
    net = gross - degradation - om
    annual = net * 8760.0 / n_hours if n_hours else 0.0
    parts = {}
    for col in ("energy_cost", "gen_capacity_cost", "net_capacity_cost",
                "net_energy_cost", "deviation_cost"):
        parts[f"saved_{col}"] = float(base_bill[col].sum() - batt_bill[col].sum())
    return {
        "baseline_bill": float(base_bill["total"].sum()),
        "bill_with_battery": float(batt_bill["total"].sum()),
        "gross_savings": gross, "degradation": degradation, "om_cost": om,
        "net_savings": net, "annual_net_savings": annual,
        "payback_years": (battery.capex_rub / annual) if annual > 0 else float("inf"),
        **parts,
    }
