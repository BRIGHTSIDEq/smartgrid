# -*- coding: utf-8 -*-
"""
Счёт потребителя по ценовым категориям 3–6 розничного рынка электроэнергии.

Модель накопителя в optimization/storage.py считает экономику по трёхзонному
тарифу с платой за мощность по максимуму месяца. Для потребителей с
максимальной мощностью от 670 кВт это не так: они работают в 3–6 ценовых
категориях с почасовым учётом, и мощность там оплачивается иначе.

  Генерирующая мощность (категории 3–6). Объём — среднее по рабочим дням
  месяца потребления в час пиковой нагрузки: для каждых рабочих суток это час
  из плановых часов СО, в который достигнут максимум потребления субъекта РФ.
  Час становится известен после окончания месяца.

  Сетевая мощность (двухставочный тариф, категории 4 и 6). Объём — среднее по
  рабочим дням месяца суточных максимумов потребителя в плановые часы пиковой
  нагрузки, которые СО ЕЭС публикует на год вперёд.

  План и отклонения (категории 5 и 6). Потребитель подаёт почасовой план и
  оплачивает отклонения в обе стороны.

Главное следствие для накопителя: вместо одного события в месяц — около
двадцати независимых (по числу рабочих дней), и каждое оплачивается долей
ставки мощности.

ВАЖНО. Ставки и окна часов по умолчанию — порядок величин для примера, а не
действующие тарифы. Перед использованием в коммерческом предложении их нужно
заменить на тарифы региона и сетевой организации и на плановые часы СО для
своей ценовой зоны, а формулы сверить с Основными положениями (ПП № 442).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Плановые часы пиковой нагрузки по месяцам: [начало, конец) в часах суток.
# Пример, а не данные СО: одно широкое окно на все месяцы. Широкое окно
# консервативно — срезать максимум в нём труднее, чем в узком.
DEFAULT_PEAK_WINDOWS: Dict[int, Tuple[int, int]] = {m: (8, 21) for m in range(1, 13)}


@dataclass(frozen=True)
class RuTariff:
    """
    Параметры счёта по ценовой категории.

    Ставки мощности — в рублях за кВт в месяц, энергетические — в рублях за
    кВт·ч. energy_price может быть и почасовым рядом, согласованным с
    отметками времени нагрузки.
    """

    category: int = 4
    energy_price: object = 3.20
    gen_capacity_rate: float = 1050.0
    net_single_rate: float = 2.60
    net_capacity_rate: float = 1300.0
    net_energy_rate: float = 0.35
    deviation_up_rate: float = 0.25
    deviation_down_rate: float = 0.15
    peak_windows: Dict[int, Tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_PEAK_WINDOWS))

    def __post_init__(self):
        if self.category not in (3, 4, 5, 6):
            raise ValueError(f"Поддерживаются категории 3–6, задана {self.category}")

    @property
    def two_rate_network(self) -> bool:
        """Двухставочный тариф на передачу: категории 4 и 6."""
        return self.category in (4, 6)

    @property
    def with_plan(self) -> bool:
        """Почасовое планирование и оплата отклонений: категории 5 и 6."""
        return self.category in (5, 6)

    def capacity_rate_per_event(self) -> float:
        """
        Ставка, которую потребитель платит за кВт суточного максимума.

        Обе мощности считаются как среднее по рабочим дням, поэтому каждый
        рабочий день несёт 1/n_рд месячной ставки. Здесь возвращается месячная
        сумма ставок; деление на число рабочих дней делается в расчёте.
        """
        rate = self.gen_capacity_rate
        if self.two_rate_network:
            rate += self.net_capacity_rate
        return rate


def _as_hourly(values, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if len(arr) != n:
        raise ValueError(f"{name}: {len(arr)} значений на {n} часов")
    return arr


def calendar_frame(timestamps, tariff: RuTariff,
                   holidays: Optional[Sequence[bool]] = None) -> pd.DataFrame:
    """
    Календарь расчёта: месяц, сутки, рабочий ли день, попадает ли час в окно СО.

    Рабочий день — будний и не праздничный. Праздники по умолчанию берутся из
    того же календаря, что у генератора и тарифных зон.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if holidays is None:
        from data.generator import holiday_flags
        holidays = holiday_flags(ts)
    holidays = np.asarray(holidays, dtype=bool)
    if len(holidays) != len(ts):
        raise ValueError("Маска праздников не совпадает по длине с рядом")

    lo = np.array([tariff.peak_windows[m][0] for m in ts.month])
    hi = np.array([tariff.peak_windows[m][1] for m in ts.month])
    return pd.DataFrame({
        "timestamp": ts,
        "month": ts.to_period("M").astype(str),
        "day": ts.normalize(),
        "working": (ts.dayofweek < 5) & ~holidays,
        "in_window": (ts.hour >= lo) & (ts.hour < hi),
    })


def peak_hours_of_region(region_load: np.ndarray, cal: pd.DataFrame) -> np.ndarray:
    """
    Маска часов пиковой нагрузки субъекта: по одному часу на рабочие сутки.

    Час — максимум потребления субъекта внутри окна СО. Если ряда субъекта нет,
    вызывающий код передаёт нагрузку самого потребителя: для крупного городского
    агрегата это разумное приближение, для отдельного объекта — нет, и это
    нужно оговаривать.
    """
    mask = np.zeros(len(cal), dtype=bool)
    frame = cal.assign(load=np.asarray(region_load, dtype=np.float64),
                       pos=np.arange(len(cal)))
    eligible = frame[frame["working"] & frame["in_window"]]
    if len(eligible):
        idx = eligible.loc[eligible.groupby("day")["load"].idxmax(), "pos"].to_numpy()
        mask[idx] = True
    return mask


def monthly_bill(load: np.ndarray, timestamps, tariff: RuTariff,
                 plan: Optional[np.ndarray] = None,
                 region_load: Optional[np.ndarray] = None,
                 holidays: Optional[Sequence[bool]] = None) -> pd.DataFrame:
    """
    Счёт по месяцам и компонентам.

    Parameters
    ----------
    load : почасовое потребление из сети, кВт·ч (= средняя мощность за час, кВт).
    plan : почасовой план для категорий 5 и 6. Без него отклонения не считаются.
    region_load : ряд субъекта для выбора часа пиковой нагрузки. По умолчанию —
        сам load. Для оценки накопителя сюда передаётся нагрузка БЕЗ
        накопителя: один потребитель не сдвигает час пика субъекта.

    Неполный месяц оплачивается пропорционально числу покрытых суток: иначе
    месяц из трёх дней весил бы в итоге как полный.
    """
    load = np.asarray(load, dtype=np.float64)
    n = len(load)
    cal = calendar_frame(timestamps, tariff, holidays)
    if len(cal) != n:
        raise ValueError("Длина ряда и отметок времени различается")
    price = _as_hourly(tariff.energy_price, n, "energy_price")
    peak_mask = peak_hours_of_region(load if region_load is None else region_load, cal)

    frame = cal.assign(load=load, price=price, peak=peak_mask)
    rows = []
    for month, g in frame.groupby("month", sort=True):
        days_covered = g["day"].nunique()
        period = pd.Period(month)
        share = days_covered / period.days_in_month
        work = g[g["working"]]
        n_work_days = work["day"].nunique()

        gen_kw = float(work.loc[work["peak"], "load"].mean()) if n_work_days else 0.0
        net_kw = (float(work[work["in_window"]].groupby("day")["load"].max().mean())
                  if n_work_days and tariff.two_rate_network else 0.0)

        energy_kwh = float(g["load"].sum())
        energy_cost = float((g["load"] * g["price"]).sum())
        net_energy = energy_kwh * (tariff.net_energy_rate if tariff.two_rate_network
                                   else tariff.net_single_rate)
        row = {
            "month": month, "days": days_covered, "working_days": n_work_days,
            "energy_kwh": energy_kwh, "energy_cost": energy_cost,
            "gen_capacity_kw": gen_kw,
            "gen_capacity_cost": gen_kw * tariff.gen_capacity_rate * share,
            "net_capacity_kw": net_kw,
            "net_capacity_cost": net_kw * tariff.net_capacity_rate * share,
            "net_energy_cost": net_energy,
            "deviation_cost": 0.0,
        }
        if tariff.with_plan and plan is not None:
            p = np.asarray(plan, dtype=np.float64)[g.index.to_numpy()]
            diff = g["load"].to_numpy() - p
            row["deviation_cost"] = float(
                tariff.deviation_up_rate * np.clip(diff, 0, None).sum()
                + tariff.deviation_down_rate * np.clip(-diff, 0, None).sum())
        row["total"] = (row["energy_cost"] + row["gen_capacity_cost"]
                        + row["net_capacity_cost"] + row["net_energy_cost"]
                        + row["deviation_cost"])
        rows.append(row)
    return pd.DataFrame(rows)


def daily_capacity_weights(timestamps, tariff: RuTariff,
                           holidays: Optional[Sequence[bool]] = None) -> pd.Series:
    """
    Цена одного кВт суточного максимума в окне СО для каждых суток.

    Среднее суточных максимумов по рабочим дням линейно по каждому дню: день
    несёт ставку, делённую на число рабочих дней месяца, с поправкой на
    неполный месяц. Этим пользуются оптимизатор (суточная задача) и бутстреп
    по суткам. Нерабочим дням соответствует ноль.
    """
    cal = calendar_frame(timestamps, tariff, holidays)
    weights = {}
    for month, g in cal.groupby("month", sort=True):
        share = g["day"].nunique() / pd.Period(month).days_in_month
        work_days = g.loc[g["working"], "day"].unique()
        per_day = tariff.capacity_rate_per_event() * share / max(len(work_days), 1)
        for day in g["day"].unique():
            weights[day] = per_day if day in set(work_days) else 0.0
    return pd.Series(weights).sort_index()
