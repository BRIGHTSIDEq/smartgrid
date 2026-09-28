# -*- coding: utf-8 -*-
"""
Базовая линия для управления спросом.

Агрегатор управления спросом (516-ФЗ, ПП № 461) получает оплату за снижение
потребления в часы события. Снижение измеряется относительно базовой линии —
оценки того, сколько объект потребил бы без команды. Ошибка базовой линии
напрямую превращается в деньги: завышенная засчитывает несуществующее
снижение, заниженная съедает реальное.

Методы:

  similar_days_baseline — классический CBL: среднее по N предыдущим
  «похожим» дням (для рабочего дня — рабочие, для выходного — выходные) в те
  же часы, по желанию с внутрисуточной поправкой по часам перед событием.
  Вариант «high» берёт N дней с наибольшим потреблением из M последних.

  forecast_baseline — прогноз модели на сутки вперёд как базовая линия.

evaluate_baselines проверяет точность на днях БЕЗ событий: там факт и есть
истинная базовая линия, поэтому ошибку можно измерить честно.

Точная формула расчёта базовой линии в российской модели управления
спросом задаётся регламентами оптового рынка; перед применением её нужно
сверить с действующей редакцией регламента АО «АТС».
"""

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _day_type(days: pd.DatetimeIndex, holidays: Optional[set]) -> np.ndarray:
    hol = holidays or set()
    return np.array([(d.dayofweek >= 5) or (d.normalize() in hol) for d in days])


def similar_days_baseline(series: pd.Series, day, n_days: int = 10,
                          pool_days: Optional[int] = None, mode: str = "mean",
                          adjust_hours: Optional[Sequence[int]] = None,
                          adjust_cap: float = 0.2,
                          holidays: Optional[Iterable] = None,
                          exclude_days: Iterable = ()) -> pd.Series:
    """
    Базовая линия на сутки day по похожим предыдущим дням.

    Parameters
    ----------
    series : почасовой ряд с DatetimeIndex.
    n_days : сколько дней усредняется.
    pool_days : из скольких последних похожих дней выбирать (для mode="high").
    mode : "mean" — последние n_days похожих дней; "high" — n_days с наибольшим
        суммарным потреблением из pool_days последних.
    adjust_hours : часы суток, по которым считается внутрисуточная поправка
        (отношение факта к базовой линии в эти часы, ограниченное ±adjust_cap).
        Часы должны предшествовать событию: иначе поправка «увидит» снижение.
    exclude_days : дни прошлых событий — они не отражают обычное потребление.
    """
    s = series.asfreq("h")
    day = pd.Timestamp(day).normalize()
    hol = {pd.Timestamp(h).normalize() for h in (holidays or [])}
    excl = {pd.Timestamp(d).normalize() for d in exclude_days}
    target_type = _day_type(pd.DatetimeIndex([day]), hol)[0]

    candidates = []
    d = day - pd.Timedelta(days=1)
    limit = pool_days or n_days
    first = s.index[0].normalize()
    while len(candidates) < limit and d >= first:
        chunk = s.loc[d:d + pd.Timedelta(hours=23)]
        if (len(chunk) == 24 and not chunk.isna().any() and d not in excl
                and _day_type(pd.DatetimeIndex([d]), hol)[0] == target_type):
            candidates.append(chunk.to_numpy())
        d -= pd.Timedelta(days=1)
    if len(candidates) < min(n_days, 3):
        raise ValueError(f"Недостаточно похожих дней до {day.date()}: {len(candidates)}")

    days = np.array(candidates)
    if mode == "high":
        days = days[np.argsort(days.sum(axis=1))[::-1][:n_days]]
    elif mode == "mean":
        days = days[:n_days]
    else:
        raise ValueError(f"Неизвестный режим {mode!r}")
    base = days.mean(axis=0)

    if adjust_hours:
        actual = s.loc[day:day + pd.Timedelta(hours=23)].to_numpy()
        idx = list(adjust_hours)
        if len(actual) == 24 and base[idx].sum() > 0:
            ratio = actual[idx].sum() / base[idx].sum()
            base = base * float(np.clip(ratio, 1 - adjust_cap, 1 + adjust_cap))
    return pd.Series(base, index=pd.date_range(day, periods=24, freq="h"))


def forecast_baseline(forecast: pd.Series, day) -> pd.Series:
    """Прогноз на сутки вперёд как базовая линия для суток day."""
    day = pd.Timestamp(day).normalize()
    return forecast.loc[day:day + pd.Timedelta(hours=23)]


def evaluate_baselines(series: pd.Series, days: Sequence, event_hours: Sequence[int],
                       forecast: Optional[pd.Series] = None,
                       holidays: Optional[Iterable] = None) -> pd.DataFrame:
    """
    Точность базовых линий в часы условного события на днях без событий.

    Для каждого метода: средняя абсолютная ошибка в часы события в процентах
    от факта и смещение (плюс — завышение, то есть засчитанное снижение,
    которого не было).
    """
    methods = {
        "10 похожих дней": dict(n_days=10, mode="mean"),
        "10 похожих дней + поправка": dict(
            n_days=10, mode="mean",
            adjust_hours=[h for h in range(min(event_hours) - 4, min(event_hours) - 1) if h >= 0]),
        "5 максимальных из 10": dict(n_days=5, pool_days=10, mode="high"),
    }
    s = series.asfreq("h")
    rows = []
    for day in days:
        day = pd.Timestamp(day).normalize()
        actual = s.loc[day:day + pd.Timedelta(hours=23)].to_numpy()[list(event_hours)]
        estimates: Dict[str, np.ndarray] = {}
        for name, kw in methods.items():
            try:
                estimates[name] = similar_days_baseline(
                    s, day, holidays=holidays, **kw).to_numpy()[list(event_hours)]
            except ValueError:
                continue
        if forecast is not None:
            fb = forecast_baseline(forecast, day)
            if len(fb) == 24:
                estimates["Прогноз модели"] = fb.to_numpy()[list(event_hours)]
        for name, est in estimates.items():
            rows.append({"day": day, "method": name,
                         "abs_pct": float(np.mean(np.abs(est - actual) / actual) * 100),
                         "bias_pct": float(np.mean((est - actual) / actual) * 100)})
    table = pd.DataFrame(rows)
    return (table.groupby("method")[["abs_pct", "bias_pct"]].mean()
                 .sort_values("abs_pct").reset_index())
