# -*- coding: utf-8 -*-
"""
Калькулятор ценности прогноза в рублях по истории одного клиента.

Вопрос заказчика не «какая MAE», а «сколько я заработаю, если прогноз станет
точнее». Калькулятор отвечает на него без обучения моделей: на последних
eval_days сутках истории он сравнивает простые прогнозы на сутки вперёд с
идеальным и переводит разницу в рубли по счёту выбранной ценовой категории:

  * отклонения от почасового плана (категории 5 и 6), если план — прогноз;
  * экономия накопителя под суточным MPC по этому прогнозу.

Простые прогнозы — то, что клиент может сделать сам: «как вчера», «как неделю
назад», средний профиль часа недели. Если модель проекта уже дала прогноз
(forecast.py), его можно добавить колонкой и увидеть, сколько он приносит
сверх этих вариантов. Разница между лучшим простым и идеальным прогнозом —
потолок того, что вообще можно заработать на точности.

    python -m analysis.value_calculator history.csv --series M1 --category 6
"""

import argparse
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

from analysis.economics import battery_for_load
from optimization.controllers import evaluate_schedule, mpc_day_ahead
from optimization.tariffs_ru import RuTariff

PERFECT = "Идеальный прогноз"


def simple_forecasts(series: pd.Series, eval_days: int) -> Dict[str, pd.Series]:
    """
    Прогнозы на сутки вперёд для последних eval_days суток.

    Профиль часа недели считается только по истории ДО оцениваемого отрезка,
    иначе он знал бы будущее.
    """
    s = series.asfreq("h")
    start = s.index[-1].normalize() - pd.Timedelta(days=eval_days - 1)
    target = s.loc[start:]
    if len(target) < 24:
        raise ValueError("Для оценки нужна хотя бы одна полная последняя сутки")
    past = s.loc[:start - pd.Timedelta(hours=1)]
    if len(past) < 14 * 24:
        raise ValueError("До оцениваемого отрезка нужно не меньше двух недель истории")

    how = past.groupby([past.index.dayofweek, past.index.hour]).mean()
    profile = pd.Series([how.get((t.dayofweek, t.hour), np.nan) for t in target.index],
                        index=target.index)
    return {
        PERFECT: target,
        "Как вчера": s.shift(24).loc[start:],
        "Как неделю назад": s.shift(168).loc[start:],
        "Профиль часа недели": profile,
    }


def forecast_value(series: pd.Series, tariff: RuTariff, eval_days: int = 60,
                   extra: Optional[Dict[str, pd.Series]] = None,
                   holidays=None) -> pd.DataFrame:
    """
    Таблица: источник прогноза → MAE, стоимость отклонений, экономия накопителя,
    итоговая ценность и её разница с «как вчера».
    """
    forecasts = simple_forecasts(series, eval_days)
    for name, fc in (extra or {}).items():
        forecasts[name] = fc.reindex(forecasts[PERFECT].index)
    actual = forecasts[PERFECT].to_numpy(dtype=np.float64)
    ts = forecasts[PERFECT].index
    battery = battery_for_load(float(np.nanmax(actual)))

    rows = []
    for name, fc in forecasts.items():
        f = fc.to_numpy(dtype=np.float64)
        if np.isnan(f).any():
            f = np.where(np.isnan(f), actual, f)       # нечем прогнозировать — факт
        dev = 0.0
        if tariff.with_plan:
            diff = actual - f
            dev = float(tariff.deviation_up_rate * np.clip(diff, 0, None).sum()
                        + tariff.deviation_down_rate * np.clip(-diff, 0, None).sum())
        sched = mpc_day_ahead(f, actual, ts, tariff, battery, holidays)
        res = evaluate_schedule(sched["grid"], sched["charged"], actual, ts, tariff,
                                battery, holidays=holidays)
        value = res["net_savings"] - dev
        rows.append({"source": name, "MAE": float(np.mean(np.abs(f - actual))),
                     "deviation_cost": dev, "battery_net_savings": res["net_savings"],
                     "value": value, "annual_value": value * 365.0 / eval_days})
    table = pd.DataFrame(rows)
    base = table.loc[table["source"] == "Как вчера", "annual_value"].iloc[0]
    table["annual_gain_vs_yesterday"] = table["annual_value"] - base
    return table.sort_values("annual_value", ascending=False).reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ценность прогноза в рублях")
    parser.add_argument("history", help="CSV: timestamp, consumption[, series]")
    parser.add_argument("--series", default=None)
    parser.add_argument("--category", type=int, default=6, choices=[3, 4, 5, 6])
    parser.add_argument("--eval-days", type=int, default=60)
    parser.add_argument("--model-forecast", default=None,
                        help="CSV прогноза модели: timestamp, forecast[, series]")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.history, parse_dates=["timestamp"])
    if args.series is not None:
        frame = frame[frame["series"].astype(str) == args.series]
    series = frame.set_index("timestamp")["consumption"].sort_index()
    extra = None
    if args.model_forecast:
        fc = pd.read_csv(args.model_forecast, parse_dates=["timestamp"])
        if args.series is not None and "series" in fc.columns:
            fc = fc[fc["series"].astype(str) == args.series]
        extra = {"Модель проекта": fc.set_index("timestamp")["forecast"]}
    table = forecast_value(series, RuTariff(category=args.category), args.eval_days, extra)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
