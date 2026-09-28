# -*- coding: utf-8 -*-
"""
Пакетный прогноз по сохранённой многорядной модели.

    python forecast.py --models results/runs/<прогон>/models \\
        --history history.csv --weather weather.csv --out forecast.csv \\
        [--model XGBoost] [--target-day 2025-04-11]

history.csv — series, timestamp, consumption, погодные и прочие измеряемые
колонки, на которых обучалась модель, и статические признаки static_*.
weather.csv — прогноз погоды на целевые часы (series необязательна).

С --target-day проверяется, покрывает ли прогноз все часы целевых суток:
модель с горизонтом 24 ч покрывает сутки D, только если история
заканчивается в 23:00 суток D−1.
"""

import argparse
import sys

import pandas as pd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Прогноз по сохранённой панельной модели")
    parser.add_argument("--models", required=True, help="Каталог models/ панельного прогона")
    parser.add_argument("--history", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="Имя модели; по умолчанию лучшая по валидации")
    parser.add_argument("--target-day", default=None)
    parser.add_argument("--quantiles", action="store_true",
                        help="Прогноз с интервалом P10/P50/P90 (по умолчанию квантильный XGBoost)")
    args = parser.parse_args(argv)

    from utils.panel_inference import check_issue_time, forecast_panel, load_panel_models

    bundle = load_panel_models(args.models, args.model, quantile=args.quantiles)
    history = pd.read_csv(args.history, parse_dates=["timestamp"])
    weather = pd.read_csv(args.weather, parse_dates=["timestamp"])
    try:
        result = forecast_panel(bundle, history, weather)
    except ValueError as exc:
        print(f"Прогноз не построен: {exc}", file=sys.stderr)
        return 1
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Модель {bundle['name']}: {result['series'].nunique()} рядов, "
          f"{len(result)} значений → {args.out}")

    if args.target_day:
        worst = 1.0
        for key, g in result.groupby("series"):
            cov = check_issue_time(g, args.target_day)
            worst = min(worst, cov["covered_share"])
            if cov["covered_share"] < 1:
                print(f"  {key}: сутки {cov['target_day']} покрыты на "
                      f"{100 * cov['covered_share']:.0f}%, первый непокрытый час "
                      f"{cov['first_missing']}", file=sys.stderr)
        if worst < 1:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
