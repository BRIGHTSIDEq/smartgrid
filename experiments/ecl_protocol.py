# -*- coding: utf-8 -*-
"""
Протокол ECL: сравнение с опубликованными результатами на том же наборе.

ECL (Electricity Consuming Load) — стандартный бенчмарк долгосрочного
прогноза: почасовое потребление клиентов UCI ElectricityLoadDiagrams за
2012–2014 годы. В статьях Informer, Autoformer, DLinear, PatchTST он
оценивается одинаково:

  * хронологическое разбиение 70/10/20, окна валидации и теста начинаются
    за длину входа до границы (вход может заходить в предыдущий отрезок);
  * нормировка z-оценкой по каждому клиенту на обучающем отрезке;
  * вход L часов (96 у Informer и Autoformer, 336 у DLinear, 336 или 512 у
    PatchTST), горизонты 96, 192, 336, 720;
  * MSE и MAE в нормированном масштабе, усреднённые по всем окнам теста с
    шагом 1, клиентам и шагам горизонта.

Протокол отличается от основного в проекте (сутки вперёд, MASE, исходный
масштаб), зато даёт прямое сравнение с DeepAR, TFT и PatchTST без
пересказа чужих настроек.

Модели здесь — те, что решаются в замкнутой форме и потому проверяются
быстро: повтор последнего значения и последних суток, канально-независимая
линейная модель (Linear из работы Zeng et al.) и DLinear (разложение на
тренд скользящим средним и остаток, по линейному слою на каждую часть).
Обе линейные модели обучаются МНК с малой гребневой регуляризацией через
нормальные уравнения, накопленные по клиентам: полная матрица признаков
всех окон всех клиентов не поместилась бы в память.

Число клиентов может отличаться от 321 из статей: там взят заранее
очищенный файл. Отбор здесь — клиенты без нулей в начале 2012 года
(подключённые к началу периода); итоговое число пишется в отчёт.

    python -m experiments.ecl_protocol --uci-path data/raw/LD2011_2014.txt
"""

import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

HORIZONS = (96, 192, 336, 720)

# Опубликованные значения MSE / MAE на ECL для сравнения. Взяты из таблиц
# статей (Zeng et al., AAAI 2023 — DLinear при L=336; Nie et al., ICLR 2023 —
# PatchTST/64 при L=512). Перед цитированием в работе сверить с оригиналом.
PUBLISHED = {
    ("DLinear (Zeng et al., L=336)", 96): (0.140, 0.237),
    ("DLinear (Zeng et al., L=336)", 192): (0.153, 0.249),
    ("DLinear (Zeng et al., L=336)", 336): (0.169, 0.267),
    ("DLinear (Zeng et al., L=336)", 720): (0.203, 0.301),
    ("PatchTST/64 (Nie et al., L=512)", 96): (0.129, 0.222),
    ("PatchTST/64 (Nie et al., L=512)", 192): (0.147, 0.240),
    ("PatchTST/64 (Nie et al., L=512)", 336): (0.163, 0.259),
    ("PatchTST/64 (Nie et al., L=512)", 720): (0.197, 0.290),
}


def load_ecl(uci_path: str, start: str = "2012-01-01", end: str = "2014-12-31 23:00",
             max_clients: Optional[int] = None) -> pd.DataFrame:
    """Почасовая широкая таблица клиентов, подключённых к началу периода."""
    from data.uci import quarter_hours_to_hourly, read_uci_raw

    hourly = quarter_hours_to_hourly(read_uci_raw(uci_path)).loc[start:end]
    first_day = hourly.iloc[:24]
    active = [c for c in hourly.columns
              if (first_day[c] > 0).all() and hourly[c].notna().all()]
    frame = hourly[active]
    if max_clients:
        frame = frame.iloc[:, :max_clients]
    return frame


def split_bounds(n: int, lookback: int):
    """Границы отрезков по соглашению Autoformer: 70/10/20 с заходом на вход."""
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    return {"train": (0, n_train),
            "val": (n_train - lookback, n_train + n_val),
            "test": (n - n_test - lookback, n)}


def normalise(values: np.ndarray, bounds) -> tuple:
    """z-оценка по каждому клиенту со статистиками ТОЛЬКО обучающего отрезка."""
    lo, hi = bounds["train"]
    mean = values[lo:hi].mean(axis=0)
    std = values[lo:hi].std(axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std, mean, std


def _windows(x: np.ndarray, lookback: int, horizon: int):
    n = len(x) - lookback - horizon + 1
    if n <= 0:
        return None, None
    idx = np.arange(n)[:, None]
    X = x[idx + np.arange(lookback)[None, :]]
    Y = x[idx + lookback + np.arange(horizon)[None, :]]
    return X, Y


def moving_average(X: np.ndarray, kernel: int = 25) -> np.ndarray:
    """Тренд DLinear: скользящее среднее с повтором крайних значений."""
    pad = (kernel - 1) // 2
    padded = np.concatenate([np.repeat(X[:, :1], pad, axis=1), X,
                             np.repeat(X[:, -1:], kernel - 1 - pad, axis=1)], axis=1)
    csum = np.cumsum(np.pad(padded, ((0, 0), (1, 0))), axis=1)
    return (csum[:, kernel:] - csum[:, :-kernel]) / kernel


def _features(X: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        return np.concatenate([X, np.ones((len(X), 1))], axis=1)
    trend = moving_average(X)
    return np.concatenate([trend, X - trend, np.ones((len(X), 1))], axis=1)


def fit_linear(series: Iterable[np.ndarray], lookback: int, horizon: int, kind: str,
               ridge: float = 1e-3) -> np.ndarray:
    """
    Канально-независимая линейная модель: одни веса для всех клиентов.

    Нормальные уравнения накапливаются по клиентам, поэтому в памяти
    находится только матрица признаков одного клиента.
    """
    xtx, xty = None, None
    for x in series:
        X, Y = _windows(x, lookback, horizon)
        if X is None:
            continue
        F = _features(X, kind)
        xtx = F.T @ F if xtx is None else xtx + F.T @ F
        xty = F.T @ Y if xty is None else xty + F.T @ Y
    reg = ridge * np.trace(xtx) / len(xtx) * np.eye(len(xtx))
    reg[-1, -1] = 0.0                     # свободный член не штрафуется
    return np.linalg.solve(xtx + reg, xty)


def evaluate(frame: pd.DataFrame, lookback: int = 336,
             horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
    """MSE и MAE по протоколу ECL для каждой модели и горизонта."""
    values = frame.to_numpy(dtype=np.float64)
    n = len(values)
    bounds = split_bounds(n, lookback)
    lo, hi = bounds["train"]
    z, _, _ = normalise(values, bounds)
    train = [z[lo:hi, j] for j in range(z.shape[1])]
    t_lo, t_hi = bounds["test"]
    test = [z[t_lo:t_hi, j] for j in range(z.shape[1])]

    rows = []
    for horizon in horizons:
        weights = {kind: fit_linear(train, lookback, horizon, kind)
                   for kind in ("linear", "dlinear")}
        acc: Dict[str, List[float]] = {}
        for x in test:
            X, Y = _windows(x, lookback, horizon)
            if X is None:
                continue
            preds = {
                "Повтор последнего значения": np.repeat(X[:, -1:], horizon, axis=1),
                "Повтор последних суток": np.tile(X[:, -24:], (1, int(np.ceil(horizon / 24))))[:, :horizon],
                "Linear (канально-независимая)": _features(X, "linear") @ weights["linear"],
                "DLinear": _features(X, "dlinear") @ weights["dlinear"],
            }
            for name, p in preds.items():
                err = p - Y
                a = acc.setdefault(name, [0.0, 0.0, 0])
                a[0] += float((err ** 2).sum())
                a[1] += float(np.abs(err).sum())
                a[2] += err.size
        for name, (se, ae, cnt) in acc.items():
            rows.append({"model": name, "horizon": horizon, "lookback": lookback,
                         "MSE": se / cnt, "MAE": ae / cnt, "clients": z.shape[1]})
    for (name, horizon), (mse, mae) in PUBLISHED.items():
        if horizon in horizons:
            rows.append({"model": name, "horizon": horizon, "lookback": None,
                         "MSE": mse, "MAE": mae, "clients": 321})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Протокол ECL")
    parser.add_argument("--uci-path", default="data/raw/LD2011_2014.txt")
    parser.add_argument("--lookback", type=int, default=336)
    parser.add_argument("--max-clients", type=int, default=None)
    parser.add_argument("--out", default="results/ecl_protocol.csv")
    args = parser.parse_args(argv)

    frame = load_ecl(args.uci_path, max_clients=args.max_clients)
    table = evaluate(frame, lookback=args.lookback)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(table.pivot_table(index="model", columns="horizon", values="MSE").round(3))
    print(f"Клиентов: {frame.shape[1]}. Результаты: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
