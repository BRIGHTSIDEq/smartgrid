# -*- coding: utf-8 -*-
"""
panel_pipeline.py — конвейер многорядного (panel) прогнозирования.

Вынесен из main.py отдельным модулем: структура данных, состав моделей и набор
метрик здесь другие. Смешивать два конвейера в одной функции означало бы
ветвление на каждом шаге и потерю читаемости обоих.

Что делает конвейер:
    генерация панели → валидация → нарезка окон → обучение глобальных моделей
    → оценка (micro, macro, худший ряд) → городской прогноз суммированием
    → экспорт результатов в изолированный каталог прогона.

Обучается ОДНА модель на всех рядах сразу. Именно ради этого panel-режим и
создавался: при десятках фидеров объём обучающих данных на порядок больше, чем
у одного агрегатного ряда, и модель учится общим закономерностям, а не
особенностям одного объекта.
"""

import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger("smart_grid.panel")

# Оценка памяти на одно окно, байт: история + будущее + статика + цель.
# Служит порогом отказа для режимов, требующих потоковой подачи.
_MAX_WINDOWS_IN_MEMORY = 1_500_000


def estimate_windows(n_series: int, days: int, history: int, horizon: int,
                     train_ratio: float = 0.70, stride: int = 1) -> int:
    """
    Оценивает число обучающих окон до генерации данных.

    Шаг прореживания учитывается здесь же: иначе защита по памяти сравнивала бы
    с порогом объём, который никогда не будет материализован, и отвергала бы
    исполнимые режимы.
    """
    hours = days * 24
    per_series = max(int(hours * train_ratio) - history - horizon, 0)
    return n_series * (per_series // max(stride, 1))


def load_panel_dataset(args, logger_obj) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Отдаёт панель либо от собственного генератора, либо из набора UCI.

    Оба источника приводятся к одному формату ДО входа в конвейер, и дальше
    обработка не ветвится. Любое расхождение в подготовке синтетики и реальных
    данных сделало бы их результаты несравнимыми, а сравнение — единственная
    причина подключать внешний набор.

    Размер панели задаёт режим: число рядов и длительность окна берутся из тех
    же PANEL_*, что и для генератора, поэтому объём вычислений сопоставим.
    """
    from data.panel import generate_panel_data

    n_series = Config.PANEL_CITIES * Config.PANEL_FEEDERS_PER_CITY

    if getattr(args, "dataset", "synthetic") == "uci":
        import pandas as pd
        from data.uci import uci_to_panel

        # Окно отсчитывается назад от конца наблюдений: свежие годы полнее по
        # составу клиентов, а привязка к фиксированной дате делает выборку
        # одинаковой при любом запуске.
        end = pd.Timestamp("2014-12-31 23:00")
        start = end - pd.Timedelta(days=Config.PANEL_DAYS) + pd.Timedelta(hours=1)

        df, specs, report = uci_to_panel(
            args.uci_path, start=str(start.date()), end=str(end.date()),
            max_series=n_series, train_ratio=Config.TRAIN_RATIO,
        )
        logger_obj.info("UCI: %d рядов × %d ч, разброс средних %.1f×, "
                        "отсеяно по активности %d",
                        report["рядов отобрано"], report["часов на ряд"],
                        report["разброс средних, раз"], report["рядов отсеяно"])
        return df, specs, report

    df, specs = generate_panel_data(
        days=Config.PANEL_DAYS,
        n_cities=Config.PANEL_CITIES,
        feeders_per_city=Config.PANEL_FEEDERS_PER_CITY,
        start_date=Config.START_DATE,
        seed=args.seed,
        kwh_per_household_month=Config.GEN_KWH_PER_HOUSEHOLD_MONTH,
        temp_annual_mean=Config.GEN_TEMP_ANNUAL_MEAN,
        temp_annual_amplitude=Config.GEN_TEMP_ANNUAL_AMPLITUDE,
        temp_min=Config.GEN_TEMP_MIN, temp_max=Config.GEN_TEMP_MAX,
        temp_setpoint=Config.GEN_TEMP_SETPOINT,
        cooling_setpoint=Config.GEN_COOLING_SETPOINT,
        ev_share_of_households=Config.GEN_EV_PENETRATION,
        solar_share_of_households=Config.GEN_SOLAR_PENETRATION,
        annual_trend=Config.GEN_ANNUAL_TREND,
        dsr_events_per_year=Config.GEN_DSR_EVENTS_PER_YEAR,
        dsr_strength_range=Config.GEN_DSR_STRENGTH,
    )
    return df, specs, {"источник": "synthetic", "рядов отобрано": len(specs)}


def build_panel_models(data: Dict[str, Any], seed: int) -> List[Tuple[Any, str]]:
    """
    Собирает состав моделей для панели.

    Наивные базлайны идут первыми: без них невозможно утверждать, что сложная
    модель вообще имеет прогностическую ценность.
    """
    from models.panel_models import (
        PanelNaive24, PanelHourlyProfile, build_panel_ridge, build_panel_xgboost,
    )
    from models.dlinear import build_dlinear

    n_hist = len(data["feature_names_hist"])
    n_future = len(data["feature_names_future"])
    n_static = len(data["static_names"])

    steps_per_epoch = max(len(data["Y_train"]) // Config.PANEL_BATCH_SIZE, 1)
    total_steps = Config.PANEL_EPOCHS * steps_per_epoch

    dlinear = build_dlinear(
        history_length=data["history_length"],
        forecast_horizon=data["forecast_horizon"],
        n_hist_features=n_hist, n_future_features=n_future,
        n_static_features=n_static,
        covariate_units=Config.PANEL_DLINEAR_UNITS,
        learning_rate=Config.PANEL_DLINEAR_LR,
        lr_schedule_total_steps=total_steps if Config.TRANSFORMER_USE_WARMUP_COSINE else 0,
        lr_warmup_fraction=Config.TRANSFORMER_WARMUP_FRACTION,
    )

    return [
        (PanelNaive24(), "Naive24"),
        (PanelHourlyProfile(), "HourlyProfile"),
        (build_panel_ridge(), "Ridge"),
        (build_panel_xgboost(n_estimators=Config.PANEL_XGB_ESTIMATORS, seed=seed),
         "XGBoost"),
        (dlinear, "DLinear"),
    ]


def run_probabilistic_block(data: Dict[str, Any], args, run_dir: str,
                           logger_obj) -> List[Dict[str, Any]]:
    """
    Обучает и оценивает вероятностные модели на той же панели.

    Блок отделён от точечного: у квантильных моделей другой выход (горизонт ×
    уровни), другая функция потерь и другой набор метрик. Смешивание в одном
    сравнении дало бы таблицу, где MAE точечной модели стоит рядом с pinball
    вероятностной, и величины несопоставимы.

    Тривиальный базлайн идёт первым: интервал по историческому разбросу ошибок
    уже даёт разумное покрытие, и без него нельзя утверждать, что обучаемая
    вероятностная модель что-то добавляет.
    """
    import pandas as pd

    from models.quantile_models import (
        DEFAULT_QUANTILES, PanelQuantileXGBoost, QuantileNaive,
        build_quantile_dlinear, evaluate_panel_quantiles,
    )

    n_hist = len(data["feature_names_hist"])
    n_future = len(data["feature_names_future"])
    n_static = len(data["static_names"])
    steps = max(len(data["Y_train"]) // Config.PANEL_BATCH_SIZE, 1)

    dlinear = build_quantile_dlinear(
        history_length=data["history_length"],
        forecast_horizon=data["forecast_horizon"],
        n_hist_features=n_hist, n_future_features=n_future,
        n_static_features=n_static,
        covariate_units=Config.PANEL_DLINEAR_UNITS,
        learning_rate=Config.PANEL_DLINEAR_LR,
        lr_schedule_total_steps=(Config.PANEL_EPOCHS * steps
                                 if Config.TRANSFORMER_USE_WARMUP_COSINE else 0),
        lr_warmup_fraction=Config.TRANSFORMER_WARMUP_FRACTION,
    )

    rows: List[Dict[str, Any]] = []
    for model, name in ((QuantileNaive(), "QuantileNaive"),
                        (PanelQuantileXGBoost(
                            n_estimators=Config.PANEL_XGB_ESTIMATORS,
                            seed=args.seed), "QuantileXGBoost"),
                        (dlinear, "QuantileDLinear")):
        t0 = time.time()
        logger_obj.info("Вероятностная модель: %s", name)
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model):
                from models.panel_trainer import make_batch
                batch_tr, batch_va = make_batch(data, "train"), make_batch(data, "val")
                names = [i.name.split(":")[0] for i in model.inputs]
                pick = lambda b: [{"hist_input": b["hist"], "future_input": b["future"],
                                   "static_input": b["static"]}[n]
                                  for n in names]
                model.fit(pick(batch_tr), data["Y_train"],
                          validation_data=(pick(batch_va), data["Y_val"]),
                          epochs=Config.PANEL_EPOCHS,
                          batch_size=Config.PANEL_BATCH_SIZE,
                          callbacks=[tf.keras.callbacks.EarlyStopping(
                              monitor="val_loss", patience=Config.PANEL_PATIENCE,
                              restore_best_weights=True, verbose=1)],
                          verbose=2)
            else:
                model.fit(data)
        except Exception as exc:
            logger_obj.error("Вероятностная модель %s не обучена: %s", name, exc)
            continue

        result = evaluate_panel_quantiles(model, data, "test", DEFAULT_QUANTILES)
        result["model"] = name
        result["train_time_sec"] = round(time.time() - t0, 1)
        rows.append(result)

        logger_obj.info(
            "%-16s pinball=%9.3f | покрытие %.3f при номинале %.2f | "
            "ширина %9.1f | пересечений %.4f",
            name, result["pinball_mean"], result["coverage"],
            result["coverage_nominal"], result["interval_width"],
            result["crossing_rate_before_sort"])

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, "quantile_metrics.csv"),
                                  index=False, encoding="utf-8-sig")
    return rows


def run_panel_pipeline(args, logger_obj) -> int:
    """
    Полный цикл panel-прогнозирования. Возвращает код завершения процесса.

    Код 2 означает частичный отказ: часть запрошенных моделей не обучилась.
    Молчаливое исключение упавшей модели с успешным кодом возврата скрывало бы
    отказ, поэтому такой запуск успешным не считается.
    """
    from data.panel import validate_panel
    from data.panel_preprocessing import prepare_panel_data
    from models.panel_trainer import (
        PanelTrainer, compare_panel_models, bottom_up_city_forecast,
    )
    from utils import reporting

    t_start = time.time()
    n_series = Config.PANEL_CITIES * Config.PANEL_FEEDERS_PER_CITY

    # ── Отказ до генерации, если режим не помещается в память ───────────────
    expected = estimate_windows(n_series, Config.PANEL_DAYS, Config.PANEL_HISTORY,
                                Config.FORECAST_HORIZON,
                                stride=Config.PANEL_WINDOW_STRIDE)
    logger_obj.info("Ожидается около %d обучающих окон на %d рядах (шаг окон %d)",
                    expected, n_series, Config.PANEL_WINDOW_STRIDE)
    if expected > _MAX_WINDOWS_IN_MEMORY:
        # Коэффициент 2 отражает удвоение при склейке кусков по рядам, а не
        # только размер готовых массивов: измеренный пик на panel-optimal с
        # прореживанием составил 8.5 ГБ против 1.2 ГБ у самих данных.
        gb = expected * (Config.PANEL_HISTORY * 20 + 24 * 15) * 4 * 2 / 1e9
        logger_obj.error(
            "Режим требует %d окон — при материализации в памяти это порядка %.0f ГБ "
            "с учётом удвоения на склейке. Уменьшите панель, увеличьте "
            "PANEL_WINDOW_STRIDE (значение должно быть взаимно простым с 24 и 168) "
            "или реализуйте потоковую подачу через tf.data. Запуск прерван, чтобы "
            "не исчерпать память посреди расчёта.",
            expected, gb,
        )
        return 1

    # ── Отказ до создания каталога, если источник данных недоступен ──────────
    # Тот же принцип, что и у проверки памяти: неудача после создания каталога
    # оставила бы пустой прогон, неотличимый от прерванного вручную.
    if getattr(args, "dataset", "synthetic") == "uci":
        from data.uci import UCI_FILENAME, UCI_URL
        if not os.path.exists(args.uci_path):
            logger_obj.error(
                "Набор UCI не найден: %s. Файл не скачивается автоматически — "
                "загрузите архив с %s и распакуйте %s по этому пути.",
                args.uci_path, UCI_URL, UCI_FILENAME)
            return 1

    run_dir = reporting.make_run_dir(Config.OUTPUT_DIR, args.mode, args.scenario, args.seed)
    Config.PLOTS_DIR = os.path.join(run_dir, "plots")
    Config.MODELS_DIR = os.path.join(run_dir, "models")
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    # ── [1/6] Данные ────────────────────────────────────────────────────────
    dataset = getattr(args, "dataset", "synthetic")
    logger_obj.info("\n[1/6] Подготовка панели (источник: %s)...", dataset)
    df, specs, source_report = load_panel_dataset(args, logger_obj)

    panel_ok, panel_rows = validate_panel(df, specs)

    # ── [2/6] Нарезка окон ──────────────────────────────────────────────────
    logger_obj.info("\n[2/6] Подготовка окон (скалеры только на train)...")
    data = prepare_panel_data(
        df, specs,
        history_length=Config.PANEL_HISTORY,
        forecast_horizon=Config.FORECAST_HORIZON,
        train_ratio=Config.TRAIN_RATIO, val_ratio=Config.VAL_RATIO,
        seed=args.seed, window_stride=Config.PANEL_WINDOW_STRIDE,
    )

    # ── [3/6] Обучение ──────────────────────────────────────────────────────
    logger_obj.info("\n[3/6] Обучение глобальных моделей...")
    to_train = build_panel_models(data, args.seed)
    requested = [name for _, name in to_train]
    trained: List[str] = []
    failed: List[Dict[str, str]] = []
    trainers: List[Any] = []

    for model, name in to_train:
        trainer = PanelTrainer(model, name)
        try:
            trainer.train(data, epochs=Config.PANEL_EPOCHS,
                          batch_size=Config.PANEL_BATCH_SIZE,
                          patience=Config.PANEL_PATIENCE)
        except Exception as exc:
            logger_obj.error("Модель %s не обучена: %s", name, exc)
            failed.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        trainers.append(trainer)
        trained.append(name)

    if not trainers:
        logger_obj.error("Ни одна модель не обучена — дальнейшие шаги невозможны.")
        return 1

    # ── [4/6] Отбор по валидации и оценка на тесте ──────────────────────────
    logger_obj.info("\n[4/6] Отбор по ВАЛИДАЦИИ, затем оценка на тесте...")
    val_metrics = compare_panel_models(trainers, data, "val")
    learned = {n: m for n, m in val_metrics.items()
               if n not in ("Naive24", "HourlyProfile")}
    pool = learned or val_metrics

    # Отбор по MASE, а не по micro-MAE. Абсолютная ошибка на панели
    # определяется крупнейшим фидером: он один выбирал бы победителя, а
    # качество на мелких рядах не влияло бы на решение. MASE безразмерна,
    # поэтому каждый ряд весит одинаково. Запасной критерий — macro-MAE:
    # он тоже уравнивает ряды, но остаётся в абсолютных величинах.
    def _selection_criterion(name: str) -> float:
        value = pool[name].get("MASE")
        if value is not None and np.isfinite(value):
            return float(value)
        return float(pool[name]["MAE_macro"])

    best_name = min(pool, key=_selection_criterion)
    best_trainer = next(t for t in trainers if t.name == best_name)
    logger_obj.info(
        "Лучшая модель по валидации: %s (MASE_val=%.3f, macro MAE_val=%.2f)",
        best_name, pool[best_name].get("MASE", float("nan")),
        pool[best_name]["MAE_macro"])

    test_metrics = compare_panel_models(trainers, data, "test")

    # ── [5/6] Иерархия ──────────────────────────────────────────────────────
    logger_obj.info("\n[5/6] Городской прогноз суммированием фидеров...")
    hierarchy = bottom_up_city_forecast(
        data, best_trainer.predict(data, "test"), "test")

    # ── Вероятностный прогноз (по запросу) ──────────────────────────────────
    quantile_rows: List[Dict[str, Any]] = []
    if getattr(args, "probabilistic", False):
        logger_obj.info("Вероятностный прогноз: квантили 0.1, 0.5, 0.9")
        quantile_rows = run_probabilistic_block(data, args, run_dir, logger_obj)

    # ── [6/6] Экспорт ───────────────────────────────────────────────────────
    logger_obj.info("\n[6/6] Экспорт результатов...")
    run_meta = {
        "mode": args.mode, "scenario": args.scenario, "seed": args.seed,
        "dataset": dataset,
        "dataset_report": source_report,
        "panel": {
            "cities": Config.PANEL_CITIES,
            "feeders_per_city": Config.PANEL_FEEDERS_PER_CITY,
            "n_series": len(data["series_index"]),
            "days": Config.PANEL_DAYS,
        },
        "window_stride": data.get("window_stride", 1),
        "history_length": data["history_length"],
        "forecast_horizon": data["forecast_horizon"],
        "n_features": {
            "hist": len(data["feature_names_hist"]),
            "future": len(data["feature_names_future"]),
            "static": len(data["static_names"]),
        },
        "split_sizes": {"train": int(len(data["Y_train"])),
                        "val": int(len(data["Y_val"])),
                        "test": int(len(data["Y_test"]))},
        "requested_models": requested,
        "trained_models": trained,
        "failed_models": failed,
        "partial_failure": bool(failed),
        "best_model_by_val": best_name,
        "quantile_models": [r["model"] for r in quantile_rows],
        "panel_validation_passed": bool(panel_ok),
        "hierarchy_city_MAE": hierarchy["MAE_mean"],
        "hierarchy_city_MAPE": hierarchy["MAPE_mean"],
        "runtime_sec": round(time.time() - t_start, 1),
    }
    reporting.write_run_metadata(run_dir, run_meta)

    # Метрики без словаря по рядам: он выгружается отдельным файлом.
    flat = {n: {k: v for k, v in m.items() if k != "per_series_MAE"}
            for n, m in test_metrics.items()}
    flat_val = {n: {k: v for k, v in m.items() if k != "per_series_MAE"}
                for n, m in val_metrics.items()}
    for target in (run_dir, Config.OUTPUT_DIR):
        reporting.export_metrics(flat, target, seed=args.seed, split="test",
                                 run_meta=run_meta, append=True)
        reporting.export_metrics(flat_val, target, seed=args.seed, split="val",
                                 run_meta=run_meta, append=True)

    rows = [{"model": n, "series": s, "MAE": v}
            for n, m in test_metrics.items()
            for s, v in m["per_series_MAE"].items()]
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "metrics_by_series.csv"),
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(hierarchy["per_city"]).to_csv(
        os.path.join(run_dir, "metrics_by_city.csv"), index=False, encoding="utf-8-sig")
    reporting.export_markdown_tables(flat, run_dir)
    reporting.aggregate_seeds(Config.OUTPUT_DIR)

    elapsed = (time.time() - t_start) / 60
    if failed:
        logger_obj.error(
            "Panel-конвейер завершён ЧАСТИЧНО за %.1f мин: не обучено %d из %d моделей.",
            elapsed, len(failed), len(requested))
        for item in failed:
            logger_obj.error("  не обучена: %s — %s", item["model"], item["error"])
        return 2

    logger_obj.info("\n" + "=" * 78)
    logger_obj.info("Panel-конвейер завершён за %.1f мин. Результаты: %s",
                    elapsed, run_dir)
    logger_obj.info("  Рядов: %d | окон: train=%d test=%d | лучшая по валидации: %s",
                    len(data["series_index"]), len(data["Y_train"]),
                    len(data["Y_test"]), best_name)
    logger_obj.info("=" * 78)
    return 0
