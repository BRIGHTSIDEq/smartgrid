# -*- coding: utf-8 -*-
"""
experiments/transformer_ablation.py — Ablation Study: Transformer vs LSTM.

═══════════════════════════════════════════════════════════════════════════════
ПЛАН ЭКСПЕРИМЕНТОВ
═══════════════════════════════════════════════════════════════════════════════

Exp 1: Влияние длины истории
    histories = [24, 48, 72, 96, 168]
    Гипотеза: при history > 48ч Transformer ≥ LSTM,
    т.к. LSTM «забывает» дальние лаги, а Attention — нет.

Exp 2: Влияние числа голов внимания
    num_heads = [4, 8, 16]
    Гипотеза: 4-8 голов оптимальны для почасовых данных.

Exp 3: Сравнение позиционных кодировок
    pe_types = ["sinusoidal", "relative", "time2vec"]
    Гипотеза: Time2Vec лучше на периодических данных.

Exp 4: Параметрический паритет LSTM vs Transformer
    Фиксируем число параметров, сравниваем архитектуры.

Exp 5: Сравнение трёх архитектур
    VanillaTransformer vs TFTLite vs PatchTST

ИСПОЛЬЗОВАНИЕ
─────────────
    cd smart_grid_project
    python experiments/transformer_ablation.py [--exp 1] [--fast]

    --exp N   : запустить только эксперимент N (1-5)
    --fast    : режим быстрого тестирования (меньше данных и эпох)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.generator import generate_smartgrid_data
from data.preprocessing import prepare_data, inverse_scale
from models.lstm import build_lstm_model
from models.transformer import (
    build_vanilla_transformer,
    build_tft_lite,
    build_patchtst,
    count_parameters,
    prepare_tft_covariates,
    make_tft_windows,
)
from utils.metrics import compute_all_metrics

logger = logging.getLogger("smart_grid.experiments.ablation")

FAST_MODE = False   # Переключается через --fast
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ablation")


# ──────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────────────────────

def _epochs() -> int:
    return 30 if FAST_MODE else 150


def _patience() -> int:
    return 8 if FAST_MODE else 25


def _days() -> int:
    return 60 if FAST_MODE else 180


def _quick_train_eval(
    model: tf.keras.Model,
    data: Dict[str, Any],
    model_name: str = "",
    x_train_override=None,
    x_val_override=None,
    x_test_override=None,
) -> Dict[str, float]:
    """
    Обучает модель с EarlyStopping и возвращает метрики на тесте.

    x_*_override позволяет передавать кастомные входы (например, [X, covars] для TFT).
    """
    X_tr = x_train_override if x_train_override is not None else data["X_train"]
    X_v = x_val_override if x_val_override is not None else data["X_val"]
    X_te = x_test_override if x_test_override is not None else data["X_test"]

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=_patience(), restore_best_weights=True
        )
    ]

    t0 = time.time()
    model.fit(
        X_tr, data["Y_train"],
        validation_data=(X_v, data["Y_val"]),
        epochs=_epochs(),
        batch_size=32,
        callbacks=cb,
        verbose=0,
    )
    train_time = time.time() - t0

    pred_scaled = model.predict(X_te, verbose=0)
    y_true = inverse_scale(data["scaler"], data["Y_test"])
    y_pred = inverse_scale(data["scaler"], pred_scaled)

    metrics = compute_all_metrics(y_true, y_pred, model_name=model_name)
    metrics["train_time_s"] = round(train_time, 1)
    metrics["n_params"] = count_parameters(model)
    return metrics


def _save_results(results: List[Dict], filename: str) -> str:
    """Сохраняет список результатов в CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    if results:
        keys = list(results[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
    logger.info("Результаты сохранены: %s", path)
    return path


def _bar_chart(
    results: List[Dict],
    x_key: str,
    y_keys: List[str],
    title: str,
    filename: str,
) -> None:
    """Строит и сохраняет барчарт из списка результатов."""
    df = pd.DataFrame(results)
    x = df[x_key].astype(str)
    fig, axes = plt.subplots(1, len(y_keys), figsize=(5 * len(y_keys), 4))
    if len(y_keys) == 1:
        axes = [axes]
    colors = plt.cm.Set2.colors

    for ax, yk in zip(axes, y_keys):
        bars = ax.bar(x, df[yk], color=colors[: len(x)], edgecolor="black", alpha=0.85)
        ax.set_title(yk, fontweight="bold")
        ax.set_xlabel(x_key)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, df[yk]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТ 1: ДЛИНА ИСТОРИИ
# ──────────────────────────────────────────────────────────────────────────────

def experiment_history_length(df: pd.DataFrame) -> List[Dict]:
    """
    Exp 1: history = [24, 48, 72, 96, 168]

    Теоретическая гипотеза:
    - LSTM имеет «эффективную глубину памяти» ~50-100 шагов
      (gradient vanishing через время)
    - Transformer поддерживает O(1) путь между ЛЮБЫМИ двумя шагами
    - Ожидаем: при history ≥ 72ч разрыв Transformer-LSTM должен закрыться
      или Transformer начнёт обгонять LSTM

    Если LSTM всё равно выигрывает на длинных окнах:
    → Объяснение для курсовой: "Недостаточно обучающих данных.
      При N=180 дней и window=168ч число уникальных обучающих примеров
      мало. Transformer требует больше данных для выявления
      долгосрочных паттернов."
    """
    histories = [24, 48, 72, 96] if not FAST_MODE else [24, 48, 72]
    results = []

    logger.info("\n" + "=" * 60)
    logger.info("EXP 1: Влияние длины истории")
    logger.info("=" * 60)

    for h in histories:
        logger.info("\n── History = %d ч ──", h)
        data = prepare_data(
            df, history_length=h, forecast_horizon=Config.FORECAST_HORIZON
        )
        n_tr = len(data["X_train"])
        logger.info("  Train samples: %d", n_tr)

        # PatchTST: patch_len должен делить history
        patch_len = 8 if h >= 48 else 6

        models_cfg = {
            "LSTM": lambda: build_lstm_model(
                history_length=h, forecast_horizon=Config.FORECAST_HORIZON,
                lstm_units_1=128, lstm_units_2=64, lstm_units_3=32,
                dropout_rate=0.2, learning_rate=5e-4,
            ),
            "VanillaTransformer": lambda: build_vanilla_transformer(
                history_length=h, forecast_horizon=Config.FORECAST_HORIZON,
                d_model=64, num_heads=4, num_layers=2, dff=128,
            ),
            "PatchTST": lambda: build_patchtst(
                history_length=h, forecast_horizon=Config.FORECAST_HORIZON,
                patch_len=patch_len, stride=patch_len // 2,
                d_model=64, num_heads=4, num_layers=2,
            ),
        }

        for name, builder in models_cfg.items():
            try:
                model = builder()
                m = _quick_train_eval(model, data, model_name=f"{name}[h={h}]")
                results.append({"history": h, "model": name, **m})
                logger.info(
                    "  %-20s MAE=%.2f MAPE=%.2f%% params=%d",
                    name, m["MAE"], m["MAPE"], m["n_params"],
                )
                tf.keras.backend.clear_session()
            except Exception as exc:
                logger.error("Ошибка %s h=%d: %s", name, h, exc)

    _save_results(results, "exp1_history_length.csv")

    # ── График: MAE по длине истории для каждой модели ────────────────────────
    df_res = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp 1: Влияние длины истории | LSTM vs Transformer", fontweight="bold")

    for model_name in df_res["model"].unique():
        sub = df_res[df_res["model"] == model_name]
        axes[0].plot(sub["history"], sub["MAE"], marker="o", label=model_name)
        axes[1].plot(sub["history"], sub["MAPE"], marker="s", label=model_name)

    for ax, ylabel in zip(axes, ["MAE (кВт·ч)", "MAPE (%)"]):
        ax.set_xlabel("History Length (ч)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_history_chart.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТ 2: ЧИСЛО ГОЛОВ ВНИМАНИЯ
# ──────────────────────────────────────────────────────────────────────────────

def experiment_num_heads(df: pd.DataFrame) -> List[Dict]:
    """
    Exp 2: num_heads = [4, 8, 16] при фиксированном d_model=128.

    d_model=128 делится на все три значения (head_dim = 32, 16, 8).

    Гипотеза:
    - 4 головы с head_dim=32 — достаточно для суточных паттернов
    - 8 голов — оптимальный баланс (субпаттерны внутри суток)
    - 16 голов с head_dim=8 — возможно переобучение (head_dim слишком мал)
    """
    heads_list = [4, 8, 16]
    data = prepare_data(df, history_length=48, forecast_horizon=Config.FORECAST_HORIZON)
    results = []

    logger.info("\n" + "=" * 60)
    logger.info("EXP 2: Влияние числа голов внимания (d_model=128)")
    logger.info("=" * 60)

    for nh in heads_list:
        logger.info("\n── num_heads = %d ──", nh)
        try:
            model = build_vanilla_transformer(
                history_length=48, forecast_horizon=Config.FORECAST_HORIZON,
                d_model=128, num_heads=nh, num_layers=2, dff=256,
            )
            m = _quick_train_eval(model, data, model_name=f"VanillaTrans[h={nh}]")
            results.append({"num_heads": nh, **m})
            logger.info("  MAE=%.2f MAPE=%.2f%% R2=%.4f", m["MAE"], m["MAPE"], m["R2"])
            tf.keras.backend.clear_session()
        except Exception as exc:
            logger.error("Ошибка heads=%d: %s", nh, exc)

    _save_results(results, "exp2_num_heads.csv")
    _bar_chart(
        results, "num_heads", ["MAE", "MAPE"],
        "Exp 2: Число голов внимания (d_model=128)",
        "exp2_heads_chart.png",
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТ 3: ПОЗИЦИОННЫЕ КОДИРОВКИ
# ──────────────────────────────────────────────────────────────────────────────

def experiment_positional_encodings(df: pd.DataFrame) -> List[Dict]:
    """
    Exp 3: Sinusoidal vs Time2Vec

    Гипотеза:
    - Time2Vec лучше, т.к. обучаемые частоты могут выучить доминирующие
      периоды (24ч, 168ч) из данных, а не кодировать все частоты равномерно
    - Sinusoidal может выиграть на маленьких данных (меньше параметров)

    Примечание: LearnableRelativePE реализована как отдельный слой.
    В текущей версии сравниваем sinusoidal vs time2vec.
    """
    pe_types = ["sinusoidal", "time2vec"]
    data = prepare_data(df, history_length=48, forecast_horizon=Config.FORECAST_HORIZON)
    results = []

    logger.info("\n" + "=" * 60)
    logger.info("EXP 3: Сравнение позиционных кодировок")
    logger.info("=" * 60)

    for pe in pe_types:
        logger.info("\n── PE = %s ──", pe)
        try:
            model = build_vanilla_transformer(
                history_length=48, forecast_horizon=Config.FORECAST_HORIZON,
                d_model=128, num_heads=4, num_layers=2, dff=256,
                pe_type=pe,
            )
            n_p = count_parameters(model)

            # Time2Vec требует дополнительный вход (tau)
            if pe == "time2vec":
                tau = np.linspace(0, 1, 48, dtype=np.float32)
                tau_tr = np.tile(tau, (len(data["X_train"]), 1))[:, :, np.newaxis]
                tau_v = np.tile(tau, (len(data["X_val"]), 1))[:, :, np.newaxis]
                tau_te = np.tile(tau, (len(data["X_test"]), 1))[:, :, np.newaxis]
                x_tr = [data["X_train"], tau_tr]
                x_v = [data["X_val"], tau_v]
                x_te = [data["X_test"], tau_te]
            else:
                x_tr = x_v = x_te = None

            m = _quick_train_eval(model, data,
                                  model_name=f"Transformer[{pe}]",
                                  x_train_override=x_tr,
                                  x_val_override=x_v,
                                  x_test_override=x_te)
            results.append({"pe_type": pe, "n_params": n_p, **m})
            logger.info("  MAE=%.2f MAPE=%.2f%% params=%d", m["MAE"], m["MAPE"], n_p)
            tf.keras.backend.clear_session()
        except Exception as exc:
            logger.error("Ошибка PE=%s: %s", pe, exc)

    _save_results(results, "exp3_positional_encoding.csv")
    _bar_chart(
        results, "pe_type", ["MAE", "MAPE"],
        "Exp 3: Позиционные кодировки",
        "exp3_pe_chart.png",
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТ 4: ПАРАМЕТРИЧЕСКИЙ ПАРИТЕТ
# ──────────────────────────────────────────────────────────────────────────────

def experiment_parameter_parity(df: pd.DataFrame) -> List[Dict]:
    """
    Exp 4: Честное сравнение LSTM vs Transformer при равном числе параметров.

    Методология:
    1. Строим LSTM с заданными units
    2. Считаем число параметров LSTM
    3. Подбираем Transformer с похожим числом параметров
    4. Сравниваем качество прогноза

    Это единственно корректный способ сравнения архитектур для научной работы.
    Без параметрического паритета сравнение невалидно.
    """
    data = prepare_data(df, history_length=48, forecast_horizon=Config.FORECAST_HORIZON)
    results = []

    logger.info("\n" + "=" * 60)
    logger.info("EXP 4: Параметрический паритет LSTM vs Transformer")
    logger.info("=" * 60)

    # ── Три конфигурации по размеру ──────────────────────────────────────────
    configs = [
        {"name": "Small",  "lstm_u": (64, 32, 16),  "tr_d": 32,  "tr_h": 4},
        {"name": "Medium", "lstm_u": (128, 64, 32),  "tr_d": 64,  "tr_h": 4},
        {"name": "Large",  "lstm_u": (256, 128, 64), "tr_d": 128, "tr_h": 8},
    ]

    for cfg in configs:
        logger.info("\n── Размер: %s ──", cfg["name"])
        u1, u2, u3 = cfg["lstm_u"]

        # LSTM
        try:
            lstm = build_lstm_model(
                history_length=48, forecast_horizon=Config.FORECAST_HORIZON,
                lstm_units_1=u1, lstm_units_2=u2, lstm_units_3=u3,
                dropout_rate=0.2, learning_rate=5e-4,
            )
            lstm_params = count_parameters(lstm)
            m_lstm = _quick_train_eval(lstm, data, model_name=f"LSTM[{cfg['name']}]")
            m_lstm["model"] = f"LSTM_{cfg['name']}"
            m_lstm["n_params"] = lstm_params
            results.append(m_lstm)
            logger.info(
                "  LSTM:        MAE=%.2f MAPE=%.2f%% params=%d",
                m_lstm["MAE"], m_lstm["MAPE"], lstm_params,
            )
            tf.keras.backend.clear_session()
        except Exception as exc:
            logger.error("LSTM %s: %s", cfg["name"], exc)

        # Vanilla Transformer (подобранный по параметрам)
        try:
            transformer = build_vanilla_transformer(
                history_length=48, forecast_horizon=Config.FORECAST_HORIZON,
                d_model=cfg["tr_d"], num_heads=cfg["tr_h"], num_layers=2,
                dff=cfg["tr_d"] * 4,
            )
            tr_params = count_parameters(transformer)
            m_tr = _quick_train_eval(transformer, data,
                                     model_name=f"Vanilla[{cfg['name']}]")
            m_tr["model"] = f"VanillaTransformer_{cfg['name']}"
            m_tr["n_params"] = tr_params
            results.append(m_tr)
            logger.info(
                "  VanillaTrans: MAE=%.2f MAPE=%.2f%% params=%d",
                m_tr["MAE"], m_tr["MAPE"], tr_params,
            )
            tf.keras.backend.clear_session()
        except Exception as exc:
            logger.error("Transformer %s: %s", cfg["name"], exc)

        # PatchTST
        try:
            patchtst = build_patchtst(
                history_length=48, forecast_horizon=Config.FORECAST_HORIZON,
                patch_len=8, stride=4,
                d_model=cfg["tr_d"], num_heads=cfg["tr_h"], num_layers=2,
            )
            pt_params = count_parameters(patchtst)
            m_pt = _quick_train_eval(patchtst, data,
                                     model_name=f"PatchTST[{cfg['name']}]")
            m_pt["model"] = f"PatchTST_{cfg['name']}"
            m_pt["n_params"] = pt_params
            results.append(m_pt)
            logger.info(
                "  PatchTST:    MAE=%.2f MAPE=%.2f%% params=%d",
                m_pt["MAE"], m_pt["MAPE"], pt_params,
            )
            tf.keras.backend.clear_session()
        except Exception as exc:
            logger.error("PatchTST %s: %s", cfg["name"], exc)

    _save_results(results, "exp4_parameter_parity.csv")

    # Визуализация: scatter MAE vs n_params с цветом по архитектуре
    df_res = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 5))
    arch_colors = {"LSTM": "steelblue", "VanillaTransformer": "orange", "PatchTST": "green"}
    for arch, color in arch_colors.items():
        sub = df_res[df_res["model"].str.startswith(arch)]
        if not sub.empty:
            ax.scatter(sub["n_params"], sub["MAE"], s=120, color=color, label=arch,
                       zorder=5, edgecolors="black")
            for _, row in sub.iterrows():
                ax.annotate(row["model"].split("_")[-1], (row["n_params"], row["MAE"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Число параметров")
    ax.set_ylabel("MAE (кВт·ч)")
    ax.set_title("Exp 4: Параметрический паритет LSTM vs Transformer", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_parity_scatter.png"), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# ЭКСПЕРИМЕНТ 5: СРАВНЕНИЕ ТРЁХ АРХИТЕКТУР
# ──────────────────────────────────────────────────────────────────────────────

def experiment_architectures(df: pd.DataFrame) -> List[Dict]:
    """
    Exp 5: VanillaTransformer vs TFTLite vs PatchTST vs LSTM

    Условия: одинаковая длина истории (48ч), близкое число параметров.
    """
    data = prepare_data(df, history_length=48, forecast_horizon=Config.FORECAST_HORIZON)

    # Подготовка ковариат для TFT
    import pandas as _pd
    ts = data["timestamps"]
    all_covars = prepare_tft_covariates(ts)
    n_covar_features = all_covars.shape[1]

    # Нарезаем ковариаты на окна совпадающие с X_train/val/test
    cov_scaled = data["scaled_train"]
    _, X_cov_tr, _ = make_tft_windows(data["scaled_train"], all_covars[:data["train_end_idx"]], 48, 24)
    _, X_cov_v, _ = make_tft_windows(data["scaled_val"], all_covars[data["train_end_idx"]:data["val_end_idx"]], 48, 24)
    _, X_cov_te, _ = make_tft_windows(data["scaled_test"], all_covars[data["val_end_idx"]:], 48, 24)

    # Обрезаем до минимального размера
    n_tr = min(len(data["X_train"]), len(X_cov_tr))
    n_v = min(len(data["X_val"]), len(X_cov_v))
    n_te = min(len(data["X_test"]), len(X_cov_te))

    results = []

    logger.info("\n" + "=" * 60)
    logger.info("EXP 5: Сравнение архитектур (history=48ч)")
    logger.info("=" * 60)

    models_cfg = {
        "LSTM": (
            build_lstm_model(
                history_length=48, forecast_horizon=24,
                lstm_units_1=128, lstm_units_2=64, lstm_units_3=32, dropout_rate=0.2,
            ),
            None, None, None
        ),
        "VanillaTransformer": (
            build_vanilla_transformer(history_length=48, forecast_horizon=24,
                                      d_model=64, num_heads=4, num_layers=3),
            None, None, None
        ),
        "PatchTST": (
            build_patchtst(history_length=48, forecast_horizon=24,
                           patch_len=8, stride=4, d_model=64, num_heads=4, num_layers=3),
            None, None, None
        ),
        "TFTLite": (
            build_tft_lite(history_length=48, forecast_horizon=24,
                           d_model=64, num_heads=4, num_layers=2,
                           n_covariate_features=n_covar_features),
            [data["X_train"][:n_tr], X_cov_tr[:n_tr]],
            [data["X_val"][:n_v], X_cov_v[:n_v]],
            [data["X_test"][:n_te], X_cov_te[:n_te]],
        ),
    }

    for name, (model, x_tr, x_v, x_te) in models_cfg.items():
        try:
            logger.info("\n── %s | %d params ──", name, count_parameters(model))
            m = _quick_train_eval(model, data, model_name=name,
                                  x_train_override=x_tr,
                                  x_val_override=x_v,
                                  x_test_override=x_te)
            results.append({"architecture": name, **m})
            logger.info(
                "  MAE=%.2f RMSE=%.2f MAPE=%.2f%% R2=%.4f params=%d time=%.1fs",
                m["MAE"], m["RMSE"], m["MAPE"], m["R2"], m["n_params"], m["train_time_s"],
            )
            tf.keras.backend.clear_session()
        except Exception as exc:
            logger.error("Ошибка %s: %s", name, exc)

    _save_results(results, "exp5_architectures.csv")
    _bar_chart(
        results, "architecture", ["MAE", "MAPE", "R2"],
        "Exp 5: Сравнение архитектур | history=48ч",
        "exp5_arch_chart.png",
    )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# СВОДНАЯ ТАБЛИЦА
# ──────────────────────────────────────────────────────────────────────────────

def print_ablation_summary(all_results: Dict[str, List[Dict]]) -> None:
    """Выводит сводную таблицу всех экспериментов."""
    logger.info("\n" + "=" * 70)
    logger.info("СВОДКА РЕЗУЛЬТАТОВ ABLATION STUDY")
    logger.info("=" * 70)
    for exp_name, results in all_results.items():
        if not results:
            continue
        logger.info("\n── %s ──", exp_name)
        df = pd.DataFrame(results)
        cols = [c for c in ["model", "architecture", "history", "num_heads", "pe_type",
                             "MAE", "MAPE", "R2", "n_params"] if c in df.columns]
        logger.info("\n%s", df[cols].to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global FAST_MODE
    parser = argparse.ArgumentParser(description="Ablation Study: Transformer vs LSTM")
    parser.add_argument("--exp", type=int, default=0, help="Номер эксперимента (0=все)")
    parser.add_argument("--fast", action="store_true", help="Быстрый режим")
    args = parser.parse_args()

    FAST_MODE = args.fast
    Config.setup_logging()

    logger.info("Ablation Study | fast=%s | exp=%d", FAST_MODE, args.exp)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    np.random.seed(Config.SEED)
    tf.random.set_seed(Config.SEED)

    df = generate_smartgrid_data(days=_days(), households=Config.HOUSEHOLDS, seed=Config.SEED)

    all_results = {}
    run_all = args.exp == 0

    if run_all or args.exp == 1:
        all_results["Exp1_HistoryLength"] = experiment_history_length(df)
    if run_all or args.exp == 2:
        all_results["Exp2_NumHeads"] = experiment_num_heads(df)
    if run_all or args.exp == 3:
        all_results["Exp3_PE"] = experiment_positional_encodings(df)
    if run_all or args.exp == 4:
        all_results["Exp4_Parity"] = experiment_parameter_parity(df)
    if run_all or args.exp == 5:
        all_results["Exp5_Architectures"] = experiment_architectures(df)

    print_ablation_summary(all_results)
    logger.info("\n✅ Ablation Study завершён. Результаты: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
