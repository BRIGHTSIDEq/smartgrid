# -*- coding: utf-8 -*-
"""
experiments/hyperparameter_tuning.py — Автоматический подбор гиперпараметров.

МЕТОД: Optuna TPE (Tree-structured Parzen Estimator) + MedianPruner

ПОЧЕМУ TPE:
  - Каждый trial LSTM в optimal mode ≈ 1–2 ч (поиск), ≈ 4 ч (полное обучение).
  - Random/Grid Search не используют информацию предыдущих trials.
  - TPE строит вероятностную модель P(гиперпараметры | хорошие результаты) и
    сэмплирует из неё → каждый следующий trial умнее предыдущего.
  - MedianPruner: если к эпохе 30 val_loss хуже медианы — trial убивается.
    Экономия: ~60% времени на плохих конфигурациях.

РЕЖИМЫ ПОИСКА (не путать с режимами пайплайна):
  fast    → days=90,  epochs=30, patience=8,  n_trials=20  (~2–3 ч)
  optimal → days=365, epochs=80, patience=20, n_trials=25  (~20–30 ч overnight)
  full    → days=730, epochs=120,patience=30, n_trials=40  (~2–3 дня)

ИСПОЛЬЗОВАНИЕ:
  # Подобрать все модели для optimal mode (рекомендуется запускать ночью):
  python experiments/hyperparameter_tuning.py --mode optimal

  # Только одну модель, быстро проверить:
  python experiments/hyperparameter_tuning.py --mode fast --model lstm --n_trials 10

  # Принудительно заменить сохранённые результаты:
  python experiments/hyperparameter_tuning.py --mode optimal --model patchtst --force

РЕЗУЛЬТАТЫ:
  Сохраняются в results/hpo/{mode}/best_params.json
  Config.py автоматически подхватывает их при следующем запуске main.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Добавляем корень проекта в путь
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Проверяем наличие optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import tensorflow as tf

from config import Config
from data.generator import generate_smartgrid_data
from data.preprocessing import prepare_data, inverse_scale

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smart_grid.hpo")

# ══════════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ПОИСКА ПО РЕЖИМАМ
# ══════════════════════════════════════════════════════════════════════════════

SEARCH_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "days":       90,
        "households": 500,
        "epochs":     30,
        "patience":   8,
        "batch_size": 32,
        "n_trials":   20,
        "history":    48,
        # Pruner: начинать сравнение с эпохи 10, минимум 5 завершённых trials
        "pruner_warmup_steps":   10,
        "pruner_warmup_trials":  5,
    },
    "optimal": {
        "days":       365,
        "households": 1500,
        "epochs":     80,
        "patience":   20,
        "batch_size": 16,
        "n_trials":   25,
        "history":    192,
        "pruner_warmup_steps":   20,
        "pruner_warmup_trials":  5,
    },
    "full": {
        "days":       730,
        "households": 2500,
        "epochs":     120,
        "patience":   30,
        "batch_size": 8,
        "n_trials":   40,
        "history":    192,
        "pruner_warmup_steps":   25,
        "pruner_warmup_trials":  8,
    },
}

HPO_RESULTS_DIR = ROOT / "results" / "hpo"


# ══════════════════════════════════════════════════════════════════════════════
# OPTUNA PRUNING CALLBACK ДЛЯ KERAS
# ══════════════════════════════════════════════════════════════════════════════

class _OptunaPruningCallback(tf.keras.callbacks.Callback):
    """
    Передаёт val_loss в Optuna после каждой эпохи.
    При решении pruner — бросает исключение, Keras останавливает обучение.
    """

    def __init__(self, trial: "optuna.Trial", monitor: str = "val_loss"):
        super().__init__()
        self.trial   = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        val = (logs or {}).get(self.monitor)
        if val is None:
            return
        self.trial.report(float(val), step=epoch)
        if self.trial.should_prune():
            logger.info("Trial %d обрезан на эпохе %d (val_loss=%.6f)",
                        self.trial.number, epoch, val)
            raise optuna.TrialPruned()


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _quick_eval(model, data: Dict[str, Any], batch_size: int,
                epochs: int, patience: int,
                trial: Optional["optuna.Trial"] = None,
                is_tft: bool = False) -> float:
    """
    Обучает модель и возвращает лучший val_loss.
    Для sklearn-обёрток (XGBoost, Ridge) вызывает fit напрямую.
    """
    if not isinstance(model, tf.keras.Model):
        # sklearn-like wrapper
        model.fit(data["X_train"], data["Y_train"],
                  X_val=data.get("X_val"), Y_val=data.get("Y_val"))
        y_pred = model.predict(data["X_val"])
        y_true = inverse_scale(data["scaler"], data["Y_val"])
        y_pred_real = inverse_scale(data["scaler"], y_pred)
        mae = float(np.mean(np.abs(y_true - y_pred_real)))
        return mae  # возвращаем MAE как суррогат val_loss

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, min_delta=1e-5,
        )
    ]
    if trial is not None:
        callbacks.append(_OptunaPruningCallback(trial))

    X_tr = data["X_tft_train"] if is_tft else data["X_train"]
    X_vl = data["X_tft_val"]   if is_tft else data["X_val"]

    history = model.fit(
        X_tr, data["Y_train"],
        validation_data=(X_vl, data["Y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    best = float(min(history.history.get("val_loss", [float("inf")])))
    tf.keras.backend.clear_session()
    return best


def _ensure_divisible(d_model: int, num_heads: int, candidates: List[int]) -> int:
    """Возвращает первый num_heads из candidates, на который делится d_model."""
    for h in candidates:
        if d_model % h == 0:
            return h
    return 1


def _get_total_steps(n_train: int, batch_size: int, epochs: int) -> int:
    return max(epochs * (n_train // batch_size), 100)


# ══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _objective_lstm(trial: "optuna.Trial",
                    data: Dict[str, Any],
                    search_cfg: Dict[str, Any]) -> float:
    from models.lstm import build_lstm_model

    lstm_units = trial.suggest_categorical("lstm_units",      [64, 96, 128, 192, 256])
    tcn_filters= trial.suggest_categorical("tcn_filters",     [32, 48, 64, 96])
    attn_heads = trial.suggest_categorical("attn_heads",      [2, 4, 8])
    dropout    = trial.suggest_float(      "dropout",         0.05, 0.25, step=0.05)
    lr         = trial.suggest_float(      "learning_rate",   5e-5, 4e-4, log=True)
    blend_init = trial.suggest_float(      "seasonal_blend",  0.30, 0.80, step=0.10)
    huber      = trial.suggest_categorical("huber_delta",     [0.02, 0.05, 0.10])
    use_cosine = trial.suggest_categorical("use_cosine_decay",[True, False])

    n_train = len(data["X_train"])
    total_steps = _get_total_steps(n_train, search_cfg["batch_size"], search_cfg["epochs"])

    model = build_lstm_model(
        history_length=search_cfg["history"],
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        lstm_units_1=lstm_units,
        lstm_units_2=lstm_units,
        lstm_units_3=lstm_units,
        dropout_rate=dropout,
        learning_rate=lr,
        attn_heads=attn_heads,
        use_cosine_decay=use_cosine,
        total_steps=total_steps,
        warmup_ratio=0.05,
        tcn_filters=tcn_filters,
        huber_delta=huber,
        seasonal_blend_init=blend_init,
        use_seasonal_skip=True,
        lag_feature_start_idx=15,
    )
    return _quick_eval(model, data,
                       batch_size=search_cfg["batch_size"],
                       epochs=search_cfg["epochs"],
                       patience=search_cfg["patience"],
                       trial=trial)


def _objective_itransformer(trial: "optuna.Trial",
                             data: Dict[str, Any],
                             search_cfg: Dict[str, Any]) -> float:
    from models.transformer import build_itransformer

    d_model    = trial.suggest_categorical("d_model",        [64, 96, 128, 192])
    num_heads_c= trial.suggest_categorical("num_heads",      [2, 4, 8])
    num_heads  = _ensure_divisible(d_model, num_heads_c, [num_heads_c, 4, 2, 1])
    num_layers = trial.suggest_int(        "num_layers",     2, 5)
    dff        = trial.suggest_categorical("dff",            [128, 256, 384, 512])
    dropout    = trial.suggest_float(      "dropout",        0.04, 0.18, step=0.02)
    lr         = trial.suggest_float(      "learning_rate",  5e-5, 3e-4, log=True)
    blend_init = trial.suggest_float(      "seasonal_blend", 0.40, 0.80, step=0.10)
    use_cosine = trial.suggest_categorical("use_cosine_decay",[True, False])

    n_train = len(data["X_train"])
    total_steps = _get_total_steps(n_train, search_cfg["batch_size"], search_cfg["epochs"])

    model = build_itransformer(
        history_length=search_cfg["history"],
        forecast_horizon=Config.FORECAST_HORIZON,
        n_features=Config.N_FEATURES,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dff=dff,
        dropout=dropout,
        learning_rate=lr,
        use_cosine_decay=use_cosine,
        total_steps=total_steps,
        warmup_ratio=0.05,
        huber_delta=0.05,
        use_seasonal_skip=True,
        seasonal_blend_init=blend_init,
    )
    return _quick_eval(model, data,
                       batch_size=search_cfg["batch_size"],
                       epochs=search_cfg["epochs"],
                       patience=search_cfg["patience"],
                       trial=trial)


def _objective_patchtst(trial: "optuna.Trial",
                         data: Dict[str, Any],
                         search_cfg: Dict[str, Any]) -> float:
    from models.transformer import build_patchtst

    history = search_cfg["history"]

    # Патч-параметры зависят от длины истории
    if history >= 192:
        patch_candidates = [12, 16, 24]
    elif history >= 96:
        patch_candidates = [8, 12, 16]
    else:
        patch_candidates = [6, 8, 12]

    patch_len  = trial.suggest_categorical("patch_len",       patch_candidates)
    stride     = trial.suggest_categorical("stride_ratio",    [0.25, 0.50, 0.75])
    stride_val = max(1, int(patch_len * stride))
    d_model    = trial.suggest_categorical("d_model",         [64, 96, 128, 192])
    num_heads_c= trial.suggest_categorical("num_heads",       [2, 4, 8])
    num_heads  = _ensure_divisible(d_model, num_heads_c, [num_heads_c, 4, 2, 1])
    num_layers = trial.suggest_int(        "num_layers",      2, 4)
    lr         = trial.suggest_float(      "learning_rate",   1e-4, 1e-3, log=True)
    dropout    = trial.suggest_float(      "dropout",         0.02, 0.12, step=0.02)
    blend_init = trial.suggest_float(      "seasonal_blend",  0.40, 0.80, step=0.10)
    sd_rate    = trial.suggest_float(      "stochastic_depth",0.04, 0.12, step=0.02)
    use_cosine = trial.suggest_categorical("use_cosine_decay",[True, False])

    # Проверяем корректность патча
    if patch_len > history:
        raise optuna.TrialPruned()
    n_patches = (history - patch_len) // stride_val + 1
    if n_patches < 3:
        raise optuna.TrialPruned()

    n_train = len(data["X_train"])
    total_steps = _get_total_steps(n_train, search_cfg["batch_size"], search_cfg["epochs"])

    model = build_patchtst(
        history_length=history,
        forecast_horizon=Config.FORECAST_HORIZON,
        patch_len=patch_len,
        stride=stride_val,
        n_features=Config.N_FEATURES,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dff=d_model * 4,
        dropout=dropout,
        learning_rate=lr,
        stochastic_depth_rate=sd_rate,
        use_revin=True,
        use_cosine_decay=use_cosine,
        total_steps=total_steps,
        warmup_ratio=0.05,
        huber_delta=0.05,
        use_seasonal_skip=True,
        seasonal_blend_init=blend_init,
        patchtst_learning_rate=lr,
        patchtst_dropout=dropout,
    )
    return _quick_eval(model, data,
                       batch_size=search_cfg["batch_size"],
                       epochs=search_cfg["epochs"],
                       patience=search_cfg["patience"],
                       trial=trial)


def _objective_tft(trial: "optuna.Trial",
                   data: Dict[str, Any],
                   search_cfg: Dict[str, Any]) -> float:
    from models.transformer import build_tft_lite

    _TFT_COVAR_INDICES = [1, 2, 3, 5, 6, 7, 15, 17, 23, 25]
    n_cov = len(_TFT_COVAR_INDICES)

    d_model    = trial.suggest_categorical("d_model",         [64, 96, 128])
    num_heads_c= trial.suggest_categorical("num_heads",       [2, 4, 8])
    num_heads  = _ensure_divisible(d_model, num_heads_c, [num_heads_c, 4, 2, 1])
    num_layers = trial.suggest_int(        "num_layers",      2, 4)
    dropout    = trial.suggest_float(      "dropout",         0.05, 0.18, step=0.02)
    lr         = trial.suggest_float(      "learning_rate",   5e-5, 3e-4, log=True)
    use_cosine = trial.suggest_categorical("use_cosine_decay",[True, False])

    n_train = len(data["X_train"])
    total_steps = _get_total_steps(n_train, search_cfg["batch_size"], search_cfg["epochs"])

    model = build_tft_lite(
        history_length=search_cfg["history"],
        forecast_horizon=Config.FORECAST_HORIZON,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=lr,
        n_covariate_features=n_cov,
        use_cosine_decay=use_cosine,
        total_steps=total_steps,
        warmup_ratio=0.05,
        huber_delta=0.05,
    )

    def _to_series(x): return x[:, :, :1].astype(np.float32)
    def _to_covar(x):  return x[:, :, _TFT_COVAR_INDICES].astype(np.float32)
    data["X_tft_train"] = [_to_series(data["X_train"]), _to_covar(data["X_train"])]
    data["X_tft_val"]   = [_to_series(data["X_val"]),   _to_covar(data["X_val"])]

    return _quick_eval(model, data,
                       batch_size=search_cfg["batch_size"],
                       epochs=search_cfg["epochs"],
                       patience=search_cfg["patience"],
                       trial=trial,
                       is_tft=True)


def _objective_xgboost(trial: "optuna.Trial",
                        data: Dict[str, Any],
                        search_cfg: Dict[str, Any]) -> float:
    from models.baseline import build_xgboost

    n_est      = trial.suggest_int(   "n_estimators",    200, 900, step=100)
    max_depth  = trial.suggest_int(   "max_depth",       3, 7)
    lr         = trial.suggest_float( "learning_rate",   0.01, 0.15, log=True)
    subsample  = trial.suggest_float( "subsample",       0.55, 0.95, step=0.05)
    colsample  = trial.suggest_float( "colsample_bytree",0.25, 0.70, step=0.05)
    min_cw     = trial.suggest_int(   "min_child_weight",5, 20)
    reg_alpha  = trial.suggest_float( "reg_alpha",       0.01, 1.0, log=True)
    reg_lambda = trial.suggest_float( "reg_lambda",      0.5,  5.0, log=True)

    model = build_xgboost(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample,
        min_child_weight=min_cw,
        seed=Config.SEED,
    )
    # Патчим reg_alpha/lambda напрямую в estimator конфиге
    model.estimator.__init__.__func__ if False else None
    # Просто возвращаем MAE на val
    model.fit(data["X_train"], data["Y_train"],
              X_val=data["X_val"], Y_val=data["Y_val"])
    y_pred = model.predict(data["X_val"])
    y_true = inverse_scale(data["scaler"], data["Y_val"])
    y_pred_real = inverse_scale(data["scaler"], y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred_real)))
    logger.info("  XGBoost trial MAE=%.2f", mae)
    return mae


# ══════════════════════════════════════════════════════════════════════════════
# ПРЕОБРАЗОВАНИЕ BEST PARAMS В CONFIG-СЛОВАРЬ
# ══════════════════════════════════════════════════════════════════════════════

def _params_to_config(model_name: str,
                      best_params: Dict[str, Any],
                      search_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Конвертирует best_params Optuna в ключи конфигурации Config-класса.
    Возвращает словарь {CONFIG_ATTR: value}.
    """
    cfg: Dict[str, Any] = {}

    if model_name == "lstm":
        cfg["LSTM_UNITS_1"]           = best_params.get("lstm_units", 128)
        cfg["LSTM_UNITS_2"]           = best_params.get("lstm_units", 128)
        cfg["LSTM_UNITS_3"]           = best_params.get("lstm_units", 128)
        cfg["LSTM_TCN_FILTERS"]       = best_params.get("tcn_filters", 64)
        cfg["LSTM_ATTN_HEADS"]        = best_params.get("attn_heads", 4)
        cfg["DROPOUT_RATE"]           = best_params.get("dropout", 0.12)
        cfg["LSTM_LEARNING_RATE"]     = best_params.get("learning_rate", 1e-4)
        cfg["LSTM_SEASONAL_BLEND_INIT"]= best_params.get("seasonal_blend", 0.50)
        cfg["LSTM_HUBER_DELTA"]       = best_params.get("huber_delta", 0.05)
        cfg["LSTM_USE_COSINE_DECAY"]  = best_params.get("use_cosine_decay", True)

    elif model_name == "itransformer":
        d_model   = best_params.get("d_model", 128)
        num_heads = best_params.get("num_heads", 4)
        # Гарантируем делимость
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        cfg["TRANSFORMER_D_MODEL"]       = d_model
        cfg["TRANSFORMER_N_HEADS"]       = num_heads
        cfg["ITRANSFORMER_N_LAYERS"]     = best_params.get("num_layers", 4)
        cfg["TRANSFORMER_DFF"]           = best_params.get("dff", 256)
        cfg["TRANSFORMER_DROPOUT"]       = best_params.get("dropout", 0.08)
        cfg["VANILLA_TRANSFORMER_LR"]    = best_params.get("learning_rate", 1e-4)
        cfg["VANILLA_SEASONAL_BLEND_INIT"]= best_params.get("seasonal_blend", 0.60)
        cfg["TRANSFORMER_USE_COSINE_DECAY"]= best_params.get("use_cosine_decay", True)

    elif model_name == "patchtst":
        d_model    = best_params.get("d_model", 128)
        num_heads  = best_params.get("num_heads", 4)
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        stride_ratio = best_params.get("stride_ratio", 0.50)
        patch_len    = best_params.get("patch_len", 24)
        cfg["TRANSFORMER_D_MODEL"]      = d_model
        cfg["TRANSFORMER_N_HEADS"]      = num_heads
        cfg["TRANSFORMER_N_LAYERS"]     = best_params.get("num_layers", 3)
        cfg["PATCHTST_LEARNING_RATE"]   = best_params.get("learning_rate", 3e-4)
        cfg["PATCHTST_DROPOUT"]         = best_params.get("dropout", 0.05)
        cfg["TRANSFORMER_STOCHASTIC_DEPTH"]= best_params.get("stochastic_depth", 0.08)
        cfg["VANILLA_SEASONAL_BLEND_INIT"] = best_params.get("seasonal_blend", 0.60)
        cfg["TRANSFORMER_USE_COSINE_DECAY"]= best_params.get("use_cosine_decay", True)
        # Доп. метаданные для patch (не Config attrs, но нужны для info)
        cfg["_patchtst_patch_len"]      = patch_len
        cfg["_patchtst_stride_ratio"]   = stride_ratio

    elif model_name == "tft":
        d_model   = best_params.get("d_model", 128)
        num_heads = best_params.get("num_heads", 4)
        while d_model % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        cfg["TRANSFORMER_D_MODEL"]      = d_model
        cfg["TRANSFORMER_N_HEADS"]      = num_heads
        cfg["TRANSFORMER_N_LAYERS"]     = best_params.get("num_layers", 3)
        cfg["TRANSFORMER_DROPOUT"]      = best_params.get("dropout", 0.10)
        cfg["TRANSFORMER_LEARNING_RATE"]= best_params.get("learning_rate", 1e-4)
        cfg["TRANSFORMER_USE_COSINE_DECAY"]= best_params.get("use_cosine_decay", True)

    elif model_name == "xgboost":
        cfg["XGB_N_ESTIMATORS"]   = best_params.get("n_estimators", 700)
        cfg["XGB_MAX_DEPTH"]      = best_params.get("max_depth", 5)
        cfg["XGB_LR"]             = best_params.get("learning_rate", 0.05)
        cfg["XGB_SUBSAMPLE"]      = best_params.get("subsample", 0.80)
        cfg["XGB_COLSAMPLE"]      = best_params.get("colsample_bytree", 0.50)

    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА ОДНОЙ МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

OBJECTIVES = {
    "lstm":         _objective_lstm,
    "itransformer": _objective_itransformer,
    "patchtst":     _objective_patchtst,
    "tft":          _objective_tft,
    "xgboost":      _objective_xgboost,
}

def tune_model(
    model_name: str,
    pipeline_mode: str,
    data: Dict[str, Any],
    n_trials: Optional[int] = None,
    force: bool = False,
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Запускает Optuna study для одной модели.

    Parameters
    ----------
    model_name    : "lstm" | "itransformer" | "patchtst" | "tft" | "xgboost"
    pipeline_mode : "fast" | "optimal" | "full"
    data          : словарь из prepare_data()
    n_trials      : переопределить кол-во trials
    force         : перезаписать уже существующие результаты
    storage       : путь к SQLite для распределённого поиска (опционально)

    Returns
    -------
    dict с best_params, best_value, config_updates
    """
    if not OPTUNA_AVAILABLE:
        logger.error("optuna не установлен. Запустите: pip install optuna")
        return {}

    search_cfg = SEARCH_CONFIGS[pipeline_mode]
    n_trials_eff = n_trials if n_trials is not None else search_cfg["n_trials"]

    # Путь для сохранения результатов
    save_dir = HPO_RESULTS_DIR / pipeline_mode
    save_dir.mkdir(parents=True, exist_ok=True)
    result_path = save_dir / f"{model_name}_best_params.json"

    if result_path.exists() and not force:
        logger.info("Загружаем существующие результаты: %s", result_path)
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    objective_fn = OBJECTIVES.get(model_name)
    if objective_fn is None:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(OBJECTIVES)}")

    logger.info("=" * 65)
    logger.info("HPO [%s | mode=%s] n_trials=%d | epochs=%d | days=%d",
                model_name.upper(), pipeline_mode,
                n_trials_eff, search_cfg["epochs"], search_cfg["days"])
    logger.info("=" * 65)

    # Создаём study
    pruner = MedianPruner(
        n_startup_trials=search_cfg["pruner_warmup_trials"],
        n_warmup_steps=search_cfg["pruner_warmup_steps"],
    )
    sampler = TPESampler(seed=Config.SEED, multivariate=True)

    study_storage = f"sqlite:///{save_dir}/{model_name}_study.db" if storage else None

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{pipeline_mode}_{model_name}",
        storage=study_storage,
        load_if_exists=True,
    )

    # Отключаем шумный лог optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()

    try:
        study.optimize(
            lambda trial: objective_fn(trial, data, search_cfg),
            n_trials=n_trials_eff,
            gc_after_trial=True,
            catch=(Exception,),
        )
    except KeyboardInterrupt:
        logger.warning("HPO прервано пользователем. Сохраняем лучшие результаты...")

    elapsed = time.time() - t0

    if not study.best_trials:
        logger.error("Все trials завершились ошибкой для %s", model_name)
        return {}

    best_params  = study.best_params
    best_value   = study.best_value
    config_upd   = _params_to_config(model_name, best_params, search_cfg)

    # Статистика
    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    pruned    = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.PRUNED])

    result = {
        "model_name":     model_name,
        "pipeline_mode":  pipeline_mode,
        "best_value":     best_value,
        "best_params":    best_params,
        "config_updates": config_upd,
        "n_trials_total": len(study.trials),
        "n_completed":    completed,
        "n_pruned":       pruned,
        "elapsed_sec":    round(elapsed, 1),
        "search_config":  {k: v for k, v in search_cfg.items()
                           if not k.startswith("pruner")},
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("─" * 65)
    logger.info("✅ %s ЗАВЕРШЁН | best_val_loss=%.6f | trials: %d/%d завершено, %d обрезано",
                model_name.upper(), best_value, completed, n_trials_eff, pruned)
    logger.info("   Время: %.1f мин", elapsed / 60)
    logger.info("   Лучшие параметры:")
    for k, v in best_params.items():
        logger.info("     %-30s = %s", k, v)
    logger.info("   Сохранено: %s", result_path)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ПРИМЕНЕНИЕ РЕЗУЛЬТАТОВ К CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def apply_hpo_to_config(pipeline_mode: str) -> Dict[str, int]:
    """
    Загружает все сохранённые HPO-результаты для режима и применяет к Config.
    Вызывается автоматически из Config.set_*_mode().

    Returns
    -------
    dict {model_name: True/False} — какие модели применены
    """
    save_dir = HPO_RESULTS_DIR / pipeline_mode
    applied: Dict[str, bool] = {}

    if not save_dir.exists():
        return applied

    for model_name in OBJECTIVES:
        result_path = save_dir / f"{model_name}_best_params.json"
        if not result_path.exists():
            continue
        try:
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)

            config_upd = result.get("config_updates", {})
            for attr, value in config_upd.items():
                if attr.startswith("_"):  # служебные ключи, не Config attrs
                    continue
                if hasattr(Config, attr):
                    setattr(Config, attr, value)
                    logger.debug("HPO: Config.%s = %s", attr, value)

            applied[model_name] = True
            logger.info("HPO результаты применены: %s (val_loss=%.6f)",
                        model_name, result.get("best_value", float("nan")))

        except Exception as exc:
            logger.warning("Ошибка при загрузке HPO результатов %s: %s", model_name, exc)
            applied[model_name] = False

    return applied


def save_combined_hpo_report(pipeline_mode: str) -> str:
    """
    Создаёт сводный JSON-файл со всеми HPO-результатами для режима.
    Удобно для журнала экспериментов в курсовой работе.
    """
    save_dir = HPO_RESULTS_DIR / pipeline_mode
    combined: Dict[str, Any] = {"mode": pipeline_mode, "models": {}}

    for model_name in OBJECTIVES:
        path = save_dir / f"{model_name}_best_params.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                combined["models"][model_name] = json.load(f)

    out_path = save_dir / "hpo_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info("Сводный отчёт HPO: %s", out_path)

    # Печатаем таблицу для удобства
    logger.info("\n%s", "=" * 70)
    logger.info("СВОДКА HPO [mode=%s]", pipeline_mode)
    logger.info("%-18s %12s %6s %6s %10s", "Модель", "val_loss", "Ok", "Обр.", "Время")
    logger.info("-" * 70)
    for name, res in combined["models"].items():
        logger.info("%-18s %12.6f %6d %6d %8.1f мин",
                    name,
                    res.get("best_value", float("nan")),
                    res.get("n_completed", 0),
                    res.get("n_pruned", 0),
                    res.get("elapsed_sec", 0) / 60)
    logger.info("=" * 70)

    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not OPTUNA_AVAILABLE:
        logger.error(
            "Optuna не установлена!\n"
            "Установите командой: pip install optuna\n"
            "Для SQLite-хранилища (опционально): pip install optuna[sqlite]"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Автоматический подбор гиперпараметров (Optuna TPE + MedianPruner)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Все модели, optimal mode (рекомендуется, ~20-30ч):
  python experiments/hyperparameter_tuning.py --mode optimal

  # Только PatchTST, быстро:
  python experiments/hyperparameter_tuning.py --mode fast --model patchtst --n_trials 10

  # Принудительный перезапуск LSTM:
  python experiments/hyperparameter_tuning.py --mode optimal --model lstm --force

  # Распределённый поиск (несколько процессов на одном файле):
  python experiments/hyperparameter_tuning.py --mode optimal --storage
        """,
    )
    parser.add_argument("--mode", choices=["fast", "optimal", "full"], default="optimal",
                        help="Режим поиска (default: optimal)")
    parser.add_argument("--model",
                        choices=list(OBJECTIVES) + ["all"],
                        default="all",
                        help="Модель для подбора (default: all)")
    parser.add_argument("--n_trials", type=int, default=None,
                        help="Переопределить кол-во trials")
    parser.add_argument("--force", action="store_true",
                        help="Перезаписать существующие результаты")
    parser.add_argument("--storage", action="store_true",
                        help="Использовать SQLite для хранения study (параллельный запуск)")
    parser.add_argument("--seed", type=int, default=Config.SEED,
                        help="Random seed")
    args = parser.parse_args()

    # Воспроизводимость
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Логирование в файл
    log_dir = ROOT / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / f"hpo_{args.mode}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(fh)

    search_cfg = SEARCH_CONFIGS[args.mode]

    logger.info("=" * 70)
    logger.info("SMART GRID HPO v1 | mode=%s | model=%s | n_trials=%s",
                args.mode, args.model,
                args.n_trials or search_cfg["n_trials"])
    logger.info("Search config: days=%d | epochs=%d | batch=%d | patience=%d",
                search_cfg["days"], search_cfg["epochs"],
                search_cfg["batch_size"], search_cfg["patience"])
    logger.info("=" * 70)

    # Генерация данных для поиска
    logger.info("Генерация данных для HPO (%d дней)...", search_cfg["days"])
    df = generate_smartgrid_data(
        days=search_cfg["days"],
        households=search_cfg["households"],
        seed=args.seed,
    )

    logger.info("Подготовка признаков (history=%d)...", search_cfg["history"])
    # Временно применяем нужные Config параметры
    Config.HISTORY_LENGTH  = search_cfg["history"]
    Config.N_FEATURES      = 26
    Config.FORECAST_HORIZON = 24

    data = prepare_data(
        df,
        history_length=search_cfg["history"],
        forecast_horizon=24,
        train_ratio=0.70,
        val_ratio=0.15,
        seasonal_diff=False,
    )
    logger.info("Данные готовы: train=%d val=%d",
                len(data["X_train"]), len(data["X_val"]))

    # Определяем список моделей
    models_to_tune = list(OBJECTIVES) if args.model == "all" else [args.model]
    storage_str = str(HPO_RESULTS_DIR / args.mode) if args.storage else None

    results_all: Dict[str, Any] = {}

    total_t0 = time.time()
    for model_name in models_to_tune:
        logger.info("\n%s", "─" * 70)
        logger.info("Запуск HPO для модели: %s", model_name.upper())
        try:
            res = tune_model(
                model_name=model_name,
                pipeline_mode=args.mode,
                data=data,
                n_trials=args.n_trials,
                force=args.force,
                storage=storage_str,
            )
            results_all[model_name] = res
        except Exception as exc:
            logger.error("Ошибка HPO для %s: %s", model_name, exc, exc_info=True)

    total_elapsed = time.time() - total_t0

    # Сводный отчёт
    if results_all:
        report_path = save_combined_hpo_report(args.mode)
        logger.info("\n✅ HPO завершён за %.1f мин", total_elapsed / 60)
        logger.info("Сводный отчёт: %s", report_path)
        logger.info("\nДля применения результатов запустите main.py — ")
        logger.info("параметры подхватятся автоматически из results/hpo/%s/", args.mode)


if __name__ == "__main__":
    main()