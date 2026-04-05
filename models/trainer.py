# -*- coding: utf-8 -*-
"""
models/trainer.py — Универсальный тренер моделей.

ИСПРАВЛЕНИЯ v2:
  1. EarlyStopping: добавлен min_delta=Config.MIN_DELTA
  2. XGBoost: передаём eval_set для early_stopping_rounds
  3. plt.show() → plt.savefig() без show() в headless-режиме

ИСПРАВЛЕНИЯ v3 (баги из code review):
  4. load_keras: добавлен TemporalAttentionBlock в custom_objects.
     БЫЛО: отсутствовал → ValueError при загрузке любой LSTM-модели.
     СТАЛО: импортируется из models.lstm и передаётся в custom_objects.

ИСПРАВЛЕНИЯ v4 (диагностика остатков):
  5. evaluate(): добавлен вызов diagnose_residuals() для детальной диагностики
     автокорреляции (DW, ACF на лагах 1/24/48/168) и тяжёлых хвостов.
     Выводит конкретные рекомендации в лог при DW < 1.5 или ACF(24) > 0.1.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.metrics import compute_all_metrics
from data.preprocessing import inverse_scale

logger = logging.getLogger("smart_grid.models.trainer")


# ══════════════════════════════════════════════════════════════════════════════
# ДИАГНОСТИКА ОСТАТКОВ (v4)
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> Dict[str, Any]:
    """
    Расширенная диагностика остатков: DW, ACF на лагах 24/48/168, кurtosis.

    Вызывается автоматически из evaluate() для каждой модели.
    Логирует рекомендации при обнаружении проблем.

    Returns
    -------
    dict: {"DW", "ACF_1", "ACF_24", "ACF_48", "ACF_168",
           "skewness", "kurtosis", "recommendations"}
    """
    residuals = y_true.flatten() - y_pred.flatten()
    n = len(residuals)

    # ── Durbin-Watson ────────────────────────────────────────────────────────
    diff_resid = np.diff(residuals)
    dw = float(np.sum(diff_resid ** 2) / (np.sum(residuals ** 2) + 1e-12))

    # ── ACF на ключевых лагах ────────────────────────────────────────────────
    def acf_lag(series: np.ndarray, lag: int) -> float:
        if len(series) <= lag:
            return float("nan")
        s = series - series.mean()
        var = np.var(s) + 1e-12
        return float(np.dot(s[lag:], s[:-lag]) / (len(s) * var))

    acf_1   = acf_lag(residuals, 1)
    acf_24  = acf_lag(residuals, 24)
    acf_48  = acf_lag(residuals, 48)
    acf_168 = acf_lag(residuals, 168)

    # ── Kurtosis и Skewness ───────────────────────────────────────────────────
    std = np.std(residuals) + 1e-12
    kurt = float(np.mean((residuals - residuals.mean()) ** 4) / std ** 4)
    skew = float(np.mean((residuals - residuals.mean()) ** 3) / std ** 3)

    # ── Автоматические рекомендации ───────────────────────────────────────────
    recommendations: List[str] = []
    if dw < 1.5:
        recommendations.append(
            f"DW={dw:.3f} < 1.5: сильная автокорреляция. "
            "→ Добавить load_lag_24h, load_lag_168h как ковариаты (preprocessing.py)."
        )
    if abs(acf_24) > 0.10:
        recommendations.append(
            f"ACF(24)={acf_24:.3f} > 0.10: суточная компонента не устранена. "
            "→ Проверить наличие load_lag_24h в feature matrix."
        )
    if abs(acf_168) > 0.10:
        recommendations.append(
            f"ACF(168)={acf_168:.3f} > 0.10: недельная компонента не устранена. "
            "→ Добавить load_lag_168h."
        )
    if kurt > 5:
        recommendations.append(
            f"Kurtosis={kurt:.2f} > 5: очень тяжёлые хвосты. "
            "→ Заменить Huber(delta=0.1) на Huber(delta=0.05) или QuantileLoss(q=0.9)."
        )
    elif kurt > 3:
        recommendations.append(
            f"Kurtosis={kurt:.2f} > 3: тяжёлые хвосты. "
            "→ Убедиться что используется Huber loss, не MSE (lstm.py)."
        )

    log = logging.getLogger("smart_grid.models.trainer")
    log.info("─ Диагностика остатков: %s (n=%d) ─────────────────────────",
             model_name or "?", n)
    log.info("  DW=%.4f  ACF(1)=%.4f  ACF(24)=%.4f  ACF(48)=%.4f  ACF(168)=%.4f",
             dw, acf_1, acf_24, acf_48, acf_168)
    log.info("  Skewness=%.4f  Kurtosis=%.4f", skew, kurt)
    if recommendations:
        log.info("  ⚠️  Рекомендации:")
        for rec in recommendations:
            log.info("    → %s", rec)
    else:
        log.info("  ✅ Остатки в норме (DW≥1.5, |ACF(24)|<0.10, Kurt<3).")

    return {
        "model":           model_name,
        "n":               n,
        "DW":              round(dw, 4),
        "ACF_1":           round(acf_1, 4),
        "ACF_24":          round(acf_24, 4),
        "ACF_48":          round(acf_48, 4),
        "ACF_168":         round(acf_168, 4),
        "skewness":        round(skew, 4),
        "kurtosis":        round(kurt, 4),
        "recommendations": recommendations,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Универсальный тренер: Keras + sklearn/xgboost через единый интерфейс.
    """

    def __init__(
        self,
        model: Any,
        model_name: str,
        models_dir: str = "results/models",
        plots_dir: str = "results/plots",
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        self.history: Optional[tf.keras.callbacks.History] = None
        self.train_time: float = 0.0

    def train(
        self,
        data: Dict[str, Any],
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 25,
        lr_patience: int = 10,
        lr_factor: float = 0.5,
        min_delta: float = 1e-5,
    ) -> "ModelTrainer":
        logger.info("=" * 60)
        logger.info("Обучение модели: %s", self.model_name)
        logger.info("=" * 60)

        t0 = time.time()
        if isinstance(self.model, tf.keras.Model):
            self._train_keras(data, epochs, batch_size, patience,
                              lr_patience, lr_factor, min_delta)
        else:
            self._train_sklearn(data)

        self.train_time = time.time() - t0
        logger.info("%s обучена за %.1f сек", self.model_name, self.train_time)
        return self

    def _train_keras(
        self,
        data: Dict[str, Any],
        epochs: int,
        batch_size: int,
        patience: int,
        lr_patience: int,
        lr_factor: float,
        min_delta: float,
    ) -> None:
        _has_schedule = False
        try:
            _opt = self.model.optimizer
            _lr_cfg = _opt.get_config().get("learning_rate", None)
            if isinstance(_lr_cfg, dict) and "class_name" in _lr_cfg:
                _has_schedule = True
            elif isinstance(_lr_cfg, tf.keras.optimizers.schedules.LearningRateSchedule):
                _has_schedule = True
        except Exception:
            _has_schedule = False

        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
        ]
        if not _has_schedule:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_delta=min_delta,
                min_lr=1e-6,
                verbose=1,
            ))
        else:
            logger.info("%s: LR schedule обнаружен → ReduceLROnPlateau отключён",
                        self.model_name)

        try:
            self.history = self.model.fit(
                data["X_train"], data["Y_train"],
                validation_data=(data["X_val"], data["Y_val"]),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            actual_epochs = len(self.history.history["loss"])
            logger.info("%s: обучение завершено за %d эпох",
                        self.model_name, actual_epochs)
        except Exception as exc:
            logger.error("Ошибка при обучении %s: %s", self.model_name, exc)
            raise

    def _train_sklearn(self, data: Dict[str, Any]) -> None:
        try:
            self.model.fit(
                data["X_train"], data["Y_train"],
                X_val=data.get("X_val"),
                Y_val=data.get("Y_val"),
            )
        except Exception as exc:
            logger.error("Ошибка при обучении %s: %s", self.model_name, exc)
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self.model, tf.keras.Model):
            return self.model.predict(X, verbose=0)
        return self.model.predict(X)

    def mc_predict(
        self,
        X: np.ndarray,
        n_samples: int = 50,
    ) -> tuple:
        """
        Monte Carlo Dropout: n_samples прогонов с training=True.

        Returns
        -------
        mean : np.ndarray shape=(N, horizon)
        std  : np.ndarray shape=(N, horizon)
        """
        if not isinstance(self.model, tf.keras.Model):
            raise TypeError("mc_predict доступен только для Keras-моделей")

        preds = np.stack(
            [self.model(X, training=True).numpy() for _ in range(n_samples)],
            axis=0,
        )
        return preds.mean(axis=0), preds.std(axis=0)

    def evaluate(
        self,
        data: Dict[str, Any],
        split: str = "test",
        run_residual_diagnostics: bool = True,
    ) -> Dict[str, float]:
        """
        Оценка модели на split-е с опциональной диагностикой остатков.

        Parameters
        ----------
        run_residual_diagnostics : bool
            Если True — запускает diagnose_residuals() и выводит
            DW/ACF/Kurt в лог с конкретными рекомендациями.
            По умолчанию включено для всех моделей.
        """
        X = data[f"X_{split}"]
        Y_true_scaled = data[f"Y_{split}"]
        scaler = data["scaler"]

        Y_pred_scaled = self.predict(X)
        Y_true = inverse_scale(scaler, Y_true_scaled)
        Y_pred = inverse_scale(scaler, Y_pred_scaled)

        metrics = compute_all_metrics(Y_true, Y_pred, model_name=self.model_name)

        # ── v4: диагностика остатков ──────────────────────────────────────────
        if run_residual_diagnostics:
            diag = diagnose_residuals(Y_true, Y_pred, model_name=self.model_name)
            metrics["DW"]       = diag["DW"]
            metrics["ACF_24"]   = diag["ACF_24"]
            metrics["ACF_168"]  = diag["ACF_168"]
            metrics["kurtosis"] = diag["kurtosis"]
        # ────────────────────────────────────────────────────────────────────

        return metrics

    def save(self) -> str:
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, self.model_name)
        try:
            if isinstance(self.model, tf.keras.Model):
                save_path = path + ".keras"
                self.model.save(save_path)
                logger.info("Keras-модель сохранена: %s", save_path)
                return save_path
            else:
                import pickle
                save_path = path + ".pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(self.model, f)
                logger.info("Sklearn-модель сохранена: %s", save_path)
                return save_path
        except Exception as exc:
            logger.error("Ошибка сохранения %s: %s", self.model_name, exc)
            raise

    @classmethod
    def load_keras(cls, path: str, model_name: str = "") -> "ModelTrainer":
        try:
            from models.transformer import (
                PreLNEncoderBlock, SinusoidalPE, Time2Vec, GatedResidualNetwork,
            )
            from models.lstm import TemporalAttentionBlock
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "PreLNEncoderBlock":      PreLNEncoderBlock,
                    "SinusoidalPE":           SinusoidalPE,
                    "Time2Vec":               Time2Vec,
                    "GatedResidualNetwork":   GatedResidualNetwork,
                    "TemporalAttentionBlock": TemporalAttentionBlock,
                },
            )
            trainer = cls(model, model_name or os.path.basename(path))
            logger.info("Загружена модель: %s", path)
            return trainer
        except Exception as exc:
            logger.error("Ошибка загрузки %s: %s", path, exc)
            raise


# ── Сравнение тренеров ────────────────────────────────────────────────────────

def compare_trainers(
    trainers: List[ModelTrainer],
    data: Dict[str, Any],
    split: str = "test",
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    logger.info("\n%s", "=" * 60)
    logger.info("СРАВНЕНИЕ МОДЕЛЕЙ (split=%s)", split.upper())
    logger.info("%s", "=" * 60)
    logger.info("%-22s %8s %8s %8s %7s %6s %7s",
                "Модель", "MAE", "RMSE", "MAPE%", "R2", "DW", "ACF24")
    logger.info("%s", "-" * 70)

    for trainer in trainers:
        try:
            m = trainer.evaluate(data, split=split, run_residual_diagnostics=True)
            results[trainer.model_name] = m
            dw_str   = f"{m.get('DW', float('nan')):.3f}"
            acf_str  = f"{m.get('ACF_24', float('nan')):.3f}"
            logger.info(
                "%-22s %8.2f %8.2f %7.2f%% %6.4f %6s %7s",
                trainer.model_name, m["MAE"], m["RMSE"],
                m["MAPE"], m["R2"], dw_str, acf_str,
            )
        except Exception as exc:
            logger.error("Ошибка оценки %s: %s", trainer.model_name, exc)

    if results:
        best = min(results, key=lambda k: results[k]["MAE"])
        logger.info("%s", "-" * 70)
        logger.info("Лучшая модель по MAE: %s", best)

    return results
