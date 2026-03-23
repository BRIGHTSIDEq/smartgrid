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
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend — НЕТ GUI, сразу в файл
import matplotlib.pyplot as plt
import tensorflow as tf

from utils.metrics import compute_all_metrics
from data.preprocessing import inverse_scale

logger = logging.getLogger("smart_grid.models.trainer")


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
        # ReduceLROnPlateau несовместим с LearningRateSchedule (CosineDecay).
        # TF2.15+: нельзя использовать hasattr(...,"__call__") — float LR тоже callable.
        #          нельзя использовать `getattr() or getattr()` — bool(KerasVariable) crash.
        # Единственный надёжный способ — inspect конфиг оптимизатора или try/except.
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
        """
        Для sklearn/xgboost моделей — делегируем в wrapper.fit().
        FlattenWrapper и LagFeaturesWrapper имеют единый интерфейс fit(X, Y).
        Никакой XGBoost-специфичной логики здесь нет — она инкапсулирована
        в соответствующих wrapper-классах из baseline.py.
        """
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

        Использование:
            mean, std = trainer.mc_predict(X_test, n_samples=50)

        Требует модель, построенную с mc_dropout=True (v7: параметр сохраняется
        в логах, но training=True надо передавать явно — что здесь и делается).

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
        )  # (n_samples, N, horizon)
        return preds.mean(axis=0), preds.std(axis=0)

    def evaluate(
        self,
        data: Dict[str, Any],
        split: str = "test",
    ) -> Dict[str, float]:
        X = data[f"X_{split}"]
        Y_true_scaled = data[f"Y_{split}"]
        scaler = data["scaler"]

        Y_pred_scaled = self.predict(X)
        Y_true = inverse_scale(scaler, Y_true_scaled)
        Y_pred = inverse_scale(scaler, Y_pred_scaled)

        return compute_all_metrics(Y_true, Y_pred, model_name=self.model_name)

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
            # ── ИСПРАВЛЕНИЕ v3 ───────────────────────────────────────────────
            # БЫЛО: TemporalAttentionBlock отсутствовал в custom_objects.
            # tf.keras.models.load_model() не может восстановить граф LSTM-модели
            # без явной регистрации кастомных слоёв → ValueError при загрузке.
            # СТАЛО: импортируем и передаём TemporalAttentionBlock явно.
            from models.lstm import TemporalAttentionBlock
            # ────────────────────────────────────────────────────────────────
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "PreLNEncoderBlock":    PreLNEncoderBlock,
                    "SinusoidalPE":         SinusoidalPE,
                    "Time2Vec":             Time2Vec,
                    "GatedResidualNetwork": GatedResidualNetwork,
                    "TemporalAttentionBlock": TemporalAttentionBlock,  # v3: добавлен
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
    logger.info("%-22s %8s %8s %8s %7s", "Модель", "MAE", "RMSE", "MAPE%", "R2")
    logger.info("%s", "-" * 58)

    for trainer in trainers:
        try:
            m = trainer.evaluate(data, split=split)
            results[trainer.model_name] = m
            logger.info(
                "%-22s %8.2f %8.2f %7.2f%% %6.4f",
                trainer.model_name, m["MAE"], m["RMSE"], m["MAPE"], m["R2"],
            )
        except Exception as exc:
            logger.error("Ошибка оценки %s: %s", trainer.model_name, exc)

    if results:
        best = min(results, key=lambda k: results[k]["MAE"])
        logger.info("%s", "-" * 58)
        logger.info("Лучшая модель по MAE: %s", best)

    return results
