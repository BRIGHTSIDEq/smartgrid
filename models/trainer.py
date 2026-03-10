# -*- coding: utf-8 -*-
"""
models/trainer.py — Универсальный тренер моделей.

ИСПРАВЛЕНИЯ v2:
  1. EarlyStopping: добавлен min_delta=Config.MIN_DELTA
     Причина: без min_delta остановка на изменении 1e-7 (шум оптимизатора),
     что приводит к преждевременной остановке (31 эпоха вместо 100+).

  2. XGBoost: передаём eval_set для early_stopping_rounds
     Причина: XGBRegressor с early_stopping_rounds требует eval_set
     при fit(), иначе RuntimeError.

  3. plt.show() → plt.savefig() без show() в headless-режиме
     Причина: plt.show() блокирует выполнение на серверах без дисплея.
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
        callbacks: List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=min_delta,        # ← ИСПРАВЛЕНИЕ: не останавливаться на шуме
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_delta=min_delta,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

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
        Для XGBoost с early_stopping_rounds нужен eval_set.
        Для обычных sklearn-моделей — просто fit().
        """
        try:
            import xgboost as xgb
            model_inner = getattr(self.model, "estimator", self.model)

            if isinstance(model_inner, xgb.XGBRegressor):
                X_tr = data["X_train"].reshape(data["X_train"].shape[0], -1)
                Y_tr = data["Y_train"]
                X_val = data["X_val"].reshape(data["X_val"].shape[0], -1)
                Y_val = data["Y_val"]
                model_inner.fit(
                    X_tr, Y_tr,
                    eval_set=[(X_val, Y_val)],
                    verbose=False,
                )
                logger.info("XGBoost обучен")
            else:
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
                PreLNEncoderBlock, SinusoidalPE, Time2Vec, GatedResidualNetwork
            )
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "PreLNEncoderBlock": PreLNEncoderBlock,
                    "SinusoidalPE": SinusoidalPE,
                    "Time2Vec": Time2Vec,
                    "GatedResidualNetwork": GatedResidualNetwork,
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
    logger.info("%-22s %8s %8s %8s %7s", "Модель", "MAE", "RMSE", "MAPE%", "R²")
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
        logger.info("🏆 Лучшая модель по MAE: %s", best)

    return results
