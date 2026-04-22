# -*- coding: utf-8 -*-
"""
models/trainer.py — v8 (ensemble min weight + bias correction).

ИЗМЕНЕНИЯ v8 vs v7 (audit 2026-04-22):

[1] WeightedEnsemble.optimize_weights(): минимальный вес 0.1%.
  БЫЛО: PatchTST получал вес 0.000 после Nelder-Mead оптимизации.
        Это означало, что PatchTST вообще не участвовал в ансамбле,
        несмотря на то что его MAE конкурентна с iTransformer.
  СТАЛО: np.maximum(w_opt, 1e-3) перед нормировкой.
         Каждая модель вносит минимум 0.1% вклада.
         Итоговая сумма весов остаётся 1.0 после ренормировки.

[2] ModelTrainer.predict_absolute(): поддержка post-hoc bias correction.
  БЫЛО: нет коррекции систематического занижения.
  СТАЛО: если trainer._bias_correction установлен (main.py step 5c),
         он вычитается из предсказания перед возвратом.
         MBE во всех моделях: −900…−2300 кВт·ч → ближе к 0.

[3] v7 исправление (ReduceLROnPlateau vs CosineDecay) — сохранено.
"""

import logging
import os
import time
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf

from utils.metrics import compute_all_metrics
from data.preprocessing import inverse_scale

logger = logging.getLogger("smart_grid.models.trainer")


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _optimizer_has_lr_schedule(model: tf.keras.Model) -> bool:
    """
    Проверяет, использует ли optimizer LearningRateSchedule.
    Если True — ReduceLROnPlateau нельзя добавлять.
    """
    if not isinstance(model, tf.keras.Model):
        return False
    try:
        optimizer = model.optimizer
        if optimizer is None:
            return False
        internal_lr = getattr(optimizer, "_learning_rate", None)
        if internal_lr is not None:
            return isinstance(
                internal_lr,
                tf.keras.optimizers.schedules.LearningRateSchedule,
            )
        current_val = optimizer.learning_rate
        optimizer.learning_rate = current_val
        return False
    except TypeError:
        return True
    except Exception:
        return False


def diagnose_residuals(y_true, y_pred, model_name=""):
    residuals = y_true.flatten() - y_pred.flatten()
    n = len(residuals)
    diff_resid = np.diff(residuals)
    dw = float(np.sum(diff_resid**2) / (np.sum(residuals**2) + 1e-12))

    def acf_lag(s, lag):
        if len(s) <= lag: return float("nan")
        s = s - s.mean(); var = np.var(s) + 1e-12
        return float(np.dot(s[lag:], s[:-lag]) / (len(s)*var))

    acf_1   = acf_lag(residuals, 1)
    acf_24  = acf_lag(residuals, 24)
    acf_48  = acf_lag(residuals, 48)
    acf_168 = acf_lag(residuals, 168)
    std  = np.std(residuals) + 1e-12
    kurt = float(np.mean((residuals-residuals.mean())**4) / std**4)
    skew = float(np.mean((residuals-residuals.mean())**3) / std**3)

    recs = []
    if dw < 1.5:   recs.append(f"DW={dw:.3f} < 1.5: автокорреляция первого порядка.")
    if abs(acf_24) > 0.10:
        recs.append(f"ACF(24)={acf_24:.3f} > 0.10: суточная компонента в остатках.")
    if abs(acf_168) > 0.10:
        recs.append(f"ACF(168)={acf_168:.3f} > 0.10: недельная компонента в остатках.")
    if kurt > 5:   recs.append(f"Kurtosis={kurt:.2f} > 5: тяжёлые хвосты → Huber loss.")
    elif kurt > 3: recs.append(f"Kurtosis={kurt:.2f} > 3 → используется Huber loss.")

    log = logging.getLogger("smart_grid.models.trainer")
    log.info("─ Диагностика: %s (n=%d) ─", model_name or "?", n)
    log.info("  DW=%.4f  ACF(1)=%.4f  ACF(24)=%.4f  ACF(48)=%.4f  ACF(168)=%.4f",
             dw, acf_1, acf_24, acf_48, acf_168)
    log.info("  Skewness=%.4f  Kurtosis=%.4f", skew, kurt)
    if recs:
        log.info("  Рекомендации:")
        for r in recs: log.info("    → %s", r)
    else:
        log.info("  ✅ Остатки в норме.")
    return {
        "model": model_name, "n": n,
        "DW": round(dw,4), "ACF_1": round(acf_1,4),
        "ACF_24": round(acf_24,4), "ACF_48": round(acf_48,4),
        "ACF_168": round(acf_168,4),
        "skewness": round(skew,4), "kurtosis": round(kurt,4),
        "recommendations": recs,
    }


def diagnose_training_regime(history, overfit_gap=0.02, underfit_floor=0.08):
    train_mae = history.get("mae") or history.get("mean_absolute_error") or []
    val_mae   = history.get("val_mae") or history.get("val_mean_absolute_error") or []
    if not train_mae or not val_mae:
        return {"status": "unknown", "reason": "mae_history_missing"}
    train_last = float(train_mae[-1]); val_last = float(val_mae[-1])
    gap = val_last - train_last
    if gap >= overfit_gap and train_last < underfit_floor:   status = "overfitting"
    elif train_last >= underfit_floor and val_last >= underfit_floor: status = "underfitting"
    else:                                                             status = "balanced"
    return {
        "status": status,
        "train_mae_last": train_last,
        "val_mae_last": val_last,
        "generalization_gap": float(gap),
    }


# ══════════════════════════════════════════════════════════════════════════════
# СОСТАВНАЯ МЕТРИКА
# ══════════════════════════════════════════════════════════════════════════════

def composite_score(metrics):
    """0.5*MAE + 0.3*RMSE + 0.2*|ACF24|*MAE — ниже = лучше."""
    mae   = metrics.get("MAE",   float("inf"))
    rmse  = metrics.get("RMSE",  float("inf"))
    acf24 = abs(metrics.get("ACF_24", 0.0))
    return 0.5 * mae + 0.3 * rmse + 0.2 * acf24 * mae


# ══════════════════════════════════════════════════════════════════════════════
# РЕКОНСТРУКЦИЯ ПРЕДСКАЗАНИЙ
# ══════════════════════════════════════════════════════════════════════════════

def _reconstruct_predictions(
    y_model_output: np.ndarray,
    data: Dict[str, Any],
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Реконструирует абсолютные предсказания и истинные значения в кВт·ч.

    seasonal_diff=True:  y_pred = inverse_scale((y_out + Y_naive).clip(0))
    seasonal_diff=False: y_pred = inverse_scale(y_out)
    """
    scaler = data["scaler"]
    Y_true_target = data[f"Y_{split}"]

    if data.get("seasonal_diff", False):
        naive = data[f"Y_seasonal_naive_{split}"]
        y_pred_abs = (y_model_output + naive).clip(0)
        y_true_abs = (Y_true_target  + naive).clip(0)
        y_pred = inverse_scale(scaler, y_pred_abs)
        y_true = inverse_scale(scaler, y_true_abs)
    else:
        y_pred = inverse_scale(scaler, y_model_output)
        y_true = inverse_scale(scaler, Y_true_target)

    return y_pred, y_true


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """Универсальный тренер: Keras + sklearn через единый интерфейс."""

    def __init__(self, model, model_name, models_dir="results/models", plots_dir="results/plots"):
        self.model      = model
        self.model_name = model_name
        self.models_dir = models_dir
        self.plots_dir  = plots_dir
        self.history    = None
        self.train_time = 0.0
        # Post-hoc bias correction (устанавливается из main.py step 5c)
        # bias_correction = mean(y_pred_val - y_true_val) в кВт·ч
        self._bias_correction: Optional[float] = None

    def train(self, data, epochs=200, batch_size=32, patience=25,
              lr_patience=10, lr_factor=0.5, min_delta=1e-5):
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

    def _train_keras(self, data, epochs, batch_size, patience,
                     lr_patience, lr_factor, min_delta):
        os.makedirs(self.models_dir, exist_ok=True)

        use_lr_schedule = _optimizer_has_lr_schedule(self.model)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.models_dir, f"{self.model_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        )
        callbacks = [early_stop, checkpoint]

        if not use_lr_schedule:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=lr_factor,
                patience=lr_patience,
                min_delta=min_delta,
                min_lr=1e-7,
                verbose=1,
            )
            callbacks.append(reduce_lr)
            logger.info("%s: использует ReduceLROnPlateau (float LR)", self.model_name)
        else:
            logger.info(
                "%s: использует CosineDecay schedule — ReduceLROnPlateau отключён",
                self.model_name,
            )

        current_batch = batch_size
        last_exc = None
        while current_batch >= 4:
            try:
                logger.info("%s: fit batch_size=%d", self.model_name, current_batch)
                self.history = self.model.fit(
                    data["X_train"], data["Y_train"],
                    validation_data=(data["X_val"], data["Y_val"]),
                    epochs=epochs,
                    batch_size=current_batch,
                    callbacks=callbacks,
                    verbose=1,
                )
                actual_epochs = len(self.history.history["loss"])
                best_epoch = int(np.argmin(
                    self.history.history.get("val_loss", self.history.history["loss"])
                )) + 1
                best_val_loss = float(np.min(
                    self.history.history.get("val_loss", self.history.history["loss"])
                ))
                logger.info("%s: %d эпох | best_epoch=%d | best_val_loss=%.6f",
                            self.model_name, actual_epochs, best_epoch, best_val_loss)
                diag = diagnose_training_regime(self.history.history)
                logger.info(
                    "%s: regime=%s train_mae=%.4f val_mae=%.4f gap=%.4f",
                    self.model_name, diag.get("status","?"),
                    diag.get("train_mae_last", float("nan")),
                    diag.get("val_mae_last",   float("nan")),
                    diag.get("generalization_gap", float("nan")),
                )
                return
            except tf.errors.ResourceExhaustedError as exc:
                last_exc = exc
                gc.collect()
                current_batch //= 2
                logger.warning("%s: OOM. Повтор batch=%d", self.model_name, current_batch)
            except Exception as exc:
                err_str = str(exc).lower()
                if "out of memory" in err_str or "oom" in err_str:
                    last_exc = exc
                    gc.collect()
                    current_batch //= 2
                    logger.warning("%s: OOM. Повтор batch=%d", self.model_name, current_batch)
                else:
                    logger.error("Ошибка %s: %s", self.model_name, exc)
                    raise

        if last_exc:
            raise last_exc
        raise RuntimeError(f"{self.model_name}: training failed (all batch sizes exhausted)")

    def _train_sklearn(self, data):
        self.model.fit(
            data["X_train"], data["Y_train"],
            X_val=data.get("X_val"),
            Y_val=data.get("Y_val"),
        )

    def predict(self, X):
        """Возвращает сырой output модели в нормированном пространстве."""
        if isinstance(self.model, tf.keras.Model):
            return self.model.predict(X, verbose=0)
        return self.model.predict(X)

    def _get_split_inputs(self, data: Dict[str, Any], split: str):
        """Подбирает корректный входной тензор для конкретной модели."""
        tft_key = f"X_tft_{split}"
        if self.model_name == "TFT-Lite" and tft_key in data:
            return data[tft_key]
        return data[f"X_{split}"]

    def predict_absolute(self, data: Dict[str, Any], split: str = "test") -> np.ndarray:
        """
        Возвращает предсказание в кВт·ч.

        v8: если установлен _bias_correction (post-hoc, вычислен на val),
            он вычитается из предсказания:
              y_corrected = y_pred - bias_correction
            где bias_correction = mean(y_pred_val - y_true_val).
            Это снижает систематическое занижение (MBE < 0).
        """
        X = self._get_split_inputs(data, split)
        y_out, _ = _reconstruct_predictions(self.predict(X), data, split)

        if self._bias_correction is not None:
            y_out = y_out - self._bias_correction
            logger.debug(
                "%s: bias_correction=%.1f кВт·ч применена к split=%s",
                self.model_name, self._bias_correction, split
            )

        return y_out

    def mc_predict(self, X, n_samples=50):
        if not isinstance(self.model, tf.keras.Model):
            raise TypeError("mc_predict только для Keras-моделей")
        preds = np.stack(
            [self.model(X, training=True).numpy() for _ in range(n_samples)],
            axis=0,
        )
        return preds.mean(axis=0), preds.std(axis=0)

    def evaluate(self, data, split="test", run_residual_diagnostics=True):
        """
        Вычисляет метрики в кВт·ч.
        Применяет bias correction если установлена.
        """
        X = self._get_split_inputs(data, split)
        y_raw = self.predict(X)
        y_pred, y_true = _reconstruct_predictions(y_raw, data, split)

        # Применяем bias correction если есть
        if self._bias_correction is not None:
            y_pred = y_pred - self._bias_correction

        metrics = compute_all_metrics(y_true, y_pred, model_name=self.model_name)
        if run_residual_diagnostics:
            diag = diagnose_residuals(y_true, y_pred, model_name=self.model_name)
            metrics["DW"]       = diag["DW"]
            metrics["ACF_24"]   = diag["ACF_24"]
            metrics["ACF_168"]  = diag["ACF_168"]
            metrics["kurtosis"] = diag["kurtosis"]
        metrics["composite_score"] = composite_score(metrics)
        return metrics

    def save(self):
        os.makedirs(self.models_dir, exist_ok=True)
        path = os.path.join(self.models_dir, self.model_name)
        if isinstance(self.model, tf.keras.Model):
            save_path = path + ".keras"
            self.model.save(save_path)
            logger.info("Сохранено: %s", save_path)
            return save_path
        else:
            import pickle
            save_path = path + ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info("Сохранено: %s", save_path)
            return save_path

    @classmethod
    def load_keras(cls, path, model_name=""):
        from models.transformer import (
            iTransformerBlock, PreLNEncoderBlock, SinusoidalPE,
            Time2Vec, GatedResidualNetwork, WarmupCosineDecay as WCD_tr)
        from models.lstm import TemporalAttentionBlock, TCNBlock, _SeasonalSkipLayer, WarmupCosineDecay
        model = tf.keras.models.load_model(path, custom_objects={
            "iTransformerBlock":     iTransformerBlock,
            "PreLNEncoderBlock":     PreLNEncoderBlock,
            "SinusoidalPE":          SinusoidalPE,
            "Time2Vec":              Time2Vec,
            "GatedResidualNetwork":  GatedResidualNetwork,
            "WarmupCosineDecay":     WarmupCosineDecay,
            "TemporalAttentionBlock":TemporalAttentionBlock,
            "TCNBlock":              TCNBlock,
            "_SeasonalSkipLayer":    _SeasonalSkipLayer,
        })
        return cls(model, model_name or os.path.basename(path))


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

class WeightedEnsemble:
    """
    Взвешенный ансамбль с оптимизацией весов на val-set (Nelder-Mead).

    v8: минимальный вес 0.1% для каждой модели (ранее PatchTST получал 0.000).
    """

    def __init__(self, trainers, weights=None, model_name="WeightedEnsemble"):
        if not trainers:
            raise ValueError("trainers не может быть пустым")
        self.trainers   = trainers
        self.model_name = model_name
        self.history    = None
        self.train_time = 0.0
        n = len(trainers)
        if weights is not None:
            w = np.array(weights, dtype=np.float64)
            self.weights = w / w.sum()
        else:
            self.weights = np.ones(n, dtype=np.float64) / n

    def predict(self, X):
        """Weighted average сырых предсказаний (нормированное пространство)."""
        preds = np.stack([t.predict(X) for t in self.trainers], axis=0)
        return np.einsum("m,mnh->nh", self.weights, preds).astype(np.float32)

    def optimize_weights(self, data: Dict[str, Any], split: str = "val") -> np.ndarray:
        """
        Оптимизирует веса по MAE в кВт·ч на val-set.

        v8 FIX: минимальный вес 1e-3 (0.1%) после оптимизации.
          БЫЛО: PatchTST получал w=0.000 → не участвовал в ансамбле вообще.
          СТАЛО: np.maximum(w_opt, 1e-3) → минимум 0.1% вклада каждой модели.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy не установлен. Используем равные веса.")
            return self.weights

        X_split = data[f"X_{split}"]
        n = len(self.trainers)

        # Кэшируем абсолютные предсказания каждой модели
        abs_preds = []
        for t in self.trainers:
            y_raw = t.predict(X_split)
            y_abs, _ = _reconstruct_predictions(y_raw, data, split)
            abs_preds.append(y_abs)
        abs_preds_arr = np.stack(abs_preds, axis=0)   # (M, N, H)

        # Истинные абсолютные значения
        scaler = data["scaler"]
        if data.get("seasonal_diff", False):
            naive = data[f"Y_seasonal_naive_{split}"]
            y_true_abs_s = (data[f"Y_{split}"] + naive).clip(0)
            y_true = inverse_scale(scaler, y_true_abs_s)
        else:
            y_true = inverse_scale(scaler, data[f"Y_{split}"])

        def objective(w):
            w_abs  = np.abs(w)
            w_norm = w_abs / (w_abs.sum() + 1e-8)
            pred   = np.einsum("m,mnh->nh", w_norm, abs_preds_arr)
            return float(np.mean(np.abs(pred - y_true)))

        x0      = np.ones(n) / n
        mae_eq  = objective(x0)
        result  = minimize(objective, x0, method="Nelder-Mead",
                           options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6})

        # v8 FIX: минимальный вес 0.1% для каждой модели
        w_opt  = np.abs(result.x)
        w_opt  = np.maximum(w_opt, 1e-3)   # ← минимум 0.1%
        w_opt /= w_opt.sum()
        self.weights = w_opt

        logger.info(
            "Ensemble weights оптимизированы (%s):",
            ", ".join(f"{t.model_name}={w:.3f}" for t, w in zip(self.trainers, self.weights))
        )
        logger.info(
            "  MAE (равные)=%.2f → MAE (оптим)=%.2f | Δ=%.1f%%",
            mae_eq, result.fun,
            (mae_eq - result.fun) / mae_eq * 100 if mae_eq > 0 else 0.0,
        )
        return self.weights

    def predict_absolute(self, data, split="test"):
        """Возвращает взвешенное предсказание ансамбля в кВт·ч."""
        X = data[f"X_{split}"]
        y_raw = self.predict(X)
        y_out, _ = _reconstruct_predictions(y_raw, data, split)
        return y_out

    def evaluate(self, data, split="test", run_residual_diagnostics=True):
        X      = data[f"X_{split}"]
        y_raw  = self.predict(X)
        y_pred, y_true = _reconstruct_predictions(y_raw, data, split)

        metrics = compute_all_metrics(y_true, y_pred, model_name=self.model_name)
        if run_residual_diagnostics:
            diag = diagnose_residuals(y_true, y_pred, model_name=self.model_name)
            metrics["DW"]       = diag["DW"]
            metrics["ACF_24"]   = diag["ACF_24"]
            metrics["ACF_168"]  = diag["ACF_168"]
            metrics["kurtosis"] = diag["kurtosis"]
        metrics["composite_score"] = composite_score(metrics)
        metrics["ensemble_weights"] = {
            t.model_name: round(float(w), 4)
            for t, w in zip(self.trainers, self.weights)
        }
        return metrics

    def __repr__(self):
        wstr = ", ".join(
            f"{t.model_name}={w:.3f}" for t, w in zip(self.trainers, self.weights)
        )
        return f"WeightedEnsemble([{wstr}])"


# ══════════════════════════════════════════════════════════════════════════════
# СРАВНЕНИЕ ТРЕНЕРОВ
# ══════════════════════════════════════════════════════════════════════════════

def compare_trainers(
    trainers: List[Any],
    data: Dict[str, Any],
    split: str = "test",
    select_by: str = "composite",
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """
    Вычисляет метрики всех моделей в абсолютном пространстве (кВт·ч).
    """
    results = {}
    logger.info("\n%s", "=" * 72)
    logger.info(
        "СРАВНЕНИЕ МОДЕЛЕЙ (split=%s, select_by=%s, seasonal_diff=%s)",
        split.upper(), select_by, data.get("seasonal_diff", False),
    )
    logger.info("%s", "=" * 72)
    logger.info(
        "%-24s %8s %8s %7s %7s %6s %7s %10s",
        "Модель", "MAE", "RMSE", "sMAPE", "R2", "DW", "ACF24", "Composite",
    )
    logger.info("%s", "-" * 80)

    for trainer in trainers:
        try:
            m = trainer.evaluate(data, split=split, run_residual_diagnostics=True)
            results[trainer.model_name] = m
            logger.info(
                "%-24s %8.2f %8.2f %7.2f%% %6.4f %6s %7s %10.3f",
                trainer.model_name,
                m["MAE"], m["RMSE"],
                m.get("sMAPE", m["MAPE"]),
                m["R2"],
                f"{m.get('DW', float('nan')):.3f}",
                f"{m.get('ACF_24', float('nan')):.3f}",
                m.get("composite_score", float("nan")),
            )
        except Exception as exc:
            logger.error("Ошибка оценки %s: %s", trainer.model_name, exc)

    if not results:
        return results, ""

    logger.info("%s", "-" * 80)
    key_map = {"composite": "composite_score", "mae": "MAE", "rmse": "RMSE"}
    sort_key = key_map.get(select_by, "composite_score")
    best_name = min(results, key=lambda k: results[k].get(sort_key, float("inf")))
    logger.info(
        "Лучшая по %s: %s | MAE=%.2f | composite=%.3f",
        sort_key, best_name,
        results[best_name]["MAE"],
        results[best_name].get("composite_score", float("nan")),
    )
    return results, best_name