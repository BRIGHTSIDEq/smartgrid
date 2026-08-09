# -*- coding: utf-8 -*-
"""
models/trainer.py — Универсальный тренер моделей.

Единый интерфейс обучения и оценки для Keras-моделей и для sklearn/xgboost/
наивных обёрток. Включает:
  * EarlyStopping + ReduceLROnPlateau (последний отключается, если у оптимизатора
    уже задано собственное расписание LR);
  * автоматическое понижение batch_size при OOM;
  * диагностику режима обучения (переобучение / недообучение);
  * корректную диагностику автокорреляции остатков (см. ниже).

О ФОРМЕ ОСТАТКОВ — важно для интерпретации ACF и Durbin–Watson.
Прогноз имеет форму (N, H): N перекрывающихся окон, H шагов горизонта.
Окна идут со сдвигом 1 час, поэтому в РАЗВЁРНУТОМ массиве соседние элементы
не образуют почасовой ряд: элемент k = i·H + h соответствует моменту i + h.
Из-за этого «лаг 24» по развёрнутому массиву равен одному часу реального
времени, а не суткам. Корректный почасовой ряд — это срез при ФИКСИРОВАННОМ
шаге горизонта: residuals[:, h] индексируется номером окна i, а соседние окна
отстоят ровно на час. Вся диагностика ниже работает именно по срезам.
История изменений — в CHANGELOG.md.
"""

import logging
import os
import time
import gc
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
# ДИАГНОСТИКА ОСТАТКОВ
# ══════════════════════════════════════════════════════════════════════════════

def _acf_lag(series: np.ndarray, lag: int) -> float:
    """Выборочная автокорреляция ряда на заданном лаге."""
    if lag <= 0 or len(series) <= lag:
        return float("nan")
    s = series - series.mean()
    var = np.var(s) + 1e-12
    return float(np.dot(s[lag:], s[:-lag]) / (len(s) * var))


def _durbin_watson(series: np.ndarray) -> float:
    d = np.diff(series)
    return float(np.sum(d ** 2) / (np.sum(series ** 2) + 1e-12))


def diagnose_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    horizon_step: int = 0,
) -> Dict[str, Any]:
    """
    Диагностика остатков: DW и ACF на лагах 1/24/48/168, асимметрия, эксцесс.

    Автокорреляция считается по срезу с ФИКСИРОВАННЫМ шагом горизонта
    (`residuals[:, horizon_step]`) — только такой срез является настоящим
    почасовым рядом. Дополнительно возвращается среднее по всем шагам
    горизонта, чтобы вывод не зависел от выбора одного среза.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N, H) либо (N,)
    horizon_step : int
        Шаг горизонта для основной диагностики. 0 = прогноз на 1 час вперёд.

    Returns
    -------
    dict: {"DW", "ACF_1", "ACF_24", "ACF_48", "ACF_168",
           "DW_mean_over_h", "ACF_24_mean_over_h",
           "skewness", "kurtosis", "recommendations"}
    """
    resid_2d = np.atleast_2d(np.asarray(y_true) - np.asarray(y_pred))
    if resid_2d.shape[0] == 1 and resid_2d.shape[1] > 1:
        # Пришёл одномерный непрерывный ряд — трактуем как единственный срез.
        resid_2d = resid_2d.T
    n_windows, horizon = resid_2d.shape

    h = int(np.clip(horizon_step, 0, horizon - 1))
    series = resid_2d[:, h].astype(np.float64)

    dw = _durbin_watson(series)
    acf_1 = _acf_lag(series, 1)
    acf_24 = _acf_lag(series, 24)
    acf_48 = _acf_lag(series, 48)
    acf_168 = _acf_lag(series, 168)

    # Усреднение по всем шагам горизонта — устойчивее к выбору среза.
    dw_all = [_durbin_watson(resid_2d[:, k].astype(np.float64)) for k in range(horizon)]
    acf24_all = [_acf_lag(resid_2d[:, k].astype(np.float64), 24) for k in range(horizon)]
    dw_mean = float(np.nanmean(dw_all))
    acf24_mean = float(np.nanmean(acf24_all))

    flat = resid_2d.flatten()
    std = np.std(flat) + 1e-12
    kurt = float(np.mean((flat - flat.mean()) ** 4) / std ** 4)
    skew = float(np.mean((flat - flat.mean()) ** 3) / std ** 3)

    recommendations: List[str] = []
    if dw_mean < 1.5:
        recommendations.append(
            f"DW={dw_mean:.3f} < 1.5: остатки положительно автокоррелированы — "
            "модель систематически запаздывает за уровнем нагрузки."
        )
    if abs(acf24_mean) > 0.10:
        recommendations.append(
            f"ACF(24)={acf24_mean:.3f} > 0.10: суточная компонента не устранена "
            "полностью, несмотря на наличие load_lag_24h среди признаков."
        )
    if abs(acf_168) > 0.10:
        recommendations.append(
            f"ACF(168)={acf_168:.3f} > 0.10: недельная компонента не устранена."
        )
    if kurt > 5:
        recommendations.append(
            f"Kurtosis={kurt:.2f} > 5: очень тяжёлые хвосты — редкие крупные "
            "промахи на аномалиях. Точечный прогноз стоит дополнить интервальным."
        )
    elif kurt > 3:
        recommendations.append(
            f"Kurtosis={kurt:.2f} > 3: тяжёлые хвосты, Huber loss оправдан."
        )

    log = logging.getLogger("smart_grid.models.trainer")
    log.info("─ Диагностика остатков: %s (окон=%d, горизонт=%d, срез h=%d) ──",
             model_name or "?", n_windows, horizon, h + 1)
    log.info("  По срезу h=%d: DW=%.4f  ACF(1)=%.4f  ACF(24)=%.4f  ACF(48)=%.4f  ACF(168)=%.4f",
             h + 1, dw, acf_1, acf_24, acf_48, acf_168)
    log.info("  Среднее по всем шагам горизонта: DW=%.4f  ACF(24)=%.4f",
             dw_mean, acf24_mean)
    log.info("  Skewness=%.4f  Kurtosis=%.4f", skew, kurt)
    if recommendations:
        log.info("  ⚠️  Замечания:")
        for rec in recommendations:
            log.info("    → %s", rec)
    else:
        log.info("  ✅ Остатки в норме (DW≥1.5, |ACF(24)|<0.10, Kurt<3).")

    return {
        "model":              model_name,
        "n_windows":          n_windows,
        "horizon":            horizon,
        "horizon_step":       h + 1,
        "DW":                 round(dw, 4),
        "ACF_1":              round(acf_1, 4),
        "ACF_24":             round(acf_24, 4),
        "ACF_48":             round(acf_48, 4),
        "ACF_168":            round(acf_168, 4),
        "DW_mean_over_h":     round(dw_mean, 4),
        "ACF_24_mean_over_h": round(acf24_mean, 4),
        "skewness":           round(skew, 4),
        "kurtosis":           round(kurt, 4),
        "recommendations":    recommendations,
    }

# ══════════════════════════════════════════════════════════════════════════════
# TRAINER

def diagnose_training_regime(
    history: Dict[str, List[float]],
    overfit_gap: float = 0.02,
    underfit_floor: float = 0.08,
) -> Dict[str, Any]:
    """
    Определяет режим обучения по ЛУЧШЕЙ эпохе, а не по последней.

    EarlyStopping с restore_best_weights возвращает веса лучшей эпохи, поэтому
    оценивается именно эта модель. Диагностика по последней эпохе описывала
    другую, уже переобученную сеть: например, PatchTST достигал минимума
    валидации на пятой эпохе, затем 25 эпох ухудшался, и в отчёт попадали
    метрики финального состояния — с выводом «сбалансировано» для модели,
    которая переобучилась почти сразу.

    Отдельно фиксируется, сколько эпох прошло после лучшей и ухудшалась ли
    валидация: это прямой признак избыточной ёмкости.

    Returns
    -------
    dict с ключами best_epoch, final_epoch, train_metric_at_best,
    val_metric_at_best, final_train_metric, final_val_metric,
    epochs_after_best, overfit_after_best, status.
    """
    train_mae = history.get("mae") or history.get("mean_absolute_error") or []
    val_mae = history.get("val_mae") or history.get("val_mean_absolute_error") or []
    val_loss = history.get("val_loss") or []

    if not train_mae or not val_mae:
        return {"status": "unknown", "reason": "mae_history_missing"}

    # Лучшая эпоха выбирается по тому же критерию, что и EarlyStopping —
    # по val_loss, а при его отсутствии по val_mae.
    criterion = val_loss if val_loss else val_mae
    best_idx = int(np.argmin(criterion))
    final_idx = len(val_mae) - 1

    train_at_best = float(train_mae[best_idx])
    val_at_best = float(val_mae[min(best_idx, len(val_mae) - 1)])
    gap = val_at_best - train_at_best

    epochs_after_best = final_idx - best_idx
    # Ухудшение валидации после лучшей эпохи при продолжающемся падении
    # ошибки на обучении — определение переобучения.
    val_worsened = float(val_mae[final_idx]) > val_at_best
    train_improved = float(train_mae[final_idx]) < train_at_best
    overfit_after_best = bool(epochs_after_best > 0 and val_worsened and train_improved)

    if overfit_after_best or (gap >= overfit_gap and train_at_best < underfit_floor):
        status = "overfitting"
    elif train_at_best >= underfit_floor and val_at_best >= underfit_floor:
        status = "underfitting"
    else:
        status = "balanced"

    return {
        "status": status,
        "best_epoch": best_idx + 1,
        "final_epoch": final_idx + 1,
        "train_metric_at_best": train_at_best,
        "val_metric_at_best": val_at_best,
        "final_train_metric": float(train_mae[final_idx]),
        "final_val_metric": float(val_mae[final_idx]),
        "epochs_after_best": epochs_after_best,
        "overfit_after_best": overfit_after_best,
        "generalization_gap": float(gap),
    }


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
        self.train_windows: int = 0
        self.fit_diagnostics: Dict[str, Any] = {}

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
        self.train_windows = int(len(data["X_train"]))
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

        current_batch = int(batch_size)
        last_exc: Optional[Exception] = None
        while current_batch >= 4:
            try:
                logger.info("%s: старт fit с batch_size=%d", self.model_name, current_batch)
                self.history = self.model.fit(
                    data["X_train"], data["Y_train"],
                    validation_data=(data["X_val"], data["Y_val"]),
                    epochs=epochs,
                    batch_size=current_batch,
                    callbacks=callbacks,
                    # verbose=2 — одна строка на эпоху вместо анимированного
                    # прогресс-бара. Бар Keras рисуется символами Unicode и на
                    # консоли с однобайтовой кодировкой (cp1251 в русской
                    # Windows) вызывает UnicodeEncodeError, обрывая обучение.
                    # Режим 2 использует только ASCII и работает везде, не
                    # требуя от пользователя выставлять PYTHONIOENCODING.
                    verbose=2,
                )
                actual_epochs = len(self.history.history["loss"])
                logger.info("%s: обучение завершено за %d эпох",
                            self.model_name, actual_epochs)
                fit_diag = diagnose_training_regime(self.history.history)
                self.fit_diagnostics = fit_diag
                logger.info(
                    "%s: режим=%s | лучшая эпоха %s из %s | "
                    "на лучшей: train_mae=%.4f val_mae=%.4f (разрыв %.4f)",
                    self.model_name, fit_diag.get("status", "unknown"),
                    fit_diag.get("best_epoch", "?"), fit_diag.get("final_epoch", "?"),
                    fit_diag.get("train_metric_at_best", float("nan")),
                    fit_diag.get("val_metric_at_best", float("nan")),
                    fit_diag.get("generalization_gap", float("nan")),
                )
                if fit_diag.get("overfit_after_best"):
                    logger.warning(
                        "%s: после лучшей эпохи ещё %d эпох ошибка на обучении падала, "
                        "а на валидации росла (%.4f → %.4f) — признак избыточной ёмкости. "
                        "Оценивается модель лучшей эпохи, восстановленная EarlyStopping.",
                        self.model_name, fit_diag.get("epochs_after_best", 0),
                        fit_diag.get("val_metric_at_best", float("nan")),
                        fit_diag.get("final_val_metric", float("nan")),
                    )
                if self.train_windows:
                    n_params = self.count_params()
                    logger.info(
                        "%s: параметров=%d | обучающих окон=%d | параметров на окно=%.1f",
                        self.model_name, n_params, self.train_windows,
                        n_params / max(self.train_windows, 1),
                    )
                return
            except tf.errors.ResourceExhaustedError as exc:
                last_exc = exc
                next_batch = current_batch // 2
                logger.warning(
                    "%s: ResourceExhausted/OOM при batch_size=%d. Повтор с batch_size=%d",
                    self.model_name, current_batch, next_batch
                )
                gc.collect()
                current_batch = next_batch
            except Exception as exc:
                if "out of memory" in str(exc).lower() or "oom" in str(exc).lower():
                    last_exc = exc
                    next_batch = current_batch // 2
                    logger.warning(
                        "%s: OOM-подобная ошибка при batch_size=%d. Повтор с batch_size=%d",
                        self.model_name, current_batch, next_batch
                    )
                    gc.collect()
                    current_batch = next_batch
                else:
                    logger.error("Ошибка при обучении %s: %s", self.model_name, exc)
                    raise

        logger.error("%s: не удалось обучить модель даже с batch_size=4", self.model_name)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"{self.model_name}: training failed for unknown reason")

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

    def predict_original_scale(
        self,
        data: Dict[str, Any],
        split: str = "test",
    ) -> np.ndarray:
        """Прогноз на split-е, приведённый к исходному масштабу (кВт·ч)."""
        return inverse_scale(data["scaler"], self.predict(data[f"X_{split}"]))

    def evaluate(
        self,
        data: Dict[str, Any],
        split: str = "test",
        run_residual_diagnostics: bool = True,
    ) -> Dict[str, float]:
        """
        Оценка модели на split-е с опциональной диагностикой остатков.

        Если в `data` присутствует ключ "mase_scale" (см.
        utils.metrics.seasonal_naive_scale), дополнительно считается MASE —
        главная метрика для ответа на вопрос «лучше ли модель наивного прогноза».

        Parameters
        ----------
        run_residual_diagnostics : bool
            Если True — запускает diagnose_residuals() и выводит DW/ACF/Kurt
            в лог. По умолчанию включено для всех моделей.
        """
        X = data[f"X_{split}"]
        Y_true_scaled = data[f"Y_{split}"]
        scaler = data["scaler"]

        Y_pred_scaled = self.predict(X)
        Y_true = inverse_scale(scaler, Y_true_scaled)
        Y_pred = inverse_scale(scaler, Y_pred_scaled)

        metrics = compute_all_metrics(
            Y_true, Y_pred,
            model_name=self.model_name,
            mase_scale=data.get("mase_scale"),
        )
        metrics["train_time_sec"] = round(self.train_time, 1)
        metrics["n_params"] = self.count_params()
        metrics["train_windows"] = self.train_windows
        if self.train_windows:
            metrics["params_per_window"] = round(
                self.count_params() / self.train_windows, 3)
        if self.fit_diagnostics:
            metrics["best_epoch"] = self.fit_diagnostics.get("best_epoch")
            metrics["final_epoch"] = self.fit_diagnostics.get("final_epoch")
            metrics["overfit_after_best"] = self.fit_diagnostics.get("overfit_after_best")

        if run_residual_diagnostics:
            diag = diagnose_residuals(Y_true, Y_pred, model_name=self.model_name)
            metrics["DW"]       = diag["DW_mean_over_h"]
            metrics["ACF_24"]   = diag["ACF_24_mean_over_h"]
            metrics["ACF_168"]  = diag["ACF_168"]
            metrics["kurtosis"] = diag["kurtosis"]

        return metrics

    def count_params(self) -> int:
        """Число обучаемых параметров (0 для моделей без параметров)."""
        if isinstance(self.model, tf.keras.Model):
            return int(self.model.count_params())
        return 0

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
                PreLNEncoderBlock, SinusoidalPE, Time2Vec,
            )
            from models.lstm import (
                TemporalAttentionBlock, TCNBlock, SeasonalSkipConnection, ConsumptionRevIN,
            )
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "PreLNEncoderBlock":      PreLNEncoderBlock,
                    "SinusoidalPE":           SinusoidalPE,
                    "Time2Vec":               Time2Vec,
                    "TemporalAttentionBlock": TemporalAttentionBlock,
                    "TCNBlock":               TCNBlock,
                    "SeasonalSkipConnection": SeasonalSkipConnection,
                    "ConsumptionRevIN":       ConsumptionRevIN,
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
    """
    Оценивает все модели на одном split-е и печатает сводную таблицу.

    Колонка MASE — ключевая: значение >= 1 означает, что модель не превзошла
    сезонно-наивный прогноз и практической ценности не имеет.
    """
    results: Dict[str, Dict[str, float]] = {}
    logger.info("\n%s", "=" * 92)
    logger.info("СРАВНЕНИЕ МОДЕЛЕЙ (split=%s)", split.upper())
    logger.info("%s", "=" * 92)

    for trainer in trainers:
        try:
            results[trainer.model_name] = trainer.evaluate(
                data, split=split, run_residual_diagnostics=True
            )
        except Exception as exc:
            logger.error("Ошибка оценки %s: %s", trainer.model_name, exc)

    if not results:
        return results

    logger.info("\n%s", "─" * 92)
    logger.info("%-22s %9s %9s %7s %8s %7s %7s %7s",
                "Модель", "MAE", "RMSE", "MAPE%", "R2", "MASE", "DW", "ACF24")
    logger.info("%s", "─" * 92)
    for name in sorted(results, key=lambda k: results[k]["MAE"]):
        m = results[name]
        logger.info(
            "%-22s %9.2f %9.2f %6.2f%% %8.4f %7s %7.3f %7.3f",
            name, m["MAE"], m["RMSE"], m["MAPE"], m["R2"],
            f"{m['MASE']:.4f}" if "MASE" in m else "н/д",
            m.get("DW", float("nan")), m.get("ACF_24", float("nan")),
        )
    logger.info("%s", "─" * 92)

    best = min(results, key=lambda k: results[k]["MAE"])
    logger.info("Лучшая модель по MAE (%s): %s", split, best)

    # Проверка на прогностическую ценность относительно наивного базлайна.
    if "MASE" in results[best]:
        skilled = [n for n, m in results.items() if m.get("MASE", 9e9) < 1.0]
        if skilled:
            logger.info(
                "Модели лучше сезонно-наивного прогноза (MASE<1): %s",
                ", ".join(sorted(skilled, key=lambda n: results[n]["MASE"])),
            )
        else:
            logger.warning(
                "⚠️  НИ ОДНА модель не превзошла сезонно-наивный прогноз (MASE>=1). "
                "Это главный результат, который нужно объяснить в работе."
            )

    return results