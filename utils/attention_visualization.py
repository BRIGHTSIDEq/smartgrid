# -*- coding: utf-8 -*-
"""
utils/attention_visualization.py — Визуализация весов внимания Transformer.

Интерпретируемость (Interpretability) — один из ключевых аргументов в пользу
Transformer для научной работы. Веса внимания показывают, какие прошлые
моменты времени модель считает наиболее важными при прогнозе.

Примеры научных инсайтов:
- Голова 1 специализируется на лаге 24ч (суточный ритм)
- Голова 2 — на лаге 168ч (недельный ритм)
- Голова 3 — на ближайших 2-3 шагах (краткосрочная динамика)
- В праздничные часы внимание перераспределяется на другие праздники в истории
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

logger = logging.getLogger("smart_grid.utils.attention_viz")


# ──────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────────────────────

def _get_enc_blocks(model: tf.keras.Model) -> List[Any]:
    """Извлекает encoder-блоки из модели (ищет PreLNEncoderBlock слои)."""
    from models.transformer import PreLNEncoderBlock
    enc_blocks = []
    # Сначала проверяем атрибут _enc_blocks (проставляем при построении)
    if hasattr(model, "_enc_blocks"):
        return model._enc_blocks
    # Иначе ищем по типу
    for layer in model.layers:
        if isinstance(layer, PreLNEncoderBlock):
            enc_blocks.append(layer)
    return enc_blocks


def extract_attention_weights(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    covariate_sample: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Выполняет forward pass и извлекает матрицы весов внимания
    из всех encoder-блоков.

    Parameters
    ----------
    model    : tf.keras.Model (VanillaTransformer, TFTLite или PatchTST)
    X_sample : np.ndarray shape=(1, history_length, 1)  — один пример
    covariate_sample : np.ndarray shape=(1, history_length, n_features) — для TFT

    Returns
    -------
    dict: {
        "block_0": np.ndarray shape=(1, num_heads, T, T),
        "block_1": ...,
        ...
    }
    """
    enc_blocks = _get_enc_blocks(model)
    if not enc_blocks:
        logger.warning("Encoder-блоки не найдены в модели %s", model.name)
        return {}

    # Forward pass (для активации _last_attn_weights в каждом блоке)
    inputs = X_sample
    if covariate_sample is not None:
        inputs = [X_sample, covariate_sample]

    model(inputs, training=False)

    weights = {}
    for i, blk in enumerate(enc_blocks):
        w = blk.get_attention_weights()
        if w is not None:
            weights[f"block_{i}"] = w.numpy() if hasattr(w, "numpy") else np.array(w)

    logger.debug("Извлечено %d матриц внимания", len(weights))
    return weights


def visualize_attention_weights(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    history_length: int = 48,
    covariate_sample: Optional[np.ndarray] = None,
    model_name: str = "Transformer",
    plots_dir: str = "results/plots",
    save: bool = True,
    timestamps: Optional[np.ndarray] = None,
) -> None:
    """
    Строит heatmap матриц внимания для всех encoder-блоков и всех голов.

    Интерпретация heatmap:
    - Строка i: позиция ЗАПРОСА (что прогнозируем / откуда смотрим)
    - Столбец j: позиция КЛЮЧА (на что обращаем внимание в истории)
    - Яркая клетка [i, j]: при формировании представления позиции i
      модель активно использует информацию из позиции j

    Что искать:
    - Диагональ: локальная авторегрессия (смотрю на себя)
    - Полосы на расстоянии 24: суточная периодичность
    - Полосы на расстоянии 168: недельная периодичность

    Parameters
    ----------
    timestamps : np.ndarray (опционально) — временные метки для оси X
    """
    os.makedirs(plots_dir, exist_ok=True)
    weights = extract_attention_weights(model, X_sample, covariate_sample)

    if not weights:
        logger.warning("Нет данных для визуализации внимания")
        return

    n_blocks = len(weights)
    for block_name, attn_matrix in weights.items():
        # attn_matrix: (batch, heads, query_len, key_len)
        # берём первый пример в батче
        attn = attn_matrix[0]           # (heads, Q, K)
        num_heads = attn.shape[0]
        Q = attn.shape[1]
        K = attn.shape[2]

        cols = min(num_heads, 4)
        rows = (num_heads + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = np.array(axes).flatten() if num_heads > 1 else [axes]

        fig.suptitle(
            f"{model_name} — {block_name} | Веса внимания\n"
            f"(строка=запрос, столбец=ключ; яркость=важность)",
            fontsize=11, fontweight="bold",
        )

        # Метки осей: временны́е индексы или часы назад
        if timestamps is not None and len(timestamps) >= K:
            import pandas as pd
            ts = pd.DatetimeIndex(timestamps[-K:])
            tick_labels = [f"{t.hour}:00" for t in ts]
        else:
            tick_labels = [f"t-{K - 1 - i}" for i in range(K)]

        for h in range(num_heads):
            ax = axes[h]
            head_attn = attn[h]  # (Q, K)

            im = ax.imshow(head_attn, aspect="auto", cmap="viridis", vmin=0, vmax=head_attn.max())
            ax.set_title(f"Голова {h + 1}", fontsize=9, fontweight="bold")

            # Подписи осей (каждые 6 шагов)
            tick_every = max(1, K // 8)
            ax.set_xticks(range(0, K, tick_every))
            ax.set_xticklabels(
                [tick_labels[i] for i in range(0, K, tick_every)],
                rotation=45, fontsize=6,
            )
            ax.set_yticks(range(0, Q, max(1, Q // 8)))
            ax.set_xlabel("Ключ (история)", fontsize=7)
            ax.set_ylabel("Запрос", fontsize=7)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Скрываем лишние subplot
        for h in range(num_heads, len(axes)):
            axes[h].set_visible(False)

        plt.tight_layout()
        if save:
            fn = os.path.join(
                plots_dir,
                f"attention_{model_name.replace(' ', '_')}_{block_name}.png",
            )
            fig.savefig(fn, dpi=150, bbox_inches="tight")
            logger.info("Attention heatmap: %s", fn)
        plt.close(fig)


def visualize_attention_summary(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    history_length: int = 48,
    model_name: str = "Transformer",
    plots_dir: str = "results/plots",
    save: bool = True,
) -> np.ndarray:
    """
    Строит агрегированный график «важности позиций в истории».

    Метод: усредняем веса внимания по всем головам, всем блокам,
    по всем позициям запросов. Получаем вектор важности для каждого
    временного шага истории.

    Это позволяет ответить: «Какой час назад в среднем наиболее важен
    для прогноза?»

    Returns
    -------
    np.ndarray shape=(history_length,) — усреднённая важность каждой позиции
    """
    os.makedirs(plots_dir, exist_ok=True)
    weights = extract_attention_weights(model, X_sample)

    if not weights:
        return np.ones(history_length) / history_length

    # Усредняем по блокам, головам, строкам запросов
    all_avg = []
    for block_name, attn_matrix in weights.items():
        attn = attn_matrix[0]           # (heads, Q, K)
        avg_over_heads = attn.mean(axis=0)   # (Q, K)
        avg_over_queries = avg_over_heads.mean(axis=0)  # (K,)
        all_avg.append(avg_over_queries)

    importance = np.mean(all_avg, axis=0)  # (K,)

    # Дополняем/обрезаем до history_length
    if len(importance) < history_length:
        # PatchTST: importance по патчам, расширяем на шаги
        patch_importance = np.repeat(importance, history_length // len(importance) + 1)
        importance = patch_importance[:history_length]
    importance = importance[:history_length]
    importance /= importance.sum() + 1e-8

    # ── График ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{model_name} — Важность временны́х позиций истории",
                 fontweight="bold")

    hours_back = np.arange(history_length - 1, -1, -1)  # [T-1, T-2, ..., 0]

    # Линейный график
    axes[0].plot(hours_back, importance[::-1], lw=2, color="steelblue")
    axes[0].fill_between(hours_back, 0, importance[::-1], alpha=0.3)
    axes[0].set_xlabel("Часов назад")
    axes[0].set_ylabel("Средний вес внимания")
    axes[0].set_title("Важность по часам назад")
    axes[0].grid(True, alpha=0.3)

    # Отмечаем ключевые лаги
    key_lags = [24, 48, 72, 168]
    for lag in key_lags:
        if lag < history_length:
            axes[0].axvline(lag, color="red", ls="--", alpha=0.5, lw=1)
            axes[0].text(lag, importance.max() * 0.9, f"{lag}ч",
                         color="red", fontsize=8, ha="center")

    # Тепловая карта по суткам
    h = history_length
    n_days = h // 24
    if n_days >= 1:
        importance_2d = importance[:n_days * 24].reshape(n_days, 24)
        im = axes[1].imshow(importance_2d, aspect="auto", cmap="YlOrRd")
        axes[1].set_xlabel("Час суток")
        axes[1].set_ylabel("Дней назад")
        axes[1].set_title("Важность: сутки × час")
        axes[1].set_xticks(range(0, 24, 4))
        axes[1].set_xticklabels([f"{h}:00" for h in range(0, 24, 4)])
        plt.colorbar(im, ax=axes[1], label="Важность")

    plt.tight_layout()
    if save:
        fn = os.path.join(plots_dir, f"attention_summary_{model_name.replace(' ', '_')}.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        logger.info("Summary attention: %s", fn)
    plt.close(fig)

    return importance


def compare_head_specialization(
    model: tf.keras.Model,
    X_sample: np.ndarray,
    model_name: str = "Transformer",
    plots_dir: str = "results/plots",
    save: bool = True,
) -> None:
    """
    Анализирует специализацию голов внимания.

    Для каждой головы вычисляем «лаг максимального внимания»:
    lag_h = argmax( mean_q(attn_h[q, :]) )

    Если head 1 специализируется на lag=24, а head 2 на lag=1 —
    это означает разделение труда: head 1 учит суточную периодичность,
    head 2 — краткосрочную динамику.
    """
    os.makedirs(plots_dir, exist_ok=True)
    weights = extract_attention_weights(model, X_sample)
    if not weights:
        return

    head_max_lags_all = {}
    for block_name, attn_matrix in weights.items():
        attn = attn_matrix[0]         # (H, Q, K)
        K = attn.shape[-1]
        # Для каждой головы — средний вектор внимания по всем запросам
        mean_per_head = attn.mean(axis=1)   # (H, K)
        max_lag_idx = np.argmax(mean_per_head, axis=-1)  # (H,)
        head_max_lags_all[block_name] = max_lag_idx

    n_blocks = len(head_max_lags_all)
    if n_blocks == 0:
        return

    first_block = list(head_max_lags_all.values())[0]
    num_heads = len(first_block)
    K = list(weights.values())[0].shape[-1]

    fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 4))
    if n_blocks == 1:
        axes = [axes]

    for ax, (block_name, max_lags) in zip(axes, head_max_lags_all.items()):
        bars = ax.barh(range(num_heads), K - 1 - max_lags,  # переводим в «лаг назад»
                       color=plt.cm.Set2.colors[:num_heads], edgecolor="black")
        ax.set_yticks(range(num_heads))
        ax.set_yticklabels([f"Head {h+1}" for h in range(num_heads)])
        ax.set_xlabel("Лаг максимального внимания (часов назад)")
        ax.set_title(f"{block_name}\n(Специализация голов)")
        ax.grid(True, alpha=0.3, axis="x")

        # Отметим ключевые лаги
        for lag in [24, 48, 168]:
            if lag <= K:
                ax.axvline(lag, color="red", ls="--", alpha=0.4, lw=1)

        for bar, lag in zip(bars, K - 1 - max_lags):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{lag}ч", va="center", fontsize=9)

    fig.suptitle(f"{model_name} — Специализация голов внимания", fontweight="bold")
    plt.tight_layout()
    if save:
        fn = os.path.join(plots_dir, f"head_specialization_{model_name.replace(' ', '_')}.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight")
        logger.info("Head specialization: %s", fn)
    plt.close(fig)
