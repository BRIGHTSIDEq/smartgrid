# -*- coding: utf-8 -*-
"""models/vanilla.py — Vanilla Transformer builders."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from .transformer_blocks import build_vanilla_transformer as _build_vanilla_transformer


def build_vanilla_transformer(**kwargs: Any) -> tf.keras.Model:
    """
    Build vanilla transformer with input argument validation.

    Args:
        **kwargs: Same kwargs as legacy `models.transformer.build_vanilla_transformer`.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If key shape-related parameters are invalid.
    """
    history = int(kwargs.get("history_length", 48))
    horizon = int(kwargs.get("forecast_horizon", 24))
    n_features = int(kwargs.get("n_features", 11))
    d_model = int(kwargs.get("d_model", 128))
    num_heads = int(kwargs.get("num_heads", 8))
    if history <= 0 or horizon <= 0 or n_features <= 0:
        raise ValueError("history_length, forecast_horizon и n_features должны быть > 0")
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) должен делиться на num_heads ({num_heads})")
    return _build_vanilla_transformer(**kwargs)
