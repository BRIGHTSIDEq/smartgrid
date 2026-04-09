# -*- coding: utf-8 -*-
"""models/patchtst.py — PatchTST builders."""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from .transformer_blocks import build_patchtst as _build_patchtst


def build_patchtst(**kwargs: Any) -> tf.keras.Model:
    """
    Build PatchTST with input argument validation.

    Args:
        **kwargs: Same kwargs as legacy `models.transformer.build_patchtst`.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If shape-defining parameters are invalid.
    """
    history = int(kwargs.get("history_length", 48))
    patch_len = int(kwargs.get("patch_len", 8))
    stride = int(kwargs.get("stride", 4))
    n_features = int(kwargs.get("n_features", 11))
    d_model = int(kwargs.get("d_model", 128))
    num_heads = int(kwargs.get("num_heads", 8))
    if history <= 0 or patch_len <= 0 or stride <= 0 or n_features <= 0:
        raise ValueError("history_length, patch_len, stride, n_features должны быть > 0")
    if patch_len > history:
        raise ValueError("patch_len не может быть больше history_length")
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) должен делиться на num_heads ({num_heads})")
    return _build_patchtst(**kwargs)
