# -*- coding: utf-8 -*-
"""
Facade module for transformer architectures.

Backward compatibility:
- Keeps historical imports from `models.transformer`.
- Delegates implementation to:
  - `models.transformer_blocks`
  - `models.vanilla`
  - `models.patchtst`
"""

from __future__ import annotations

from typing import Literal

import tensorflow as tf

from .transformer_blocks import (
    GatedResidualNetwork,
    LearnedQueryPooling,
    PreLNEncoderBlock,
    ProbSparseAttention,
    RevIN,
    RevINDenorm,
    RevINNorm,
    SinusoidalPE,
    StochasticDepth,
    Time2Vec,
    build_tft_lite,
    count_parameters,
    make_tft_windows,
    prepare_tft_covariates,
)
from .vanilla import build_vanilla_transformer
from .patchtst import build_patchtst


def build_transformer(model_type: Literal["vanilla", "patchtst"], **kwargs) -> tf.keras.Model:
    """
    Unified transformer builder API.

    Args:
        model_type: One of {"vanilla", "patchtst"}.
        **kwargs: Forwarded to the selected builder.

    Returns:
        Compiled Keras model.

    Raises:
        ValueError: If model_type is not supported.
    """
    if model_type == "vanilla":
        return build_vanilla_transformer(**kwargs)
    if model_type == "patchtst":
        return build_patchtst(**kwargs)
    raise ValueError(f"Unsupported model_type={model_type!r}. Use 'vanilla' or 'patchtst'.")


__all__ = [
    "count_parameters",
    "RevINNorm",
    "RevINDenorm",
    "RevIN",
    "StochasticDepth",
    "LearnedQueryPooling",
    "SinusoidalPE",
    "Time2Vec",
    "ProbSparseAttention",
    "PreLNEncoderBlock",
    "GatedResidualNetwork",
    "build_vanilla_transformer",
    "build_patchtst",
    "build_tft_lite",
    "prepare_tft_covariates",
    "make_tft_windows",
    "build_transformer",
]
