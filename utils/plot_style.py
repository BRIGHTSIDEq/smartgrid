# -*- coding: utf-8 -*-
"""
utils/plot_style.py — единый стиль графиков для публикаций.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (10, 6)
WIDE_FIGSIZE = (14, 6)
TALL_FIGSIZE = (10, 10)


def get_palette() -> Dict[str, str]:
    """
    Цветовая палитра с хорошей контрастностью (colorblind-safe, publication friendly).
    """
    return {
        "primary": "#0072B2",     # blue
        "secondary": "#009E73",   # green
        "accent": "#D55E00",      # vermillion
        "warning": "#E69F00",     # orange
        "neutral": "#4D4D4D",     # dark gray
        "highlight": "#CC79A7",   # purple
        "baseline": "#000000",    # black
        "positive": "#009E73",
        "negative": "#D55E00",
    }


def apply_publication_style() -> None:
    """
    Применяет единый стиль matplotlib/seaborn для всех графиков проекта.
    """
    palette = list(get_palette().values())
    sns.set_theme(style="whitegrid", palette=palette)
    mpl.rcParams.update(
        {
            "figure.dpi": DEFAULT_DPI,
            "savefig.dpi": DEFAULT_DPI,
            "figure.figsize": DEFAULT_FIGSIZE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.loc": "best",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(
    fig: plt.Figure,
    output_path: str,
    *,
    save: bool = True,
    dpi: int = DEFAULT_DPI,
) -> Tuple[str, str, str] | Tuple[None, None, None]:
    """
    Сохраняет график в PNG + векторные форматы (PDF/SVG).
    output_path может быть с .png или без расширения.
    """
    if not save:
        return (None, None, None)

    base = Path(output_path)
    if base.suffix:
        base = base.with_suffix("")
    os.makedirs(base.parent, exist_ok=True)

    png_path = str(base.with_suffix(".png"))
    pdf_path = str(base.with_suffix(".pdf"))
    svg_path = str(base.with_suffix(".svg"))

    fig.savefig(png_path, dpi=max(dpi, DEFAULT_DPI), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, pdf_path, svg_path
