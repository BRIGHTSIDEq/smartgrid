# -*- coding: utf-8 -*-
"""
Иконки для шаблонов: {{ icon('circuit-battery') }}.

SVG читаются один раз при старте из static/vendor/tabler (Tabler Icons, MIT)
и static/icons (свои знаки в той же сетке 24 px: счётчик, опора ЛЭП) и
вставляются в разметку напрямую: так они наследуют цвет текста и не требуют
отдельных запросов. Размеры и класс задаются здесь, лишние атрибуты
исходников убираются.
"""

import os
import re
from typing import Dict, Optional

from markupsafe import Markup

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = (os.path.join(HERE, "static", "vendor", "tabler"), os.path.join(HERE, "static", "icons"))


def _load() -> Dict[str, str]:
    icons = {}
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.endswith(".svg"):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                svg = f.read()
            body = re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S).group(1)
            body = re.sub(r'<path stroke="none" d="M0 0h24v24H0z" fill="none"\s*/>', "", body)
            icons[name[:-4]] = " ".join(body.split())
    return icons


ICONS = _load()


def icon(name: str, size: int = 20, label: Optional[str] = None, cls: str = "") -> Markup:
    body = ICONS.get(name)
    if body is None:
        return Markup("")
    a11y = f'role="img" aria-label="{label}"' if label else 'aria-hidden="true"'
    return Markup(
        f'<svg class="i {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round" {a11y}>{body}</svg>')
