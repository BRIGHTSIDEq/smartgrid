# -*- coding: utf-8 -*-
"""
Форматирование чисел для шаблонов — одно место на всё приложение.

Правила (docs/design/VISUAL_SYSTEM.md): десятичная запятая; разряды узким
неразрывным пробелом; минус U+2212; единица через неразрывный пробел;
годовые рубли — три значащие цифры, без копеек.
"""

import math
from typing import Any, Optional

NNBSP = " "      # узкий неразрывный пробел — разряды
NBSP = " "       # неразрывный пробел — между числом и единицей
MINUS = "−"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (isinstance(value, float) and
                                                    (math.isnan(value)))


def num(value: Any, digits: int = 0, group_from: int = 1000) -> str:
    """Число с десятичной запятой и разрядами."""
    if value is None or not _is_number(value):
        return "—"
    if math.isinf(value):
        return "∞"
    text = f"{abs(value):,.{digits}f}".replace(",", " ").replace(".", ",")
    if abs(value) < group_from:
        text = text.replace(" ", "")
    text = text.replace(" ", NNBSP)
    return (MINUS if value < 0 and float(text.replace(",", ".").replace(NNBSP, "")) != 0
            else "") + text


def sig3(value: float) -> str:
    """Три значащие цифры без единицы."""
    if value == 0:
        return "0"
    digits = max(0, 2 - int(math.floor(math.log10(abs(value)))))
    return num(round(value, digits), digits)


def rub(value: Any) -> str:
    """Рубли: «842 тыс. ₽», «1,52 млн ₽», «9 800 ₽»."""
    if value is None or not _is_number(value):
        return "—"
    a = abs(value)
    if a >= 1e9:
        return f"{sig3(value / 1e9)}{NBSP}млрд{NBSP}₽"
    if a >= 1e6:
        return f"{sig3(value / 1e6)}{NBSP}млн{NBSP}₽"
    if a >= 1e4:
        return f"{sig3(value / 1e3)}{NBSP}тыс.{NBSP}₽"
    return f"{num(value)}{NBSP}₽"


def rub_range(lo: Any, hi: Any) -> str:
    """Диапазон рублей в одних единицах: «1,2–1,9 млн ₽», «от −0,2 до 0,4 млн ₽»."""
    if not (_is_number(lo) and _is_number(hi)):
        return "—"
    top = max(abs(lo), abs(hi))
    scale, unit = (1e9, "млрд") if top >= 1e9 else (1e6, "млн") if top >= 1e6 else \
        (1e3, "тыс.") if top >= 1e4 else (1, "")
    a, b = sig3(lo / scale), sig3(hi / scale)
    tail = f"{NBSP}{unit}{NBSP}₽" if unit else f"{NBSP}₽"
    if lo < 0:
        return f"от {a} до {b}{tail}"
    return f"{a}–{b}{tail}"


def rub_between(lo: Any, hi: Any) -> str:
    """«от 5,70 до 5,99 млн ₽» — для фраз вида «скорее всего от … до …»."""
    text = rub_range(lo, hi)
    if text == "—" or text.startswith("от "):
        return text
    return "от " + text.replace("–", " до ", 1)


def years(value: Any) -> str:
    """Окупаемость: «6,4 года», «больше 25 лет», «не окупается»."""
    if value is None or not _is_number(value) or math.isinf(value) or value <= 0:
        return "не окупается"
    if value > 25:
        return "больше 25 лет"
    n = round(value, 1)
    whole = int(n)
    word = "года" if n != whole or 2 <= whole % 10 <= 4 and not 12 <= whole % 100 <= 14 \
        else "год" if whole % 10 == 1 and whole % 100 != 11 else "лет"
    return f"{num(n, 0 if n == whole else 1)}{NBSP}{word}"


def years_range(lo: Any, hi: Any) -> str:
    """
    Окупаемость по краям разброса: «9–14 лет».

    Слова те же, что у years(): «больше 25 лет» — конечный, но слишком
    долгий срок, «не окупается» — экономии нет. Прежде years_range называл
    «не окупается» и срок в 40 лет, а years — «больше 25 лет».
    """
    vals = [v for v in (lo, hi) if _is_number(v)]
    finite = [v for v in vals if not math.isinf(v) and v > 0]
    if not finite:
        return "не окупается"
    within = sorted(v for v in finite if v <= 25)
    if not within:
        return "больше 25 лет"
    if len(within) == 1:
        if len(finite) == 2:
            return f"от {years(within[0])} до больше 25 лет"
        if len(vals) == 2:
            return f"от {years(within[0])}, в неблагоприятном варианте — не окупается"
        return years(within[0])
    a, b = within
    if round(a, 1) == round(b, 1):
        return years(a)
    # Слово согласуется с верхним краем в том виде, в каком он показан:
    # «3,1–3,3 года», «9,2–14 лет» (14,1 выводится как 14).
    word = years(round(b) if b >= 10 else round(b, 1)).split(NBSP)[-1]
    return f"{num(a, 0 if a >= 10 else 1)}–{num(b, 0 if b >= 10 else 1)}{NBSP}{word}"


def pct(value: Any, digits: int = 0, signed: bool = False) -> str:
    if value is None or not _is_number(value):
        return "—"
    text = num(abs(value) * 100, digits)
    sign = (MINUS if value < 0 else "+" if signed and value > 0 else "")
    return f"{sign}{text}{NBSP}%"


def kwh(value: Any, digits: int = 0) -> str:
    return f"{num(value, digits)}{NBSP}кВт·ч" if _is_number(value) else "—"


def dmy(value: Any) -> str:
    """ISO-дата или Timestamp → «14.04.2025»."""
    text = str(value)[:10]
    parts = text.split("-")
    return f"{parts[2]}.{parts[1]}.{parts[0]}" if len(parts) == 3 else text


def register(env) -> None:
    env.filters.update(num=num, rub=rub, rub_range=rub_range, years=years,
                       years_range=years_range, pct=pct, kwh=kwh, dmy=dmy,
                       rub_between=rub_between)
