# -*- coding: utf-8 -*-
"""
Понятные сообщения об ошибках данных.

Функции проекта сообщают об ошибках входных данных через ValueError с
русским текстом — его можно показывать как есть. Остальные исключения
переводятся здесь в «что случилось и что сделать». Трассировку пользователь
не видит никогда: она уходит в лог.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class DataError(Exception):
    title: str
    detail: str = ""
    hint: str = ""

    def __str__(self) -> str:
        return f"{self.title}: {self.detail}" if self.detail else self.title


def friendly(exc: BaseException, encoding: Optional[str] = None) -> DataError:
    """Исключение → сообщение для пользователя."""
    if isinstance(exc, DataError):
        return exc
    if isinstance(exc, UnicodeDecodeError):
        other = "cp1251" if (encoding or "").lower().replace("-", "") != "cp1251" else "utf-8"
        return DataError("Файл не читается в выбранной кодировке",
                         f"кодировка {encoding or 'utf-8'} не подходит",
                         f"Выберите кодировку {other}: выгрузки АСКУЭ часто сохраняются в cp1251.")
    if isinstance(exc, pd.errors.EmptyDataError):
        return DataError("Файл пуст", "", "Проверьте, что выгрузка содержит строки с данными.")
    if isinstance(exc, pd.errors.ParserError):
        return DataError("Не удалось разобрать таблицу", str(exc).splitlines()[0],
                         "Проверьте разделитель столбцов и десятичный знак.")
    if isinstance(exc, KeyError):
        return DataError("Нет нужного столбца", str(exc).strip("'\""),
                         "Выберите столбцы в форме формата выгрузки.")
    if isinstance(exc, FileNotFoundError):
        return DataError("Нужный файл не найден", str(exc))
    if isinstance(exc, ValueError):
        return DataError("Данные не подходят", str(exc))
    return DataError("Внутренняя ошибка", type(exc).__name__,
                     "Подробности записаны в журнал приложения.")
