# -*- coding: utf-8 -*-
"""
Конфигурация клиента в YAML.

Всё, что отличает одного заказчика от другого, собрано в одном файле:
источник и формат выгрузки, ценовая категория, ставки и плановые часы СО,
параметры накопителя, каталог моделей. Код не меняется между клиентами.
Пример — config/client_example.yaml.

Файл проверяется при загрузке: неизвестный ключ или неверный тип дают
ошибку с указанием поля. Опечатка в названии ставки иначе молча оставила
бы значение по умолчанию, и счёт был бы посчитан по чужому тарифу.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

import yaml

from data.connector import ExportFormat
from optimization.tariffs_ru import RuTariff


@dataclass
class BatteryConfig:
    capacity_kwh: Optional[float] = None       # None — типоразмер от пика нагрузки
    power_kw: Optional[float] = None
    capex_rub_per_kwh: float = 16_000.0


@dataclass
class ClientConfig:
    name: str
    data_path: str
    export: ExportFormat = field(default_factory=ExportFormat)
    tariff: RuTariff = field(default_factory=RuTariff)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    models_dir: Optional[str] = None
    target_issue_hour: int = 10               # час подачи заявки на сутки вперёд


def _build(cls, values: Dict[str, Any], path: str):
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{path}: неизвестные поля {', '.join(sorted(unknown))}; "
                         f"допустимы {', '.join(sorted(allowed))}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_client_config(path: str) -> ClientConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: ожидается словарь верхнего уровня")

    raw = dict(raw)
    export = _build(ExportFormat, raw.pop("export", {}) or {}, "export")
    tariff_raw = dict(raw.pop("tariff", {}) or {})
    if "peak_windows" in tariff_raw:
        tariff_raw["peak_windows"] = {int(k): tuple(v)
                                      for k, v in tariff_raw["peak_windows"].items()}
    tariff = _build(RuTariff, tariff_raw, "tariff")
    battery = _build(BatteryConfig, raw.pop("battery", {}) or {}, "battery")
    return _build(ClientConfig, {**raw, "export": export, "tariff": tariff,
                                 "battery": battery}, "клиент")
