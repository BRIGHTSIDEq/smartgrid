# -*- coding: utf-8 -*-
"""
Рабочий каталог загрузок.

Каждая загрузка — отдельный каталог u_<32 hex> с исходным файлом, meta.json и
промежуточными таблицами. Состояние живёт в URL (/u/<uid>/…), без базы и
сессий: страницу можно обновить, открыть заново или отправить ссылку.

Идентификатор проверяется регулярным выражением до обращения к диску:
иначе «../» в адресе открыл бы доступ к произвольным файлам. В каталог
results/ приложение не пишет никогда — это результаты исследования.
"""

import json
import os
import re
import shutil
import time
import uuid
from typing import Any, BinaryIO, Dict, Optional

import pandas as pd

UID_RE = re.compile(r"^[0-9a-f]{32}$")


def default_root() -> str:
    """
    Каталог загрузок вне репозитория.

    Выгрузки — данные заказчика. В папке репозитория одна ошибка в
    .gitignore отправила бы их в push, поэтому по умолчанию они лежат в
    %LOCALAPPDATA%\smartgrid-web; путь переопределяется SMARTGRID_WEB_DIR.
    """
    env = os.environ.get("SMARTGRID_WEB_DIR")
    if env:
        return env
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "smartgrid-web")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
KEEP_DAYS = 7


class Workspace:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def path(self, uid: str, *parts: str) -> str:
        if not UID_RE.match(uid or ""):
            raise KeyError("неверный идентификатор загрузки")
        base = os.path.join(self.root, f"u_{uid}")
        if not os.path.isdir(base):
            raise KeyError("загрузка не найдена")
        return os.path.join(base, *parts)

    def create(self, stream: BinaryIO, filename: str,
               max_bytes: int = MAX_UPLOAD_BYTES) -> str:
        """
        Сохраняет поток кусками и возвращает uid.

        Размер проверяется по ходу записи: файл больше лимита не читается в
        память целиком и не остаётся на диске.
        """
        uid = uuid.uuid4().hex
        base = os.path.join(self.root, f"u_{uid}")
        os.makedirs(base)
        ext = os.path.splitext(filename or "")[1].lower() or ".csv"
        target = os.path.join(base, f"original{ext}")
        written = 0
        try:
            with open(target, "wb") as f:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise OverflowError(max_bytes)
                    f.write(chunk)
        except OverflowError:
            shutil.rmtree(base, ignore_errors=True)
            raise
        self.write_meta(uid, {"filename": os.path.basename(filename or "выгрузка.csv"),
                              "original": os.path.basename(target),
                              "size": written, "created": time.time()})
        return uid

    def meta(self, uid: str) -> Dict[str, Any]:
        with open(self.path(uid, "meta.json"), encoding="utf-8") as f:
            return json.load(f)

    def write_meta(self, uid: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        base = os.path.join(self.root, f"u_{uid}")
        path = os.path.join(base, "meta.json")
        meta: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        meta.update(updates)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        return meta

    def original(self, uid: str) -> str:
        return self.path(uid, self.meta(uid)["original"])

    def save_frame(self, uid: str, name: str, frame: pd.DataFrame) -> str:
        path = self.path(uid, f"{name}.csv")
        frame.to_csv(path, index=False, encoding="utf-8")
        return path

    def load_frame(self, uid: str, name: str) -> Optional[pd.DataFrame]:
        path = self.path(uid, f"{name}.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path, parse_dates=["timestamp"])

    def cleanup(self, keep_days: int = KEEP_DAYS) -> int:
        """Удаляет загрузки старше keep_days; только внутри своего каталога."""
        limit = time.time() - keep_days * 86400
        removed = 0
        for name in os.listdir(self.root):
            full = os.path.join(self.root, name)
            if name.startswith("u_") and os.path.isdir(full) and os.path.getmtime(full) < limit:
                shutil.rmtree(full, ignore_errors=True)
                removed += 1
        return removed
