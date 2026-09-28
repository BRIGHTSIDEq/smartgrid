# -*- coding: utf-8 -*-
"""
Фоновые расчёты с прогрессом и кэшем.

Экономика по сотне суток считается секунды, чувствительность — десятки
секунд. Синхронный запрос на это время замораживал бы страницу, поэтому
расчёт уходит в фоновый поток, а страница опрашивает его состояние.

Поток один намеренно: машина может одновременно считать прогоны, и
интерфейс не должен отнимать у них все ядра.

Результат кэшируется по ключу параметров: повторный запрос с теми же
параметрами — например, на защите — отдаётся мгновенно.
"""

import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


def params_key(params: Dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class Job:
    id: str
    key: str
    title: str
    status: str = "running"          # running | done | error
    progress: float = 0.0
    stage: str = ""
    result: Any = None
    error: Any = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    # Флаг остановки: расчёт проверяет его между вариантами и завершается,
    # сохранив посчитанное.
    stop: threading.Event = field(default_factory=threading.Event)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started


class ImmediateExecutor(Executor):
    """Исполнитель для тестов: задача выполняется сразу, внутри запроса."""

    def submit(self, fn, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:          # noqa: BLE001
            future.set_exception(exc)
        return future


class JobRunner:
    def __init__(self, executor: Optional[Executor] = None):
        self.executor = executor or ThreadPoolExecutor(max_workers=1)
        self.jobs: Dict[str, Job] = {}
        self.by_key: Dict[str, str] = {}
        self._lock = threading.Lock()

    def submit(self, key: str, title: str,
               fn: Callable[["Job"], Any]) -> Job:
        """
        Запускает fn(job) или возвращает уже посчитанную задачу с тем же ключом.

        fn сама обновляет job.progress и job.stage по ходу работы.
        Задача с ошибкой не кэшируется: повтор запускает расчёт заново.
        """
        with self._lock:
            known = self.by_key.get(key)
            if known and self.jobs[known].status in ("running", "done"):
                return self.jobs[known]
            job = Job(id=uuid.uuid4().hex[:12], key=key, title=title)
            self.jobs[job.id] = job
            self.by_key[key] = job.id

        def run():
            try:
                job.result = fn(job)
                job.progress = 1.0
                job.status = "done"
            except BaseException as exc:      # noqa: BLE001
                job.error = exc
                job.status = "error"
            finally:
                job.finished = time.time()

        self.executor.submit(run)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)
