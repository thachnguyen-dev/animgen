"""Long work runs as one task at a time on a worker thread; the UI polls its progress and can cancel it.

Cancel is honoured between steps (between frames and between retries), never in the middle of a model call.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import AppError


class Cancelled(Exception):
    pass


@dataclass
class Task:
    id: str
    kind: str
    state: str = "queued"          # queued, running, completed, failed, cancelled
    stage: str = ""                # loading_models, generating, retrying, exporting, ...
    frame: int = 0                 # 1..8 while generating
    frames_total: int = 0
    attempt: int = 0               # 0 = first try, 1.. = automatic retry number
    done: int = 0
    total: int = 0
    message: str = ""
    error: dict | None = None
    result: Any = None
    warnings: list[dict] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kw: Any) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def check_cancel(self) -> None:
        if self.cancel_requested:
            raise Cancelled()

    def to_json(self) -> dict:
        with self._lock:
            return {"id": self.id, "kind": self.kind, "state": self.state, "stage": self.stage,
                    "frame": self.frame, "frames_total": self.frames_total, "attempt": self.attempt,
                    "done": self.done, "total": self.total, "message": self.message, "error": self.error,
                    "result": self.result, "warnings": list(self.warnings),
                    "elapsed": round((self.finished or time.time()) - self.started, 1)}


class TaskRunner:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._current: Task | None = None
        self._lock = threading.Lock()

    def busy(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.state in ("queued", "running")

    def start(self, kind: str, fn: Callable[[Task], Any]) -> Task:
        with self._lock:
            if self._current is not None and self._current.state in ("queued", "running"):
                raise AppError("busy")
            task = Task(id=uuid.uuid4().hex[:12], kind=kind)
            self._tasks[task.id] = task
            self._current = task
            # keep the last few tasks only
            if len(self._tasks) > 50:
                for old in list(self._tasks)[:-50]:
                    self._tasks.pop(old, None)
        threading.Thread(target=self._run, args=(task, fn), name=f"task-{kind}", daemon=True).start()
        return task

    def _run(self, task: Task, fn: Callable[[Task], Any]) -> None:
        task.update(state="running")
        try:
            result = fn(task)
            task.update(state="completed", result=result, finished=time.time())
        except Cancelled:
            task.update(state="cancelled", error=AppError("cancelled").to_json(), finished=time.time())
        except AppError as exc:
            task.update(state="failed", error=exc.to_json(), finished=time.time())
        except Exception as exc:  # noqa: BLE001 — every failure must reach the UI explicitly
            traceback.print_exc()
            err = AppError("inference_failed", detail=f"{type(exc).__name__}: {exc}")
            task.update(state="failed", error=err.to_json(), finished=time.time())

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        t = self._tasks.get(task_id)
        if t is None:
            return False
        t.update(cancel_requested=True)
        return True

    def wait(self, task_id: str, timeout: float = 60.0) -> Task:
        t = self._tasks[task_id]
        end = time.time() + timeout
        while t.state in ("queued", "running") and time.time() < end:
            time.sleep(0.01)
        return t
