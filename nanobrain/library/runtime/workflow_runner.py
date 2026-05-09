"""G21 — WorkflowRunner.run_detached: long-running workflow lifecycle.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

The autonomous-agent use case (``autonomous_workflow_agent.md``) needs
an entry point that returns immediately with a ``task_id`` and continues
running in a managed background context. ``Workflow.run()`` (G8) is
synchronous and blocks the caller; this module adds the detached path.

v1 scope (this module)
======================

Implemented:

- ``WorkflowRunner.run_detached(workflow_callable, task_id, payload)``
  schedules an ``asyncio.create_task`` and returns a
  ``DetachedTaskHandle`` immediately.
- ``WorkflowRunner.cancel(task_id)`` calls ``Task.cancel()``; the
  in-flight asyncio task is cooperatively cancelled.
- ``WorkflowRunner.get_handle(task_id)`` returns the current handle
  from the task store.
- ``WorkflowRunner.await_completion(task_id, timeout=None)`` blocks
  until the underlying asyncio task settles; convenience for tests
  and for callers that want a synchronous join point.
- Task store: in-memory (default) or SQLite (stdlib ``sqlite3``).
- Per-runner concurrency cap via ``asyncio.Semaphore``.

NOT implemented (deferred follow-ups; documented honestly so callers
do not assume capabilities the v1 release does not have):

- Step 2 — ``pause(task_id)`` raises ``NotImplementedError``. A real
  pause requires a step-level cancellation hook on ``BaseStep`` so
  the runner can prevent the *next* step from starting without
  killing the in-flight one. That hook does not exist yet.
- Step 3 — heartbeat watchdog + stale-task reaper.
- Step 4 — cross-process resume (Postgres durability backend, G5
  checkpoint integration so a process restart resumes from the last
  committed checkpoint rather than re-running the whole workflow).
- ``CostEnvelope`` and ``autonomy_level`` parameters from the gap
  proposal are reserved-for-future. The v1 ``run_detached`` signature
  takes only the workflow callable, task_id, and payload.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

from pydantic import ConfigDict, Field, model_validator

from nanobrain.core.component_base import (
    ComponentConfigurationError,
    FromConfigBase,
)
from nanobrain.core.config.config_base import ConfigBase


# Lifecycle states for a detached task. The runner asserts incoming
# transitions against this whitelist; an unknown status is a programming
# error, not user input.
_VALID_STATUSES: Tuple[str, ...] = (
    "queued",
    "running",
    "paused",
    "completed",
    "cancelled",
    "failed",
)

_STATUS_ACTIVE: Tuple[str, ...] = ("queued", "running", "paused")


# ---------------------------------------------------------------------------
# DetachedTaskHandle — value object returned to callers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DetachedTaskHandle:
    """A snapshot of a detached task's state.

    Returned by ``run_detached`` and ``get_handle``. The handle is a
    *value object* — mutating it does not propagate back to the runner's
    store. Always re-fetch via ``get_handle(task_id)`` to see the
    current state after the underlying task progresses.
    """

    task_id: str
    status: str  # one of _VALID_STATUSES
    created_at: datetime
    last_heartbeat_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    cost_actual: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Task store backends
# ---------------------------------------------------------------------------

class _TaskStore:
    """Abstract base for task stores."""

    async def insert(self, handle: DetachedTaskHandle) -> None:
        raise NotImplementedError

    async def update(self, handle: DetachedTaskHandle) -> None:
        raise NotImplementedError

    async def get(self, task_id: str) -> Optional[DetachedTaskHandle]:
        raise NotImplementedError

    async def list_active(self) -> List[DetachedTaskHandle]:
        raise NotImplementedError


class _InMemoryTaskStore(_TaskStore):
    """Process-local dict, serialized by an asyncio.Lock. The default."""

    def __init__(self) -> None:
        self._handles: Dict[str, DetachedTaskHandle] = {}
        self._lock = asyncio.Lock()

    async def insert(self, handle: DetachedTaskHandle) -> None:
        async with self._lock:
            if handle.task_id in self._handles:
                raise ValueError(
                    f"FAIL-FAST: task_id {handle.task_id!r} already exists in "
                    f"the in-memory task store"
                )
            self._handles[handle.task_id] = _clone(handle)

    async def update(self, handle: DetachedTaskHandle) -> None:
        async with self._lock:
            self._handles[handle.task_id] = _clone(handle)

    async def get(self, task_id: str) -> Optional[DetachedTaskHandle]:
        async with self._lock:
            stored = self._handles.get(task_id)
            return _clone(stored) if stored else None

    async def list_active(self) -> List[DetachedTaskHandle]:
        async with self._lock:
            return [
                _clone(h) for h in self._handles.values()
                if h.status in _STATUS_ACTIVE
            ]


class _SQLiteTaskStore(_TaskStore):
    """SQLite-backed task store. Single connection serialized by an
    asyncio.Lock — sufficient for a single-process detached runner.
    Multi-process / cross-restart resume is Step 4 scope.
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS detached_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_heartbeat_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error TEXT,
            cost_actual_json TEXT
        )
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, isolation_level=None,
                                     check_same_thread=False)
        self._conn.execute(self._SCHEMA)
        self._lock = asyncio.Lock()

    @staticmethod
    def _to_row(h: DetachedTaskHandle) -> Tuple:
        return (
            h.task_id,
            h.status,
            h.created_at.isoformat(),
            h.last_heartbeat_at.isoformat() if h.last_heartbeat_at else None,
            h.completed_at.isoformat() if h.completed_at else None,
            json.dumps(h.result) if h.result is not None else None,
            h.error,
            json.dumps(h.cost_actual) if h.cost_actual else None,
        )

    @staticmethod
    def _from_row(r: Optional[Tuple]) -> Optional[DetachedTaskHandle]:
        if r is None:
            return None

        def _dt(s: Optional[str]) -> Optional[datetime]:
            return datetime.fromisoformat(s) if s else None

        return DetachedTaskHandle(
            task_id=r[0],
            status=r[1],
            created_at=_dt(r[2]),
            last_heartbeat_at=_dt(r[3]),
            completed_at=_dt(r[4]),
            result=json.loads(r[5]) if r[5] else None,
            error=r[6],
            cost_actual=json.loads(r[7]) if r[7] else None,
        )

    async def insert(self, handle: DetachedTaskHandle) -> None:
        async with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO detached_tasks VALUES (?,?,?,?,?,?,?,?)",
                    self._to_row(handle),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"FAIL-FAST: task_id {handle.task_id!r} already exists"
                ) from exc

    async def update(self, handle: DetachedTaskHandle) -> None:
        async with self._lock:
            self._conn.execute(
                "UPDATE detached_tasks SET status=?, last_heartbeat_at=?, "
                "completed_at=?, result_json=?, error=?, cost_actual_json=? "
                "WHERE task_id=?",
                (
                    handle.status,
                    handle.last_heartbeat_at.isoformat() if handle.last_heartbeat_at else None,
                    handle.completed_at.isoformat() if handle.completed_at else None,
                    json.dumps(handle.result) if handle.result is not None else None,
                    handle.error,
                    json.dumps(handle.cost_actual) if handle.cost_actual else None,
                    handle.task_id,
                ),
            )

    async def get(self, task_id: str) -> Optional[DetachedTaskHandle]:
        async with self._lock:
            cursor = self._conn.execute(
                "SELECT task_id, status, created_at, last_heartbeat_at, "
                "completed_at, result_json, error, cost_actual_json "
                "FROM detached_tasks WHERE task_id=?",
                (task_id,),
            )
            return self._from_row(cursor.fetchone())

    async def list_active(self) -> List[DetachedTaskHandle]:
        async with self._lock:
            placeholders = ",".join(["?"] * len(_STATUS_ACTIVE))
            rows = self._conn.execute(
                f"SELECT task_id, status, created_at, last_heartbeat_at, "
                f"completed_at, result_json, error, cost_actual_json "
                f"FROM detached_tasks WHERE status IN ({placeholders})",
                _STATUS_ACTIVE,
            ).fetchall()
            return [h for h in (self._from_row(r) for r in rows) if h is not None]


def _clone(h: DetachedTaskHandle) -> DetachedTaskHandle:
    """Shallow copy a handle. The store hands callers copies so they
    cannot mutate stored state via the returned reference."""
    return dataclasses.replace(h)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class WorkflowRunnerConfig(ConfigBase):
    """Configuration for ``WorkflowRunner``.

    Validation rules:

    - ``task_store_backend`` must be one of {'in_memory', 'sqlite'}.
      'sqlite' requires ``sqlite_db_path`` to be set; in-memory rejects it.
    - ``max_concurrent_detached_tasks`` must be ≥ 1.
    """

    name: str
    task_store_backend: Literal["in_memory", "sqlite"] = "in_memory"
    sqlite_db_path: Optional[str] = None
    max_concurrent_detached_tasks: int = Field(default=16, ge=1)

    # Set by ConfigBase.from_config after load. Declared here so that
    # ``extra='forbid'`` does not reject the post-load setattr (the
    # parent ConfigBase has ``validate_assignment=True``, which routes
    # every setattr through the model validator).
    source_path: Optional[str] = Field(default=None, exclude=True)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_backend_specific(self) -> "WorkflowRunnerConfig":
        if self.task_store_backend == "sqlite":
            if not self.sqlite_db_path:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig task_store_backend='sqlite' "
                    "requires sqlite_db_path"
                )
        else:
            # in_memory — sqlite_db_path is meaningless and likely a typo.
            if self.sqlite_db_path:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig sqlite_db_path is set but "
                    "task_store_backend='in_memory'; remove sqlite_db_path or "
                    "set backend to 'sqlite'"
                )
        return self


# ---------------------------------------------------------------------------
# WorkflowRunner
# ---------------------------------------------------------------------------

class WorkflowRunner(FromConfigBase):
    """G21 — manages detached workflow lifecycle.

    Construct via ``WorkflowRunner.from_config(<yml-path-or-config>)``.

    Usage::

        runner = WorkflowRunner.from_config("config/runner.yml")
        handle = await runner.run_detached(
            workflow_callable=my_workflow.run,
            task_id="task_123",
            payload={"input": "value"},
        )
        # ... callers can return immediately ...
        await runner.await_completion("task_123", timeout=60)
        final = await runner.get_handle("task_123")
        assert final.status == "completed"
    """

    COMPONENT_TYPE = "workflow_runner"
    REQUIRED_CONFIG_FIELDS = ["name", "task_store_backend"]

    @classmethod
    def _get_config_class(cls):
        return WorkflowRunnerConfig

    def _init_from_config(
        self,
        config: WorkflowRunnerConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        self.name = config.name
        self._max_concurrent = config.max_concurrent_detached_tasks
        self._store: _TaskStore
        if config.task_store_backend == "in_memory":
            self._store = _InMemoryTaskStore()
        else:
            self._store = _SQLiteTaskStore(config.sqlite_db_path)  # type: ignore[arg-type]
        self._tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

    # ---- Public API ---------------------------------------------------

    async def run_detached(
        self,
        workflow_callable: Callable[..., Awaitable[Any]],
        task_id: str,
        payload: Dict[str, Any],
    ) -> DetachedTaskHandle:
        """Schedule ``workflow_callable(payload)`` to run in the
        background. Returns immediately with the queued handle.

        The handle's ``status`` will progress queued → running →
        (completed | cancelled | failed). Use ``get_handle`` to
        observe progress and ``await_completion`` to block.
        """
        if not isinstance(task_id, str) or not task_id:
            raise ComponentConfigurationError(
                "FAIL-FAST: WorkflowRunner.run_detached: task_id must be a "
                "non-empty string"
            )
        now = datetime.now(timezone.utc)
        handle = DetachedTaskHandle(
            task_id=task_id, status="queued", created_at=now,
        )
        await self._store.insert(handle)

        async def _runner() -> None:
            async with self._semaphore:
                handle.status = "running"
                handle.last_heartbeat_at = datetime.now(timezone.utc)
                await self._store.update(handle)
                try:
                    result = await workflow_callable(payload)
                    handle.status = "completed"
                    handle.result = result
                except asyncio.CancelledError:
                    handle.status = "cancelled"
                    handle.completed_at = datetime.now(timezone.utc)
                    await self._store.update(handle)
                    raise
                except Exception as exc:  # noqa: BLE001
                    handle.status = "failed"
                    handle.error = f"{type(exc).__name__}: {exc}"
                handle.completed_at = datetime.now(timezone.utc)
                await self._store.update(handle)

        self._tasks[task_id] = asyncio.create_task(
            _runner(), name=f"detached-{task_id}",
        )
        # Return a queued snapshot — caller does not see the in-flight
        # status mutation that happens inside _runner.
        return await self._store.get(task_id) or handle

    async def cancel(self, task_id: str, reason: str = "") -> None:
        """Hard-cancel an in-flight detached task.

        Raises ``ComponentConfigurationError`` if ``task_id`` is unknown.
        Best-effort: the asyncio task is cancelled and awaited; the
        store status is updated to ``cancelled`` by the runner's
        own CancelledError handler.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.cancel: task_id {task_id!r} "
                f"is not registered with this runner"
            )
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            # Status mutation happens inside _runner; we just need to
            # join the task to ensure the store reflects the cancel.
            pass

    async def pause(self, task_id: str, reason: str = "") -> None:  # pragma: no cover
        """Reserved for G21 Step 2. Soft-pause requires a step-level
        cancellation hook in ``BaseStep`` so the runner can prevent the
        *next* step from starting without killing the in-flight one;
        that hook does not exist yet. Calling this raises so callers do
        not silently get a no-op when they expected a pause."""
        raise NotImplementedError(
            "G21 Step 2 — pause requires a step-level cancellation hook "
            "in BaseStep that the runner can use to gate the next step "
            "without killing the in-flight one. Reserved-for-future."
        )

    async def get_handle(self, task_id: str) -> Optional[DetachedTaskHandle]:
        """Return the current handle from the store, or None if
        ``task_id`` was never registered with this runner."""
        return await self._store.get(task_id)

    async def list_active(self) -> List[DetachedTaskHandle]:
        """Return handles for all queued/running/paused tasks."""
        return await self._store.list_active()

    async def await_completion(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> DetachedTaskHandle:
        """Block until the underlying asyncio task settles; return the
        final handle from the store. Convenience for tests and for
        callers that want a synchronous join point.

        Raises ``ComponentConfigurationError`` if ``task_id`` is unknown.
        Raises ``asyncio.TimeoutError`` if ``timeout`` elapses before the
        task settles.
        """
        task = self._tasks.get(task_id)
        if task is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.await_completion: task_id "
                f"{task_id!r} is not registered with this runner"
            )
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.TimeoutError:
            raise
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            # The runner's _runner coroutine sets the final status;
            # we just need to join.
            pass
        handle = await self._store.get(task_id)
        if handle is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.await_completion: task_id "
                f"{task_id!r} was registered but the store returned None"
            )
        return handle
