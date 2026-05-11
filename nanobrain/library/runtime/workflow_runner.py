"""G21 — WorkflowRunner.run_detached: long-running workflow lifecycle.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

The autonomous-agent use case (``autonomous_workflow_agent.md``) needs
an entry point that returns immediately with a ``task_id`` and continues
running in a managed background context. ``Workflow.run()`` (G8) is
synchronous and blocks the caller; this module adds the detached path.

v1 + Step 2 scope (this module)
================================

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
- **G21 Step 2** — ``pause(task_id)`` / ``resume(task_id)`` /
  ``is_paused(task_id)`` cooperative-pause primitives. A pause
  request:
    * Sets a per-task ``_PauseSignal`` contextvar that is published
      inside the runner coroutine wrapping the workflow callable.
      Step authors who want pause-aware behavior call
      ``current_pause_signal()`` and check ``is_paused()`` between
      step boundaries; the framework does NOT enforce pause inside
      the standard step-execution path (that requires per-step
      cooperation hooks that are out-of-scope for Step 2).
    * Updates the task handle's ``status`` to ``"paused"`` so
      callers polling ``get_handle`` see it immediately.
    * In-flight asyncio tasks are NOT cancelled; the workflow
      continues to run unless its step authors honor the signal.
  Brutal truth: a workflow that does NOT consult the contextvar
  will run to completion regardless of pause requests. Pause is a
  cooperative protocol, not a hard preemption. ``resume`` clears
  the signal and updates status back to ``"running"``.

NOT implemented (deferred follow-ups; documented honestly so callers
do not assume capabilities the v1 release does not have):

- Step 3 — heartbeat watchdog + stale-task reaper.
- Step 4 — cross-process resume (Postgres durability backend, G5
  checkpoint integration so a process restart resumes from the last
  committed checkpoint rather than re-running the whole workflow).
- ``CostEnvelope`` and ``autonomy_level`` parameters from the gap
  proposal are reserved-for-future. The v1 ``run_detached`` signature
  takes only the workflow callable, task_id, and payload.
- The standard ``BaseStep`` does not yet consult the pause signal;
  pause-aware step authoring is on the framework user today. A
  future framework change can wire BaseStep to check
  ``current_pause_signal()`` automatically without touching this
  module — that is the contract Step 2 ships.
"""

from __future__ import annotations

import asyncio
import contextvars
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
    # G27 Option A v1 (2026-05-11) — soft-suspend on
    # ApprovalPendingError. Distinct from 'paused' which is a
    # cooperative-pause signal at step boundaries; 'suspended' means
    # the workflow ITSELF raised ApprovalPendingError and the run is
    # waiting for an external approval resolution. ``resume(task_id)``
    # re-spawns the asyncio task with the original callable + payload;
    # the deterministic approval_id (G27 P6+a default) means the
    # re-run finds the now-resolved Approval and returns.
    "suspended",
    "completed",
    "cancelled",
    "failed",
)

_STATUS_ACTIVE: Tuple[str, ...] = (
    "queued", "running", "paused", "suspended",
)


# ---------------------------------------------------------------------------
# G21 Step 2 — cooperative pause signal
# ---------------------------------------------------------------------------

class PauseSignal:
    """Per-task pause signal published by the runner inside the
    coroutine that wraps the workflow callable.

    Step authors who want pause-aware behavior call
    :func:`current_pause_signal` from inside ``process()`` and consult
    :meth:`is_paused` between meaningful work boundaries.
    Implementations that want to BLOCK on pause can ``await
    signal.wait_until_resumed()``; implementations that want to
    cooperative-exit can check ``signal.is_paused()`` and return early.

    The signal is a thin wrapper around an :class:`asyncio.Event` —
    the framework reuses asyncio's primitives rather than rolling its
    own polling loop. ``set()`` semantics are inverted from a normal
    Event: the event is *set* when the task is RUNNING (callers can
    proceed) and *cleared* when paused (callers wait).
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._event.set()  # default: not paused

    def is_paused(self) -> bool:
        return not self._event.is_set()

    def pause(self) -> None:
        self._event.clear()

    def resume(self) -> None:
        self._event.set()

    async def wait_until_resumed(self) -> None:
        """Block until the signal is resumed. No-op if not paused."""
        await self._event.wait()


# Module-global contextvar. Asyncio-task-local via PEP 567, so two
# concurrent workflow runs see independent signals. None when no
# detached run is active.
_current_pause_signal: contextvars.ContextVar[Optional[PauseSignal]] = (
    contextvars.ContextVar("current_pause_signal", default=None)
)


def current_pause_signal() -> Optional[PauseSignal]:
    """Return the pause signal for the current detached run, or None
    when called outside a detached run context. Step authors who want
    pause-aware behavior call this from inside ``process()`` and
    consult the returned signal."""
    return _current_pause_signal.get()


def _is_approval_pending(exc: BaseException) -> bool:
    """G27 Option A — duck-type detect ApprovalPendingError without
    importing the deferred_hitl_step module at workflow_runner import
    time. Two signals must both hold:

      * the exception class is exactly named ``ApprovalPendingError``
      * the exception carries an ``approval_id`` attribute

    Both conditions guard against false positives if another package
    ships a same-named exception with different semantics.
    """
    return (
        type(exc).__name__ == "ApprovalPendingError"
        and hasattr(exc, "approval_id")
    )


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
    # G27 Option A v1 — populated when status == "suspended".
    # Carries approval_id + step_name + prompt so external resolvers
    # know what to look up in the ApprovalStore.
    suspension_info: Optional[Dict[str, Any]] = None
    # G27 Option B evaluation instrumentation (2026-05-11).
    # Counts how many resume_suspended() calls this task has gone
    # through. Combined with cost_actual (G26 tracking) operators
    # measure "what fraction of cumulative cost is re-run cost?" —
    # the load-bearing question for deciding whether Option A's
    # re-run-from-start semantic is acceptable in their deployment.
    # See nanobrain/docs/g27_g21_wiring_design.md "Decision needed"
    # section + the G27-Option-B evaluation framework note.
    resume_count: int = 0


# ---------------------------------------------------------------------------
# Task store backends
# ---------------------------------------------------------------------------

class TaskStore:
    """Abstract base for task stores. Public extension point.

    Three implementations ship in this module:
      - InMemoryTaskStore (default; tests + single-process dev)
      - SqliteTaskStore (single-process production with durability)
      - PostgresTaskStore (multi-process production; psycopg 3 optional dep)

    Deployments with other backends (Redis, DynamoDB, MongoDB,
    custom-managed-DB) subclass this and implement the four async
    methods. The runner consumes the abstract interface, so any
    conforming backend works without runner changes.

    Wire a custom backend at runner-construction time by either:
      (a) instantiating WorkflowRunner.from_config(...) then assigning
          ``runner._store = MyStore(...)`` BEFORE the first run_detached
          call (programmatic; bypasses the YAML factory); or
      (b) extending WorkflowRunnerConfig.task_store_backend with a new
          Literal value AND patching _init_from_config to dispatch to
          your backend (canonical; recommended for first-class support).
    """

    async def insert(self, handle: DetachedTaskHandle) -> None:
        raise NotImplementedError

    async def update(self, handle: DetachedTaskHandle) -> None:
        raise NotImplementedError

    async def get(self, task_id: str) -> Optional[DetachedTaskHandle]:
        raise NotImplementedError

    async def list_active(self) -> List[DetachedTaskHandle]:
        raise NotImplementedError


# Backwards-compat alias — keep the underscore-prefixed name working
# for any external code that imported it. Will be removed after one
# release.
_TaskStore = TaskStore


class InMemoryTaskStore(TaskStore):
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


# Backwards-compat alias.
_InMemoryTaskStore = InMemoryTaskStore


class SqliteTaskStore(TaskStore):
    """SQLite-backed task store. Single connection serialized by an
    asyncio.Lock — sufficient for a single-process detached runner.
    Multi-process / cross-restart resume is Step 4 scope.
    """

    # G27 wiring (2026-05-11): ``suspension_info_json`` column added
    # for soft-suspend persistence + ``resume_count`` for Option B
    # evaluation instrumentation. Existing databases get the columns
    # via the idempotent ALTERs below.
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS detached_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_heartbeat_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error TEXT,
            cost_actual_json TEXT,
            suspension_info_json TEXT,
            resume_count INTEGER NOT NULL DEFAULT 0
        )
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, isolation_level=None,
                                     check_same_thread=False)
        self._conn.execute(self._SCHEMA)
        # G27 wiring — additive ALTERs for tables created before
        # these columns existed. SQLite raises "duplicate column
        # name" when the column is already present; we swallow that
        # specific error class so the migration is idempotent.
        for stmt in (
            "ALTER TABLE detached_tasks ADD COLUMN suspension_info_json TEXT",
            "ALTER TABLE detached_tasks "
            "ADD COLUMN resume_count INTEGER NOT NULL DEFAULT 0",
        ):
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
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
            json.dumps(h.suspension_info) if h.suspension_info else None,
            int(h.resume_count or 0),
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
            # G27 wiring — column may be absent on rows written by a
            # pre-migration build. Tolerate len(r) of 8 / 9 / 10.
            suspension_info=(
                json.loads(r[8]) if len(r) > 8 and r[8] else None
            ),
            resume_count=int(r[9]) if len(r) > 9 and r[9] is not None else 0,
        )

    async def insert(self, handle: DetachedTaskHandle) -> None:
        async with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO detached_tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
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
                "completed_at=?, result_json=?, error=?, cost_actual_json=?, "
                "suspension_info_json=?, resume_count=? "
                "WHERE task_id=?",
                (
                    handle.status,
                    handle.last_heartbeat_at.isoformat() if handle.last_heartbeat_at else None,
                    handle.completed_at.isoformat() if handle.completed_at else None,
                    json.dumps(handle.result) if handle.result is not None else None,
                    handle.error,
                    json.dumps(handle.cost_actual) if handle.cost_actual else None,
                    json.dumps(handle.suspension_info) if handle.suspension_info else None,
                    int(handle.resume_count or 0),
                    handle.task_id,
                ),
            )

    async def get(self, task_id: str) -> Optional[DetachedTaskHandle]:
        async with self._lock:
            cursor = self._conn.execute(
                "SELECT task_id, status, created_at, last_heartbeat_at, "
                "completed_at, result_json, error, cost_actual_json, "
                "suspension_info_json, resume_count "
                "FROM detached_tasks WHERE task_id=?",
                (task_id,),
            )
            return self._from_row(cursor.fetchone())

    async def list_active(self) -> List[DetachedTaskHandle]:
        async with self._lock:
            placeholders = ",".join(["?"] * len(_STATUS_ACTIVE))
            rows = self._conn.execute(
                f"SELECT task_id, status, created_at, last_heartbeat_at, "
                f"completed_at, result_json, error, cost_actual_json, "
                f"suspension_info_json, resume_count "
                f"FROM detached_tasks WHERE status IN ({placeholders})",
                _STATUS_ACTIVE,
            ).fetchall()
            return [h for h in (self._from_row(r) for r in rows) if h is not None]


# Backwards-compat alias.
_SQLiteTaskStore = SqliteTaskStore


class PostgresTaskStore(TaskStore):
    """G21 Step 4 — Postgres-backed TaskStore for cross-process resume.

    Uses ``psycopg`` (psycopg 3) as a lazy import — it is NOT a hard
    dependency of nanobrain. Install with:

        pip install psycopg[binary]

    Instantiating this class without psycopg installed raises
    ``ImportError`` with the install hint.

    Usage::

        from nanobrain.library.runtime import PostgresTaskStore
        store = PostgresTaskStore(dsn="postgresql://user:pw@host/db")
        await store.initialize()  # create the table if missing
        # then attach to runner: runner._store = store

    DDL: see ``_SCHEMA``. The table is created with ``IF NOT EXISTS``
    so deployments can pre-create it via migrations and just rely on
    this code at runtime.

    Cross-process semantics (the value-add over SqliteTaskStore):
    multiple worker processes can each hold a WorkflowRunner pointed
    at the same Postgres DB. Each process sees the union of all
    tasks; ``list_active`` returns running tasks across the whole
    fleet. The G21 Step 3 watchdog reaps tasks whose heartbeat went
    stale across the fleet — useful for crash recovery (a worker
    process died mid-task; the watchdog on a surviving worker reaps
    the orphan).

    Limitations (Step 4 honest deferrals):
    - The runner does NOT YET resume orphan tasks (i.e., re-launch
      a workflow whose runner crashed). Orphans are reaped to
      ``failed`` by the watchdog; rebuilding via G5 ResumeStep is
      the deployment author's job.
    - Connection pooling is not implemented; v1 uses a single
      connection serialized by an asyncio.Lock. For higher throughput,
      subclass and swap in psycopg's AsyncConnectionPool.
    """

    # G27 wiring (2026-05-11): ``suspension_info_json`` column added
    # for soft-suspend persistence + ``resume_count`` for Option B
    # evaluation instrumentation. Existing databases get the columns
    # via the idempotent ALTERs in ``initialize()``.
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS nanobrain_detached_tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_heartbeat_at TEXT,
            completed_at TEXT,
            result_json TEXT,
            error TEXT,
            cost_actual_json TEXT,
            suspension_info_json TEXT,
            resume_count INTEGER NOT NULL DEFAULT 0
        )
    """

    def __init__(self, dsn: str, *, table_name: str = "nanobrain_detached_tasks") -> None:
        try:
            import psycopg as _psycopg  # noqa: F401  (probe only)
        except ImportError as exc:
            raise ImportError(
                "PostgresTaskStore requires psycopg (psycopg 3). Install "
                "with: pip install 'psycopg[binary]'"
            ) from exc
        self._dsn = dsn
        self._table = table_name
        self._conn = None  # opened lazily in initialize()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Open the connection and ensure the table exists. Idempotent."""
        if self._initialized:
            return
        import psycopg
        self._conn = await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True,
        )
        # Schema; replace the table-name placeholder safely (we own the
        # name, not user input, but psycopg.sql.SQL would be more idiomatic).
        async with self._conn.cursor() as cur:
            await cur.execute(self._SCHEMA.replace(
                "nanobrain_detached_tasks", self._table,
            ))
            # G27 wiring — additive ALTERs for tables created before
            # these columns existed. Postgres supports ``IF NOT EXISTS``
            # on ADD COLUMN since 9.6 (a 2016 minimum); the integration
            # already requires psycopg 3 which depends on a much later
            # server. Idempotent + safe across re-initialize().
            await cur.execute(
                f"ALTER TABLE {self._table} "
                f"ADD COLUMN IF NOT EXISTS suspension_info_json TEXT"
            )
            await cur.execute(
                f"ALTER TABLE {self._table} "
                f"ADD COLUMN IF NOT EXISTS resume_count INTEGER NOT NULL "
                f"DEFAULT 0"
            )
        self._initialized = True

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            self._initialized = False

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
            json.dumps(h.suspension_info) if h.suspension_info else None,
            int(h.resume_count or 0),
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
            # G27 wiring — tolerate rows that predate the columns.
            suspension_info=(
                json.loads(r[8]) if len(r) > 8 and r[8] else None
            ),
            resume_count=int(r[9]) if len(r) > 9 and r[9] is not None else 0,
        )

    async def insert(self, handle: DetachedTaskHandle) -> None:
        if not self._initialized:
            await self.initialize()
        import psycopg
        async with self._lock:
            try:
                async with self._conn.cursor() as cur:
                    await cur.execute(
                        f"INSERT INTO {self._table} "
                        f"VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        self._to_row(handle),
                    )
            except psycopg.errors.UniqueViolation as exc:
                raise ValueError(
                    f"FAIL-FAST: task_id {handle.task_id!r} already exists"
                ) from exc

    async def update(self, handle: DetachedTaskHandle) -> None:
        if not self._initialized:
            await self.initialize()
        async with self._lock:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE {self._table} SET status=%s, last_heartbeat_at=%s, "
                    f"completed_at=%s, result_json=%s, error=%s, "
                    f"cost_actual_json=%s, suspension_info_json=%s, "
                    f"resume_count=%s "
                    f"WHERE task_id=%s",
                    (
                        handle.status,
                        handle.last_heartbeat_at.isoformat() if handle.last_heartbeat_at else None,
                        handle.completed_at.isoformat() if handle.completed_at else None,
                        json.dumps(handle.result) if handle.result is not None else None,
                        handle.error,
                        json.dumps(handle.cost_actual) if handle.cost_actual else None,
                        json.dumps(handle.suspension_info) if handle.suspension_info else None,
                        int(handle.resume_count or 0),
                        handle.task_id,
                    ),
                )

    async def get(self, task_id: str) -> Optional[DetachedTaskHandle]:
        if not self._initialized:
            await self.initialize()
        async with self._lock:
            async with self._conn.cursor() as cur:
                await cur.execute(
                    f"SELECT task_id, status, created_at, last_heartbeat_at, "
                    f"completed_at, result_json, error, cost_actual_json, "
                    f"suspension_info_json, resume_count "
                    f"FROM {self._table} WHERE task_id=%s",
                    (task_id,),
                )
                return self._from_row(await cur.fetchone())

    async def list_active(self) -> List[DetachedTaskHandle]:
        if not self._initialized:
            await self.initialize()
        async with self._lock:
            placeholders = ",".join(["%s"] * len(_STATUS_ACTIVE))
            async with self._conn.cursor() as cur:
                await cur.execute(
                    f"SELECT task_id, status, created_at, last_heartbeat_at, "
                    f"completed_at, result_json, error, cost_actual_json, "
                    f"suspension_info_json, resume_count "
                    f"FROM {self._table} WHERE status IN ({placeholders})",
                    _STATUS_ACTIVE,
                )
                rows = await cur.fetchall()
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
    task_store_backend: Literal["in_memory", "sqlite", "postgres"] = "in_memory"
    sqlite_db_path: Optional[str] = None
    postgres_dsn: Optional[str] = Field(
        default=None,
        description="DSN for the postgres backend, e.g. "
                    "'postgresql://user:pw@host/db'. Required when "
                    "task_store_backend='postgres'. psycopg 3 must be "
                    "installed (pip install 'psycopg[binary]')."
    )
    postgres_table_name: str = Field(
        default="nanobrain_detached_tasks",
        description="Table name for the postgres backend. Default "
                    "'nanobrain_detached_tasks'. Override when sharing "
                    "a database with other applications."
    )
    max_concurrent_detached_tasks: int = Field(default=16, ge=1)

    # G21 Step 3 — heartbeat watchdog. The runner spawns a background
    # asyncio task that updates last_heartbeat_at on every running
    # task at this interval, and reaps tasks whose last_heartbeat_at
    # is older than `watchdog_stale_threshold_seconds`.
    #
    # Defaults from the gap proposal (autonomous_workflow_agent.md §5.2):
    # 60s heartbeat, 600s stale threshold (10x the heartbeat). Set
    # heartbeat_interval_seconds to 0 to DISABLE the watchdog entirely
    # (useful for tests that want deterministic timestamps).
    heartbeat_interval_seconds: float = Field(
        default=60.0, ge=0.0,
        description="How often the watchdog refreshes last_heartbeat_at "
                    "for running tasks. Set to 0 to disable the watchdog "
                    "entirely (last_heartbeat_at then only updates at "
                    "task lifecycle transitions)."
    )
    watchdog_stale_threshold_seconds: float = Field(
        default=600.0, ge=0.01,
        description="A running task whose last_heartbeat_at is older "
                    "than this threshold is reaped — its status is set "
                    "to 'failed' with an explicit error string. Should "
                    "be a multiple of heartbeat_interval_seconds (10x "
                    "is the proposal default) to tolerate transient "
                    "scheduling jitter."
    )

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
            if self.postgres_dsn:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig postgres_dsn set but "
                    "task_store_backend='sqlite'; remove postgres_dsn or "
                    "set backend to 'postgres'"
                )
        elif self.task_store_backend == "postgres":
            if not self.postgres_dsn:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig task_store_backend="
                    "'postgres' requires postgres_dsn"
                )
            if self.sqlite_db_path:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig sqlite_db_path set but "
                    "task_store_backend='postgres'; remove sqlite_db_path or "
                    "set backend to 'sqlite'"
                )
        else:
            # in_memory — neither backend-specific field should be set.
            if self.sqlite_db_path:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig sqlite_db_path is set but "
                    "task_store_backend='in_memory'; remove sqlite_db_path or "
                    "set backend to 'sqlite'"
                )
            if self.postgres_dsn:
                raise ValueError(
                    "FAIL-FAST: WorkflowRunnerConfig postgres_dsn is set but "
                    "task_store_backend='in_memory'; remove postgres_dsn or "
                    "set backend to 'postgres'"
                )
        # G21 Step 3 — sanity: stale threshold must exceed the heartbeat
        # interval, otherwise the watchdog reaps healthy tasks. Skipped
        # when heartbeat is disabled (interval == 0).
        if self.heartbeat_interval_seconds > 0 and (
            self.watchdog_stale_threshold_seconds
            <= self.heartbeat_interval_seconds
        ):
            raise ValueError(
                f"FAIL-FAST: watchdog_stale_threshold_seconds "
                f"({self.watchdog_stale_threshold_seconds}) must exceed "
                f"heartbeat_interval_seconds "
                f"({self.heartbeat_interval_seconds}) — otherwise the "
                f"watchdog reaps healthy tasks before they can refresh"
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
        self._store: TaskStore
        if config.task_store_backend == "in_memory":
            self._store = InMemoryTaskStore()
        elif config.task_store_backend == "sqlite":
            self._store = SqliteTaskStore(config.sqlite_db_path)  # type: ignore[arg-type]
        elif config.task_store_backend == "postgres":
            self._store = PostgresTaskStore(
                dsn=config.postgres_dsn,  # type: ignore[arg-type]
                table_name=config.postgres_table_name,
            )
        else:
            # Defensive — Pydantic Literal already rejects unknowns.
            raise ComponentConfigurationError(
                f"FAIL-FAST: unknown task_store_backend "
                f"{config.task_store_backend!r}"
            )
        self._tasks: Dict[str, asyncio.Task] = {}
        # G21 Step 2 — per-task PauseSignal registry. The runner owns
        # the signal; pause/resume mutate it; the inner _runner
        # coroutine publishes it as a contextvar so step code can read.
        self._pause_signals: Dict[str, PauseSignal] = {}
        self._semaphore = asyncio.Semaphore(self._max_concurrent)

        # G21 Step 3 — heartbeat watchdog. Lazily started; the first
        # run_detached call brings it up so a runner that never runs a
        # task pays no idle cost.
        self._heartbeat_interval = config.heartbeat_interval_seconds
        self._stale_threshold = config.watchdog_stale_threshold_seconds
        self._watchdog_task: Optional[asyncio.Task] = None
        self._watchdog_lock = asyncio.Lock()

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
        # G21 Step 3 — lazy-start the watchdog on first detached run.
        await self._maybe_start_watchdog()
        now = datetime.now(timezone.utc)
        handle = DetachedTaskHandle(
            task_id=task_id, status="queued", created_at=now,
        )
        await self._store.insert(handle)

        # G21 Step 2 — per-task pause signal. Created BEFORE the asyncio
        # task is scheduled so a caller can call pause() on the handle
        # in between run_detached() returning and _runner actually
        # entering the workflow callable; the workflow then sees the
        # signal already paused on its first contextvar read.
        signal = PauseSignal()
        self._pause_signals[task_id] = signal

        # G27 Option A v1 — record callable + payload so resume() can
        # re-spawn the asyncio task with the original args. In-memory
        # only; cross-process resume requires Option B (G5 checkpoint
        # integration). See nanobrain/docs/g27_g21_wiring_design.md.
        if not hasattr(self, "_suspended_callables"):
            self._suspended_callables = {}
            self._suspended_payloads = {}
        self._suspended_callables[task_id] = workflow_callable
        self._suspended_payloads[task_id] = payload

        async def _runner() -> None:
            # Publish the pause signal as a contextvar BEFORE entering
            # the workflow callable so any nested process() that calls
            # current_pause_signal() sees this task's signal.
            token = _current_pause_signal.set(signal)
            try:
                async with self._semaphore:
                    handle.status = "running"
                    handle.last_heartbeat_at = datetime.now(timezone.utc)
                    await self._store.update(handle)
                    try:
                        result = await workflow_callable(payload)
                        handle.status = "completed"
                        handle.result = result
                    except asyncio.CancelledError:
                        # Discriminate watchdog-initiated cancel (store
                        # already says 'failed') from external cancel:
                        # don't stomp the watchdog's verdict.
                        current = await self._store.get(task_id)
                        if current is not None and current.status == "failed":
                            raise
                        handle.status = "cancelled"
                        handle.completed_at = datetime.now(timezone.utc)
                        await self._store.update(handle)
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # G27 Option A v1 — soft-suspend on
                        # ApprovalPendingError. Lazy import keeps
                        # workflow_runner from depending on
                        # library/steps/ at import time. Other
                        # exceptions fall through to the failed branch.
                        if _is_approval_pending(exc):
                            handle.status = "suspended"
                            handle.suspension_info = {
                                "kind": "deferred_hitl",
                                "approval_id": getattr(
                                    exc, "approval_id", None
                                ),
                                "step_name": getattr(
                                    exc, "step_name", None
                                ),
                                "prompt": getattr(exc, "prompt", None),
                            }
                            # Do NOT set completed_at; the task is
                            # waiting for resolve(), not done.
                            await self._store.update(handle)
                            return  # exit _runner; resume() re-spawns
                        handle.status = "failed"
                        handle.error = f"{type(exc).__name__}: {exc}"
                    handle.completed_at = datetime.now(timezone.utc)
                    await self._store.update(handle)
            finally:
                _current_pause_signal.reset(token)

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

    async def pause(self, task_id: str, reason: str = "") -> None:
        """G21 Step 2 — cooperative soft-pause for a detached task.

        Sets the per-task ``PauseSignal``'s paused state and updates the
        handle's ``status`` to ``"paused"`` so callers polling
        ``get_handle`` see it immediately.

        Brutal truth: pause is a *cooperative* protocol. Step authors
        who want pause-aware behavior MUST consult
        ``current_pause_signal()`` from inside ``process()`` and either
        ``await signal.wait_until_resumed()`` or check ``is_paused()``
        between work units. A workflow that does NOT consult the signal
        will run to completion regardless of pause requests. The
        framework does NOT preempt running steps.

        Raises ``ComponentConfigurationError`` if ``task_id`` is unknown.
        Returns silently when the task is already terminal (completed
        / cancelled / failed) — pause-after-done is a no-op.
        """
        signal = self._pause_signals.get(task_id)
        if signal is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.pause: task_id {task_id!r} "
                f"is not registered with this runner"
            )
        handle = await self._store.get(task_id)
        if handle is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.pause: task_id {task_id!r} "
                f"missing from task store"
            )
        if handle.status not in _STATUS_ACTIVE:
            # Already terminal — pause is a no-op.
            return
        signal.pause()
        handle.status = "paused"
        await self._store.update(handle)

    async def resume(self, task_id: str) -> None:
        """G21 Step 2 — clear a pause signal and update status.

        Symmetric to ``pause``. Status moves paused → running; if the
        task was never paused, this is a no-op (still updates status to
        running for idempotency). Raises ``ComponentConfigurationError``
        for unknown ``task_id``.
        """
        signal = self._pause_signals.get(task_id)
        if signal is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.resume: task_id {task_id!r} "
                f"is not registered with this runner"
            )
        handle = await self._store.get(task_id)
        if handle is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.resume: task_id {task_id!r} "
                f"missing from task store"
            )
        if handle.status not in _STATUS_ACTIVE:
            return
        signal.resume()
        if handle.status == "paused":
            handle.status = "running"
            await self._store.update(handle)

    def is_paused(self, task_id: str) -> bool:
        """G21 Step 2 — query whether the task's pause signal is set.

        Synchronous because reading the signal flag is non-blocking.
        Returns False for unknown task_ids (does NOT raise) so that
        polling code can treat unknown-or-not-paused identically.
        """
        signal = self._pause_signals.get(task_id)
        if signal is None:
            return False
        return signal.is_paused()

    async def resume_suspended(self, task_id: str) -> None:
        """G27 Option A v1 (2026-05-11) — re-spawn a soft-suspended task.

        Distinct from ``resume`` (which clears the cooperative G21
        pause signal); this method handles the new ``"suspended"``
        lifecycle state introduced for deferred-HITL gates.

        Semantics (Option A — see ``nanobrain/docs/g27_g21_wiring_design.md``):
          * The original workflow callable + payload are re-invoked.
          * The deterministic ``approval_id`` (G27 P6+a default)
            ensures the re-running DeferredHITLStep finds the now-
            resolved Approval and returns its decision payload.
          * Steps BEFORE the DeferredHITLStep run again. Pure-compute
            steps fine; LLM-bound steps re-charge cost; side-effecting
            steps double-fire. Operators who want Option B's
            no-re-run semantic compose with CheckpointStep manually.

        Raises:
            ComponentConfigurationError: when ``task_id`` is unknown
                OR the task is not in ``"suspended"`` state.

        Returns when the asyncio task has been re-scheduled. The
        caller observes progress via ``get_handle`` / ``await_completion``.
        """
        handle = await self._store.get(task_id)
        if handle is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.resume_suspended: task_id "
                f"{task_id!r} is not registered with this runner"
            )
        if handle.status != "suspended":
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.resume_suspended: task_id "
                f"{task_id!r} is in status={handle.status!r}, not "
                f"'suspended'. Use cancel() or wait for completion."
            )
        # Adversarial-probe finding (2026-05-11): two concurrent
        # resume_suspended callers both pass the status check (no
        # atomic compare-and-swap), both re-spawn the asyncio task,
        # two _runner instances race to write the terminal status.
        # Guard via the existing ``_tasks`` dict: if there's already
        # an asyncio Task registered for this task_id AND it is not
        # done, refuse the second resume. The first resume's _runner
        # transitions status from suspended -> queued -> running, so
        # by the time a second caller wins the lookup, the existing
        # task is observably in-flight.
        existing_task = self._tasks.get(task_id)
        if existing_task is not None and not existing_task.done():
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.resume_suspended: task_id "
                f"{task_id!r} already has an in-flight asyncio task "
                f"(name={existing_task.get_name()!r}). A concurrent "
                f"resume is racing with this call. Wait for the first "
                f"resume to complete, then check status."
            )

        callable_ = getattr(self, "_suspended_callables", {}).get(task_id)
        payload = getattr(self, "_suspended_payloads", {}).get(task_id)
        if callable_ is None or payload is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WorkflowRunner.resume_suspended: cannot "
                f"resume task_id {task_id!r} — the original workflow "
                f"callable + payload are missing from the runner's "
                f"in-memory store. Cross-process resume requires "
                f"Option B (G5 checkpoint integration); v1 supports "
                f"same-process resume only."
            )

        # Clear suspension_info + re-spawn the asyncio task with the
        # original args. Use a fresh PauseSignal — the prior one's
        # event-loop scope ends with the prior _runner.
        # G27 Option B evaluation: bump resume_count so operators
        # can measure re-run frequency (cost is per-resume × per-
        # pre-HITL-step cost).
        handle.suspension_info = None
        handle.status = "queued"
        handle.resume_count = (handle.resume_count or 0) + 1
        await self._store.update(handle)

        signal = PauseSignal()
        self._pause_signals[task_id] = signal

        async def _resumed_runner() -> None:
            token = _current_pause_signal.set(signal)
            try:
                async with self._semaphore:
                    handle.status = "running"
                    handle.last_heartbeat_at = datetime.now(timezone.utc)
                    await self._store.update(handle)
                    try:
                        result = await callable_(payload)
                        handle.status = "completed"
                        handle.result = result
                    except asyncio.CancelledError:
                        current = await self._store.get(task_id)
                        if current is not None and current.status == "failed":
                            raise
                        handle.status = "cancelled"
                        handle.completed_at = datetime.now(timezone.utc)
                        await self._store.update(handle)
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # If the re-run hits a NEW ApprovalPendingError
                        # (multi-gate workflow), soft-suspend again.
                        if _is_approval_pending(exc):
                            handle.status = "suspended"
                            handle.suspension_info = {
                                "kind": "deferred_hitl",
                                "approval_id": getattr(
                                    exc, "approval_id", None
                                ),
                                "step_name": getattr(
                                    exc, "step_name", None
                                ),
                                "prompt": getattr(exc, "prompt", None),
                            }
                            await self._store.update(handle)
                            return
                        handle.status = "failed"
                        handle.error = f"{type(exc).__name__}: {exc}"
                    handle.completed_at = datetime.now(timezone.utc)
                    await self._store.update(handle)
            finally:
                _current_pause_signal.reset(token)

        self._tasks[task_id] = asyncio.create_task(
            _resumed_runner(), name=f"detached-{task_id}-resumed",
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
            # asyncio.shield prevents the inner task from being
            # cancelled when wait_for cancels the outer wait on timeout.
            # Without shield, await_completion(timeout=...) silently
            # cancels paused / slow workflows — a footgun the pause
            # smoke test surfaced.
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
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

    # ---- G21 Step 3 — heartbeat watchdog --------------------------------

    async def _maybe_start_watchdog(self) -> None:
        """Start the watchdog background task on first detached run.
        No-op when ``heartbeat_interval_seconds == 0`` (disabled by config)
        or when the watchdog is already running."""
        if self._heartbeat_interval <= 0:
            return
        async with self._watchdog_lock:
            if self._watchdog_task is not None and not self._watchdog_task.done():
                return
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(),
                name=f"watchdog-{self.name}",
            )

    async def stop_watchdog(self) -> None:
        """Stop the watchdog. Idempotent. Tests use this to ensure
        the watchdog doesn't leak between cases."""
        if self._watchdog_task is None:
            return
        self._watchdog_task.cancel()
        try:
            await self._watchdog_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._watchdog_task = None

    async def _watchdog_loop(self) -> None:
        """Periodic heartbeat updater + stale-task reaper.

        Wakes every ``heartbeat_interval_seconds`` and:
        1. For every running task whose asyncio task is not done,
           refresh ``last_heartbeat_at`` to now. This is the "I'm alive"
           signal.
        2. For every running task whose ``last_heartbeat_at`` is older
           than ``stale_threshold_seconds`` AND whose asyncio task is
           NOT done — reap it. The task's status flips to 'failed' with
           an error string and the asyncio task is cancelled.
        """
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                await self._tick_watchdog()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001
            # Watchdog must never bring down the runner; log and exit.
            # In production logging this would be a WARNING, but we
            # avoid pulling in get_logger here (would couple the runner
            # to the framework logging). Re-raising is wrong — this
            # daemon should fail gracefully.
            return

    async def _tick_watchdog(self) -> None:
        """One pass of heartbeat-refresh + stale-task reap. Public-ish
        for testability (sub-second tests can drive this directly
        instead of waiting for the periodic loop)."""
        now_dt = datetime.now(timezone.utc)
        active = await self._store.list_active()
        for handle in active:
            if handle.status != "running":
                continue
            asyncio_task = self._tasks.get(handle.task_id)
            if asyncio_task is None or asyncio_task.done():
                continue

            stale = False
            if handle.last_heartbeat_at is not None:
                age = (now_dt - handle.last_heartbeat_at).total_seconds()
                stale = age > self._stale_threshold

            if stale:
                # Reap. Mark status BEFORE cancelling so the task's own
                # CancelledError handler doesn't overwrite to 'cancelled'.
                handle.status = "failed"
                handle.error = (
                    f"WorkflowRunner watchdog reaped task: "
                    f"last_heartbeat_at older than "
                    f"watchdog_stale_threshold_seconds="
                    f"{self._stale_threshold}s"
                )
                handle.completed_at = now_dt
                await self._store.update(handle)
                asyncio_task.cancel()
                continue

            # Healthy — refresh heartbeat.
            handle.last_heartbeat_at = now_dt
            await self._store.update(handle)
