"""G27 wiring — pin suspension_info round-trip through SQLite + Postgres TaskStores.

Follow-up to the G27 Option A v1 in-memory wiring (commit ``abd41b5``):
the runner's suspension_info field was in-memory only. This commit
wires it into both SQLite and Postgres TaskStore _to_row / _from_row
mappings AND the schema (with an idempotent ALTER for existing DBs).

Cross-process resume v1 — a workflow suspended in process A whose
state is in a shared SQLite/Postgres TaskStore can be inspected (and,
once the in-memory ``_suspended_callables`` map is replaced with a
durable callable-reference store, resumed) from process B.

This test pins:
  1. SQLite: insert + get round-trips suspension_info as a dict
  2. SQLite: update can mutate suspension_info -> None (resume clears)
  3. SQLite: pre-existing DB without the column gets ALTER'd (idempotent)
  4. SQLite: list_active includes suspended tasks
  5. SQLite: _from_row tolerates rows from a pre-migration shape (8 cols)
  6. Postgres: insert + get round-trips suspension_info as a dict
     (gated on POSTGRES_TEST_DSN env var)

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G27 (DB-serialization follow-up to wiring);
``nanobrain/docs/g27_g21_wiring_design.md`` Option A scope note.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import datetime, timezone

import pytest

from nanobrain.library.runtime.workflow_runner import (
    DetachedTaskHandle,
    PostgresTaskStore,
    SqliteTaskStore,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# SQLite tests — exercise the canonical disk-backed path
# ---------------------------------------------------------------------------


def test_sqlite_round_trips_suspension_info(tmp_path):
    """insert(handle) with suspension_info; get(task_id) returns a
    handle with equal suspension_info. JSON serialization round-trip
    must be lossless for the dict shape that G27 emits."""
    db_path = str(tmp_path / "tasks.db")
    store = SqliteTaskStore(db_path)

    h = DetachedTaskHandle(
        task_id="task-suspend-rt-1",
        status="suspended",
        created_at=_now(),
        suspension_info={
            "kind": "deferred_hitl",
            "approval_id": "abc123def456",
            "step_name": "approve_step",
            "prompt": "Approve change to size from 100 to 200?",
        },
    )

    asyncio.run(store.insert(h))
    fetched = asyncio.run(store.get("task-suspend-rt-1"))

    assert fetched is not None
    assert fetched.status == "suspended"
    assert fetched.suspension_info == {
        "kind": "deferred_hitl",
        "approval_id": "abc123def456",
        "step_name": "approve_step",
        "prompt": "Approve change to size from 100 to 200?",
    }


def test_sqlite_update_clears_suspension_info(tmp_path):
    """When a task resumes (status -> running, suspension_info -> None),
    the update path must persist the cleared value."""
    db_path = str(tmp_path / "tasks.db")
    store = SqliteTaskStore(db_path)

    h = DetachedTaskHandle(
        task_id="task-clear-1",
        status="suspended",
        created_at=_now(),
        suspension_info={"kind": "deferred_hitl", "approval_id": "x"},
    )
    asyncio.run(store.insert(h))

    h.status = "running"
    h.suspension_info = None
    asyncio.run(store.update(h))

    fetched = asyncio.run(store.get("task-clear-1"))
    assert fetched.status == "running"
    assert fetched.suspension_info is None


def test_sqlite_pre_existing_db_gets_alter_idempotent(tmp_path):
    """A SQLite DB created by a build before this column existed must
    still work — the SqliteTaskStore constructor issues a defensive
    ``ALTER TABLE ADD COLUMN`` that catches the "duplicate column name"
    error class for already-migrated DBs."""
    db_path = str(tmp_path / "tasks.db")

    # Simulate a pre-migration DB: schema without suspension_info_json.
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute(
        """
        CREATE TABLE detached_tasks (
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
    )
    conn.close()

    # Constructing SqliteTaskStore against this DB must ALTER the table.
    store = SqliteTaskStore(db_path)

    # Subsequent operations work with the new column.
    h = DetachedTaskHandle(
        task_id="post-migration-1",
        status="suspended",
        created_at=_now(),
        suspension_info={"kind": "deferred_hitl", "approval_id": "y"},
    )
    asyncio.run(store.insert(h))
    fetched = asyncio.run(store.get("post-migration-1"))
    assert fetched.suspension_info == {
        "kind": "deferred_hitl",
        "approval_id": "y",
    }

    # Second construction is also fine (the ALTER is idempotent).
    SqliteTaskStore(db_path)


def test_sqlite_list_active_includes_suspended(tmp_path):
    """``"suspended"`` is in _STATUS_ACTIVE (per G27 wiring); list_active
    must surface suspended tasks alongside queued/running/paused."""
    db_path = str(tmp_path / "tasks.db")
    store = SqliteTaskStore(db_path)

    asyncio.run(
        store.insert(
            DetachedTaskHandle(
                task_id="t-suspended",
                status="suspended",
                created_at=_now(),
                suspension_info={"approval_id": "abc"},
            )
        )
    )
    asyncio.run(
        store.insert(
            DetachedTaskHandle(
                task_id="t-completed",
                status="completed",
                created_at=_now(),
            )
        )
    )

    active = asyncio.run(store.list_active())
    active_ids = {h.task_id for h in active}
    assert "t-suspended" in active_ids
    assert "t-completed" not in active_ids


def test_sqlite_from_row_tolerates_pre_migration_row_shape():
    """_from_row must accept tuples of length 8 (pre-migration) and
    9 (post-migration) and produce equivalent handles. Important when
    a row was inserted before the migration and is read after."""
    from datetime import datetime as _dt

    short_row = (
        "old-task",
        "completed",
        _dt(2026, 5, 11, 10, 0, 0).isoformat(),
        None,
        None,
        None,
        None,
        None,
    )
    h = SqliteTaskStore._from_row(short_row)
    assert h is not None
    assert h.task_id == "old-task"
    assert h.suspension_info is None  # tolerated absent

    long_row = (*short_row, '{"approval_id": "x"}')
    h2 = SqliteTaskStore._from_row(long_row)
    assert h2.suspension_info == {"approval_id": "x"}


# ---------------------------------------------------------------------------
# Postgres test — gated on POSTGRES_TEST_DSN env var
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("POSTGRES_TEST_DSN"),
    reason="POSTGRES_TEST_DSN env var not set; skipping live Postgres test",
)
def test_postgres_round_trips_suspension_info():
    """Postgres-backed store; gated by env var. Schema + ALTER are
    issued during initialize(); the round-trip should produce a handle
    with equal suspension_info."""
    dsn = os.environ["POSTGRES_TEST_DSN"]
    table = f"nb_g27_test_{int(_now().timestamp())}"
    store = PostgresTaskStore(dsn, table_name=table)

    async def _run():
        await store.initialize()
        h = DetachedTaskHandle(
            task_id="pg-suspend-1",
            status="suspended",
            created_at=_now(),
            suspension_info={
                "kind": "deferred_hitl",
                "approval_id": "pg_approval_abc",
                "step_name": "approve_step",
                "prompt": "Approve PG round-trip?",
            },
        )
        await store.insert(h)
        fetched = await store.get("pg-suspend-1")
        try:
            assert fetched is not None
            assert fetched.status == "suspended"
            assert fetched.suspension_info == {
                "kind": "deferred_hitl",
                "approval_id": "pg_approval_abc",
                "step_name": "approve_step",
                "prompt": "Approve PG round-trip?",
            }
        finally:
            # Cleanup: drop the per-test table.
            import psycopg

            conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
            async with conn.cursor() as cur:
                await cur.execute(f"DROP TABLE IF EXISTS {table}")
            await conn.close()
            await store.close()

    asyncio.run(_run())
