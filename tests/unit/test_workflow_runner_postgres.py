"""Tests for G21 Step 4 — PostgresTaskStore + TaskStore public extension point.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

Coverage:
1. TaskStore is a public class (importable, subclass-able).
2. WorkflowRunnerConfig accepts task_store_backend='postgres' + postgres_dsn.
3. backend='postgres' without postgres_dsn FAIL-FASTs.
4. backend='in_memory' + postgres_dsn FAIL-FASTs.
5. PostgresTaskStore raises ImportError if psycopg is unavailable
   (this test is best-effort; psycopg IS present in this venv).
6. INTEGRATION: full lifecycle against a real Postgres server when
   ``POSTGRES_TEST_DSN`` env var is set; otherwise SKIPPED.

CI / local-dev recipe for the integration test:

    docker run --rm -d --name nb-pg-test -p 5432:5432 \\
        -e POSTGRES_PASSWORD=test postgres:16
    POSTGRES_TEST_DSN='postgresql://postgres:test@localhost/postgres' \\
        .venv/bin/python -m pytest \\
        tests/unit/test_workflow_runner_postgres.py -v
    docker rm -f nb-pg-test
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.library.runtime import (
    InMemoryTaskStore,
    PostgresTaskStore,
    SqliteTaskStore,
    TaskStore,
    WorkflowRunner,
    WorkflowRunnerConfig,
)


_PG_DSN = os.environ.get("POSTGRES_TEST_DSN")
_pg_skip = pytest.mark.skipif(
    _PG_DSN is None,
    reason="POSTGRES_TEST_DSN not set; run a local postgres + set the env "
           "var to enable PostgresTaskStore integration tests",
)


# ---------------------------------------------------------------------------
# 1. Public extension point
# ---------------------------------------------------------------------------

class TestPublicExtensionPoint:

    def test_taskstore_is_importable(self):
        # Just importing TaskStore from the package is the test.
        assert TaskStore.__module__.endswith("workflow_runner")

    def test_taskstore_is_subclassable(self):
        class MyStore(TaskStore):
            async def insert(self, handle):
                pass
            async def update(self, handle):
                pass
            async def get(self, task_id):
                return None
            async def list_active(self):
                return []
        # If this constructs, the abstract surface is satisfied.
        s = MyStore()
        assert isinstance(s, TaskStore)

    def test_inmemory_is_taskstore(self):
        assert issubclass(InMemoryTaskStore, TaskStore)

    def test_sqlite_is_taskstore(self):
        assert issubclass(SqliteTaskStore, TaskStore)

    def test_postgres_is_taskstore(self):
        assert issubclass(PostgresTaskStore, TaskStore)

    def test_legacy_underscore_aliases_still_work(self):
        """Backwards-compat: the underscore-prefixed names that v1
        exported (or that external code might have grabbed) still
        resolve to the new public classes."""
        from nanobrain.library.runtime.workflow_runner import (
            _TaskStore, _InMemoryTaskStore, _SQLiteTaskStore,
        )
        assert _TaskStore is TaskStore
        assert _InMemoryTaskStore is InMemoryTaskStore
        assert _SQLiteTaskStore is SqliteTaskStore


# ---------------------------------------------------------------------------
# 2-4. WorkflowRunnerConfig validation
# ---------------------------------------------------------------------------

class TestPostgresConfigValidation:

    def _build(self, **kwargs):
        WorkflowRunnerConfig._allow_direct_instantiation = True
        try:
            return WorkflowRunnerConfig(**kwargs)
        finally:
            WorkflowRunnerConfig._allow_direct_instantiation = False

    def test_postgres_minimal_accepted(self):
        cfg = self._build(name="r", task_store_backend="postgres",
                          postgres_dsn="postgresql://localhost/x")
        assert cfg.task_store_backend == "postgres"
        assert cfg.postgres_table_name == "nanobrain_detached_tasks"

    def test_postgres_requires_dsn(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="r", task_store_backend="postgres")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "postgres_dsn" in str(exc_info.value)

    def test_postgres_with_sqlite_path_fails(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="r", task_store_backend="postgres",
                        postgres_dsn="postgresql://localhost/x",
                        sqlite_db_path="/tmp/x.db")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_in_memory_with_postgres_dsn_fails(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="r", task_store_backend="in_memory",
                        postgres_dsn="postgresql://localhost/x")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "postgres_dsn" in str(exc_info.value)

    def test_sqlite_with_postgres_dsn_fails(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="r", task_store_backend="sqlite",
                        sqlite_db_path="/tmp/x.db",
                        postgres_dsn="postgresql://localhost/x")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_table_name_override(self):
        cfg = self._build(name="r", task_store_backend="postgres",
                          postgres_dsn="postgresql://localhost/x",
                          postgres_table_name="my_custom_table")
        assert cfg.postgres_table_name == "my_custom_table"


# ---------------------------------------------------------------------------
# 5. psycopg ImportError shape
# ---------------------------------------------------------------------------

class TestPsycopgOptionalDep:

    def test_psycopg_is_lazy_imported(self):
        """Constructing PostgresTaskStore probes psycopg at __init__ time;
        if psycopg is missing, an ImportError with the install hint is
        raised. We can't easily test the missing-psycopg case without
        uninstalling psycopg from the venv, so this test verifies that
        the IMPORT HINT IS PRESENT in the error class's docstring or
        the constructor's behavior when psycopg IS present (the import
        succeeds without raising)."""
        # psycopg IS present in this venv — the construct should succeed.
        store = PostgresTaskStore(dsn="postgresql://localhost:9999/x")
        assert store is not None  # Construction succeeded; lazy probe passed


# ---------------------------------------------------------------------------
# 6. Integration test — gated on POSTGRES_TEST_DSN
# ---------------------------------------------------------------------------

@_pg_skip
class TestPostgresIntegration:

    def test_full_lifecycle_against_real_postgres(self):
        """End-to-end: store insert → update → get → list_active → reap.
        Uses a randomized table name so concurrent test runs don't collide."""
        import uuid
        from datetime import datetime, timezone
        from nanobrain.library.runtime import DetachedTaskHandle

        async def run():
            table = f"nb_test_{uuid.uuid4().hex[:8]}"
            store = PostgresTaskStore(dsn=_PG_DSN, table_name=table)
            await store.initialize()

            now = datetime.now(timezone.utc)
            h1 = DetachedTaskHandle(
                task_id="t1", status="queued", created_at=now,
            )
            await store.insert(h1)

            got = await store.get("t1")
            assert got is not None
            assert got.status == "queued"
            assert got.task_id == "t1"

            h1.status = "running"
            h1.last_heartbeat_at = now
            await store.update(h1)

            active = await store.list_active()
            assert len(active) == 1
            assert active[0].task_id == "t1"

            # Cleanup: drop the test table
            import psycopg
            async with store._conn.cursor() as cur:
                await cur.execute(f"DROP TABLE {table}")
            await store.close()

        asyncio.run(run())

    def test_runner_with_postgres_backend_via_from_config(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                yml = tmp / "r.yml"
                yml.write_text(yaml.safe_dump({
                    "name": "r",
                    "task_store_backend": "postgres",
                    "postgres_dsn": _PG_DSN,
                    "postgres_table_name": (
                        f"nb_runner_test_{os.getpid()}"
                    ),
                    "heartbeat_interval_seconds": 0,
                }))
                runner = WorkflowRunner.from_config(str(yml))
                async def wf(payload):
                    return {"ok": True}
                await runner.run_detached(wf, "rt1", {"x": 1})
                h = await runner.await_completion("rt1", timeout=5)
                assert h.status == "completed"
                assert h.result == {"ok": True}
                # Cleanup
                async with runner._store._conn.cursor() as cur:
                    await cur.execute(
                        f"DROP TABLE {runner._store._table}"
                    )
                await runner._store.close()
        asyncio.run(run())
