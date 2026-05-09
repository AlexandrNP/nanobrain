"""Tests for G21 — WorkflowRunner.run_detached.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

v1 scope coverage:

1. WorkflowRunnerConfig validation (backend-specific fields).
2. WorkflowRunner via from_config builds with the in-memory store.
3. WorkflowRunner via from_config builds with the SQLite store.
4. run_detached returns immediately (status='queued').
5. The background task progresses queued → running → completed.
6. The result is captured in the handle.
7. A workflow that raises is captured as status='failed' with error string.
8. cancel() transitions an in-flight task to status='cancelled'.
9. cancel() on unknown task_id raises FAIL-FAST.
10. await_completion respects timeout (raises asyncio.TimeoutError).
11. await_completion on unknown task_id raises FAIL-FAST.
12. pause raises NotImplementedError (Step 2 deferred).
13. Concurrent run_detached calls + list_active accuracy.
14. Duplicate task_id rejected.
15. SQLite store: state survives a fresh runner build (durability check).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.runtime import (
    DetachedTaskHandle,
    WorkflowRunner,
    WorkflowRunnerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path, backend: str = "in_memory", **overrides) -> WorkflowRunner:
    cfg = {"name": "r", "task_store_backend": backend}
    if backend == "sqlite":
        cfg["sqlite_db_path"] = str(tmp / "tasks.db")
    cfg.update(overrides)
    yml = tmp / f"runner_{backend}.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return WorkflowRunner.from_config(str(yml))


async def _ok_workflow(payload):
    await asyncio.sleep(0.01)
    return {"echo": payload}


async def _slow_workflow(payload):
    await asyncio.sleep(2.0)
    return payload


async def _failing_workflow(payload):
    raise RuntimeError("intentional test failure")


# ---------------------------------------------------------------------------
# 1. WorkflowRunnerConfig validation
# ---------------------------------------------------------------------------

class TestRunnerConfig:

    def _build(self, **kwargs):
        WorkflowRunnerConfig._allow_direct_instantiation = True
        try:
            return WorkflowRunnerConfig(**kwargs)
        finally:
            WorkflowRunnerConfig._allow_direct_instantiation = False

    def test_in_memory_minimal(self):
        cfg = self._build(name="r")
        assert cfg.task_store_backend == "in_memory"
        assert cfg.sqlite_db_path is None

    def test_sqlite_requires_path(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="r", task_store_backend="sqlite")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "sqlite_db_path" in str(exc_info.value)

    def test_in_memory_rejects_sqlite_path(self):
        """A path with in_memory backend is almost certainly a typo."""
        with pytest.raises(Exception) as exc_info:
            self._build(name="r", task_store_backend="in_memory",
                        sqlite_db_path="/tmp/x.db")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_max_concurrent_must_be_positive(self):
        with pytest.raises(Exception):
            self._build(name="r", max_concurrent_detached_tasks=0)


# ---------------------------------------------------------------------------
# 2-3. Build paths
# ---------------------------------------------------------------------------

class TestRunnerBuilds:

    def test_in_memory_runner_via_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _build_runner(Path(tmp), "in_memory")
            assert r.name == "r"

    def test_sqlite_runner_via_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = _build_runner(Path(tmp), "sqlite")
            assert r.name == "r"
            assert (Path(tmp) / "tasks.db").exists()


# ---------------------------------------------------------------------------
# 4-7. run_detached lifecycle
# ---------------------------------------------------------------------------

class TestRunDetachedLifecycle:

    def test_returns_immediately_status_queued(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                handle = await r.run_detached(_ok_workflow, "t1", {"x": 1})
                assert isinstance(handle, DetachedTaskHandle)
                assert handle.task_id == "t1"
                assert handle.status == "queued"
                assert handle.created_at is not None
                # Cleanup:
                await r.await_completion("t1", timeout=5)
        asyncio.run(run())

    def test_progresses_to_completed(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_ok_workflow, "t1", {"x": 1})
                final = await r.await_completion("t1", timeout=5)
                assert final.status == "completed"
                assert final.result == {"echo": {"x": 1}}
                assert final.completed_at is not None
                assert final.error is None
        asyncio.run(run())

    def test_failed_workflow_captured(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_failing_workflow, "t1", {})
                final = await r.await_completion("t1", timeout=5)
                assert final.status == "failed"
                assert "intentional test failure" in final.error
                assert final.result is None
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 8-9. cancel
# ---------------------------------------------------------------------------

class TestCancel:

    def test_cancels_in_flight_task(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_slow_workflow, "t1", {})
                await asyncio.sleep(0.05)  # let it transition to running
                await r.cancel("t1", reason="test cancel")
                final = await r.get_handle("t1")
                assert final.status == "cancelled"
        asyncio.run(run())

    def test_cancel_unknown_task_id_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await r.cancel("ghost", reason="x")
                assert "FAIL-FAST" in str(exc_info.value)
                assert "ghost" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 10-11. await_completion
# ---------------------------------------------------------------------------

class TestAwaitCompletion:

    def test_timeout_raises(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_slow_workflow, "t1", {})
                with pytest.raises(asyncio.TimeoutError):
                    await r.await_completion("t1", timeout=0.05)
                # Cleanup:
                await r.cancel("t1")
        asyncio.run(run())

    def test_unknown_task_id_fails_fast(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                with pytest.raises(ComponentConfigurationError):
                    await r.await_completion("ghost", timeout=1)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 12. pause is reserved-for-future
# ---------------------------------------------------------------------------

class TestPauseDeferred:

    def test_pause_raises_not_implemented(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                with pytest.raises(NotImplementedError) as exc_info:
                    await r.pause("anything", reason="x")
                assert "G21 Step 2" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 13-14. Concurrency + duplicates
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_list_active_reflects_in_flight(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_slow_workflow, "t1", {})
                await r.run_detached(_slow_workflow, "t2", {})
                await asyncio.sleep(0.05)
                active = await r.list_active()
                ids = sorted(h.task_id for h in active)
                assert ids == ["t1", "t2"]
                # Cleanup:
                await r.cancel("t1")
                await r.cancel("t2")
        asyncio.run(run())

    def test_duplicate_task_id_rejected(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_slow_workflow, "t1", {})
                with pytest.raises(ValueError) as exc_info:
                    await r.run_detached(_ok_workflow, "t1", {})
                assert "already exists" in str(exc_info.value)
                await r.cancel("t1")
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 15. SQLite durability
# ---------------------------------------------------------------------------

class TestSqliteDurability:

    def test_state_survives_fresh_runner_build(self):
        """Run a task to completion under runner A, then build a fresh
        runner B pointing at the same SQLite file and read the same handle."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp_p = Path(tmp)
                r1 = _build_runner(tmp_p, "sqlite")
                await r1.run_detached(_ok_workflow, "t1", {"x": 99})
                final1 = await r1.await_completion("t1", timeout=5)
                assert final1.status == "completed"

                # Fresh runner — same SQLite file:
                r2 = _build_runner(tmp_p, "sqlite")
                read_back = await r2.get_handle("t1")
                assert read_back is not None
                assert read_back.status == "completed"
                assert read_back.result == {"echo": {"x": 99}}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# Handle hand-out is a copy (mutation isolation)
# ---------------------------------------------------------------------------

class TestHandleIsolation:

    def test_returned_handle_is_a_copy(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(_ok_workflow, "t1", {"x": 1})
                h1 = await r.get_handle("t1")
                h1.status = "MUTATED"  # local mutation
                h2 = await r.get_handle("t1")
                assert h2.status != "MUTATED", \
                    "Store handed out the same object — mutation leaked"
                await r.await_completion("t1", timeout=5)
        asyncio.run(run())
