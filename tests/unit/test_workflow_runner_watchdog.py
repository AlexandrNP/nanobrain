"""Tests for G21 Step 3 — heartbeat watchdog + stale-task reaper.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

Coverage:
1. Default watchdog values (60s / 600s) accepted.
2. Stale threshold ≤ heartbeat interval rejected (FAIL-FAST).
3. heartbeat_interval_seconds = 0 disables the watchdog.
4. Heartbeat advances last_heartbeat_at on running tasks.
5. Stale task is reaped: status -> failed with watchdog error string.
6. Reaped task's asyncio task is cancelled.
7. Reaped status survives the CancelledError handler in _runner
   (race-condition fix).
8. stop_watchdog is idempotent.
9. Healthy fast workflows complete normally; watchdog does not interfere.
10. Watchdog never starts when interval == 0 (no leaked task).
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from nanobrain.library.runtime import WorkflowRunner, WorkflowRunnerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path, **runner_overrides) -> WorkflowRunner:
    cfg = {"name": "r", "task_store_backend": "in_memory"}
    cfg.update(runner_overrides)
    yml = tmp / "runner.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return WorkflowRunner.from_config(str(yml))


async def _slow_workflow(payload):
    await asyncio.sleep(5)  # outlives any reasonable watchdog test
    return {"done": True}


async def _quick_workflow(payload):
    return {"ok": True}


# ---------------------------------------------------------------------------
# 1-3. Config field validation
# ---------------------------------------------------------------------------

class TestWatchdogConfig:

    def _build(self, **kwargs):
        WorkflowRunnerConfig._allow_direct_instantiation = True
        try:
            return WorkflowRunnerConfig(**kwargs)
        finally:
            WorkflowRunnerConfig._allow_direct_instantiation = False

    def test_defaults(self):
        cfg = self._build(name="r")
        assert cfg.heartbeat_interval_seconds == 60.0
        assert cfg.watchdog_stale_threshold_seconds == 600.0

    def test_stale_threshold_must_exceed_interval(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="r",
                        heartbeat_interval_seconds=10.0,
                        watchdog_stale_threshold_seconds=5.0)
        assert "FAIL-FAST" in str(exc_info.value)

    def test_stale_threshold_equal_to_interval_rejected(self):
        with pytest.raises(Exception):
            self._build(name="r",
                        heartbeat_interval_seconds=10.0,
                        watchdog_stale_threshold_seconds=10.0)

    def test_heartbeat_zero_disables_validation(self):
        # When heartbeat is 0, the threshold-vs-interval check is
        # skipped (the watchdog is disabled, so threshold is irrelevant).
        cfg = self._build(name="r",
                          heartbeat_interval_seconds=0,
                          watchdog_stale_threshold_seconds=1.0)
        assert cfg.heartbeat_interval_seconds == 0


# ---------------------------------------------------------------------------
# 4. Heartbeat advances
# ---------------------------------------------------------------------------

class TestHeartbeatAdvance:

    def test_heartbeat_advances_on_running_task(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0.05,
                    watchdog_stale_threshold_seconds=10.0,
                )
                await r.run_detached(_slow_workflow, "t1", {})
                # First snapshot
                await asyncio.sleep(0.1)
                h1 = await r.get_handle("t1")
                first_hb = h1.last_heartbeat_at
                assert first_hb is not None
                # Second snapshot — must move forward
                await asyncio.sleep(0.15)
                h2 = await r.get_handle("t1")
                assert h2.last_heartbeat_at > first_hb
                # Cleanup
                await r.cancel("t1")
                await r.stop_watchdog()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5-7. Stale-task reaping
# ---------------------------------------------------------------------------

class TestStaleReaping:

    def test_stale_task_marked_failed(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0.05,
                    watchdog_stale_threshold_seconds=0.2,
                )
                await r.run_detached(_slow_workflow, "t1", {})
                await asyncio.sleep(0.1)

                # Force staleness by rewinding the stored heartbeat.
                stored = await r._store.get("t1")
                stored.last_heartbeat_at = (
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                )
                await r._store.update(stored)

                # Wait for the watchdog to notice and reap.
                await asyncio.sleep(0.25)
                h = await r.get_handle("t1")
                assert h.status == "failed", f"expected failed, got {h.status}"
                assert "watchdog reaped" in h.error
                assert "watchdog_stale_threshold_seconds=0.2" in h.error
                await r.stop_watchdog()
        asyncio.run(run())

    def test_reaped_task_asyncio_task_cancelled(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0.05,
                    watchdog_stale_threshold_seconds=0.2,
                )
                await r.run_detached(_slow_workflow, "t1", {})
                await asyncio.sleep(0.1)
                stored = await r._store.get("t1")
                stored.last_heartbeat_at = (
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                )
                await r._store.update(stored)
                await asyncio.sleep(0.25)
                # The asyncio task should be done (cancelled by reaper).
                t = r._tasks["t1"]
                assert t.done()
                await r.stop_watchdog()
        asyncio.run(run())

    def test_reaped_status_survives_cancellederror_handler(self):
        """Race-condition pinning test. The watchdog sets status=failed
        AND cancels the asyncio task. The task's CancelledError handler
        must NOT overwrite to 'cancelled'; the watchdog's verdict wins."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0.05,
                    watchdog_stale_threshold_seconds=0.2,
                )
                await r.run_detached(_slow_workflow, "t1", {})
                await asyncio.sleep(0.1)
                stored = await r._store.get("t1")
                stored.last_heartbeat_at = (
                    datetime.now(timezone.utc) - timedelta(seconds=10)
                )
                await r._store.update(stored)
                # Generous wait so the CancelledError handler has time
                # to run and (incorrectly) overwrite if the race is back.
                await asyncio.sleep(0.4)
                h = await r.get_handle("t1")
                assert h.status == "failed", \
                    f"CancelledError handler stomped watchdog verdict: {h.status}"
                await r.stop_watchdog()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 8. stop_watchdog idempotency
# ---------------------------------------------------------------------------

class TestStopWatchdog:

    def test_stop_watchdog_idempotent(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0.05,
                    watchdog_stale_threshold_seconds=10.0,
                )
                await r.run_detached(_quick_workflow, "t1", {})
                await r.await_completion("t1", timeout=2)
                await r.stop_watchdog()
                await r.stop_watchdog()  # second call must not raise
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 9. Healthy workflow not interfered with
# ---------------------------------------------------------------------------

class TestHealthyWorkflow:

    def test_quick_workflow_completes_normally(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0.05,
                    watchdog_stale_threshold_seconds=10.0,
                )
                await r.run_detached(_quick_workflow, "t1", {})
                h = await r.await_completion("t1", timeout=2)
                assert h.status == "completed"
                assert h.result == {"ok": True}
                await r.stop_watchdog()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 10. Watchdog disabled
# ---------------------------------------------------------------------------

class TestWatchdogDisabled:

    def test_zero_interval_disables_watchdog(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(
                    Path(tmp),
                    heartbeat_interval_seconds=0,
                )
                await r.run_detached(_quick_workflow, "t1", {})
                await r.await_completion("t1", timeout=2)
                # Watchdog task must NOT have started.
                assert r._watchdog_task is None
        asyncio.run(run())
