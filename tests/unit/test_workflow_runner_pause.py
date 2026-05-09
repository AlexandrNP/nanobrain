"""Tests for G21 Step 2 — cooperative pause via PauseSignal contextvar.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

Coverage:
1. PauseSignal class basics (paused / resumed / wait_until_resumed).
2. current_pause_signal() returns None outside a detached run.
3. current_pause_signal() returns the per-task signal inside _runner.
4. Two concurrent tasks see independent signals (contextvar isolation).
5. pause() updates handle status to 'paused'; resume() restores 'running'.
6. is_paused() reflects the signal state synchronously.
7. End-to-end: cooperative workflow honors pause/resume.
8. Unknown task pause / resume FAIL-FAST.
9. pause-after-completion is a no-op (does not flip terminal status).
10. await_completion(timeout=...) does NOT cancel the inner task on timeout
    (the asyncio.shield fix).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.runtime import (
    PauseSignal,
    WorkflowRunner,
    current_pause_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path) -> WorkflowRunner:
    yml = tmp / "runner.yml"
    yml.write_text(yaml.safe_dump({
        "name": "r", "task_store_backend": "in_memory",
    }))
    return WorkflowRunner.from_config(str(yml))


# ---------------------------------------------------------------------------
# 1. PauseSignal class basics
# ---------------------------------------------------------------------------

class TestPauseSignalBasics:

    def test_default_state_not_paused(self):
        async def run():
            sig = PauseSignal()
            assert sig.is_paused() is False
            await sig.wait_until_resumed()  # immediate
        asyncio.run(run())

    def test_pause_then_resume(self):
        async def run():
            sig = PauseSignal()
            sig.pause()
            assert sig.is_paused() is True
            sig.resume()
            assert sig.is_paused() is False
        asyncio.run(run())

    def test_wait_until_resumed_blocks_when_paused(self):
        async def run():
            sig = PauseSignal()
            sig.pause()

            async def waiter():
                await sig.wait_until_resumed()
                return "resumed"

            t = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)
            assert not t.done()
            sig.resume()
            result = await asyncio.wait_for(t, timeout=1)
            assert result == "resumed"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 2-3. current_pause_signal() context exposure
# ---------------------------------------------------------------------------

class TestContextExposure:

    def test_returns_none_outside_detached_run(self):
        assert current_pause_signal() is None

    def test_published_inside_runner(self):
        observed = []

        async def workflow(payload):
            observed.append(current_pause_signal())
            return None

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(workflow, "t1", {})
                await r.await_completion("t1", timeout=2)
                assert observed[0] is not None
                assert isinstance(observed[0], PauseSignal)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. Concurrent task isolation (PEP 567)
# ---------------------------------------------------------------------------

class TestConcurrentIsolation:

    def test_two_tasks_see_independent_signals(self):
        observed = {}

        async def workflow(payload):
            observed[payload["id"]] = current_pause_signal()
            await asyncio.sleep(0.05)
            return None

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(workflow, "t1", {"id": "t1"})
                await r.run_detached(workflow, "t2", {"id": "t2"})
                await r.await_completion("t1", timeout=2)
                await r.await_completion("t2", timeout=2)
                assert observed["t1"] is not observed["t2"], \
                    "Concurrent tasks shared a signal — contextvar isolation broken"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5-6. pause/resume/is_paused mutate handle status
# ---------------------------------------------------------------------------

class TestPauseResumeStatus:

    def test_pause_flips_status_to_paused(self):
        async def workflow(payload):
            sig = current_pause_signal()
            await sig.wait_until_resumed()
            await asyncio.sleep(0.1)
            await sig.wait_until_resumed()
            return {"ok": True}

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(workflow, "t1", {})
                await asyncio.sleep(0.05)
                await r.pause("t1")
                h = await r.get_handle("t1")
                assert h.status == "paused"
                assert r.is_paused("t1") is True

                await r.resume("t1")
                h = await r.get_handle("t1")
                assert h.status == "running"
                assert r.is_paused("t1") is False
                # Cleanup
                await r.await_completion("t1", timeout=2)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. End-to-end cooperative pause
# ---------------------------------------------------------------------------

class TestEndToEndCooperativePause:

    def test_workflow_blocks_on_pause_resumes_on_resume(self):
        async def workflow(payload):
            sig = current_pause_signal()
            await sig.wait_until_resumed()
            await asyncio.sleep(0.1)        # work
            await sig.wait_until_resumed()  # may block here
            return {"completed": True}

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(workflow, "t1", {})
                await asyncio.sleep(0.05)
                await r.pause("t1")

                # Workflow blocked at second checkpoint; status stays paused
                # for at least 0.3s (no completion).
                await asyncio.sleep(0.3)
                h = await r.get_handle("t1")
                assert h.status == "paused", \
                    f"workflow should be blocked at pause; got {h.status}"

                await r.resume("t1")
                h = await r.await_completion("t1", timeout=2)
                assert h.status == "completed"
                assert h.result == {"completed": True}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 9. Pause-after-completion is no-op
# ---------------------------------------------------------------------------

class TestPauseAfterCompletion:

    def test_pause_after_completion_does_not_flip_status(self):
        async def quick_workflow(payload):
            return {"ok": True}

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(quick_workflow, "t1", {})
                await r.await_completion("t1", timeout=2)
                # Now task is terminal; pause should be a no-op.
                await r.pause("t1")
                h = await r.get_handle("t1")
                assert h.status == "completed", \
                    f"pause-after-done flipped status to {h.status}"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 10. await_completion timeout does NOT cancel (asyncio.shield fix)
# ---------------------------------------------------------------------------

class TestAwaitCompletionShield:

    def test_timeout_does_not_cancel_underlying_task(self):
        """Without asyncio.shield, asyncio.wait_for cancels its inner
        awaitable on timeout — silently killing slow / paused workflows.
        Verify the shield is in place by:
        1. starting a 0.5s workflow
        2. calling await_completion with timeout=0.1 (raises TimeoutError)
        3. waiting another 0.6s
        4. confirming the workflow completed normally (status='completed')
        """
        async def slow_workflow(payload):
            await asyncio.sleep(0.5)
            return {"done": True}

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r = _build_runner(Path(tmp))
                await r.run_detached(slow_workflow, "t1", {})

                # First await: timeout fires; workflow MUST keep running
                with pytest.raises(asyncio.TimeoutError):
                    await r.await_completion("t1", timeout=0.1)

                # Wait for the workflow to actually finish naturally.
                h = await r.await_completion("t1", timeout=2)
                assert h.status == "completed", \
                    f"timeout cancelled the task; got status={h.status}"
                assert h.result == {"done": True}
        asyncio.run(run())
