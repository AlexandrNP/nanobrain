"""Tests for G21 Step 5 — automatic BaseStep pause cooperation.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G21``.

Coverage:
1. _await_pause_signal_if_present is a no-op when no signal is published.
2. _await_pause_signal_if_present returns immediately when signal is not paused.
3. _await_pause_signal_if_present blocks when signal is paused; returns
   when resumed.
4. The lazy import is cached after first call (no repeated probing).
5. End-to-end: a step with NO awareness of PauseSignal still honors pause
   when running inside a detached workflow. (The Step 2 protocol becomes
   automatic.)
6. Cancel-during-pause terminates the awaiting step (asyncio.CancelledError).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from nanobrain.core.step import (
    BaseStep,
    StepConfig,
    _await_pause_signal_if_present,
)
from nanobrain.library.runtime import (
    PauseSignal,
    WorkflowRunner,
    current_pause_signal,
)
from nanobrain.library.runtime.workflow_runner import _current_pause_signal


# ---------------------------------------------------------------------------
# 1-3. Helper behavior in isolation
# ---------------------------------------------------------------------------

class TestHelperBehavior:

    def test_no_signal_is_noop(self):
        async def run():
            # Outside any detached run, the contextvar is None.
            assert current_pause_signal() is None
            # Should return immediately — no exception, no block.
            await asyncio.wait_for(_await_pause_signal_if_present(), timeout=1.0)
        asyncio.run(run())

    def test_unpaused_signal_returns_immediately(self):
        async def run():
            sig = PauseSignal()
            token = _current_pause_signal.set(sig)
            try:
                # Default is not-paused
                assert sig.is_paused() is False
                await asyncio.wait_for(_await_pause_signal_if_present(), timeout=1.0)
            finally:
                _current_pause_signal.reset(token)
        asyncio.run(run())

    def test_paused_signal_blocks_until_resumed(self):
        async def run():
            sig = PauseSignal()
            token = _current_pause_signal.set(sig)
            try:
                sig.pause()
                # Schedule a resume after 100ms
                async def resume_later():
                    await asyncio.sleep(0.1)
                    sig.resume()
                resumer = asyncio.create_task(resume_later())
                start = asyncio.get_event_loop().time()
                await _await_pause_signal_if_present()
                elapsed = asyncio.get_event_loop().time() - start
                assert elapsed >= 0.09, \
                    f"Helper did not block on paused signal; elapsed={elapsed}"
                await resumer
            finally:
                _current_pause_signal.reset(token)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. Lazy-import caching
# ---------------------------------------------------------------------------

class TestImportCaching:

    def test_repeated_calls_dont_reimport(self):
        """After the first call, the cached getter must be a function;
        subsequent calls should not re-probe the import system. We verify
        by reading the module-level cache flag."""
        async def run():
            from nanobrain.core import step as step_module
            await _await_pause_signal_if_present()
            assert step_module._CURRENT_PAUSE_SIGNAL_PROBED is True
            # The getter is set when the import succeeded
            assert step_module._CURRENT_PAUSE_SIGNAL_GETTER is not None
            # Second call should not re-probe (cache hit)
            await _await_pause_signal_if_present()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. End-to-end with a real BaseStep subclass
# ---------------------------------------------------------------------------

class _NaivePauseUnawareStep(BaseStep):
    """A step that does NOT consult current_pause_signal() at all.
    The framework's Step 5 wiring is the sole pause-cooperation
    mechanism for this step."""
    COMPONENT_TYPE = "naive_step"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    def _init_from_config(self, config, component_config, dependencies):
        super()._init_from_config(config, component_config, dependencies)
        self.process_call_count = 0

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        self.process_call_count += 1
        return {"called": self.process_call_count, "input": input_data}


class TestEndToEndAutoCooperation:

    def test_step_blocks_on_paused_signal_without_user_awareness(self):
        """A naive step that does NOT consult current_pause_signal()
        should still block at _execute_process boundaries when pause is
        active in the contextvar."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                yml = tmp / "step.yml"
                yml.write_text(yaml.safe_dump({"name": "naive"}))
                step = _NaivePauseUnawareStep.from_config(str(yml))

                sig = PauseSignal()
                token = _current_pause_signal.set(sig)
                try:
                    sig.pause()

                    # Schedule a resume after 100ms
                    async def resume_later():
                        await asyncio.sleep(0.1)
                        sig.resume()
                    resumer = asyncio.create_task(resume_later())

                    start = asyncio.get_event_loop().time()
                    result = await step._execute_process({"x": 1})
                    elapsed = asyncio.get_event_loop().time() - start

                    assert elapsed >= 0.09, \
                        f"Step ran without honoring pause; elapsed={elapsed}"
                    assert result["called"] == 1
                    assert result["input"] == {"x": 1}
                    await resumer
                finally:
                    _current_pause_signal.reset(token)
        asyncio.run(run())

    def test_no_signal_no_overhead(self):
        """A step run with no pause signal published runs as before
        (this is the "preserves historical behavior" guarantee)."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                yml = tmp / "step.yml"
                yml.write_text(yaml.safe_dump({"name": "naive"}))
                step = _NaivePauseUnawareStep.from_config(str(yml))

                # No contextvar set — runs immediately
                start = asyncio.get_event_loop().time()
                result = await step._execute_process({"x": 1})
                elapsed = asyncio.get_event_loop().time() - start
                assert elapsed < 0.05, \
                    f"No-signal path took {elapsed}s; should be ~0"
                assert result["called"] == 1
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Cancel during pause
# ---------------------------------------------------------------------------

class TestCancelDuringPause:

    def test_cancel_terminates_paused_step(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                yml = tmp / "step.yml"
                yml.write_text(yaml.safe_dump({"name": "naive"}))
                step = _NaivePauseUnawareStep.from_config(str(yml))

                sig = PauseSignal()
                token = _current_pause_signal.set(sig)
                try:
                    sig.pause()

                    step_task = asyncio.create_task(
                        step._execute_process({"x": 1})
                    )
                    await asyncio.sleep(0.05)
                    assert not step_task.done()
                    step_task.cancel()
                    try:
                        await step_task
                    except asyncio.CancelledError:
                        pass
                    assert step_task.done()
                    # process() must NOT have been called (cancelled
                    # while waiting on the pause signal)
                    assert step.process_call_count == 0
                finally:
                    _current_pause_signal.reset(token)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. End-to-end with a real WorkflowRunner
# ---------------------------------------------------------------------------

class TestEndToEndWithRunner:

    def test_runner_published_signal_makes_naive_step_pause(self):
        """Wire-up test: WorkflowRunner publishes the contextvar; a
        naive step running inside the workflow callable honors pause
        without ever consulting current_pause_signal() in its own code."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner_yml = tmp / "r.yml"
                runner_yml.write_text(yaml.safe_dump({
                    "name": "r", "task_store_backend": "in_memory",
                    "heartbeat_interval_seconds": 0,
                }))
                runner = WorkflowRunner.from_config(str(runner_yml))

                step_yml = tmp / "step.yml"
                step_yml.write_text(yaml.safe_dump({"name": "naive"}))

                async def two_step_workflow(payload):
                    # Build two naive steps; run them sequentially.
                    s1 = _NaivePauseUnawareStep.from_config(str(step_yml))
                    s2 = _NaivePauseUnawareStep.from_config(str(step_yml))
                    r1 = await s1._execute_process({"step": 1})
                    # Sleep gives the test a chance to pause between
                    # step boundaries.
                    await asyncio.sleep(0.1)
                    r2 = await s2._execute_process({"step": 2})
                    return {"r1": r1, "r2": r2}

                await runner.run_detached(two_step_workflow, "t1", {})
                # Pause arrives mid-workflow (between s1 and s2)
                await asyncio.sleep(0.05)
                await runner.pause("t1")

                # Wait long enough that, without auto-cooperation,
                # the workflow would have completed.
                await asyncio.sleep(0.3)
                h = await runner.get_handle("t1")
                assert h.status == "paused", \
                    f"naive step did not auto-honor pause; status={h.status}"

                await runner.resume("t1")
                final = await runner.await_completion("t1", timeout=2)
                assert final.status == "completed"
                assert final.result["r1"]["called"] == 1
                assert final.result["r2"]["called"] == 1
        asyncio.run(run())
