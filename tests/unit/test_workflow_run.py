"""Tests for G8 — Workflow.run() canonical synchronous entry.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G8``:
``Workflow.run()`` wraps process() + wait_for_cascade() + output collection
into a single sync entry. Eliminates the dominant "process() returned but
nothing happened" silent-failure shape (architecture.md §13 brutal-truth #1).

These tests cover the helper API surface in isolation. Full end-to-end
integration with a real workflow that actually drives a trigger cascade
is exercised by the existing integration tests in apecx-mcp-integration
(once they migrate to use run() instead of process()+wait_for_cascade()).

This file targets:
1. Workflow.run signature (no_first_step shape, await_cascade flag)
2. _collect_workflow_output_data_units helper
3. Status-field semantics in the returned dict
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobrain.core.workflow import Workflow


# ---------------------------------------------------------------------------
# Helpers — minimal Workflow stub via attribute injection
# ---------------------------------------------------------------------------
#
# Building a real Workflow via Workflow.from_config() requires a YAML file
# describing steps, links, and triggers. For testing the run() helper in
# isolation, we create a minimal object that has the methods run() needs.

class _FakeDataUnit:
    """Minimal stand-in for DataUnitBase — supports the .get() async API."""

    def __init__(self, value):
        self._value = value

    async def get(self):
        return self._value


def _make_workflow_stub(
    *,
    process_returns=None,
    cascade_drains=True,
    output_units=None,
):
    """Build an object that quacks like a Workflow for run()-helper tests.

    We sidestep the real Workflow constructor (which requires from_config
    + many dependencies); the run() method only consults a few attributes
    and async methods, so we mock those directly. Real-cascade integration
    tests live elsewhere.
    """
    wf = MagicMock(spec=Workflow)
    wf.name = "stub"
    wf.process = AsyncMock(
        return_value=process_returns if process_returns is not None
        else {"status": "data_flow_initiated"}
    )
    wf.wait_for_cascade = AsyncMock(return_value=cascade_drains)

    wf.step_output_data_units = {
        name: _FakeDataUnit(value) for name, value in (output_units or {}).items()
    }

    # Bind the REAL run() and helper from Workflow to the stub:
    wf.run = Workflow.run.__get__(wf, type(wf))
    wf._collect_workflow_output_data_units = (
        Workflow._collect_workflow_output_data_units.__get__(wf, type(wf))
    )
    return wf


# ---------------------------------------------------------------------------
# 1. Output collection
# ---------------------------------------------------------------------------

class TestCollectWorkflowOutputDataUnits:

    def test_empty_owned_returns_empty_dict(self):
        async def run():
            wf = _make_workflow_stub(output_units={})
            result = await wf._collect_workflow_output_data_units()
            assert result == {}
        asyncio.run(run())

    def test_collects_all_owned_units(self):
        async def run():
            wf = _make_workflow_stub(output_units={
                "answer": "hello",
                "score": 0.95,
            })
            result = await wf._collect_workflow_output_data_units()
            assert result == {"answer": "hello", "score": 0.95}
        asyncio.run(run())

    def test_data_unit_get_failure_does_not_poison_others(self):
        """A bad data unit records the error inline; siblings still
        return their values. Workflow-level robustness for partial outputs."""
        async def run():
            class _BadDataUnit:
                async def get(self):
                    raise RuntimeError("boom")

            wf = _make_workflow_stub(output_units={"good": "value"})
            wf.step_output_data_units["bad"] = _BadDataUnit()

            result = await wf._collect_workflow_output_data_units()
            assert result["good"] == "value"
            assert result["bad"] is None
            assert "_errors" in result
            assert "bad" in result["_errors"]
            assert "boom" in result["_errors"]["bad"]
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 2. Workflow.run() — happy path + status semantics
# ---------------------------------------------------------------------------

class TestWorkflowRun:

    def test_completed_status_on_clean_drain(self):
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=True,
                output_units={"answer": "hi"},
            )
            result = await wf.run({"query": "x"})
            assert result["status"] == "completed"
            assert result["answer"] == "hi"
            wf.process.assert_called_once_with({"query": "x"})
            wf.wait_for_cascade.assert_called_once()
        asyncio.run(run())

    def test_default_input_data_empty_dict(self):
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=True,
                output_units={},
            )
            result = await wf.run()  # no input_data
            assert result["status"] == "completed"
            wf.process.assert_called_once_with({})
        asyncio.run(run())

    def test_cascade_timeout_returns_status_field(self):
        """Default behavior: timeout returns a status, not an exception.
        Operators get partial outputs to diagnose."""
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=False,
                output_units={"partial": "data"},
            )
            result = await wf.run({}, timeout=0.1)
            assert result["status"] == "cascade_timeout"
            assert result["_timeout_seconds"] == 0.1
            assert result["partial"] == "data"
        asyncio.run(run())

    def test_cascade_timeout_raises_when_opted_in(self):
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=False,
                output_units={},
            )
            with pytest.raises(TimeoutError) as exc_info:
                await wf.run({}, timeout=0.1, raise_on_cascade_timeout=True)
            assert "stub" in str(exc_info.value)
            assert "cascade did not drain" in str(exc_info.value)
        asyncio.run(run())

    def test_no_first_step_short_circuits(self):
        """Workflow.process() returns no_first_step status; run() echoes
        it without waiting for a cascade that won't fire."""
        async def run():
            wf = _make_workflow_stub(
                process_returns={"status": "no_first_step", "workflow": "stub"},
                output_units={},
            )
            result = await wf.run({})
            assert result["status"] == "no_first_step"
            wf.wait_for_cascade.assert_not_called()
        asyncio.run(run())

    def test_await_cascade_false_skips_wait(self):
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=True,
                output_units={"answer": "current"},
            )
            result = await wf.run({}, await_cascade=False)
            assert result["status"] == "completed_no_await"
            assert result["answer"] == "current"
            wf.wait_for_cascade.assert_not_called()
            assert "_process_return" in result
        asyncio.run(run())

    def test_passes_kwargs_through_to_process(self):
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=True, output_units={})
            await wf.run({"x": 1}, my_custom_kwarg="foo")
            wf.process.assert_called_once_with({"x": 1}, my_custom_kwarg="foo")
        asyncio.run(run())

    def test_passes_timeout_settle_to_wait_for_cascade(self):
        async def run():
            wf = _make_workflow_stub(
                cascade_drains=True, output_units={})
            await wf.run({}, timeout=12.5, settle_ms=200)
            wf.wait_for_cascade.assert_called_once_with(
                timeout=12.5, settle_ms=200)
        asyncio.run(run())
