"""G124 (2026-05-18) — ``settle_ms`` safe-floor on ``wait_for_cascade``.

Pin the framework-side guard against the silent-failure shape that
hid for weeks in the ``apecx-mcp-integration`` codegen adapter:
``Workflow.wait_for_cascade(settle_ms=100)`` against real LLM
workloads occasionally returned drained BEFORE the cascade had
propagated all listener-task writes, producing empty output DUs.

The G124 contract:

* ``Workflow.wait_for_cascade``'s ``settle_ms`` default is **500**
  (up from 50). New code that doesn't override sees the safe value.
* ``Workflow.run``'s ``settle_ms`` default is also **500** (up
  from 50) for symmetry.
* Callers that pass ``settle_ms < 500`` get a ``WARNING`` log
  every call, naming the workflow and the silent-failure shape.
* Setting ``NANOBRAIN_ALLOW_SHORT_SETTLE_MS=1`` suppresses the
  warning — intended for short pure-compute test fixtures where
  the 500ms overhead is prohibitive.

# ALLOWED_WAIT_FOR_CASCADE: this test pins the wait_for_cascade
# safe-floor contract itself — the workspace lint should permit it.
"""

from __future__ import annotations

import asyncio
import logging
import os

import pytest

from nanobrain.core.workflow import Workflow, WorkflowConfig


def _empty_workflow_config(name: str = "g124_test_wf") -> dict:
    """Minimal one-step workflow that satisfies the framework's
    structural validators. We need a real Workflow instance to call
    wait_for_cascade against; this is the smallest legal shape."""
    return {
        "class": "nanobrain.core.workflow.Workflow",
        "name": name,
        "config_version": 2,
        "input_data_units": {
            "wf_in": {
                "class": "nanobrain.core.data_unit.DataUnitMemory",
                "name": "wf_in",
            }
        },
        "output_data_units": {
            "wf_out": {
                "class": "nanobrain.core.data_unit.DataUnitMemory",
                "name": "wf_out",
            }
        },
        "steps": {},  # validator requires non-empty, but we use a fake below
    }


def test_wait_for_cascade_default_settle_ms_is_500():
    """G124: the default settle_ms is 500ms.

    A future PR that flips it back to 50 (or any value < 500) would
    re-introduce the silent-failure shape that hid the codegen
    adapter bug for weeks. This test catches that regression.
    """
    import inspect

    sig = inspect.signature(Workflow.wait_for_cascade)
    settle_default = sig.parameters["settle_ms"].default
    assert settle_default == 500, (
        f"Workflow.wait_for_cascade settle_ms default is {settle_default}; "
        f"G124 requires 500. A change here re-opens the silent-failure "
        f"window — see apecx-mcp-integration/docs/CODEGEN_CANARY_AND_PARITY.md."
    )


def test_workflow_run_default_settle_ms_is_500():
    """G124: Workflow.run defaults to settle_ms=500 for symmetry."""
    import inspect

    sig = inspect.signature(Workflow.run)
    settle_default = sig.parameters["settle_ms"].default
    assert settle_default == 500, (
        f"Workflow.run settle_ms default is {settle_default}; G124 "
        f"requires 500 for parity with wait_for_cascade."
    )


def test_safe_floor_constant_is_500():
    """G124: ``Workflow._SETTLE_MS_SAFE_FLOOR`` is exposed on the
    class so callers can read the floor + reason about it."""
    assert Workflow._SETTLE_MS_SAFE_FLOOR == 500


@pytest.mark.asyncio
async def test_below_floor_emits_warning(caplog, monkeypatch):
    """G124: passing settle_ms < 500 emits a WARNING log.

    The warning names the workflow and points to the incident doc.
    This is the loud-failure-mode guarantee — operators can't
    silently regress without surfacing it in logs.
    """
    monkeypatch.delenv("NANOBRAIN_ALLOW_SHORT_SETTLE_MS", raising=False)

    # Build a tiny workflow stand-in. We only need ``self.name`` and
    # ``self._SETTLE_MS_SAFE_FLOOR`` + the ``wait_for_cascade`` method.
    # Real workflow init is heavyweight; the warning logic is the
    # first thing the method does, so we can stub the executor call.
    class _StubWF(Workflow):
        def __init__(self):
            self.name = "g124_warn_test"
        def _g115_workflow_id(self):
            return "g124-warn-test-wfid"

    # Bypass the FromConfigBase constructor prohibition by direct
    # __new__ (we're the framework's own test, not user code).
    wf = _StubWF.__new__(_StubWF)
    wf.name = "g124_warn_test"

    # Stub the executor so the test doesn't need a full Workflow.
    from nanobrain.core import trigger as _trig

    class _StubExecutor:
        async def wait_for_all_tasks(self, **kwargs):
            return True

    async def _stub_get_instance():
        return _StubExecutor()

    monkeypatch.setattr(_trig.AsyncTriggerExecutor, "get_instance", _stub_get_instance)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
        result = await wf.wait_for_cascade(timeout=1.0, settle_ms=100)
    assert result is True

    warning_messages = [
        rec.message for rec in caplog.records
        if rec.levelno == logging.WARNING and "safe-floor" in rec.message
    ]
    assert warning_messages, (
        f"Expected a 'safe-floor' WARNING log when settle_ms=100; got "
        f"records: {[(r.levelno, r.message) for r in caplog.records]!r}"
    )
    assert "g124_warn_test" in warning_messages[0], (
        "Warning should name the workflow so operators can identify "
        "the culprit in a multi-workflow process."
    )


@pytest.mark.asyncio
async def test_at_or_above_floor_emits_no_warning(caplog, monkeypatch):
    """G124: passing settle_ms >= 500 does NOT emit a warning.

    Operators who've explicitly chosen the safe floor don't need
    log spam.
    """
    monkeypatch.delenv("NANOBRAIN_ALLOW_SHORT_SETTLE_MS", raising=False)

    wf = Workflow.__new__(Workflow)
    wf.name = "g124_noisy_test"

    from nanobrain.core import trigger as _trig

    class _StubExecutor:
        async def wait_for_all_tasks(self, **kwargs):
            return True

    async def _stub_get_instance():
        return _StubExecutor()

    monkeypatch.setattr(_trig.AsyncTriggerExecutor, "get_instance", _stub_get_instance)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
        await wf.wait_for_cascade(timeout=1.0, settle_ms=500)
        await wf.wait_for_cascade(timeout=1.0, settle_ms=2000)
    safe_floor_warnings = [
        rec for rec in caplog.records if "safe-floor" in rec.message
    ]
    assert not safe_floor_warnings, (
        f"Did not expect a safe-floor warning at settle_ms>=500; "
        f"got: {[r.message for r in safe_floor_warnings]!r}"
    )


@pytest.mark.asyncio
async def test_opt_out_env_var_suppresses_warning(caplog, monkeypatch):
    """G124: NANOBRAIN_ALLOW_SHORT_SETTLE_MS=1 suppresses the warning.

    Intended for short pure-compute test fixtures where the 500ms
    overhead is prohibitive AND the author has explicitly accepted
    the silent-failure risk.
    """
    monkeypatch.setenv("NANOBRAIN_ALLOW_SHORT_SETTLE_MS", "1")

    wf = Workflow.__new__(Workflow)
    wf.name = "g124_optout_test"

    from nanobrain.core import trigger as _trig

    class _StubExecutor:
        async def wait_for_all_tasks(self, **kwargs):
            return True

    async def _stub_get_instance():
        return _StubExecutor()

    monkeypatch.setattr(_trig.AsyncTriggerExecutor, "get_instance", _stub_get_instance)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="nanobrain.core.workflow"):
        await wf.wait_for_cascade(timeout=1.0, settle_ms=50)
    safe_floor_warnings = [
        rec for rec in caplog.records if "safe-floor" in rec.message
    ]
    assert not safe_floor_warnings, (
        f"NANOBRAIN_ALLOW_SHORT_SETTLE_MS=1 should suppress the warning; "
        f"got: {[r.message for r in safe_floor_warnings]!r}"
    )
