"""Adversarial-probe regression tests for the eval_03 arc.

This module pins three bugs the adversarial probes (2026-05-11)
found in the recently-shipped framework primitives:

  1. ``CostTracker.record`` accepted NaN (silent ledger corruption).
  2. ``_subscriber_stack`` contextvar default was a mutable list
     (tripwire — future code paths could mutate the shared default).
  3. ``WorkflowRunner.resume_suspended`` had a concurrent-call race
     (two callers both pass the status check, both spawn duplicate
     asyncio tasks for the same task_id).

Each pin is a small adversarial input that would have triggered the
original bug. A regression that re-introduces the bug fires the
matching test.

Source: chain of 2026-05-11; adversarial bug-hunt across eval_03
arc primitives.
"""
from __future__ import annotations

import asyncio
import math
import tempfile
from typing import Any, Dict

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.cost_envelope import (
    CostEnvelope,
    CostTracker,
)
from nanobrain.core.step_events import (
    StepEvent,
    publish_step_event,
    subscribe_to_step_events,
)
from nanobrain.library.runtime.approval_store import InMemoryApprovalStore
from nanobrain.library.runtime.workflow_runner import (
    WorkflowRunner,
)
from nanobrain.library.steps.deferred_hitl_step import (
    ApprovalPendingError,
    DeferredHITLStep,
)


# ---------------------------------------------------------------------------
# Probe 1: CostTracker NaN / Inf
# ---------------------------------------------------------------------------


def test_record_rejects_nan_explicitly():
    """NaN comparisons always return False, so ``NaN < 0`` is False
    AND ``NaN > cap`` is False. Without an explicit isnan guard,
    NaN slips past every check and corrupts the ledger silently.
    """
    tracker = CostTracker(CostEnvelope(usd=100.0))
    with pytest.raises(ValueError, match="NaN"):
        tracker.record("usd", float("nan"))
    # Ledger MUST be unchanged after the rejection.
    assert tracker.cumulative("usd") == 0.0


def test_record_rejects_positive_infinity():
    """Infinity would corrupt the ledger by locking it into a
    perpetual breach state."""
    tracker = CostTracker(CostEnvelope(usd=100.0))
    with pytest.raises(ValueError, match="finite"):
        tracker.record("usd", float("inf"))
    assert tracker.cumulative("usd") == 0.0


def test_record_rejects_negative_infinity():
    """-inf would underflow the cumulative ledger. Same fix path."""
    tracker = CostTracker(CostEnvelope(usd=100.0))
    with pytest.raises(ValueError):
        tracker.record("usd", float("-inf"))
    assert tracker.cumulative("usd") == 0.0


def test_record_still_accepts_finite_zero():
    """Probe didn't break the boundary: 0.0 is valid (no-op record)."""
    tracker = CostTracker(CostEnvelope(usd=100.0))
    tracker.record("usd", 0.0)
    assert tracker.cumulative("usd") == 0.0


def test_record_still_accepts_finite_positive():
    """Boundary case: a normal finite positive value records normally."""
    tracker = CostTracker(CostEnvelope(usd=100.0))
    tracker.record("usd", 42.5)
    assert tracker.cumulative("usd") == 42.5


# ---------------------------------------------------------------------------
# Probe 2: _subscriber_stack mutable-default tripwire
# ---------------------------------------------------------------------------


def test_subscriber_stack_default_is_immutable_tuple():
    """The contextvar default must be a tuple (immutable), not a list.
    Mutating the returned default would leak across every context
    that hadn't ``.set()``-ed — a silent multi-context bug class."""
    from nanobrain.core.step_events import _subscriber_stack

    default = _subscriber_stack.get()
    assert isinstance(default, tuple), (
        f"_subscriber_stack default must be a tuple; got "
        f"{type(default).__name__}. Mutable defaults are a tripwire."
    )
    # And calling .append on a tuple raises AttributeError, which
    # is the protection mechanism (compared to silent corruption
    # of the shared list default).
    with pytest.raises(AttributeError):
        default.append("rogue")  # type: ignore[attr-defined]


def test_subscribers_propagate_via_immutable_tuples():
    """End-to-end check that the immutable-default change doesn't
    break the subscribe/publish flow."""
    captured = []

    def _sub(event: StepEvent) -> None:
        captured.append(event)

    fake_event = StepEvent(
        event_type="step_start",
        step_name="probe",
        run_id=None,
        timestamp_iso="2026-05-11T00:00:00Z",
    )
    with subscribe_to_step_events(_sub):
        publish_step_event(fake_event)
    assert captured == [fake_event]


# ---------------------------------------------------------------------------
# Probe 3: resume_suspended concurrent-call race
# ---------------------------------------------------------------------------


def _build_runner_via_tmp_yaml() -> WorkflowRunner:
    import yaml as _yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        _yaml.safe_dump(
            {
                "name": "probe_runner",
                "task_store_backend": "in_memory",
            },
            f,
        )
        path = f.name
    return WorkflowRunner.from_config(path)


def _make_hitl_step(approval_store) -> DeferredHITLStep:
    import yaml as _yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        _yaml.safe_dump(
            {
                "name": "probe_hitl",
                "prompt_template": "approve?",
            },
            f,
        )
        path = f.name
    return DeferredHITLStep.from_config(path, approval_store=approval_store)


def test_resume_suspended_refuses_concurrent_call():
    """Two callers race resume_suspended on the same suspended
    task. The second caller must FAIL-FAST instead of spawning a
    duplicate asyncio task (which would corrupt status writes)."""

    async def _run() -> str:
        runner = _build_runner_via_tmp_yaml()
        store = InMemoryApprovalStore()
        step = _make_hitl_step(store)

        async def _workflow(payload):
            return await step.process(payload)

        await runner.run_detached(_workflow, "race-task-1", {"x": 1})
        # Wait for suspension.
        for _ in range(50):
            h = await runner.get_handle("race-task-1")
            if h.status == "suspended":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError("did not suspend")

        # Resolve the approval (otherwise the resume re-suspends).
        store.resolve(
            h.suspension_info["approval_id"],
            decision="approved",
            decided_by="op",
        )

        # First resume: starts the asyncio task. We do NOT await
        # completion; second resume must see the in-flight task and
        # refuse.
        await runner.resume_suspended("race-task-1")
        try:
            await runner.resume_suspended("race-task-1")
        except ComponentConfigurationError as exc:
            # Expected: in-flight asyncio task already running.
            # Let the first resume drain to clean teardown.
            for _ in range(50):
                h = await runner.get_handle("race-task-1")
                if h.status in ("completed", "failed"):
                    return str(exc)
                await asyncio.sleep(0.02)
            return str(exc)
        # Wait for first resume to drain regardless of second-call
        # outcome.
        for _ in range(50):
            h = await runner.get_handle("race-task-1")
            if h.status in ("completed", "failed"):
                break
            await asyncio.sleep(0.02)
        pytest.fail(
            "second resume_suspended did not raise — concurrent-call "
            "race guard is missing"
        )
        return ""  # unreachable; satisfy mypy

    msg = asyncio.run(_run())
    # The race guard manifests in two ways depending on micro-timing:
    #   (a) second caller passes the status-check window microseconds
    #       after the first resume has flipped status suspended ->
    #       queued. Second caller's status check fails with
    #       "is in status='queued', not 'suspended'". This is the
    #       FIRST line of defense and the most common observation.
    #   (b) second caller passes the status check BEFORE the first
    #       resume updates the store. Status is still "suspended".
    #       Then the explicit _tasks-dict guard catches it with the
    #       "in-flight" / "racing" message.
    # Either outcome is a valid race-prevention path; the bug
    # would manifest as BOTH callers spawning duplicate asyncio
    # tasks AND silently corrupting status writes.
    assert (
        "in status='queued'" in msg
        or "in-flight" in msg
        or "racing" in msg
    ), (
        f"race-guard error message must hint at status mismatch "
        f"OR in-flight detection; got: {msg!r}"
    )


def test_resume_suspended_does_not_refuse_re_resume_after_complete():
    """After a resume completes (status -> completed), the in-flight
    task IS done — a fresh resume_suspended on a NEW suspended cycle
    must work. Verify the race guard doesn't false-positive on a
    cleanly-finished prior task slot."""
    # This is the multi-gate / multi-resume scenario. We test it by
    # forcing the task back into "suspended" manually (the in-memory
    # store permits direct mutation; in real flow this happens when a
    # multi-gate workflow re-raises ApprovalPendingError on resume).

    async def _run():
        runner = _build_runner_via_tmp_yaml()
        store = InMemoryApprovalStore()
        step = _make_hitl_step(store)

        async def _workflow(payload):
            return await step.process(payload)

        await runner.run_detached(_workflow, "multi-1", {"x": 1})
        for _ in range(50):
            h = await runner.get_handle("multi-1")
            if h.status == "suspended":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError

        store.resolve(
            h.suspension_info["approval_id"],
            decision="approved",
            decided_by="op",
        )
        await runner.resume_suspended("multi-1")
        for _ in range(50):
            h = await runner.get_handle("multi-1")
            if h.status == "completed":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError

        # Now simulate a second suspend cycle by hand: set status
        # back to "suspended" + reinstate suspension_info. (The
        # real-flow trigger for this is a multi-gate workflow.)
        h = await runner._store.get("multi-1")
        h.status = "suspended"
        h.suspension_info = {
            "kind": "deferred_hitl",
            "approval_id": h.suspension_info["approval_id"]
            if h.suspension_info
            else "x",
            "step_name": "probe_hitl",
            "prompt": "approve?",
        }
        await runner._store.update(h)

        # Resume the new suspension. The prior task slot is done,
        # so the race guard must NOT false-positive.
        await runner.resume_suspended("multi-1")
        return True

    asyncio.run(_run())
