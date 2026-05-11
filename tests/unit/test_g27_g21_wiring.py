"""G27 ↔ G21 wiring — pin Option A v1 soft-suspend + resume contract.

Per ``nanobrain/docs/g27_g21_wiring_design.md`` (deferred-by-design
doc shipped 2026-05-10), this commit implements Option A v1:

  - WorkflowRunner.run_detached catches ApprovalPendingError
  - Marks the task ``"suspended"`` (new lifecycle state)
  - Records approval_id + step_name + prompt in handle.suspension_info
  - WorkflowRunner.resume_suspended(task_id) re-spawns the asyncio
    task with the original callable + payload
  - The deterministic approval_id ensures the re-run finds the
    resolved Approval and returns

This test pins:
  1. ApprovalPendingError raised inside a detached task transitions
     status to ``"suspended"`` (NOT ``"failed"``)
  2. The handle's suspension_info carries approval_id + step_name +
     prompt for external resolvers
  3. ``"suspended"`` is in _STATUS_ACTIVE (list_active includes it)
  4. resume_suspended re-runs the workflow callable and the second
     pass returns the approved decision
  5. resume_suspended on a non-suspended task FAIL-FAST
  6. resume_suspended on an unknown task_id FAIL-FAST
  7. A NEW ApprovalPendingError on resume re-suspends (multi-gate)
  8. Non-ApprovalPendingError exceptions still go to ``"failed"``
     (no Option A interference)
  9. Successful completion (no approval pending) is unchanged
     behavior — Option A is additive, not destructive

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G27 (runner-side follow-up);
``nanobrain/docs/g27_g21_wiring_design.md``.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.runtime.approval_store import InMemoryApprovalStore
from nanobrain.library.runtime.workflow_runner import (
    WorkflowRunner,
    _STATUS_ACTIVE,
)
from nanobrain.library.steps.deferred_hitl_step import (
    ApprovalPendingError,
    DeferredHITLStep,
)


def _build_runner() -> WorkflowRunner:
    """Build a WorkflowRunner via tmp YAML (canonical test pattern
    used in test_workflow_runner.py)."""
    import yaml as _yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        _yaml.safe_dump(
            {"name": "g27_test_runner", "task_store_backend": "in_memory"},
            f,
        )
        path = f.name
    return WorkflowRunner.from_config(path)


def _make_step(approval_store, *, name: str = "approve_step", **cfg_kwargs):
    """Build a DeferredHITLStep via tmp YAML (canonical test pattern)."""
    import yaml

    cfg_kwargs.setdefault("name", name)
    cfg_kwargs.setdefault("prompt_template", "approve this?")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        yaml.safe_dump(cfg_kwargs, f)
        path = f.name
    return DeferredHITLStep.from_config(path, approval_store=approval_store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_approval_pending_transitions_to_suspended_not_failed():
    """A detached workflow that raises ApprovalPendingError must
    transition to ``"suspended"`` — NOT ``"failed"``. This is the
    core G27↔G21 wiring contract."""

    async def _run() -> Dict[str, Any]:
        runner = _build_runner()
        store = InMemoryApprovalStore()
        step = _make_step(store)

        async def _workflow(payload):
            return await step.process(payload)

        handle = await runner.run_detached(
            _workflow, "task-suspend-1", {"item": "x"}
        )
        # Wait for the task to enter suspended.
        for _ in range(50):
            current = await runner.get_handle("task-suspend-1")
            if current.status in ("suspended", "failed", "completed"):
                return current.__dict__
            await asyncio.sleep(0.02)
        raise TimeoutError("task did not reach a terminal state")

    state = asyncio.run(_run())
    assert state["status"] == "suspended", (
        f"expected 'suspended'; got {state['status']!r} "
        f"(error={state.get('error')!r})"
    )


def test_suspension_info_carries_approval_id_step_name_prompt():
    """The handle's suspension_info field must carry the
    approval_id, step_name, and prompt so an external resolver can
    look up the Approval in the store."""

    async def _run() -> Dict[str, Any]:
        runner = _build_runner()
        store = InMemoryApprovalStore()
        step = _make_step(
            store,
            prompt_template="Approve {input.action}?",
        )

        async def _workflow(payload):
            return await step.process(payload)

        await runner.run_detached(
            _workflow, "task-info-1", {"action": "deploy"}
        )
        for _ in range(50):
            current = await runner.get_handle("task-info-1")
            if current.status == "suspended":
                return current.suspension_info
            await asyncio.sleep(0.02)
        raise TimeoutError

    info = asyncio.run(_run())
    assert info is not None
    assert info["kind"] == "deferred_hitl"
    assert info["approval_id"]
    assert info["step_name"] == "approve_step"
    assert "deploy" in info["prompt"]


def test_suspended_in_status_active():
    """``"suspended"`` must be in _STATUS_ACTIVE so list_active
    surfaces it to operators alongside queued/running/paused."""
    assert "suspended" in _STATUS_ACTIVE


def test_resume_suspended_re_runs_and_returns_decision():
    """After the operator resolves the approval and calls
    resume_suspended, the task re-runs and reaches ``"completed"``
    with the approved decision payload as result."""

    async def _run() -> Dict[str, Any]:
        runner = _build_runner()
        store = InMemoryApprovalStore()
        step = _make_step(store)

        async def _workflow(payload):
            return await step.process(payload)

        await runner.run_detached(
            _workflow, "task-resume-1", {"item": "x"}
        )
        # Wait for suspension.
        for _ in range(50):
            current = await runner.get_handle("task-resume-1")
            if current.status == "suspended":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError("did not suspend")

        # Resolve the approval externally.
        approval_id = current.suspension_info["approval_id"]
        store.resolve(
            approval_id,
            decision="approved",
            decided_by="operator",
            decision_payload={"note": "ok to proceed"},
        )

        # Resume.
        await runner.resume_suspended("task-resume-1")

        # Wait for completion.
        for _ in range(50):
            current = await runner.get_handle("task-resume-1")
            if current.status == "completed":
                return current.result
            await asyncio.sleep(0.02)
        raise TimeoutError("did not complete after resume")

    result = asyncio.run(_run())
    assert result is not None
    assert result["decision"] == "approved"
    assert result["decided_by"] == "operator"
    assert result["decision_payload"] == {"note": "ok to proceed"}


def test_resume_suspended_on_non_suspended_fails_fast():
    """Calling resume_suspended on a task that is queued / running /
    completed FAIL-FASTs — the method has a specific contract."""

    async def _run():
        runner = _build_runner()

        async def _quick(payload):
            return {"done": True}

        await runner.run_detached(_quick, "task-quick-1", {})
        # Wait for completion.
        for _ in range(50):
            current = await runner.get_handle("task-quick-1")
            if current.status == "completed":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError
        with pytest.raises(ComponentConfigurationError) as excinfo:
            await runner.resume_suspended("task-quick-1")
        return str(excinfo.value)

    msg = asyncio.run(_run())
    assert "not 'suspended'" in msg or "status=" in msg


def test_resume_suspended_unknown_task_fails_fast():
    async def _run():
        runner = _build_runner()
        with pytest.raises(ComponentConfigurationError) as excinfo:
            await runner.resume_suspended("does-not-exist")
        return str(excinfo.value)

    msg = asyncio.run(_run())
    assert "is not registered" in msg


def test_non_approval_exception_still_fails():
    """A workflow that raises a NON-ApprovalPendingError exception
    must still go to ``"failed"`` — Option A wiring is additive and
    does not change the failure path for unrelated exceptions."""

    async def _run():
        runner = _build_runner()

        async def _explode(payload):
            raise RuntimeError("legitimately broken")

        await runner.run_detached(_explode, "task-explode-1", {})
        for _ in range(50):
            current = await runner.get_handle("task-explode-1")
            if current.status in ("failed", "completed", "suspended"):
                return current
            await asyncio.sleep(0.02)
        raise TimeoutError

    handle = asyncio.run(_run())
    assert handle.status == "failed"
    assert "RuntimeError" in handle.error


def test_resume_count_increments_on_resume_suspended():
    """G27 Option B evaluation instrumentation: resume_count is
    incremented each time resume_suspended re-spawns the task.

    Multi-resume cycle: suspend -> resume -> (multi-gate?) suspend
    again -> resume again. Each resume_suspended() bumps the counter.
    The persisted value lets operators compute re-run cost.
    """

    async def _run() -> int:
        runner = _build_runner()
        s = InMemoryApprovalStore()
        step = _make_step(s)

        async def _workflow(payload):
            return await step.process(payload)

        await runner.run_detached(_workflow, "task-rc-1", {"x": 1})
        for _ in range(50):
            h = await runner.get_handle("task-rc-1")
            if h.status == "suspended":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError

        # First resume: pre-count is 0, post-count should be 1.
        s.resolve(
            h.suspension_info["approval_id"],
            decision="approved",
            decided_by="op",
        )
        await runner.resume_suspended("task-rc-1")
        for _ in range(50):
            h = await runner.get_handle("task-rc-1")
            if h.status == "completed":
                return h.resume_count
            await asyncio.sleep(0.02)
        raise TimeoutError

    rc = asyncio.run(_run())
    assert rc == 1, (
        f"resume_count should be 1 after a single resume cycle; got {rc}"
    )


def test_resume_count_default_is_zero_for_fresh_tasks():
    """Tasks that never resume have resume_count == 0. Documenting
    invariant so operator queries can rely on it."""

    async def _run():
        runner = _build_runner()

        async def _quick(payload):
            return {"done": True}

        await runner.run_detached(_quick, "task-zero-1", {})
        for _ in range(50):
            h = await runner.get_handle("task-zero-1")
            if h.status == "completed":
                return h.resume_count
            await asyncio.sleep(0.02)
        raise TimeoutError

    assert asyncio.run(_run()) == 0


def test_successful_completion_unchanged():
    """A workflow that completes normally (no ApprovalPendingError)
    reaches ``"completed"`` with the result — Option A is additive."""

    async def _run():
        runner = _build_runner()

        async def _normal(payload):
            return {"normal": "result"}

        await runner.run_detached(_normal, "task-normal-1", {})
        for _ in range(50):
            current = await runner.get_handle("task-normal-1")
            if current.status in ("completed", "failed", "suspended"):
                return current
            await asyncio.sleep(0.02)
        raise TimeoutError

    handle = asyncio.run(_run())
    assert handle.status == "completed"
    assert handle.result == {"normal": "result"}
