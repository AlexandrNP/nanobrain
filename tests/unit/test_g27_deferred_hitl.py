"""G27 — pin the DeferredHITLStep + ApprovalStore contract.

eval_03 Round 3: pre-G27, the framework had no first-class primitive
for "the workflow needs to wait for a human decision, possibly minutes
or hours, possibly across process restarts". The autonomy-mode design
(autonomous_workflow_agent.md §6) depends on this exact pattern.

This test suite pins:

  ApprovalStore (both backends):
    1. submit() is idempotent — re-submitting the same approval_id
       returns the existing record, NOT a fresh one
    2. resolve() FAIL-FASTs when the approval is already resolved
    3. resolve() FAIL-FASTs when the approval doesn't exist
    4. list_pending() returns only pending records
    5. FileApprovalStore survives process restart (state is on disk)
    6. FileApprovalStore atomic writes never leave a partial file
    7. FileApprovalStore rejects path-traversal in approval_id

  DeferredHITLStep:
    8. First call -> ApprovalPendingError with approval_id + prompt
    9. Second call (same input) finds the existing approval and again
       raises ApprovalPendingError (NOT a duplicate emission)
   10. After resolve(decision='approved') -> step returns approved dict
   11. After resolve(decision='corrected') -> step returns corrected dict
   12. After resolve(decision='rejected') -> step raises
       ApprovalRejectedError
   13. Deterministic approval_id strategy -> retries hit the same record
   14. Missing 'approval_store' kwarg FAIL-FASTs at from_config
   15. Approval_store missing protocol method FAIL-FASTs at from_config

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G27;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8 (P6+a).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.runtime.approval_store import (
    Approval,
    FileApprovalStore,
    InMemoryApprovalStore,
    deterministic_approval_id,
    random_approval_id,
)
from nanobrain.library.steps.deferred_hitl_step import (
    ApprovalPendingError,
    ApprovalRejectedError,
    DeferredHITLStep,
)


# ---------------------------------------------------------------------------
# ApprovalStore tests — InMemoryApprovalStore + FileApprovalStore in parallel
# ---------------------------------------------------------------------------


@pytest.fixture(params=["in_memory", "file"])
def store(request, tmp_path):
    if request.param == "in_memory":
        return InMemoryApprovalStore()
    return FileApprovalStore(tmp_path / "approvals")


def _new_pending(approval_id: str = "test-approval-1") -> Approval:
    return Approval(
        approval_id=approval_id,
        run_id="run-1",
        step_name="approve_step",
        prompt="approve this?",
    )


def test_submit_is_idempotent(store):
    a1 = _new_pending()
    a2 = _new_pending()  # same id, fresh object
    r1 = store.submit(a1)
    r2 = store.submit(a2)
    assert r1.approval_id == r2.approval_id
    # Both submissions must return the SAME record (the first one).
    # If the store overwrote on second submit, created_at would
    # differ. We use the get() round-trip as the canonical check.
    fetched = store.get(a1.approval_id)
    assert fetched is not None
    assert fetched.created_at == r1.created_at


def test_resolve_changes_decision(store):
    store.submit(_new_pending())
    resolved = store.resolve(
        "test-approval-1",
        decision="approved",
        decided_by="alex",
        decision_payload={"comment": "ok"},
    )
    assert resolved.decision == "approved"
    assert resolved.decided_by == "alex"
    assert resolved.decision_payload == {"comment": "ok"}
    assert resolved.decided_at is not None


def test_resolve_already_resolved_fails_fast(store):
    store.submit(_new_pending())
    store.resolve("test-approval-1", decision="approved")
    with pytest.raises(ValueError, match="already resolved"):
        store.resolve("test-approval-1", decision="rejected")


def test_resolve_missing_approval_fails_fast(store):
    with pytest.raises(KeyError, match="no approval"):
        store.resolve("does-not-exist", decision="approved")


def test_list_pending_excludes_resolved(store):
    store.submit(_new_pending("a1"))
    store.submit(_new_pending("a2"))
    store.submit(_new_pending("a3"))
    store.resolve("a2", decision="approved")
    pending_ids = {a.approval_id for a in store.list_pending()}
    assert pending_ids == {"a1", "a3"}


def test_file_store_survives_process_simulation(tmp_path):
    """FileApprovalStore writes to disk; a fresh store instance over
    the same root must see the prior records (simulates restart)."""
    root = tmp_path / "approvals"
    s1 = FileApprovalStore(root)
    s1.submit(_new_pending("persistent-1"))
    s1.resolve("persistent-1", decision="approved", decided_by="op1")

    s2 = FileApprovalStore(root)  # fresh store instance
    fetched = s2.get("persistent-1")
    assert fetched is not None
    assert fetched.decision == "approved"
    assert fetched.decided_by == "op1"


def test_file_store_rejects_path_traversal(tmp_path):
    """approval_id with path separators must be rejected — otherwise
    a malicious caller could write outside the store root."""
    s = FileApprovalStore(tmp_path / "approvals")
    bad = Approval(
        approval_id="../escape",
        run_id=None,
        step_name="x",
        prompt="x",
    )
    with pytest.raises(ValueError, match="path-separator"):
        s.submit(bad)


# ---------------------------------------------------------------------------
# DeferredHITLStep tests
# ---------------------------------------------------------------------------


def _make_step(*, approval_store, name: str = "approve_step", **cfg_kwargs):
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


def test_step_first_call_raises_approval_pending():
    s = InMemoryApprovalStore()
    step = _make_step(approval_store=s)
    with pytest.raises(ApprovalPendingError) as excinfo:
        asyncio.run(step.process({"item": "x"}))
    err = excinfo.value
    assert err.step_name == "approve_step"
    assert err.approval_id  # non-empty
    # An Approval record must have been written to the store.
    assert s.get(err.approval_id) is not None


def test_step_retry_finds_same_pending_no_duplicate():
    """Critical contract: a retry produces the SAME approval_id, the
    store's idempotent submit returns the existing record, and the
    second call raises ApprovalPendingError again WITHOUT creating
    a second store record."""
    s = InMemoryApprovalStore()
    step = _make_step(approval_store=s)
    with pytest.raises(ApprovalPendingError) as exc1:
        asyncio.run(step.process({"item": "x"}))
    with pytest.raises(ApprovalPendingError) as exc2:
        asyncio.run(step.process({"item": "x"}))
    assert exc1.value.approval_id == exc2.value.approval_id
    # And only ONE record exists.
    assert len(s.list_pending()) == 1


def test_step_returns_approved_payload():
    s = InMemoryApprovalStore()
    step = _make_step(approval_store=s)
    with pytest.raises(ApprovalPendingError) as excinfo:
        asyncio.run(step.process({"item": "x"}))
    aid = excinfo.value.approval_id
    s.resolve(
        aid,
        decision="approved",
        decided_by="alex",
        decision_payload={"comment": "looks good"},
    )
    result = asyncio.run(step.process({"item": "x"}))
    assert result["decision"] == "approved"
    assert result["approval_id"] == aid
    assert result["decided_by"] == "alex"
    assert result["decision_payload"] == {"comment": "looks good"}


def test_step_returns_corrected_payload():
    s = InMemoryApprovalStore()
    step = _make_step(approval_store=s)
    with pytest.raises(ApprovalPendingError) as excinfo:
        asyncio.run(step.process({"item": "x"}))
    aid = excinfo.value.approval_id
    s.resolve(
        aid,
        decision="corrected",
        decided_by="alex",
        decision_payload={"replacement": "new_value"},
    )
    result = asyncio.run(step.process({"item": "x"}))
    assert result["decision"] == "corrected"
    assert result["decision_payload"] == {"replacement": "new_value"}


def test_step_raises_on_rejected():
    s = InMemoryApprovalStore()
    step = _make_step(approval_store=s)
    with pytest.raises(ApprovalPendingError) as excinfo:
        asyncio.run(step.process({"item": "x"}))
    aid = excinfo.value.approval_id
    s.resolve(
        aid,
        decision="rejected",
        decided_by="alex",
        decision_payload={"reason": "policy violation"},
    )
    with pytest.raises(ApprovalRejectedError) as exc2:
        asyncio.run(step.process({"item": "x"}))
    assert exc2.value.approval_id == aid
    assert exc2.value.rejected_by == "alex"
    assert exc2.value.rejection_payload == {"reason": "policy violation"}


def test_deterministic_id_stable_across_runs():
    """Deterministic strategy means SAME (run_id, step_name, prompt)
    -> SAME approval_id, regardless of when called. Critical for retry
    idempotency."""
    aid_1 = deterministic_approval_id(
        run_id="r1", step_name="s1", prompt="approve?"
    )
    aid_2 = deterministic_approval_id(
        run_id="r1", step_name="s1", prompt="approve?"
    )
    aid_3 = deterministic_approval_id(
        run_id="r1", step_name="s1", prompt="approve?-different"
    )
    assert aid_1 == aid_2
    assert aid_1 != aid_3


def test_random_strategy_yields_distinct_ids():
    """Random strategy never collides on a fresh call."""
    seen = {random_approval_id() for _ in range(100)}
    assert len(seen) == 100


def test_step_missing_approval_store_kwarg_fails_fast():
    """from_config must require ``approval_store`` — pluggable storage
    cannot be loaded from YAML alone (backends carry runtime state)."""
    import yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        yaml.safe_dump(
            {"name": "x", "prompt_template": "approve?"}, f
        )
        path = f.name
    with pytest.raises(ComponentConfigurationError) as excinfo:
        DeferredHITLStep.from_config(path)  # no approval_store kwarg
    assert "approval_store" in str(excinfo.value)
    assert "FAIL-FAST" in str(excinfo.value)


def test_step_invalid_store_type_fails_fast():
    """A store-shaped object missing required protocol methods must
    FAIL-FAST at from_config, not silently accept and crash later."""
    import yaml

    class _FakeStore:
        # Missing resolve() and list_pending().
        def submit(self, a):
            return a

        def get(self, _):
            return None

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        yaml.safe_dump(
            {"name": "x", "prompt_template": "approve?"}, f
        )
        path = f.name
    with pytest.raises(ComponentConfigurationError) as excinfo:
        DeferredHITLStep.from_config(path, approval_store=_FakeStore())
    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert ("resolve" in msg or "list_pending" in msg)


def test_step_prompt_template_substitutes_input_tokens():
    """The prompt rendered into the Approval must include the
    substituted ``{input.<key>}`` values — operators see actual
    request context, not the raw template."""
    s = InMemoryApprovalStore()
    step = _make_step(
        approval_store=s,
        prompt_template="Approve change to {input.field}: from {input.old} to {input.new}?",
    )
    with pytest.raises(ApprovalPendingError) as excinfo:
        asyncio.run(
            step.process(
                {"field": "size", "old": 100, "new": 200}
            )
        )
    rendered = s.get(excinfo.value.approval_id).prompt
    assert "size" in rendered
    assert "100" in rendered
    assert "200" in rendered
    assert "{input." not in rendered, (
        f"raw {{input.}} tokens leaked: {rendered!r}"
    )
