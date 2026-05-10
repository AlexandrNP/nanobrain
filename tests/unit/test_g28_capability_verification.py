"""G28 — pin the capability-token verification contract.

eval_03 Round 3 G28: G15 (UnifiedToolDescriptor) shipped a
``requires_capability: List[str]`` field, but no enforcement existed
at the workflow loader / step dispatcher. Tools could declare
arbitrary capabilities and no caller checked. The safety surface
described in hitl_safety_gates.md §7 was paper-only.

P4+b decision: WorkflowRunContext is the SINGLE source of truth for
capability tokens. No env var, no per-step kwarg, no global.

This test pins:
  1. verify_capability with empty required = no-op (no context check)
  2. verify_capability with all-granted tokens = no exception
  3. verify_capability with any missing token raises
     CapabilityNotGranted naming the missing token
  4. CapabilityNotGranted carries required + granted + missing
     attributes for programmatic inspection
  5. error message names target_name when provided
  6. no active run context = no granted tokens (strict default)
  7. WorkflowRunContext exposes capability_tokens property
  8. WorkflowRunContext.has_capability(token) convenience check
  9. capability_tokens property returns a copy (mutation does not
     leak into context state)
 10. ToolExecutionStep.process() FAIL-FASTs before invoking adapter
     when UTD requires_capability has unmet tokens

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G28;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.9 (P4+b).
"""
from __future__ import annotations

import pytest

from nanobrain.core.capabilities import (
    CapabilityNotGranted,
    verify_capability,
)
from nanobrain.library.orchestration.run_context import (
    WorkflowRunContext,
)


def _ctx_with(tokens):
    return WorkflowRunContext.from_config(
        {"capability_tokens": list(tokens), "run_id": "g28-test"}
    )


# ---------------------------------------------------------------------------
# verify_capability tests
# ---------------------------------------------------------------------------


def test_empty_required_is_noop():
    """No capability required = fast no-op. Doesn't even consult
    the run context (so no exception when no context is active)."""
    verify_capability([])  # empty list
    verify_capability("")  # empty string
    verify_capability(None)  # None


def test_all_granted_tokens_pass():
    ctx = _ctx_with({"hpc.submit", "data.read"})
    with ctx.activate():
        verify_capability(["hpc.submit", "data.read"])


def test_missing_token_raises_capability_not_granted():
    ctx = _ctx_with({"data.read"})
    with ctx.activate():
        with pytest.raises(CapabilityNotGranted) as excinfo:
            verify_capability(["data.read", "hpc.submit"])
    err = excinfo.value
    assert "hpc.submit" in str(err)
    assert err.missing == ["hpc.submit"]
    assert "data.read" in err.granted
    assert sorted(err.required) == ["data.read", "hpc.submit"]


def test_target_name_in_error_message():
    """Error must name WHICH tool failed so operators can act on
    the right call site."""
    ctx = _ctx_with([])
    with ctx.activate():
        with pytest.raises(CapabilityNotGranted) as excinfo:
            verify_capability(
                ["hpc.submit"], target_name="rhea:hpc.run_alphafold"
            )
    msg = str(excinfo.value)
    assert "rhea:hpc.run_alphafold" in msg
    assert "hpc.submit" in msg


def test_no_active_context_treated_as_no_granted_tokens():
    """Strict default: when no context is active, ANY required token
    fails verification. The permissive alternative (silently grant
    everything) would be a security footgun."""
    with pytest.raises(CapabilityNotGranted) as excinfo:
        verify_capability(["any.token"])
    assert excinfo.value.granted == []
    assert excinfo.value.missing == ["any.token"]


def test_string_required_treated_as_single_token():
    """A single string is treated as a one-element required list
    so callers don't have to wrap every individual token in a list."""
    ctx = _ctx_with(["hpc.submit"])
    with ctx.activate():
        verify_capability("hpc.submit")  # passes
        with pytest.raises(CapabilityNotGranted):
            verify_capability("data.write")


# ---------------------------------------------------------------------------
# WorkflowRunContext capability_tokens tests
# ---------------------------------------------------------------------------


def test_workflow_run_context_capability_tokens_property():
    ctx = _ctx_with(["a", "b", "c"])
    assert ctx.capability_tokens == ["a", "b", "c"]


def test_workflow_run_context_has_capability():
    ctx = _ctx_with(["hpc.submit"])
    assert ctx.has_capability("hpc.submit") is True
    assert ctx.has_capability("data.write") is False


def test_capability_tokens_property_returns_copy():
    """Mutating the returned list MUST NOT change the context's
    grant set (otherwise downstream code could escalate privileges
    after the runner set up the context)."""
    ctx = _ctx_with(["a", "b"])
    tokens = ctx.capability_tokens
    tokens.append("ESCALATED")
    # Context still has only the original tokens.
    assert ctx.capability_tokens == ["a", "b"]
    assert ctx.has_capability("ESCALATED") is False


def test_workflow_run_context_default_no_tokens():
    """A context built with no explicit tokens has an empty list —
    NOT None — so verify_capability sees a consistent shape."""
    ctx = WorkflowRunContext.from_config({})
    assert ctx.capability_tokens == []


# ---------------------------------------------------------------------------
# ToolExecutionStep integration test — the framework-boundary hook
# ---------------------------------------------------------------------------


def test_tool_execution_step_fails_fast_on_missing_capability():
    """ToolExecutionStep.process() MUST verify the UTD's
    requires_capability against the active context BEFORE invoking
    the adapter. A missing capability raises CapabilityNotGranted —
    the adapter never sees the call."""
    import asyncio

    from nanobrain.core.unified_tool_descriptor import (
        UnifiedToolDescriptor,
    )
    from nanobrain.library.steps.tool_execution_step import (
        ToolBackendAdapter,
        ToolBackendRegistry,
        ToolExecutionStep,
    )

    # Build an UTD that requires a capability.
    def _privileged_op() -> dict:
        return {"ok": True}

    utd = UnifiedToolDescriptor.from_python_callable(
        _privileged_op,
        backend="g28_test",
        version="0.1.0",
        provenance_class_path=f"{_privileged_op.__module__}.{_privileged_op.__name__}",
        requires_capability=["hpc.submit"],
    )

    # Simple adapter that records whether it was invoked.
    adapter_invocations: list[bool] = []

    class _RecordingAdapter(ToolBackendAdapter):
        BACKEND_NAME = "g28_test"

        async def invoke(self, utd, inputs, **_kwargs):
            adapter_invocations.append(True)
            return {"ok": True}

    ToolBackendRegistry.register(_RecordingAdapter())
    try:
        # Build the step via tmp YAML.
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            yaml.safe_dump(
                {
                    "name": "privileged_step",
                    "tool_descriptor": utd.model_dump(mode="json"),
                },
                f,
            )
            cfg_path = f.name

        step = ToolExecutionStep.from_config(cfg_path)

        # No run context active — verify_capability strict-default
        # rejects the privileged call.
        with pytest.raises(CapabilityNotGranted) as excinfo:
            asyncio.run(step.process({}))
        assert "hpc.submit" in str(excinfo.value)
        assert adapter_invocations == [], (
            f"adapter was invoked despite capability check; this is "
            f"the silent-failure shape G28 is preventing"
        )

        # With an active context that grants the capability, the
        # adapter IS invoked.
        ctx = _ctx_with(["hpc.submit"])
        with ctx.activate():
            result = asyncio.run(step.process({}))
        assert result == {"ok": True}
        assert adapter_invocations == [True]
    finally:
        ToolBackendRegistry.unregister("g28_test")
