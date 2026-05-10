"""Capability tokens — G28 framework-boundary verification primitive.

eval_03 Round 3 G28: G15 (UnifiedToolDescriptor) shipped with a
``requires_capability: List[str]`` field, but no enforcement hook
existed at the workflow loader / step dispatcher boundary. Tool
authors who declared ``requires_capability: ["hpc.submit"]`` got a
free-form attribute that no caller checked — the safety surface in
``hitl_safety_gates.md §7`` and ``tool_descriptor_contract.md §6``
was paper-only.

Post-G28 the framework ships:

  * ``CapabilityNotGranted`` exception — workflow-terminal failure
    when a step requires a capability not present in the active
    WorkflowRunContext.
  * ``verify_capability(required: list[str] | str)`` — dispatch
    helper. Resolves current_run_context(); raises CapabilityNotGranted
    when any required token is absent.

P4+b decision (open question §8.9):
  * **WorkflowRunContext is the single source of truth** for tokens.
    No env var, no per-step kwarg, no global. The runner that creates
    the context is responsible for populating ``capability_tokens``
    based on the operator / API caller / agent identity. Downstream
    code does NOT consult any other source.

Sibling commit extends WorkflowRunContext with a ``capability_tokens``
field; this module exports the verification surface that consumes it.

## Why a separate module (not just a helper on WorkflowRunContext)

Verification is invoked from `core/` code paths (ToolExecutionStep
in `library/`; Workflow loader in `core/`). Keeping the helper in
`core/` lets `library/` consumers import it without circular-dep
contortion. WorkflowRunContext stays in `library/orchestration/`
because it depends on FromConfigBase patterns that core's leaf
modules avoid.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G28;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.9 (P4+b).
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class CapabilityNotGranted(Exception):
    """Raised when an operation requires a capability the active
    run context does not hold. Workflow-terminal: steps must FAIL
    rather than silently proceed without the granted privilege."""

    def __init__(
        self,
        *,
        required: List[str],
        granted: List[str],
        target_name: Optional[str] = None,
    ) -> None:
        missing = sorted(set(required) - set(granted))
        target_clause = (
            f" for {target_name!r}" if target_name else ""
        )
        super().__init__(
            f"CapabilityNotGranted{target_clause}: "
            f"required={sorted(required)}, granted={sorted(granted)}, "
            f"missing={missing}. The runner / workflow loader should "
            f"populate WorkflowRunContext.capability_tokens with the "
            f"caller's granted set BEFORE invoking the workflow OR "
            f"refuse the call upstream."
        )
        self.required = list(required)
        self.granted = list(granted)
        self.missing = missing
        self.target_name = target_name


def verify_capability(
    required: Union[str, Sequence[str]],
    *,
    target_name: Optional[str] = None,
) -> None:
    """Verify the active WorkflowRunContext holds every required token.

    Args:
        required: A single capability string, or a sequence of them.
            Empty/None = no capability required (no-op fast path).
        target_name: Optional name of the capability-requiring target
            (tool name, step name, etc.) for inclusion in error
            messages. Helps operators identify which call site failed.

    Raises:
        CapabilityNotGranted: when any required token is missing AND a
            run context is active.
        CapabilityNotGranted: ALSO when no run context is active and
            any token is required — defaults to "treat absent context
            as no granted tokens" (the strict, safe default; the
            permissive fallback would silently grant every request).
    """
    if not required:
        return  # no capability required = no-op

    if isinstance(required, str):
        required_list = [required]
    else:
        required_list = list(required)
    if not required_list:
        return

    granted = _granted_capabilities()
    missing = [t for t in required_list if t not in granted]
    if missing:
        raise CapabilityNotGranted(
            required=required_list,
            granted=granted,
            target_name=target_name,
        )


def _granted_capabilities() -> List[str]:
    """Resolve the active context's capability_tokens. Returns an
    empty list when no context is active (the strict default)."""
    try:
        # Lazy import: capabilities is core/, run_context is
        # library/orchestration/. Avoid eager dep.
        from nanobrain.library.orchestration.run_context import (
            current_run_context,
        )
    except ImportError:
        return []
    ctx = current_run_context()
    if ctx is None:
        return []
    tokens = getattr(ctx, "capability_tokens", None)
    if tokens is None:
        return []
    return list(tokens)


__all__ = ["CapabilityNotGranted", "verify_capability"]
