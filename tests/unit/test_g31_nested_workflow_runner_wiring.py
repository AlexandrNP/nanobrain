"""G31 runner-side wiring — pin Workflow.run(nest_under_active_context=True).

Follow-up to G31 primitive (commit 7f37a33): the namespace_strategy
field + derive_nested_namespace helper shipped, but the runner-side
auto-install of a nested WorkflowRunContext was deferred. This commit
adds it as an explicit ``nest_under_active_context: bool = False``
kwarg on ``Workflow.run`` — caller-explicit (lower regression risk
than implicit auto-detect).

This test pins:
  1. Default behavior (no kwarg) is unchanged — no nested context,
     no namespace mutation
  2. nest_under_active_context=True with NO outer context logs a
     warning and falls through to non-nested run (does not crash)
  3. nest_under_active_context=True with outer context derives
     nested namespace via derive_nested_namespace under default
     strategy (scoped)
  4. namespace_strategy="inherit" on the nested workflow yields
     parent's namespace verbatim
  5. capability_tokens propagate from outer to nested (no privilege
     drop on the way down)
  6. nested run_id is f"{parent_run_id}.{workflow_name}" for
     audit-trail co-derivability
  7. context restoration: after the nested run completes, outer
     context is the active context again

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G31 (runner-side follow-up);
``apecx-mcp-integration/docs/development_roadmap.md`` 8.9 (P4+a).
"""
from __future__ import annotations

import asyncio
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from nanobrain.core.workflow import Workflow
from nanobrain.library.orchestration.run_context import (
    WorkflowRunContext,
    current_run_context,
)


_MINIMAL_WORKFLOW = textwrap.dedent(
    """\
    name: g31_nested_test_workflow
    description: "Minimal workflow for G31 runner-side wiring tests"
    version: "0.1.0"
    config_version: 2

    steps: {}
    links: {}
    """
)


def _stage_workflow(tmp_path: Path, *, namespace_strategy: str = "scoped") -> Path:
    body = _MINIMAL_WORKFLOW
    if namespace_strategy != "scoped":
        body = body.replace(
            "config_version: 2",
            f"config_version: 2\nnamespace_strategy: {namespace_strategy}",
        )
    workflow_path = tmp_path / "workflow.yml"
    workflow_path.write_text(body)
    return workflow_path


@pytest.fixture
def workflow(tmp_path):
    path = _stage_workflow(tmp_path)
    return Workflow.from_config(str(path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_no_kwarg_does_not_install_nested_context(workflow, caplog):
    """Default behavior (nest_under_active_context=False) must not
    mutate the contextvar. Existing callers see zero change."""
    outer = WorkflowRunContext.from_config(
        {"run_id": "outer-run", "capability_tokens": ["cap.a"]}
    )
    seen_run_ids: List[str | None] = []

    with outer.activate():
        # Run WITHOUT nest_under_active_context.
        asyncio.run(workflow.run({}))
        # The active context AFTER the run should still be outer.
        ctx = current_run_context()
        seen_run_ids.append(ctx.run_id if ctx else None)

    assert seen_run_ids == ["outer-run"], (
        f"outer context must remain active after default run; "
        f"saw {seen_run_ids}"
    )


def test_nest_with_no_outer_context_warns_and_falls_through(
    workflow, caplog
):
    """nest_under_active_context=True with NO outer context must
    log a warning and complete the run anyway — it does NOT crash.
    The caller asked for nesting but there's no parent; we can't
    auto-construct a meaningful nested context."""
    import logging

    caplog.set_level(logging.WARNING)
    # No outer context active.
    result = asyncio.run(
        workflow.run({}, nest_under_active_context=True)
    )
    assert isinstance(result, dict)
    assert "no outer WorkflowRunContext" in caplog.text


def test_nest_with_outer_context_installs_nested(workflow, tmp_path):
    """When an outer context is active and nest_under_active_context=True,
    the workflow runs UNDER a nested WorkflowRunContext whose namespace
    derives from the parent + workflow name."""
    outer = WorkflowRunContext.from_config(
        {"run_id": "outer-run-42", "capability_tokens": []}
    )
    captured_inside: Dict[str, Any] = {}

    # Patch the workflow's process() to capture the active context.
    original_process = workflow.process

    async def _capturing_process(input_data, **kwargs):
        ctx = current_run_context()
        if ctx is not None:
            captured_inside["run_id"] = ctx.run_id
            captured_inside["namespace"] = ctx.proxystore_namespace
            captured_inside["capability_tokens"] = list(ctx.capability_tokens)
        return await original_process(input_data, **kwargs)

    workflow.process = _capturing_process

    with outer.activate():
        asyncio.run(workflow.run({}, nest_under_active_context=True))

    assert captured_inside["run_id"] == "outer-run-42.g31_nested_test_workflow", (
        f"nested run_id should be parent.child; got "
        f"{captured_inside['run_id']!r}"
    )
    # Default strategy is "scoped" (per G31 P4+a).
    assert captured_inside["namespace"] == "run_outer-run-42.g31_nested_test_workflow", (
        f"scoped namespace should be parent_namespace.child_name; got "
        f"{captured_inside['namespace']!r}"
    )


def test_nest_inherit_strategy_yields_parent_namespace(tmp_path):
    """A workflow with namespace_strategy='inherit' shares the parent
    namespace verbatim (P4+a opt-in path)."""
    path = _stage_workflow(tmp_path, namespace_strategy="inherit")
    workflow = Workflow.from_config(str(path))

    outer = WorkflowRunContext.from_config(
        {"run_id": "outer-run", "capability_tokens": []}
    )
    captured: Dict[str, Any] = {}

    original_process = workflow.process

    async def _capturing_process(input_data, **kwargs):
        ctx = current_run_context()
        captured["namespace"] = ctx.proxystore_namespace
        return await original_process(input_data, **kwargs)

    workflow.process = _capturing_process

    with outer.activate():
        asyncio.run(workflow.run({}, nest_under_active_context=True))

    # Inherit means SAME namespace as parent (no append).
    assert captured["namespace"] == outer.proxystore_namespace


def test_nest_propagates_capability_tokens(workflow):
    """Capability tokens must propagate from outer to nested — a
    nested workflow does NOT lose privileges its parent had. (Authz
    is a separate concern; G28 handles per-tool checks.)"""
    outer = WorkflowRunContext.from_config(
        {
            "run_id": "outer",
            "capability_tokens": ["hpc.submit", "data.read"],
        }
    )
    captured: Dict[str, Any] = {}

    original_process = workflow.process

    async def _capturing_process(input_data, **kwargs):
        ctx = current_run_context()
        captured["tokens"] = list(ctx.capability_tokens)
        return await original_process(input_data, **kwargs)

    workflow.process = _capturing_process

    with outer.activate():
        asyncio.run(workflow.run({}, nest_under_active_context=True))

    assert sorted(captured["tokens"]) == ["data.read", "hpc.submit"]


def test_outer_context_restored_after_nested_run(workflow):
    """PEP 567 contextvar restore: after the nested run, the outer
    context is the active context again."""
    outer = WorkflowRunContext.from_config(
        {"run_id": "outer-restore-test"}
    )
    with outer.activate():
        # Before: outer
        assert current_run_context().run_id == "outer-restore-test"
        asyncio.run(workflow.run({}, nest_under_active_context=True))
        # After: outer is restored.
        assert current_run_context().run_id == "outer-restore-test", (
            "nested context did not restore outer on exit"
        )


def test_nested_run_id_is_parent_dot_workflow_name(workflow):
    """Nested run_id is f'{parent}.{workflow_name}' so audit logs
    can correlate across levels."""
    outer = WorkflowRunContext.from_config({"run_id": "abc123"})
    captured: Dict[str, Any] = {}

    original_process = workflow.process

    async def _capturing_process(input_data, **kwargs):
        captured["nested_run_id"] = current_run_context().run_id
        return await original_process(input_data, **kwargs)

    workflow.process = _capturing_process

    with outer.activate():
        asyncio.run(workflow.run({}, nest_under_active_context=True))

    assert captured["nested_run_id"] == f"abc123.{workflow.name}"
