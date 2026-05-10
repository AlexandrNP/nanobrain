"""G31 — pin the nested-workflow / namespace-strategy contract.

eval_03 Round 3 G31: pre-G31 the framework had "Workflow IS a Step"
in the design but no first-class primitive for COMPOSING workflows
(loading a workflow YAML as a step inside another workflow). The
meta-workflow design's Strategy B (skeleton composition) produces a
parent workflow with K skeletons embedded as sub-workflows; without
G31 those nested workflows had to be flattened or hand-orchestrated.

This commit ships:

  * ``WorkflowConfig.namespace_strategy`` field (Literal["scoped",
    "inherit"]; default "scoped"). When this Workflow is loaded as a
    nested step, the strategy controls how its data-units namespace
    derives from the outer parent's run context.

  * ``derive_nested_namespace(parent_namespace, child_workflow_name,
    strategy)`` — pure helper. The runner / framework loader calls
    this when activating a nested run context for a child workflow.

P4+a decision (open question §8.9): ``scoped`` is the default because
silent-namespace-collision is a worse failure than over-isolation.

This test pins:
  1. WorkflowConfig.namespace_strategy default is "scoped"
  2. WorkflowConfig.namespace_strategy accepts "inherit"
  3. WorkflowConfig.namespace_strategy rejects bogus values
  4. derive_nested_namespace scoped strategy appends child name
  5. derive_nested_namespace inherit strategy returns parent verbatim
  6. derive_nested_namespace empty parent + scoped = child name only
  7. derive_nested_namespace scoped + empty child FAIL-FAST
  8. derive_nested_namespace unknown strategy FAIL-FAST
  9. inherit + empty parent works (returns "")
 10. nested namespaces compose cleanly (a.b.c shape under double nest)
 11. namespace_strategy field roundtrips through model_dump

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G31;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.9 (P4+a).
"""
from __future__ import annotations

import pytest

from nanobrain.core.workflow import (
    WorkflowConfig,
    derive_nested_namespace,
)


def _build_workflow_config(**overrides) -> WorkflowConfig:
    """Build a minimal valid WorkflowConfig via the framework backdoor."""
    base = {"name": "test_workflow", **overrides}
    WorkflowConfig._allow_direct_instantiation = True
    try:
        return WorkflowConfig(**base)
    finally:
        WorkflowConfig._allow_direct_instantiation = False


# ---------------------------------------------------------------------------
# WorkflowConfig.namespace_strategy field tests
# ---------------------------------------------------------------------------


def test_namespace_strategy_default_is_scoped():
    cfg = _build_workflow_config()
    assert cfg.namespace_strategy == "scoped"


def test_namespace_strategy_accepts_inherit():
    cfg = _build_workflow_config(namespace_strategy="inherit")
    assert cfg.namespace_strategy == "inherit"


def test_namespace_strategy_rejects_bogus_value():
    """The Literal type guards against typos like 'isolated' /
    'shared' that operators might intuit but the framework does
    not understand."""
    with pytest.raises(Exception) as excinfo:
        _build_workflow_config(namespace_strategy="bogus_value")
    msg = str(excinfo.value).lower()
    assert (
        "scoped" in msg or "inherit" in msg or "literal" in msg
    ), f"validation error must hint at valid values; got: {excinfo.value}"


def test_namespace_strategy_field_roundtrips():
    cfg = _build_workflow_config(namespace_strategy="inherit")
    dumped = cfg.model_dump()
    assert dumped["namespace_strategy"] == "inherit"


# ---------------------------------------------------------------------------
# derive_nested_namespace tests
# ---------------------------------------------------------------------------


def test_scoped_strategy_appends_child_name():
    result = derive_nested_namespace(
        parent_namespace="run_abc123",
        child_workflow_name="rag_synthesis",
        strategy="scoped",
    )
    assert result == "run_abc123.rag_synthesis"


def test_scoped_is_the_default_strategy():
    """The default strategy is 'scoped' — call without strategy=
    to verify the default applies."""
    result = derive_nested_namespace(
        parent_namespace="run_abc123",
        child_workflow_name="rag_synthesis",
    )
    assert result == "run_abc123.rag_synthesis"


def test_inherit_strategy_returns_parent_verbatim():
    result = derive_nested_namespace(
        parent_namespace="run_abc123",
        child_workflow_name="rag_synthesis",
        strategy="inherit",
    )
    assert result == "run_abc123"


def test_scoped_with_empty_parent_returns_child_name():
    """No parent namespace = top-level child. The child's name
    becomes the full namespace verbatim — equivalent to a top-level
    workflow with that name as namespace."""
    result = derive_nested_namespace(
        parent_namespace="",
        child_workflow_name="rag_synthesis",
        strategy="scoped",
    )
    assert result == "rag_synthesis"


def test_scoped_with_empty_child_fails_fast():
    """An empty child name under scoped strategy would produce
    'parent.' which is ambiguous and likely a programmer error."""
    with pytest.raises(ValueError, match="non-empty child_workflow_name"):
        derive_nested_namespace(
            parent_namespace="run_abc",
            child_workflow_name="",
            strategy="scoped",
        )


def test_unknown_strategy_fails_fast():
    with pytest.raises(ValueError, match="must be 'scoped' or 'inherit'"):
        derive_nested_namespace(
            parent_namespace="x",
            child_workflow_name="y",
            strategy="rogue_strategy",
        )


def test_inherit_with_empty_parent():
    """Inherit + empty parent = empty result. Edge case but well-
    defined; matches "child shares whatever the parent had"."""
    result = derive_nested_namespace(
        parent_namespace="",
        child_workflow_name="anything",
        strategy="inherit",
    )
    assert result == ""


def test_double_nesting_composes_cleanly():
    """Two levels of nesting produce a 3-component dotted namespace.
    This is the core multi-skeleton-composition contract — every
    layer's namespace is unique."""
    parent = "run_abc"
    level_1 = derive_nested_namespace(
        parent_namespace=parent,
        child_workflow_name="orchestrator",
        strategy="scoped",
    )
    level_2 = derive_nested_namespace(
        parent_namespace=level_1,
        child_workflow_name="rag_inner",
        strategy="scoped",
    )
    assert level_2 == "run_abc.orchestrator.rag_inner"


def test_mixed_strategies_in_chain():
    """Inherit at one level + scoped at another. The child of an
    inherit-shared layer still gets scoped under the parent's
    namespace, NOT the inheritor's name."""
    parent = "run_abc"
    layer_1 = derive_nested_namespace(
        parent_namespace=parent,
        child_workflow_name="passthrough",
        strategy="inherit",
    )
    # layer_1 == parent
    layer_2 = derive_nested_namespace(
        parent_namespace=layer_1,
        child_workflow_name="leaf",
        strategy="scoped",
    )
    assert layer_2 == "run_abc.leaf"


def test_namespace_collision_avoidance_via_scoped():
    """Two child workflows with the same name under different
    parents get distinct namespaces — the silent-collision shape
    G31 prevents."""
    a = derive_nested_namespace(
        parent_namespace="tenant_a",
        child_workflow_name="rag",
        strategy="scoped",
    )
    b = derive_nested_namespace(
        parent_namespace="tenant_b",
        child_workflow_name="rag",
        strategy="scoped",
    )
    assert a != b
    assert a == "tenant_a.rag"
    assert b == "tenant_b.rag"
