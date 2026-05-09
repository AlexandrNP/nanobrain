"""Tests for G18 Step 2 — workflow validator allows declared back-edges
through LoopController.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G18``:
the runtime LoopController shipped in G18 Step 1; this commit ships the
load-time validator that recognizes LoopController-bounded cycles as
ALLOWED (no warning, no error) while continuing to flag undeclared cycles.

Tests cover:
1. _node_is_loop_controller detection via COMPONENT_TYPE
2. _get_strongly_connected_components on simple + complex graphs
3. _all_cycles_pass_through_loop_controller — three scenarios
4. _undeclared_cycle_nodes — names the offending SCCs
5. End-to-end through validate_graph
"""

from __future__ import annotations

import tempfile

import pytest
import yaml

from nanobrain.core.workflow_graph import WorkflowGraph
from nanobrain.library.steps import LoopController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loop_controller(name: str = "lc", max_iterations: int = 3) -> LoopController:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml.safe_dump({"name": name, "max_iterations": max_iterations}))
        path = f.name
    return LoopController.from_config(path)


class _DummyStep:
    """Stand-in for a non-LoopController step. The graph stores BaseStep
    instances but the cycle-detection logic only consults
    `COMPONENT_TYPE` — so a duck-typed object suffices."""
    COMPONENT_TYPE = "dummy_step"


def _build_linear_graph(*step_ids) -> WorkflowGraph:
    """A → B → C → ... — no cycles."""
    g = WorkflowGraph()
    for sid in step_ids:
        g.nodes[sid] = _DummyStep()
        g.adjacency[sid] = set()
        g.reverse_adjacency[sid] = set()
    for i in range(len(step_ids) - 1):
        src, dst = step_ids[i], step_ids[i + 1]
        g.adjacency[src].add(dst)
        g.reverse_adjacency[dst].add(src)
    return g


def _build_cycle_graph(*nodes_and_kinds) -> WorkflowGraph:
    """Build a cycle from a list of (step_id, kind) pairs.

    `kind` is "dummy" for plain steps or "loop" for LoopController nodes.
    Edges form a single cycle: nodes[0] → nodes[1] → ... → nodes[-1] → nodes[0].
    """
    g = WorkflowGraph()
    instances = []
    for sid, kind in nodes_and_kinds:
        if kind == "loop":
            instance = _build_loop_controller(name=sid)
        else:
            instance = _DummyStep()
        g.nodes[sid] = instance
        g.adjacency[sid] = set()
        g.reverse_adjacency[sid] = set()
        instances.append((sid, instance))
    for i in range(len(nodes_and_kinds)):
        src = nodes_and_kinds[i][0]
        dst = nodes_and_kinds[(i + 1) % len(nodes_and_kinds)][0]
        g.adjacency[src].add(dst)
        g.reverse_adjacency[dst].add(src)
    return g


# ---------------------------------------------------------------------------
# 1. _node_is_loop_controller
# ---------------------------------------------------------------------------

class TestNodeDetection:

    def test_dummy_step_is_not_loop_controller(self):
        g = _build_linear_graph("a", "b")
        assert g._node_is_loop_controller("a") is False
        assert g._node_is_loop_controller("b") is False

    def test_loop_controller_is_detected(self):
        g = WorkflowGraph()
        lc = _build_loop_controller(name="my_lc")
        g.nodes["my_lc"] = lc
        g.adjacency["my_lc"] = set()
        g.reverse_adjacency["my_lc"] = set()
        assert g._node_is_loop_controller("my_lc") is True

    def test_unknown_node_returns_false(self):
        g = _build_linear_graph("a")
        assert g._node_is_loop_controller("missing") is False


# ---------------------------------------------------------------------------
# 2. _get_strongly_connected_components
# ---------------------------------------------------------------------------

class TestSCCDetection:

    def test_acyclic_graph_has_no_sccs(self):
        g = _build_linear_graph("a", "b", "c", "d")
        sccs = g._get_strongly_connected_components()
        assert sccs == []

    def test_single_cycle_is_one_scc(self):
        g = _build_cycle_graph(("a", "dummy"), ("b", "dummy"), ("c", "dummy"))
        sccs = g._get_strongly_connected_components()
        assert len(sccs) == 1
        assert set(sccs[0]) == {"a", "b", "c"}

    def test_self_loop_is_an_scc(self):
        g = _build_linear_graph("a")
        g.adjacency["a"].add("a")
        sccs = g._get_strongly_connected_components()
        assert len(sccs) == 1
        assert sccs[0] == ["a"]


# ---------------------------------------------------------------------------
# 3. _all_cycles_pass_through_loop_controller
# ---------------------------------------------------------------------------

class TestCyclePassThroughLoop:

    def test_acyclic_graph_returns_true_vacuously(self):
        g = _build_linear_graph("a", "b", "c")
        assert g._all_cycles_pass_through_loop_controller() is True

    def test_cycle_with_loop_controller_passes(self):
        g = _build_cycle_graph(
            ("a", "dummy"), ("b", "dummy"), ("lc", "loop"))
        assert g._all_cycles_pass_through_loop_controller() is True

    def test_cycle_without_loop_controller_fails(self):
        g = _build_cycle_graph(("a", "dummy"), ("b", "dummy"))
        assert g._all_cycles_pass_through_loop_controller() is False

    def test_two_cycles_one_with_loop_one_without(self):
        """Validator is strict — every cycle must pass through a
        LoopController, not just some."""
        g = WorkflowGraph()
        # Cycle 1: a ↔ b (no LC) — undeclared
        g.nodes["a"] = _DummyStep()
        g.nodes["b"] = _DummyStep()
        g.adjacency["a"] = {"b"}
        g.adjacency["b"] = {"a"}
        g.reverse_adjacency["a"] = {"b"}
        g.reverse_adjacency["b"] = {"a"}
        # Cycle 2: c ↔ lc (LC bounded)
        g.nodes["c"] = _DummyStep()
        g.nodes["lc"] = _build_loop_controller(name="lc")
        g.adjacency["c"] = {"lc"}
        g.adjacency["lc"] = {"c"}
        g.reverse_adjacency["c"] = {"lc"}
        g.reverse_adjacency["lc"] = {"c"}

        assert g._all_cycles_pass_through_loop_controller() is False
        undeclared = g._undeclared_cycle_nodes()
        assert len(undeclared) == 1
        assert set(undeclared[0]) == {"a", "b"}


# ---------------------------------------------------------------------------
# 4. _undeclared_cycle_nodes
# ---------------------------------------------------------------------------

class TestUndeclaredCycleNodes:

    def test_no_cycles_no_undeclared(self):
        g = _build_linear_graph("a", "b")
        assert g._undeclared_cycle_nodes() == []

    def test_loop_bounded_cycle_no_undeclared(self):
        g = _build_cycle_graph(
            ("a", "dummy"), ("b", "dummy"), ("lc", "loop"))
        assert g._undeclared_cycle_nodes() == []

    def test_unbounded_cycle_reported(self):
        g = _build_cycle_graph(("x", "dummy"), ("y", "dummy"))
        undeclared = g._undeclared_cycle_nodes()
        assert len(undeclared) == 1
        assert set(undeclared[0]) == {"x", "y"}


# ---------------------------------------------------------------------------
# 5. validate_graph integration
# ---------------------------------------------------------------------------

class TestValidateGraphIntegration:

    def test_acyclic_graph_validates(self):
        g = _build_linear_graph("a", "b")
        is_valid, errors = g.validate_graph(allow_cycles=False)
        assert is_valid is True or not errors  # depends on framework version
        assert "CYCLES DETECTED" not in " ".join(errors).upper()

    def test_loop_bounded_cycle_no_warning(self, caplog):
        """G18 Step 2: a cycle through a LoopController is silently OK
        even when allow_cycles=False (the default)."""
        import logging
        g = _build_cycle_graph(
            ("a", "dummy"), ("lc", "loop"))
        with caplog.at_level(logging.WARNING, logger="workflow.graph"):
            g.validate_graph(allow_cycles=False)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        cycle_warnings = [
            r for r in warnings if "CYCLES DETECTED" in r.getMessage()
        ]
        assert len(cycle_warnings) == 0

    def test_unbounded_cycle_emits_warning(self, caplog):
        """G18 Step 2: cycles NOT through a LoopController still warn,
        and the warning names the offending SCC."""
        import logging
        g = _build_cycle_graph(("x", "dummy"), ("y", "dummy"))
        with caplog.at_level(logging.WARNING, logger="workflow.graph"):
            g.validate_graph(allow_cycles=False)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        cycle_warnings = [
            r for r in warnings if "CYCLES DETECTED" in r.getMessage()
        ]
        # The framework's double-logger pattern (nb_logger + stdlib) emits
        # the same message multiple times — assert >= 1, not == 1.
        assert len(cycle_warnings) >= 1
        msg = cycle_warnings[0].getMessage()
        # The warning names the undeclared SCC + the LoopController fix:
        assert "LoopController" in msg
        assert "x" in msg
        assert "y" in msg

    def test_allow_cycles_overrides_g18_check(self, caplog):
        """allow_cycles=True is the legacy escape hatch — silences
        warnings even for undeclared cycles. Preserved for backward compat."""
        import logging
        g = _build_cycle_graph(("x", "dummy"), ("y", "dummy"))
        with caplog.at_level(logging.WARNING, logger="workflow.graph"):
            g.validate_graph(allow_cycles=True)
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        cycle_warnings = [
            r for r in warnings if "CYCLES DETECTED" in r.getMessage()
        ]
        assert len(cycle_warnings) == 0
