"""Regression tests for the four framework fixes surfaced during the
2026-05-17 G99 TDR-as-YAML integration test.

The G99 integration test (in apecx-mcp-integration:
``tests/integration/test_tdr_workflow_against_ollama.py``) drove a
cycle-bearing workflow end-to-end against real Ollama and surfaced
four framework bugs that the previous validator+link unit tests
missed because they exercised the load-time path only. The fixes:

  1. ``WorkflowGraph._get_cycles_info`` — added; was referenced by
     ``validate_graph`` on the G18 cycle-allowed success branch but
     undefined. The resulting AttributeError was swallowed by
     ``handle_error`` which then falsely reported validation failure.
     Net effect: cycle-bearing workflows with LoopController bounding
     loaded but refused to run.

  2. ``ConditionalLink._init_from_config`` now sets ``self.auto_transfer``
     from component_config. Before this fix the attribute was never
     bound on ConditionalLink instances, so the LinkBase auto-transfer
     path read ``getattr(self, 'auto_transfer', False)`` and treated
     EVERY ConditionalLink as a no-op link.

  3. ``ConditionalLink.transfer`` (condition-met branch) gained a
     ``self.target.set(data)`` fallback for when target is a data unit
     directly. Before, the branch only handled targets with
     ``input_data_units`` or ``set_input``; data-unit targets silently
     no-op'd. The ``gate_to_bottom`` branch already had this fallback;
     this is parity.

  4. ``LoopController.process`` now unwraps the trigger envelope
     (``{<input_data_unit_name>: <payload>}``). Before, when invoked
     via the data-driven cascade, the controller passed the wrapped
     dict through under ``payload``, so downstream Steps consuming
     the controller output via a back-edge received doubly-wrapped
     data.

All four bugs were silent failures: ``load + start`` succeeded, the
workflow logs reported "transfer complete" / "iteration N", but no
actual data ever propagated through the cycle. The bugs are tested
both in isolation (this file) and end-to-end via the apecx-side TDR
workflow integration test.
"""

from __future__ import annotations

import asyncio
import tempfile

import pytest
import yaml

from nanobrain.core.link import ConditionalLink
from nanobrain.core.workflow_graph import WorkflowGraph
from nanobrain.library.steps.loop_controller import LoopController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_loop_controller(name: str = "lc", max_iterations: int = 3) -> LoopController:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml.safe_dump({"name": name, "max_iterations": max_iterations}))
        path = f.name
    return LoopController.from_config(path)


class _DummyStep:
    COMPONENT_TYPE = "dummy_step"


# ---------------------------------------------------------------------------
# Fix 1 — WorkflowGraph._get_cycles_info exists and returns useful string
# ---------------------------------------------------------------------------


class TestGetCyclesInfo:
    """The method was referenced by validate_graph on the cycle-allowed
    branches but didn't exist. Pin its presence + output shape."""

    def test_method_exists(self):
        g = WorkflowGraph()
        assert callable(getattr(g, "_get_cycles_info", None))

    def test_empty_graph_returns_empty_brackets(self):
        g = WorkflowGraph()
        assert g._get_cycles_info() == "[]"

    def test_acyclic_graph_returns_empty_brackets(self):
        g = WorkflowGraph()
        for sid in ("a", "b", "c"):
            g.nodes[sid] = _DummyStep()
            g.adjacency[sid] = set()
            g.reverse_adjacency[sid] = set()
        g.adjacency["a"].add("b")
        g.adjacency["b"].add("c")
        g.reverse_adjacency["b"].add("a")
        g.reverse_adjacency["c"].add("b")
        assert g._get_cycles_info() == "[]"

    def test_cycle_renders_arrow_chain(self):
        """Multi-node SCC renders as 'a -> b -> a' (loop-back marker)."""
        g = WorkflowGraph()
        for sid in ("a", "b"):
            g.nodes[sid] = _DummyStep()
            g.adjacency[sid] = set()
            g.reverse_adjacency[sid] = set()
        # a → b → a
        g.adjacency["a"].add("b")
        g.adjacency["b"].add("a")
        g.reverse_adjacency["b"].add("a")
        g.reverse_adjacency["a"].add("b")
        rendered = g._get_cycles_info()
        # SCC order isn't guaranteed; both renderings are valid.
        assert rendered in ("['a -> b -> a']", "['b -> a -> b']")

    def test_self_loop_renders(self):
        g = WorkflowGraph()
        g.nodes["x"] = _DummyStep()
        g.adjacency["x"] = {"x"}
        g.reverse_adjacency["x"] = {"x"}
        rendered = g._get_cycles_info()
        assert rendered == "['x -> x']"

    def test_validate_graph_through_cycle_allowed_branch_does_not_raise(self):
        """End-to-end check: validate_graph used to crash on the
        success branch where _get_cycles_info was logged. After the
        fix, it returns (True, []) cleanly."""
        g = WorkflowGraph()
        lc = _build_loop_controller(name="my_lc")
        a = _DummyStep()
        g.nodes["a"] = a
        g.nodes["my_lc"] = lc
        g.adjacency["a"] = {"my_lc"}
        g.adjacency["my_lc"] = {"a"}
        g.reverse_adjacency["a"] = {"my_lc"}
        g.reverse_adjacency["my_lc"] = {"a"}

        # Cycle present; allow_cycles=False; but LoopController is in the cycle,
        # so G18 Step 2 should allow it. Before the fix, the log statement on
        # this branch raised AttributeError on _get_cycles_info, which
        # handle_error swallowed and turned into (False, ["Graph validation failed"]).
        is_valid, errors = g.validate_graph(allow_cycles=False, require_connected=True)
        assert is_valid is True, f"expected valid, got errors: {errors}"


# ---------------------------------------------------------------------------
# Fix 2 — ConditionalLink._init_from_config sets self.auto_transfer
# ---------------------------------------------------------------------------


class TestConditionalLinkAutoTransferAttribute:
    """LinkBase._setup_automatic_transfer_if_possible reads
    ``getattr(self, 'auto_transfer', False)``. ConditionalLink never
    set the attribute, so every ConditionalLink was silently treated
    as a no-op link."""

    def _build_conditional_link(self, auto_transfer: bool = True) -> ConditionalLink:
        # Build a minimal ConditionalLink. Needs source + target + condition.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml.safe_dump({
                "name": "cl",
                "link_type": "conditional",
                "source": "src_step.src_du",
                "target": "tgt_step.tgt_du",
                "condition": {"op": "eq", "field": "x", "value": 1},
                "auto_transfer": auto_transfer,
            }))
            path = f.name
        return ConditionalLink.from_config(path)

    def test_auto_transfer_true_sets_instance_attr(self):
        link = self._build_conditional_link(auto_transfer=True)
        assert hasattr(link, "auto_transfer"), (
            "ConditionalLink instance is missing the auto_transfer attribute. "
            "LinkBase._setup_automatic_transfer_if_possible reads via "
            "getattr(self, 'auto_transfer', False) — missing attribute "
            "silently disables the link."
        )
        assert link.auto_transfer is True

    def test_auto_transfer_false_sets_instance_attr(self):
        link = self._build_conditional_link(auto_transfer=False)
        assert hasattr(link, "auto_transfer")
        assert link.auto_transfer is False


# ---------------------------------------------------------------------------
# Fix 3 — ConditionalLink.transfer data-unit fallback when condition matches
# ---------------------------------------------------------------------------


class _FakeDataUnit:
    """Tiny stand-in: just records what was set()."""

    def __init__(self):
        self.values = []

    async def set(self, value):
        self.values.append(value)

    async def get(self):
        return self.values[-1] if self.values else None


class TestConditionalLinkTransferDataUnitFallback:
    """When ConditionalLink.transfer's condition matches, the
    condition-met branch tried target.input_data_units[0] OR
    target.set_input. If target is itself a data unit (which is the
    normal case for workflow-level outputs and step input data units
    accessed by name), neither path triggers, so the transfer
    silently no-op'd. The fix adds a final ``self.target.set(data)``
    fallback in parity with the gate_to_bottom branch."""

    @pytest.mark.asyncio
    async def test_condition_met_writes_to_data_unit_target(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml.safe_dump({
                "name": "cl",
                "link_type": "conditional",
                "source": "src",
                "target": "tgt",
                "condition": {"op": "eq", "field": "x", "value": 1},
                "auto_transfer": True,
            }))
            path = f.name
        link = ConditionalLink.from_config(path)
        # Swap the target to a bare data unit (the production case
        # surfaced by workflow-level output_data_units).
        target_du = _FakeDataUnit()
        link.target = target_du
        link._is_active = True

        payload = {"x": 1, "data": "hello"}
        await link.transfer(payload)

        assert target_du.values == [payload], (
            "ConditionalLink with condition met failed to write to a "
            "data-unit target. The condition-met branch lacked the "
            "fallback that the gate_to_bottom branch had — silent no-op."
        )

    @pytest.mark.asyncio
    async def test_condition_not_met_does_not_write_in_publish_empty_mode(self):
        """Sanity: in the default publish_empty mode, condition-miss
        is a no-op. Confirms the fix doesn't accidentally introduce a
        write."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml.safe_dump({
                "name": "cl",
                "link_type": "conditional",
                "source": "src",
                "target": "tgt",
                "condition": {"op": "eq", "field": "x", "value": 1},
                "auto_transfer": True,
            }))
            path = f.name
        link = ConditionalLink.from_config(path)
        target_du = _FakeDataUnit()
        link.target = target_du
        link._is_active = True

        await link.transfer({"x": 999, "data": "no-match"})
        assert target_du.values == []


# ---------------------------------------------------------------------------
# Fix 4 — LoopController unwraps the trigger envelope
# ---------------------------------------------------------------------------


class TestLoopControllerEnvelopeUnwrap:
    """When a Step is invoked through the data-driven cascade, the
    framework delivers ``{<input_data_unit_name>: <payload>}`` to the
    Step's process(). Most Steps that consume payload-shaped input
    (CodeWriteStep, IsolatedPyExecStep) explicitly unwrap. LoopController
    did not — it passed the wrapper through under ``payload``, breaking
    any back-edge cycle that downstream consumed payload-as-envelope."""

    @pytest.mark.asyncio
    async def test_direct_process_call_passes_envelope_through(self):
        """Direct ``.process(envelope)`` (test-time) still works —
        not wrapped, not detected as envelope, used as-is."""
        lc = _build_loop_controller(max_iterations=2)
        # Need step_input_data_units to be empty so the heuristic
        # doesn't fire. Default LoopController instances don't have
        # ``step_input_data_units`` populated unless built via
        # full Workflow init — verify empty dict default.
        out = await lc.process({"some_key": "some_value"})
        # The default passthrough_key is "payload"; the controller
        # echoes the input under that key.
        assert out["payload"] == {"some_key": "some_value"}

    @pytest.mark.asyncio
    async def test_unwrap_fires_when_single_key_matches_input_unit_name(self):
        """Simulate the trigger envelope: input is
        ``{<input_unit_name>: <real payload>}`` AND the controller's
        step_input_data_units contains <input_unit_name>. Should unwrap
        so payload echo carries the real payload, not the wrapper."""
        lc = _build_loop_controller(max_iterations=2)
        # Inject a step_input_data_units mapping matching the envelope key.
        lc.step_input_data_units = {"loop_gate_input": _FakeDataUnit()}
        wrapped = {"loop_gate_input": {"real": "data", "iteration": 1}}
        out = await lc.process(wrapped)
        # If unwrap fired, payload contains the inner dict.
        # If it didn't, payload contains the wrapper.
        assert out["payload"] == {"real": "data", "iteration": 1}, (
            "LoopController did not unwrap its trigger envelope. "
            "Downstream Steps would receive doubly-wrapped data."
        )

    @pytest.mark.asyncio
    async def test_no_unwrap_when_key_does_not_match(self):
        """If the single-key wrapper key is NOT a known input unit
        name, leave the input alone — the controller can't tell
        whether it's an envelope or a 1-key user payload."""
        lc = _build_loop_controller(max_iterations=2)
        lc.step_input_data_units = {"different_name": _FakeDataUnit()}
        wrapped = {"loop_gate_input": {"real": "data"}}
        out = await lc.process(wrapped)
        # Conservative: did NOT unwrap; payload is the original input.
        assert out["payload"] == wrapped

    @pytest.mark.asyncio
    async def test_no_unwrap_when_value_is_not_dict(self):
        """Single-key wrapper but value is a scalar — not an envelope shape."""
        lc = _build_loop_controller(max_iterations=2)
        lc.step_input_data_units = {"loop_gate_input": _FakeDataUnit()}
        wrapped = {"loop_gate_input": 42}
        out = await lc.process(wrapped)
        # Conservative: did NOT unwrap.
        assert out["payload"] == wrapped

    @pytest.mark.asyncio
    async def test_iteration_counter_still_increments_after_unwrap(self):
        """Sanity: unwrap shouldn't affect the iteration counter logic."""
        lc = _build_loop_controller(max_iterations=2)
        lc.step_input_data_units = {"in": _FakeDataUnit()}
        out1 = await lc.process({"in": {"x": 1}})
        out2 = await lc.process({"in": {"x": 2}})
        out3 = await lc.process({"in": {"x": 3}})
        assert out1["allow_continue"] is True
        assert out2["allow_continue"] is True
        assert out3["allow_continue"] is False
        assert out3["loop_exhausted"] is True
