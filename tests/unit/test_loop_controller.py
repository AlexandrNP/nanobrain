"""Tests for G18 — LoopController bounded-cycle iteration counter.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G18``: a
BaseStep subclass that owns the iteration counter for a back-edge through
the workflow DAG. Without an iteration cap, a back-edge becomes an
infinite loop. This commit ships the runtime primitive (the counter +
output shape); the workflow integrity validator extension allowing
declared back-edges through this step is a separate task.

Tests cover:
1. LoopControllerConfig validation (positive max_iterations etc.)
2. Output shape on under-cap iterations (allow_continue=True)
3. Output shape on cap-reached iterations (allow_continue=False)
4. Iteration counter behavior across many calls
5. Reset semantics
6. Custom payload_passthrough_key
7. Realistic agent-repair-loop scenario from agent_workflow_authoring.md §7
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.library.steps import LoopController, LoopControllerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# Per the canonical nanobrain test pattern (see tests/unit/test_approval_step.py
# line 94+211), step instances are constructed by writing a tmp YAML file
# and loading it through from_config. BaseStep.from_config requires a file
# path — direct dict / Pydantic-object construction raises a clear error.
# We use NamedTemporaryFile so each call creates an isolated config file.


def _build_controller(**cfg_kwargs) -> LoopController:
    """Build a LoopController via tmp YAML. Defaults: name='ctrl',
    max_iterations=2. Test owns the tmp file lifetime — pytest's
    tmp-file cleanup handles teardown."""
    cfg_kwargs.setdefault("name", "ctrl")
    cfg_kwargs.setdefault("max_iterations", 2)
    yaml_text = yaml.safe_dump(cfg_kwargs)
    # delete=False so the file persists after close(); pytest doesn't
    # need to clean up because /tmp is rotated by the OS.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        f.write(yaml_text)
        path = f.name
    return LoopController.from_config(path)


# ---------------------------------------------------------------------------
# 1. LoopControllerConfig validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def _build_cfg(self, **kwargs):
        kwargs.setdefault("name", "x")
        LoopControllerConfig._allow_direct_instantiation = True
        try:
            return LoopControllerConfig(**kwargs)
        finally:
            LoopControllerConfig._allow_direct_instantiation = False

    def test_minimal(self):
        cfg = self._build_cfg(max_iterations=3)
        assert cfg.max_iterations == 3
        assert cfg.initial_count == 0
        assert cfg.payload_passthrough_key == "payload"

    def test_max_iterations_must_be_positive(self):
        with pytest.raises(Exception):
            self._build_cfg(max_iterations=0)

    def test_max_iterations_string_int_coerces(self):
        """Pydantic coerces 'numeric strings' to int. We accept this as
        ergonomic — operators editing YAML may accidentally quote a number;
        the cap behavior is unchanged. Strict-mode rejection would require
        opt-in via model_config['strict']=True; we don't need it here."""
        cfg = self._build_cfg(max_iterations="3")
        assert cfg.max_iterations == 3

    def test_max_iterations_non_numeric_string_rejected(self):
        """A genuinely-invalid value (non-numeric string) IS rejected."""
        with pytest.raises(Exception):
            self._build_cfg(max_iterations="not-a-number")

    def test_initial_count_negative_rejected(self):
        with pytest.raises(Exception):
            self._build_cfg(max_iterations=3, initial_count=-1)


# ---------------------------------------------------------------------------
# 2-3. Output shape — under cap and at cap
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_first_iteration_allows_continue(self):
        async def run():
            ctrl = _build_controller(max_iterations=2)
            out = await ctrl.process({"plan": "v1"})
            assert out["allow_continue"] is True
            assert out["loop_exhausted"] is False
            assert out["iteration"] == 0  # the count BEFORE the gate decision
            assert out["max_iterations"] == 2
            assert out["payload"] == {"plan": "v1"}
        asyncio.run(run())

    def test_second_iteration_under_cap(self):
        async def run():
            ctrl = _build_controller(max_iterations=2)
            await ctrl.process({"plan": "v1"})
            out = await ctrl.process({"plan": "v2"})
            assert out["allow_continue"] is True
            assert out["iteration"] == 1
            assert out["payload"] == {"plan": "v2"}
        asyncio.run(run())

    def test_third_iteration_exhausted(self):
        async def run():
            ctrl = _build_controller(max_iterations=2)
            await ctrl.process({"plan": "v1"})
            await ctrl.process({"plan": "v2"})
            out = await ctrl.process({"plan": "v3"})
            assert out["allow_continue"] is False
            assert out["loop_exhausted"] is True
            assert out["iteration"] == 2  # the cap
            # The payload IS preserved even when exhausted, so escalation
            # path can include the last attempted plan in its message:
            assert out["payload"] == {"plan": "v3"}
        asyncio.run(run())

    def test_repeated_calls_after_exhaustion_idempotent(self):
        """The counter does not advance past the cap on repeated calls."""
        async def run():
            ctrl = _build_controller(max_iterations=2)
            await ctrl.process({})
            await ctrl.process({})
            out_a = await ctrl.process({"a": 1})
            out_b = await ctrl.process({"b": 2})
            assert out_a["iteration"] == 2
            assert out_b["iteration"] == 2  # NOT 3 — counter stops at cap
            assert ctrl.iteration_count == 2
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. Iteration counter
# ---------------------------------------------------------------------------

class TestIterationCounter:

    def test_counter_advances_with_each_call(self):
        async def run():
            ctrl = _build_controller(max_iterations=5)
            assert ctrl.iteration_count == 0
            await ctrl.process({})
            assert ctrl.iteration_count == 1
            await ctrl.process({})
            assert ctrl.iteration_count == 2
            await ctrl.process({})
            assert ctrl.iteration_count == 3
        asyncio.run(run())

    def test_initial_count_seeds_counter(self):
        """Allows resuming a checkpointed loop with a non-zero starting point."""
        async def run():
            ctrl = _build_controller(max_iterations=3, initial_count=2)
            assert ctrl.iteration_count == 2
            # Already at 2, one more call brings it to 3 (= cap).
            out = await ctrl.process({})
            assert out["allow_continue"] is True
            assert ctrl.iteration_count == 3
            # Next call exhausts.
            out = await ctrl.process({})
            assert out["allow_continue"] is False
        asyncio.run(run())

    def test_is_exhausted_property(self):
        async def run():
            ctrl = _build_controller(max_iterations=2)
            assert ctrl.is_exhausted is False
            await ctrl.process({})
            assert ctrl.is_exhausted is False
            await ctrl.process({})
            assert ctrl.is_exhausted is True
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. reset()
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_after_exhaustion(self):
        async def run():
            ctrl = _build_controller(max_iterations=2)
            await ctrl.process({})
            await ctrl.process({})
            assert ctrl.is_exhausted
            ctrl.reset()
            assert ctrl.iteration_count == 0
            assert ctrl.is_exhausted is False
            # Subsequent process() works as fresh:
            out = await ctrl.process({"v": 1})
            assert out["allow_continue"] is True
        asyncio.run(run())

    def test_reset_to_initial_count(self):
        """reset() returns to initial_count, not 0."""
        async def run():
            ctrl = _build_controller(max_iterations=5, initial_count=2)
            await ctrl.process({})  # → count 3
            await ctrl.process({})  # → count 4
            ctrl.reset()
            assert ctrl.iteration_count == 2
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Custom payload_passthrough_key
# ---------------------------------------------------------------------------

class TestPayloadPassthrough:

    def test_default_key_is_payload(self):
        async def run():
            ctrl = _build_controller(max_iterations=2)
            out = await ctrl.process({"x": 1})
            assert "payload" in out
            assert out["payload"] == {"x": 1}
        asyncio.run(run())

    def test_custom_key(self):
        async def run():
            ctrl = _build_controller(
                max_iterations=2,
                payload_passthrough_key="execution_plan",
            )
            out = await ctrl.process({"intent": "investigate"})
            assert "execution_plan" in out
            assert out["execution_plan"] == {"intent": "investigate"}
            # Default key is NOT also present:
            assert "payload" not in out
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. Realistic agent-repair-loop scenario
# ---------------------------------------------------------------------------

class TestRepairLoopScenario:
    """Per agent_workflow_authoring.md §7: two-attempt cap on repair.
    First failure → repair attempt 1. Second failure → repair attempt 2.
    Third failure → escalation."""

    def test_two_repair_attempts_then_escalate(self):
        async def run():
            ctrl = _build_controller(
                max_iterations=2,
                name="repair_gate",
                payload_passthrough_key="execution_plan",
            )

            # Initial plan rejected → first repair attempt:
            out1 = await ctrl.process({"plan_hash": "v1"})
            assert out1["allow_continue"] is True
            assert out1["iteration"] == 0  # first repair attempt slot

            # Repair v2 rejected → second repair attempt:
            out2 = await ctrl.process({"plan_hash": "v2"})
            assert out2["allow_continue"] is True
            assert out2["iteration"] == 1

            # Repair v3 rejected → ESCALATE (cap reached):
            out3 = await ctrl.process({"plan_hash": "v3"})
            assert out3["allow_continue"] is False
            assert out3["loop_exhausted"] is True
            assert out3["execution_plan"] == {"plan_hash": "v3"}
            assert out3["max_iterations"] == 2
        asyncio.run(run())
