"""Tests for G10 — ConditionalLink + AllDataReceivedTrigger gate-to-bottom semantics.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G10``.

Coverage:
1. ConditionalLink.GATED_OFF_SENTINEL constant exists and is the documented value.
2. ConditionalLink with publish_empty (default) is a no-op when condition is False.
3. ConditionalLink with gate_to_bottom writes the sentinel when condition is False.
4. ConditionalLink with gate_to_bottom passes through real data when condition is True.
5. ConditionalLink rejects unknown gate_semantics value.
6-9. AllDataReceivedTrigger._is_satisfied — None, sentinel under both modes, real payload.
10. AllDataReceivedTrigger rejects unknown gate_semantics value.
11. End-to-end: 2 producers + 1 fan-in, one branch gated, fan-in fires with 1 key.
12. End-to-end legacy: same setup under publish_empty deadlocks.
13. End-to-end: gate_to_bottom with all branches real behaves identically.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.link import (
    ConditionalLink,
    LinkType,
)
from nanobrain.core.trigger import (
    AllDataReceivedTrigger,
    TriggerConfig,
    TriggerType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeStepUnit:
    """Test double that satisfies BOTH the step-target shape (has
    ``set_input``) used by ConditionalLink.transfer() and the data-unit
    shape (has ``set`` / ``get``) used by AllDataReceivedTrigger. A
    single instance can therefore serve as both link target and
    trigger source, which is what an end-to-end gating test needs."""

    def __init__(self, name: str):
        self.name = name
        self._value: Any = None

    async def set_input(self, value: Any) -> None:
        self._value = value

    async def set(self, value: Any) -> None:
        self._value = value

    async def get(self) -> Any:
        return self._value


def _du(name: str) -> _FakeStepUnit:
    return _FakeStepUnit(name)


# G1 predicate that returns True iff the input dict has key 'present'.
_COND_TRUE = {"op": "exists", "field": "present"}
# G1 predicate that returns False (looks for a key the test never sets).
_COND_FALSE = {"op": "exists", "field": "missing"}


def _build_conditional_link(target_unit, condition_dict, gate_semantics: str):
    """Build a ConditionalLink and inject real target unit via the property setter
    (the workflow loader's role in production; we substitute it for unit tests)."""
    link = ConditionalLink.from_config({
        "link_type": LinkType.CONDITIONAL,
        "source": "src_placeholder",
        "target": "tgt_placeholder",
        "condition": condition_dict,
        "gate_semantics": gate_semantics,
    })
    link.target = target_unit
    return link


def _build_alldata_trigger(units, gate_semantics: str = "publish_empty",
                           name: str = "tg") -> AllDataReceivedTrigger:
    cfg_dict = {
        "trigger_type": TriggerType.ALL_DATA_RECEIVED,
        "name": name,
        "gate_semantics": gate_semantics,
    }
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(**cfg_dict)
    finally:
        TriggerConfig._allow_direct_instantiation = False
    return AllDataReceivedTrigger.from_config(cfg, data_units=units)


# ---------------------------------------------------------------------------
# 1. Sentinel constant
# ---------------------------------------------------------------------------

class TestSentinelConstant:

    def test_sentinel_value(self):
        assert ConditionalLink.GATED_OFF_SENTINEL == "__nanobrain_gated_off__"

    def test_sentinel_is_distinct_from_none(self):
        assert ConditionalLink.GATED_OFF_SENTINEL is not None


# ---------------------------------------------------------------------------
# 2-4. ConditionalLink behavior
# ---------------------------------------------------------------------------

class TestConditionalLinkBehavior:

    def test_publish_empty_legacy_noop(self):
        async def run():
            tgt = _du("tgt")
            assert await tgt.get() is None
            link = _build_conditional_link(tgt, _COND_FALSE, "publish_empty")
            await link.start()
            await link.transfer({"present": 1})  # condition still False (path: missing)
            assert await tgt.get() is None
        asyncio.run(run())

    def test_publish_empty_passes_through_when_true(self):
        async def run():
            tgt = _du("tgt")
            link = _build_conditional_link(tgt, _COND_TRUE, "publish_empty")
            await link.start()
            await link.transfer({"present": 1})
            assert await tgt.get() == {"present": 1}
        asyncio.run(run())

    def test_gate_to_bottom_writes_sentinel_on_false(self):
        async def run():
            tgt = _du("tgt")
            link = _build_conditional_link(tgt, _COND_FALSE, "gate_to_bottom")
            await link.start()
            await link.transfer({"present": 1})
            assert await tgt.get() == ConditionalLink.GATED_OFF_SENTINEL
        asyncio.run(run())

    def test_gate_to_bottom_passes_through_on_true(self):
        async def run():
            tgt = _du("tgt")
            link = _build_conditional_link(tgt, _COND_TRUE, "gate_to_bottom")
            await link.start()
            await link.transfer({"present": 1})
            assert await tgt.get() == {"present": 1}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. ConditionalLink validation
# ---------------------------------------------------------------------------

class TestConditionalLinkValidation:

    def test_unknown_gate_semantics_rejected(self):
        with pytest.raises(Exception) as exc_info:
            _build_conditional_link(_du("tgt"), _COND_TRUE, "explode_loudly")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "gate_semantics" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6-9. _is_satisfied predicate
# ---------------------------------------------------------------------------

class TestIsSatisfied:

    def test_none_payload_unsatisfied(self):
        tg = _build_alldata_trigger([_du("a")], gate_semantics="publish_empty")
        assert tg._is_satisfied(None) == (False, False)

    def test_sentinel_under_publish_empty_unsatisfied(self):
        tg = _build_alldata_trigger([_du("a")], gate_semantics="publish_empty")
        assert tg._is_satisfied(ConditionalLink.GATED_OFF_SENTINEL) == (False, False)

    def test_sentinel_under_gate_to_bottom_satisfied_excluded(self):
        tg = _build_alldata_trigger([_du("a")], gate_semantics="gate_to_bottom")
        assert tg._is_satisfied(ConditionalLink.GATED_OFF_SENTINEL) == (True, False)

    def test_real_payload_satisfied_included(self):
        tg = _build_alldata_trigger([_du("a")], gate_semantics="gate_to_bottom")
        assert tg._is_satisfied({"x": 1}) == (True, True)

    def test_zero_and_empty_payloads_satisfied(self):
        """Edge case: 0 and '' are not None and must count as data."""
        tg = _build_alldata_trigger([_du("a")], gate_semantics="publish_empty")
        assert tg._is_satisfied(0) == (True, True)
        assert tg._is_satisfied("") == (True, True)
        assert tg._is_satisfied([]) == (True, True)
        assert tg._is_satisfied({}) == (True, True)


# ---------------------------------------------------------------------------
# 10. AllDataReceivedTrigger validation
# ---------------------------------------------------------------------------

class TestTriggerValidation:

    def test_unknown_gate_semantics_rejected(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _build_alldata_trigger([_du("a")], gate_semantics="ghost")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "gate_semantics" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 11-13. End-to-end fan-in integration
# ---------------------------------------------------------------------------

class TestFanInIntegration:

    def test_one_branch_gated_fan_in_fires_with_one_key(self):
        """G10 happy path: ConditionalLink writes sentinel → trigger fires
        with the non-gated key only, and the sentinel is excluded."""
        async def run():
            unit_a = _du("a")
            unit_b = _du("b")
            tg = _build_alldata_trigger([unit_a, unit_b],
                                        gate_semantics="gate_to_bottom")

            fired_with: Dict[str, Any] = {}

            async def callback(payload):
                fired_with.update(payload)

            await tg.add_callback(callback)
            await tg.start_monitoring()

            link_a = _build_conditional_link(unit_a, _COND_TRUE, "gate_to_bottom")
            link_b = _build_conditional_link(unit_b, _COND_FALSE, "gate_to_bottom")
            await link_a.start()
            await link_b.start()
            await link_a.transfer({"present": "A_data"})
            await link_b.transfer({"present": "B_data"})  # gated off

            for _ in range(20):
                await asyncio.sleep(0.05)
                if fired_with:
                    break

            await tg.stop_monitoring()

            assert fired_with, "Trigger did not fire — gate_to_bottom failed"
            assert "input_0" in fired_with
            assert fired_with["input_0"] == {"present": "A_data"}
            assert "input_1" not in fired_with, \
                "Sentinel leaked into payload — user code would see magic string"
        asyncio.run(run())

    def test_legacy_publish_empty_deadlocks_on_gate(self):
        """The G10-cured failure shape: under publish_empty a False
        condition produces no write; AllDataReceivedTrigger waits forever.
        We wait briefly and confirm the trigger has NOT fired."""
        async def run():
            unit_a = _du("a")
            unit_b = _du("b")
            tg = _build_alldata_trigger([unit_a, unit_b],
                                        gate_semantics="publish_empty")

            fired: list = []

            async def callback(payload):
                fired.append(payload)

            await tg.add_callback(callback)
            await tg.start_monitoring()

            link_a = _build_conditional_link(unit_a, _COND_TRUE, "publish_empty")
            link_b = _build_conditional_link(unit_b, _COND_FALSE, "publish_empty")
            await link_a.start()
            await link_b.start()
            await link_a.transfer({"present": "A_data"})
            await link_b.transfer({"present": "B_data"})  # legacy no-op

            await asyncio.sleep(0.5)
            await tg.stop_monitoring()

            assert not fired, \
                "Legacy publish_empty fired anyway — semantics broken"
        asyncio.run(run())

    def test_gate_to_bottom_all_branches_real(self):
        """Sanity: gate_to_bottom does NOT alter behavior when every
        branch produces real data."""
        async def run():
            unit_a = _du("a")
            unit_b = _du("b")
            tg = _build_alldata_trigger([unit_a, unit_b],
                                        gate_semantics="gate_to_bottom")

            fired_with: Dict[str, Any] = {}

            async def callback(payload):
                fired_with.update(payload)

            await tg.add_callback(callback)
            await tg.start_monitoring()

            link_a = _build_conditional_link(unit_a, _COND_TRUE, "gate_to_bottom")
            link_b = _build_conditional_link(unit_b, _COND_TRUE, "gate_to_bottom")
            await link_a.start()
            await link_b.start()
            await link_a.transfer({"present": "A"})
            await link_b.transfer({"present": "B"})

            for _ in range(20):
                await asyncio.sleep(0.05)
                if fired_with:
                    break

            await tg.stop_monitoring()

            assert "input_0" in fired_with
            assert "input_1" in fired_with
        asyncio.run(run())
