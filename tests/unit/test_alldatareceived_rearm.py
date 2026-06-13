"""Regression: AllDataReceivedTrigger is RE-ARMABLE (2026-06-13).

The event-driven fan-in trigger used to fire ONCE then deactivate + unregister,
so a cached/re-runnable workflow's 2nd+ run returned STALE run-1 output (the
fan-in step never re-fired). It is now **value-comparison re-armable**: it fires
the first time every input is present, and then re-fires whenever the input
value-tuple DIFFERS from the last fire (i.e. ANY input changed). "Any changed" —
not "all changed" — is load-bearing: a same-query re-run that only changes a
control input (e.g. adds an approval token) while the evidence input is unchanged
must still re-fire. A byte-identical re-run does NOT re-fire — the prior output is
already the correct deterministic answer for those exact inputs.

This replaces the earlier ``_armed`` + ``reset_for_run()`` mechanism (which forced
the Workflow to clear every fan-in's inputs before each run); the trigger now
self-determines re-fire from its own observable state, with no Workflow→Trigger
reset hook.
"""

from __future__ import annotations

import asyncio

from nanobrain.core.data_unit import DataUnitMemory
from nanobrain.core.trigger import AllDataReceivedTrigger


def _du(name: str) -> DataUnitMemory:
    return DataUnitMemory.from_config(
        {"class": "nanobrain.core.data_unit.DataUnitMemory", "name": name}
    )


async def _make(n: int = 2):
    dus = [_du(f"du_{i}") for i in range(n)]
    for d in dus:
        await d.initialize()
    fired: list = []

    async def _cb(data):
        fired.append(data)

    trig = AllDataReceivedTrigger.from_config(
        {"trigger_type": "all_data_received", "name": "t"}, data_units=dus
    )
    trig.bind_action(_cb)
    return dus, fired, trig


def test_fires_once_when_all_inputs_present():
    async def go():
        dus, fired, trig = await _make()
        await trig.start_monitoring()
        await dus[0].set({"v": 1})
        await asyncio.sleep(0.25)
        assert len(fired) == 0, "must NOT fire with only one input"
        await dus[1].set({"v": 2})
        await asyncio.sleep(0.4)
        assert len(fired) == 1
        assert fired[0] == {"input_0": {"v": 1}, "input_1": {"v": 2}}

    asyncio.run(go())


def test_re_fires_when_any_input_changes():
    """Cached-workflow re-run: a second cycle with a CHANGED input re-fires
    (no reset_for_run() needed — value comparison drives the re-fire)."""

    async def go():
        dus, fired, trig = await _make()
        await trig.start_monitoring()
        await dus[0].set({"v": 1})
        await dus[1].set({"v": 2})
        await asyncio.sleep(0.4)
        assert len(fired) == 1
        # Change both inputs (a fresh "run") → re-fires.
        await dus[0].set({"v": 3})
        await dus[1].set({"v": 4})
        await asyncio.sleep(0.4)
        assert len(fired) == 2
        assert fired[1] == {"input_0": {"v": 3}, "input_1": {"v": 4}}

    asyncio.run(go())


def test_re_fires_when_only_one_input_changes():
    """The load-bearing design-gate case: the evidence input (review) is UNCHANGED
    but the control input gains an approval token → the value-tuple still DIFFERS,
    so the gate MUST re-fire. 'Any changed', not 'all changed'."""

    async def go():
        dus, fired, trig = await _make()
        review, control = dus
        await trig.start_monitoring()
        await review.set({"synthesis": "same"})
        await control.set({"query": "chikv"})
        await asyncio.sleep(0.4)
        assert len(fired) == 1
        # Only control changes (adds approval); review stays identical.
        await control.set({"query": "chikv", "design_approval_id": "appr-1"})
        await asyncio.sleep(0.4)
        assert len(fired) == 2, "must re-fire when only the control input changes"
        assert fired[1]["input_1"] == {"query": "chikv", "design_approval_id": "appr-1"}
        assert fired[1]["input_0"] == {"synthesis": "same"}

    asyncio.run(go())


def test_does_not_refire_on_identical_values():
    """A byte-identical re-run does NOT re-fire — the prior output is already the
    correct deterministic answer for those exact inputs. (Distinct from the
    one-input-changed case above, which DOES re-fire.)"""

    async def go():
        dus, fired, trig = await _make()
        await trig.start_monitoring()
        await dus[0].set({"v": 1})
        await dus[1].set({"v": 2})
        await asyncio.sleep(0.4)
        assert len(fired) == 1
        # Re-set the SAME values — value-tuple unchanged → no re-fire.
        await dus[0].set({"v": 1})
        await dus[1].set({"v": 2})
        await asyncio.sleep(0.4)
        assert len(fired) == 1, "identical re-run must NOT re-fire"

    asyncio.run(go())


def test_early_arrival_then_late_fires_once():
    """The design-gate scenario: one input set BEFORE monitoring, the other after."""

    async def go():
        dus, fired, trig = await _make()
        await dus[0].set({"v": "early"})
        await trig.start_monitoring()
        await asyncio.sleep(0.25)
        assert len(fired) == 0
        await dus[1].set({"v": "late"})
        await asyncio.sleep(0.4)
        assert len(fired) == 1

    asyncio.run(go())


def test_prepopulated_fires_immediately():
    async def go():
        dus, fired, trig = await _make()
        await dus[0].set({"v": 1})
        await dus[1].set({"v": 2})
        await trig.start_monitoring()
        await asyncio.sleep(0.4)
        assert len(fired) == 1

    asyncio.run(go())
