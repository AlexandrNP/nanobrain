"""G123 (2026-05-18) — default ``bind_action`` on TriggerBase.

Before G123, TimerTrigger and EventTrigger did not implement
``bind_action``. ``step.py`` unconditionally calls
``trigger_instance.bind_action(self._execute_on_trigger)`` at step
trigger initialization (see ``nanobrain/core/step.py`` line ~1238),
so authoring a YAML step that declared a ``TimerTrigger`` or
``EventTrigger`` as a step-level trigger raised
``AttributeError: 'TimerTrigger' object has no attribute 'bind_action'``
at workflow load — silent in design intent, loud in stack trace,
unhelpful in error message.

After G123 (``nanobrain/core/trigger.py``), ``TriggerBase`` carries
a default ``bind_action`` that registers the action into
``self._callbacks``. Subclasses with their own ``bound_actions``
list (DataUnitChangeTrigger / AllDataReceivedTrigger / ManualTrigger)
keep their overrides; the new default only applies to subclasses
that don't override.

This module pins:
  1. ``TimerTrigger.bind_action(action)`` does not raise and the
     action is registered + fires on each timer tick.
  2. ``EventTrigger.bind_action(action)`` does not raise and the
     action fires on ``fire_event``.
  3. ``unbind_action`` removes the action and prevents subsequent
     fires.
  4. The existing ``bind_action`` overrides on DataUnitChangeTrigger /
     AllDataReceivedTrigger / ManualTrigger still work (regression
     guard — confirms G123 didn't shadow them).
"""

from __future__ import annotations

import asyncio

import pytest
from nanobrain.core.trigger import (
    AllDataReceivedTrigger,
    DataUnitChangeTrigger,
    EventTrigger,
    ManualTrigger,
    TimerTrigger,
)


def _make_timer(interval_ms: int = 50) -> TimerTrigger:
    cfg = {
        "name": "g123_timer",
        "trigger_type": "timer",
        "timer_interval_ms": interval_ms,
        # The default debounce_ms is 100 (rate-limit safeguard); set to 0
        # so the test's tight interval actually fires.
        "debounce_ms": 0,
        "max_frequency_hz": 1000.0,
    }
    return TimerTrigger.from_config(cfg)


def _make_event() -> EventTrigger:
    cfg = {
        "name": "g123_event",
        "trigger_type": "event",
        "debounce_ms": 0,
        "max_frequency_hz": 1000.0,
    }
    return EventTrigger.from_config(cfg)


@pytest.mark.asyncio
async def test_timer_trigger_bind_action_default_routes_through_callbacks():
    """G123 pin: TimerTrigger inherits a working ``bind_action`` that
    routes through ``_callbacks``. Without the default, this raises
    AttributeError at the ``bind_action`` call."""
    trigger = _make_timer(interval_ms=30)
    fires: list[object] = []

    async def action(data):
        fires.append(data)

    trigger.bind_action(action)
    assert action in trigger._callbacks, (
        "G123 regression — bind_action did not register into _callbacks"
    )

    await trigger.start_monitoring()
    await asyncio.sleep(0.12)
    await trigger.stop_monitoring()

    assert len(fires) >= 2, (
        f"TimerTrigger fired bound action only {len(fires)} times in 120ms "
        f"with 30ms interval — bind_action is not wired through the fire path"
    )


@pytest.mark.asyncio
async def test_event_trigger_bind_action_default_routes_through_callbacks():
    """G123 pin: EventTrigger inherits a working ``bind_action`` and
    the bound action fires once per ``fire_event`` call."""
    trigger = _make_event()
    fires: list[object] = []

    async def action(data):
        fires.append(data)

    trigger.bind_action(action)
    await trigger.start_monitoring()

    fired = await trigger.fire_event({"kind": "novel"})
    assert fired is True
    # Yield to the loop so the rate limiter's per-tick min_interval window
    # passes (max_frequency_hz=1000 → min_interval=1ms; two fires in the
    # same tick would trip the limiter).
    await asyncio.sleep(0.005)
    fired = await trigger.fire_event({"kind": "stale"})
    assert fired is True

    await trigger.stop_monitoring()
    assert fires == [{"kind": "novel"}, {"kind": "stale"}], (
        f"EventTrigger bound action did not receive both events; got {fires!r}"
    )


@pytest.mark.asyncio
async def test_unbind_action_default_removes_from_callbacks():
    """G123 pin: ``unbind_action`` is symmetric — after unbind, the
    action does NOT fire on subsequent trigger fires."""
    trigger = _make_event()
    fires: list[object] = []

    async def action(data):
        fires.append(data)

    trigger.bind_action(action)
    trigger.unbind_action(action)
    assert action not in trigger._callbacks, (
        "G123 regression — unbind_action left the action in _callbacks"
    )

    await trigger.start_monitoring()
    await trigger.fire_event({"kind": "after_unbind"})
    await trigger.stop_monitoring()

    assert fires == [], (
        f"Unbound action still fired — {fires!r}; unbind_action is broken"
    )


@pytest.mark.asyncio
async def test_data_unit_change_trigger_override_still_works():
    """Regression: DataUnitChangeTrigger ships its own ``bind_action``
    that maintains a separate ``bound_actions`` list (independent from
    ``_callbacks``). G123's TriggerBase default must NOT shadow that
    override."""
    from nanobrain.core.data_unit import DataUnitMemory

    data_unit = DataUnitMemory.from_config(
        {"class": "nanobrain.core.data_unit.DataUnitMemory", "name": "g123_du"}
    )

    trigger = DataUnitChangeTrigger.from_config(
        {
            "name": "g123_du_change",
            "trigger_type": "data_updated",
            "data_unit": data_unit,
        }
    )

    async def action(event):
        pass

    trigger.bind_action(action)
    # The subclass-specific list must carry the action.
    assert action in trigger.bound_actions, (
        "DataUnitChangeTrigger.bind_action override no longer registers "
        "into bound_actions — G123 may have shadowed the override"
    )


@pytest.mark.asyncio
async def test_all_data_received_trigger_override_still_works():
    """Regression: AllDataReceivedTrigger's ``bind_action`` (G118)
    registers into BOTH ``bound_actions`` and ``_callbacks``. G123's
    TriggerBase default must NOT shadow this override."""
    from nanobrain.core.data_unit import DataUnitMemory

    du_a = DataUnitMemory.from_config(
        {"class": "nanobrain.core.data_unit.DataUnitMemory", "name": "g123_a"}
    )
    du_b = DataUnitMemory.from_config(
        {"class": "nanobrain.core.data_unit.DataUnitMemory", "name": "g123_b"}
    )

    trigger = AllDataReceivedTrigger.from_config(
        {
            "name": "g123_all_received",
            "trigger_type": "all_data_received",
            "data_units": [du_a, du_b],
        }
    )

    async def action(payload):
        pass

    trigger.bind_action(action)
    assert action in trigger.bound_actions, (
        "AllDataReceivedTrigger.bind_action no longer registers into "
        "bound_actions — G118 override may have been shadowed by G123"
    )
    assert action in trigger._callbacks, (
        "AllDataReceivedTrigger.bind_action no longer registers into "
        "_callbacks — G118 dual-list registration broke"
    )


@pytest.mark.asyncio
async def test_manual_trigger_override_still_works():
    """Regression: ManualTrigger's ``bind_action`` (legacy) registers
    into ``bound_actions`` only. G123's TriggerBase default must NOT
    shadow this override."""
    trigger = ManualTrigger.from_config(
        {"name": "g123_manual", "trigger_type": "manual"}
    )

    async def action(data):
        pass

    trigger.bind_action(action)
    assert action in trigger.bound_actions, (
        "ManualTrigger.bind_action override no longer registers into "
        "bound_actions — G123 may have shadowed the override"
    )


def test_bind_action_is_synchronous_on_all_trigger_classes():
    """G123 contract: ``bind_action`` is synchronous on EVERY trigger
    subclass (so ``step.py``'s sync call site at line ~1238 works
    uniformly). Compare to ``add_callback`` which is async on
    TriggerBase. This pins the sync-vs-async asymmetry that motivated
    a separate ``bind_action`` method in the first place."""
    import inspect

    for cls in [
        TimerTrigger,
        EventTrigger,
        ManualTrigger,
        DataUnitChangeTrigger,
        AllDataReceivedTrigger,
    ]:
        method = cls.bind_action
        assert not inspect.iscoroutinefunction(method), (
            f"{cls.__name__}.bind_action is async; step.py calls it "
            f"synchronously — must remain sync"
        )
