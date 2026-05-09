"""Tests for G22 — EventTrigger (the event-source half).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G22``.

Coverage:
1. EventTrigger constructs via from_config (dict, TriggerConfig, str path).
2. fire_event without a filter fires every time, returns True.
3. fire_event with a G1 predicate filter fires only on match.
4. fire_event on inactive trigger returns False, no callbacks.
5. fire_event passes the event body to callbacks unchanged.
6. EventTrigger TriggerType is registered.
7. event_filter rejects malformed predicate dicts (FAIL-FAST).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.trigger import (
    EventTrigger,
    TriggerConfig,
    TriggerType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_event_trigger(
    *,
    event_filter=None,
    debounce_ms=0,
    max_frequency_hz=1000.0,
    name="ev",
):
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(
            name=name,
            trigger_type=TriggerType.EVENT,
            debounce_ms=debounce_ms,
            max_frequency_hz=max_frequency_hz,
            event_filter=event_filter,
        )
    finally:
        TriggerConfig._allow_direct_instantiation = False
    return EventTrigger.from_config(cfg)


# ---------------------------------------------------------------------------
# 1. Construction paths
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_from_config_via_trigger_config(self):
        t = _build_event_trigger()
        assert t.name == "ev"
        assert t.config.trigger_type == TriggerType.EVENT

    def test_from_config_via_dict(self):
        t = EventTrigger.from_config({
            "name": "evd",
            "trigger_type": TriggerType.EVENT,
            "debounce_ms": 0,
        })
        assert t.name == "evd"

    def test_from_config_via_yaml_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            yml = Path(tmp) / "ev.yml"
            yml.write_text(yaml.safe_dump({
                "name": "evp",
                "trigger_type": "event",
                "debounce_ms": 0,
            }))
            t = EventTrigger.from_config(str(yml))
            assert t.name == "evp"


# ---------------------------------------------------------------------------
# 2-5. fire_event behavior
# ---------------------------------------------------------------------------

class TestFireEvent:

    def test_no_filter_always_fires(self):
        async def run():
            t = _build_event_trigger()
            fired = []
            async def cb(payload):
                fired.append(payload)
            await t.add_callback(cb)
            await t.start_monitoring()
            r = await t.fire_event({"a": 1})
            assert r is True
            assert fired == [{"a": 1}]
        asyncio.run(run())

    def test_filter_passes_match(self):
        async def run():
            t = _build_event_trigger(
                event_filter={"op": "eq", "field": "kind", "value": "novel"}
            )
            fired = []
            async def cb(payload):
                fired.append(payload)
            await t.add_callback(cb)
            await t.start_monitoring()
            r = await t.fire_event({"kind": "novel", "id": 1})
            assert r is True
            assert fired == [{"kind": "novel", "id": 1}]
        asyncio.run(run())

    def test_filter_blocks_miss(self):
        async def run():
            t = _build_event_trigger(
                event_filter={"op": "eq", "field": "kind", "value": "novel"}
            )
            fired = []
            async def cb(payload):
                fired.append(payload)
            await t.add_callback(cb)
            await t.start_monitoring()
            r = await t.fire_event({"kind": "boring", "id": 2})
            assert r is False
            assert fired == []
        asyncio.run(run())

    def test_inactive_trigger_returns_false(self):
        async def run():
            t = _build_event_trigger()
            fired = []
            async def cb(payload):
                fired.append(payload)
            await t.add_callback(cb)
            # NOT started
            r = await t.fire_event({"x": 1})
            assert r is False
            assert fired == []
        asyncio.run(run())

    def test_event_body_passed_through(self):
        async def run():
            t = _build_event_trigger()
            seen = []
            async def cb(payload):
                seen.append(payload)
            await t.add_callback(cb)
            await t.start_monitoring()
            await t.fire_event({"a": 1, "b": [2, 3]})
            # Sleep beyond the 1/max_frequency_hz min_interval (1ms at 1000Hz)
            # so each fire is not rate-limited.
            await asyncio.sleep(0.005)
            await t.fire_event("a string")
            await asyncio.sleep(0.005)
            await t.fire_event(42)
            assert seen == [{"a": 1, "b": [2, 3]}, "a string", 42]
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. TriggerType
# ---------------------------------------------------------------------------

class TestTriggerType:

    def test_event_value_registered(self):
        assert TriggerType.EVENT.value == "event"


# ---------------------------------------------------------------------------
# 7. Filter validation
# ---------------------------------------------------------------------------

class TestFilterValidation:

    def test_malformed_predicate_rejected_at_init(self):
        """A G1 predicate dict with an unknown 'op' fails at trigger
        construction time (FAIL-FAST), not at fire_event."""
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _build_event_trigger(
                event_filter={"op": "nonexistent_op", "field": "x", "value": 1}
            )
        assert "FAIL-FAST" in str(exc_info.value)
