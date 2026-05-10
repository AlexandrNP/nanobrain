"""G37 — pin the cascade-aware step-event hook contract.

eval_03 Round 3 G37: G4-completion records provenance to a durable
sink, but the integration's recorder needed LIVE access to step
events without tailing the JSONL sink. G37 exposes
``subscribe_to_step_events(subscriber)`` as the supported hook.

This test pins:
  1. step_start fires before process() runs
  2. step_complete fires after a successful process() with outputs +
     duration
  3. step_failed fires when process() raises, with type + message +
     traceback + duration; original exception still propagates
  4. multiple subscribers all receive the same event
  5. nested subscriptions stack — outer + inner both receive
  6. subscriber exceptions are swallowed (do NOT mask step exception)
  7. event_schema_version is 1
  8. no active subscribers = zero overhead path (publish is no-op)
  9. non-dict process() return wraps under _result in the event payload
 10. context isolation: subscriber added in one with-block does not
     leak to sibling code AFTER the block exits

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G37;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.9 (P4+c).
"""
from __future__ import annotations

import asyncio
from typing import Any, List

import pytest

from nanobrain.core.step_events import (
    EVENT_SCHEMA_VERSION,
    StepEvent,
    publish_step_event,
    subscribe_to_step_events,
)


# ---------------------------------------------------------------------------
# Helpers — mirror the test pattern used by G4-completion (concrete BaseStep
# subclass via the _allow_direct_instantiation backdoor).
# ---------------------------------------------------------------------------


def _make_toy_step(*, raise_for: str | None = None):
    from nanobrain.core.step import BaseStep

    class _ToyStep(BaseStep):
        async def process(
            self, input_data: dict, **kwargs: Any
        ) -> Any:
            if raise_for and input_data.get("seed") == raise_for:
                raise ValueError(f"toy boom on seed={raise_for!r}")
            return {"result": f"processed:{input_data.get('seed', '')}"}

    _ToyStep._allow_direct_instantiation = True
    try:
        step = object.__new__(_ToyStep)
    finally:
        _ToyStep._allow_direct_instantiation = False
    step.name = "g37_toy_step"
    step._step_config_object = None
    step.config = None
    return step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_step_start_fires_before_process_runs():
    captured: List[StepEvent] = []
    step = _make_toy_step()
    with subscribe_to_step_events(captured.append):
        asyncio.run(step._execute_process({"seed": "alpha"}))
    starts = [e for e in captured if e.event_type == "step_start"]
    assert len(starts) == 1
    e = starts[0]
    assert e.step_name == "g37_toy_step"
    assert e.payload["inputs"] == {"seed": "alpha"}
    assert e.event_schema_version == 1


def test_step_complete_fires_with_outputs_and_duration():
    captured: List[StepEvent] = []
    step = _make_toy_step()
    with subscribe_to_step_events(captured.append):
        asyncio.run(step._execute_process({"seed": "alpha"}))
    completes = [e for e in captured if e.event_type == "step_complete"]
    assert len(completes) == 1
    e = completes[0]
    assert e.step_name == "g37_toy_step"
    assert e.payload["outputs"] == {"result": "processed:alpha"}
    assert e.payload["duration_seconds"] >= 0


def test_step_failed_fires_then_exception_propagates():
    captured: List[StepEvent] = []
    step = _make_toy_step(raise_for="boom")
    with subscribe_to_step_events(captured.append):
        with pytest.raises(ValueError, match="toy boom"):
            asyncio.run(step._execute_process({"seed": "boom"}))
    failed = [e for e in captured if e.event_type == "step_failed"]
    assert len(failed) == 1
    e = failed[0]
    assert e.step_name == "g37_toy_step"
    exc = e.payload["exception"]
    assert exc["type"] == "ValueError"
    assert "toy boom" in exc["message"]
    assert "traceback" in exc
    assert e.payload["duration_seconds"] >= 0
    # And step_start was emitted before step_failed.
    types = [e.event_type for e in captured]
    assert types == ["step_start", "step_failed"]


def test_multiple_subscribers_all_receive_events():
    a: List[StepEvent] = []
    b: List[StepEvent] = []
    step = _make_toy_step()
    with subscribe_to_step_events(a.append), subscribe_to_step_events(b.append):
        asyncio.run(step._execute_process({"seed": "x"}))
    # Both subscribers see start + complete.
    assert [e.event_type for e in a] == ["step_start", "step_complete"]
    assert [e.event_type for e in b] == ["step_start", "step_complete"]


def test_subscriber_exception_swallowed_does_not_mask_step():
    """A buggy subscriber must NOT break the step it observes. Mirrors
    the G4-completion recorder-failure-non-fatal contract."""
    captured: List[StepEvent] = []

    def _raising_sub(event: StepEvent) -> None:
        raise RuntimeError("subscriber bug")

    step = _make_toy_step()
    with subscribe_to_step_events(_raising_sub), subscribe_to_step_events(
        captured.append
    ):
        result = asyncio.run(step._execute_process({"seed": "x"}))
    # The good subscriber still got events.
    assert len(captured) == 2
    # The step's return value is unaffected.
    assert result == {"result": "processed:x"}


def test_no_subscribers_is_a_noop():
    """Outside a subscribe_to_step_events block, publish_step_event
    is a fast no-op. Step still runs normally."""
    step = _make_toy_step()
    result = asyncio.run(step._execute_process({"seed": "y"}))
    assert result == {"result": "processed:y"}
    # Direct API: publishing with no subscribers does not raise.
    publish_step_event(
        StepEvent(
            event_type="step_start",
            step_name="x",
            run_id=None,
            timestamp_iso="2026-05-09T00:00:00Z",
        )
    )


def test_event_schema_version_is_one():
    """Pin the v1 schema declaration."""
    assert EVENT_SCHEMA_VERSION == 1
    captured: List[StepEvent] = []
    step = _make_toy_step()
    with subscribe_to_step_events(captured.append):
        asyncio.run(step._execute_process({"seed": "z"}))
    for e in captured:
        assert e.event_schema_version == 1


def test_non_dict_return_wrapped_under_result():
    """Steps that return non-dicts have their return wrapped under
    ``_result`` in the step_complete event payload — uniform shape
    for downstream consumers."""
    from nanobrain.core.step import BaseStep

    class _ScalarStep(BaseStep):
        async def process(self, input_data: dict, **kwargs: Any) -> Any:
            return 42

    _ScalarStep._allow_direct_instantiation = True
    try:
        step = object.__new__(_ScalarStep)
    finally:
        _ScalarStep._allow_direct_instantiation = False
    step.name = "scalar_step"
    step._step_config_object = None
    step.config = None

    captured: List[StepEvent] = []
    with subscribe_to_step_events(captured.append):
        asyncio.run(step._execute_process({}))
    complete = next(e for e in captured if e.event_type == "step_complete")
    assert complete.payload["outputs"] == {"_result": 42}


def test_subscription_does_not_leak_outside_with_block():
    """The contextvar stack must restore on exit so a subscriber
    registered in one block does not leak to sibling code that runs
    AFTER."""
    inner_received: List[StepEvent] = []
    outer_received: List[StepEvent] = []
    step = _make_toy_step()

    with subscribe_to_step_events(inner_received.append):
        asyncio.run(step._execute_process({"seed": "first"}))

    # Outside the with-block: only outer receives.
    with subscribe_to_step_events(outer_received.append):
        asyncio.run(step._execute_process({"seed": "second"}))

    assert len(inner_received) == 2  # first run only
    assert all(
        e.payload.get("inputs", {}).get("seed") in ("first", None)
        for e in inner_received
    )
    assert len(outer_received) == 2  # second run only
    assert all(
        e.payload.get("inputs", {}).get("seed") in ("second", None)
        for e in outer_received
    )


def test_event_carries_iso_timestamp():
    """Each event has a UTC-ISO-8601 timestamp_iso for cross-system
    correlation. The framework does not impose a fixed timezone-naive
    or naive form — UTC ISO is the canonical."""
    captured: List[StepEvent] = []
    step = _make_toy_step()
    with subscribe_to_step_events(captured.append):
        asyncio.run(step._execute_process({}))
    for e in captured:
        # Crude check: ISO-8601 with timezone marker.
        assert "T" in e.timestamp_iso
        assert e.timestamp_iso.endswith("+00:00") or e.timestamp_iso.endswith("Z")
