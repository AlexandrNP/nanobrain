"""step_progress event + BaseStep.emit_progress (mid-process() streaming).

Pins the additive G37 extension: a step can emit incremental progress between step_start and
step_complete, flowing through the SAME subscriber stack so a long step is not silent. See the
nanobrain streaming chain (2026-06-17).
"""

from __future__ import annotations

import asyncio
from typing import Any

from nanobrain.core.step_events import (
    _make_step_progress_event,
    subscribe_to_step_events,
)


def test_progress_event_shape_and_fraction_clamp():
    ev = _make_step_progress_event(
        step_name="s", run_id="r", message="halfway", data={"n": 5}, fraction=1.7
    )
    assert ev.event_type == "step_progress"
    assert ev.step_name == "s" and ev.run_id == "r"
    assert ev.payload["message"] == "halfway"
    assert ev.payload["data"] == {"n": 5}
    assert ev.payload["fraction"] == 1.0  # clamped to [0,1]
    # negative clamps to 0; omitted data/fraction stay absent
    assert _make_step_progress_event(step_name="s", run_id=None, message="x", fraction=-3).payload[
        "fraction"
    ] == 0.0
    bare = _make_step_progress_event(step_name="s", run_id=None, message="x")
    assert "fraction" not in bare.payload and "data" not in bare.payload


def _make_emitting_step(n_emits: int):
    """A real BaseStep subclass whose process() emits N progress events then completes.
    Built via the _allow_direct_instantiation backdoor (mirrors test_g4_completion)."""
    from nanobrain.core.step import BaseStep

    class _EmitStep(BaseStep):
        async def process(self, input_data: dict[str, Any], **kwargs: Any) -> Any:
            for i in range(n_emits):
                self.emit_progress(f"step {i + 1}/{n_emits}", fraction=(i + 1) / n_emits)
            return {"ok": True}

    _EmitStep._allow_direct_instantiation = True
    try:
        step = object.__new__(_EmitStep)
    finally:
        _EmitStep._allow_direct_instantiation = False
    step.name = "emit_step"
    step._step_config_object = None
    step.config = None
    return step


def test_emit_progress_reaches_subscriber_interleaved_before_complete():
    events: list = []
    step = _make_emitting_step(3)
    with subscribe_to_step_events(events.append):
        asyncio.run(step._execute_process({"seed": "x"}))

    types = [e.event_type for e in events]
    assert types == ["step_start", "step_progress", "step_progress", "step_progress", "step_complete"]
    progs = [e for e in events if e.event_type == "step_progress"]
    assert [p.payload["message"] for p in progs] == ["step 1/3", "step 2/3", "step 3/3"]
    assert all(p.step_name == "emit_step" for p in progs)


def test_emit_progress_noop_without_subscriber():
    # No subscriber installed → emit_progress must not raise.
    step = _make_emitting_step(2)
    result = asyncio.run(step._execute_process({"seed": "x"}))
    assert result == {"ok": True}
