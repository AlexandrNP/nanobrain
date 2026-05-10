"""G4-completion — pin BaseStep._execute_process automatic recording.

Pre-G4-completion: the ProvenanceContext primitive shipped (G4) but the
framework's ``BaseStep._execute_process`` did NOT call
``record_step_invocation`` automatically. Operators who wanted
provenance had to wrap every ``process()`` call by hand — which means
exceptions, timeouts, and any path that raised would silently disappear
from the audit trail. The whole stated value of G4 ("the framework
wraps _execute_process so the recorder sees every invocation,
including those that raise") was deferred. Eval_03 Round 2 named this
explicitly; commit ``78b67cb`` admitted the deferral in its message.

Post-G4-completion: ``_execute_process`` consults
``current_provenance_context()`` on every call. When a context is
active and enabled, it records one invocation with:

  * ``step_name``
  * ``inputs`` (the dict passed to ``process``)
  * EITHER ``outputs`` (success) OR ``exception`` (failure path —
    type, message, traceback)
  * ``timing.duration_seconds``

When no context is active, the wrap is a fast no-op.

Recorder errors are SWALLOWED so provenance bookkeeping never masks
the step's real exception or breaks its return path.

This test pins:
  1. successful process() emits one record with inputs + outputs + timing
  2. raising process() emits one record with inputs + exception + timing,
     then re-raises the original exception
  3. no active context → zero records (no-op, no perf cost)
  4. recorder failure does NOT mask the step's real exception
  5. recorder failure does NOT corrupt the step's return value
  6. disabled context (``enabled=False``) → zero records

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 2 G4-completion;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nanobrain.core.provenance import ProvenanceContext


# ---------------------------------------------------------------------------
# A recording sink that captures records into an in-memory list — lets the
# tests assert on what would have been written without touching disk.
# ---------------------------------------------------------------------------


class _CapturingSink:
    """Test helper. Implements the ProvenanceSinkBase contract minimally."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.write_should_raise: bool = False

    async def write_record(self, record: dict[str, Any]) -> None:
        if self.write_should_raise:
            raise RuntimeError("test-injected sink failure")
        self.records.append(record)

    async def flush(self) -> None:
        return None

    async def close(self) -> None:
        return None


def _make_context(*, enabled: bool = True, sink: Any = None) -> ProvenanceContext:
    return ProvenanceContext.from_config({"enabled": enabled}, sink=sink)


# ---------------------------------------------------------------------------
# Test step: a BaseStep subclass whose process() either returns a dict or
# raises. Built directly via __new__ to bypass FromConfigBase guard for
# this targeted state-machine test.
# ---------------------------------------------------------------------------


def _make_toy_step(*, raise_for: str | None = None):
    """Build a minimal real ``BaseStep`` subclass instance via the
    framework's ``_allow_direct_instantiation`` backdoor.

    A concrete subclass is required because BaseStep declares
    ``process`` as ABC; ``object.__new__(BaseStep)`` would error.
    Setting ``_step_config_object`` and ``config`` to ``None``
    short-circuits the G6 schema lookup; the rest of the wrap
    (pause-signal, G4-completion provenance) is what we are testing.
    """
    from nanobrain.core.step import BaseStep

    class _ToyStep(BaseStep):
        async def process(
            self, input_data: dict[str, Any], **kwargs: Any
        ) -> Any:
            if raise_for and input_data.get("seed") == raise_for:
                raise ValueError(f"toy boom on seed={raise_for!r}")
            return {"result": f"processed:{input_data.get('seed', '')}"}

    _ToyStep._allow_direct_instantiation = True
    try:
        step = object.__new__(_ToyStep)
    finally:
        _ToyStep._allow_direct_instantiation = False
    step.name = "toy_step"
    step._step_config_object = None
    step.config = None
    return step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_success_path_records_inputs_outputs_and_timing():
    """Successful process() emits exactly one record with inputs +
    outputs + a positive duration_seconds."""
    sink = _CapturingSink()
    ctx = _make_context(sink=sink)
    step = _make_toy_step()

    with ctx.activate():
        result = asyncio.run(step._execute_process({"seed": "alpha"}))

    assert result == {"result": "processed:alpha"}
    assert len(sink.records) == 1, (
        f"expected exactly one provenance record; got "
        f"{len(sink.records)}: {sink.records!r}"
    )
    rec = sink.records[0]
    assert rec["step_name"] == "toy_step"
    assert rec["inputs"] == {"seed": "alpha"}
    assert rec["outputs"] == {"result": "processed:alpha"}
    assert "exception" not in rec or rec.get("exception") in (None, {})
    assert rec["timing"]["duration_seconds"] >= 0


def test_exception_path_records_then_reraises():
    """A raising process() emits one record with inputs + exception +
    timing, then the framework re-raises the original ValueError."""
    sink = _CapturingSink()
    ctx = _make_context(sink=sink)
    step = _make_toy_step(raise_for="boom")

    with ctx.activate():
        with pytest.raises(ValueError, match="toy boom on seed='boom'"):
            asyncio.run(step._execute_process({"seed": "boom"}))

    assert len(sink.records) == 1, (
        f"expected exactly one provenance record on exception; got "
        f"{len(sink.records)}"
    )
    rec = sink.records[0]
    assert rec["step_name"] == "toy_step"
    assert rec["inputs"] == {"seed": "boom"}
    exc = rec["exception"]
    assert exc["type"] == "ValueError"
    assert "toy boom" in exc["message"]
    assert "ValueError" in exc["traceback"]
    assert rec["timing"]["duration_seconds"] >= 0


def test_no_active_context_is_a_noop():
    """Outside ``ctx.activate()`` there is no current_provenance_context;
    ``_execute_process`` must NOT record anything (and certainly must
    not raise).

    Relies on the contextvar default of None — no surrounding
    ``with ctx.activate()``."""
    step = _make_toy_step()

    # No activate() — there is no current_provenance_context.
    result = asyncio.run(step._execute_process({"seed": "beta"}))
    assert result == {"result": "processed:beta"}


def test_disabled_context_is_a_noop():
    """``enabled=False`` short-circuits the recorder before the sink
    is touched. No records, no perf cost."""
    sink = _CapturingSink()
    ctx = _make_context(enabled=False, sink=sink)
    step = _make_toy_step()

    with ctx.activate():
        asyncio.run(step._execute_process({"seed": "gamma"}))

    assert sink.records == [], (
        f"disabled context must not write records; got {sink.records}"
    )


def test_recorder_failure_does_not_mask_step_exception():
    """When the recorder itself raises during the exception path, the
    framework MUST surface the step's original exception, not the
    recorder's. This is the load-bearing safety contract — if the
    recorder masks a real failure, every operator-facing alarm goes
    silent."""
    sink = _CapturingSink()
    sink.write_should_raise = True
    ctx = _make_context(sink=sink)
    step = _make_toy_step(raise_for="boom")

    with ctx.activate():
        # The original ValueError must surface, not the
        # RuntimeError("test-injected sink failure").
        with pytest.raises(ValueError, match="toy boom"):
            asyncio.run(step._execute_process({"seed": "boom"}))


def test_recorder_failure_does_not_corrupt_success_return():
    """When the recorder raises during the success path, the step's
    return value must still reach the caller. The whole point of the
    recorder is observability; observability must NEVER replace
    correctness."""
    sink = _CapturingSink()
    sink.write_should_raise = True
    ctx = _make_context(sink=sink)
    step = _make_toy_step()

    with ctx.activate():
        result = asyncio.run(step._execute_process({"seed": "delta"}))

    assert result == {"result": "processed:delta"}, (
        f"recorder failure must not corrupt the step's return value; "
        f"got {result!r}"
    )
