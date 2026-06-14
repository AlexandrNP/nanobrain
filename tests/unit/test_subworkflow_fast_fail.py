"""Regression test for BUG A — fast inner-step-failure detection in SubworkflowStep.

When an inner step of a nested workflow RAISES, the trigger executor SWALLOWS the
exception (G127 — ``Workflow.run`` does not propagate it), so the inner output data
unit never populates. Historically ``SubworkflowStep`` then polled the inner output
until ``timeout_seconds`` elapsed before giving up — an N-minute hang for a failure
that happened in seconds, surfaced only as a generic ``TimeoutError`` that hid the
real reason.

The fix subscribes to the inner cascade's G37 ``step_failed`` events (the inner step
tasks are ``create_task``-spawned transitively within ``process()``'s task context, so
they inherit the contextvar-based subscriber). The poll loop checks the capture each
iteration and re-raises the inner step's REAL exception immediately.

These tests pin BOTH halves of the win:
  * **wall-time** — detection happens in << ``timeout_seconds`` (not at the deadline).
  * **accuracy** — the surfaced error names the inner step + its real exception
    message, not a generic timeout.

Module-level step class + builder so the dotted-path resolver can import them (same
shape as ``test_subworkflow_step_builder.py``).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import Workflow
from nanobrain.lightweight.workflow_builder import WorkflowBuilder
from nanobrain.library.steps.subworkflow_step import SubworkflowStep

_DU = "nanobrain.core.data_unit.DataUnitMemory"
_TRIGGER = "nanobrain.core.trigger.DataUnitChangeTrigger"

_RAISE_MESSAGE = "inner fetch failed: length filter left 1 sequence"


class _RaisingInnerStep(BaseStep):
    """Inner-workflow first step that ALWAYS raises — stands in for a real inner
    step (e.g. BvbrcProteinFastaStep's <2-sequences ValueError) whose exception the
    trigger executor swallows."""

    COMPONENT_TYPE = "test_raising_inner_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kwargs):
        raise ValueError(_RAISE_MESSAGE)


class _SecondInnerStep(BaseStep):
    """Deterministic downstream step. It must NEVER run — its input DU never
    populates because the first step raised. Present so the inner workflow has a
    real LAST step whose output DU the poll loop would otherwise wait on."""

    COMPONENT_TYPE = "test_second_inner_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kwargs):
        return {"never": "reached"}


def build_failing_inner_workflow() -> Workflow:
    """NO-ARG builder: a two-step inner workflow whose FIRST step raises. The
    second step's output DU is what the SubworkflowStep poll loop waits on — so
    without fast-fail detection the wait would run the full timeout."""
    b = WorkflowBuilder("failing_inner_wf", "inner workflow whose first step raises")
    b.add_input("inner_wf_in", "DataUnitMemory")
    b.add_output("inner_wf_out", "DataUnitMemory")
    b.add_step(
        "fetch",
        f"{__name__}._RaisingInnerStep",
        input_data_units={"fetch_in": {"class": _DU, "name": "fetch_in"}},
        output_data_units={"fetch_out": {"class": _DU, "name": "fetch_out"}},
        triggers=[{"class": _TRIGGER, "data_unit": "fetch_in"}],
    )
    b.add_step(
        "consume",
        f"{__name__}._SecondInnerStep",
        input_data_units={"consume_in": {"class": _DU, "name": "consume_in"}},
        output_data_units={"consume_out": {"class": _DU, "name": "consume_out"}},
        triggers=[{"class": _TRIGGER, "data_unit": "consume_in"}],
    )
    b.add_link("inner_wf_in", "fetch.fetch_in", link_type="direct")
    b.add_link("fetch.fetch_out", "consume.consume_in", link_type="direct")
    b.add_link("consume.consume_out", "inner_wf_out", link_type="direct")
    return b.load()


_LONG_TIMEOUT = 30.0  # the would-be hang; detection must be FAR under this.


def _stage_step(tmp_path) -> SubworkflowStep:
    p = tmp_path / "fast_fail_step.yml"
    p.write_text(
        "name: fast_fail_step\n"
        f"inner_workflow_builder: {__name__}.build_failing_inner_workflow\n"
        f"timeout_seconds: {_LONG_TIMEOUT}\n"
    )
    return SubworkflowStep.from_config(str(p))


def test_inner_failure_surfaced_fast_and_accurate(tmp_path):
    step = _stage_step(tmp_path)

    started = time.monotonic()
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(step.process({"taxon_id": 12637, "protein": "envelope"}))
    elapsed = time.monotonic() - started

    msg = str(exc_info.value)
    # Accuracy: the REAL inner step name + its real exception text, not a timeout.
    assert "fetch" in msg, msg
    assert "ValueError" in msg, msg
    assert _RAISE_MESSAGE in msg, msg
    assert "step_failed event" in msg, msg
    # Wall-time: surfaced in FAR less than the inner-cascade timeout (the old
    # behavior waited the full _LONG_TIMEOUT before a generic TimeoutError).
    assert elapsed < _LONG_TIMEOUT / 3, (
        f"inner failure took {elapsed:.2f}s to surface — fast-fail detection did "
        f"not short-circuit the {_LONG_TIMEOUT}s timeout"
    )


def test_inner_failure_is_not_a_timeout_error(tmp_path):
    # Negative pin: the surfaced exception must NOT be a TimeoutError (which is what
    # the deadline path raises). A TimeoutError here would mean detection regressed
    # to waiting out the clock.
    step = _stage_step(tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(step.process({"taxon_id": 1, "protein": "x"}))
    assert not isinstance(exc_info.value, TimeoutError)
