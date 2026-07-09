"""Regression: an AllDataReceivedTrigger fan-in must not silently HALT the cascade (G127).

A fan-in trigger fires its downstream cascade through a DEBOUNCED task (`TriggerBase.trigger`,
`debounce_ms>0`) and an immediate pre-populated check (`AllDataReceivedTrigger.start_monitoring`),
both of which `asyncio.sleep` before running their callbacks. Historically neither task was added to
`AsyncTriggerExecutor.background_tasks`, so during that sleep they were invisible to the cascade drain
(`wait_for_all_tasks` → `_scoped(background_tasks)`). If a debounce task was the only in-flight cascade
work at a settle checkpoint, the drain returned early and `Workflow.run` reported ``completed`` with the
terminal output data unit still None — a silent halt (reproduced deterministically on the real
``viral_epitope_analysis`` flagship workflow, whose ``merge``/``envelope`` fan-ins never ran).

The fix (`_track_cascade_task`) registers + tags both tasks so the drain waits for them. These tests
pin the SYMPTOM (workflow-level: terminal output is populated under a debounce>settle race) and the
MECHANISM (unit-level: the debounce task is tracked in ``background_tasks`` while in flight).
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.lightweight.workflow_builder import WorkflowBuilder

_DU = "nanobrain.core.data_unit.DataUnitMemory"
_CHANGE = "nanobrain.core.trigger.DataUnitChangeTrigger"
_ALLDATA = "nanobrain.core.trigger.AllDataReceivedTrigger"


# --- workflow steps (module-level so class paths resolve) -------------------------------------


class _Producer(BaseStep):
    COMPONENT_TYPE = "fanin_prod"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        return {"left": "L", "right": "R"}


class _LeftArm(BaseStep):
    COMPONENT_TYPE = "fanin_left"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        p = input_data.get("l_in", input_data)
        return {"lval": p.get("left", "?")}


class _RightArm(BaseStep):
    COMPONENT_TYPE = "fanin_right"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        p = input_data.get("r_in", input_data)
        return {"rval": p.get("right", "?")}


class _Assemble(BaseStep):
    """Fan-in over both arms via AllDataReceivedTrigger."""

    COMPONENT_TYPE = "fanin_asm"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        return {"assembled": sorted(k for k in input_data.keys())}


class _Consumer(BaseStep):
    COMPONENT_TYPE = "fanin_cons"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        return {"final": True}


def _build_fanin(debounce_ms: int):
    b = WorkflowBuilder("fanin_drain_wf", "producer -> {left,right} -> assemble(fanin) -> consumer")
    b.add_input("wf_in", "DataUnitMemory")
    b.add_output("wf_out", "DataUnitMemory")

    b.add_step(
        "producer",
        f"{__name__}._Producer",
        input_data_units={"p_in": {"class": _DU, "name": "p_in"}},
        output_data_units={"p_out": {"class": _DU, "name": "p_out"}},
        triggers=[{"class": _CHANGE, "data_unit": "p_in"}],
    )
    b.add_step(
        "left",
        f"{__name__}._LeftArm",
        input_data_units={"l_in": {"class": _DU, "name": "l_in"}},
        output_data_units={"l_out": {"class": _DU, "name": "l_out"}},
        triggers=[{"class": _CHANGE, "data_unit": "l_in"}],
    )
    b.add_step(
        "right",
        f"{__name__}._RightArm",
        input_data_units={"r_in": {"class": _DU, "name": "r_in"}},
        output_data_units={"r_out": {"class": _DU, "name": "r_out"}},
        triggers=[{"class": _CHANGE, "data_unit": "r_in"}],
    )
    b.add_step(
        "assemble",
        f"{__name__}._Assemble",
        input_data_units={
            "asm_l": {"class": _DU, "name": "asm_l"},
            "asm_r": {"class": _DU, "name": "asm_r"},
        },
        output_data_units={"asm_out": {"class": _DU, "name": "asm_out"}},
        triggers=[
            {"class": _ALLDATA, "data_units": ["asm_l", "asm_r"], "debounce_ms": debounce_ms}
        ],
    )
    b.add_step(
        "consumer",
        f"{__name__}._Consumer",
        input_data_units={"c_in": {"class": _DU, "name": "c_in"}},
        output_data_units={"c_out": {"class": _DU, "name": "c_out"}},
        triggers=[{"class": _CHANGE, "data_unit": "c_in"}],
    )

    b.add_link("wf_in", "producer.p_in", link_type="direct")
    b.add_link("producer.p_out", "left.l_in", link_type="direct")
    b.add_link("producer.p_out", "right.r_in", link_type="direct")
    b.add_link("left.l_out", "assemble.asm_l", link_type="direct")
    b.add_link("right.r_out", "assemble.asm_r", link_type="direct")
    b.add_link("assemble.asm_out", "consumer.c_in", link_type="direct")
    b.add_link("consumer.c_out", "wf_out", link_type="direct")
    return b.load()


@pytest.mark.asyncio
async def test_fanin_debounce_does_not_silently_halt_cascade(monkeypatch):
    """producer -> {left,right} -> assemble(AllDataReceivedTrigger, debounce>settle) -> consumer.

    Under a debounce(80ms) that exceeds the settle window (short settle), the pre-fix cascade drain
    returned before the fan-in's debounced downstream fired — ``wf_out`` came back None with
    ``status: completed`` (15/15 in the root-cause repro). With ``_track_cascade_task`` the drain
    waits for the debounce task, so the consumer runs and ``wf_out`` is populated EVERY run.
    """
    # Allow the short settle that exposes the race (the drain's 500ms floor would otherwise mask it).
    monkeypatch.setenv("NANOBRAIN_ALLOW_SHORT_SETTLE_MS", "1")
    reps = 6
    for i in range(reps):
        wf = _build_fanin(debounce_ms=80)
        result = await wf.run({"wf_in": {"i": i}}, timeout=30, settle_ms=60)
        assert result.get("status") == "completed", f"run {i}: {result!r}"
        # G127 honesty — decide success from the OUTPUT VALUE, not status.
        assert result.get("wf_out") is not None, (
            f"run {i}: fan-in cascade silently halted — wf_out is None (status={result.get('status')})"
        )
        assert result["wf_out"].get("final") is True, f"run {i}: consumer did not run: {result['wf_out']!r}"


@pytest.mark.asyncio
async def test_fanin_immediate_prepopulated_check_does_not_halt(monkeypatch):
    """The pre-populated fan-in path (immediate check in start_monitoring) must also drain.

    When both fan-in inputs are already present at ``start_monitoring`` (a common cascade shape), the
    trigger fires via the immediate ``_check_and_maybe_fire`` task rather than a change event. That
    task was also untracked; a debounce on top made it invisible to the drain. With the fix it is
    tracked, so the terminal output is populated. Uses a debounce so both untracked-task sites are
    exercised together.
    """
    monkeypatch.setenv("NANOBRAIN_ALLOW_SHORT_SETTLE_MS", "1")
    for i in range(4):
        wf = _build_fanin(debounce_ms=40)
        result = await wf.run({"wf_in": {"i": i}}, timeout=30, settle_ms=50)
        assert result.get("wf_out") is not None, (
            f"run {i}: pre-populated fan-in halted — wf_out None (status={result.get('status')})"
        )
