"""Regression: concurrent ``Workflow.run()`` on ONE cached instance must NOT
cross-contaminate (2026-06-14).

A ``Workflow`` owns MUTABLE data units; a cached instance reused across concurrent
``run()`` calls (the norm for a long-lived MCP server caching workflows per process)
had every overlapping caller clobber the shared workflow-level inputs/outputs (the
G122 deposit) and silently receive ANOTHER call's result — all with
``status: completed``. A 3-way concurrent echo repro reproduced it with no LLM and no
domain code: ``sent=AAA got=BBB``.

Fix: ``Workflow.run`` serializes overlapping runs on the same instance via a
per-instance, loop-rebound ``asyncio.Lock`` (``_get_run_lock``), acquired AFTER the
``nest_under_active_context`` dispatch so the nested path (which re-enters
``run(nest=False)``) takes it exactly once. Distinct instances keep their own locks
and run fully in parallel; a SubworkflowStep in the cascade drives a DIFFERENT
instance and so does not deadlock.

These tests assert the OUTPUT VALUE per concurrent run (G127 — never trust
``status`` alone), which is the only thing that catches the silent clobber.
"""

from __future__ import annotations

import asyncio

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import Workflow
from nanobrain.lightweight.workflow_builder import WorkflowBuilder

_DU = "nanobrain.core.data_unit.DataUnitMemory"
_TRIG = "nanobrain.core.trigger.DataUnitChangeTrigger"


class _EchoSlowStep(BaseStep):
    """Echo the input's ``v`` after a delay wide enough to overlap concurrent runs.

    The delay is load-bearing: an instant step would finish run A before run B even
    starts, hiding the race. The input arrives as the trigger envelope keyed by the
    step's input DU name (``{"in": {"v": ...}}``)."""

    COMPONENT_TYPE = "echo_slow_isolation_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kwargs):
        v = input_data["in"]["v"]
        await asyncio.sleep(0.4)
        return {"echoed": v}


def _build_echo_workflow() -> Workflow:
    b = WorkflowBuilder("echo_isolation_wf", "echo a value through one step")
    b.add_input("wf_in", "DataUnitMemory")
    b.add_output("wf_out", "DataUnitMemory")
    b.add_step(
        "echo",
        f"{__name__}._EchoSlowStep",
        input_data_units={"in": {"class": _DU, "name": "in"}},
        output_data_units={"out": {"class": _DU, "name": "out"}},
        triggers=[{"class": _TRIG, "data_unit": "in"}],
    )
    b.add_link("wf_in", "echo.in", link_type="direct")
    b.add_link("echo.out", "wf_out", link_type="direct")
    return b.load()


def test_concurrent_runs_on_one_instance_are_isolated():
    """Three overlapping run() calls on ONE instance each get their OWN result."""
    wf = _build_echo_workflow()

    async def _run(v):
        return await wf.run(
            {"wf_in": {"v": v}}, timeout=15.0, settle_ms=500, raise_on_cascade_timeout=False
        )

    async def _go():
        return await asyncio.gather(_run("AAA"), _run("BBB"), _run("CCC"))

    a, b, c = asyncio.run(_go())
    got = {
        "AAA": (a.get("wf_out") or {}).get("echoed"),
        "BBB": (b.get("wf_out") or {}).get("echoed"),
        "CCC": (c.get("wf_out") or {}).get("echoed"),
    }
    assert got == {"AAA": "AAA", "BBB": "BBB", "CCC": "CCC"}, (
        f"concurrent runs cross-contaminated: {got}"
    )


def test_get_run_lock_is_stable_within_a_loop_and_rebinds_across_loops():
    """Same loop → same lock object (so overlapping runs actually contend). A fresh
    loop → a fresh lock (a cached workflow is re-run across asyncio.run boundaries; a
    lock bound to the first loop would raise 'bound to a different event loop')."""
    wf = _build_echo_workflow()

    async def _two_calls_same_loop():
        return wf._get_run_lock(), wf._get_run_lock()

    l1, l2 = asyncio.run(_two_calls_same_loop())
    assert l1 is l2, "lock must be stable within one loop so concurrent runs contend"

    async def _one_call():
        return wf._get_run_lock()

    l3 = asyncio.run(_one_call())  # a brand-new event loop
    assert l3 is not l1, "lock must rebind on a fresh loop (no cross-loop binding error)"
