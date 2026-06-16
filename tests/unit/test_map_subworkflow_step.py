"""Tests for ``MapSubworkflowStep`` — run an inner workflow once per list item.

Per the G99 cycle-bearing rule, load/smoke is necessary but not sufficient: each
test drives a REAL inner ``Workflow.run()`` (built via the lightweight builder,
real DataUnitChangeTrigger cascade, real DirectLinks) and asserts on concrete
OUTPUT VALUES. G117 is honored: the map step's own input DU (``map_step_in``)
differs from the inner workflow's input DU (``map_inner_in``).
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import Workflow
from nanobrain.lightweight.workflow_builder import WorkflowBuilder
from nanobrain.library.steps.map_subworkflow_step import MapSubworkflowStep

_DU = "nanobrain.core.data_unit.DataUnitMemory"
_TRIGGER = "nanobrain.core.trigger.DataUnitChangeTrigger"


class _ComputeStep(BaseStep):
    """Inner domain step: returns {result: n + base}; raises when n < 0."""

    COMPONENT_TYPE = "test_map_compute_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        await asyncio.sleep(0.05)  # let gathered runs actually overlap
        # The framework delivers the step input as {<input_du>: payload}; unwrap.
        payload = input_data.get("compute_in", input_data)
        n = payload.get("n")
        base = payload.get("base", 0)
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"bad n={n!r}")
        return {"result": n + base}


def _build_map_inner_workflow() -> Workflow:
    """NO-ARG builder returning a real loaded inner Workflow (per-item compute)."""
    b = WorkflowBuilder("map_inner_wf", "per-item compute inner workflow")
    b.add_input("map_inner_in", "DataUnitMemory")
    b.add_output("map_inner_out", "DataUnitMemory")
    b.add_step(
        "compute",
        f"{__name__}._ComputeStep",
        input_data_units={"compute_in": {"class": _DU, "name": "compute_in"}},
        output_data_units={"compute_out": {"class": _DU, "name": "compute_out"}},
        triggers=[{"class": _TRIGGER, "data_unit": "compute_in"}],
    )
    b.add_link("map_inner_in", "compute.compute_in", link_type="direct")
    b.add_link("compute.compute_out", "map_inner_out", link_type="direct")
    return b.load()


def _map_step(tmp_path, **overrides) -> MapSubworkflowStep:
    body = (
        "name: map_test\n"
        f"inner_workflow_builder: {__name__}._build_map_inner_workflow\n"
        "item_list_key: numbers\n"
        "item_param_key: n\n"
        "static_params_keys: [base]\n"
        "step_input_data_unit_name: map_step_in\n"
        "max_concurrency: 4\n"
        "timeout_seconds: 30\n"
        "settle_ms: 500\n"
    )
    for k, v in overrides.items():
        body += f"{k}: {v}\n"
    p = tmp_path / "map.yml"
    p.write_text(body)
    return MapSubworkflowStep.from_config(str(p))


def test_loads_via_from_config(tmp_path):
    step = _map_step(tmp_path)
    assert step.name == "map_test"
    assert step._item_list_key == "numbers"


def test_maps_over_list_collects_results_in_order(tmp_path):
    step = _map_step(tmp_path)
    out = asyncio.run(step.process({"numbers": [1, 2, 3], "base": 10}))
    items = out["items"]
    assert [it["result"] for it in items] == [11, 12, 13]
    assert out["_map_errors"] == {}


def test_per_item_failure_is_named_note_not_whole_step_failure(tmp_path):
    step = _map_step(tmp_path)
    out = asyncio.run(step.process({"numbers": [1, -1, 2], "base": 0}))
    items = out["items"]
    assert items[0] == {"result": 1}
    assert "_map_item_error" in items[1] and "ValueError" in items[1]["_map_item_error"]
    assert items[2] == {"result": 2}
    assert set(out["_map_errors"].keys()) == {1}


def test_empty_list_returns_empty(tmp_path):
    step = _map_step(tmp_path)
    out = asyncio.run(step.process({"numbers": [], "base": 5}))
    assert out["items"] == [] and out["_map_errors"] == {}


def test_trigger_envelope_unwrap(tmp_path):
    step = _map_step(tmp_path)
    out = asyncio.run(step.process({"map_step_in": {"numbers": [4], "base": 1}}))
    assert out["items"] == [{"result": 5}]


def test_missing_list_fails_fast(tmp_path):
    step = _map_step(tmp_path)
    with pytest.raises(ComponentConfigurationError, match="no list under"):
        asyncio.run(step.process({"base": 1}))
