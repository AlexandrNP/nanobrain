"""Workflow.process() status dict must NOT pollute output data units.

Source: 2026-05-11 audit — surfaced by apecx-mcp-integration's
``test_workflow_native_framework_cascade``. Before the fix,
``BaseStep.execute()`` called ``_update_output_data_units(result)``
with ``result`` being the Workflow.process() status dict
(``{"status": "data_flow_initiated", "workflow": ..., "first_step":
..., "populated_units": N}``). The single-output-fallback branch
in ``_update_output_data_units`` then wrote that entire status
dict to the workflow's lone output data unit, masquerading as
real data.

Silent-failure shape: a caller reading the output unit got a
non-null value (the status dict) and might assume the cascade
fired. The cascade's actual not-firing was masked.

Fix: ``Workflow._update_output_data_units`` overrides the base
class to be a no-op for the data-driven status-dict shape. The
cascade is the data-flow mechanism; process()'s return is
observability metadata for the caller.

This test pins:
  1. After ``await wf.process({})``, the workflow's output data
     units are NOT polluted with the status dict.
  2. Subclasses that legitimately return a non-status-dict payload
     from process() still get the BaseStep behavior (no regression).
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from nanobrain.core.data_unit import DataUnitMemory
from nanobrain.core.workflow import Workflow


def _build_minimal_workflow_yaml() -> str:
    """Minimal workflow with one input data unit + one output data
    unit + one step. Used for the pollution check."""
    import yaml as _yaml

    # We use the lightweight WorkflowBuilder path to materialize a
    # real workflow YAML — testing through from_config keeps us on
    # the framework-native path.
    yml = {
        "name": "test_status_no_pollution",
        "config_version": 2,
        "input_data_units": {
            "in_a": {
                "class": "nanobrain.core.data_unit.DataUnitMemory",
                "name": "in_a",
                "persistent": False,
            },
        },
        "output_data_units": {
            "out_a": {
                "class": "nanobrain.core.data_unit.DataUnitMemory",
                "name": "out_a",
                "persistent": False,
            },
        },
        "steps": {},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        _yaml.safe_dump(yml, f)
        return f.name


def test_status_dict_does_not_land_in_output_data_unit():
    """The bug shape pin: process() returns a status dict, and
    ``execute()``'s post-call ``_update_output_data_units`` MUST NOT
    write that status dict into the workflow's output data unit."""

    async def _scenario() -> None:
        yml_path = _build_minimal_workflow_yaml()
        wf = Workflow.from_config(yml_path)
        # Call process() — returns a status dict in data-driven mode.
        result = await wf.process({})
        assert isinstance(result, dict)
        assert result.get("status") in (
            "data_flow_initiated",
            "no_first_step",
            "no_steps",
        )

        # Trigger the BaseStep.execute() path that calls
        # _update_output_data_units after process() returns. Use the
        # framework's actual call to mimic what runners do.
        await wf._update_output_data_units(result)

        # Every output data unit must NOT have the status dict.
        for unit_name, du in wf.step_output_data_units.items():
            value = await du.get()
            assert value != result, (
                f"Output unit {unit_name!r} got polluted with the "
                f"status dict {result!r}. Override semantics broke."
            )
            # Specifically — the status dict's identifying key
            # MUST NOT be in any captured output. This pins the
            # silent-failure shape: a caller doing
            # ``isinstance(value, dict) and "status" in value`` to
            # detect "is this real data" would have false-negatived
            # before the fix.
            if isinstance(value, dict):
                assert "status" not in value or value.get(
                    "status"
                ) not in (
                    "data_flow_initiated",
                    "no_first_step",
                    "no_steps",
                ), (
                    f"Output unit {unit_name!r} value contains the "
                    f"status-dict marker. Pollution regression."
                )

    asyncio.run(_scenario())


def test_workflow_subclass_with_payload_still_propagates():
    """A Workflow subclass whose process() legitimately returns a
    payload dict (not a status dict) MUST still get the BaseStep
    propagation. Pinning: my override is conditioned on the
    status-dict shape, not blanket."""

    async def _scenario() -> None:
        yml_path = _build_minimal_workflow_yaml()
        wf = Workflow.from_config(yml_path)

        # Mimic an imperative workflow's process() return shape —
        # has a key that matches the output unit name.
        payload = {"out_a": [1, 2, 3]}
        await wf._update_output_data_units(payload)

        # The output unit was updated with the matching value.
        unit = wf.step_output_data_units["out_a"]
        value = await unit.get()
        assert value == [1, 2, 3], (
            f"BaseStep behavior MUST be preserved for non-status "
            f"results; got {value!r}"
        )

    asyncio.run(_scenario())
