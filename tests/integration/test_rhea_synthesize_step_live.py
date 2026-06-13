"""LIVE integration: synthesize a Step from a REAL Galaxy tool and run it.

GATED on ``$RHEA_MCP_URL`` — skipped when no Rhea worker is reachable
(the state of the dev laptop that authored E2-R: docker daemon down, env
var unset). When a worker IS available, this is the real-data proof the
synthesizer + determinism wire actually work end-to-end:

1. Synthesize a Step for a real Galaxy tool surfaced via ``find_tools``.
2. Assert the synthesized UTD carries REAL determinism pins (a real
   version — not ``@unpinned`` — and, for a containerized tool, a real
   determinism class R1/R2 with filesystem side-effects), proving the
   Rhea-side ``apecx_provenance`` annotation reached discovery.
3. Run the synthesized Step through a real ``Workflow.run`` cascade and
   assert a concrete OUTPUT VALUE (never the run ``status`` — G127).

Configure via env:
* ``RHEA_MCP_URL``                     — the worker endpoint (required).
* ``RHEA_SYNTH_TOOL_NAME``             — tool to synthesize (default
                                         ``"muscle"``).
* ``RHEA_SYNTH_FIND_QUERY``            — find_tools query to surface it
                                         (default ``"sequence alignment"``).

This test deliberately makes NO determinism-pin assertion that a
non-containerized / unpinned tool would fail — if the chosen tool is
unpinned, only the run path is asserted, and the pin state is reported.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from nanobrain.library.steps.tool_execution_step import ToolBackendRegistry
from nanobrain.library.tools.rhea_adapter import RheaAdapter
from nanobrain.library.tools.rhea_step_synthesizer import synthesize_rhea_step
from nanobrain.lightweight.workflow_builder import WorkflowBuilder

pytestmark = pytest.mark.integration

_RHEA_URL = os.environ.get("RHEA_MCP_URL")
_rhea_skip = pytest.mark.skipif(_RHEA_URL is None, reason="RHEA_MCP_URL not set")

_TOOL_NAME = os.environ.get("RHEA_SYNTH_TOOL_NAME", "muscle")
_FIND_QUERY = os.environ.get(
    "RHEA_SYNTH_FIND_QUERY", "muscle multiple sequence alignment"
)
# Caller overrides for the tool's REQUIRED non-file params that carry no
# schema default (the synthesizer now FAILS LOUD on these rather than
# silently dropping them). Defaults to muscle's required ``diags`` since the
# default tool is muscle; override via $RHEA_SYNTH_STATIC_ARGS (JSON) when
# pointing RHEA_SYNTH_TOOL_NAME at a different file tool.
_STATIC_ARGS = json.loads(
    os.environ.get("RHEA_SYNTH_STATIC_ARGS", '{"diags": false}')
)


@_rhea_skip
def test_synthesize_real_tool_carries_determinism_pins():
    """A discovered+synthesized UTD must carry the worker's REAL version
    + (for a containerized tool) honest determinism, proving the
    apecx_provenance wire reached discovery."""
    spec = asyncio.run(
        synthesize_rhea_step(
            _TOOL_NAME,
            mcp_url=_RHEA_URL,
            find_tools_query=_FIND_QUERY,
            static_tool_args=_STATIC_ARGS,
        )
    )
    utd = spec.utd
    print(
        f"\n[LIVE] synthesized {spec.descriptor_id!r} "
        f"determinism={utd.get('determinism')} "
        f"side_effects={utd.get('side_effects')} "
        f"file_input={spec.uses_file_input} pinned={spec.is_pinned}"
    )
    # The worker MUST have surfaced a determinism class.
    assert utd.get("determinism") in {"R1", "R2", "R3"}
    if spec.is_pinned:
        # A real version → it must NOT be the fabricated @unpinned sentinel.
        assert not spec.descriptor_id.endswith("@unpinned")
        # A pinned + containerized tool reports R1/R2 + filesystem effects.
        mcp_support = (utd.get("provenance_pin") or {}).get("mcp_support") or {}
        if mcp_support.get("container_image_ref") or (utd.get("provenance_pin") or {}).get(
            "container_image_digest"
        ):
            assert utd["determinism"] in {"R1", "R2"}
            assert utd["side_effects"] == "filesystem_write"


@_rhea_skip
def test_synthesized_file_tool_runs_end_to_end():
    """Run the synthesized Step through a real Workflow.run cascade and
    assert a concrete output value (G127 — never trust status)."""
    spec = asyncio.run(
        synthesize_rhea_step(
            _TOOL_NAME,
            mcp_url=_RHEA_URL,
            find_tools_query=_FIND_QUERY,
            static_tool_args=_STATIC_ARGS,
        )
    )

    builder = WorkflowBuilder("rhea_live_synth", "live synthesized run")
    builder.add_input("wf_in", "DataUnitMemory")
    builder.add_output("wf_out", "DataUnitMemory")

    if spec.uses_file_input:
        # RheaFileToolStep — output_files is the terminal key.
        builder.add_rhea_step(
            "tool",
            spec,
            input_data_units={
                "tool_in": {
                    "class": "nanobrain.core.data_unit.DataUnitMemory",
                    "name": "tool_in",
                }
            },
            output_data_units={
                "output_files": {
                    "class": "nanobrain.core.data_unit.DataUnitMemory",
                    "name": "output_files",
                }
            },
            triggers=[
                {
                    "class": "nanobrain.core.trigger.DataUnitChangeTrigger",
                    "data_unit": "tool_in",
                }
            ],
        )
        builder.add_link("wf_in", "tool.tool_in", link_type="direct")
        builder.add_link("tool.output_files", "wf_out", link_type="direct")
        payload = {
            "wf_in": {
                "fasta_name": "seqs.fasta",
                "fasta_text": ">a\nMVLSPADKTNVKAAW\n>b\nMVLSAADKTNVKAAW\n",
            }
        }
    else:
        adapter = RheaAdapter(mcp_url=_RHEA_URL)
        if "rhea" not in ToolBackendRegistry.list_backends():
            ToolBackendRegistry.register(adapter)
        builder.add_rhea_step(
            "tool",
            spec,
            input_data_units={
                "tool_in": {
                    "class": "nanobrain.core.data_unit.DataUnitMemory",
                    "name": "tool_in",
                }
            },
            output_data_units={
                "return": {
                    "class": "nanobrain.core.data_unit.DataUnitMemory",
                    "name": "return",
                }
            },
            triggers=[
                {
                    "class": "nanobrain.core.trigger.DataUnitChangeTrigger",
                    "data_unit": "tool_in",
                }
            ],
        )
        builder.add_link("wf_in", "tool.tool_in", link_type="direct")
        builder.add_link("tool.return", "wf_out", link_type="direct")
        # A minimal JSON payload — caller should set RHEA_SYNTH_TOOL_NAME to a
        # tool whose required inputs this satisfies.
        payload = {"wf_in": {i["name"]: "" for i in spec.utd.get("inputs", [])}}

    wf = builder.load()

    async def _run():
        return await wf.run(
            payload, timeout=900.0, settle_ms=1000, raise_on_cascade_timeout=False
        )

    out = asyncio.run(_run())
    print(f"\n[LIVE] workflow output: {str(out)[:400]}")
    assert out.get("wf_out") is not None, (
        f"synthesized tool produced no output (status={out.get('status')}); "
        f"G127 — a 'completed' status with empty output is a silent failure"
    )
