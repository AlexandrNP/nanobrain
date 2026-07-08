"""LIVE end-to-end: synthesize MUSCLE and run it to a real alignment.

This is the run that E3-4 could not close. Before the E3-R-followup fix,
``synthesize_rhea_step("muscle")`` mapped ONLY the file param into the
RheaFileToolStep and dropped every required non-file param, so the worker's
``muscleArguments`` pydantic model rejected the call with
``diags Field required [type=missing]``. The synthesizer now derives the
non-file params from the tool's inputSchema (schema defaults + caller
overrides, FAIL LOUD on a required-no-default param the caller omits), so a
real MUSCLE alignment actually RUNS.

GATED on:
* ``$RHEA_MCP_URL`` — a reachable Rhea MCP worker with MUSCLE ingested.
* the ``rhea`` repo importable client-side — ``RheaFileToolStep`` stages the
  input into Rhea's ProxyStore via ``rhea.utils.proxy.RheaFileProxy`` (the
  genuine class, by module reference), so the client process needs the
  ``rhea`` package on PYTHONPATH plus proxystore/redis/cloudpickle.

Run it with:

    PYTHONPATH=<nanobrain>:<rhea> RHEA_MCP_URL=http://localhost:3001/mcp/ \
        python -m pytest tests/integration/test_rhea_synthesize_muscle_run_live.py -s
"""

from __future__ import annotations

import asyncio
import os

import pytest

from nanobrain.library.tools.rhea_step_synthesizer import synthesize_rhea_step
from nanobrain.lightweight.workflow_builder import WorkflowBuilder

pytestmark = pytest.mark.integration

_RHEA_URL = os.environ.get("RHEA_MCP_URL")

_skip = pytest.mark.skipif(
    _RHEA_URL is None,
    reason="needs $RHEA_MCP_URL set (a reachable Rhea MCP worker with MUSCLE ingested)",
)
# NOTE: the client no longer needs the 'rhea' package importable — RheaFileToolStep is now a
# thin HTTP client (POST /upload, MCP tools/call, GET /download, POST /delete). The old
# `find_spec('rhea')` gate was removed with the redis-direct HTTP transport migration.

# Three short, near-identical globin N-termini — a real MSA the worker can
# align in seconds.
_FASTA = (
    ">seqA\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF\n"
    ">seqB\nMVLSAADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF\n"
    ">seqC\nMVLSGEDKSNIKAAWGKIGGHGAEYGAEALERMFASFPTTKTYFPHF\n"
)


def _parse_fasta(text: str) -> list[tuple[str, str]]:
    """Minimal FASTA parser → list of (header, sequence)."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
            header = line[1:].strip()
            seq = []
        elif line.strip():
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


@_skip
def test_synthesize_muscle_runs_to_nonempty_alignment():
    """synthesize_rhea_step('muscle') → RheaFileToolStep → Workflow.run on a
    3-sequence FASTA → assert a NON-EMPTY aligned FASTA (>=1 record, all
    records equal length = real alignment columns). G127: success is read
    from the OUTPUT VALUE, never from the run status."""
    # 'diags' is REQUIRED with NO schema default — the caller must supply it,
    # exactly the param the synthesizer used to silently drop. The remaining
    # required/optional non-file params (cluster, run, outputFormat, ...) are
    # auto-mapped from the inputSchema defaults.
    spec = asyncio.run(
        synthesize_rhea_step(
            "muscle",
            mcp_url=_RHEA_URL,
            find_tools_query="muscle multiple sequence alignment",
            static_tool_args={"diags": False},
        )
    )
    assert spec.uses_file_input is True
    assert spec.step_config["file_input_arg"] == "input_seqs"
    # CC-1: the synthesized args dict is NON-EMPTY and carries the required
    # param the worker's model would otherwise reject the call without.
    static_args = spec.step_config["static_tool_args"]
    assert static_args, "static_tool_args must be non-empty (the original bug)"
    assert static_args["diags"] is False
    assert "input_seqs" not in static_args  # file param staged at runtime

    builder = WorkflowBuilder("muscle_live_run", "synthesized muscle e2e")
    builder.add_input("wf_in", "DataUnitMemory")
    builder.add_output("wf_out", "DataUnitMemory")
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
    wf = builder.load()

    async def _run():
        return await wf.run(
            {"wf_in": {"fasta_name": "seqs.fasta", "fasta_text": _FASTA}},
            timeout=900.0,
            settle_ms=1000,
            raise_on_cascade_timeout=False,
        )

    out = asyncio.run(_run())
    wf_out = out.get("wf_out")
    assert isinstance(wf_out, dict) and wf_out, (
        f"muscle produced no output files (status={out.get('status')}); "
        f"G127 — a 'completed' status with empty output is a silent failure"
    )
    # MUSCLE returns the alignment under an 'out_align'-class key.
    align_key = next(
        (k for k in wf_out if "align" in k.lower() and "html" not in k.lower()),
        None,
    )
    assert align_key is not None, f"no alignment FASTA in outputs: {list(wf_out)}"
    aligned_fasta = wf_out[align_key]
    print(f"\n[LIVE] muscle alignment ({align_key!r}):\n{aligned_fasta}")

    records = _parse_fasta(aligned_fasta)
    assert len(records) >= 1, f"alignment had no records: {aligned_fasta!r}"
    lengths = {len(seq) for _, seq in records}
    # The defining property of a multiple sequence alignment: every aligned
    # record has the SAME number of columns.
    assert len(lengths) == 1, (
        f"aligned records are not equal length (not a real alignment): {lengths}"
    )
    assert lengths.pop() > 0, "aligned records are empty"
    # All three input sequences survived the round-trip.
    assert {h for h, _ in records} >= {"seqA", "seqB", "seqC"}
