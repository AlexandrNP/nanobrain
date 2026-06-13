"""Real Workflow.run regression for the E2-R synthesizer seam.

Drives a tool Step SYNTHESIZED by ``synthesize_rhea_step`` end-to-end
through an actual ``Workflow.run`` cascade — no mocked nanobrain internals.
Only the MCP WIRE is faked (httpx MockTransport), per the workspace mocks
carve-out (wire-shape only); the trigger cascade, link transfer, step
dispatch, adapter lookup, and output collection are all real.

This is the load-bearing assertion the workspace requires for a framework
change: success is read from a concrete OUTPUT VALUE (the workflow's
terminal data unit), NOT from the run's ``status`` field (which
``Workflow.run`` reports ``completed`` even when a step silently produced
nothing — G127).

The matching live test (synthesize from a REAL Galaxy tool and run it
against a REAL Rhea worker) is
``tests/integration/test_rhea_synthesize_step_live.py`` (gated on
``$RHEA_MCP_URL``).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from nanobrain.library.steps.tool_execution_step import ToolBackendRegistry
from nanobrain.library.tools.rhea_adapter import RheaAdapter
from nanobrain.library.tools.rhea_discovery import RheaMCPDiscovery
from nanobrain.library.tools.rhea_step_synthesizer import synthesize_rhea_step
from nanobrain.lightweight.workflow_builder import WorkflowBuilder


def _sse(obj: dict) -> str:
    return "event: message\ndata: " + json.dumps(obj) + "\n"


_JSON_TOOL = {
    "name": "uniprot_search",
    "title": "UniProt Search",
    "description": "Search UniProt",
    "inputSchema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "annotations": {
        "apecx_provenance": {
            "schema": 1,
            "tool_version": "2.0",
            "requirements": [],
            "containers": [],
            "version_command": "",
            "file_input_args": [],
            "stochastic": False,
        }
    },
}

_TOOL_RESULT_TEXT = "UNIPROT:P04637"


def _mcp_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    method = body.get("method")
    if method == "initialize":
        return httpx.Response(
            200,
            text=_sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
            headers={"mcp-session-id": "sess-1"},
        )
    if method == "notifications/initialized":
        return httpx.Response(202, text="")
    if method == "tools/list":
        return httpx.Response(
            200,
            text=_sse(
                {"jsonrpc": "2.0", "id": 2, "result": {"tools": [_JSON_TOOL]}}
            ),
        )
    if method == "tools/call":
        # Plain-text (non-JSON) content → parse_tool_call_result returns the
        # string → the single-output adapter wraps it under "return".
        return httpx.Response(
            200,
            text=_sse(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "result": {
                        "content": [{"type": "text", "text": _TOOL_RESULT_TEXT}],
                        "isError": False,
                    },
                }
            ),
        )
    return httpx.Response(400, text="unexpected method")


@pytest.fixture
def _fake_mcp(monkeypatch):
    """Patch discovery to use the fake MCP; register a RheaAdapter on the
    same fake wire; clean the registry up after."""
    real_init = RheaMCPDiscovery.__init__

    def fake_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.transport.client = httpx.AsyncClient(
            transport=httpx.MockTransport(_mcp_handler), timeout=5.0
        )

    monkeypatch.setattr(RheaMCPDiscovery, "__init__", fake_init)

    adapter = RheaAdapter(mcp_url="http://fake/mcp/")
    adapter.transport.client = httpx.AsyncClient(
        transport=httpx.MockTransport(_mcp_handler), timeout=5.0
    )
    ToolBackendRegistry.register(adapter)
    try:
        yield
    finally:
        ToolBackendRegistry.unregister("rhea")
        asyncio.run(adapter.aclose())


def test_synthesized_tool_step_runs_in_workflow(_fake_mcp):
    spec = asyncio.run(
        synthesize_rhea_step("uniprot_search", mcp_url="http://fake/mcp/")
    )
    assert spec.uses_file_input is False

    builder = WorkflowBuilder("rhea_synth_wf", "synthesized tool step e2e")
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

    wf = builder.load()

    async def _run():
        return await wf.run(
            {"wf_in": {"query": "p53"}},
            timeout=20.0,
            settle_ms=500,
            raise_on_cascade_timeout=False,
        )

    out = asyncio.run(_run())
    # Success is read from the OUTPUT VALUE, never from status (G127).
    assert out.get("wf_out") == _TOOL_RESULT_TEXT, (
        f"synthesized tool step did not propagate its result: {out}"
    )
