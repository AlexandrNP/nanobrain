"""Unit tests for RheaAdapter — the BACKEND_NAME='rhea' ToolBackendAdapter.

Runs UNCONDITIONALLY against a fake MCP server (httpx MockTransport).
No live Rhea worker needed. The end-to-end path against a real Rhea
worker is covered by the gated integration test
(apecx-mcp-integration tests/integration/test_open_rosalind_rhea_workflow.py,
skipped unless $RHEA_MCP_URL is set).

Mirrors the fake-MCP-server pattern from test_rhea_mcp_dispatcher.py.
The MockTransport is installed via the ``adapter.transport.client``
test seam (RheaAdapter delegates the MCP wire protocol to the shared
MCPTransport).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.steps.tool_execution_step import ToolBackendRegistry
from nanobrain.library.tools.rhea_adapter import RheaAdapter

# ---- fake MCP server helpers (httpx MockTransport) ----


def _utd(descriptor_id: str = "rhea:sequence.analyze@1.0.0", *, outputs=None, inputs=None):
    return UnifiedToolDescriptor.from_dict(
        {
            "descriptor_id": descriptor_id,
            "display_name": "Sequence Analyze",
            "summary": "Synthetic UTD for RheaAdapter tests.",
            "long_description": "",
            "inputs": inputs
            or [
                {
                    "name": "sequence",
                    "type": "string",
                    "description": "",
                    "required": True,
                    "default": None,
                }
            ],
            "outputs": outputs or [{"name": "return", "type": "object", "description": ""}],
            "side_effects": "none",
            "determinism": "R3",
            "resource_class": "cpu_light",
            "provenance_pin": {
                "class_path": "nanobrain.library.tools.rhea_adapter.RheaAdapter",
            },
        }
    )


def _sse(obj: dict) -> str:
    return "event: message\ndata: " + json.dumps(obj) + "\n"


def _adapter_with_mock(handler) -> RheaAdapter:
    """RheaAdapter whose MCP transport client is swapped for a MockTransport."""
    adapter = RheaAdapter(mcp_url="http://fake/mcp/", timeout_seconds=5.0)
    adapter.transport.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=5.0
    )
    return adapter


def _standard_handler(tool_result: Any):
    """Handler: initialize -> session header; tools/call -> tool_result."""

    def handler(request: httpx.Request) -> httpx.Response:
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
        if method == "tools/call":
            return httpx.Response(
                200, text=_sse({"jsonrpc": "2.0", "id": 2, "result": tool_result})
            )
        return httpx.Response(400, text="unexpected method")

    return handler


# ---- tests ----


def test_backend_name_and_subclass():
    from nanobrain.library.steps.tool_execution_step import ToolBackendAdapter

    assert RheaAdapter.BACKEND_NAME == "rhea"
    assert issubclass(RheaAdapter, ToolBackendAdapter)


def test_construction_rejects_empty_url():
    with pytest.raises(ComponentConfigurationError, match="non-empty mcp_url"):
        RheaAdapter(mcp_url="")


def test_from_env_fails_loud_without_var(monkeypatch):
    monkeypatch.delenv("RHEA_MCP_URL", raising=False)
    with pytest.raises(ComponentConfigurationError, match="RHEA_MCP_URL"):
        RheaAdapter.from_env()


def test_from_env_builds_and_registers(monkeypatch):
    monkeypatch.setenv("RHEA_MCP_URL", "http://localhost:3001/mcp/")
    ToolBackendRegistry.unregister("rhea")
    try:
        adapter = RheaAdapter.from_env()
        assert "rhea" in ToolBackendRegistry.list_backends()
        assert ToolBackendRegistry.get("rhea") is adapter
    finally:
        ToolBackendRegistry.unregister("rhea")


def test_invoke_happy_path_dict_result():
    # MCP tool returns structuredContent dict -> returned as-is.
    handler = _standard_handler(
        {"structuredContent": {"type": "dna", "length": 9}, "isError": False}
    )
    adapter = _adapter_with_mock(handler)
    out = asyncio.run(adapter.invoke(_utd(), {"sequence": "ATGAAACGT"}))
    assert out == {"type": "dna", "length": 9}
    asyncio.run(adapter.aclose())


def test_invoke_single_output_wraps_scalar():
    # MCP tool returns a bare text payload, JSON-parseable -> wrapped
    # under the single UTD output name.
    handler = _standard_handler(
        {"content": [{"type": "text", "text": json.dumps("dna, length 9")}]}
    )
    adapter = _adapter_with_mock(handler)
    out = asyncio.run(adapter.invoke(_utd(), {"sequence": "ATGAAACGT"}))
    assert out == {"return": "dna, length 9"}
    asyncio.run(adapter.aclose())


def test_invoke_missing_required_input_fails_loud():
    adapter = _adapter_with_mock(_standard_handler({"structuredContent": {}}))
    with pytest.raises(ComponentConfigurationError, match="missing required input"):
        asyncio.run(adapter.invoke(_utd(), {}))  # 'sequence' is required
    asyncio.run(adapter.aclose())


def test_invoke_propagates_mcp_jsonrpc_error():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                text=_sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
                headers={"mcp-session-id": "sess-1"},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202, text="")
        # tools/call -> JSON-RPC error
        return httpx.Response(
            200,
            text=_sse(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": -32000, "message": "tool blew up"},
                }
            ),
        )

    adapter = _adapter_with_mock(handler)
    with pytest.raises(ComponentConfigurationError, match="tool blew up"):
        asyncio.run(adapter.invoke(_utd(), {"sequence": "ATG"}))
    asyncio.run(adapter.aclose())


def test_invoke_propagates_is_error():
    handler = _standard_handler(
        {"isError": True, "content": [{"type": "text", "text": "bad input"}]}
    )
    adapter = _adapter_with_mock(handler)
    with pytest.raises(ComponentConfigurationError, match="isError=True"):
        asyncio.run(adapter.invoke(_utd(), {"sequence": "ATG"}))
    asyncio.run(adapter.aclose())


def test_invoke_non_200_fails_loud():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                text=_sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
                headers={"mcp-session-id": "sess-1"},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202, text="")
        return httpx.Response(500, text="rhea worker on fire")

    adapter = _adapter_with_mock(handler)
    with pytest.raises(ComponentConfigurationError, match="HTTP 500"):
        asyncio.run(adapter.invoke(_utd(), {"sequence": "ATG"}))
    asyncio.run(adapter.aclose())


def test_resolve_tool_name_strips_prefix_and_version():
    assert (
        RheaAdapter._resolve_tool_name(_utd("rhea:uniprot.search@2.1.0"))
        == "uniprot.search"
    )


def test_resolve_tool_name_honors_provenance_pin_override():
    # A UTD whose MCP-side name differs from the sanitized tool_id.
    utd = UnifiedToolDescriptor.from_dict(
        {
            "descriptor_id": "rhea:sequence_analyze@1.0.0",
            "display_name": "Sequence Analyze",
            "summary": "override test",
            "long_description": "",
            "inputs": [],
            "outputs": [{"name": "return", "type": "object", "description": ""}],
            "side_effects": "network",
            "determinism": "R3",
            "resource_class": "cpu_light",
            "provenance_pin": {
                "class_path": "nanobrain.library.tools.rhea_adapter.RheaAdapter",
                "mcp_support": {"rhea_tool_name": "Sequence-Analyze"},
            },
        }
    )
    assert RheaAdapter._resolve_tool_name(utd) == "Sequence-Analyze"


def test_initialize_missing_session_header_fails_loud():
    def handler(request: httpx.Request) -> httpx.Response:
        # initialize returns 200 but NO mcp-session-id header
        return httpx.Response(200, text=_sse({"jsonrpc": "2.0", "id": 1, "result": {}}))

    adapter = _adapter_with_mock(handler)
    with pytest.raises(ComponentConfigurationError, match="mcp-session-id"):
        asyncio.run(adapter.invoke(_utd(), {"sequence": "ATG"}))
    asyncio.run(adapter.aclose())
