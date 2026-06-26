"""Unit tests for the unified find-and-establish seam.

Two paths, one return type (:class:`UnifiedToolDescriptor`):

* RHEA path — runs against a fake MCP server (httpx MockTransport, the
  same seam RheaAdapter uses). Asserts find_and_establish_tool calls
  ``find_tools`` and returns parseable UTDs, both when the result is
  directly parseable AND when it falls back to ``tools/list``.
* CUSTOM path — registers a dummy ToolBackendAdapter and asserts the
  returned UTD's descriptor_id / backend / provenance class_path; plus
  the FAIL-LOUD KeyError for an unregistered backend.

The live-Rhea path is exercised by the gated integration tests
(skipped unless $RHEA_MCP_URL is set).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.steps.tool_execution_step import (
    ToolBackendAdapter,
    ToolBackendRegistry,
)
from nanobrain.library.tools._mcp_transport import MCPTransport
from nanobrain.library.tools.tool_discovery import find_and_establish_tool


def _sse(obj: dict) -> str:
    return "event: message\ndata: " + json.dumps(obj) + "\n"


_SAMPLE_TOOLS = [
    {
        "name": "muscle",
        "title": "MUSCLE",
        "description": "Multiple sequence alignment.",
        "inputSchema": {
            "type": "object",
            "properties": {"input_seqs": {"type": "string"}},
            "required": ["input_seqs"],
        },
    },
    {
        "name": "UniProt-Search",  # violates UTD tool_id grammar -> sanitized
        "description": "Search UniProt.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


def _make_handler(
    *,
    find_tools_result: dict,
    tools_list: list[dict] | None = None,
    calls: list[str] | None = None,
):
    """Fake MCP server: initialize -> session; tools/call & tools/list."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get("method")
        if calls is not None:
            name = (body.get("params") or {}).get("name")
            calls.append(f"{method}:{name}" if name else method)
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
                200,
                text=_sse({"jsonrpc": "2.0", "id": 2, "result": find_tools_result}),
            )
        if method == "tools/list":
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {"tools": tools_list or []},
                    }
                ),
            )
        return httpx.Response(400, text="unexpected method")

    return handler


def _transport_with_mock(handler) -> MCPTransport:
    transport = MCPTransport(mcp_url="http://fake/mcp/", timeout_seconds=5.0)
    transport.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=5.0
    )
    return transport


# ---------------------------------------------------------------------------
# RHEA path
# ---------------------------------------------------------------------------

def test_rhea_path_parses_find_tools_result():
    """find_tools result carries the matched tools -> parsed into UTDs
    WITHOUT a second tools/list round-trip."""
    calls: list[str] = []
    # structuredContent wraps the matched tools under 'tools'.
    handler = _make_handler(
        find_tools_result={
            "structuredContent": {"tools": _SAMPLE_TOOLS},
            "isError": False,
        },
        calls=calls,
    )
    transport = _transport_with_mock(handler)
    utds = asyncio.run(find_and_establish_tool("align sequences", rhea_transport=transport))
    asyncio.run(transport.aclose())

    assert all(isinstance(u, UnifiedToolDescriptor) for u in utds)
    ids = {u.descriptor_id for u in utds}
    assert "rhea:muscle@unpinned" in ids
    assert "rhea:uniprot_search@unpinned" in ids
    # find_tools was called; tools/list was NOT (result was parseable).
    assert "tools/call:find_tools" in calls
    assert "tools/list" not in calls


def test_rhea_path_bare_list_result():
    """A find_tools result that is a bare list of tool dicts (via the
    text content channel) is parsed directly."""
    handler = _make_handler(
        find_tools_result={
            "content": [{"type": "text", "text": json.dumps(_SAMPLE_TOOLS)}],
            "isError": False,
        },
    )
    transport = _transport_with_mock(handler)
    utds = asyncio.run(find_and_establish_tool("align", rhea_transport=transport))
    asyncio.run(transport.aclose())
    ids = {u.descriptor_id for u in utds}
    assert "rhea:muscle@unpinned" in ids


def test_rhea_path_falls_back_to_tools_list():
    """An opaque find_tools result (no parseable tool dicts) falls back
    to a tools/list over the same session."""
    calls: list[str] = []
    handler = _make_handler(
        find_tools_result={
            "content": [{"type": "text", "text": "surfaced 2 tools"}],
            "isError": False,
        },
        tools_list=_SAMPLE_TOOLS,
        calls=calls,
    )
    transport = _transport_with_mock(handler)
    utds = asyncio.run(find_and_establish_tool("align", rhea_transport=transport))
    asyncio.run(transport.aclose())
    ids = {u.descriptor_id for u in utds}
    assert "rhea:muscle@unpinned" in ids
    assert "tools/call:find_tools" in calls
    assert "tools/list" in calls  # the fallback fired


def test_rhea_path_preserves_raw_name_for_dispatch():
    handler = _make_handler(
        find_tools_result={
            "structuredContent": {"tools": _SAMPLE_TOOLS},
            "isError": False,
        },
    )
    transport = _transport_with_mock(handler)
    utds = asyncio.run(find_and_establish_tool("x", rhea_transport=transport))
    asyncio.run(transport.aclose())
    by_id = {u.descriptor_id: u for u in utds}
    uniprot = by_id["rhea:uniprot_search@unpinned"]
    assert uniprot.provenance_pin.mcp_support["rhea_tool_name"] == "UniProt-Search"


# ---------------------------------------------------------------------------
# CUSTOM path
# ---------------------------------------------------------------------------

class _DummyAdapter(ToolBackendAdapter):
    BACKEND_NAME = "dummytestbackend"

    async def invoke(self, utd, inputs, *, run_context_namespace="", **kwargs):
        return {"result": "ok"}


@pytest.fixture
def dummy_backend():
    adapter = _DummyAdapter()
    ToolBackendRegistry.register(adapter)
    try:
        yield adapter
    finally:
        ToolBackendRegistry.unregister("dummytestbackend")


def test_custom_path_builds_utd_for_backend_tool(dummy_backend):
    utds = asyncio.run(find_and_establish_tool("dummytestbackend:render"))
    assert len(utds) == 1
    utd = utds[0]
    assert isinstance(utd, UnifiedToolDescriptor)
    assert utd.descriptor_id == "dummytestbackend:render@0.0.0"
    assert utd.descriptor_backend == "dummytestbackend"
    assert utd.descriptor_tool_id == "render"
    # provenance_pin.class_path points at the registered adapter's class.
    assert utd.provenance_pin.class_path.endswith("_DummyAdapter")


def test_custom_path_bare_backend_name(dummy_backend):
    """A bare BACKEND_NAME doubles as the tool_id."""
    utds = asyncio.run(find_and_establish_tool("dummytestbackend"))
    assert utds[0].descriptor_id == "dummytestbackend:dummytestbackend@0.0.0"


def test_custom_path_honors_passed_version(dummy_backend):
    utds = asyncio.run(
        find_and_establish_tool("dummytestbackend:render", version="2.1.0")
    )
    assert utds[0].descriptor_id == "dummytestbackend:render@2.1.0"


def test_custom_path_unknown_backend_fails_loud():
    with pytest.raises(KeyError, match="no ToolBackendAdapter registered"):
        asyncio.run(find_and_establish_tool("nosuchbackend:thing"))


class _EstablishableAdapter(ToolBackendAdapter):
    """A custom backend that ESTABLISHES itself (the docker-source shape)."""

    BACKEND_NAME = "establishtestbackend"

    def __init__(self):
        super().__init__()
        self.established = 0
        self.progress_seen = None

    async def invoke(self, utd, inputs, *, run_context_namespace="", **kwargs):
        return {"result": "ok"}

    async def ensure_established(self, *, on_progress=None):
        self.established += 1
        self.progress_seen = on_progress


@pytest.fixture
def establishable_backend():
    adapter = _EstablishableAdapter()
    ToolBackendRegistry.register(adapter)
    try:
        yield adapter
    finally:
        ToolBackendRegistry.unregister("establishtestbackend")


def test_custom_path_establishes_establishable_backend(establishable_backend):
    """An Establishable adapter is ESTABLISHED (ensure_established awaited,
    on_progress forwarded) before its UTD is returned — the custom path is a
    real find-AND-establish (PyMOL's docker-source shape), not a bare lookup."""
    sentinel = object()
    utds = asyncio.run(
        find_and_establish_tool("establishtestbackend:sasa", on_progress=sentinel)
    )
    assert establishable_backend.established == 1
    assert establishable_backend.progress_seen is sentinel
    assert utds[0].descriptor_id == "establishtestbackend:sasa@0.0.0"
