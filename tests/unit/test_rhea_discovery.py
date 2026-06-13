"""Unit tests for RheaMCPDiscovery — the codegen-as-MCP-client mechanism.

Runs UNCONDITIONALLY against a fake MCP server (httpx MockTransport).
The live-Rhea discovery path is exercised by the gated integration
test (apecx-mcp-integration tests/integration/test_open_rosalind_rhea_workflow.py,
skipped unless $RHEA_MCP_URL is set).

Key contract assertions:
* Discovered MCP tools convert to UTD dicts that actually parse via
  UnifiedToolDescriptor.from_dict (the round-trip the codegen depends on).
* MCP tool names that violate the UTD tool_id grammar are sanitized
  in the descriptor_id BUT preserved verbatim in
  provenance_pin.mcp_support.rhea_tool_name (so dispatch still works).
* An empty tools list FAILS LOUD — never a silent zero-tool catalog.

The MockTransport is installed via the ``discovery.transport.client``
test seam (RheaMCPDiscovery delegates the MCP wire protocol to the
shared MCPTransport).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.tools.rhea_discovery import RheaMCPDiscovery


def _sse(obj: dict) -> str:
    return "event: message\ndata: " + json.dumps(obj) + "\n"


def _make_handler(tools: list[dict] | None, *, tools_key_present: bool = True):
    """Fake MCP server: initialize -> session; tools/list -> tools."""

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
        if method == "tools/list":
            result: dict = {}
            if tools_key_present:
                result["tools"] = tools if tools is not None else []
            return httpx.Response(
                200, text=_sse({"jsonrpc": "2.0", "id": 2, "result": result})
            )
        return httpx.Response(400, text="unexpected method")

    return handler


def _discovery_with_mock(handler) -> RheaMCPDiscovery:
    disco = RheaMCPDiscovery(mcp_url="http://fake/mcp/", timeout_seconds=5.0)
    disco.transport.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=5.0
    )
    return disco


_SAMPLE_TOOLS = [
    {
        "name": "sequence.analyze",
        "description": "Analyze a biological sequence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "the input sequence"},
            },
            "required": ["sequence"],
        },
    },
    {
        "name": "UniProt-Search",  # violates UTD tool_id grammar -> sanitized
        "description": "Search UniProt for a protein.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def test_from_env_fails_loud_without_var(monkeypatch):
    monkeypatch.delenv("RHEA_MCP_URL", raising=False)
    with pytest.raises(ComponentConfigurationError, match="RHEA_MCP_URL"):
        RheaMCPDiscovery.from_env()


def test_construction_rejects_empty_url():
    with pytest.raises(ComponentConfigurationError, match="non-empty mcp_url"):
        RheaMCPDiscovery(mcp_url="")


def test_discover_returns_parseable_utds():
    disco = _discovery_with_mock(_make_handler(_SAMPLE_TOOLS))
    utds = asyncio.run(disco.discover())
    asyncio.run(disco.aclose())
    assert len(utds) == 2
    # The load-bearing assertion: every discovered dict parses as a UTD.
    parsed = [UnifiedToolDescriptor.from_dict(u) for u in utds]
    ids = {p.descriptor_id for p in parsed}
    # These sample tools carry NO apecx_provenance annotation, so they are
    # honestly UNPINNED (@unpinned) — never a fabricated @1.0.0.
    assert "rhea:sequence.analyze@unpinned" in ids
    # 'UniProt-Search' sanitized into the tool_id grammar.
    assert "rhea:uniprot_search@unpinned" in ids


def test_discover_preserves_original_mcp_name_for_dispatch():
    disco = _discovery_with_mock(_make_handler(_SAMPLE_TOOLS))
    utds = asyncio.run(disco.discover())
    asyncio.run(disco.aclose())
    by_id = {u["descriptor_id"]: u for u in utds}
    # The sanitized one must keep the ORIGINAL name for RheaAdapter dispatch.
    sanitized = by_id["rhea:uniprot_search@unpinned"]
    assert sanitized["provenance_pin"]["mcp_support"]["rhea_tool_name"] == "UniProt-Search"


def test_discover_maps_input_schema():
    disco = _discovery_with_mock(_make_handler(_SAMPLE_TOOLS))
    utds = asyncio.run(disco.discover())
    asyncio.run(disco.aclose())
    by_id = {u["descriptor_id"]: u for u in utds}
    uniprot = by_id["rhea:uniprot_search@unpinned"]
    input_names = {i["name"] for i in uniprot["inputs"]}
    assert input_names == {"query", "limit"}
    query_input = next(i for i in uniprot["inputs"] if i["name"] == "query")
    assert query_input["required"] is True
    limit_input = next(i for i in uniprot["inputs"] if i["name"] == "limit")
    assert limit_input["required"] is False  # not in 'required' array
    assert limit_input["default"] == 5
    assert limit_input["has_default"] is True  # schema declared a default
    # 'query' has no default key → has_default False (distinct from default: null)
    assert query_input["has_default"] is False


def test_discover_provenance_pin_points_at_rhea_adapter():
    disco = _discovery_with_mock(_make_handler(_SAMPLE_TOOLS))
    utds = asyncio.run(disco.discover())
    asyncio.run(disco.aclose())
    for u in utds:
        assert (
            u["provenance_pin"]["class_path"]
            == "nanobrain.library.tools.rhea_adapter.RheaAdapter"
        )


def test_discover_empty_tool_list_fails_loud():
    """An empty tools list must FAIL LOUD — a codegen handed zero tools
    would generate an empty/no-op workflow (silent-failure shape)."""
    disco = _discovery_with_mock(_make_handler([]))
    with pytest.raises(ComponentConfigurationError, match="empty tools list"):
        asyncio.run(disco.discover())
    asyncio.run(disco.aclose())


def test_discover_missing_tools_key_fails_loud():
    disco = _discovery_with_mock(_make_handler(None, tools_key_present=False))
    with pytest.raises(ComponentConfigurationError, match="no 'tools' array"):
        asyncio.run(disco.discover())
    asyncio.run(disco.aclose())


def test_discover_jsonrpc_error_fails_loud():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                text=_sse({"jsonrpc": "2.0", "id": 1, "result": {}}),
                headers={"mcp-session-id": "s1"},
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202, text="")
        return httpx.Response(
            200,
            text=_sse(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": -32601, "message": "tools/list not supported"},
                }
            ),
        )

    disco = _discovery_with_mock(handler)
    with pytest.raises(ComponentConfigurationError, match="tools/list not supported"):
        asyncio.run(disco.discover())
    asyncio.run(disco.aclose())


def test_sanitize_tool_id_grammar():
    # leading non-alpha gets a 't_' prefix.
    assert RheaMCPDiscovery._sanitize_tool_id("3prime-utr") == "t_3prime_utr"
    assert RheaMCPDiscovery._sanitize_tool_id("Sequence.Analyze") == "sequence.analyze"


# ---------------------------------------------------------------------------
# E2-R Priority 2 — discovery reads the apecx_provenance determinism block.
# ---------------------------------------------------------------------------

def _tool_with_provenance(prov: dict, *, name="muscle", file_param=True) -> dict:
    """A tools/list entry carrying an apecx_provenance annotation block."""
    props = {"input_seqs": {"type": "string"}} if file_param else {
        "query": {"type": "string"}
    }
    return {
        "name": name,
        "title": name.upper(),
        "description": f"{name} tool",
        "inputSchema": {"type": "object", "properties": props, "required": []},
        "annotations": {"title": name.upper(), "apecx_provenance": prov},
    }


_MUSCLE_PROV = {
    "schema": 1,
    "tool_version": "5.1.0",
    "requirements": [{"type": "package", "name": "muscle", "version": "5.1"}],
    "containers": [
        {"type": "docker", "value": "quay.io/biocontainers/muscle:5.1--h99_0"}
    ],
    "version_command": "muscle -version",
    "file_input_args": ["input_seqs"],
    "stochastic": False,
}


def _discover_one(tool: dict) -> dict:
    disco = _discovery_with_mock(_make_handler([tool]))
    utds = asyncio.run(disco.discover())
    asyncio.run(disco.aclose())
    return utds[0]


def test_discover_reads_real_version_into_descriptor_id():
    """A worker-pinned version produces a REAL descriptor_id version —
    the bug this task fixes (was blanket @1.0.0)."""
    u = _discover_one(_tool_with_provenance(_MUSCLE_PROV))
    assert u["descriptor_id"] == "rhea:muscle@5.1.0"
    # And it round-trips through the UTD validator.
    parsed = UnifiedToolDescriptor.from_dict(u)
    assert parsed.descriptor_version == "5.1.0"


def test_discover_versioned_containerized_tool_is_r2_filesystem():
    """Versioned + containerized => honest R2 (not blanket R3) and
    filesystem_write side-effects (not blanket network)."""
    u = _discover_one(_tool_with_provenance(_MUSCLE_PROV))
    assert u["determinism"] == "R2"
    assert u["side_effects"] == "filesystem_write"
    # The container is a TAG ref, not a digest — recorded as ref, NOT as
    # a false digest.
    assert "container_image_digest" not in u["provenance_pin"]
    assert (
        u["provenance_pin"]["mcp_support"]["container_image_ref"]
        == "quay.io/biocontainers/muscle:5.1--h99_0"
    )


def test_discover_reads_oci_digest_as_digest():
    """A real @sha256 digest IS pinned into container_image_digest."""
    prov = dict(_MUSCLE_PROV)
    prov["containers"] = [
        {
            "type": "docker",
            "value": "quay.io/biocontainers/muscle@sha256:" + "a" * 64,
        }
    ]
    u = _discover_one(_tool_with_provenance(prov))
    assert u["provenance_pin"]["container_image_digest"] == (
        "quay.io/biocontainers/muscle@sha256:" + "a" * 64
    )
    assert u["determinism"] == "R2"


def test_discover_stochastic_tool_is_r3_even_when_containerized():
    prov = dict(_MUSCLE_PROV)
    prov["stochastic"] = True
    u = _discover_one(_tool_with_provenance(prov))
    assert u["determinism"] == "R3"


def test_discover_file_input_args_surfaced_for_synthesizer():
    u = _discover_one(_tool_with_provenance(_MUSCLE_PROV))
    assert u["provenance_pin"]["mcp_support"]["file_input_args"] == ["input_seqs"]


def test_discover_pure_json_tool_reports_empty_file_inputs():
    prov = {
        "schema": 1,
        "tool_version": "2.0",
        "requirements": [],
        "containers": [],
        "version_command": "",
        "file_input_args": [],
        "stochastic": False,
    }
    u = _discover_one(_tool_with_provenance(prov, name="search", file_param=False))
    # Explicitly empty (a JSON tool), present so the synthesizer knows.
    assert u["provenance_pin"]["mcp_support"]["file_input_args"] == []
    # No container => network side-effects, and unversioned-by-container =>
    # R3 (no reproducibility claim without a pinned binary).
    assert u["side_effects"] == "network"
    assert u["determinism"] == "R3"


def test_discover_unpinned_when_provenance_absent():
    """An old worker (no apecx_provenance) yields an honest @unpinned UTD,
    R3, network — and NO file_input_args (synthesizer must FAIL LOUD)."""
    tool = {
        "name": "legacy_tool",
        "description": "old worker tool",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    }
    u = _discover_one(tool)
    assert u["descriptor_id"] == "rhea:legacy_tool@unpinned"
    assert u["determinism"] == "R3"
    assert "file_input_args" not in u["provenance_pin"]["mcp_support"]
