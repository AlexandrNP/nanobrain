"""Unit tests for synthesize_rhea_step (E2-R Priority 1).

Runs UNCONDITIONALLY against a fake MCP server (httpx MockTransport).
The live-worker path (synthesize from a REAL Galaxy tool + run it) is the
gated integration test
``tests/integration/test_rhea_synthesize_step_live.py`` (skipped unless
``$RHEA_MCP_URL`` is set).

Contract assertions:
* A pure-JSON tool synthesizes a ToolExecutionStep (backend=rhea).
* A file (Galaxy type="data") tool synthesizes a RheaFileToolStep with
  the correct file_input_arg.
* The file-vs-JSON branch uses the worker's AUTHORITATIVE file_input_args
  discriminator — and FAILS LOUD (never guesses) when the worker did not
  surface it and the caller gave no override.
* An unknown tool name FAILS LOUD; a multi-file tool FAILS LOUD.
* The synthesized spec drops into a WorkflowBuilder DAG.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.tools.rhea_discovery import RheaMCPDiscovery
from nanobrain.library.tools.rhea_step_synthesizer import (
    RheaStepSpec,
    synthesize_rhea_step,
)


def _sse(obj: dict) -> str:
    return "event: message\ndata: " + json.dumps(obj) + "\n"


def _make_handler(tools: list[dict], *, record: dict | None = None):
    """Fake Rhea MCP server: initialize, find_tools (tools/call), tools/list."""

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
            name = body.get("params", {}).get("name")
            if record is not None and name == "find_tools":
                record["find_tools_called"] = True
                record["find_tools_query"] = (
                    body["params"].get("arguments", {}).get("query")
                )
            return httpx.Response(
                200,
                text=_sse(
                    {
                        "jsonrpc": "2.0",
                        "id": 9,
                        "result": {
                            "content": [{"type": "text", "text": "ok"}],
                            "isError": False,
                        },
                    }
                ),
            )
        if method == "tools/list":
            return httpx.Response(
                200,
                text=_sse(
                    {"jsonrpc": "2.0", "id": 2, "result": {"tools": tools}}
                ),
            )
        return httpx.Response(400, text="unexpected method")

    return handler


def _patch_discovery(monkeypatch, tools, record=None):
    """Make every RheaMCPDiscovery use the fake MCP transport."""
    real_init = RheaMCPDiscovery.__init__

    def fake_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        self.transport.client = httpx.AsyncClient(
            transport=httpx.MockTransport(_make_handler(tools, record=record)),
            timeout=5.0,
        )

    monkeypatch.setattr(RheaMCPDiscovery, "__init__", fake_init)


def _file_tool(name="muscle"):
    return {
        "name": name,
        "title": name.upper(),
        "description": "Multiple sequence alignment",
        "inputSchema": {
            "type": "object",
            "properties": {"input_seqs": {"type": "string"}},
            "required": ["input_seqs"],
        },
        "annotations": {
            "apecx_provenance": {
                "schema": 1,
                "tool_version": "5.1.0",
                "requirements": [],
                "containers": [{"type": "docker", "value": "muscle:5.1"}],
                "version_command": "muscle -version",
                "file_input_args": ["input_seqs"],
                "stochastic": False,
            }
        },
    }


def _json_tool(name="uniprot_search"):
    return {
        "name": name,
        "title": name,
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


def _no_prov_tool(name="legacy"):
    return {
        "name": name,
        "description": "old-worker tool, no apecx_provenance",
        "inputSchema": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [],
        },
    }


def test_synthesize_json_tool_makes_tool_execution_step(monkeypatch):
    _patch_discovery(monkeypatch, [_json_tool()])
    spec = asyncio.run(
        synthesize_rhea_step("uniprot_search", mcp_url="http://fake/mcp/")
    )
    assert isinstance(spec, RheaStepSpec)
    assert spec.uses_file_input is False
    assert spec.step_class.endswith("ToolExecutionStep")
    assert spec.step_config["tool_descriptor"]["descriptor_id"] == "rhea:uniprot_search@2.0"
    assert spec.is_pinned is True


def _muscle_like_tool(name="muscle"):
    """A file tool that ALSO declares required + optional non-file params.

    Mirrors the live muscle inputSchema: ``input_seqs`` is the file param;
    ``diags`` is REQUIRED with NO default (the param the synthesizer used to
    drop, causing ``muscleArguments: diags Field required``); ``cluster`` /
    ``run`` are non-file params WITH schema defaults; ``extra`` is optional
    with no default.
    """
    return {
        "name": name,
        "title": name.upper(),
        "description": "Multiple sequence alignment",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_seqs": {"type": "string"},
                "diags": {"type": "boolean"},
                "cluster": {"type": "string", "default": "upgmb"},
                "run": {"type": "string", "default": "16"},
                "extra": {"type": "string"},
            },
            "required": ["input_seqs", "diags"],
        },
        "annotations": {
            "apecx_provenance": {
                "schema": 1,
                "tool_version": "3.8.1551+galaxy0",
                "requirements": [],
                "containers": [{"type": "docker", "value": "muscle:3.8"}],
                "version_command": "muscle -version",
                "file_input_args": ["input_seqs"],
                "stochastic": False,
            }
        },
    }


def test_value_params_populated_from_schema_defaults(monkeypatch):
    """Non-file params WITH defaults land in static_tool_args; the required-
    no-default param is supplied via override → no FAIL LOUD."""
    _patch_discovery(monkeypatch, [_muscle_like_tool()])
    spec = asyncio.run(
        synthesize_rhea_step(
            "muscle",
            mcp_url="http://fake/mcp/",
            static_tool_args={"diags": False},
        )
    )
    args = spec.step_config["static_tool_args"]
    assert args["cluster"] == "upgmb"  # schema default
    assert args["run"] == "16"  # schema default
    assert args["diags"] is False  # caller override (required, no default)
    assert "extra" not in args  # optional, no default → omitted
    assert "input_seqs" not in args  # file param → staged, never in static args


def test_required_no_default_param_fails_loud(monkeypatch):
    """A required non-file param with no default and no override is the bug
    this fix targets — it must FAIL LOUD, never be silently omitted."""
    _patch_discovery(monkeypatch, [_muscle_like_tool()])
    with pytest.raises(
        ComponentConfigurationError, match="cannot map required value param"
    ) as exc:
        asyncio.run(synthesize_rhea_step("muscle", mcp_url="http://fake/mcp/"))
    assert "diags" in str(exc.value)


def test_caller_override_beats_schema_default(monkeypatch):
    """An override for a defaulted param wins over the schema default."""
    _patch_discovery(monkeypatch, [_muscle_like_tool()])
    spec = asyncio.run(
        synthesize_rhea_step(
            "muscle",
            mcp_url="http://fake/mcp/",
            static_tool_args={"diags": True, "cluster": "neighborjoining"},
        )
    )
    args = spec.step_config["static_tool_args"]
    assert args["cluster"] == "neighborjoining"  # override beat default "upgmb"
    assert args["diags"] is True
    assert args["run"] == "16"  # untouched default


def test_synthesize_file_tool_makes_rhea_file_tool_step(monkeypatch):
    _patch_discovery(monkeypatch, [_file_tool()])
    spec = asyncio.run(
        synthesize_rhea_step(
            "muscle", mcp_url="http://fake/mcp/", find_tools_query="align"
        )
    )
    assert spec.uses_file_input is True
    assert spec.step_class.endswith("RheaFileToolStep")
    assert spec.step_config["tool_name"] == "muscle"
    assert spec.step_config["file_input_arg"] == "input_seqs"
    assert spec.step_config["find_tools_query"] == "align"
    assert spec.step_config["mcp_url"] == "http://fake/mcp/"


def test_find_tools_called_when_query_given(monkeypatch):
    record: dict = {}
    _patch_discovery(monkeypatch, [_file_tool()], record=record)
    asyncio.run(
        synthesize_rhea_step(
            "muscle", mcp_url="http://fake/mcp/", find_tools_query="align proteins"
        )
    )
    assert record.get("find_tools_called") is True
    assert record.get("find_tools_query") == "align proteins"


def test_unknown_tool_fails_loud(monkeypatch):
    _patch_discovery(monkeypatch, [_json_tool()])
    with pytest.raises(ComponentConfigurationError, match="could not find a Rhea tool"):
        asyncio.run(synthesize_rhea_step("nonexistent", mcp_url="http://fake/mcp/"))


def test_unknown_file_vs_json_fails_loud(monkeypatch):
    """Old worker (no file_input_args) + no override => FAIL LOUD, no guess."""
    _patch_discovery(monkeypatch, [_no_prov_tool()])
    with pytest.raises(ComponentConfigurationError, match="cannot determine whether"):
        asyncio.run(synthesize_rhea_step("legacy", mcp_url="http://fake/mcp/"))


def test_explicit_override_forces_json_path(monkeypatch):
    """Caller override resolves the ambiguity for an old worker."""
    _patch_discovery(monkeypatch, [_no_prov_tool()])
    spec = asyncio.run(
        synthesize_rhea_step(
            "legacy", mcp_url="http://fake/mcp/", file_input_args=[]
        )
    )
    assert spec.uses_file_input is False
    assert spec.step_class.endswith("ToolExecutionStep")


def test_explicit_override_forces_file_path(monkeypatch):
    _patch_discovery(monkeypatch, [_no_prov_tool()])
    spec = asyncio.run(
        synthesize_rhea_step(
            "legacy", mcp_url="http://fake/mcp/", file_input_args=["x"]
        )
    )
    assert spec.uses_file_input is True
    assert spec.step_config["file_input_arg"] == "x"


def test_multi_file_tool_fails_loud(monkeypatch):
    tool = _file_tool()
    tool["annotations"]["apecx_provenance"]["file_input_args"] = ["a", "b"]
    _patch_discovery(monkeypatch, [tool])
    with pytest.raises(ComponentConfigurationError, match="multiple file"):
        asyncio.run(synthesize_rhea_step("muscle", mcp_url="http://fake/mcp/"))


def test_unpinned_tool_spec_is_not_pinned(monkeypatch):
    _patch_discovery(monkeypatch, [_no_prov_tool()])
    spec = asyncio.run(
        synthesize_rhea_step("legacy", mcp_url="http://fake/mcp/", file_input_args=[])
    )
    assert spec.descriptor_id == "rhea:legacy@unpinned"
    assert spec.is_pinned is False


def test_synthesized_step_drops_into_workflow_builder(monkeypatch):
    """The synthesized spec must produce a valid WorkflowBuilder step entry."""
    from nanobrain.lightweight.workflow_builder import WorkflowBuilder

    _patch_discovery(monkeypatch, [_json_tool()])
    spec = asyncio.run(
        synthesize_rhea_step("uniprot_search", mcp_url="http://fake/mcp/")
    )
    builder = WorkflowBuilder("rhea_wf", "synthesized")
    builder.add_rhea_step("search", spec)
    cfg = builder.get_config()
    assert "search" in cfg["steps"]
    entry = cfg["steps"]["search"]
    assert entry["class"].endswith("ToolExecutionStep")
    assert entry["tool_descriptor"]["descriptor_id"] == "rhea:uniprot_search@2.0"


def test_add_rhea_step_rejects_non_spec():
    from nanobrain.lightweight.workflow_builder import WorkflowBuilder

    builder = WorkflowBuilder("rhea_wf", "")
    with pytest.raises(ValueError, match="expects a RheaStepSpec"):
        builder.add_rhea_step("x", {"not": "a spec"})
