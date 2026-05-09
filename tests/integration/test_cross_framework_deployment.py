"""Cross-framework end-to-end deployment + tool-wrapping integration test.

The most important test in the deployment-validation chain. Exercises:

1. **Live Rhea worker container** — discovered via the
   ``RHEA_MCP_URL`` env var (skipped without it).
2. **MCP discovery** — JSON-RPC ``tools/list`` against the live Rhea
   server returns its tool catalogue.
3. **Wire-format conversion** — each MCP tool is converted to a UTD
   dict using the same logic that ships in
   ``rhea/extensions/apecx_utd_extension/utd_producer.py``. The dict
   is the cross-framework contract; both sides agree on the shape.
4. **Nanobrain UTD validation** — the dict feeds
   ``UnifiedToolDescriptor.from_dict`` and produces a validated UTD
   with a stable descriptor_hash.
5. **Academy in the same workflow** — a separate Academy agent
   (spawned + scraped + utilized via the patterns validated in
   ``test_academy_in_workflow.py``) runs alongside the Rhea
   discovery, both inside ONE WorkflowRunner.run_detached invocation.
6. **End-to-end through the WorkflowRunner** — both halves run as
   one detached task; the runner confirms completion.

This is the canonical "does cross-framework deployment actually
work" validation. Pre-this-chain, the discovery → UTD → workflow
seam was unproven end-to-end.

## Skip semantics

- ``RHEA_MCP_URL`` env var unset → SKIPPED (Rhea not running).
- ``RHEA_MCP_URL`` set but unreachable → test FAILS (operator
  error, not silent skip).

## Local-dev recipe

    docker network create rhea-net 2>/dev/null
    docker run --rm -d --name rhea-redis --network rhea-net redis:7
    docker build -t rhea-server:apecx-integration <rhea-checkout>
    docker run --rm -d --name rhea-server-test \\
        --network rhea-net -p 3001:3001 \\
        -e REDIS_HOST=rhea-redis -e AGENT_REDIS_HOST=rhea-redis \\
        -e HOST=0.0.0.0 \\
        rhea-server:apecx-integration \\
        uv run -m rhea.server.mcp_server --transport streamable-http
    RHEA_MCP_URL=http://localhost:3001/mcp/ \\
      .venv/bin/python -m pytest \\
      nanobrain/tests/integration/test_cross_framework_deployment.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pytest
import yaml

pytestmark = pytest.mark.integration


_RHEA_MCP_URL = os.environ.get("RHEA_MCP_URL")
_rhea_skip = pytest.mark.skipif(
    _RHEA_MCP_URL is None,
    reason="RHEA_MCP_URL env var not set; bring up Rhea + set the var "
           "to enable the cross-framework integration tests "
           "(see file docstring for the local-dev recipe).",
)


# ---------------------------------------------------------------------------
# MCP client helper — minimal JSON-RPC over the streamable-HTTP transport
# ---------------------------------------------------------------------------

class _MCPClient:
    """Minimal MCP HTTP client. Handles the session-id handshake,
    the post-initialize notification, and JSON-RPC request/response."""

    def __init__(self, base_url: str) -> None:
        self._url = base_url
        self._session_id: Optional[str] = None
        self._client = httpx.AsyncClient(timeout=15.0)

    async def __aenter__(self) -> "_MCPClient":
        await self._initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def _initialize(self) -> None:
        resp = await self._client.post(
            self._url,
            headers={
                "Accept": "application/json,text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nb-cf-test", "version": "0.1"},
                },
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"FAIL-FAST: Rhea MCP initialize returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        self._session_id = resp.headers.get("mcp-session-id")
        if not self._session_id:
            raise RuntimeError(
                "FAIL-FAST: Rhea MCP initialize response missing "
                "mcp-session-id header"
            )
        # Required initialized notification
        await self._client.post(
            self._url,
            headers={
                "Accept": "application/json,text/event-stream",
                "Content-Type": "application/json",
                "mcp-session-id": self._session_id,
            },
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    async def list_tools(self) -> List[Dict[str, Any]]:
        resp = await self._client.post(
            self._url,
            headers={
                "Accept": "application/json,text/event-stream",
                "Content-Type": "application/json",
                "mcp-session-id": self._session_id,  # type: ignore[arg-type]
            },
            json={
                "jsonrpc": "2.0", "id": 2,
                "method": "tools/list", "params": {},
            },
        )
        # The streamable-HTTP transport returns SSE-formatted bodies:
        # ``event: message\ndata: <json>\n\n``. Parse the first data line.
        text = resp.text
        for line in text.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                if "result" in payload:
                    return payload["result"].get("tools", [])
        raise RuntimeError(
            f"FAIL-FAST: Rhea MCP tools/list returned no parseable data: "
            f"{text[:300]}"
        )


# ---------------------------------------------------------------------------
# Rhea → UTD dict producer (mirror of utd_producer.py for test isolation)
# ---------------------------------------------------------------------------

_TOOL_ID_RE = re.compile(r"[^a-z0-9_.]")


def _sanitize_tool_id(raw: str) -> str:
    sanitized = _TOOL_ID_RE.sub("_", raw.lower())
    if not sanitized or not sanitized[0].isalpha():
        sanitized = "rhea_" + sanitized.lstrip("_.")
    return sanitized


def _input_specs_from_json_schema(schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return [{"name": "input", "type": "Any", "description": "",
                 "required": True, "default": None}]
    properties = schema.get("properties") or {}
    required_set = set(schema.get("required") or [])
    specs = []
    for pname, pschema in properties.items():
        if not isinstance(pschema, dict):
            continue
        specs.append({
            "name": pname,
            "type": pschema.get("type", "Any"),
            "description": pschema.get("description", ""),
            "required": pname in required_set,
            "default": pschema.get("default"),
        })
    return specs


def mcp_tool_to_utd_dict(
    mcp_tool: Dict[str, Any], *, backend: str = "rhea", version: str = "1.0.0",
) -> Dict[str, Any]:
    """Convert an MCP Tool dict (as returned by tools/list) to a
    UTD-shaped dict consumable by UnifiedToolDescriptor.from_dict."""
    name = mcp_tool["name"]
    title = mcp_tool.get("title") or name
    description = (mcp_tool.get("description") or "").strip()
    parts = description.split("\n", 1) if description else [title, ""]
    summary = parts[0].strip() or title
    long_description = parts[1].strip() if len(parts) > 1 else ""

    output_schema = mcp_tool.get("outputSchema") or {}
    output_type = output_schema.get("type", "Any") if isinstance(output_schema, dict) else "Any"

    return {
        "descriptor_id": f"{backend}:{_sanitize_tool_id(name)}@{version}",
        "display_name": title,
        "summary": summary,
        "long_description": long_description,
        "inputs": _input_specs_from_json_schema(mcp_tool.get("inputSchema")),
        "outputs": [{"name": "return", "type": output_type, "description": ""}],
        "side_effects": "none",
        "determinism": "R3",
        "resource_class": "cpu_light",
        "provenance_pin": {
            "class_path": "rhea.extensions.apecx_utd_extension.dispatchers.RheaMCPDispatcher",
        },
    }


# ---------------------------------------------------------------------------
# Academy fixture (Academy is process-local; uses LocalExchangeFactory)
# ---------------------------------------------------------------------------

from academy.agent import Agent, action  # noqa: E402


class _SummarizerAgent(Agent):
    """Trivial Academy agent — produces a one-line summary of a list."""

    @action
    async def summarize(self, items: List[str]) -> str:
        return f"summarized {len(items)} items: " + ", ".join(items[:3])


# ---------------------------------------------------------------------------
# 1. Pure wire-format test — runs without a live Rhea server
# ---------------------------------------------------------------------------

class TestWireFormatRoundtrip:

    def test_synthetic_mcp_tool_validates_as_utd(self):
        """Cross-framework wire format works without a live Rhea — we
        feed a synthetic MCP-shaped dict into the producer and validate."""
        from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
        synthetic = {
            "name": "muscle.align",
            "title": "MUSCLE Align",
            "description": "Align protein sequences with MUSCLE.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sequences": {"type": "array",
                                  "description": "FASTA input"},
                    "max_iters": {"type": "integer", "default": 16},
                },
                "required": ["sequences"],
            },
            "outputSchema": {"type": "string"},
        }
        utd_dict = mcp_tool_to_utd_dict(synthetic, backend="rhea",
                                         version="5.1.0")
        utd = UnifiedToolDescriptor.from_dict(utd_dict)
        assert utd.descriptor_id == "rhea:muscle.align@5.1.0"
        assert utd.descriptor_hash  # non-empty SHA-256
        names = [i.name for i in utd.inputs]
        assert names == ["sequences", "max_iters"]
        assert utd.inputs[0].required is True
        assert utd.inputs[1].required is False
        assert utd.inputs[1].default == 16


# ---------------------------------------------------------------------------
# 2. Live Rhea — MCP discovery + UTD validation
# ---------------------------------------------------------------------------

@_rhea_skip
class TestLiveRheaDiscovery:

    def test_list_tools_returns_at_least_one(self):
        """Bare-MCP discovery: Rhea must expose at least the find_tools
        meta-tool by default."""
        async def run():
            async with _MCPClient(_RHEA_MCP_URL) as cli:
                tools = await cli.list_tools()
                assert len(tools) >= 1, f"no tools returned; got {tools!r}"
                names = [t["name"] for t in tools]
                assert "find_tools" in names, names
        asyncio.run(run())

    def test_every_tool_converts_to_valid_utd(self):
        """Every tool Rhea exposes must round-trip through the producer
        + nanobrain UTD validator. If even ONE tool's schema breaks the
        converter, the whole catalogue is unusable from the apecx side."""
        from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor

        async def run():
            async with _MCPClient(_RHEA_MCP_URL) as cli:
                tools = await cli.list_tools()
                for t in tools:
                    utd_dict = mcp_tool_to_utd_dict(
                        t, backend="rhea", version="1.10.1",
                    )
                    utd = UnifiedToolDescriptor.from_dict(utd_dict)
                    # Each UTD has a stable hash
                    assert utd.descriptor_hash
                    # And a parseable descriptor_id
                    assert utd.descriptor_backend == "rhea"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3. Mixed deployment — Rhea discovery + Academy step in the same workflow,
#    driven by WorkflowRunner.run_detached
# ---------------------------------------------------------------------------

@_rhea_skip
class TestCrossFrameworkWorkflow:

    def test_workflow_does_rhea_discovery_and_academy_dispatch(self):
        """The canonical end-to-end shape: ONE detached workflow
        invocation that:
        (a) discovers Rhea tools as UTDs,
        (b) spawns + utilizes an Academy agent,
        (c) returns a unified result dict.
        """
        from nanobrain.core.academy_integration import (
            AcademyIntegration,
            shutdown_academy_manager,
        )
        from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
        from nanobrain.library.runtime import WorkflowRunner

        async def cross_framework_workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
            # PHASE 1 — Rhea discovery
            async with _MCPClient(payload["rhea_url"]) as cli:
                mcp_tools = await cli.list_tools()
            utd_dicts = [
                mcp_tool_to_utd_dict(t, backend="rhea", version="1.10.1")
                for t in mcp_tools
            ]
            utds = [UnifiedToolDescriptor.from_dict(d) for d in utd_dicts]
            tool_names = [u.descriptor_tool_id for u in utds]

            # PHASE 2 — Academy dispatch
            await shutdown_academy_manager()  # ensure clean
            mgr = AcademyIntegration.setup_academy_manager()
            try:
                handle = await mgr.register_agent_class(
                    "summarizer", _SummarizerAgent,
                )
                summary = await handle.summarize(tool_names)
            finally:
                await shutdown_academy_manager()

            return {
                "rhea_tool_count": len(utds),
                "rhea_tool_descriptor_ids": [u.descriptor_id for u in utds],
                "academy_summary": summary,
            }

        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                yml = tmp / "runner.yml"
                yml.write_text(yaml.safe_dump({
                    "name": "cross_runner",
                    "task_store_backend": "in_memory",
                    "heartbeat_interval_seconds": 0,
                }))
                runner = WorkflowRunner.from_config(str(yml))
                await runner.run_detached(
                    cross_framework_workflow,
                    "cross_task",
                    {"rhea_url": _RHEA_MCP_URL},
                )
                final = await runner.await_completion("cross_task", timeout=30)
                assert final.status == "completed", final
                result = final.result
                assert result["rhea_tool_count"] >= 1
                assert all(
                    did.startswith("rhea:") for did in result["rhea_tool_descriptor_ids"]
                )
                assert "summarized" in result["academy_summary"]
        asyncio.run(run())
