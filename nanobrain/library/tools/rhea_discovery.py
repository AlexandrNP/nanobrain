"""RheaMCPDiscovery — discover a Rhea MCP server's tool catalog as UTDs.

A thin MCP client that connects to a running Rhea worker, calls
``tools/list``, and converts each discovered MCP tool into a
``UnifiedToolDescriptor``-shaped dict. The intended consumer is a
code generator / workflow composer: discover Rhea's tools at
generation time, then wire the matching tool into a workflow via
``ToolExecutionStep`` + the ``RheaAdapter`` backend.

This is the "code generator uses Rhea as an MCP server" mechanism —
the generator is itself an MCP client of Rhea.

Why the wire shape, not Rhea's in-process objects
-------------------------------------------------

The Rhea fork ships ``rhea.extensions.apecx_utd_extension.utd_producer``
which converts Rhea's in-process ``Tool`` objects to UTD dicts. This
module does the equivalent from the *wire* shape (the MCP
``tools/list`` response) instead — so it has zero ``rhea/`` import
dependency. The MCP ``tools/list`` JSON is the cross-framework
contract; both sides agree on it. Use this module when you only have
the MCP endpoint, not the Rhea process.

Honest scope (v1)
-----------------

* ``tools/list`` only (no ``resources/list`` / ``prompts/list``).
* The generated UTD is minimal-but-valid: ``descriptor_id`` =
  ``rhea:<sanitized_tool_id>@<version>``, ``inputs`` derived from the
  MCP tool's ``inputSchema.properties``, a single ``return`` output
  (MCP ``tools/list`` carries no output schema in the base spec).
* FAIL-FAST on: ``$RHEA_MCP_URL`` unset, empty tool catalog, missing
  ``tools`` array, and anything ``MCPTransport`` raises. An empty
  catalog is a LOUD error — a generator handed zero tools would emit
  an empty/no-op workflow.
* Shares the MCP wire protocol with ``RheaAdapter`` +
  ``RheaMCPDispatcher`` via ``MCPTransport``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.tools._mcp_transport import MCPTransport

_RHEA_ENV_VAR = "RHEA_MCP_URL"
_RHEA_ADAPTER_CLASS_PATH = "nanobrain.library.tools.rhea_adapter.RheaAdapter"
# UTD tool_id grammar: [a-z][a-z0-9_.]* — sanitize MCP names to fit.
_TOOL_ID_SANITIZE = re.compile(r"[^a-z0-9_.]")


class RheaMCPDiscovery:
    """Connects to a Rhea MCP server and discovers its tools as UTD dicts."""

    def __init__(
        self,
        *,
        mcp_url: str,
        timeout_seconds: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        default_tool_version: str = "1.0.0",
    ) -> None:
        if not mcp_url or not isinstance(mcp_url, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDiscovery requires a non-empty mcp_url; "
                f"got {mcp_url!r}"
            )
        self._default_tool_version = default_tool_version
        self._transport = MCPTransport(
            mcp_url=mcp_url,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            client_name="nanobrain-rhea-discovery",
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> RheaMCPDiscovery:
        """Build from ``$RHEA_MCP_URL``; FAIL-FAST when unset."""
        url = os.environ.get(_RHEA_ENV_VAR)
        if not url:
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDiscovery.from_env requires ${_RHEA_ENV_VAR} "
                f"(e.g. export {_RHEA_ENV_VAR}='http://localhost:3001/mcp/'). "
                f"The Rhea worker must be running and reachable."
            )
        return cls(mcp_url=url, **kwargs)

    # Test seam — swap in an httpx.MockTransport-backed client.
    @property
    def transport(self) -> MCPTransport:
        return self._transport

    async def discover(self) -> list[dict[str, Any]]:
        """Return a list of UTD-shaped dicts for every tool Rhea exposes.

        FAIL-FAST if Rhea reports zero tools — a generator handed an
        empty tool catalog would emit an empty/no-op workflow.
        """
        result = await self._transport.call("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        if tools is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCP tools/list result had no 'tools' array: "
                f"{result!r}"
            )
        if not tools:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Rhea MCP server at {self._transport.mcp_url} "
                f"returned an empty tools list. A code generator cannot "
                f"compose a workflow with zero tools. Check that the Rhea "
                f"worker is loaded with the expected tool catalog."
            )
        return [self._mcp_tool_to_utd(t) for t in tools]

    async def aclose(self) -> None:
        await self._transport.aclose()

    # ---- MCP tool -> UTD conversion --------------------------------------

    def _mcp_tool_to_utd(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert one MCP ``tools/list`` entry into a UTD dict."""
        raw_name = str(tool.get("name", "")).strip()
        if not raw_name:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Rhea MCP tool entry has no 'name': {tool!r}"
            )
        tool_id = self._sanitize_tool_id(raw_name)
        descriptor_id = f"rhea:{tool_id}@{self._default_tool_version}"

        input_schema = tool.get("inputSchema") or {}
        properties = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or [])
        inputs = []
        for prop_name, prop_spec in properties.items():
            inputs.append(
                {
                    "name": prop_name,
                    "type": _json_type_to_utd_type(prop_spec.get("type")),
                    "description": str(prop_spec.get("description", "")),
                    "required": prop_name in required,
                    "default": prop_spec.get("default"),
                }
            )

        return {
            "descriptor_id": descriptor_id,
            "display_name": tool.get("title") or raw_name,
            "summary": str(tool.get("description", "") or f"Rhea tool {raw_name}")[:200],
            "long_description": str(tool.get("description", "") or ""),
            "inputs": inputs,
            "outputs": [
                {
                    "name": "return",
                    "type": "object",
                    "description": "MCP tool result (Rhea base spec carries no output schema).",
                }
            ],
            # Rhea bio tools hit external APIs (UniProt, PubMed, NCBI, ...);
            # 'network' is the honest side-effect class.
            "side_effects": "network",
            "determinism": "R3",
            "resource_class": "cpu_light",
            "provenance_pin": {
                "class_path": _RHEA_ADAPTER_CLASS_PATH,
                # mcp_support carries MCP plumbing. The MCP-side tool name
                # may differ from the sanitized UTD tool_id;
                # RheaAdapter._resolve_tool_name reads this override.
                "mcp_support": {
                    "rhea_tool_name": raw_name,
                    "discovered_from": self._transport.mcp_url,
                },
            },
        }

    @staticmethod
    def _sanitize_tool_id(raw_name: str) -> str:
        """Coerce an MCP tool name into the UTD tool_id grammar
        ``[a-z][a-z0-9_.]*``."""
        lowered = raw_name.lower()
        sanitized = _TOOL_ID_SANITIZE.sub("_", lowered)
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "t_" + sanitized
        return sanitized


def _json_type_to_utd_type(json_type: Any) -> str:
    """Map a JSON-schema type to the UTD type vocabulary."""
    mapping = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    if isinstance(json_type, str):
        return mapping.get(json_type, "object")
    return "object"


__all__ = ["RheaMCPDiscovery"]
