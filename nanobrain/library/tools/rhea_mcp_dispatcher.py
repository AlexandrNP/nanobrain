"""RheaMCPDispatcher — ToolBase that dispatches via the MCP HTTP transport.

The consumer side of the cross-framework tool-wrapping contract (T-RH-03):
a ``ToolBase`` subclass that materializes from a UTD with
``provenance_pin.class_path`` pointing at this class, and dispatches
the tool's ``execute`` call as a JSON-RPC ``tools/call`` over MCP's
streamable-HTTP transport.

Wire-format independence: this module does NOT import from rhea/. The
contract is the UTD dict + the MCP protocol; both Rhea-side and
nanobrain-side speak the same wire format and stay decoupled.

## Usage (canonical, end-to-end)

    # 1. Rhea worker is up at http://localhost:3001/mcp/
    # 2. Discover Rhea's tools as UTD dicts
    from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
    rhea_tool_dict = ...  # from rhea.extensions.apecx_utd_extension.utd_producer
    # Override provenance_pin to point at this dispatcher with the
    # Rhea endpoint baked into config_path-equivalent kwargs:
    rhea_tool_dict["provenance_pin"]["class_path"] = (
        "nanobrain.library.tools.rhea_mcp_dispatcher.RheaMCPDispatcher"
    )

    # 3. Materialize via from_descriptor (passes mcp_url + tool_name kwargs)
    from nanobrain.core.tool import ToolBase
    tool = ToolBase.from_descriptor(
        rhea_tool_dict,
        mcp_url="http://localhost:3001/mcp/",
        rhea_tool_name="muscle.align",  # the original Rhea tool name
    )

    # 4. Dispatch via execute (returns the MCP tool's result)
    result = await tool.execute({"sequences": [...], "max_iters": 16})

## Honest scope (T-RH-03 minimum)

- Calls MCP ``tools/call`` synchronously and returns the result. The
  MCP transport is HTTP+SSE; we parse the first ``data:`` line.
- One MCP session per dispatcher instance. The dispatcher caches the
  session_id after the first call; subsequent calls reuse it. When
  the server invalidates the session, the next call re-initializes.
- httpx is the HTTP client. The framework already depends on httpx
  via the bioinformatics tools (G15 follow-on); no new dep.
- FAIL-FAST on:
  * MCP HTTP returning non-200
  * MCP JSON-RPC returning an ``error`` object
  * Response missing the expected ``result.content`` shape
  * Network/connection errors propagate from httpx unchanged

## NOT shipped (deferred)

- Streaming responses (MCP supports server-sent events for long-
  running tools; the dispatcher only handles the simple result-
  in-one-event case for v1).
- Authentication. Rhea's MCP server doesn't currently require auth.
  When it does (Bearer token / mTLS), authors override
  ``_request_headers`` or pass ``extra_headers`` per call.
- Connection pooling tuning. Each dispatcher owns one httpx.AsyncClient.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import ToolBase, ToolConfig
from nanobrain.library.tools._mcp_transport import (
    MCPTransport,
    parse_tool_call_result,
)

logger = logging.getLogger(__name__)


class RheaMCPDispatcher(ToolBase):
    """ToolBase subclass that dispatches a UTD-described tool over MCP HTTP.

    Materialized via ``ToolBase.from_descriptor(utd, mcp_url=..., rhea_tool_name=...)``.
    The MCP URL + tool name are NOT in the UTD itself (they're per-deployment
    plumbing); the descriptor declares the tool's CONTRACT, the dispatcher
    holds the plumbing.
    """

    COMPONENT_TYPE = "rhea_mcp_dispatcher"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return ToolConfig

    def _init_from_config(
        self,
        config: ToolConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        # MCP URL + tool name MUST come via dependencies (kwargs to
        # from_descriptor). FAIL-FAST when missing — the dispatcher
        # cannot dispatch without knowing where to dispatch to.
        mcp_url = dependencies.get("mcp_url")
        if not mcp_url or not isinstance(mcp_url, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDispatcher {config.name!r} requires "
                f"'mcp_url' kwarg (e.g. mcp_url='http://localhost:3001/mcp/'); "
                f"got {mcp_url!r}"
            )
        rhea_tool_name = dependencies.get("rhea_tool_name")
        if not rhea_tool_name or not isinstance(rhea_tool_name, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDispatcher {config.name!r} requires "
                f"'rhea_tool_name' kwarg (the original MCP tool name on "
                f"the Rhea side); got {rhea_tool_name!r}"
            )

        self._mcp_url: str = mcp_url
        self._rhea_tool_name: str = rhea_tool_name

        # MCP wire protocol is delegated to the shared MCPTransport
        # (nanobrain.library.tools._mcp_transport) — single source of
        # truth shared with RheaAdapter + RheaMCPDiscovery.
        self._transport = MCPTransport(
            mcp_url=mcp_url,
            timeout_seconds=float(dependencies.get("timeout_seconds", 30.0)),
            extra_headers=dict(dependencies.get("extra_headers", {})),
            client_name="nanobrain-rhea-dispatcher",
        )

    @classmethod
    def resolve_dependencies(
        cls, component_config: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """Pass through the dispatcher-specific kwargs from from_config.

        Mirrors the WorkflowEntryTrigger pattern: the framework's standard
        resolve_dependencies doesn't know about mcp_url / rhea_tool_name,
        so we surface them explicitly.
        """
        return {
            "mcp_url": kwargs.get("mcp_url"),
            "rhea_tool_name": kwargs.get("rhea_tool_name"),
            "timeout_seconds": kwargs.get("timeout_seconds", 30.0),
            "extra_headers": kwargs.get("extra_headers", {}),
        }

    # ---- Public API -----------------------------------------------------

    async def execute(self, payload: Dict[str, Any]) -> Any:
        """Dispatch ``payload`` to the Rhea MCP tool via JSON-RPC.

        Args:
            payload: Dict matching the MCP tool's ``inputSchema``.

        Returns:
            The MCP tool's ``result.content[0]`` payload (text or
            structured), parsed from JSON when possible. For tools that
            return multiple ``content`` items, the full list is returned.

        Raises:
            ComponentConfigurationError: on MCP HTTP non-200, JSON-RPC
                error response, malformed result shape, or session
                exhaustion.
            httpx.HTTPError: on network/transport failure.
        """
        if not isinstance(payload, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDispatcher.execute payload must be a "
                f"dict matching the MCP tool's inputSchema; got "
                f"{type(payload).__name__}"
            )

        raw = await self._transport.call(
            "tools/call",
            {"name": self._rhea_tool_name, "arguments": payload},
        )
        return parse_tool_call_result(raw, self._rhea_tool_name)

    async def aclose(self) -> None:
        """Close the underlying MCP transport. Idempotent."""
        await self._transport.aclose()


__all__ = ["RheaMCPDispatcher"]
