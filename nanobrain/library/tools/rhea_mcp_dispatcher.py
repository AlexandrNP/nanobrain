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

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import ToolBase, ToolConfig

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
        self._timeout_seconds: float = float(
            dependencies.get("timeout_seconds", 30.0)
        )
        self._extra_headers: Dict[str, str] = dict(
            dependencies.get("extra_headers", {})
        )

        # Lazily-created session state
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        self._client_lock = asyncio.Lock()

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

        await self._ensure_session()
        return await self._dispatch_tool_call(payload, _retry_on_session=True)

    async def aclose(self) -> None:
        """Close the underlying httpx client. Idempotent."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._session_id = None

    # ---- Internals ------------------------------------------------------

    async def _ensure_session(self) -> None:
        """Open the httpx client + perform the MCP initialize handshake.

        Idempotent + asyncio.Lock-serialized so concurrent ``execute``
        calls don't race on session creation.
        """
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
            if self._session_id is not None:
                return

            init_resp = await self._client.post(
                self._mcp_url,
                headers=self._request_headers(include_session=False),
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "nanobrain-rhea-dispatcher",
                            "version": "0.1.0",
                        },
                    },
                },
            )
            if init_resp.status_code != 200:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: MCP initialize at {self._mcp_url} returned "
                    f"{init_resp.status_code}: {init_resp.text[:300]}"
                )
            session_id = init_resp.headers.get("mcp-session-id")
            if not session_id:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: MCP initialize at {self._mcp_url} did not "
                    f"return an 'mcp-session-id' header — server may not be "
                    f"speaking the MCP streamable-HTTP protocol"
                )
            self._session_id = session_id

            # Required initialized notification.
            await self._client.post(
                self._mcp_url,
                headers=self._request_headers(include_session=True),
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                },
            )

    def _request_headers(self, *, include_session: bool) -> Dict[str, str]:
        """Build request headers; optionally include the session id."""
        headers = {
            "Accept": "application/json,text/event-stream",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        if include_session and self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    async def _dispatch_tool_call(
        self, payload: Dict[str, Any], *, _retry_on_session: bool,
    ) -> Any:
        """Send the tools/call JSON-RPC request and parse the response.

        On session-invalid errors (4xx with explicit "session" mention),
        clears the cached session and retries ONCE — protects against
        Rhea-side server restarts mid-session.
        """
        assert self._client is not None  # _ensure_session should set it
        resp = await self._client.post(
            self._mcp_url,
            headers=self._request_headers(include_session=True),
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": self._rhea_tool_name,
                    "arguments": payload,
                },
            },
        )

        if resp.status_code in (400, 401, 404) and _retry_on_session:
            # Session may have expired (server restart). Clear + retry once.
            text_lower = resp.text.lower()
            if "session" in text_lower:
                logger.warning(
                    "MCP session %s invalid (HTTP %d); re-initializing + "
                    "retrying once", self._session_id, resp.status_code,
                )
                self._session_id = None
                await self._ensure_session()
                return await self._dispatch_tool_call(
                    payload, _retry_on_session=False,
                )

        if resp.status_code != 200:
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCP tools/call '{self._rhea_tool_name}' at "
                f"{self._mcp_url} returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        return self._parse_mcp_result(resp.text)

    def _parse_mcp_result(self, body: str) -> Any:
        """Parse an MCP streamable-HTTP response body.

        Format: ``event: message\\ndata: <json>\\n\\n`` — possibly with
        multiple data lines. We extract the FIRST ``data:`` whose payload
        contains a ``result`` field.

        Raises ``ComponentConfigurationError`` if:
        - No parseable ``data:`` line found
        - JSON-RPC ``error`` object returned
        - Result shape is unexpected
        """
        result_payload = None
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue
            if "error" in payload:
                err = payload["error"]
                raise ComponentConfigurationError(
                    f"FAIL-FAST: MCP tools/call '{self._rhea_tool_name}' "
                    f"JSON-RPC error: code={err.get('code')!r} "
                    f"message={err.get('message')!r}"
                )
            if "result" in payload:
                result_payload = payload["result"]
                break

        if result_payload is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCP tools/call '{self._rhea_tool_name}' "
                f"response had no parseable 'data:' line with a result "
                f"field. Body head: {body[:300]!r}"
            )

        # MCP wraps tool output in result.content (list of content items).
        # The shape contract:
        #   {"content": [{"type": "text", "text": "..."}, ...],
        #    "structuredContent": {...},
        #    "isError": bool}
        if not isinstance(result_payload, dict):
            return result_payload

        if result_payload.get("isError"):
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCP tool '{self._rhea_tool_name}' returned "
                f"isError=True. Content: {result_payload.get('content')!r}"
            )

        # Prefer structuredContent if present (MCP spec for typed outputs)
        if "structuredContent" in result_payload:
            return result_payload["structuredContent"]

        content = result_payload.get("content")
        if not isinstance(content, list) or not content:
            return result_payload  # unknown shape — return raw

        # Single text item — try to JSON-parse it for caller convenience
        if len(content) == 1 and isinstance(content[0], dict):
            item = content[0]
            text = item.get("text")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except (TypeError, json.JSONDecodeError):
                    return text
        return content
