"""Shared MCP streamable-HTTP transport for Rhea-facing components.

Single source of truth for the MCP streamable-HTTP wire protocol:
the ``initialize`` handshake, the ``notifications/initialized``
follow-up, JSON-RPC ``tools/call`` / ``tools/list`` dispatch, the
``mcp-session-id`` header lifecycle (with one-shot re-init on
server-restart), and the SSE ``data:`` line parse.

Before this module, the ~90 lines of MCP wire logic were duplicated
across three components:

* ``RheaMCPDispatcher`` (a ``ToolBase`` — the Agent-facing path)
* ``RheaAdapter`` (a ``ToolBackendAdapter`` — the ToolExecutionStep path)
* ``RheaMCPDiscovery`` (an MCP ``tools/list`` client)

All three now consume :class:`MCPTransport`. A bug fix or protocol
update lands once.

Honest scope (v1)
-----------------

* Streamable-HTTP only. The first ``data:`` SSE line carrying a
  ``result`` is taken; multi-event / long-running streaming responses
  are NOT handled (deferred — same scope cut the components carried
  individually).
* One ``httpx.AsyncClient`` + one MCP session per transport instance,
  ``asyncio.Lock``-serialized so concurrent callers don't race on
  session creation.
* FAIL-FAST (``ComponentConfigurationError``) on: non-200 HTTP,
  JSON-RPC ``error`` object, missing ``mcp-session-id`` header, no
  parseable ``data:`` line with a ``result``. Network/transport
  failures propagate as ``httpx`` errors unchanged.
* Session-invalid recovery: a 4xx whose body mentions "session"
  clears the cached session and retries the call ONCE (covers a Rhea
  worker restarting mid-session).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from nanobrain.core.component_base import ComponentConfigurationError

logger = logging.getLogger(__name__)

_MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPTransport:
    """An MCP streamable-HTTP session against one endpoint.

    Construct one per ``mcp_url``. Call :meth:`call` for JSON-RPC
    dispatch — it lazily performs the ``initialize`` handshake on
    first use and reuses the session afterwards.
    """

    def __init__(
        self,
        *,
        mcp_url: str,
        timeout_seconds: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        client_name: str = "nanobrain-mcp-transport",
        client_version: str = "0.1.0",
    ) -> None:
        if not mcp_url or not isinstance(mcp_url, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCPTransport requires a non-empty mcp_url "
                f"string; got {mcp_url!r}"
            )
        self._mcp_url = mcp_url
        self._timeout_seconds = float(timeout_seconds)
        self._extra_headers = dict(extra_headers or {})
        self._client_name = client_name
        self._client_version = client_version
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def mcp_url(self) -> str:
        return self._mcp_url

    # Test seam: callers (and their test suites) swap in an
    # httpx.MockTransport-backed client by setting ``transport.client``.
    @property
    def client(self) -> httpx.AsyncClient | None:
        return self._client

    @client.setter
    def client(self, value: httpx.AsyncClient | None) -> None:
        self._client = value

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: int = 2,
        _retry_on_session: bool = True,
    ) -> Any:
        """JSON-RPC ``method`` dispatch. Returns the raw ``result`` object.

        The caller is responsible for interpreting the ``result``
        shape (``tools/call`` → ``{content, structuredContent,
        isError}``; ``tools/list`` → ``{tools: [...]}``). Use
        :func:`parse_tool_call_result` for the ``tools/call`` shape.
        """
        await self._ensure_session()
        assert self._client is not None
        resp = await self._client.post(
            self._mcp_url,
            headers=self._headers(include_session=True),
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            },
        )
        if resp.status_code in (400, 401, 404) and _retry_on_session:
            if "session" in resp.text.lower():
                logger.warning(
                    "MCP session %s invalid (HTTP %d) on %s; re-initializing "
                    "+ retrying once",
                    self._session_id,
                    resp.status_code,
                    method,
                )
                self._session_id = None
                await self._ensure_session()
                return await self.call(
                    method, params, request_id=request_id, _retry_on_session=False
                )
        if resp.status_code != 200:
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCP {method} at {self._mcp_url} returned HTTP "
                f"{resp.status_code}: {resp.text[:300]}"
            )
        return self._parse_sse_result(resp.text, method)

    async def aclose(self) -> None:
        """Close the underlying httpx client. Idempotent."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._session_id = None

    # ---- internals ------------------------------------------------------

    async def _ensure_session(self) -> None:
        """Lazily open the client + perform the MCP initialize handshake.

        Idempotent + Lock-serialized so concurrent ``call`` invocations
        don't race on session creation.
        """
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
            if self._session_id is not None:
                return
            init_resp = await self._client.post(
                self._mcp_url,
                headers=self._headers(include_session=False),
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": _MCP_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {
                            "name": self._client_name,
                            "version": self._client_version,
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
                    f"return an 'mcp-session-id' header — the server may not "
                    f"be speaking the MCP streamable-HTTP protocol."
                )
            self._session_id = session_id
            # Required initialized notification.
            await self._client.post(
                self._mcp_url,
                headers=self._headers(include_session=True),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )

    def _headers(self, *, include_session: bool) -> dict[str, str]:
        headers = {
            "Accept": "application/json,text/event-stream",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        if include_session and self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    @staticmethod
    def _parse_sse_result(body: str, method: str) -> Any:
        """Extract the first ``data:`` SSE line carrying a JSON-RPC result.

        Raises ``ComponentConfigurationError`` on a JSON-RPC ``error``
        object or when no parseable ``data:`` line with a ``result``
        field is found.
        """
        for line in body.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                continue
            if "error" in payload:
                err = payload["error"]
                raise ComponentConfigurationError(
                    f"FAIL-FAST: MCP {method} JSON-RPC error: "
                    f"code={err.get('code')!r} message={err.get('message')!r}"
                )
            if "result" in payload:
                return payload["result"]
        raise ComponentConfigurationError(
            f"FAIL-FAST: MCP {method} response had no parseable 'data:' line "
            f"with a result field. Body head: {body[:300]!r}"
        )


def parse_tool_call_result(result: Any, tool_name: str) -> Any:
    """Unwrap a JSON-RPC ``tools/call`` ``result`` into the tool's output.

    MCP wraps tool output as
    ``{"content": [{"type": "text", "text": ...}], "structuredContent":
    {...}, "isError": bool}``. This helper:

    * raises ``ComponentConfigurationError`` when ``isError`` is true,
    * prefers ``structuredContent`` (the MCP typed-output slot),
    * else, for a single text content item, JSON-parses it when
      possible (caller convenience),
    * else returns the raw ``content`` list / raw ``result``.
    """
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        raise ComponentConfigurationError(
            f"FAIL-FAST: MCP tool {tool_name!r} returned isError=True. "
            f"Content: {result.get('content')!r}"
        )
    if "structuredContent" in result:
        return result["structuredContent"]
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return result
    if len(content) == 1 and isinstance(content[0], dict):
        text = content[0].get("text")
        if isinstance(text, str):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return content


__all__ = ["MCPTransport", "parse_tool_call_result"]
