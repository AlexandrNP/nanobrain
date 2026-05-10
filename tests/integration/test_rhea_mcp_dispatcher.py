"""Tests for RheaMCPDispatcher (T-RH-03) and the from_descriptor fixes.

Two test surfaces:

1. **Unit tests** (run unconditionally) — exercise the dispatcher's
   contract behavior against a *fake* MCP server (httpx mock-transport
   based). Verifies session handshake, JSON-RPC envelope, error
   propagation, and result parsing without needing a live Rhea.

2. **Integration tests** (gated on ``RHEA_MCP_URL`` env var) — exercise
   the dispatcher against a live Rhea MCP server. Validates the
   end-to-end path including from_descriptor + from_dict + tempfile
   YAML materialization + dispatch.

Bonus: regression pin for two ``from_descriptor`` bugs surfaced
during the smoke:
- Pre-fix, the bare ``UnifiedToolDescriptor(**utd)`` constructor
  didn't open nested classes (UTDProvenancePin); any dict-form UTD
  was rejected. Fix: switched to ``from_dict``.
- Pre-fix, ``ImplCls.from_config(inline_config_dict)`` was rejected
  by ToolConfig (YAML-first discipline). Fix: materialize the
  inline config to a NamedTemporaryFile.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import ToolBase
from nanobrain.library.tools.rhea_mcp_dispatcher import RheaMCPDispatcher

pytestmark = pytest.mark.integration


_RHEA_URL = os.environ.get("RHEA_MCP_URL")
_rhea_skip = pytest.mark.skipif(
    _RHEA_URL is None, reason="RHEA_MCP_URL not set",
)


# ---------------------------------------------------------------------------
# Helpers — fake MCP server via httpx MockTransport
# ---------------------------------------------------------------------------

def _build_fake_mcp(handler):
    """Build a (tool, handler-state) pair where the dispatcher's
    underlying httpx client has been replaced with a MockTransport that
    forwards to ``handler(request)`` returning ``httpx.Response``."""
    transport = httpx.MockTransport(handler)
    # Build the dispatcher first; then swap its client.
    utd_dict = _basic_utd_dict()
    tool = ToolBase.from_descriptor(
        utd_dict, mcp_url="http://fake/mcp/", rhea_tool_name="testtool",
    )
    tool._client = httpx.AsyncClient(transport=transport, timeout=5.0)
    return tool


def _basic_utd_dict():
    return {
        "descriptor_id": "rhea:testtool@0.1.0",
        "display_name": "Test Tool",
        "summary": "Synthetic UTD for dispatcher tests.",
        "long_description": "",
        "inputs": [{"name": "x", "type": "string", "description": "",
                    "required": True, "default": None}],
        "outputs": [{"name": "return", "type": "object", "description": ""}],
        "side_effects": "none",
        "determinism": "R3",
        "resource_class": "cpu_light",
        "provenance_pin": {
            "class_path": (
                "nanobrain.library.tools.rhea_mcp_dispatcher.RheaMCPDispatcher"
            ),
        },
    }


def _mcp_response(status: int, body: str, *, session_id: str = "fake-sess-1"):
    return httpx.Response(
        status,
        text=body,
        headers={"mcp-session-id": session_id},
    )


def _initialize_response(session_id: str = "fake-sess-1"):
    return _mcp_response(
        200,
        'event: message\ndata: ' + json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "Fake", "version": "0.0.1"},
            },
        }) + "\n",
        session_id=session_id,
    )


def _tools_call_response(payload: Dict[str, Any]):
    return _mcp_response(
        200,
        'event: message\ndata: ' + json.dumps({
            "jsonrpc": "2.0", "id": 2, "result": payload,
        }) + "\n",
    )


# ---------------------------------------------------------------------------
# 1. Construction via from_descriptor
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_from_descriptor_creates_dispatcher(self):
        tool = ToolBase.from_descriptor(
            _basic_utd_dict(),
            mcp_url="http://fake/mcp/",
            rhea_tool_name="testtool",
        )
        assert isinstance(tool, RheaMCPDispatcher)
        assert tool._mcp_url == "http://fake/mcp/"
        assert tool._rhea_tool_name == "testtool"

    def test_missing_mcp_url_fails_fast(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(
                _basic_utd_dict(),
                rhea_tool_name="testtool",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "mcp_url" in str(exc_info.value)

    def test_missing_rhea_tool_name_fails_fast(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(
                _basic_utd_dict(),
                mcp_url="http://fake/mcp/",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "rhea_tool_name" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 2. Regression pins for from_descriptor bugs (T-RH-03 surfaced)
# ---------------------------------------------------------------------------

class TestFromDescriptorRegressions:

    def test_dict_utd_with_nested_provenance_pin_works(self):
        """Pre-T-RH-03 fix, the bare UnifiedToolDescriptor(**dict)
        constructor didn't open UTDProvenancePin; dict-form UTDs were
        rejected. Pin via this test that from_descriptor correctly
        opens nested classes."""
        utd_dict = _basic_utd_dict()
        # Verify the nested provenance_pin survives
        tool = ToolBase.from_descriptor(
            utd_dict, mcp_url="http://fake/", rhea_tool_name="x",
        )
        assert tool is not None

    def test_inline_config_does_not_get_dict_passed_to_from_config(self):
        """Pre-T-RH-03 fix, from_descriptor passed the inline_config
        dict directly to ImplCls.from_config(...) which the framework's
        YAML-first discipline rejected. Pin: dict-form UTD → from_descriptor
        → tool builds without error."""
        # If the bug were back, this would raise "ToolConfig requires
        # file path but got dictionary".
        tool = ToolBase.from_descriptor(
            _basic_utd_dict(),
            mcp_url="http://fake/", rhea_tool_name="x",
        )
        assert isinstance(tool, RheaMCPDispatcher)


# ---------------------------------------------------------------------------
# 3. Dispatch behavior against a fake MCP server
# ---------------------------------------------------------------------------

class TestDispatchBehavior:

    def test_dispatch_happy_path_text_content(self):
        """Tool returns a single text content item with JSON inside;
        the dispatcher should parse the JSON for caller convenience."""
        async def run():
            calls = []
            def handler(request: httpx.Request) -> httpx.Response:
                body = request.read().decode("utf-8")
                payload = json.loads(body)
                calls.append(payload)
                if payload.get("method") == "initialize":
                    return _initialize_response()
                if payload.get("method") == "notifications/initialized":
                    return httpx.Response(202)
                if payload.get("method") == "tools/call":
                    return _tools_call_response({
                        "content": [
                            {"type": "text", "text": json.dumps({"echo": "hi"})}
                        ],
                        "isError": False,
                    })
                return httpx.Response(404)

            tool = _build_fake_mcp(handler)
            try:
                result = await tool.execute({"x": "hi"})
                assert result == {"echo": "hi"}
            finally:
                await tool.aclose()
        asyncio.run(run())

    def test_dispatch_iserror_true_raises(self):
        """isError=True in the MCP response must FAIL-FAST."""
        async def run():
            def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.read())
                if payload.get("method") == "initialize":
                    return _initialize_response()
                if payload.get("method") == "notifications/initialized":
                    return httpx.Response(202)
                if payload.get("method") == "tools/call":
                    return _tools_call_response({
                        "content": [{"type": "text", "text": "remote barf"}],
                        "isError": True,
                    })
                return httpx.Response(404)

            tool = _build_fake_mcp(handler)
            try:
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await tool.execute({"x": "hi"})
                assert "FAIL-FAST" in str(exc_info.value)
                assert "isError=True" in str(exc_info.value)
            finally:
                await tool.aclose()
        asyncio.run(run())

    def test_jsonrpc_error_object_raises(self):
        """A JSON-RPC error response must FAIL-FAST with the code +
        message in the exception body."""
        async def run():
            def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.read())
                if payload.get("method") == "initialize":
                    return _initialize_response()
                if payload.get("method") == "notifications/initialized":
                    return httpx.Response(202)
                if payload.get("method") == "tools/call":
                    return _mcp_response(
                        200,
                        'event: message\ndata: ' + json.dumps({
                            "jsonrpc": "2.0", "id": 2,
                            "error": {"code": -32602, "message": "Invalid params"},
                        }) + "\n",
                    )
                return httpx.Response(404)

            tool = _build_fake_mcp(handler)
            try:
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await tool.execute({"x": "hi"})
                assert "JSON-RPC error" in str(exc_info.value)
                assert "-32602" in str(exc_info.value)
                assert "Invalid params" in str(exc_info.value)
            finally:
                await tool.aclose()
        asyncio.run(run())

    def test_http_500_raises_with_body(self):
        async def run():
            def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.read())
                if payload.get("method") == "initialize":
                    return _initialize_response()
                if payload.get("method") == "notifications/initialized":
                    return httpx.Response(202)
                if payload.get("method") == "tools/call":
                    return httpx.Response(500, text="server crashed")
                return httpx.Response(404)

            tool = _build_fake_mcp(handler)
            try:
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await tool.execute({"x": "hi"})
                assert "HTTP 500" in str(exc_info.value)
                assert "server crashed" in str(exc_info.value)
            finally:
                await tool.aclose()
        asyncio.run(run())

    def test_session_invalidation_triggers_one_retry(self):
        """When the server says 400 + 'session' in body, the dispatcher
        clears its session_id and retries the call exactly once."""
        async def run():
            init_count = [0]
            call_count = [0]
            def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.read())
                method = payload.get("method")
                if method == "initialize":
                    init_count[0] += 1
                    return _initialize_response(
                        session_id=f"sess-{init_count[0]}",
                    )
                if method == "notifications/initialized":
                    return httpx.Response(202)
                if method == "tools/call":
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return httpx.Response(
                            400, text="Bad Request: Missing session ID",
                        )
                    return _tools_call_response({
                        "content": [{"type": "text", "text": "ok-after-retry"}],
                        "isError": False,
                    })
                return httpx.Response(404)

            tool = _build_fake_mcp(handler)
            try:
                result = await tool.execute({"x": "hi"})
                assert result == "ok-after-retry"
                # Two initializes (original + retry-after-session-clear)
                assert init_count[0] == 2
                # Two tools/call attempts (the first failed, the second succeeded)
                assert call_count[0] == 2
            finally:
                await tool.aclose()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. Payload validation
# ---------------------------------------------------------------------------

class TestPayloadValidation:

    def test_non_dict_payload_fails_fast(self):
        async def run():
            tool = ToolBase.from_descriptor(
                _basic_utd_dict(),
                mcp_url="http://fake/", rhea_tool_name="x",
            )
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await tool.execute("not a dict")
            assert "FAIL-FAST" in str(exc_info.value)
            assert "must be a dict" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. Integration tests against live Rhea
# ---------------------------------------------------------------------------

@_rhea_skip
class TestLiveRheaDispatch:

    def test_live_dispatch_to_find_tools(self):
        """Dispatch the find_tools meta-tool. Without the embedding
        service running, find_tools fails internally with a Connection
        error — the dispatcher must surface this as a FAIL-FAST
        (isError=True propagation), NOT a silent success."""
        async def run():
            utd = _basic_utd_dict()
            utd["descriptor_id"] = "rhea:find_tools@1.10.1"
            tool = ToolBase.from_descriptor(
                utd, mcp_url=_RHEA_URL, rhea_tool_name="find_tools",
            )
            try:
                # find_tools needs the embedding service; expect FAIL-FAST
                # from the dispatcher's isError handling.
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await tool.execute({"query": "sequence alignment"})
                msg = str(exc_info.value)
                # Either the embedding service is down (Connection
                # error from Rhea) OR the embedding model returned
                # something the tool doesn't like — both surface as
                # isError=True with a text content body.
                assert "isError=True" in msg
            finally:
                await tool.aclose()
        asyncio.run(run())
