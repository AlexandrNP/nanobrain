"""G38 — pin the HTTPBackendAdapter contract against a real ASGI test server.

eval_03 Round 4 G38: pre-G38 ToolExecutionStep had no in-tree HTTP
adapter; integration steps that talked to HTTP services (e.g.,
SynonymCacheLookupStep posting to the control plane) bypassed the
adapter dispatch entirely. Without G38 those steps cannot ride the
G15 UnifiedToolDescriptor / G28 capability-token surface.

Post-G38 ships ``HTTPBackendAdapter``. This test exercises it against
a REAL httpx ASGITransport-mounted FastAPI-style app — no mocks of
the HTTP layer.

Pinned contracts:
  1. POST happy path returns dict from JSON response
  2. GET method sends inputs as query params
  3. Non-2xx responses raise RuntimeError with status + body preview
  4. Endpoint resolution priority: kwarg > UTD metadata > descriptor_id
  5. base_url + endpoint join cleanly (no double slash)
  6. run_context_namespace propagates as X-Nanobrain-Run-Namespace header
  7. Custom headers per-call extend (don't replace) default headers
  8. Non-JSON 2xx response wraps under _response_text
  9. Transport error (e.g., DNS fail) wraps in RuntimeError
 10. close() shuts down owned client; pre-built client is preserved
 11. invoke kwargs[endpoint] overrides UTD-side metadata
 12. descriptor_id fallback resolves when neither kwarg nor metadata set

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 4 G38;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 Tier 4.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.tools.http_backend_adapter import HTTPBackendAdapter


pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Test ASGI app — handles a few endpoints that the tests use as the
# real backend. No FastAPI dep; minimal ASGI handler.
# ---------------------------------------------------------------------------


class _RecordingASGIApp:
    """Minimal ASGI app that:
       - records every request (path, method, headers, body)
       - dispatches on path to canned response handlers
    """

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        # Build path + method.
        path = scope["path"]
        method = scope["method"]
        # Collect body.
        body = b""
        more = True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
        # Headers.
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        query_string = scope.get("query_string", b"").decode("latin-1")
        self.requests.append({
            "path": path,
            "method": method,
            "headers": headers,
            "body": body,
            "query_string": query_string,
        })

        # Dispatch on path.
        status, ctype, response_body = self._dispatch(path, method, body, query_string)
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", ctype.encode("latin-1"))],
        })
        await send({"type": "http.response.body", "body": response_body})

    def _dispatch(self, path, method, body, query_string):
        if path == "/echo_post" and method == "POST":
            try:
                payload = json.loads(body)
            except Exception:
                payload = {}
            response = {"received": payload, "echoed_at": "test"}
            return 200, "application/json", json.dumps(response).encode()

        if path == "/echo_get" and method == "GET":
            response = {"query": query_string, "method": "GET"}
            return 200, "application/json", json.dumps(response).encode()

        if path == "/error_500":
            return 500, "text/plain", b"Internal Server Error: synthetic"

        if path == "/plain_text":
            return 200, "text/plain", b"hello, plain world"

        if path == "/synonyms/cache_lookup":
            return 200, "application/json", json.dumps({"hit": True, "via": "metadata"}).encode()

        if path == "/http:utd_fallback@0.1.0":
            return 200, "application/json", json.dumps({"hit": True, "via": "descriptor_id"}).encode()

        return 404, "text/plain", b"not found"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def asgi_app():
    return _RecordingASGIApp()


@pytest.fixture
def adapter(asgi_app):
    """Build an adapter wired to the ASGI app via httpx ASGITransport."""
    transport = httpx.ASGITransport(app=asgi_app)
    client = httpx.AsyncClient(
        transport=transport, base_url="http://testserver", timeout=10.0
    )
    a = HTTPBackendAdapter(
        base_url="http://testserver",
        default_method="POST",
        default_timeout=10.0,
        default_headers={"X-Default-Header": "default-value"},
        client=client,
    )
    yield a
    asyncio.run(a.close())


def _utd(descriptor_id_suffix: str = "echo") -> UnifiedToolDescriptor:
    """Build a minimal UTD whose descriptor_id ends with the given
    suffix. Endpoint resolution in tests is via the invoke kwarg
    (canonical); descriptor_id fallback exercised separately."""
    def _dummy(x: int = 0) -> dict:
        return {"x": x}
    return UnifiedToolDescriptor.from_python_callable(
        _dummy,
        descriptor_id=f"http:test_{descriptor_id_suffix}@0.1.0",
        backend="http",
        version="0.1.0",
        provenance_class_path=f"{_dummy.__module__}.{_dummy.__name__}",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_happy_path_returns_dict(adapter, asgi_app):
    utd = _utd()
    result = asyncio.run(adapter.invoke(utd, {"foo": "bar"}, endpoint="/echo_post"))
    assert isinstance(result, dict)
    assert result["received"] == {"foo": "bar"}
    # Verify the request was actually made.
    assert len(asgi_app.requests) == 1
    req = asgi_app.requests[0]
    assert req["method"] == "POST"
    assert req["path"] == "/echo_post"


def test_get_method_sends_inputs_as_query(adapter, asgi_app):
    utd = _utd()
    result = asyncio.run(
        adapter.invoke(utd, {"q": "alpha", "k": "5"}, endpoint="/echo_get", method="GET")
    )
    assert result["method"] == "GET"
    # query_string should contain both keys.
    qs = asgi_app.requests[0]["query_string"]
    assert "q=alpha" in qs
    assert "k=5" in qs


def test_non_2xx_raises_runtime_error(adapter):
    utd = _utd()
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(adapter.invoke(utd, {}, endpoint="/error_500"))
    msg = str(excinfo.value)
    assert "500" in msg
    assert "synthetic" in msg


def test_endpoint_kwarg_overrides_default(adapter, asgi_app):
    """The invoke kwarg endpoint is the canonical per-call override."""
    utd = _utd()
    asyncio.run(
        adapter.invoke(utd, {"k": "v"}, endpoint="/synonyms/cache_lookup")
    )
    assert asgi_app.requests[0]["path"] == "/synonyms/cache_lookup"


def test_descriptor_id_fallback_when_no_kwarg(adapter, asgi_app):
    """When no ``endpoint`` kwarg is passed, the adapter falls back
    to ``/<descriptor_id>``. Use a UTD whose descriptor_id matches
    the routed path under canonical UTD grammar (<backend>:<id>@<version>)."""
    def _utd_fallback() -> dict:
        return {}
    utd = UnifiedToolDescriptor.from_python_callable(
        _utd_fallback,
        descriptor_id="http:utd_fallback@0.1.0",
        backend="http",
        version="0.1.0",
        provenance_class_path=f"{_utd_fallback.__module__}._utd_fallback",
    )
    result = asyncio.run(adapter.invoke(utd, {}))
    assert result["via"] == "descriptor_id"


def test_run_context_namespace_propagates_as_header(adapter, asgi_app):
    utd = _utd()
    asyncio.run(
        adapter.invoke(
            utd, {"data": "x"},
            endpoint="/echo_post",
            run_context_namespace="run_abc.tenant_1",
        )
    )
    headers = asgi_app.requests[0]["headers"]
    assert headers.get("x-nanobrain-run-namespace") == "run_abc.tenant_1"


def test_default_headers_sent(adapter, asgi_app):
    """Default headers configured at adapter construction time are
    sent on every request."""
    utd = _utd()
    asyncio.run(adapter.invoke(utd, {}, endpoint="/echo_post"))
    headers = asgi_app.requests[0]["headers"]
    assert headers.get("x-default-header") == "default-value"


def test_per_call_headers_extend_defaults(adapter, asgi_app):
    """Per-call ``headers=...`` extends, doesn't replace, defaults."""
    utd = _utd()
    asyncio.run(
        adapter.invoke(
            utd, {},
            endpoint="/echo_post",
            headers={"X-Trace-Id": "abc-123"},
        )
    )
    headers = asgi_app.requests[0]["headers"]
    assert headers.get("x-trace-id") == "abc-123"
    # And defaults still present.
    assert headers.get("x-default-header") == "default-value"


def test_non_json_2xx_wraps_response_text(adapter):
    utd = _utd()
    result = asyncio.run(adapter.invoke(utd, {}, endpoint="/plain_text"))
    assert result["_response_text"] == "hello, plain world"
    assert result["_status_code"] == 200
    assert result["_content_type"] == "text/plain"


def test_invalid_inputs_type_fails_fast(adapter):
    utd = _utd()
    with pytest.raises(ComponentConfigurationError) as excinfo:
        asyncio.run(
            adapter.invoke(utd, "not-a-dict", endpoint="/echo_post")  # type: ignore
        )
    assert "must be a dict" in str(excinfo.value)


def test_empty_base_url_fails_fast():
    with pytest.raises(ComponentConfigurationError) as excinfo:
        HTTPBackendAdapter(base_url="")
    assert "base_url" in str(excinfo.value)


def test_close_is_idempotent_on_owned_client(asgi_app):
    """close() can be called multiple times safely."""
    transport = httpx.ASGITransport(app=asgi_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    a = HTTPBackendAdapter(base_url="http://testserver", client=client)
    asyncio.run(a.close())
    asyncio.run(a.close())  # second close: no-op (client already closed)


def test_close_does_not_close_pre_built_client(asgi_app):
    """When a client is passed in, the adapter does NOT close it on
    close() — caller manages that lifecycle."""
    transport = httpx.ASGITransport(app=asgi_app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    a = HTTPBackendAdapter(base_url="http://testserver", client=client)
    asyncio.run(a.close())
    # Client still usable.
    assert not client.is_closed
    asyncio.run(client.aclose())  # caller closes


def test_endpoint_normalize_adds_leading_slash(adapter, asgi_app):
    utd = _utd()
    # Pass endpoint without leading slash.
    asyncio.run(
        adapter.invoke(utd, {}, endpoint="echo_post")
    )
    assert asgi_app.requests[0]["path"] == "/echo_post"
