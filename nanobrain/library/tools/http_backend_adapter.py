"""HTTPBackendAdapter — G38 generic HTTP ToolBackendAdapter.

eval_03 Round 4 G38: ``ToolExecutionStep`` (G11) shipped with the
abstract ``ToolBackendAdapter`` protocol; concrete adapters lived
in their respective integration packages (Rhea fork ships
RheaAdapter / RheaMCPDispatcher; G11-completion shipped LocalParslAdapter
in-tree). The integration's
``apecx-mcp-integration/composition/steps/synonym_cache.py`` ships
``SynonymCacheLookupStep`` + ``VerifiedSynonymWritebackStep`` that POST
to control-plane HTTP endpoints DIRECTLY — bypassing the
``ToolExecutionStep`` -> adapter dispatch path entirely. This means
those steps cannot ride the G15 ``UnifiedToolDescriptor`` /
G28 capability-token surface; HTTP is hardcoded.

Post-G38 ships ``HTTPBackendAdapter`` (BACKEND_NAME="http"): a
generic HTTP adapter that POSTs (or GETs / PUTs / etc.) inputs to a
configured endpoint and returns the JSON response. Consumers point
their UTD's ``provenance_pin.class_path`` at this adapter and supply
the endpoint via UTD metadata or invoke kwargs; the integration's
direct-HTTP steps can migrate to ``ToolExecutionStep`` + this adapter.

## Configuration shapes

The adapter is constructed once per backend (typically one HTTPBackendAdapter
per service URL — control plane, an external REST API, etc.):

    adapter = HTTPBackendAdapter(
        base_url="http://localhost:8080",
        default_method="POST",
        default_timeout=30.0,
        default_headers={"Authorization": "Bearer ${API_TOKEN}"},
    )
    ToolBackendRegistry.register(adapter)

Per-call overrides flow through invoke kwargs:

    await adapter.invoke(
        utd, inputs,
        endpoint="/synonyms/cache_lookup",
        method="POST",
        timeout=10.0,
        headers={"X-Trace-Id": "abc"},
    )

Endpoint resolution order:
    1. invoke kwargs ``endpoint=...`` (per-call override; canonical)
    2. f"/{utd.descriptor_id}" (last-resort fallback when no kwarg)

(Earlier drafts considered a per-UTD ``provenance_pin.metadata.endpoint``
field but ``UTDProvenancePin`` has ``extra="forbid"`` and no
``metadata`` field; per-tool defaults are best surfaced through the
caller's step config, not the UTD descriptor.)

## Response parsing

JSON content-type → ``response.json()`` returned as a dict.
Non-JSON 2xx → ``{"_response_text": <body>, "_status_code": <code>}``.
Non-2xx → ``RuntimeError`` with status + body summary.

## Why httpx, not aiohttp

Both are venv-available; httpx has a cleaner async API + native
ASGI test client (``httpx.AsyncClient(transport=ASGITransport(app))``)
which we use in the test suite to exercise the adapter against a
real ASGI app without a network bind. aiohttp's ``aiohttp.test_utils``
is more invasive.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 4 G38;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 Tier 4.
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, Optional

import httpx

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.steps.tool_execution_step import (
    ToolBackendAdapter,
)

logger = logging.getLogger(__name__)


class HTTPBackendAdapter(ToolBackendAdapter):
    """ToolBackendAdapter that dispatches tool calls over HTTP.

    Args:
        base_url: Base URL for the backend service (e.g.,
            ``"http://localhost:8080"``). Required; trailing slash
            is stripped.
        default_method: HTTP method for invokes that don't specify one.
            Defaults to ``"POST"``. Per-call ``method=...`` overrides.
        default_timeout: Request timeout in seconds (httpx semantics).
            Defaults to ``30.0``.
        default_headers: Headers sent with every request. Per-call
            ``headers=...`` extends (does NOT replace) these.
        client: Optional pre-built ``httpx.AsyncClient``. Useful for
            tests that need an ASGI transport. When None, the adapter
            builds its own client at construction time.
    """

    BACKEND_NAME: ClassVar[str] = "http"

    def __init__(
        self,
        *,
        base_url: str,
        default_method: str = "POST",
        default_timeout: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not base_url:
            raise ComponentConfigurationError(
                "FAIL-FAST: HTTPBackendAdapter requires a non-empty "
                "base_url"
            )
        self._base_url = base_url.rstrip("/")
        self._default_method = default_method.upper()
        self._default_timeout = float(default_timeout)
        self._default_headers: Dict[str, str] = dict(default_headers or {})
        # Track whether we own the client so close() only shuts down
        # the ones we built.
        self._client_owned = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._default_timeout,
            headers=self._default_headers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def invoke(
        self,
        utd: UnifiedToolDescriptor,
        inputs: Dict[str, Any],
        *,
        run_context_namespace: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """POST (or other method) ``inputs`` to the resolved endpoint."""
        if not isinstance(inputs, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: HTTPBackendAdapter.invoke inputs must be "
                f"a dict; got {type(inputs).__name__}"
            )

        endpoint = self._resolve_endpoint(utd, kwargs)
        method = str(kwargs.get("method") or self._default_method).upper()
        timeout = float(kwargs.get("timeout", self._default_timeout))
        # Merge default_headers + per-call headers. Default first so
        # per-call values can override individual entries; this mirrors
        # the documented "extends, doesn't replace" contract. When a
        # pre-built client is passed, the client may not have our
        # default_headers attached, so merging at request time is the
        # canonical place.
        merged_headers: Dict[str, str] = dict(self._default_headers)
        merged_headers.update(kwargs.get("headers") or {})
        # run_context_namespace propagates as a header for backends
        # that want to honor multi-tenant isolation.
        if run_context_namespace:
            merged_headers.setdefault(
                "X-Nanobrain-Run-Namespace", run_context_namespace
            )
        per_call_headers = merged_headers

        try:
            if method == "GET":
                response = await self._client.get(
                    endpoint,
                    params=inputs,
                    timeout=timeout,
                    headers=per_call_headers,
                )
            else:
                response = await self._client.request(
                    method,
                    endpoint,
                    json=inputs,
                    timeout=timeout,
                    headers=per_call_headers,
                )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"HTTPBackendAdapter transport error to "
                f"{self._base_url}{endpoint} ({method}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            # Truncate body for readable error messages.
            body_preview = response.text[:512]
            raise RuntimeError(
                f"HTTPBackendAdapter HTTP {response.status_code} from "
                f"{self._base_url}{endpoint} ({method}); "
                f"body: {body_preview!r}"
            )

        # Parse response. JSON → dict. Non-JSON 2xx → wrap.
        ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype == "application/json":
            try:
                payload = response.json()
            except Exception as exc:  # malformed JSON despite content-type
                raise RuntimeError(
                    f"HTTPBackendAdapter received content-type=json from "
                    f"{self._base_url}{endpoint} but body did not parse: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if isinstance(payload, dict):
                return payload
            return {"_response_payload": payload, "_status_code": response.status_code}
        return {
            "_response_text": response.text,
            "_status_code": response.status_code,
            "_content_type": ctype,
        }

    async def close(self) -> None:
        """Close the underlying httpx client (if we own it).

        Safe to call multiple times. Adapters that received a
        pre-built client via the ``client=`` kwarg do NOT close it —
        the caller manages that lifecycle.
        """
        if self._client_owned and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_endpoint(
        self,
        utd: UnifiedToolDescriptor,
        kwargs: Dict[str, Any],
    ) -> str:
        """Resolve the endpoint path to call.

        Priority:
          1. invoke kwarg ``endpoint=...`` (per-call override; canonical)
          2. f"/{utd.descriptor_id}" (fallback)
        """
        if "endpoint" in kwargs and kwargs["endpoint"]:
            return self._normalize_path(kwargs["endpoint"])
        descriptor_id = getattr(utd, "descriptor_id", None)
        if descriptor_id:
            return self._normalize_path(str(descriptor_id))
        raise ComponentConfigurationError(
            f"FAIL-FAST: HTTPBackendAdapter cannot resolve an endpoint "
            f"for UTD {utd!r}: no invoke kwarg 'endpoint' AND no "
            f"descriptor_id. Pass endpoint=... at invoke time."
        )

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Ensure the path starts with '/' so httpx joins it cleanly
        with the base_url."""
        if not path.startswith("/"):
            return "/" + path
        return path


__all__ = ["HTTPBackendAdapter"]
