"""WebSearchTool — a pluggable-backend web search ``ToolBase``.

A generic, framework-native web search capability for nanobrain Agents
(and, indirectly, Steps that wrap it). It is the Agent-path counterpart
to the domain-specific search tools the framework already ships
(``bv_brc_tool``, the Elasticsearch suite) — but general-purpose: it
answers an arbitrary text query with a ranked list of web results.

Why it lives in nanobrain proper
--------------------------------

Web search is not benchmark-specific or apecx-specific — it is a
generic Agent capability, the same way ``HTTPBackendAdapter`` (G38) is
a generic Step capability. It belongs alongside the other
``library/tools`` primitives. Benchmark-specific *composition* of this
tool (the step that wires it into a codegen workflow) lives apecx-side.

Pluggable backends
------------------

``WebSearchTool`` does not hard-code a search provider. The
``WebSearchBackend`` ABC is the extension point; concrete backends are
registered in ``_BACKENDS``. Two ship today:

* ``duckduckgo`` (default) — keyless, via the ``ddgs`` package. No API
  key, no cost. **Honest caveat**: keyless DDG rate-limits under a
  sustained sweep (hundreds of queries). The on-disk result cache
  (below) is the mitigation — a re-run hits the cache, not the network.
* ``tavily`` — an API-key backend (``$TAVILY_API_KEY``). Reliable, but
  costs per query. Real code; its integration test is *gated* on the
  key being present (same pattern as the Rhea gated tests).

Adding a backend (Brave, Serper, an MCP search server, ...) is a clean
``WebSearchBackend`` subclass + one ``_BACKENDS`` entry.

Result cache
------------

When ``parameters.cache_dir`` is set, every search result is cached on
disk keyed by ``sha256(backend | max_results | query)``. This is a
genuine reliability + reproducibility feature, not just a perf hack:

* an ablation sweep that re-runs is **reproducible** — identical
  queries return identical results regardless of how the live web has
  drifted;
* a rate-limited or flaky backend does not re-fail on a re-run;
* the network is hit exactly once per distinct query.

When ``cache_dir`` is unset, caching is disabled (every call is live).

Honesty contract — what is and is NOT a failure
------------------------------------------------

* A backend error (network failure, rate-limit rejection, HTTP non-200,
  missing API key, missing ``ddgs`` package) **FAILS LOUD** —
  ``ComponentConfigurationError`` or the backend's transport exception
  propagates. A web search tool that silently returned ``[]`` on a
  transport failure would make a workflow *look* like it ran while
  the drafter got zero context — exactly the silent-failure shape the
  workspace policy forbids.
* A search that **succeeds but finds nothing** returns ``results: []``.
  That is a legitimate, honest outcome (the query was obscure), NOT a
  failure — the caller distinguishes the two by: exception = failed,
  empty list = searched-OK-found-nothing.

**Non-determinism**: live web results drift over time. Any step or
agent that consumes ``WebSearchTool`` without the cache is a
non-deterministic component and must be labelled as such — it is not
under the framework's deterministic-step contract. The cache makes a
*given cache directory* deterministic, but the first population of the
cache is still live.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import ToolBase, ToolConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol + concrete backends
# ---------------------------------------------------------------------------

class WebSearchBackend(abc.ABC):
    """One web search provider.

    A backend's :meth:`search` returns a list of normalized result
    dicts ``{"title": str, "url": str, "snippet": str}``. It MUST raise
    on a transport/provider failure (never return ``[]`` to mask one).
    An empty list is reserved for the honest "searched OK, found
    nothing" outcome.
    """

    #: Backend identifier — matches ``parameters.backend`` in the YAML.
    name: str = ""

    @abc.abstractmethod
    async def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        """Run one search. Return normalized results; raise on failure."""
        ...


class DuckDuckGoBackend(WebSearchBackend):
    """Keyless DuckDuckGo backend via the ``ddgs`` package.

    No API key, no cost. ``ddgs`` is a synchronous library, so the
    blocking call is offloaded to a thread to keep the event loop
    responsive.
    """

    name = "duckduckgo"

    def __init__(self) -> None:
        try:
            from ddgs import DDGS  # noqa: PLC0415
        except ImportError as e:
            raise ComponentConfigurationError(
                "FAIL-FAST: WebSearchTool backend 'duckduckgo' requires the "
                "'ddgs' package, which is not installed. Install it with "
                "`pip install ddgs`, or configure a different backend via "
                "parameters.backend."
            ) from e
        self._DDGS = DDGS

    async def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        def _run() -> list[dict]:
            # ddgs is sync; DDGS().text(query, max_results=N) -> list of
            # {"title","href","body"} dicts. A transport / rate-limit
            # failure raises here and propagates — we do NOT swallow it.
            return self._DDGS().text(query, max_results=max_results) or []

        raw = await asyncio.to_thread(_run)
        return [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("href", "")),
                "snippet": str(r.get("body", "")),
            }
            for r in raw
        ]


class TavilyBackend(WebSearchBackend):
    """Tavily Search API backend (key via ``$TAVILY_API_KEY``).

    Reliable + low-latency, but billed per query. FAIL-LOUD at
    construction when the key env var is unset — a key-backed search
    tool that silently degraded to no-op is the forbidden
    silent-failure shape.

    The Tavily request/response shape below reflects Tavily's public
    API as of this writing; if Tavily changes it, the gated
    integration test (``$TAVILY_API_KEY`` present) surfaces the drift
    loudly rather than silently returning malformed results.
    """

    name = "tavily"
    _ENV_VAR = "TAVILY_API_KEY"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self) -> None:
        key = os.environ.get(self._ENV_VAR)
        if not key:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool backend 'tavily' requires "
                f"${self._ENV_VAR} to be set. Either export the key, or "
                f"configure parameters.backend: duckduckgo (keyless)."
            )
        self._api_key = key

    async def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        import httpx  # noqa: PLC0415 — framework already depends on httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._ENDPOINT,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                },
            )
        if resp.status_code != 200:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Tavily search returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        payload = resp.json()
        results = payload.get("results")
        if not isinstance(results, list):
            raise ComponentConfigurationError(
                f"FAIL-FAST: Tavily response had no 'results' list: "
                f"{payload!r}"
            )
        return [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "snippet": str(r.get("content", "")),
            }
            for r in results
        ]


#: Backend registry. Adding a provider = one subclass + one entry here.
_BACKENDS: dict[str, type[WebSearchBackend]] = {
    DuckDuckGoBackend.name: DuckDuckGoBackend,
    TavilyBackend.name: TavilyBackend,
}


def _build_backend(name: str) -> WebSearchBackend:
    """Instantiate a backend by name. FAIL-LOUD on an unknown name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ComponentConfigurationError(
            f"FAIL-FAST: WebSearchTool unknown backend {name!r}. "
            f"Available: {sorted(_BACKENDS)}."
        )
    return cls()


# ---------------------------------------------------------------------------
# On-disk result cache
# ---------------------------------------------------------------------------

class _SearchCache:
    """Content-addressed on-disk cache of search results.

    Disabled (every method a no-op / miss) when ``cache_dir`` is None.
    The key is ``sha256(backend | max_results | query)`` so a different
    backend or result-count is a different cache entry.
    """

    def __init__(self, cache_dir: Path | None) -> None:
        self._dir = cache_dir
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self._dir is not None

    @staticmethod
    def _key(backend: str, query: str, max_results: int) -> str:
        return hashlib.sha256(
            f"{backend}|{max_results}|{query}".encode()
        ).hexdigest()

    def _path(self, backend: str, query: str, max_results: int) -> Path | None:
        if self._dir is None:
            return None
        return self._dir / f"{self._key(backend, query, max_results)}.json"

    def get(
        self, backend: str, query: str, max_results: int
    ) -> list[dict[str, str]] | None:
        path = self._path(backend, query, max_results)
        if path is None or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # A corrupt cache entry is a cache MISS, not a hard failure —
            # we re-fetch. But log it so a systematically broken cache
            # dir is visible.
            logger.warning("WebSearchTool cache entry %s unreadable: %s", path, e)
            return None

    def put(
        self,
        backend: str,
        query: str,
        max_results: int,
        results: list[dict[str, str]],
    ) -> None:
        path = self._path(backend, query, max_results)
        if path is None:
            return
        # Atomic write: tmp + rename, so a concurrent reader never sees
        # a half-written file.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        tmp.rename(path)


# ---------------------------------------------------------------------------
# WebSearchTool
# ---------------------------------------------------------------------------

class WebSearchTool(ToolBase):
    """A pluggable-backend web search tool.

    Configuration (all under ``parameters`` in the tool YAML — the
    framework's tool-specific config slot)::

        name: web_search
        tool_type: external
        description: "General-purpose web search."
        parameters:
          backend: duckduckgo          # duckduckgo | tavily
          max_results: 5               # default result count
          cache_dir: /path/to/cache    # optional; omit to disable cache
        tool_card:
          capabilities: ["web_search"]

    ``execute`` accepts either a bare query string or a dict
    ``{"query": str, "max_results": int?}`` and returns::

        {
          "query": "<the query>",
          "backend": "duckduckgo",
          "from_cache": bool,
          "results": [{"title", "url", "snippet"}, ...]
        }

    An exception means the search FAILED. ``results: []`` means the
    search SUCCEEDED and found nothing — these are distinct.
    """

    COMPONENT_TYPE = "web_search_tool"
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

        params = component_config.get("parameters") or {}
        if not isinstance(params, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} 'parameters' must be "
                f"a dict; got {type(params).__name__}"
            )

        backend_name = str(params.get("backend", "duckduckgo"))
        # _build_backend FAIL-FASTs on an unknown name OR a missing
        # dependency / API key — surfaced here, at construction, not
        # silently at first use.
        self._backend = _build_backend(backend_name)
        self._backend_name = backend_name

        max_results = params.get("max_results", 5)
        try:
            self._max_results = int(max_results)
        except (TypeError, ValueError) as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} parameters.max_results "
                f"must be an int; got {max_results!r}"
            ) from e
        if self._max_results < 1:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} parameters.max_results "
                f"must be >= 1; got {self._max_results}"
            )

        cache_dir = self._resolve_cache_dir(params.get("cache_dir"))
        self._cache = _SearchCache(cache_dir)

        logger.info(
            "WebSearchTool %r initialized: backend=%s max_results=%d cache=%s",
            self.name,
            self._backend_name,
            self._max_results,
            "on" if self._cache.enabled else "off",
        )

    @staticmethod
    def _resolve_cache_dir(raw: Any) -> Path | None:
        """Resolve the configured ``cache_dir``.

        * Empty / None  -> caching disabled (returns None).
        * Absolute path -> used as-is.
        * Relative path -> resolved against the workspace root (via the
          G40 ``locate_workflow_root`` helper) so the cache location is
          stable regardless of the process CWD. Falls back to CWD only
          if the workspace root cannot be located.
        """
        if not raw:
            return None
        p = Path(str(raw)).expanduser()
        if p.is_absolute():
            return p
        root: Path
        try:
            from nanobrain.library.runtime.workspace_root import (  # noqa: PLC0415
                locate_workflow_root,
            )

            located = locate_workflow_root()
            root = located if located is not None else Path.cwd()
        except Exception:  # noqa: BLE001 — never let cache-path resolution break init
            root = Path.cwd()
        return (root / p).resolve()

    # Test seam — unit tests inject a fake backend without touching the
    # network. (Construction always builds a real backend; tests then
    # swap it.)
    @property
    def backend(self) -> WebSearchBackend:
        return self._backend

    @backend.setter
    def backend(self, value: WebSearchBackend) -> None:
        self._backend = value
        self._backend_name = value.name

    def get_schema(self) -> Dict[str, Any]:
        """OpenAI function-call spec for this tool.

        The framework's native agents (``SimpleAgent`` /
        ``ConversationalAgent``) build their ``tools`` list by calling
        ``tool.get_schema()`` on each registered tool. Returning the
        OpenAI tool-spec shape makes ``WebSearchTool`` a first-class
        citizen of the Agent tool-calling path.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    self.description
                    or "General-purpose web search. Returns a ranked list "
                    "of {title, url, snippet} results for a query."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The web search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": (
                                f"How many results to return "
                                f"(default {self._max_results})."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, payload: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Run a web search.

        ``payload`` may be a bare query string or a dict
        ``{"query": str, "max_results": int?}``. ``max_results`` may
        also be passed as a kwarg. FAIL-LOUD on a missing/blank query
        or a backend failure.
        """
        # Accept: execute("query"), execute({"query": ...}),
        # execute(query="...") — agent tool-call paths vary.
        if payload is None:
            payload = kwargs

        if isinstance(payload, str):
            query = payload
            max_results = kwargs.get("max_results", self._max_results)
        elif isinstance(payload, dict):
            query = payload.get("query", payload.get("q"))
            max_results = payload.get(
                "max_results", kwargs.get("max_results", self._max_results)
            )
        else:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} execute expects a "
                f"query string or a dict with a 'query' key; got "
                f"{type(payload).__name__}"
            )

        if not query or not isinstance(query, str) or not query.strip():
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} execute requires a "
                f"non-empty 'query'; got {query!r}"
            )
        query = query.strip()

        try:
            max_results = int(max_results)
        except (TypeError, ValueError) as e:
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} max_results must be "
                f"an int; got {max_results!r}"
            ) from e
        max_results = max(1, max_results)

        cached = self._cache.get(self._backend_name, query, max_results)
        if cached is not None:
            self._call_count += 1
            return {
                "query": query,
                "backend": self._backend_name,
                "from_cache": True,
                "results": cached,
            }

        # Cache miss — hit the live backend. A backend failure raises
        # here and propagates LOUD; we do not catch-and-empty-list.
        try:
            results = await self._backend.search(query, max_results=max_results)
        except ComponentConfigurationError:
            self._error_count += 1
            raise
        except Exception as e:
            self._error_count += 1
            raise ComponentConfigurationError(
                f"FAIL-FAST: WebSearchTool {self.name!r} backend "
                f"{self._backend_name!r} failed on query {query!r}: "
                f"{type(e).__name__}: {e}"
            ) from e

        self._call_count += 1
        self._cache.put(self._backend_name, query, max_results, results)
        return {
            "query": query,
            "backend": self._backend_name,
            "from_cache": False,
            "results": results,
        }


__all__ = ["WebSearchTool", "WebSearchBackend", "DuckDuckGoBackend", "TavilyBackend"]
