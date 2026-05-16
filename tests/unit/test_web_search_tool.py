"""Unit tests for WebSearchTool — the pluggable-backend web search ToolBase.

Most tests run UNCONDITIONALLY with a fake backend (no network). Two
integration tests are GATED:

* ``$WEB_SEARCH_LIVE_DDG=1`` — runs a real keyless DuckDuckGo query.
* ``$TAVILY_API_KEY`` set — runs a real Tavily query.

Gating pattern mirrors nanobrain's test_rhea_mcp_dispatcher.py.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.tools.web_search import (
    DuckDuckGoBackend,
    WebSearchBackend,
    WebSearchTool,
)


# ---- helpers --------------------------------------------------------------

def _build_tool(parameters: dict) -> WebSearchTool:
    """Write a tmp tool YAML and load via from_config."""
    cfg = {
        "name": "web_search",
        "tool_type": "external",
        "description": "test web search tool",
        "parameters": parameters,
        "tool_card": {"capabilities": ["web_search"]},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, encoding="utf-8"
    ) as f:
        yaml.safe_dump(cfg, f)
        path = f.name
    return WebSearchTool.from_config(path)


class _FakeBackend(WebSearchBackend):
    """Records every search() call and returns a canned result list."""

    name = "fake"

    def __init__(self, results=None, raises: Exception | None = None):
        self._results = results if results is not None else [
            {"title": "T1", "url": "http://x/1", "snippet": "s1"},
        ]
        self._raises = raises
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, max_results: int):
        self.calls.append((query, max_results))
        if self._raises is not None:
            raise self._raises
        return list(self._results)


# ---- construction ---------------------------------------------------------

def test_construction_default_duckduckgo_backend():
    tool = _build_tool({"backend": "duckduckgo", "max_results": 3})
    assert tool.backend.name == "duckduckgo"
    assert isinstance(tool.backend, DuckDuckGoBackend)


def test_construction_backend_defaults_to_duckduckgo_when_omitted():
    tool = _build_tool({"max_results": 5})
    assert tool.backend.name == "duckduckgo"


def test_construction_unknown_backend_fails_loud():
    with pytest.raises(ComponentConfigurationError, match="unknown backend"):
        _build_tool({"backend": "not_a_real_backend"})


def test_construction_tavily_without_key_fails_loud(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ComponentConfigurationError, match="TAVILY_API_KEY"):
        _build_tool({"backend": "tavily"})


def test_construction_bad_max_results_fails_loud():
    with pytest.raises(ComponentConfigurationError, match="max_results"):
        _build_tool({"backend": "duckduckgo", "max_results": "not-an-int"})


def test_construction_zero_max_results_fails_loud():
    with pytest.raises(ComponentConfigurationError, match="max_results"):
        _build_tool({"backend": "duckduckgo", "max_results": 0})


# ---- execute (fake backend) ----------------------------------------------

def test_execute_string_query_happy_path():
    async def run():
        tool = _build_tool({"backend": "duckduckgo", "max_results": 4})
        fake = _FakeBackend()
        tool.backend = fake
        out = await tool.execute("what is nanobrain")
        assert out["query"] == "what is nanobrain"
        assert out["backend"] == "fake"
        assert out["from_cache"] is False
        assert out["results"] == [
            {"title": "T1", "url": "http://x/1", "snippet": "s1"}
        ]
        # default max_results from config flowed through
        assert fake.calls == [("what is nanobrain", 4)]

    asyncio.run(run())


def test_execute_dict_query_with_max_results_override():
    async def run():
        tool = _build_tool({"backend": "duckduckgo", "max_results": 4})
        fake = _FakeBackend()
        tool.backend = fake
        out = await tool.execute({"query": "deseq2 usage", "max_results": 2})
        assert out["query"] == "deseq2 usage"
        assert fake.calls == [("deseq2 usage", 2)]

    asyncio.run(run())


def test_execute_kwarg_query():
    async def run():
        tool = _build_tool({"backend": "duckduckgo"})
        fake = _FakeBackend()
        tool.backend = fake
        out = await tool.execute(query="kwarg style")
        assert out["query"] == "kwarg style"

    asyncio.run(run())


def test_execute_empty_query_fails_loud():
    async def run():
        tool = _build_tool({"backend": "duckduckgo"})
        tool.backend = _FakeBackend()
        with pytest.raises(ComponentConfigurationError, match="non-empty 'query'"):
            await tool.execute("   ")

    asyncio.run(run())


def test_execute_bad_payload_type_fails_loud():
    async def run():
        tool = _build_tool({"backend": "duckduckgo"})
        tool.backend = _FakeBackend()
        with pytest.raises(ComponentConfigurationError, match="query string or a dict"):
            await tool.execute(12345)

    asyncio.run(run())


def test_execute_backend_failure_fails_loud():
    """A backend transport failure must propagate LOUD, never become []."""
    async def run():
        tool = _build_tool({"backend": "duckduckgo"})
        tool.backend = _FakeBackend(raises=RuntimeError("rate limited"))
        with pytest.raises(ComponentConfigurationError, match="rate limited"):
            await tool.execute("anything")
        assert tool.error_count == 1

    asyncio.run(run())


def test_execute_empty_results_is_not_a_failure():
    """A search that succeeds but finds nothing returns [] — honest, not
    a failure. (Distinct from a backend error, which raises.)"""
    async def run():
        tool = _build_tool({"backend": "duckduckgo"})
        tool.backend = _FakeBackend(results=[])
        out = await tool.execute("query with no hits")
        assert out["results"] == []
        assert out["from_cache"] is False
        assert tool.error_count == 0  # NOT counted as an error

    asyncio.run(run())


# ---- cache ----------------------------------------------------------------

def test_cache_miss_then_hit():
    async def run():
        with tempfile.TemporaryDirectory() as cache_dir:
            tool = _build_tool(
                {"backend": "duckduckgo", "max_results": 3, "cache_dir": cache_dir}
            )
            fake = _FakeBackend(
                results=[{"title": "C", "url": "http://c", "snippet": "cached"}]
            )
            tool.backend = fake

            first = await tool.execute("cached query")
            assert first["from_cache"] is False
            assert len(fake.calls) == 1

            # Second identical call -> served from disk cache, backend NOT hit.
            second = await tool.execute("cached query")
            assert second["from_cache"] is True
            assert second["results"] == first["results"]
            assert len(fake.calls) == 1  # backend was NOT called again

            # A cache file actually exists on disk.
            assert list(Path(cache_dir).glob("*.json"))

    asyncio.run(run())


def test_cache_key_distinguishes_max_results():
    async def run():
        with tempfile.TemporaryDirectory() as cache_dir:
            tool = _build_tool(
                {"backend": "duckduckgo", "max_results": 3, "cache_dir": cache_dir}
            )
            fake = _FakeBackend()
            tool.backend = fake
            await tool.execute({"query": "q", "max_results": 2})
            await tool.execute({"query": "q", "max_results": 5})
            # Different max_results -> different cache key -> 2 live calls.
            assert len(fake.calls) == 2

    asyncio.run(run())


def test_cache_disabled_when_no_cache_dir():
    async def run():
        tool = _build_tool({"backend": "duckduckgo"})
        fake = _FakeBackend()
        tool.backend = fake
        await tool.execute("q")
        await tool.execute("q")
        # No cache_dir -> every call is live.
        assert len(fake.calls) == 2

    asyncio.run(run())


# ---- gated integration tests ---------------------------------------------

@pytest.mark.skipif(
    os.environ.get("WEB_SEARCH_LIVE_DDG") != "1",
    reason="WEB_SEARCH_LIVE_DDG != 1 — set it to run a real DuckDuckGo query",
)
def test_live_duckduckgo_search():
    async def run():
        tool = _build_tool({"backend": "duckduckgo", "max_results": 3})
        out = await tool.execute("python list comprehension")
        assert out["backend"] == "duckduckgo"
        assert out["from_cache"] is False
        # A real query for a common term should return SOMETHING.
        assert out["results"], "live DDG returned zero results for a common query"
        assert all(r["url"] for r in out["results"])

    asyncio.run(run())


@pytest.mark.skipif(
    not os.environ.get("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY unset — set it to run a real Tavily query",
)
def test_live_tavily_search():
    async def run():
        tool = _build_tool({"backend": "tavily", "max_results": 3})
        out = await tool.execute("python list comprehension")
        assert out["backend"] == "tavily"
        assert out["results"], "live Tavily returned zero results"

    asyncio.run(run())
