"""The unified find-and-establish seam — acquire a tool from a source.

This module is the single entry point for the question "I have a *need*;
give me the tool(s) that satisfy it, as :class:`UnifiedToolDescriptor`
records I can dispatch." A *source* is anything that can surface tools:
a Rhea MCP worker, or a custom in-process backend (PyMOL, a local Python
callable, an HTTP service). The seam is deliberately source-agnostic —
**Rhea is one source, not THE source**; custom backends are first-class
citizens of the same API.

Two paths, one return type
---------------------------

* **Rhea path** (``rhea_transport`` given): SURFACE the matching tools
  into the Rhea session catalog via ``find_tools``, then build UTDs by
  parsing that ``find_tools`` result (reusing
  :meth:`RheaMCPDiscovery._mcp_tool_to_utd`, the wire→UTD converter).
  When the result shape can't be parsed into MCP tool dicts, fall back
  to a ``tools/list`` over the **same** (find_tools-populated) session.
* **Custom path** (no ``rhea_transport``): resolve a registered
  :class:`ToolBackendAdapter` from :class:`ToolBackendRegistry`; if it is
  :class:`Establishable`, ESTABLISH it (e.g. auto-build its docker image)
  via ``ensure_established`` BEFORE returning its UTD. Any
  ``BACKEND_NAME``-registered adapter rides this path; PyMOL is the first
  (a docker source — it establishes by building ``apecx-pymol:3.1.0`` from
  its Dockerfile). This is what makes docker a first-class establishment
  source alongside Rhea, not a bare registry lookup.

Honesty / FAIL-LOUD discipline (workspace CLAUDE.md):

* The Rhea path lets :class:`MCPTransport`'s existing errors propagate
  unchanged — an unreachable Rhea worker FAILS LOUD, it is never
  swallowed into an empty list.
* The custom path lets ``ToolBackendRegistry.get`` raise ``KeyError``
  when the backend is not registered, and lets ``ensure_established``
  (e.g. a docker build failure) propagate — a misnamed backend or an
  unbuildable source is a loud error, not a silent no-op.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.steps.tool_execution_step import ToolBackendRegistry
from nanobrain.library.tools._mcp_transport import (
    MCPTransport,
    parse_tool_call_result,
)
from nanobrain.library.tools.rhea_discovery import RheaMCPDiscovery

logger = logging.getLogger(__name__)

# Keys a find_tools result might carry the matched-tool list under. The
# first list-valued one wins; if none match (and the result is not itself
# a list / single tool dict) we fall back to a tools/list round-trip.
_TOOL_LIST_KEYS = ("tools", "results", "matches", "items", "found_tools")


@runtime_checkable
class Establishable(Protocol):
    """A tool adapter that can ESTABLISH itself from its declared source.

    Building its docker image, cloning+building a repo, conda-installing — any
    one-time provisioning that must happen before the tool can run. The
    find-and-establish seam's custom path calls ``ensure_established`` so a
    source-backed tool is provisioned THROUGH the seam, the same way Rhea tools
    are established via ``find_tools`` — not self-provisioned around it.
    """

    async def ensure_established(self, *, on_progress: Any = None) -> None: ...


async def find_and_establish_tool(
    need: str,
    *,
    rhea_transport: MCPTransport | None = None,
    version: str = "0.0.0",
    on_progress: Any = None,
) -> list[UnifiedToolDescriptor]:
    """Find and establish the tool(s) that satisfy ``need``.

    Args:
        need: For the Rhea path, a semantic ``find_tools`` query. For the
            custom path, a ``"<backend>:<tool_id>"`` identifier (or a
            bare registered ``BACKEND_NAME``, in which case the backend
            name doubles as the tool_id).
        rhea_transport: A live :class:`MCPTransport` against a Rhea MCP
            worker. When given, the Rhea path runs; when ``None``, the
            custom-backend path runs.
        version: The descriptor version token for the custom path's UTD
            (the Rhea path takes the version from the worker's
            provenance block). Ignored on the Rhea path.
        on_progress: Optional callback forwarded to a custom backend's
            ``ensure_established`` (e.g. the docker-build progress line).
            Ignored on the Rhea path.

    Returns:
        A list of :class:`UnifiedToolDescriptor`. The Rhea path returns
        every tool ``find_tools`` surfaced for ``need``; the custom path
        returns exactly one UTD for the named backend tool.

    Raises:
        ComponentConfigurationError: on a malformed ``need`` (custom
            path) or anything :class:`MCPTransport` raises (Rhea path).
        KeyError: when ``need`` names an unregistered backend (custom
            path) — surfaced verbatim from ``ToolBackendRegistry.get``.
    """
    if rhea_transport is not None:
        return await _find_and_establish_rhea(need, rhea_transport)
    return await _establish_custom(need, version, on_progress)


# ---------------------------------------------------------------------------
# Rhea path
# ---------------------------------------------------------------------------

async def _find_and_establish_rhea(
    need: str, rhea_transport: MCPTransport
) -> list[UnifiedToolDescriptor]:
    """Surface ``find_tools`` matches into the session catalog, return UTDs.

    Reuses :meth:`RheaMCPDiscovery._mcp_tool_to_utd` as the wire→UTD
    converter and, for the fallback, ``RheaMCPDiscovery.discover()`` —
    both bound to the CALLER's transport so the ``tools/list`` fallback
    runs over the SAME session ``find_tools`` just populated (Rhea's
    tool catalog is session-scoped; a fresh session would not see the
    surfaced tools).
    """
    raw = await rhea_transport.call(
        "tools/call",
        {"name": "find_tools", "arguments": {"query": need}},
    )
    parsed = parse_tool_call_result(raw, "find_tools")

    # RheaMCPDiscovery owns the wire→UTD conversion. Reuse the caller's
    # live, find_tools-populated transport for both the converter's
    # mcp_url and the discover() fallback (same session). The discovery's
    # own (lazily-opened) transport is never touched, so nothing leaks;
    # the caller owns aclose() of rhea_transport.
    discovery = RheaMCPDiscovery(mcp_url=rhea_transport.mcp_url)
    discovery._transport = rhea_transport

    tool_dicts = _extract_tool_dicts(parsed)
    if tool_dicts:
        utd_dicts = [discovery._mcp_tool_to_utd(t) for t in tool_dicts]
    else:
        # The find_tools result shape wasn't directly parseable into MCP
        # tool dicts — fall back to a tools/list over the same session.
        logger.debug(
            "find_and_establish_tool: find_tools result not directly "
            "parseable (type=%s); falling back to tools/list",
            type(parsed).__name__,
        )
        utd_dicts = await discovery.discover()

    return [UnifiedToolDescriptor.from_dict(d) for d in utd_dicts]


def _extract_tool_dicts(parsed: Any) -> list[dict[str, Any]]:
    """Normalize a find_tools result into a list of MCP tool dicts.

    Accepts a bare list of tool dicts, a single tool dict, or a wrapper
    dict carrying the list under one of :data:`_TOOL_LIST_KEYS`. Returns
    only entries that carry a ``name`` (the minimum
    ``_mcp_tool_to_utd`` requires). An empty return signals the caller
    to fall back to ``tools/list``.
    """
    if isinstance(parsed, list):
        return [t for t in parsed if isinstance(t, dict) and t.get("name")]
    if isinstance(parsed, dict):
        if parsed.get("name") and ("inputSchema" in parsed or "annotations" in parsed):
            return [parsed]
        for key in _TOOL_LIST_KEYS:
            val = parsed.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict) and t.get("name")]
    return []


# ---------------------------------------------------------------------------
# Custom-backend path
# ---------------------------------------------------------------------------

async def _establish_custom(
    need: str, version: str, on_progress: Any = None
) -> list[UnifiedToolDescriptor]:
    """Establish (if the adapter supports it) and return a minimal UTD for a
    registered custom backend tool.

    ``need`` is ``"<backend>:<tool_id>"`` or a bare ``BACKEND_NAME`` (the
    backend name then doubles as the tool_id). The adapter is resolved from
    :class:`ToolBackendRegistry` (``KeyError`` FAIL-LOUD when absent); if it is
    :class:`Establishable` it is ESTABLISHED (e.g. its docker image is
    auto-built) before the UTD is returned. The UTD's
    ``provenance_pin.class_path`` pins the adapter's importable class path so
    the tool can be re-materialized.
    """
    if ":" in need:
        backend, tool_id = need.split(":", 1)
    else:
        backend = tool_id = need
    backend = backend.strip()
    tool_id = tool_id.strip()
    if not backend or not tool_id:
        raise ComponentConfigurationError(
            f"FAIL-FAST: find_and_establish_tool custom path needs a "
            f"'<backend>:<tool_id>' (or bare BACKEND_NAME) identifier; "
            f"got {need!r}"
        )

    # FAIL-LOUD KeyError when the backend is not registered.
    adapter = ToolBackendRegistry.get(backend)
    # ESTABLISH from the adapter's own source (e.g. docker auto-build) BEFORE
    # handing back the UTD — this is what makes the custom path a real
    # find-AND-establish, not a bare lookup. Non-Establishable adapters skip it.
    if isinstance(adapter, Establishable):
        await adapter.ensure_established(on_progress=on_progress)
    adapter_cls = type(adapter)
    class_path = f"{adapter_cls.__module__}.{adapter_cls.__qualname__}"

    utd = UnifiedToolDescriptor.from_dict(
        {
            "descriptor_id": f"{backend}:{tool_id}@{version}",
            "display_name": tool_id,
            "summary": f"{backend} backend tool {tool_id}",
            "inputs": [],
            "outputs": [],
            "provenance_pin": {"class_path": class_path},
        }
    )
    return [utd]


__all__ = ["Establishable", "find_and_establish_tool"]
