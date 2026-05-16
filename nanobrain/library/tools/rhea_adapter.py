"""RheaAdapter — the ``ToolBackendAdapter`` (``BACKEND_NAME="rhea"``).

``ToolExecutionStep`` (G11) dispatches a UTD-described tool to a
``ToolBackendAdapter`` looked up from ``ToolBackendRegistry`` by the
UTD's backend prefix. nanobrain ships ``HTTPBackendAdapter`` and
``LocalParslAdapter``; ``RheaAdapter`` is the third concrete adapter —
the ``rhea`` backend. It dispatches to a Rhea MCP worker over the MCP
streamable-HTTP transport.

(Historical note: an earlier ``CLAUDE.md`` claimed the Rhea adapter
"ships from the Rhea fork (Track C T-RH-04)" — it never did. It was
first built apecx-side as a workaround, then promoted here, its
canonical home, alongside the other ``ToolBackendAdapter``s.)

RheaAdapter vs. RheaMCPDispatcher
---------------------------------

Both dispatch a tool to a Rhea MCP worker; they differ in WHO calls
them:

* ``RheaMCPDispatcher`` is a ``ToolBase`` — the Agent path. You
  materialize ONE dispatcher per tool via
  ``ToolBase.from_descriptor(utd, mcp_url=, rhea_tool_name=)`` and an
  Agent holds the Tool.
* ``RheaAdapter`` is a ``ToolBackendAdapter`` — the Step path. ONE
  adapter per Rhea endpoint, registered once with
  ``ToolBackendRegistry``; ``ToolExecutionStep`` consults the registry
  and calls ``adapter.invoke(utd, inputs, ...)``.

Both share the MCP wire protocol via
``nanobrain.library.tools._mcp_transport.MCPTransport``.

How a UTD names its Rhea tool
-----------------------------

``ToolExecutionStep`` resolves the backend from
``utd.descriptor_backend`` — the ``<backend>:`` prefix of the
``descriptor_id``. For a Rhea tool the ``descriptor_id`` is
``rhea:<tool_id>@<version>`` (e.g. ``rhea:sequence.analyze@1.0.0``).
The adapter strips the ``rhea:`` prefix and the ``@version`` suffix to
get the MCP-side tool name. When the MCP-side name differs from the
sanitized ``tool_id`` (``RheaMCPDiscovery`` sanitizes names like
``UniProt-Search`` → ``uniprot_search``), the original is read from
``utd.provenance_pin.mcp_support['rhea_tool_name']``.

Configuration
-------------

The adapter is constructed once per Rhea endpoint and registered::

    from nanobrain.library.tools.rhea_adapter import RheaAdapter
    from nanobrain.library.steps.tool_execution_step import ToolBackendRegistry

    adapter = RheaAdapter(mcp_url="http://localhost:3001/mcp/")
    ToolBackendRegistry.register(adapter)

``$RHEA_MCP_URL`` is the canonical env var; ``RheaAdapter.from_env()``
builds + registers in one call and FAIL-FASTs when the var is unset.

Honest scope (v1)
-----------------

* Synchronous ``tools/call`` via ``MCPTransport`` — streaming /
  multi-event responses are NOT handled (same scope cut as
  ``RheaMCPDispatcher`` v1).
* FAIL-FAST on: non-empty-url violation, missing UTD-required inputs,
  and anything ``MCPTransport`` raises (non-200, JSON-RPC error,
  missing session header, ``isError`` result).
* Does NOT re-validate the UTD shape (``ToolExecutionStep`` already
  did). It DOES check every UTD-required input is present in
  ``inputs``.
"""

from __future__ import annotations

import os
from typing import Any

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.steps.tool_execution_step import (
    ToolBackendAdapter,
    ToolBackendRegistry,
)
from nanobrain.library.tools._mcp_transport import (
    MCPTransport,
    parse_tool_call_result,
)

_RHEA_ENV_VAR = "RHEA_MCP_URL"


class RheaAdapter(ToolBackendAdapter):
    """ToolBackendAdapter that dispatches UTD tools to a Rhea MCP worker."""

    BACKEND_NAME = "rhea"

    def __init__(
        self,
        *,
        mcp_url: str,
        timeout_seconds: float = 30.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        # MCPTransport FAIL-FASTs on an empty mcp_url; this surfaces the
        # same check with the adapter's name in the message.
        if not mcp_url or not isinstance(mcp_url, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaAdapter requires a non-empty mcp_url string; "
                f"got {mcp_url!r}"
            )
        self._transport = MCPTransport(
            mcp_url=mcp_url,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            client_name="nanobrain-rhea-adapter",
        )

    # ---- construction / registration helpers ----------------------------

    @classmethod
    def from_env(cls, *, register: bool = True, **kwargs: Any) -> RheaAdapter:
        """Build a RheaAdapter from ``$RHEA_MCP_URL``; optionally register it.

        FAIL-FAST when the env var is unset — a Rhea workflow that
        silently no-ops because the endpoint is missing is exactly the
        silent-failure shape the framework's discipline forbids.
        """
        url = os.environ.get(_RHEA_ENV_VAR)
        if not url:
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaAdapter.from_env requires ${_RHEA_ENV_VAR} "
                f"to be set (e.g. export {_RHEA_ENV_VAR}="
                f"'http://localhost:3001/mcp/'). The Rhea worker must be "
                f"running and reachable."
            )
        adapter = cls(mcp_url=url, **kwargs)
        if register:
            ToolBackendRegistry.register(adapter)
        return adapter

    # Test seam — swap in an httpx.MockTransport-backed client.
    @property
    def transport(self) -> MCPTransport:
        return self._transport

    # ---- ToolBackendAdapter contract ------------------------------------

    async def invoke(
        self,
        utd: UnifiedToolDescriptor,
        inputs: dict[str, Any],
        *,
        run_context_namespace: str = "",  # noqa: ARG002 — Rhea has no per-tenant store yet
        **kwargs: Any,  # noqa: ARG002 — no Rhea-specific backend kwargs in v1
    ) -> dict[str, Any]:
        """Dispatch ``inputs`` to the Rhea MCP tool described by ``utd``.

        Returns a dict keyed by the UTD's output names. When the MCP
        tool returns a dict, it is returned as-is (keys assumed to
        match the UTD outputs). When it returns a scalar and the UTD
        declares a single output, the scalar is placed under that
        output's name.
        """
        rhea_tool_name = self._resolve_tool_name(utd)
        self._check_required_inputs(utd, inputs)

        raw = await self._transport.call(
            "tools/call",
            {"name": rhea_tool_name, "arguments": inputs},
        )
        result = parse_tool_call_result(raw, rhea_tool_name)

        output_names = self._utd_output_names(utd)
        if isinstance(result, dict):
            return result
        if len(output_names) == 1:
            return {output_names[0]: result}
        return {"result": result}

    async def aclose(self) -> None:
        """Close the underlying MCP transport. Idempotent."""
        await self._transport.aclose()

    # ---- UTD helpers ----------------------------------------------------

    @staticmethod
    def _resolve_tool_name(utd: UnifiedToolDescriptor) -> str:
        """MCP-side tool name.

        Priority: ``utd.provenance_pin.mcp_support['rhea_tool_name']``
        override, else the ``tool_id`` portion of the descriptor_id
        (``<backend>:<tool_id>@<version>`` — both the ``rhea:`` prefix
        and the ``@version`` suffix stripped).
        """
        pin = getattr(utd, "provenance_pin", None)
        mcp_support = getattr(pin, "mcp_support", None) or {}
        override = mcp_support.get("rhea_tool_name")
        if override:
            return str(override)
        descriptor_id = getattr(utd, "descriptor_id", "") or ""
        if ":" in descriptor_id:
            descriptor_id = descriptor_id.split(":", 1)[1]
        if "@" in descriptor_id:
            descriptor_id = descriptor_id.rsplit("@", 1)[0]
        return descriptor_id

    @staticmethod
    def _utd_output_names(utd: UnifiedToolDescriptor) -> list[str]:
        outputs = getattr(utd, "outputs", None) or []
        names = []
        for o in outputs:
            name = getattr(o, "name", None) or (
                o.get("name") if isinstance(o, dict) else None
            )
            if name:
                names.append(str(name))
        return names

    @staticmethod
    def _check_required_inputs(
        utd: UnifiedToolDescriptor, inputs: dict[str, Any]
    ) -> None:
        """FAIL-FAST when a UTD-required input is missing from ``inputs``."""
        declared = getattr(utd, "inputs", None) or []
        missing = []
        for spec in declared:
            if isinstance(spec, dict):
                name = spec.get("name")
                required = spec.get("required", True)
            else:
                name = getattr(spec, "name", None)
                required = getattr(spec, "required", True)
            if required and name and name not in inputs:
                missing.append(name)
        if missing:
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaAdapter.invoke missing required input(s) "
                f"{missing} for tool {getattr(utd, 'descriptor_id', '?')!r}; "
                f"got inputs {sorted(inputs)}"
            )


__all__ = ["RheaAdapter"]
