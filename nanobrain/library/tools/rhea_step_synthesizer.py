"""synthesize_rhea_step (E2-R Priority 1) — tool name → nanobrain Step.

The missing seam: turn "a Rhea/Galaxy tool name" into a from_config-ready
nanobrain ``Step`` config that drops straight into a ``WorkflowBuilder``
DAG, with DETERMINISTIC provenance (carried by the discovered UTD) — the
core ask of E2-R.

Flow
----

1. Connect to a Rhea MCP worker (``RheaMCPDiscovery``).
2. (Optional) ``find_tools(query)`` first — Rhea's catalog is DYNAMIC; a
   tool is not in ``tools/list`` until a semantic query surfaces it into
   the session catalog.
3. ``discover()`` → UTD dicts (each carries the honest determinism pins
   read from the worker's ``apecx_provenance`` annotation — real version,
   container ref/digest, R1/R2/R3, side-effects).
4. Select the UTD matching ``tool_name``; FAIL LOUD if absent.
5. Branch file-vs-JSON on the UTD's ``file_input_args`` discriminator:
   - a tool with file (Galaxy ``type="data"``) inputs → a
     ``RheaFileToolStep`` config (ProxyStore file staging).
   - a pure-JSON tool → a ``ToolExecutionStep`` config (backend=rhea,
     dispatched by ``RheaAdapter``).

Honesty / no-silent-failure discipline
---------------------------------------

The file-vs-JSON branch is the place a wrong guess does real damage (a
JSON config for a file tool would pass a raw string where the tool needs
a staged redis_key, or vice-versa). So:

- The discriminator is the AUTHORITATIVE ``file_input_args`` the worker
  surfaced (Galaxy ``type="data"`` params) — NOT a heuristic on the
  inputSchema (which serializes a file param as an indistinguishable
  ``string``).
- When the worker did NOT surface it (an old worker, no apecx_provenance
  annotation) AND the caller passed no explicit ``file_input_args``
  override, synthesis FAILS LOUD — it never guesses.
- ``RheaFileToolStep`` v1 dispatches exactly ONE file argument; a tool
  with multiple file inputs FAILS LOUD (honest scope limit) rather than
  silently dropping the extras.

The synthesized step's determinism is exactly what the worker reported:
a versioned + containerized tool is R2, an unpinned tool is R3@unpinned,
a flagged-stochastic tool is R3 — never a fabricated R3@1.0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.tools.rhea_discovery import RheaMCPDiscovery

_TOOL_EXECUTION_STEP_CLASS = (
    "nanobrain.library.steps.tool_execution_step.ToolExecutionStep"
)
_RHEA_FILE_TOOL_STEP_CLASS = (
    "nanobrain.library.steps.rhea_file_tool_step.RheaFileToolStep"
)
# Mirrors rhea_discovery._UNPINNED_VERSION; a synthesized step over an
# unpinned tool is honestly labelled, but is still runnable.
_UNPINNED_VERSION = "unpinned"


@dataclass
class RheaStepSpec:
    """A synthesized, from_config-ready nanobrain Step config.

    Consume via ``WorkflowBuilder.add_rhea_step(name, spec)`` (the sync DAG
    assembly seam) or read ``step_class`` / ``step_config`` directly to
    hand-author the step YAML.
    """

    #: Dotted import path of the Step class (ToolExecutionStep or
    #: RheaFileToolStep).
    step_class: str
    #: Config fields for the step (passed to add_step kwargs / written to
    #: the step's YAML at WorkflowBuilder.load()).
    step_config: dict[str, Any]
    #: True when the synthesized step stages a file input (RheaFileToolStep).
    uses_file_input: bool
    #: The discovered UTD's descriptor_id (carries the honest version pin).
    descriptor_id: str
    #: The full discovered UTD dict — provenance pins included.
    utd: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def is_pinned(self) -> bool:
        """True when the tool carried a real version (not @unpinned)."""
        return not self.descriptor_id.endswith(f"@{_UNPINNED_VERSION}")


def _select_utd(utds: list[dict[str, Any]], tool_name: str) -> dict[str, Any]:
    """Find the UTD matching ``tool_name`` or FAIL LOUD.

    Matches (in priority order): the original MCP tool name preserved in
    ``provenance_pin.mcp_support.rhea_tool_name``, then the sanitized
    descriptor tool_id, then a case-insensitive name compare.
    """
    sanitized = RheaMCPDiscovery._sanitize_tool_id(tool_name)
    by_mcp_name: dict[str, dict] = {}
    by_tool_id: dict[str, dict] = {}
    for u in utds:
        pin = u.get("provenance_pin") or {}
        mcp_support = pin.get("mcp_support") or {}
        raw = mcp_support.get("rhea_tool_name")
        if raw:
            by_mcp_name[str(raw)] = u
        did = str(u.get("descriptor_id", ""))
        # descriptor_id is rhea:<tool_id>@<version>
        if ":" in did and "@" in did:
            tool_id = did.split(":", 1)[1].rsplit("@", 1)[0]
            by_tool_id[tool_id] = u

    if tool_name in by_mcp_name:
        return by_mcp_name[tool_name]
    if sanitized in by_tool_id:
        return by_tool_id[sanitized]
    # case-insensitive last resort
    lowered = tool_name.lower()
    for raw, u in by_mcp_name.items():
        if raw.lower() == lowered:
            return u

    available = sorted(by_mcp_name.keys())
    raise ComponentConfigurationError(
        f"FAIL-FAST: synthesize_rhea_step could not find a Rhea tool named "
        f"{tool_name!r} in the worker catalog. Available tool names: "
        f"{available}. If the tool is dynamic, pass find_tools_query= so it "
        f"is surfaced into the session catalog before discovery."
    )


def _resolve_file_input_args(
    utd: dict[str, Any], override: list[str] | None
) -> list[str]:
    """Resolve the file-vs-JSON discriminator, FAIL LOUD when unknown.

    An explicit caller override always wins. Otherwise read the worker's
    authoritative ``file_input_args``. When neither is available the worker
    is too old to disambiguate — FAIL LOUD rather than guess.
    """
    if override is not None:
        return list(override)
    mcp_support = (utd.get("provenance_pin") or {}).get("mcp_support") or {}
    file_args = mcp_support.get("file_input_args")
    if file_args is None:
        raise ComponentConfigurationError(
            f"FAIL-FAST: synthesize_rhea_step cannot determine whether "
            f"{utd.get('descriptor_id')!r} takes FILE inputs or JSON inputs. "
            f"The Rhea worker did not surface 'file_input_args' (its "
            f"apecx_provenance annotation is missing — likely an older "
            f"worker). Refusing to guess (a wrong guess passes a raw string "
            f"where a staged file is required, or vice-versa). Fix: upgrade "
            f"the Rhea worker, OR pass an explicit file_input_args=[...] "
            f"(empty list = JSON tool)."
        )
    return list(file_args)


async def synthesize_rhea_step(
    tool_name: str,
    *,
    mcp_url: str | None = None,
    find_tools_query: str | None = None,
    file_input_args: list[str] | None = None,
    static_tool_args: dict[str, Any] | None = None,
    output_file_args: list[str] | None = None,
    timeout_seconds: float = 30.0,
) -> RheaStepSpec:
    """Synthesize a nanobrain Step config for a Rhea/Galaxy tool by name.

    Args:
        tool_name: The Rhea/Galaxy tool name (the MCP-side name, e.g.
            ``"muscle"``).
        mcp_url: Rhea MCP endpoint. When None, ``$RHEA_MCP_URL`` is used
            (FAIL LOUD if unset).
        find_tools_query: Optional semantic query passed to Rhea's
            ``find_tools`` BEFORE discovery, so a dynamic tool is surfaced
            into the session catalog. Also used as the file-step's own
            ``find_tools_query`` at run time. Strongly recommended — Rhea's
            catalog is dynamic.
        file_input_args: Optional explicit file-vs-JSON override. An empty
            list forces the JSON (ToolExecutionStep) path; a non-empty list
            forces the file (RheaFileToolStep) path. When omitted, the
            worker's authoritative discriminator is used (FAIL LOUD if the
            worker did not surface it).
        static_tool_args: Non-file tool arguments forwarded verbatim by a
            file step (ignored for the JSON path).
        output_file_args: Output file names a file step fetches back
            (empty/None = all). Ignored for the JSON path.
        timeout_seconds: Per-MCP-call timeout for discovery.

    Returns:
        A :class:`RheaStepSpec`.

    Raises:
        ComponentConfigurationError: tool not found, file-vs-JSON
            undeterminable, or a multi-file tool (file-step v1 limit).
    """
    if mcp_url:
        disco = RheaMCPDiscovery(mcp_url=mcp_url, timeout_seconds=timeout_seconds)
    else:
        disco = RheaMCPDiscovery.from_env(timeout_seconds=timeout_seconds)

    try:
        # Surface a dynamic tool into the session catalog first.
        if find_tools_query:
            await disco.transport.call(
                "tools/call",
                {"name": "find_tools", "arguments": {"query": find_tools_query}},
            )
        utds = await disco.discover()
    finally:
        await disco.aclose()

    utd = _select_utd(utds, tool_name)
    descriptor_id = str(utd.get("descriptor_id", ""))
    mcp_support = (utd.get("provenance_pin") or {}).get("mcp_support") or {}
    rhea_tool_name = mcp_support.get("rhea_tool_name") or tool_name

    resolved_file_args = _resolve_file_input_args(utd, file_input_args)

    if not resolved_file_args:
        # JSON tool → ToolExecutionStep, backend=rhea (RheaAdapter).
        step_config: dict[str, Any] = {"tool_descriptor": utd}
        return RheaStepSpec(
            step_class=_TOOL_EXECUTION_STEP_CLASS,
            step_config=step_config,
            uses_file_input=False,
            descriptor_id=descriptor_id,
            utd=utd,
        )

    # File tool → RheaFileToolStep. v1 dispatches exactly one file arg.
    if len(resolved_file_args) > 1:
        raise ComponentConfigurationError(
            f"FAIL-FAST: tool {descriptor_id!r} declares multiple file "
            f"inputs {resolved_file_args}; RheaFileToolStep v1 stages "
            f"exactly ONE file argument. Synthesize a custom multi-file step "
            f"or pass file_input_args=[<one>] to pick the primary input."
        )

    file_step_config: dict[str, Any] = {
        "tool_name": str(rhea_tool_name),
        "find_tools_query": find_tools_query or utd.get("summary") or str(rhea_tool_name),
        "file_input_arg": resolved_file_args[0],
        "static_tool_args": dict(static_tool_args or {}),
        "output_file_args": list(output_file_args or []),
    }
    if mcp_url:
        file_step_config["mcp_url"] = mcp_url
    return RheaStepSpec(
        step_class=_RHEA_FILE_TOOL_STEP_CLASS,
        step_config=file_step_config,
        uses_file_input=True,
        descriptor_id=descriptor_id,
        utd=utd,
    )


__all__ = ["RheaStepSpec", "synthesize_rhea_step"]
