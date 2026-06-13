"""RheaMCPDiscovery — discover a Rhea MCP server's tool catalog as UTDs.

A thin MCP client that connects to a running Rhea worker, calls
``tools/list``, and converts each discovered MCP tool into a
``UnifiedToolDescriptor``-shaped dict. The intended consumer is a
code generator / workflow composer: discover Rhea's tools at
generation time, then wire the matching tool into a workflow via
``ToolExecutionStep`` + the ``RheaAdapter`` backend.

This is the "code generator uses Rhea as an MCP server" mechanism —
the generator is itself an MCP client of Rhea.

Why the wire shape, not Rhea's in-process objects
-------------------------------------------------

The Rhea fork ships ``rhea.extensions.apecx_utd_extension.utd_producer``
which converts Rhea's in-process ``Tool`` objects to UTD dicts. This
module does the equivalent from the *wire* shape (the MCP
``tools/list`` response) instead — so it has zero ``rhea/`` import
dependency. The MCP ``tools/list`` JSON is the cross-framework
contract; both sides agree on it. Use this module when you only have
the MCP endpoint, not the Rhea process.

Honest scope (v1)
-----------------

* ``tools/list`` only (no ``resources/list`` / ``prompts/list``).
* The generated UTD is minimal-but-valid: ``descriptor_id`` =
  ``rhea:<sanitized_tool_id>@<version>``, ``inputs`` derived from the
  MCP tool's ``inputSchema.properties``, a single ``return`` output
  (MCP ``tools/list`` carries no output schema in the base spec).
* FAIL-FAST on: ``$RHEA_MCP_URL`` unset, empty tool catalog, missing
  ``tools`` array, and anything ``MCPTransport`` raises. An empty
  catalog is a LOUD error — a generator handed zero tools would emit
  an empty/no-op workflow.
* Shares the MCP wire protocol with ``RheaAdapter`` +
  ``RheaMCPDispatcher`` via ``MCPTransport``.
"""

from __future__ import annotations

import os
import re
from typing import Any

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.tools._mcp_transport import MCPTransport

_RHEA_ENV_VAR = "RHEA_MCP_URL"
_RHEA_ADAPTER_CLASS_PATH = "nanobrain.library.tools.rhea_adapter.RheaAdapter"
# UTD tool_id grammar: [a-z][a-z0-9_.]* — sanitize MCP names to fit.
_TOOL_ID_SANITIZE = re.compile(r"[^a-z0-9_.]")
# UTD version token grammar (descriptor_id @<version> segment):
# [0-9A-Za-z-_.+]+ — sanitize a real tool version to fit, never fabricate.
_VERSION_SANITIZE = re.compile(r"[^0-9A-Za-z\-_.+]")
# The key the Rhea apecx extension writes its determinism/provenance block
# under, inside the MCP ToolAnnotations field. Mirrors
# rhea.extensions.apecx_utd_extension.provenance_annotations.APECX_PROVENANCE_KEY.
_APECX_PROVENANCE_KEY = "apecx_provenance"
# Explicit "we could not pin this tool" version token. Honest — NOT a
# fabricated '1.0.0' that lies about provenance.
_UNPINNED_VERSION = "unpinned"


class RheaMCPDiscovery:
    """Connects to a Rhea MCP server and discovers its tools as UTD dicts."""

    def __init__(
        self,
        *,
        mcp_url: str,
        timeout_seconds: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        default_tool_version: str = _UNPINNED_VERSION,
    ) -> None:
        if not mcp_url or not isinstance(mcp_url, str):
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDiscovery requires a non-empty mcp_url; "
                f"got {mcp_url!r}"
            )
        self._default_tool_version = default_tool_version
        self._transport = MCPTransport(
            mcp_url=mcp_url,
            timeout_seconds=timeout_seconds,
            extra_headers=extra_headers,
            client_name="nanobrain-rhea-discovery",
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> RheaMCPDiscovery:
        """Build from ``$RHEA_MCP_URL``; FAIL-FAST when unset."""
        url = os.environ.get(_RHEA_ENV_VAR)
        if not url:
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaMCPDiscovery.from_env requires ${_RHEA_ENV_VAR} "
                f"(e.g. export {_RHEA_ENV_VAR}='http://localhost:3001/mcp/'). "
                f"The Rhea worker must be running and reachable."
            )
        return cls(mcp_url=url, **kwargs)

    # Test seam — swap in an httpx.MockTransport-backed client.
    @property
    def transport(self) -> MCPTransport:
        return self._transport

    async def discover(self) -> list[dict[str, Any]]:
        """Return a list of UTD-shaped dicts for every tool Rhea exposes.

        FAIL-FAST if Rhea reports zero tools — a generator handed an
        empty tool catalog would emit an empty/no-op workflow.
        """
        result = await self._transport.call("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        if tools is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: MCP tools/list result had no 'tools' array: "
                f"{result!r}"
            )
        if not tools:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Rhea MCP server at {self._transport.mcp_url} "
                f"returned an empty tools list. A code generator cannot "
                f"compose a workflow with zero tools. Check that the Rhea "
                f"worker is loaded with the expected tool catalog."
            )
        return [self._mcp_tool_to_utd(t) for t in tools]

    async def aclose(self) -> None:
        await self._transport.aclose()

    # ---- MCP tool -> UTD conversion --------------------------------------

    def _mcp_tool_to_utd(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Convert one MCP ``tools/list`` entry into a UTD dict.

        Reads the determinism/provenance block the Rhea apecx extension
        surfaces under ``annotations.apecx_provenance`` (E2-R Priority 2):
        the real tool version, container refs/digests, and the file-vs-JSON
        ``file_input_args`` discriminator. The resulting UTD is HONEST:
        a tool the worker pinned carries its real version + container; a
        tool the worker could NOT pin (or an old worker with no apecx block)
        stays explicitly ``@unpinned`` with determinism ``R3`` — never a
        fabricated ``R3@1.0.0`` that lies about provenance.
        """
        raw_name = str(tool.get("name", "")).strip()
        if not raw_name:
            raise ComponentConfigurationError(
                f"FAIL-FAST: Rhea MCP tool entry has no 'name': {tool!r}"
            )
        tool_id = self._sanitize_tool_id(raw_name)

        prov = self._extract_provenance_block(tool)
        version = self._resolve_version(prov)
        descriptor_id = f"rhea:{tool_id}@{version}"

        digest, container_ref = _split_container_digest(
            prov.get("containers") if prov else None
        )
        determinism = _honest_determinism(prov, version, digest, container_ref)
        side_effects = _honest_side_effects(digest, container_ref)

        # file_input_args is the file-vs-JSON discriminator the synthesizer
        # branches on. None => the worker did NOT surface it (old worker);
        # the synthesizer FAILS LOUD rather than guess. A present (possibly
        # empty) list is authoritative.
        file_input_args = prov.get("file_input_args") if prov else None

        input_schema = tool.get("inputSchema") or {}
        properties = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or [])
        inputs = []
        for prop_name, prop_spec in properties.items():
            # ``has_default`` preserves whether the schema DECLARED a default
            # at all — ``default`` alone cannot distinguish an absent default
            # from an explicit ``default: null``. The step synthesizer needs
            # this to tell a required-no-default param (FAIL LOUD) apart from
            # a param whose declared default happens to be null.
            inputs.append(
                {
                    "name": prop_name,
                    "type": _json_type_to_utd_type(prop_spec.get("type")),
                    "description": str(prop_spec.get("description", "")),
                    "required": prop_name in required,
                    "has_default": "default" in prop_spec,
                    "default": prop_spec.get("default"),
                }
            )

        # mcp_support carries MCP plumbing + the determinism evidence that
        # has no first-class UTD field (container REF as opposed to digest,
        # version_command, requirements, file_input_args). RheaAdapter and
        # the step synthesizer read these back.
        mcp_support: dict[str, Any] = {
            "rhea_tool_name": raw_name,
            "discovered_from": self._transport.mcp_url,
        }
        if file_input_args is not None:
            mcp_support["file_input_args"] = list(file_input_args)
        if container_ref:
            mcp_support["container_image_ref"] = container_ref
        if prov:
            if prov.get("version_command"):
                mcp_support["version_command"] = prov["version_command"]
            if prov.get("requirements"):
                mcp_support["requirements"] = prov["requirements"]
            mcp_support["determinism_pinned"] = (
                version != _UNPINNED_VERSION and bool(digest or container_ref)
            )

        provenance_pin: dict[str, Any] = {
            "class_path": _RHEA_ADAPTER_CLASS_PATH,
            "mcp_support": mcp_support,
        }
        # Only set the digest field when we actually have a digest — a
        # mutable tag ref is NOT a digest and must not masquerade as one.
        if digest:
            provenance_pin["container_image_digest"] = digest

        return {
            "descriptor_id": descriptor_id,
            "display_name": tool.get("title") or raw_name,
            "summary": str(tool.get("description", "") or f"Rhea tool {raw_name}")[:200],
            "long_description": str(tool.get("description", "") or ""),
            "inputs": inputs,
            "outputs": [
                {
                    "name": "return",
                    "type": "object",
                    "description": "MCP tool result (Rhea base spec carries no output schema).",
                }
            ],
            "side_effects": side_effects,
            "determinism": determinism,
            "resource_class": "cpu_light",
            "provenance_pin": provenance_pin,
        }

    @staticmethod
    def _extract_provenance_block(tool: dict[str, Any]) -> dict[str, Any] | None:
        """Return the apecx_provenance block from a tool's annotations.

        Returns None when the worker did not surface it (old worker, or a
        tool whose annotation build failed). A None block is honest:
        downstream treats the tool as unpinned + file-vs-JSON-unknown.
        """
        annotations = tool.get("annotations")
        if not isinstance(annotations, dict):
            return None
        block = annotations.get(_APECX_PROVENANCE_KEY)
        return block if isinstance(block, dict) else None

    def _resolve_version(self, prov: dict[str, Any] | None) -> str:
        """Honest descriptor version.

        A real version from the worker wins (sanitized to the UTD version
        grammar). Empty / missing => the configured default (``unpinned``).
        """
        raw = (prov or {}).get("tool_version") or ""
        raw = str(raw).strip()
        if not raw:
            return self._default_tool_version
        sanitized = _VERSION_SANITIZE.sub("_", raw)
        return sanitized or self._default_tool_version

    @staticmethod
    def _sanitize_tool_id(raw_name: str) -> str:
        """Coerce an MCP tool name into the UTD tool_id grammar
        ``[a-z][a-z0-9_.]*``."""
        lowered = raw_name.lower()
        sanitized = _TOOL_ID_SANITIZE.sub("_", lowered)
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "t_" + sanitized
        return sanitized


def _split_container_digest(
    containers: list[dict[str, Any]] | None,
) -> tuple[str | None, str | None]:
    """Split a container list into (digest, ref).

    A value carrying an OCI digest (``...@sha256:<hex>`` or a bare
    ``sha256:<hex>``) is an immutable pin → returned as ``digest``. A plain
    ``image:tag`` is MUTABLE — returned as ``ref`` only, never as a digest
    (a tag masquerading as a digest is exactly the false-provenance shape
    this code refuses). Returns ``(None, None)`` when there are no
    containers.
    """
    if not containers:
        return None, None
    ref: str | None = None
    for cont in containers:
        if not isinstance(cont, dict):
            continue
        val = str(cont.get("value", "") or "").strip()
        if not val:
            continue
        if "@sha256:" in val or val.startswith("sha256:"):
            return val, val
        if ref is None:
            ref = val
    return None, ref


def _honest_determinism(
    prov: dict[str, Any] | None,
    version: str,
    digest: str | None,
    container_ref: str | None,
) -> str:
    """Honest DeterminismClass from real evidence — never blanket R3.

    - Explicitly-flagged stochastic tool (sampling / ML / random seed) → R3.
    - Versioned AND containerized → R2 (a versioned binary run in a pinned
      container is reproducible up to floating point; we do NOT assert R1
      bit-exactness, which Galaxy metadata cannot prove).
    - Otherwise (unpinned / unknown) → R3 (we cannot claim reproducibility).

    R2 is the strongest HONEST claim from Galaxy metadata; R3 here means
    "unknown / unpinned", paired with an ``@unpinned`` version so the pair
    reads coherently.
    """
    if prov is None:
        return "R3"
    if prov.get("stochastic"):
        return "R3"
    if version != _UNPINNED_VERSION and (digest or container_ref):
        return "R2"
    return "R3"


def _honest_side_effects(digest: str | None, container_ref: str | None) -> str:
    """Honest SideEffectClass — never blanket 'network'.

    A containerized Galaxy tool reads its inputs and writes its outputs
    inside its own container/ProxyStore → ``filesystem_write``. A tool with
    NO container (a pure MCP function: find_tools, a search API) reaches out
    over the network → ``network``. This is a per-tool read of the
    available evidence, not a one-size-fits-all default.
    """
    if digest or container_ref:
        return "filesystem_write"
    return "network"


def _json_type_to_utd_type(json_type: Any) -> str:
    """Map a JSON-schema type to the UTD type vocabulary."""
    mapping = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    if isinstance(json_type, str):
        return mapping.get(json_type, "object")
    return "object"


__all__ = ["RheaMCPDiscovery"]
