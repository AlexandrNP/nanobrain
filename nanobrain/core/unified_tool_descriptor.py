"""UnifiedToolDescriptor (G15) — typed tool-card primitive.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G15`` and
``apecx-mcp-integration/docs/tool_descriptor_contract.md §2``: a typed
shape that EVERY tool backend (Rhea, native nanobrain, GalaxyMCP) speaks.
Promotes the ``ToolConfig.tool_card`` field from a free-form dict to a
content-hash-pinned, capability-aware primitive.

This module owns the on-the-wire schema; the semantic spec lives in
``tool_descriptor_contract.md``. New tool authors should subclass
ToolBase as before; opting into the UTD path means populating
``tool_card`` with a UTD instance (or YAML dict in the UTD shape).

Design notes
------------

- ``model_config = {"extra": "forbid"}`` everywhere (workspace
  ``pydantic_extra_forbid_rule``).
- ``descriptor_hash`` is auto-computed from the canonical fields
  (descriptor_id, inputs, outputs, side_effects, determinism,
  resource_class) when set to the sentinel ``"<computed-at-load>"`` or
  left empty. Mirrors G14's auto-hash pattern.
- ``ToolBase.from_descriptor(utd)`` is a thin convenience that resolves
  the descriptor's ``provenance_pin.config_path`` and calls
  ``ToolBase.from_config(...)``. The full from_descriptor spec
  (auto-derived configs, default-minimal UTD computation) is deferred
  to follow-up tasks; this commit ships the typed schema + the
  convenience constructor.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from .config.config_base import ConfigBase


# ---------------------------------------------------------------------------
# Enum-like literals — match tool_descriptor_contract.md §2 vocabulary.
# ---------------------------------------------------------------------------

# SideEffectClass — what touching the tool does to the world. Used by
# the sandbox + capability check.
SideEffectClass = Literal[
    "none",                # pure compute, no I/O
    "filesystem_read",     # read-only filesystem access
    "filesystem_write",    # creates/modifies files
    "network",             # outbound HTTP / sockets
    "external_database",   # writes to a shared DB
    "destructive",         # may delete or overwrite irreversibly
]

# DeterminismClass — per hpc_reproducibility_spec.md three-tier model.
# R1 = bit-exact reproducible; R2 = reproducible up to floating-point;
# R3 = stochastic (e.g. LLM completion).
DeterminismClass = Literal["R1", "R2", "R3"]

# ResourceClass — coarse-grained resource bucket for HPC scheduling.
ResourceClass = Literal[
    "cpu_light",   # < 1 core, < 1 GB
    "cpu_medium",  # 1-4 cores, 1-8 GB
    "cpu_heavy",   # > 4 cores, > 8 GB
    "gpu_single",  # 1 GPU
    "gpu_multi",   # > 1 GPU
    "io_heavy",    # bottlenecked on I/O, not compute
]


# Descriptor ID grammar: <backend>:<tool_id>@<version>.
# backend ∈ {rhea, galaxy, native, ...}; tool_id is dotted lowercase.
# version is semver-ish (loose to admit non-strict registry tags).
_DESCRIPTOR_ID_RE = re.compile(
    r"^(?P<backend>[a-z][a-z0-9_]*):(?P<tool_id>[a-z][a-z0-9_.]*)"
    r"@(?P<version>[0-9A-Za-z\-_.+]+)$"  # `+` admitted for semver build metadata
)


class UTDInputSpec(ConfigBase):
    """One input slot in a tool descriptor."""
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = Field(
        ...,
        description="JSON-Schema type name OR dotted Pydantic class path. "
                    "The orchestrator's input-binding step resolves the "
                    "expected payload shape from this.",
    )
    description: str = ""
    required: bool = True
    default: Any = None


class UTDOutputSpec(ConfigBase):
    """One output slot."""
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str = ""
    # Per gap proposal: future enhancement maps `type` to the appropriate
    # DataUnit class (DataUnitMemory for small dicts; DataUnitFile for
    # paths; DataUnitProxyRef for HPC-scale). NOT enforced today.


class UTDCostEstimate(ConfigBase):
    """Coarse cost projection for HITL gating + scheduling."""
    model_config = ConfigDict(extra="forbid")

    estimated_seconds: float = Field(default=0.0, ge=0.0)
    estimated_usd: float = Field(default=0.0, ge=0.0)
    confidence: Literal["high", "medium", "low"] = "low"


class UTDFailureMode(ConfigBase):
    """A documented failure mode of the tool."""
    model_config = ConfigDict(extra="forbid")

    code: str  # short identifier, e.g. "input_too_large"
    detail: str  # one-sentence description
    detection_signal: str = ""  # how a caller knows it happened
    recovery: str = ""  # what to do


class UTDProvenancePin(ConfigBase):
    """How to materialize the tool from this descriptor.

    Per the gap proposal: ``ToolBase.from_descriptor(utd)`` reads
    ``provenance_pin.config_path`` and calls ``from_config`` on it.
    Backends may also pin a container_image_digest or tool_binary_sha256
    for reproducibility audit.
    """
    model_config = ConfigDict(extra="forbid")

    class_path: str = Field(
        ...,
        description="Dotted Python class path of the ToolBase subclass.",
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Path to a YAML config file consumed by from_config(). "
                    "When None, ToolBase.from_descriptor uses an empty "
                    "config (the descriptor itself is the config).",
    )
    container_image_digest: Optional[str] = Field(
        default=None,
        description="OCI image digest (e.g. 'sha256:abc...'). Auditable "
                    "per HPC reproducibility spec §6 — pinned by digest "
                    "not tag.",
    )


class UTDVersionEntry(ConfigBase):
    """One entry in the descriptor's version history."""
    model_config = ConfigDict(extra="forbid")

    version: str
    published_at: str  # ISO-8601 timestamp
    deprecated: bool = False
    notes: str = ""


def compute_descriptor_hash(
    *,
    descriptor_id: str,
    inputs: List[Dict[str, Any]],
    outputs: List[Dict[str, Any]],
    side_effects: str,
    determinism: str,
    resource_class: str,
) -> str:
    """G15 — canonical descriptor-content hash.

    Covers the fields that determine the tool's BEHAVIOR (descriptor_id
    + I/O contract + side-effect class + determinism + resource class).
    NOT included: display_name, summary, long_description, cost_estimate,
    failure_modes, provenance_pin, requires_capability, version_history
    — these are descriptive or operational; tampering with them does
    NOT change the LLM-visible tool contract.

    Used by:
    - ``apecx-bundle verify`` to detect tampering between bundle export
      and replay.
    - Catalogue indices (Rhea pgvector) to detect that a tool's contract
      has materially changed and re-embed.
    """
    canonical = json.dumps(
        {
            "descriptor_id": descriptor_id,
            "inputs": sorted(inputs, key=lambda d: d.get("name", "")),
            "outputs": sorted(outputs, key=lambda d: d.get("name", "")),
            "side_effects": side_effects,
            "determinism": determinism,
            "resource_class": resource_class,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class UnifiedToolDescriptor(ConfigBase):
    """G15 — typed Unified Tool Descriptor.

    Replaces the free-form ``ToolConfig.tool_card: Dict`` with a
    schema-validated primitive. Tool authors who opt into UTD get
    catalogue discovery, capability gating, and cost estimation
    uniformly across backends.

    Cross-references:
    - ``apecx-mcp-integration/docs/tool_descriptor_contract.md §2`` —
      semantic spec.
    - ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G15`` —
      gap proposal.
    """
    model_config = ConfigDict(extra="forbid")

    descriptor_id: str = Field(
        ...,
        description="'<backend>:<tool_id>@<version>' grammar. Unique key "
                    "in the catalogue.",
    )
    descriptor_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 over the canonical contract fields. "
                    "Auto-computed at load when empty or set to "
                    "'<computed-at-load>'.",
    )

    # Human-facing.
    display_name: str
    summary: str
    long_description: str = ""

    # Contract.
    inputs: List[UTDInputSpec] = Field(default_factory=list)
    outputs: List[UTDOutputSpec] = Field(default_factory=list)
    side_effects: SideEffectClass = "none"
    determinism: DeterminismClass = "R3"  # safest default
    resource_class: ResourceClass = "cpu_light"

    # Operational.
    cost_estimate: Optional[UTDCostEstimate] = None
    failure_modes: List[UTDFailureMode] = Field(default_factory=list)
    provenance_pin: UTDProvenancePin

    # Authorization.
    requires_capability: List[str] = Field(
        default_factory=list,
        description="Capability tokens (HITL gates §7) the user must "
                    "hold to invoke this tool. Empty list = no special "
                    "capability required.",
    )

    # Lifecycle.
    version_history: List[UTDVersionEntry] = Field(default_factory=list)

    @field_validator("descriptor_id")
    @classmethod
    def _validate_descriptor_id(cls, v: str) -> str:
        if not _DESCRIPTOR_ID_RE.match(v):
            raise ValueError(
                f"FAIL-FAST: descriptor_id={v!r} must match "
                f"'<backend>:<tool_id>@<version>' "
                f"(e.g. 'rhea:muscle.align@5.1.0')"
            )
        return v

    @model_validator(mode="after")
    def _resolve_descriptor_hash(self) -> "UnifiedToolDescriptor":
        sentinels = (None, "", "<computed-at-load>")
        if self.descriptor_hash in sentinels:
            self.descriptor_hash = compute_descriptor_hash(
                descriptor_id=self.descriptor_id,
                inputs=[i.model_dump() for i in self.inputs],
                outputs=[o.model_dump() for o in self.outputs],
                side_effects=self.side_effects,
                determinism=self.determinism,
                resource_class=self.resource_class,
            )
        return self

    @property
    def descriptor_backend(self) -> str:
        """Extract the backend segment ('rhea', 'galaxy', 'native', ...)."""
        m = _DESCRIPTOR_ID_RE.match(self.descriptor_id)
        return m.group("backend") if m else ""

    @property
    def descriptor_tool_id(self) -> str:
        m = _DESCRIPTOR_ID_RE.match(self.descriptor_id)
        return m.group("tool_id") if m else ""

    @property
    def descriptor_version(self) -> str:
        m = _DESCRIPTOR_ID_RE.match(self.descriptor_id)
        return m.group("version") if m else ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedToolDescriptor":
        """Build a UnifiedToolDescriptor from a plain dict, handling
        every nested ConfigBase model's direct-instantiation admittance.

        Without this helper, callers have to flip ``_allow_direct_instantiation``
        on the UTD AND on every nested model (UTDInputSpec, UTDOutputSpec,
        UTDProvenancePin, UTDCostEstimate, UTDFailureMode,
        UTDVersionEntry) — which is brittle and easy to get wrong.

        Use this for in-memory UTD construction (tests + apecx-mcp-side
        programmatic UTD authoring). For file-loaded UTDs, use
        ``UnifiedToolDescriptor.from_config(path)`` (the standard route).
        """
        nested_classes = (
            UTDInputSpec, UTDOutputSpec, UTDCostEstimate,
            UTDFailureMode, UTDProvenancePin, UTDVersionEntry,
            cls,
        )
        # Open every nested class for direct construction.
        for nested in nested_classes:
            nested._allow_direct_instantiation = True
        try:
            normalized = dict(data)

            # Pre-build the simple nested fields:
            if isinstance(normalized.get("provenance_pin"), dict):
                normalized["provenance_pin"] = UTDProvenancePin(
                    **normalized["provenance_pin"])
            if isinstance(normalized.get("cost_estimate"), dict):
                normalized["cost_estimate"] = UTDCostEstimate(
                    **normalized["cost_estimate"])

            # Pre-build list nested fields:
            for list_field, nested_cls in [
                ("inputs", UTDInputSpec),
                ("outputs", UTDOutputSpec),
                ("failure_modes", UTDFailureMode),
                ("version_history", UTDVersionEntry),
            ]:
                if list_field in normalized and isinstance(normalized[list_field], list):
                    built = []
                    for item in normalized[list_field]:
                        if isinstance(item, nested_cls):
                            built.append(item)
                        elif isinstance(item, dict):
                            built.append(nested_cls(**item))
                        else:
                            raise ValueError(
                                f"FAIL-FAST: UTD {list_field}[*] must be a "
                                f"dict or {nested_cls.__name__}, got "
                                f"{type(item).__name__}"
                            )
                    normalized[list_field] = built

            return cls(**normalized)
        finally:
            for nested in nested_classes:
                nested._allow_direct_instantiation = False
