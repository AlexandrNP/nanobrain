"""UnifiedToolDescriptor (G15) — typed tool-card primitive.

Per ``apecx-mcp-integration/docs/CONTRACTS.md#g15`` and
``apecx-mcp-integration/docs/CONTRACTS.md#td-vocab``: a typed
shape that EVERY tool backend (Rhea, native nanobrain, GalaxyMCP) speaks.
Promotes the ``ToolConfig.tool_card`` field from a free-form dict to a
content-hash-pinned, capability-aware primitive.

This module owns the on-the-wire schema; the semantic spec lives in
``CONTRACTS.md#td-vocab``. New tool authors should subclass
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
# Enum-like literals — match CONTRACTS.md#td-vocab vocabulary.
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

# DeterminismClass — per CONTRACTS.md#hpc-determinism three-tier model.
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
    has_default: bool = Field(
        default=False,
        description="Whether the source schema DECLARED a default for this "
                    "input. Distinguishes a required-no-default param (must be "
                    "supplied) from one whose declared default happens to be "
                    "null — 'default' alone cannot tell them apart.",
    )


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
    - ``apecx-mcp-integration/docs/CONTRACTS.md#td-vocab`` —
      semantic spec.
    - ``apecx-mcp-integration/docs/CONTRACTS.md#g15`` —
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

    @classmethod
    def from_python_callable(
        cls,
        fn: Any,
        *,
        descriptor_id: Optional[str] = None,
        backend: str = "native",
        version: str = "0.1.0",
        provenance_class_path: Optional[str] = None,
        side_effects: str = "none",
        determinism: str = "R3",
        resource_class: str = "cpu_light",
        **overrides: Any,
    ) -> "UnifiedToolDescriptor":
        """Build a UTD by introspecting a Python callable.

        Mirrors the convenience pattern Rhea / FastMCP use for tool
        registration: the author writes a regular Python function with
        type hints + docstring, and the framework derives the
        machine-readable descriptor for free.

        Resolution rules:

        - ``inputs`` derived from ``inspect.signature(fn).parameters``.
          Each non-``self``/``cls`` parameter becomes a ``UTDInputSpec``.
          The ``type`` field uses ``typing.get_type_hints`` (falls back
          to ``Any`` when an annotation is missing). ``required`` is
          ``True`` iff the parameter has no default; the default value
          (when present) is recorded in ``UTDInputSpec.default``.
        - ``outputs`` derived from the return-type annotation. A single
          named output ``"return"`` is generated; the type is the
          return annotation's name (e.g. ``"dict"``, ``"str"``, or a
          dotted class path).
        - ``display_name`` defaults to ``fn.__qualname__``; first line
          of the docstring becomes ``summary``; the rest becomes
          ``long_description``.
        - ``descriptor_id`` defaults to
          ``"<backend>:<module>.<qualname>@<version>"`` (lowercased) so
          the same callable always produces the same descriptor_id.
        - ``provenance_pin.class_path`` defaults to
          ``"<module>.<qualname>"`` so ``ToolBase.from_descriptor``
          can locate the callable. Override via
          ``provenance_class_path`` if the callable is bound to a
          different importable path (test fixtures, dynamically
          generated functions).

        Override any auto-derived field via ``**overrides`` —
        e.g. ``cost_estimate=UTDCostEstimate(estimated_seconds=120.0,
        confidence='high')``.

        Brutal limitations (documented honestly):
        - Only basic type-name extraction. ``Dict[str, int]`` becomes
          ``"Dict"`` not ``"Dict[str, int]"``. For richer schemas,
          override ``inputs=`` / ``outputs=`` explicitly.
        - The descriptor's ``determinism`` defaults to ``R3`` (least
          assertive); the author should override to ``R0`` (deterministic)
          if true. The framework cannot infer determinism from a
          signature.
        """
        import inspect as _inspect

        try:
            sig = _inspect.signature(fn)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"FAIL-FAST: from_python_callable cannot introspect "
                f"{fn!r}: {exc}"
            ) from exc

        # Type hints (best-effort; some annotations are PEP-604 unions
        # or string forwards that get_type_hints can't resolve).
        try:
            hints = _inspect.get_annotations(fn, eval_str=True)
        except Exception:  # noqa: BLE001
            hints = getattr(fn, "__annotations__", {}) or {}

        def _type_name(annotation: Any) -> str:
            if annotation is _inspect.Parameter.empty:
                return "Any"
            # typing.X has __name__; classes have __name__; everything else str()
            name = getattr(annotation, "__name__", None)
            if name:
                return name
            origin = getattr(annotation, "__origin__", None)
            if origin is not None:
                origin_name = getattr(origin, "__name__", None)
                if origin_name:
                    return origin_name
            return str(annotation).replace("typing.", "")

        # Build inputs from non-self/cls parameters.
        inputs: List[Dict[str, Any]] = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (_inspect.Parameter.VAR_POSITIONAL,
                              _inspect.Parameter.VAR_KEYWORD):
                # Skip *args / **kwargs — UTD has no concept of variadic
                # inputs; authors should bind these explicitly.
                continue
            ann = hints.get(pname, param.annotation)
            type_name = _type_name(ann)
            has_default = param.default is not _inspect.Parameter.empty
            inputs.append({
                "name": pname,
                "type": type_name,
                "required": not has_default,
                "default": param.default if has_default else None,
                "description": "",
            })

        # Build outputs from return annotation.
        return_ann = hints.get("return", sig.return_annotation)
        if return_ann is _inspect.Parameter.empty:
            output_type = "Any"
        else:
            output_type = _type_name(return_ann)
        outputs = [{
            "name": "return",
            "type": output_type,
            "description": "",
        }]

        # display_name + summary + long_description from qualname + docstring.
        display_name = fn.__qualname__
        doc = (fn.__doc__ or "").strip()
        if doc:
            doc_lines = doc.split("\n", 1)
            summary = doc_lines[0].strip()
            long_description = doc_lines[1].strip() if len(doc_lines) > 1 else ""
        else:
            summary = display_name
            long_description = ""

        # descriptor_id default — module.qualname for stability.
        if descriptor_id is None:
            module = getattr(fn, "__module__", "unknown") or "unknown"
            tool_id = f"{module}.{fn.__qualname__}".lower()
            # tool_id grammar in _DESCRIPTOR_ID_RE allows [a-z0-9_.] only,
            # AND requires the FIRST character to be [a-z]. Replace
            # disallowed chars with underscore; if the result doesn't
            # start with [a-z], prefix ``fn_`` so module names like
            # ``__main__`` produce a valid descriptor_id.
            tool_id = re.sub(r"[^a-z0-9_.]", "_", tool_id)
            if not tool_id or not tool_id[0].isalpha():
                tool_id = "fn_" + tool_id.lstrip("_.")
            descriptor_id = f"{backend}:{tool_id}@{version}"

        # provenance_pin default — module.qualname.
        if provenance_class_path is None:
            module = getattr(fn, "__module__", "unknown") or "unknown"
            provenance_class_path = f"{module}.{fn.__qualname__}"

        # Compose the dict + delegate to from_dict for the nested-class
        # admittance dance.
        data: Dict[str, Any] = {
            "descriptor_id": descriptor_id,
            "display_name": display_name,
            "summary": summary,
            "long_description": long_description,
            "inputs": inputs,
            "outputs": outputs,
            "side_effects": side_effects,
            "determinism": determinism,
            "resource_class": resource_class,
            "provenance_pin": {"class_path": provenance_class_path},
        }
        # Apply overrides last — author-supplied fields trump defaults.
        data.update(overrides)
        return cls.from_dict(data)
