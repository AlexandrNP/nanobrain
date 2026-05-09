"""Skeleton (G9) — first-class workflow-skeleton primitive.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G9`` and
``apecx-mcp-integration/docs/agent_workflow_authoring.md §4``: a skeleton
is a pre-validated nanobrain workflow YAML with typed *holes* — named
placeholders that the orchestrator fills at authoring time. Holes are
declared in a sibling ``skeleton.schema.json`` (or inline) with type,
required-flag, default, and description.

This module ships the framework primitives:

- :class:`SkeletonHole` — typed parameter slot
- :class:`Skeleton` — the workflow-shape carrier (YAML body + hole schema)
- :class:`SkeletonRegistry` — content-addressed lookup by digest or
  semver tag

What this module does NOT do (deferred to G17):

- ``PlanLoweringStep`` — applies the 7 lowering steps from
  ``agent_workflow_authoring.md §5``. Lives in ``plan_lowering_step.py``.
- ``SkeletonLoaderStep`` — wraps registry lookup as a workflow step.

Workspace constraints honored:
- Holes use the same fixed type vocabulary as PromptHole (G14) for
  consistency: string / integer / number / boolean / array / object /
  any. The skeleton hole grammar is a strict superset (adds the
  ``tool_descriptor_ref`` type per the gap proposal).
- Content hash auto-computed from the canonical skeleton YAML body
  (whitespace-normalized, sorted-keys JSON projection of the YAML).
- ``extra='forbid'`` on every ConfigBase subclass.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import ConfigDict, Field, field_validator, model_validator

from nanobrain.core.config.config_base import ConfigBase


# Hole types — superset of G14 PromptHole types: adds tool_descriptor_ref
# per agent_workflow_authoring.md §4.1 (the type triggers special handling
# in the lowering pipeline that resolves the ref against the UTD catalogue).
SkeletonHoleType = Literal[
    "string", "integer", "number", "boolean",
    "array", "object", "any", "tool_descriptor_ref",
]


# Skeleton-version grammar:
#   * 12-char hex prefix of the SHA-256 digest (preferred for production)
#   * full 64-char hex digest
#   * semver-ish tag (loose; admits "1.2.3-rc.1+build.456")
_SKELETON_VERSION_RE = re.compile(
    r"^([a-f0-9]{12}|[a-f0-9]{64}|[0-9A-Za-z\-_.+]+)$"
)


# Inline-hole token grammar in skeleton.yml: "{{<name>: <type>}}" or
# "{{<name>: <type> | default=<value>}}". Tolerant of surrounding whitespace.
_INLINE_HOLE_RE = re.compile(
    r"^\{\{\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*"
    r"(?P<type>string|integer|number|boolean|array|object|any|tool_descriptor_ref)"
    r"(?:\s*\|\s*default\s*=\s*(?P<default>[^}]+?))?\s*\}\}$"
)


class SkeletonHole(ConfigBase):
    """A typed hole declared in skeleton.schema.json (or inline alongside
    the skeleton body).

    Per ``agent_workflow_authoring.md §4.1``: each hole has a type, a
    required flag, an optional default for optional holes, and a human
    description. The lowering pipeline's Gate-3 validates parameter
    bindings against this shape.
    """
    model_config = ConfigDict(extra="forbid")

    type: SkeletonHoleType = "string"
    required: bool = True
    default: Any = None
    description: str = ""


class Skeleton(ConfigBase):
    """G9 — first-class skeleton primitive.

    Pairs a workflow YAML body (string, NOT yet parsed) with a typed
    hole schema. The body contains ``{{<name>: <type>}}`` tokens that
    the lowering pipeline (G17) replaces with bound values.

    Identity:
    - ``skeleton_id`` is the human-readable handle (e.g.
      ``"multi_source_discovery"``).
    - ``skeleton_version`` is the operator-facing version (semver tag
      OR hex digest of body).
    - ``content_hash`` is the canonical SHA-256 of the body, auto-computed
      at load time. Two Skeleton instances with the same body hash to
      the same value regardless of holes/description differences — the
      body IS the contract.

    Three construction paths:
    1. ``Skeleton.from_config('path/to/skeleton_config.yml')`` — standard
       file-based path matching every other ConfigBase subclass.
    2. ``Skeleton.from_config({...})`` — inline-dict admittance for
       in-memory construction (tests, programmatic skeleton authoring).
       Skeleton joins DataUnit/Link/Trigger as a config primitive that
       bypasses the file-only rule.
    3. ``Skeleton.load_from_directory(skeleton_dir)`` — convenience for
       the canonical layout: ``skeleton.yml`` + ``skeleton.schema.json``
       per ``agent_workflow_authoring.md §4.1``.
    """
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_config(cls, config, **kwargs) -> "Skeleton":
        """Override the framework default to admit inline dicts (tests +
        programmatic construction). File-path input still routes through
        ``ConfigBase.from_config``.

        Nested ``SkeletonHole`` instances are pre-built here so each hole
        gets the same direct-instantiation admittance as the parent —
        Pydantic otherwise tries to construct SkeletonHole from the dict
        and trips its own FromConfigBase prohibition.
        """
        if isinstance(config, dict):
            # Pre-build SkeletonHole nested instances.
            normalized = dict(config)
            holes_raw = normalized.get("holes", {})
            if isinstance(holes_raw, dict):
                built_holes: Dict[str, SkeletonHole] = {}
                SkeletonHole._allow_direct_instantiation = True
                try:
                    for name, hole_data in holes_raw.items():
                        if isinstance(hole_data, SkeletonHole):
                            built_holes[name] = hole_data
                        elif isinstance(hole_data, dict):
                            built_holes[name] = SkeletonHole(**hole_data)
                        else:
                            raise ValueError(
                                f"FAIL-FAST: skeleton hole {name!r} value "
                                f"must be a dict or SkeletonHole, got "
                                f"{type(hole_data).__name__}"
                            )
                finally:
                    SkeletonHole._allow_direct_instantiation = False
                normalized["holes"] = built_holes

            cls._allow_direct_instantiation = True
            try:
                return cls(**normalized)
            finally:
                cls._allow_direct_instantiation = False
        if isinstance(config, cls):
            return config
        # Delegate file-path / Path input to ConfigBase.from_config:
        return super().from_config(config, **kwargs)

    skeleton_id: str
    skeleton_version: str
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 over the canonical YAML body. Auto-computed "
                    "at load when empty or set to '<computed-at-load>'.",
    )
    description: str = ""
    body: str = Field(
        ...,
        description="The skeleton's workflow YAML body (the raw text). "
                    "Holes appear inline as '{{<name>: <type>}}' tokens.",
    )
    holes: Dict[str, SkeletonHole] = Field(default_factory=dict)

    @field_validator("skeleton_version")
    @classmethod
    def _validate_skeleton_version(cls, v: str) -> str:
        if not v or not v.strip() or any(c.isspace() for c in v):
            raise ValueError(
                "FAIL-FAST: Skeleton.skeleton_version must be a non-empty "
                "whitespace-free string (12/64-char hex digest OR semver tag)"
            )
        if not _SKELETON_VERSION_RE.match(v):
            raise ValueError(
                f"FAIL-FAST: Skeleton.skeleton_version={v!r} must match "
                f"a 12-char or 64-char hex digest, OR a semver-ish tag"
            )
        return v

    @model_validator(mode="after")
    def _resolve_content_hash(self) -> "Skeleton":
        sentinels = (None, "", "<computed-at-load>")
        if self.content_hash in sentinels:
            self.content_hash = compute_skeleton_body_hash(self.body)
        return self

    def declared_hole_names(self) -> List[str]:
        """Names of holes declared in the schema. Used by Gate-3 of the
        lowering pipeline to verify parameter bindings cover every
        required hole and don't introduce extras."""
        return list(self.holes.keys())

    def find_inline_hole_tokens(self) -> List[Tuple[str, str, Optional[str]]]:
        """Walk the body string and return one tuple per inline
        ``{{<name>: <type> | default=<value>}}`` token found.

        Returns:
            List of (hole_name, hole_type, default_or_None) tuples. The
            order matches the token order in the body — useful for
            substitution.

        The lowering pipeline's hole substitution step (G17 Step 3)
        consumes this list to know which tokens to replace.
        """
        result: List[Tuple[str, str, Optional[str]]] = []
        # Match every {{...}} block, then validate each one against the
        # full grammar. This avoids regex catastrophic backtracking on
        # malformed input.
        for raw_match in re.finditer(r"\{\{[^{}]*\}\}", self.body):
            token = raw_match.group(0)
            m = _INLINE_HOLE_RE.match(token)
            if m is None:
                # Mention the token but don't FAIL-FAST here — schema
                # validation has its own pass that surfaces this with
                # better context (line/column). The lowering pipeline
                # is the right place to fail.
                continue
            result.append((m.group("name"), m.group("type"), m.group("default")))
        return result

    def validate_against_schema(self) -> List[str]:
        """Per agent_workflow_authoring.md §4.1: cross-check that every
        inline ``{{...}}`` token in the body has a corresponding entry
        in the holes schema, and vice versa.

        Returns:
            A list of human-readable validation issues. Empty list means
            the skeleton is internally consistent.

        The lowering pipeline's Gate-3 calls this at validation time;
        Skeleton itself doesn't FAIL-FAST in __init__ because we want
        the issue list reported BEFORE the framework raises.
        """
        issues: List[str] = []

        inline_tokens = self.find_inline_hole_tokens()
        inline_names = {n for n, _, _ in inline_tokens}
        declared_names = set(self.holes.keys())

        # Tokens used in the body but not declared in the schema:
        used_undeclared = inline_names - declared_names
        for name in sorted(used_undeclared):
            issues.append(
                f"hole {name!r} used in skeleton body but not declared "
                f"in holes schema"
            )

        # Holes declared in the schema but not used in the body:
        declared_unused = declared_names - inline_names
        for name in sorted(declared_unused):
            issues.append(
                f"hole {name!r} declared in holes schema but not used "
                f"in skeleton body"
            )

        # Per-token: type in inline must match type in schema (if
        # both are present).
        for name, inline_type, _ in inline_tokens:
            if name in self.holes:
                schema_type = self.holes[name].type
                if inline_type != schema_type:
                    issues.append(
                        f"hole {name!r} type mismatch: body says "
                        f"{inline_type!r}, schema says {schema_type!r}"
                    )

        # Required holes without defaults must NOT have an inline default
        # (the schema's `required: true` is the source of truth).
        for name, _, inline_default in inline_tokens:
            if name in self.holes and self.holes[name].required and inline_default is not None:
                issues.append(
                    f"hole {name!r} declared required in schema but has "
                    f"inline default in body"
                )

        return issues

    @classmethod
    def load_from_directory(
        cls,
        skeleton_dir: str | Path,
        skeleton_id: Optional[str] = None,
        skeleton_version: Optional[str] = None,
    ) -> "Skeleton":
        """Convenience: load a skeleton from
        ``<skeleton_dir>/skeleton.yml`` + ``<skeleton_dir>/skeleton.schema.json``.

        The directory layout matches ``agent_workflow_authoring.md §4.1``:

        .. code-block:: text

            composition/workflows/<skeleton_id>/
                skeleton.yml          # workflow body with {{...}} tokens
                skeleton.schema.json  # {"holes": {...}, "metadata": {...}}

        ``skeleton_id`` defaults to the directory's basename;
        ``skeleton_version`` defaults to the body's content hash.
        """
        dir_path = Path(skeleton_dir)
        body_path = dir_path / "skeleton.yml"
        schema_path = dir_path / "skeleton.schema.json"

        if not body_path.is_file():
            raise FileNotFoundError(
                f"FAIL-FAST: skeleton body not found at {body_path}"
            )
        body_text = body_path.read_text()

        holes_dict: Dict[str, Any] = {}
        description = ""
        if schema_path.is_file():
            schema_data = json.loads(schema_path.read_text())
            holes_dict = schema_data.get("holes", {})
            metadata = schema_data.get("metadata", {})
            description = metadata.get("description", "")

        # If skeleton_version is missing, use the content hash (12-char
        # prefix) so the load is content-addressed without operator burden.
        if skeleton_version is None:
            skeleton_version = compute_skeleton_body_hash(body_text)[:12]

        return cls.from_config({
            "skeleton_id": skeleton_id or dir_path.name,
            "skeleton_version": skeleton_version,
            "description": description,
            "body": body_text,
            "holes": holes_dict,
        })


def compute_skeleton_body_hash(body: str) -> str:
    """G9 — canonical SHA-256 of the skeleton body.

    The hash is over the body's canonical form: the YAML is parsed,
    serialized to JSON with sorted keys + minimal separators, then
    hashed. This makes the hash invariant under cosmetic YAML changes
    (whitespace, key order, comment placement).

    For bodies that don't parse as YAML (rare — usually a typo), we
    fall back to the raw bytes — operators get a hash they can pin
    even when the body is malformed, but the hash is then sensitive
    to whitespace.
    """
    try:
        parsed = yaml.safe_load(body)
    except yaml.YAMLError:
        # Fallback: hash the raw bytes. This is honest about the body
        # being non-canonical; if the operator wants stable hashing
        # they need to fix the YAML.
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    canonical = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SkeletonRegistry:
    """G9 — content-addressed registry for skeleton lookup.

    The registry maps ``skeleton_id`` to a list of (skeleton_version,
    Skeleton) pairs. Lookup by ``skeleton_id + skeleton_version``
    where version is one of:

    1. The full 64-char content hash.
    2. A 12-char hex prefix of the content hash.
    3. A semver-ish tag the registry resolves to a specific hash.

    The registry is in-memory; a full implementation in apecx-mcp
    persists this to the control plane DB. This module ships the
    framework-side primitive; persistence is a follow-up.
    """

    def __init__(self) -> None:
        # skeleton_id → list of (version_string, content_hash, Skeleton)
        # tuples. Order = registration order = chronological.
        self._entries: Dict[str, List[Tuple[str, str, Skeleton]]] = {}

    def register(self, skeleton: Skeleton) -> None:
        """Register a skeleton. Re-registering an existing
        (skeleton_id, content_hash) pair is a no-op (idempotent).
        Re-registering with the SAME version_string but a DIFFERENT
        content_hash FAIL-FASTs (contract drift between two skeleton
        instances with the same human handle)."""
        entries = self._entries.setdefault(skeleton.skeleton_id, [])

        for existing_version, existing_hash, _ in entries:
            if existing_hash == skeleton.content_hash:
                return  # already registered (idempotent)
            if existing_version == skeleton.skeleton_version:
                raise ValueError(
                    f"FAIL-FAST: skeleton {skeleton.skeleton_id!r} "
                    f"version {skeleton.skeleton_version!r} already "
                    f"registered with content_hash {existing_hash!r}; "
                    f"new registration has content_hash "
                    f"{skeleton.content_hash!r} — contract drift"
                )

        entries.append((skeleton.skeleton_version, skeleton.content_hash, skeleton))

    def lookup(
        self,
        skeleton_id: str,
        skeleton_version: str,
    ) -> Skeleton:
        """Resolve (skeleton_id, skeleton_version) to a Skeleton.

        Resolution rules (tried in order):
        1. Exact content_hash match (full 64-char OR 12-char prefix).
        2. Exact version_string match (semver tag).

        Raises ``KeyError`` when no match.
        """
        entries = self._entries.get(skeleton_id)
        if not entries:
            raise KeyError(
                f"FAIL-FAST: skeleton {skeleton_id!r} not in registry"
            )

        # First pass: exact content_hash match (full or prefix).
        for _, content_hash, sk in entries:
            if content_hash == skeleton_version:
                return sk
            if len(skeleton_version) == 12 and content_hash.startswith(skeleton_version):
                return sk

        # Second pass: exact version_string match.
        for version_string, _, sk in entries:
            if version_string == skeleton_version:
                return sk

        # Build a helpful error listing what's available.
        available = [
            f"{vs} ({ch[:12]}...)" for vs, ch, _ in entries
        ]
        raise KeyError(
            f"FAIL-FAST: skeleton {skeleton_id!r} version "
            f"{skeleton_version!r} not in registry; available versions: "
            f"{available}"
        )

    def list_skeletons(self) -> List[str]:
        """Return all registered skeleton_ids."""
        return list(self._entries.keys())

    def versions_of(self, skeleton_id: str) -> List[str]:
        """Return all version_strings registered for a skeleton_id."""
        entries = self._entries.get(skeleton_id, [])
        return [vs for vs, _, _ in entries]
