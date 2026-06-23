"""Gradual-typed data-unit I/O contracts (Project A, Step 1).

A *contract* is an optional, structured declaration of what a data unit carries, so a workflow
that REUSES a component can have producer→consumer interface drift CAUGHT instead of silently
consumed (the dominant `auto_transfer=False` / G127 silent-failure class). Gradual: an undeclared
side is `any` and always compatible — existing untyped workflows are unaffected.

This module is PURE (no I/O, no framework deps): the typed `Contract`, `parse_contract` (FAIL
LOUD on an unknown kind), and the `compatible(producer, consumer)` relation. Step 1 only WARNs on
incompatibility at load; the runtime `set()` guard + the config_version:3 FAIL-flip are Step 2.

Kind lattice + refinement:
* ``text``        — opaque string; no refinement.
* ``file``        — a file path/handle; refinement = accepted extensions (``extensions``).
* ``record``      — a dict; refinement = ``required`` (key → nested contract, or null = any) and/or
                    ``required_keys`` (list, kinds = any). Structural width + covariant depth.
* ``collection``  — a homogeneous sequence; refinement = ``element`` (nested contract).
* ``handle``      — an opaque reference; refinement = ``referent`` (nested contract).

`record` compatibility (producer P → consumer C, both record):
* width  — ``C.required ⊆ P.guaranteed`` (producer is the structural subtype: adding a key stays
           compatible, removing a required one breaks).
* depth  — for each key C requires, the value kinds recurse-compatible; `any` when either side
           leaves that key's kind undeclared.
* a consumer's ``additionalProperties: false`` is IGNORED here — link-time compat is open-world.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

KINDS = ("text", "file", "record", "collection", "handle")

# Project A Step 2 — the active workflow's config_version, set by Workflow.run()/process()
# (mirrors the G115 `_active_workflow_id` ContextVar). The runtime set() guard reads it to
# decide RAISE (>=3, binding) vs WARN (<3, non-binding). Lives here, in the pure leaf module,
# so both workflow.py (sets it) and data_unit.py (reads it) import without a cycle.
# NOTE: binding enforcement applies only INSIDE Workflow.run()/process(); a direct set() on a
# contract-bearing DU outside a run sees the default (1) and downgrades to WARN.
_active_config_version: ContextVar[int] = ContextVar("_active_config_version", default=1)


class ContractViolationError(Exception):
    """A data unit's ACTUAL value violated its declared contract at runtime. Raised by the
    set() guard only under config_version>=3 (binding); under <3 the guard warns instead."""


@dataclass(frozen=True)
class Contract:
    """A parsed data-unit contract. Refinement fields are kind-specific + all optional."""

    kind: str
    extensions: tuple[str, ...] = ()  # file: accepted/produced extensions (lowercased, no dot)
    required: dict[str, "Contract | None"] = field(default_factory=dict)  # record: key -> contract|any
    element: "Contract | None" = None  # collection: element contract
    referent: "Contract | None" = None  # handle: pointed-to contract


def parse_contract(spec: Any) -> Contract:
    """Parse a contract dict into a Contract. FAIL LOUD on an unknown/missing kind.

    ``None`` is NOT accepted here — callers check for a declared contract first; an explicitly
    declared-but-malformed contract is an authoring error worth raising on.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"contract must be a mapping with a 'kind', got {type(spec).__name__}")
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"contract kind {kind!r} is not one of {KINDS}")

    if kind == "file":
        exts = spec.get("extensions") or []
        if not isinstance(exts, (list, tuple)):
            raise ValueError("file contract 'extensions' must be a list")
        return Contract(kind="file", extensions=tuple(_norm_ext(e) for e in exts))

    if kind == "record":
        required: dict[str, Contract | None] = {}
        # `required`: key -> nested contract dict (or null = any). Typed form.
        for k, v in (spec.get("required") or {}).items():
            required[str(k)] = parse_contract(v) if v is not None else None
        # `required_keys`: list of keys with `any` value-kind. Gradual/width-only form.
        for k in spec.get("required_keys") or []:
            required.setdefault(str(k), None)
        return Contract(kind="record", required=required)

    if kind == "collection":
        el = spec.get("element")
        return Contract(kind="collection", element=parse_contract(el) if el is not None else None)

    if kind == "handle":
        ref = spec.get("referent")
        return Contract(kind="handle", referent=parse_contract(ref) if ref is not None else None)

    return Contract(kind="text")


def _norm_ext(e: Any) -> str:
    return str(e).lower().lstrip(".")


def compatible(producer: Contract, consumer: Contract) -> tuple[bool, str]:
    """True iff a value satisfying ``producer`` is acceptable to ``consumer``. Returns
    (ok, reason); reason is '' when ok. Covariant/structural per the module docstring."""
    if producer.kind != consumer.kind:
        return False, f"kind mismatch: producer {producer.kind!r} != consumer {consumer.kind!r}"

    if producer.kind == "file":
        # Producer's emitted extensions must all be acceptable to the consumer. Either side
        # undeclared (empty) -> no constraint (gradual).
        if producer.extensions and consumer.extensions:
            extra = set(producer.extensions) - set(consumer.extensions)
            if extra:
                return False, f"producer may emit extensions {sorted(extra)} not accepted by consumer"
        return True, ""

    if producer.kind == "record":
        # width: every key the consumer requires must be guaranteed by the producer.
        missing = set(consumer.required) - set(producer.required)
        if missing:
            return False, f"consumer requires keys not guaranteed by producer: {sorted(missing)}"
        # depth: recurse on the consumer-required keys when BOTH declare a value kind.
        for key, c_sub in consumer.required.items():
            p_sub = producer.required.get(key)
            if c_sub is None or p_sub is None:
                continue  # `any` on either side -> compatible (gradual)
            ok, why = compatible(p_sub, c_sub)
            if not ok:
                return False, f"key {key!r}: {why}"
        return True, ""

    if producer.kind == "collection":
        if producer.element is None or consumer.element is None:
            return True, ""  # element kind undeclared on a side -> any
        ok, why = compatible(producer.element, consumer.element)
        return (ok, "" if ok else f"element: {why}")

    if producer.kind == "handle":
        if producer.referent is None or consumer.referent is None:
            return True, ""
        ok, why = compatible(producer.referent, consumer.referent)
        return (ok, "" if ok else f"referent: {why}")

    # text
    return True, ""


def validate_value(contract: Contract, value: Any) -> tuple[bool, str]:
    """Does an ACTUAL value satisfy ``contract``? The RUNTIME complement to ``compatible``
    (which is declaration-vs-declaration). Returns (ok, reason); reason '' when ok. Pure.

    Open-world like ``compatible``: a record may carry EXTRA keys; an undeclared value-kind on
    a required key is `any` (not checked). Caller decides None/sentinel handling before calling.

    By-design edges: `collection` is list/tuple ONLY (a set/generator/str is rejected — a set is
    unordered and a generator is single-pass, neither a safe data-unit payload); `file` extension
    matches the LAST dotted segment only (``a.tar.gz`` → ``gz``); dict/str subclasses + namedtuple
    + pathlib.Path are accepted (isinstance).
    """
    k = contract.kind
    if k == "text":
        return (True, "") if isinstance(value, str) else (
            False, f"expected text (str), got {type(value).__name__}")

    if k == "file":
        if not isinstance(value, (str, os.PathLike)):
            return False, f"expected file path (str/PathLike), got {type(value).__name__}"
        if contract.extensions:
            ext = str(value).lower().rsplit(".", 1)
            ext = ext[-1] if len(ext) == 2 else ""
            if ext not in contract.extensions:
                return False, f"file extension {ext!r} not in {list(contract.extensions)}"
        return True, ""

    if k == "record":
        if not isinstance(value, dict):
            return False, f"expected record (dict), got {type(value).__name__}"
        for key, sub in contract.required.items():
            if key not in value:
                return False, f"missing required key {key!r}"
            if sub is not None:  # declared value-kind -> recurse; None = any -> skip
                ok, why = validate_value(sub, value[key])
                if not ok:
                    return False, f"key {key!r}: {why}"
        return True, ""

    if k == "collection":
        if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
            return False, f"expected collection (list/tuple), got {type(value).__name__}"
        if contract.element is not None:
            for i, item in enumerate(value):
                ok, why = validate_value(contract.element, item)
                if not ok:
                    return False, f"element[{i}]: {why}"
        return True, ""

    if k == "handle":
        return (True, "") if value is not None else (False, "expected a handle (non-None)")

    return True, ""


__all__ = [
    "Contract",
    "ContractViolationError",
    "KINDS",
    "compatible",
    "parse_contract",
    "validate_value",
]
