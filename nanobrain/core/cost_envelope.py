"""CostEnvelope — G26 workflow-level cost-cap enforcement primitive.

eval_03 Round 3 G26: pre-G26 the framework had ``ResourceEnvelope``
(G12 — declarative per-step + workflow-level resource declarations)
but no *enforcement* primitive. ``hitl_safety_gates.md §8`` GATE-R1
describes "halt the task when its cumulative cost exceeds the cap";
that halt did not exist. The per-deployment-per-day ceiling that
``autonomous_workflow_agent.md §8`` calls out cannot stop a single
runaway task.

Post-G26 the framework ships:

  * ``CostEnvelope`` — a declared cap (usd / tokens / walltime_seconds)
  * ``CostTracker`` — per-run cumulative-cost ledger; check-and-raise
    ``CostEnvelopeBreach`` on any record() call that would exceed
    any cap on the active envelope
  * ``current_cost_tracker()`` — contextvar-backed accessor; LLM
    clients, tool calls, and any other cost-emitting code path
    consult this and call ``record(kind, amount)`` to charge cost

P6+b decision (open question §8.8 of development_roadmap):
  * **Per-step cap AND per-workflow cap, both declarative.** The
    workflow's CostEnvelope is the OUTER cap (cumulative across
    every step's record()). Per-step caps live on StepConfig and are
    checked separately at step boundary (sibling concern; this
    primitive ships the workflow-level surface, the step-level
    surface composes via a stack of trackers).
  * Both caps are optional fields. Absent = no cap on that dimension.

## Lifecycle

1. The runner / executor / driver creates a CostTracker bound to a
   workflow-level CostEnvelope and ``activate()``s it as the
   contextvar.
2. Every cost-emitting operation under the active context calls
   ``current_cost_tracker().record(kind, amount)``.
3. record() checks the cumulative against the cap; ``raise
   CostEnvelopeBreach`` if any cap is exceeded.
4. The runner catches CostEnvelopeBreach and treats it as a terminal
   failure (workflow status -> FAILED with reason="cost_envelope_breach").

## Why a contextvar (not a plain attribute)

Concurrent asyncio tasks each get their own contextvar value (PEP 567).
A workflow with parallel branches gets one tracker per branch, OR all
branches share one tracker — the choice is up to the runner that
``activate()``s the tracker. The framework primitive supports both.

## Cost kinds

Three canonical kinds ship:
  * ``"usd"``     — dollar cost (LLM API spend)
  * ``"tokens"``  — token count (LLM input + output)
  * ``"walltime_seconds"`` — wall-clock seconds

Custom kinds are NOT rejected — operators can record arbitrary kinds
for their own tracking, but only the canonical three are enforced
against the envelope (since the envelope only declares those three).

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G26;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8 (P6+b).
"""
from __future__ import annotations

import contextvars
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

# Canonical cost kinds. Custom kinds may be recorded but are not
# enforced against the envelope.
CANONICAL_KINDS = ("usd", "tokens", "walltime_seconds")


@dataclass(frozen=True)
class CostEnvelope:
    """A declared cap. All fields optional; absent = no cap.

    The frozen dataclass makes the envelope identity-bearing — two
    envelopes with the same fields hash equal so the runner can dedupe
    nested-envelope semantics.
    """

    usd: Optional[float] = None
    tokens: Optional[int] = None
    walltime_seconds: Optional[float] = None

    def cap_for(self, kind: str) -> Optional[float]:
        """Return the cap for a given kind, or None if uncapped."""
        if kind == "usd":
            return self.usd
        if kind == "tokens":
            return float(self.tokens) if self.tokens is not None else None
        if kind == "walltime_seconds":
            return self.walltime_seconds
        return None  # custom kinds are unenforced


class CostEnvelopeBreach(Exception):
    """Raised when a record() call would push cumulative cost over
    the envelope cap.

    Attributes:
        kind: which cost dimension was breached
        cap: the declared cap
        attempted: cumulative-after-this-record (if allowed)
        previous: cumulative-before-this-record (the unbreached state)
    """

    def __init__(
        self,
        *,
        kind: str,
        cap: float,
        attempted: float,
        previous: float,
    ) -> None:
        super().__init__(
            f"CostEnvelopeBreach: {kind}={attempted:g} would exceed "
            f"cap={cap:g} (previous cumulative={previous:g}). "
            f"The runner should treat this as a terminal failure "
            f"with reason='cost_envelope_breach'."
        )
        self.kind = kind
        self.cap = cap
        self.attempted = attempted
        self.previous = previous


@dataclass
class CostLedger:
    """Per-run cumulative ledger. NOT thread-safe on its own — callers
    serialize access via the tracker's lock."""

    cumulative: Dict[str, float] = field(default_factory=dict)

    def add(self, kind: str, amount: float) -> float:
        new_total = self.cumulative.get(kind, 0.0) + float(amount)
        self.cumulative[kind] = new_total
        return new_total

    def get(self, kind: str) -> float:
        return self.cumulative.get(kind, 0.0)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.cumulative)


# Module-global contextvar. Concurrent asyncio tasks see their own
# value (PEP 567). The runner is responsible for activating the
# correct tracker for each task; the framework primitive does not
# auto-share trackers across tasks.
_current_cost_tracker: contextvars.ContextVar[
    Optional["CostTracker"]
] = contextvars.ContextVar("current_cost_tracker", default=None)


class CostTracker:
    """Active cost-cap enforcer for one workflow run.

    Construct with an envelope; activate via ``with tracker.activate():``;
    cost-emitting code calls ``current_cost_tracker().record(kind, amount)``.

    Thread-safe: a lock serializes record() calls so concurrent
    branches sharing one tracker cannot race past the cap.
    """

    def __init__(self, envelope: CostEnvelope) -> None:
        self._envelope = envelope
        self._ledger = CostLedger()
        self._lock = threading.Lock()

    @property
    def envelope(self) -> CostEnvelope:
        return self._envelope

    def record(self, kind: str, amount: float) -> float:
        """Record ``amount`` of cost in dimension ``kind``.

        Returns the new cumulative for that kind.

        Raises:
            CostEnvelopeBreach: if the cumulative-after-this-record
                would exceed the envelope cap on ``kind``.

        Atomicity: under the lock, we compute the would-be new total
        and compare to the cap BEFORE writing it back. A breach leaves
        the ledger in its pre-record state (failure-atomic).
        """
        # Adversarial-probe finding (2026-05-11): NaN passes ``< 0``
        # checks (every comparison with NaN returns False). Without
        # an explicit math.isnan guard, NaN values silently propagate
        # into the cumulative ledger AND bypass the cap check
        # downstream (``NaN > cap`` is also False). isinf check is
        # added for symmetry — record(inf) would corrupt the ledger
        # similarly even though the cap check happens to catch it.
        if math.isnan(amount):
            raise ValueError(
                f"FAIL-FAST: CostTracker.record amount must be a "
                f"finite number; got NaN. NaN silently corrupts the "
                f"ledger because every NaN comparison returns False."
            )
        if math.isinf(amount):
            raise ValueError(
                f"FAIL-FAST: CostTracker.record amount must be finite; "
                f"got {amount}. Infinity would lock the ledger into a "
                f"breach state regardless of subsequent records."
            )
        if amount < 0:
            raise ValueError(
                f"FAIL-FAST: CostTracker.record amount must be >= 0; "
                f"got {amount}"
            )
        with self._lock:
            previous = self._ledger.get(kind)
            attempted = previous + float(amount)
            cap = self._envelope.cap_for(kind)
            if cap is not None and attempted > cap:
                raise CostEnvelopeBreach(
                    kind=kind,
                    cap=cap,
                    attempted=attempted,
                    previous=previous,
                )
            return self._ledger.add(kind, amount)

    def cumulative(self, kind: str) -> float:
        """Read the cumulative cost for a kind. Read-only; does not
        check the cap."""
        with self._lock:
            return self._ledger.get(kind)

    def snapshot(self) -> Dict[str, float]:
        """Return a copy of the ledger's cumulative dict. Useful for
        provenance recording."""
        with self._lock:
            return self._ledger.snapshot()

    @contextmanager
    def activate(self) -> Iterator["CostTracker"]:
        """Install this tracker as the current_cost_tracker for the
        duration of the block. Mirrors ProvenanceContext.activate."""
        token = _current_cost_tracker.set(self)
        try:
            yield self
        finally:
            _current_cost_tracker.reset(token)


def current_cost_tracker() -> Optional[CostTracker]:
    """Return the active CostTracker, or None if no envelope is in
    effect. Cost-emitting code paths consult this; if None, recording
    is a fast no-op (no envelope, no enforcement)."""
    return _current_cost_tracker.get()


def record_cost(kind: str, amount: float) -> Optional[float]:
    """Convenience: record against the active tracker if one exists.

    Returns the new cumulative, or None if no tracker is active.
    Raises CostEnvelopeBreach on cap violation; passes through.

    This is the canonical entry point for cost-emitting code (LLM
    clients, tool dispatchers, etc.). It avoids hand-checking the
    contextvar at every call site.
    """
    tracker = current_cost_tracker()
    if tracker is None:
        return None
    return tracker.record(kind, amount)


__all__ = [
    "CANONICAL_KINDS",
    "CostEnvelope",
    "CostEnvelopeBreach",
    "CostLedger",
    "CostTracker",
    "current_cost_tracker",
    "record_cost",
]
