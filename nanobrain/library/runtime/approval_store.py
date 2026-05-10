"""ApprovalStore — pluggable persistence for deferred-HITL gates (G27).

Pre-G27 the framework had no first-class primitive for "the workflow
needs to wait for a human decision, possibly minutes or hours". The
``ApprovalPolicy`` in apecx-mcp-integration sketched the policy side
but not the framework-side suspend/resume contract.

ApprovalStore is the persistence half of G27. It carries one
``Approval`` record per pending request and exposes an idempotent
write surface that the ``DeferredHITLStep`` (sibling commit) drives.

Two storage backends ship in this module:

  * ``InMemoryApprovalStore`` — process-local; for tests and
    single-process deployments. State is lost on restart.
  * ``FileApprovalStore`` — content-addressed JSON files under a
    configurable directory. Survives restart; integration-test
    canonical. Pluggable per record so the operator does not have
    to fork the store class for a custom backend.

The Postgres-backed store (the production-canonical one for the
integration's control plane) is intentionally NOT in this module —
the integration layer carries DB schema concerns and the framework
must stay backend-neutral. A future ``PostgresApprovalStore`` can
implement the same protocol without forking the Step.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G27;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8 (P6+a).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol

logger = logging.getLogger(__name__)


ApprovalDecision = Literal["pending", "approved", "rejected", "corrected"]


@dataclass(frozen=True)
class Approval:
    """One approval request + its (eventual) resolution.

    Fields:
        approval_id: Stable identifier (uuid4 by default; deterministic
            hash if the caller wants idempotent re-emission across
            workflow retries — see DeferredHITLStep's
            ``approval_id_strategy``).
        run_id: The workflow run that emitted this approval (or
            ``None`` if outside a run context — uncommon).
        step_name: The step that emitted the approval, for audit.
        prompt: The request body shown to the operator. Free-form
            string; the step's prompt template owns formatting.
        decision: One of ``pending`` / ``approved`` / ``rejected`` /
            ``corrected``. Defaults to ``pending`` at creation.
        decision_payload: Operator-supplied data on resolution. For
            ``approved`` this is typically empty; for ``corrected``
            it carries the corrected output that should replace the
            step's default.
        decided_by: Operator identity at resolution time.
        created_at: ISO-8601 UTC timestamp at emission.
        decided_at: ISO-8601 UTC timestamp at resolution (None when
            still pending).
    """

    approval_id: str
    run_id: Optional[str]
    step_name: str
    prompt: str
    decision: ApprovalDecision = "pending"
    decision_payload: Optional[Dict[str, Any]] = None
    decided_by: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    decided_at: Optional[str] = None

    def is_pending(self) -> bool:
        return self.decision == "pending"

    def is_resolved(self) -> bool:
        return self.decision in ("approved", "rejected", "corrected")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "step_name": self.step_name,
            "prompt": self.prompt,
            "decision": self.decision,
            "decision_payload": self.decision_payload,
            "decided_by": self.decided_by,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Approval":
        # Coerce missing fields to defaults; pass-through on extras
        # would be silent-failure-shaped, so we explicitly require the
        # known field set and reject extras.
        known = {
            "approval_id", "run_id", "step_name", "prompt",
            "decision", "decision_payload", "decided_by",
            "created_at", "decided_at",
        }
        extras = set(data.keys()) - known
        if extras:
            raise ValueError(
                f"FAIL-FAST: Approval.from_dict received unexpected "
                f"keys {sorted(extras)}; known fields: {sorted(known)}"
            )
        return cls(
            approval_id=data["approval_id"],
            run_id=data.get("run_id"),
            step_name=data["step_name"],
            prompt=data["prompt"],
            decision=data.get("decision", "pending"),
            decision_payload=data.get("decision_payload"),
            decided_by=data.get("decided_by"),
            created_at=data.get(
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            ),
            decided_at=data.get("decided_at"),
        )


class ApprovalStoreProtocol(Protocol):
    """Storage protocol for Approval records. Both shipped backends
    (InMemory and File) implement this; custom backends should too."""

    def submit(self, approval: Approval) -> Approval:
        """Idempotent write. If an Approval with the same approval_id
        already exists, return the EXISTING record (do NOT overwrite).
        This is the contract that lets the Step retry safely."""
        ...

    def get(self, approval_id: str) -> Optional[Approval]:
        """Return the Approval or None if not found."""
        ...

    def resolve(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: Optional[str] = None,
        decision_payload: Optional[Dict[str, Any]] = None,
    ) -> Approval:
        """Resolve a pending Approval. FAIL-FAST if the approval is
        already resolved (changing the decision after-the-fact would
        break audit-trail integrity)."""
        ...

    def list_pending(self) -> List[Approval]:
        """Return all pending approvals; ordering is implementation-
        specific (most-recent-first preferred but not required)."""
        ...


class InMemoryApprovalStore:
    """Process-local approval store. State is held in a dict + lock;
    state is lost on process restart.

    Use cases:
      * unit tests
      * single-process deployments where durability is not a goal
      * driving local CLI flows that only need to survive within one
        Python invocation
    """

    def __init__(self) -> None:
        self._records: Dict[str, Approval] = {}
        self._lock = threading.Lock()

    def submit(self, approval: Approval) -> Approval:
        with self._lock:
            existing = self._records.get(approval.approval_id)
            if existing is not None:
                return existing
            self._records[approval.approval_id] = approval
            return approval

    def get(self, approval_id: str) -> Optional[Approval]:
        with self._lock:
            return self._records.get(approval_id)

    def resolve(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: Optional[str] = None,
        decision_payload: Optional[Dict[str, Any]] = None,
    ) -> Approval:
        with self._lock:
            existing = self._records.get(approval_id)
            if existing is None:
                raise KeyError(
                    f"FAIL-FAST: ApprovalStore.resolve: no approval "
                    f"with id {approval_id!r}"
                )
            if existing.is_resolved():
                raise ValueError(
                    f"FAIL-FAST: ApprovalStore.resolve: approval "
                    f"{approval_id!r} is already resolved (decision="
                    f"{existing.decision!r}); changing it would break "
                    f"audit-trail integrity"
                )
            resolved = Approval(
                approval_id=existing.approval_id,
                run_id=existing.run_id,
                step_name=existing.step_name,
                prompt=existing.prompt,
                decision=decision,
                decision_payload=decision_payload,
                decided_by=decided_by,
                created_at=existing.created_at,
                decided_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[approval_id] = resolved
            return resolved

    def list_pending(self) -> List[Approval]:
        with self._lock:
            return [
                a for a in self._records.values() if a.is_pending()
            ]


class FileApprovalStore:
    """File-backed approval store. One JSON file per approval_id under
    a configurable directory. Atomic writes via write-tmp-then-rename
    so a crash mid-write cannot corrupt an existing record.

    Locking strategy: a directory-level lock file
    (``.approval_store.lock``) serializes mutations within a process.
    Cross-process serialization on the same store is OS-dependent; the
    integration test suite drives this single-process. Multi-writer
    workloads should use a backend with proper transactions (Postgres).
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- Helpers ---------------------------------------------------------

    def _path_for(self, approval_id: str) -> Path:
        # approval_id is constrained to UUID-like or hex hash; reject
        # anything containing a path separator to prevent traversal.
        if "/" in approval_id or "\\" in approval_id or ".." in approval_id:
            raise ValueError(
                f"FAIL-FAST: approval_id {approval_id!r} contains "
                f"path-separator characters; refusing to write."
            )
        return self._root / f"{approval_id}.json"

    def _atomic_write(self, target: Path, data: Dict[str, Any]) -> None:
        tmp = target.with_suffix(target.suffix + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, target)

    def _load_record(self, path: Path) -> Optional[Approval]:
        if not path.is_file():
            return None
        try:
            return Approval.from_dict(json.loads(path.read_text()))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "FileApprovalStore: dropping unreadable record at "
                "%s: %s",
                path,
                exc,
            )
            return None

    # ---- Public API ------------------------------------------------------

    def submit(self, approval: Approval) -> Approval:
        with self._lock:
            target = self._path_for(approval.approval_id)
            existing = self._load_record(target)
            if existing is not None:
                return existing
            self._atomic_write(target, approval.to_dict())
            return approval

    def get(self, approval_id: str) -> Optional[Approval]:
        with self._lock:
            return self._load_record(self._path_for(approval_id))

    def resolve(
        self,
        approval_id: str,
        decision: ApprovalDecision,
        *,
        decided_by: Optional[str] = None,
        decision_payload: Optional[Dict[str, Any]] = None,
    ) -> Approval:
        with self._lock:
            target = self._path_for(approval_id)
            existing = self._load_record(target)
            if existing is None:
                raise KeyError(
                    f"FAIL-FAST: ApprovalStore.resolve: no approval "
                    f"with id {approval_id!r}"
                )
            if existing.is_resolved():
                raise ValueError(
                    f"FAIL-FAST: ApprovalStore.resolve: approval "
                    f"{approval_id!r} already resolved "
                    f"(decision={existing.decision!r})"
                )
            resolved = Approval(
                approval_id=existing.approval_id,
                run_id=existing.run_id,
                step_name=existing.step_name,
                prompt=existing.prompt,
                decision=decision,
                decision_payload=decision_payload,
                decided_by=decided_by,
                created_at=existing.created_at,
                decided_at=datetime.now(timezone.utc).isoformat(),
            )
            self._atomic_write(target, resolved.to_dict())
            return resolved

    def list_pending(self) -> List[Approval]:
        with self._lock:
            results: List[Approval] = []
            for path in self._root.glob("*.json"):
                rec = self._load_record(path)
                if rec is not None and rec.is_pending():
                    results.append(rec)
            # Most-recent first (created_at descending) — best-effort.
            results.sort(key=lambda a: a.created_at, reverse=True)
            return results


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def deterministic_approval_id(
    *,
    run_id: Optional[str],
    step_name: str,
    prompt: str,
) -> str:
    """Build a deterministic approval_id from (run_id, step_name, prompt).

    Use case: a step that retries (e.g., after a transient failure)
    wants the same approval_id so the second submit() finds the
    first record and does NOT create a duplicate. The hash is
    SHA-256 truncated to 32 hex chars (UUID-shaped).
    """
    payload = json.dumps(
        {"run_id": run_id, "step_name": step_name, "prompt": prompt},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def random_approval_id() -> str:
    """Build a random uuid4-shaped approval_id. Use when retries
    should produce distinct approvals (rare; the deterministic path
    is typically what you want)."""
    return uuid.uuid4().hex[:32]


__all__ = [
    "Approval",
    "ApprovalDecision",
    "ApprovalStoreProtocol",
    "InMemoryApprovalStore",
    "FileApprovalStore",
    "deterministic_approval_id",
    "random_approval_id",
]
