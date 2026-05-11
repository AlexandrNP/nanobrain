"""DeferredHITLStep — G27 deferred-HITL approval Step primitive.

eval_03 Round 3 G27: pre-G27, the framework had ``ApprovalStep`` and
``ApprovalPolicy`` (apecx-mcp side) that emit approval rows
synchronously, but no first-class primitive for "the workflow needs
to wait for a human decision, possibly minutes or hours, possibly
across process restarts". The autonomy-mode design depends on this:
``CONTRACTS.md#autonomy-hitl`` describes a "pause for human input,
notify the operator, resume on resolution" flow that has no framework
support.

DeferredHITLStep provides the framework half of that contract:

  1. On first invocation, the step submits an Approval to its
     configured ApprovalStore and raises ``ApprovalPendingError``.
  2. The runner / executor catches this specific exception and treats
     it as a soft-suspend (NOT a failure). The Approval ID is exposed
     in the exception so an external resolver knows what to look up.
  3. When the operator resolves the approval (via MCP /approve, REST,
     CLI, etc), they call ``approval_store.resolve(id, decision, ...)``.
  4. When the workflow re-enters the step (the runner re-invokes
     after observing the resolution OR a polling caller retries), the
     step's process() finds a resolved approval and returns the
     decision payload.

This shape is NOT a polling loop inside the step — the step is
*idempotent and stateless*. Suspension is signaled by the exception;
resumption is just re-running. The runner's pause/wake logic is
separate concern (G21 WorkflowRunner; integration is a sibling task).

## Option B — opt-in G5 checkpoint integration (2026-05-11)

The design doc ``nanobrain/docs/g27_g21_wiring_design.md`` records two
resume semantics:

  * Option A (shipped): resume re-invokes the workflow_callable from
    start. Pre-HITL steps run again. Cheap to ship; expensive to run
    when pre-HITL work is heavy.
  * Option B (this addition): the step optionally writes a G5
    ``WorkflowCheckpoint`` of its ``input_data`` immediately before
    raising ``ApprovalPendingError``. The exception carries
    ``checkpoint_manifest_handle``; the runner stores it on
    ``suspension_info`` and forwards it to the resumed workflow.
    Workflow authors check for the handle and short-circuit pre-HITL
    work — pre-HITL steps fire exactly once across suspend + resume.

Option B is **opt-in via the ``checkpoint_dir`` config field**. When
absent, Option A's exact behavior is preserved. When set, the manifest
is content-addressed (G5 ``_FilesystemStorage`` idempotency) by the
deterministic ``approval_id``, so re-suspension across multi-gate
flows reuses the same manifest path. Authors recover ``input_data``
via ``ResumeStep`` against the manifest path.

The framework does NOT auto-skip pre-HITL steps for the workflow author.
That decision lives in workflow code (e.g., ``if
payload.get("__resume_checkpoint_handle__"): jump_to_hitl_step()``),
because the runner has no introspection into an opaque
``workflow_callable``. This keeps the runner's contract minimal and
framework-native (data flows through DataUnits/Links, not magic).

## Decision payload semantics

  * ``approved`` → step output is ``{"decision": "approved",
    "decision_payload": <payload-or-{}>, "approval_id": "...",
    "approved_by": "..."}``. Downstream steps proceed.
  * ``rejected`` → step raises ``ApprovalRejectedError``. The runner
    treats this as a workflow-terminal failure (the human said no).
  * ``corrected`` → step output is ``{"decision": "corrected",
    "decision_payload": <corrected-payload>, "approval_id": "..."}``.
    The corrected payload replaces the upstream input that the
    operator deemed wrong.

## Approval-ID strategy (P6+a decision)

DeferredHITLStep defaults to ``approval_id_strategy="deterministic"``:
the approval_id is SHA-256 of ``(run_id, step_name, prompt)``. This
means a workflow retry produces the SAME approval_id, the
ApprovalStore's idempotent submit() returns the existing record, and
the operator does NOT see duplicate approvals after a transient
failure. Tests that want distinct approvals per call configure
``approval_id_strategy="random"``.

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G27;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import Field

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.library.runtime.approval_store import (
    Approval,
    ApprovalStoreProtocol,
    deterministic_approval_id,
    random_approval_id,
)

logger = logging.getLogger(__name__)


ApprovalIDStrategy = Literal["deterministic", "random"]


class ApprovalPendingError(Exception):
    """Raised by DeferredHITLStep when the approval is still pending.

    Carriers (these MUST stay on the exception so external resolvers
    can route the decision back):

      * ``approval_id``: stable identifier; resolvers call
        ``store.resolve(approval_id, decision, ...)`` against this
      * ``step_name``: which step suspended
      * ``prompt``: human-readable request body the operator sees
      * ``checkpoint_manifest_handle`` (Option B, optional): absolute
        path to a G5 manifest file capturing the step's input_data.
        ``None`` when the step is configured for Option A (no
        checkpoint_dir). The runner forwards this to suspension_info
        so the resumed workflow can rehydrate via ResumeStep.

    The runner / executor catches this and treats it as a soft-suspend.
    User code should NOT catch this — let it propagate to the runner
    layer that knows how to handle suspend/resume.
    """

    def __init__(
        self,
        *,
        approval_id: str,
        step_name: str,
        prompt: str,
        checkpoint_manifest_handle: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"DeferredHITL approval pending for step {step_name!r} "
            f"(approval_id={approval_id!r}). Resolve via "
            f"approval_store.resolve(approval_id, decision, ...)."
        )
        self.approval_id = approval_id
        self.step_name = step_name
        self.prompt = prompt
        self.checkpoint_manifest_handle = checkpoint_manifest_handle


class ApprovalRejectedError(Exception):
    """Raised by DeferredHITLStep when an approval is resolved with
    decision='rejected'. Workflow-terminal: the operator said no."""

    def __init__(
        self,
        *,
        approval_id: str,
        step_name: str,
        rejected_by: Optional[str] = None,
        rejection_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"DeferredHITL approval {approval_id!r} (step {step_name!r}) "
            f"REJECTED by {rejected_by!r}. The workflow should fail "
            f"closed."
        )
        self.approval_id = approval_id
        self.step_name = step_name
        self.rejected_by = rejected_by
        self.rejection_payload = rejection_payload or {}


class DeferredHITLStepConfig(StepConfig):
    """Configuration for DeferredHITLStep.

    The ``approval_store`` field is NOT a config-loadable component —
    stores are not safely YAML-roundtrippable (a FileApprovalStore
    holds an absolute path, an InMemoryApprovalStore holds a singleton
    dict). Pass the store via the ``approval_store=`` kwarg to
    ``DeferredHITLStep.from_config()`` instead.
    """

    prompt_template: str = Field(
        ...,
        description=(
            "Human-readable prompt template. ``{input.<key>}`` tokens "
            "are substituted from the step's input_data dict. Plain "
            "strings (no tokens) are also valid."
        ),
    )
    approval_id_strategy: ApprovalIDStrategy = Field(
        default="deterministic",
        description=(
            "How to derive approval_id. 'deterministic' -> SHA-256 of "
            "(run_id, step_name, rendered_prompt); 'random' -> uuid4. "
            "Deterministic is the default because retries should NOT "
            "create duplicate approvals."
        ),
    )
    checkpoint_dir: Optional[str] = Field(
        default=None,
        description=(
            "Option B (opt-in): when set, the step writes a G5 "
            "filesystem-backed checkpoint of input_data to "
            "``<checkpoint_dir>/<approval_id>.manifest.json`` before "
            "raising ApprovalPendingError. The manifest path is carried "
            "on the exception (``checkpoint_manifest_handle``) and "
            "propagates to suspension_info. Resume workflows recover "
            "input_data via ``ResumeStep`` against the manifest. "
            "Leave None for Option A (re-run-from-start semantics; "
            "no checkpoint written)."
        ),
    )


class DeferredHITLStep(BaseStep):
    """Step that suspends the workflow for human approval.

    Lifecycle:
      * First call: render prompt -> submit Approval to store -> raise
        ApprovalPendingError carrying approval_id.
      * Subsequent call (same approval_id thanks to deterministic
        strategy): fetch the Approval -> dispatch on decision:
          - pending -> raise ApprovalPendingError again (still waiting)
          - approved -> return {"decision": "approved", ...}
          - rejected -> raise ApprovalRejectedError
          - corrected -> return {"decision": "corrected",
                                 "decision_payload": <corrected>}

    The Step is intentionally stateless: the ApprovalStore holds all
    state, and the deterministic approval_id strategy is what makes
    retries idempotent.
    """

    COMPONENT_TYPE: str = "deferred_hitl_step"
    REQUIRED_CONFIG_FIELDS = ["name", "prompt_template"]

    @classmethod
    def _get_config_class(cls):
        return DeferredHITLStepConfig

    @classmethod
    def resolve_dependencies(
        cls, component_config: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        # Inherit BaseStep's executor-resolution chain (step config ->
        # workflow kwarg -> default LocalExecutor) so the step has the
        # standard executor surface alongside the approval store.
        base_deps = super().resolve_dependencies(component_config, **kwargs)

        store = kwargs.get("approval_store")
        if store is None:
            raise ComponentConfigurationError(
                "FAIL-FAST: DeferredHITLStep requires an "
                "'approval_store' kwarg. Pass via from_config(path, "
                "approval_store=<InMemoryApprovalStore() | "
                "FileApprovalStore(...) | custom>). The store is not "
                "YAML-loadable because backends carry runtime state "
                "(filesystem paths, DB connections)."
            )
        # Duck-type check on the protocol — we don't import the
        # Protocol class for runtime isinstance because the
        # integration's PostgresApprovalStore (and similar) won't
        # subclass it directly.
        for method in ("submit", "get", "resolve", "list_pending"):
            if not callable(getattr(store, method, None)):
                raise ComponentConfigurationError(
                    f"FAIL-FAST: DeferredHITLStep approval_store "
                    f"{type(store).__name__} missing required method "
                    f"{method!r}; must implement ApprovalStoreProtocol."
                )
        return {**base_deps, "approval_store": store}

    def _init_from_config(
        self,
        config: DeferredHITLStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)
        self._approval_store: ApprovalStoreProtocol = dependencies[
            "approval_store"
        ]
        self._prompt_template: str = config.prompt_template
        self._approval_id_strategy: ApprovalIDStrategy = (
            config.approval_id_strategy
        )
        self._checkpoint_dir: Optional[Path] = (
            Path(config.checkpoint_dir) if config.checkpoint_dir else None
        )

    # ---- Public API -----------------------------------------------------

    async def process(
        self, input_data: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise ComponentConfigurationError(
                f"FAIL-FAST: DeferredHITLStep {self.name!r} input_data "
                f"must be dict; got {type(input_data).__name__}"
            )

        prompt = self._render_prompt(input_data)
        run_id = self._current_run_id()

        approval_id = self._derive_approval_id(
            run_id=run_id, prompt=prompt
        )

        # Look up first; the deterministic-strategy retry path gets a
        # fast hit on resolved approvals without re-emitting.
        existing = self._approval_store.get(approval_id)
        if existing is not None:
            return self._dispatch_resolution(
                existing, input_data=input_data
            )

        # Fresh request: emit a pending Approval via idempotent submit.
        new_approval = Approval(
            approval_id=approval_id,
            run_id=run_id,
            step_name=self.name,
            prompt=prompt,
        )
        recorded = self._approval_store.submit(new_approval)
        # If submit() returned a different (already-existing) record
        # — raced submission — dispatch on whatever's there.
        if recorded.is_resolved():
            return self._dispatch_resolution(
                recorded, input_data=input_data
            )

        # Option B (opt-in): write G5 checkpoint of input_data before
        # raising. The manifest path becomes the ``checkpoint_manifest_handle``
        # the runner forwards to suspension_info. Idempotent: re-running
        # the same suspended step uses the same approval_id, hence the
        # same manifest path (G5 _FilesystemStorage content-addresses
        # value blobs, but the manifest path itself is deterministic).
        checkpoint_handle = self._maybe_write_checkpoint(
            input_data=input_data, approval_id=recorded.approval_id
        )

        raise ApprovalPendingError(
            approval_id=recorded.approval_id,
            step_name=self.name,
            prompt=prompt,
            checkpoint_manifest_handle=checkpoint_handle,
        )

    # ---- Internals ------------------------------------------------------

    def _render_prompt(self, input_data: Dict[str, Any]) -> str:
        """Substitute ``{input.<key>}`` tokens. Missing keys leave
        the token in place — operators can decide whether to require
        the key or accept partial templating. Future enhancement:
        per-template strict mode."""
        rendered = self._prompt_template
        for key, value in input_data.items():
            token = "{input." + key + "}"
            rendered = rendered.replace(token, str(value))
        return rendered

    def _current_run_id(self) -> Optional[str]:
        """Resolve the active WorkflowRunContext's run_id (if any).
        Lazy import keeps core/step.py from depending on library/."""
        try:
            from nanobrain.library.orchestration.run_context import (
                current_run_context,
            )
        except ImportError:
            return None
        ctx = current_run_context()
        if ctx is None:
            return None
        return getattr(ctx, "run_id", None)

    def _derive_approval_id(
        self, *, run_id: Optional[str], prompt: str
    ) -> str:
        if self._approval_id_strategy == "deterministic":
            return deterministic_approval_id(
                run_id=run_id, step_name=self.name, prompt=prompt
            )
        return random_approval_id()

    def _dispatch_resolution(
        self,
        approval: Approval,
        *,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the step output for the given Approval state.

        Pending -> raise ApprovalPendingError (retry).
        Approved -> return decision payload.
        Rejected -> raise ApprovalRejectedError (workflow-terminal).
        Corrected -> return decision payload + corrected flag.

        ``input_data`` is only consulted on the pending re-raise path
        (Option B re-emits the checkpoint so the same manifest covers
        the second-pending-observation). On resolved paths the
        input_data is ignored — the decision payload comes from the
        store.
        """
        if approval.decision == "pending":
            handle = (
                self._maybe_write_checkpoint(
                    input_data=input_data,
                    approval_id=approval.approval_id,
                )
                if input_data is not None
                else None
            )
            raise ApprovalPendingError(
                approval_id=approval.approval_id,
                step_name=self.name,
                prompt=approval.prompt,
                checkpoint_manifest_handle=handle,
            )
        if approval.decision == "approved":
            return {
                "decision": "approved",
                "approval_id": approval.approval_id,
                "decision_payload": approval.decision_payload or {},
                "decided_by": approval.decided_by,
                "decided_at": approval.decided_at,
            }
        if approval.decision == "corrected":
            return {
                "decision": "corrected",
                "approval_id": approval.approval_id,
                "decision_payload": approval.decision_payload or {},
                "decided_by": approval.decided_by,
                "decided_at": approval.decided_at,
            }
        # "rejected" — workflow-terminal failure.
        raise ApprovalRejectedError(
            approval_id=approval.approval_id,
            step_name=self.name,
            rejected_by=approval.decided_by,
            rejection_payload=approval.decision_payload,
        )

    def _maybe_write_checkpoint(
        self,
        *,
        input_data: Dict[str, Any],
        approval_id: str,
    ) -> Optional[str]:
        """Option B: write a G5 checkpoint of ``input_data`` so the
        resumed workflow can rehydrate via ResumeStep without re-running
        upstream steps. Returns the manifest path, or ``None`` when
        Option B is not configured (preserving Option A behavior).

        We piggy-back on G5's ``_FilesystemStorage`` rather than spawning
        a ``CheckpointStep`` instance because the step's lifecycle
        (no triggers, no executor) is orthogonal to the HITL gate's
        process() call. The manifest schema MUST stay compatible with
        ``ResumeStep`` — which is why we import the private storage
        class and ``_MANIFEST_VERSION`` from checkpoint_resume rather
        than duplicating either.

        Manifest path is ``<checkpoint_dir>/<approval_id>.manifest.json``,
        making re-suspensions of the same approval idempotent (same
        path; G5 content-addresses value blobs by hash).
        """
        if self._checkpoint_dir is None:
            return None
        if not isinstance(input_data, dict):
            # process() already enforces this contract, but guard
            # against direct callers of _dispatch_resolution.
            raise ComponentConfigurationError(
                f"FAIL-FAST: DeferredHITLStep {self.name!r} "
                f"checkpoint_dir set but input_data is not a dict "
                f"(got {type(input_data).__name__})"
            )

        # Lazy import: keeps DeferredHITLStep importable without the
        # G5 module loaded (Option A users pay zero import cost).
        from nanobrain.library.steps.checkpoint_resume import (
            _MANIFEST_VERSION,
            _FilesystemStorage,
            _capture_code_identity,
        )

        ckpt_dir = self._checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = ckpt_dir / f"{approval_id}.manifest.json"

        storage = _FilesystemStorage(base_dir=ckpt_dir)
        entries: Dict[str, Dict[str, Any]] = {}
        # G5's ``_FilesystemStorage.write_value`` declares ``async`` but
        # its body has no awaitable I/O. We mirror its logic inline
        # (via ``_sync_write_value``) so we don't need to schedule a
        # coroutine while already inside ``async def process()``. The
        # descriptor shape MUST stay in sync with G5 — tests cover this.
        for key, value in input_data.items():
            entries[key] = _sync_write_value(storage, value, key)

        manifest = {
            "manifest_version": _MANIFEST_VERSION,
            "step_name": self.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "filesystem",
            "captured": list(entries.keys()),
            "entries": entries,
            "code_identity": _capture_code_identity(),
            # G27-specific metadata; ResumeStep ignores unknown top-level
            # keys but operators can inspect this to confirm the
            # checkpoint originated from a HITL gate.
            "g27_source": {
                "approval_id": approval_id,
                "step_name": self.name,
            },
        }
        # Atomic write: write to .tmp then rename (matches G5).
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        tmp_path.replace(manifest_path)
        return str(manifest_path)


def _sync_write_value(
    storage: Any, value: Any, hint: str
) -> Dict[str, Any]:
    """Sync re-implementation of ``_FilesystemStorage.write_value``.

    The G5 class declares the method ``async`` but its body has no
    awaitable I/O — it canonicalizes the value to JSON, hashes the
    bytes, and writes a content-addressed file. Calling it from inside
    a running event loop would normally require ``await``; we replicate
    the body synchronously here to avoid forcing the caller into
    ``run_until_complete`` (which raises inside a running loop) or
    ``asyncio.run`` (which spawns a competing loop).

    Source of truth for the descriptor shape: G5
    ``_FilesystemStorage.write_value`` at
    ``library/steps/checkpoint_resume.py``. Stays in sync with that
    shape because we'd break ResumeStep otherwise — covered by tests.
    """
    import hashlib

    # Reject stream-shaped values per G5 spec (mirrored from G5).
    if hasattr(value, "__aiter__") and not isinstance(
        value, (str, bytes, dict, list)
    ):
        raise ComponentConfigurationError(
            f"FAIL-FAST: DeferredHITLStep checkpoint cannot snapshot "
            f"stream-shaped value {hint!r} (async iterator); per G5 spec "
            f"streams are not snapshottable"
        )
    try:
        canonical = json.dumps(value, sort_keys=True, default=str)
    except TypeError as e:
        raise ComponentConfigurationError(
            f"FAIL-FAST: DeferredHITLStep checkpoint value {hint!r} "
            f"not JSON-serializable: {e}"
        ) from e
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    target_path = storage._base_dir / f"{content_hash}.json"
    if not target_path.exists():
        target_path.write_text(canonical)
    return {
        "backend": "filesystem",
        "path": str(target_path),
        "content_hash": content_hash,
        "size_bytes": len(canonical.encode("utf-8")),
    }


__all__ = [
    "ApprovalIDStrategy",
    "ApprovalPendingError",
    "ApprovalRejectedError",
    "DeferredHITLStep",
    "DeferredHITLStepConfig",
]
