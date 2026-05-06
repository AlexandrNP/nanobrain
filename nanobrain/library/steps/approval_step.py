"""
ApprovalStep — HITL pause-token primitive for nanobrain workflows.

Pauses a workflow run by POSTing a pending approval to the APECX Control Plane and
polling for the human decision. When the decision arrives, the step applies it to
the input data and returns, resuming the workflow.

Durability model
----------------
Pause state lives in the Control Plane DB, NOT in this process. If the nanobrain
process dies during a pause, the pending approval stays in the Control Plane. On
restart, the caller (workflow runner) is expected to pass ``resume_approval_id``
via ``kwargs`` when re-entering ``process()`` for the same step — the step then
skips the POST and polls the existing approval. Without ``resume_approval_id``
the step always creates a new approval. See the T00.2 spike verdict
(``docs/spikes/async_pause_resume.md``) for why in-process ``asyncio.Event``
was rejected.

Concurrency caveat (T00.2 §3.2)
-------------------------------
``LocalExecutor`` holds a semaphore (default ``max_workers=5``) for every task,
including ones paused here. Single-user laptop deployments are fine; shared
backends need a sized executor or a dedicated blocking-allowed executor class.

Configuration (see ``approval_step.yml`` for an example)
--------------------------------------------------------
- ``gate_policy.kind``: ``"hard"`` | ``"soft"`` | ``"silent"`` | ``"allocation"``.
- ``gate_policy.timeout_seconds``: ``null`` for hard; a number for soft.
- ``gate_policy.on_timeout``: ``"auto_approve"`` | ``"reject"`` (soft only).
- ``control_plane.base_url``: Control Plane HTTP base URL.
- ``control_plane.poll_interval_seconds``: poll cadence while pending.
- ``control_plane.request_timeout_seconds``: per-HTTP-call timeout.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import httpx
from pydantic import Field

from nanobrain.core.step import BaseStep, StepConfig


# Status strings (match Control Plane ``ApprovalStatus`` enum values).
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_APPROVED_WITH_MODIFICATIONS = "approved_with_modifications"
STATUS_REJECTED = "rejected"
STATUS_AUTO_APPROVED = "auto_approved"
STATUS_TIMED_OUT = "timed_out"

# Kind strings (match Control Plane ``ApprovalKind`` enum values).
VALID_KINDS = {"hard", "soft", "silent", "allocation"}

# Soft-gate timeout policies.
ON_TIMEOUT_AUTO_APPROVE = "auto_approve"
ON_TIMEOUT_REJECT = "reject"


class StepRejected(Exception):
    """Raised when an ApprovalStep receives a REJECTED decision from a human reviewer."""

    def __init__(self, reason: str, approval_id: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.approval_id = approval_id


class ApprovalStepConfig(StepConfig):
    """Configuration schema for ApprovalStep.

    Extends ``StepConfig`` with two required nested blocks: ``gate_policy`` and
    ``control_plane``. Both are passed as dicts to keep YAML interpolation
    (``${CONTROL_PLANE_URL}``) simple; shape is validated in
    ``_init_from_config``.
    """

    gate_policy: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gate configuration: kind, timeout_seconds, on_timeout.",
    )
    control_plane: Dict[str, Any] = Field(
        default_factory=dict,
        description="Control Plane client config: base_url, poll_interval_seconds, request_timeout_seconds.",
    )


class ApprovalStep(BaseStep):
    """Pauses a workflow at an HITL gate by POSTing and polling the APECX Control Plane.

    Lifecycle in ``process()``:
        1. Determine approval id (reuse ``kwargs['resume_approval_id']`` if provided,
           otherwise POST ``/approvals/`` to create a new pending row).
        2. Poll ``GET /approvals/{id}`` every ``poll_interval_seconds`` until the
           status leaves ``pending``. For soft gates, ``asyncio.wait_for`` enforces
           ``timeout_seconds`` and the ``on_timeout`` policy applies locally.
        3. Apply the decision:
             - ``approved`` / ``auto_approved`` / ``timed_out`` → return input unchanged.
             - ``approved_with_modifications`` → shallow-merge
               ``approval.policy["modifications"]`` into input_data.
             - ``rejected`` → raise ``StepRejected``.

    Hard design constraints (do not change without re-reading the scope memo):
        - NO in-process ``asyncio.Event`` state. All persistence via HTTP.
        - NO direct DB access to the Control Plane. HTTP only.
        - NO in-process cache of approval decisions across runs.
        - NO SSE for v1. Polling is sufficient per scope decision memo 02.

    Authoritative references:
        - ``apecx-mcp-integration/docs/scope_decisions/02_approval_step_in_nanobrain.md``
        - ``apecx-mcp-integration/docs/spikes/async_pause_resume.md``
    """

    COMPONENT_TYPE: str = "approval_step"
    REQUIRED_CONFIG_FIELDS: List[str] = ["name", "gate_policy", "control_plane"]

    # Defaults used when a sub-field is missing from the YAML.
    DEFAULT_POLL_INTERVAL_SECONDS: float = 2.0
    DEFAULT_REQUEST_TIMEOUT_SECONDS: float = 10.0
    SUMMARY_MAX_CHARS: int = 2000

    @classmethod
    def _get_config_class(cls):
        """Return the ApprovalStepConfig Pydantic class for from_config validation."""
        return ApprovalStepConfig

    @classmethod
    def extract_component_config(cls, config: ApprovalStepConfig) -> Dict[str, Any]:
        """Include gate_policy and control_plane alongside inherited BaseStep fields."""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            "gate_policy": getattr(config, "gate_policy", {}) or {},
            "control_plane": getattr(config, "control_plane", {}) or {},
        }

    def _init_from_config(
        self,
        config: ApprovalStepConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        """Validate and cache gate + control-plane config for use in process()."""
        super()._init_from_config(config, component_config, dependencies)

        gate_policy = component_config.get("gate_policy") or {}
        control_plane = component_config.get("control_plane") or {}

        # Validate gate_policy.kind — wrong values mean the Control Plane will reject the POST
        # later, which is a worse error surface than failing at step init.
        kind = gate_policy.get("kind")
        if kind not in VALID_KINDS:
            raise ValueError(
                f"ApprovalStep '{self.name}': gate_policy.kind must be one of "
                f"{sorted(VALID_KINDS)}, got {kind!r}."
            )
        self._gate_kind: str = kind

        timeout_seconds = gate_policy.get("timeout_seconds")
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0
        ):
            raise ValueError(
                f"ApprovalStep '{self.name}': gate_policy.timeout_seconds must be a positive "
                f"number or null, got {timeout_seconds!r}."
            )
        self._timeout_seconds: Optional[float] = (
            float(timeout_seconds) if timeout_seconds is not None else None
        )

        on_timeout = gate_policy.get("on_timeout", ON_TIMEOUT_REJECT)
        if on_timeout not in {ON_TIMEOUT_AUTO_APPROVE, ON_TIMEOUT_REJECT}:
            raise ValueError(
                f"ApprovalStep '{self.name}': gate_policy.on_timeout must be "
                f"'auto_approve' or 'reject', got {on_timeout!r}."
            )
        self._on_timeout: str = on_timeout

        base_url = control_plane.get("base_url")
        if not base_url or not isinstance(base_url, str):
            raise ValueError(
                f"ApprovalStep '{self.name}': control_plane.base_url is required (got {base_url!r})."
            )
        # Strip trailing slash so concatenation is predictable.
        self._base_url: str = base_url.rstrip("/")
        self._poll_interval_seconds: float = float(
            control_plane.get("poll_interval_seconds", self.DEFAULT_POLL_INTERVAL_SECONDS)
        )
        self._request_timeout_seconds: float = float(
            control_plane.get("request_timeout_seconds", self.DEFAULT_REQUEST_TIMEOUT_SECONDS)
        )

        # Dependency-injection hook: tests replace this with an in-memory fake.
        # Default = real httpx.AsyncClient factory.
        self._http_client_factory = self._default_http_client_factory

        self.nb_logger.info(
            f"ApprovalStep {self.name} initialized",
            gate_kind=self._gate_kind,
            timeout_seconds=self._timeout_seconds,
            on_timeout=self._on_timeout,
            control_plane_base_url=self._base_url,
            poll_interval_seconds=self._poll_interval_seconds,
        )

    def _default_http_client_factory(self) -> httpx.AsyncClient:
        """Build a real httpx.AsyncClient configured with the request timeout."""
        return httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._request_timeout_seconds,
        )

    def _format_summary(self, input_data: Dict[str, Any]) -> str:
        """Default summary = truncated pretty-printed JSON of input_data.

        Subclasses may override this to build a more human-readable summary
        from domain-specific fields (e.g., synonym-approval-gate surfacing
        candidate terms). This is code, not a prompt — the human sees it in
        the MCP approval surface.
        """
        try:
            rendered = json.dumps(input_data, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            # Should not happen because default=str coerces everything,
            # but defend against non-JSON-serializable top-level objects.
            rendered = f"<unserializable input_data: {exc}>"
        return rendered[: self.SUMMARY_MAX_CHARS]

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Pause the workflow on an approval, then apply the decision.

        Args:
            input_data: Dict keyed by input data unit names; shallow-merged with
                any modifications returned from an ``approved_with_modifications``
                decision before being returned.
            **kwargs: Recognized extras:
                - ``run_id`` (str, required): the workflow run id. Propagated to
                  ``POST /approvals/``.
                - ``step_id`` (str, required): the concrete step-instance id.
                - ``artifact_ids`` (list[str], optional): artifact references.
                - ``resume_approval_id`` (str, optional): if set, SKIP the POST
                  and poll this existing approval. This is how restart recovery
                  works — the workflow runner persists the approval id alongside
                  the run state and replays it on resume.

        Returns:
            The input_data, possibly shallow-merged with the reviewer's
            modifications.

        Raises:
            StepRejected: if the reviewer rejected the gate (hard) or timed out
                with ``on_timeout=reject`` (soft).
            ValueError: if required kwargs are missing.
            httpx.HTTPError: on unrecoverable transport failures.
        """
        run_id = kwargs.get("run_id")
        step_id = kwargs.get("step_id")
        if not run_id or not step_id:
            raise ValueError(
                f"ApprovalStep '{self.name}': process() requires run_id and step_id kwargs "
                f"(got run_id={run_id!r}, step_id={step_id!r})."
            )

        artifact_ids = kwargs.get("artifact_ids") or []
        resume_approval_id = kwargs.get("resume_approval_id")

        async with self._http_client_factory() as client:
            if resume_approval_id is not None:
                self.nb_logger.info(
                    f"ApprovalStep {self.name}: resuming existing approval",
                    approval_id=resume_approval_id,
                    run_id=run_id,
                    step_id=step_id,
                )
                approval_id = resume_approval_id
            else:
                approval_id = await self._create_approval(
                    client=client,
                    run_id=run_id,
                    step_id=step_id,
                    input_data=input_data,
                    artifact_ids=artifact_ids,
                )
                self.nb_logger.info(
                    f"ApprovalStep {self.name}: created approval",
                    approval_id=approval_id,
                    run_id=run_id,
                    step_id=step_id,
                )

            approval = await self._poll_until_decided(client=client, approval_id=approval_id)

        return self._apply_decision(approval=approval, input_data=input_data)

    async def _create_approval(
        self,
        *,
        client: httpx.AsyncClient,
        run_id: str,
        step_id: str,
        input_data: Dict[str, Any],
        artifact_ids: List[str],
    ) -> str:
        """POST /approvals/ to create a pending approval row. Returns the approval id."""
        payload = {
            "run_id": run_id,
            "step_id": step_id,
            "kind": self._gate_kind,
            "summary": self._format_summary(input_data),
            "artifact_ids": list(artifact_ids),
            "policy": dict(self._gate_policy_payload()),
        }
        response = await client.post("/approvals/", json=payload)
        response.raise_for_status()
        body = response.json()
        # The Control Plane (TX1) wraps responses in a CreateApprovalResponse
        # envelope: ``{"approval": {...}}``. Unwrap to the inner approval row.
        approval = body.get("approval") if isinstance(body, dict) else None
        approval_id = approval.get("id") if isinstance(approval, dict) else None
        if not approval_id:
            raise ValueError(
                f"ApprovalStep '{self.name}': POST /approvals/ returned no approval.id "
                f"(body={body!r})."
            )
        return str(approval_id)

    def _gate_policy_payload(self) -> Dict[str, Any]:
        """Policy block the Control Plane persists alongside the approval row."""
        return {
            "kind": self._gate_kind,
            "timeout_seconds": self._timeout_seconds,
            "on_timeout": self._on_timeout,
        }

    async def _poll_until_decided(
        self,
        *,
        client: httpx.AsyncClient,
        approval_id: str,
    ) -> Dict[str, Any]:
        """Poll GET /approvals/{id} until status != pending.

        For soft gates with a timeout, wraps the polling loop in
        ``asyncio.wait_for``; on timeout returns a synthetic approval dict whose
        status reflects the ``on_timeout`` policy. The synthetic decision is
        applied LOCALLY per the scope memo — we do NOT call the Control Plane
        to record the timeout (the Control Plane's own timer handles
        durable-timeout cases; the local override is only for the in-flight
        process that's waiting here).
        """
        if self._timeout_seconds is None:
            return await self._poll_loop(client=client, approval_id=approval_id)

        try:
            return await asyncio.wait_for(
                self._poll_loop(client=client, approval_id=approval_id),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            # Local-only synthetic decision. See docstring.
            synthesized_status = (
                STATUS_AUTO_APPROVED
                if self._on_timeout == ON_TIMEOUT_AUTO_APPROVE
                else STATUS_REJECTED
            )
            self.nb_logger.info(
                f"ApprovalStep {self.name}: soft gate timed out locally",
                approval_id=approval_id,
                timeout_seconds=self._timeout_seconds,
                on_timeout=self._on_timeout,
                synthesized_status=synthesized_status,
            )
            return {
                "id": approval_id,
                "status": synthesized_status,
                "policy": {},
                "comment": (
                    f"soft gate timed out after {self._timeout_seconds}s; "
                    f"applied on_timeout={self._on_timeout}"
                ),
            }

    async def _poll_loop(
        self,
        *,
        client: httpx.AsyncClient,
        approval_id: str,
    ) -> Dict[str, Any]:
        """Plain polling loop. No timeout concerns here — the caller wraps this.

        GET /approvals/{id} returns an ``ApprovalResponse`` envelope
        (``{"approval": {...}}``). We unwrap to the inner approval row
        before reading ``status``.
        """
        path = f"/approvals/{approval_id}"
        while True:
            response = await client.get(path)
            response.raise_for_status()
            body = response.json()
            approval = body.get("approval") if isinstance(body, dict) else None
            if not isinstance(approval, dict):
                raise ValueError(
                    f"ApprovalStep '{self.name}': GET {path} returned no approval "
                    f"envelope (body={body!r})."
                )
            status = approval.get("status")
            if status and status != STATUS_PENDING:
                return approval
            await asyncio.sleep(self._poll_interval_seconds)

    def _apply_decision(
        self,
        *,
        approval: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Turn the approval row into a return value (or raise StepRejected)."""
        status = approval.get("status")
        approval_id = approval.get("id")

        if status in (STATUS_APPROVED, STATUS_AUTO_APPROVED, STATUS_TIMED_OUT):
            self.nb_logger.info(
                f"ApprovalStep {self.name}: decision applied (pass-through)",
                approval_id=approval_id,
                status=status,
            )
            return dict(input_data)

        if status == STATUS_APPROVED_WITH_MODIFICATIONS:
            modifications = (approval.get("policy") or {}).get("modifications") or {}
            if not isinstance(modifications, dict):
                raise ValueError(
                    f"ApprovalStep '{self.name}': approval.policy.modifications must be a dict, "
                    f"got {type(modifications).__name__}."
                )
            merged = dict(input_data)
            merged.update(modifications)
            self.nb_logger.info(
                f"ApprovalStep {self.name}: decision applied (with modifications)",
                approval_id=approval_id,
                status=status,
                modified_keys=sorted(modifications.keys()),
            )
            return merged

        if status == STATUS_REJECTED:
            reason = approval.get("comment") or "rejected by user"
            self.nb_logger.info(
                f"ApprovalStep {self.name}: decision applied (rejected)",
                approval_id=approval_id,
                reason=reason,
            )
            raise StepRejected(reason=reason, approval_id=approval_id)

        raise ValueError(
            f"ApprovalStep '{self.name}': unknown approval status {status!r} "
            f"(approval_id={approval_id!r})."
        )
