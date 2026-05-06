"""Unit tests for ``ApprovalStep`` (T10 HITL pause-token primitive).

Mocking strategy
----------------
The HTTP calls to the APECX Control Plane are intercepted via
``httpx.MockTransport``. This is a unit-level contract: these tests verify
``ApprovalStep.process()`` makes the expected HTTP requests and correctly
applies the returned decisions. They do NOT verify that the Control Plane
actually persists anything.

Integration parity (workspace policy)
-------------------------------------
Per the workspace unit-mock / integration-test parity rule, every behavior
mocked here MUST be exercised against a real Control Plane in a companion
integration test. That integration test is the responsibility of the main
agent working in ``apecx-mcp-integration`` and is tracked as part of T10;
location:

    apecx-mcp-integration/tests/integration/test_approval_step_integration.py
    (to be authored — spin up Control Plane via docker-compose or in-process
    ASGI test app, then drive a real workflow through the step.)

Until that integration test runs green, the ApprovalStep is NOT considered
"done" per workspace policy.

Scope of this file
------------------
AC1 (create + poll happy path), AC3 (modifications merge), AC4 (rejection
raises), AC5 (soft-gate timeout + on_timeout), AC6 (resume_approval_id skips
POST) at the step-behavior level. AC2 (server-side persistence across
restart) is out of scope — it requires a real Control Plane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from nanobrain.library.steps.approval_step import (
    ApprovalStep,
    StepRejected,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _write_config(
    tmp_path: Path,
    *,
    kind: str = "hard",
    timeout_seconds: Optional[float] = None,
    on_timeout: str = "reject",
    poll_interval_seconds: float = 0.01,
) -> Path:
    """Write a minimal ApprovalStep YAML to ``tmp_path``. Uses a sub-millisecond
    poll interval so tests that legitimately poll a few times don't add latency.
    """
    timeout_yaml = "null" if timeout_seconds is None else str(timeout_seconds)
    yaml_body = f"""
name: test_approval_gate
description: "Unit-test fixture for ApprovalStep."

input_data_units:
  proposals_input:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: proposals_input

output_data_units:
  approved_output:
    class: "nanobrain.core.data_unit.DataUnitMemory"
    name: approved_output

triggers:
  - class: "nanobrain.core.trigger.DataUnitChangeTrigger"
    data_unit: "proposals_input"

gate_policy:
  kind: {kind}
  timeout_seconds: {timeout_yaml}
  on_timeout: {on_timeout}

control_plane:
  base_url: "http://control-plane.test"
  poll_interval_seconds: {poll_interval_seconds}
  request_timeout_seconds: 5.0
""".strip()
    config_path = tmp_path / "approval_step_test.yml"
    config_path.write_text(yaml_body)
    return config_path


class FakeControlPlane:
    """In-memory stand-in for the Control Plane approvals API.

    Supports:
        - ``POST /approvals/`` → allocates an incrementing id and records the payload.
        - ``GET /approvals/{id}`` → returns the current record, with the status
          taken from a caller-supplied script (list of statuses to return in
          order; the last value is repeated indefinitely).
    """

    def __init__(
        self,
        *,
        status_script: List[str],
        final_approval_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._status_script = list(status_script)
        self._final_overrides = dict(final_approval_overrides or {})
        self._posts: List[Dict[str, Any]] = []
        self._gets_by_id: Dict[str, int] = {}
        self._records: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1

    # --- call-log accessors (for assertions) ---

    @property
    def posts(self) -> List[Dict[str, Any]]:
        return self._posts

    def get_count(self, approval_id: str) -> int:
        return self._gets_by_id.get(approval_id, 0)

    # --- seeding a pre-existing approval (for resume tests) ---

    def seed(self, approval_id: str, *, kind: str = "hard") -> None:
        self._records[approval_id] = {
            "id": approval_id,
            "kind": kind,
            "status": "pending",
            "policy": {},
            "comment": None,
        }

    # --- httpx.MockTransport handler ---

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/approvals/":
            payload = json.loads(request.content.decode("utf-8"))
            approval_id = str(self._next_id)
            self._next_id += 1
            self._posts.append(payload)
            self._records[approval_id] = {
                "id": approval_id,
                "kind": payload.get("kind"),
                "status": "pending",
                "policy": payload.get("policy") or {},
                "comment": None,
            }
            # Real TX1 wraps in CreateApprovalResponse envelope — mirror it so
            # the unit test's contract matches the live Control Plane's.
            return httpx.Response(
                201,
                json={"approval": {"id": approval_id, "status": "pending"}},
            )

        if request.method == "GET" and request.url.path.startswith("/approvals/"):
            approval_id = request.url.path.rsplit("/", 1)[-1]
            if approval_id not in self._records:
                return httpx.Response(404, json={"detail": "not found"})

            n = self._gets_by_id.get(approval_id, 0)
            idx = min(n, len(self._status_script) - 1)
            status = self._status_script[idx]
            self._gets_by_id[approval_id] = n + 1

            record = dict(self._records[approval_id])
            record["status"] = status
            # Merge final-overrides (e.g., modifications, comment) only once the
            # script has reached a terminal status; otherwise leave the record
            # pristine so a partially-polled record looks realistic.
            if status != "pending":
                # Deep-ish merge for "policy" so injected modifications survive.
                overrides = dict(self._final_overrides)
                policy_override = overrides.pop("policy", None)
                if policy_override is not None:
                    record_policy = dict(record.get("policy") or {})
                    record_policy.update(policy_override)
                    record["policy"] = record_policy
                record.update(overrides)
            # Real TX1 wraps in ApprovalResponse envelope.
            return httpx.Response(200, json={"approval": record})

        return httpx.Response(500, json={"detail": "unhandled request in FakeControlPlane"})


def _install_fake(step: ApprovalStep, fake: FakeControlPlane) -> None:
    """Swap ``step._http_client_factory`` for one that uses ``fake.handler``.

    The factory must return an ``httpx.AsyncClient`` (since ``process()`` uses
    it via ``async with``), so we return a client backed by
    ``httpx.MockTransport``.
    """

    def factory() -> httpx.AsyncClient:
        transport = httpx.MockTransport(fake.handler)
        return httpx.AsyncClient(transport=transport, base_url="http://control-plane.test")

    step._http_client_factory = factory


def _make_step(tmp_path: Path, **overrides: Any) -> ApprovalStep:
    config_path = _write_config(tmp_path, **overrides)
    return ApprovalStep.from_config(str(config_path))


# ---------------------------------------------------------------------------
# AC1 — create + poll happy path (hard gate, approved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hard_gate_approved_returns_input_unchanged(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    # First GET still pending (proves we actually polled), second GET approved.
    fake = FakeControlPlane(status_script=["pending", "approved"])
    _install_fake(step, fake)

    input_data = {"proposals": [{"entity": "chikv", "candidate": "Chikungunya virus"}]}
    result = await step.process(input_data, run_id="run-42", step_id="step-42")

    assert result == input_data
    # Exactly one POST, at least two GETs (pending then approved).
    assert len(fake.posts) == 1
    assert fake.posts[0]["run_id"] == "run-42"
    assert fake.posts[0]["step_id"] == "step-42"
    assert fake.posts[0]["kind"] == "hard"
    # summary is the default truncated JSON of input_data.
    assert "chikv" in fake.posts[0]["summary"]
    assert fake.get_count("1") >= 2


# ---------------------------------------------------------------------------
# AC3 — approved_with_modifications merges modifications into input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approved_with_modifications_merges_into_input(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    fake = FakeControlPlane(
        status_script=["approved_with_modifications"],
        final_approval_overrides={
            "policy": {"modifications": {"proposals": ["CHIKV vaccine Z"]}},
            "comment": "preferring more recent candidate",
        },
    )
    _install_fake(step, fake)

    input_data = {"proposals": ["CHIKV vaccine X"], "context": "batch-01"}
    result = await step.process(input_data, run_id="run-99", step_id="step-99")

    # Shallow merge: "proposals" is overwritten, "context" stays.
    assert result["proposals"] == ["CHIKV vaccine Z"]
    assert result["context"] == "batch-01"
    assert len(fake.posts) == 1


# ---------------------------------------------------------------------------
# AC4 — rejected decision raises StepRejected with reviewer comment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_raises_step_rejected(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    fake = FakeControlPlane(
        status_script=["rejected"],
        final_approval_overrides={"comment": "insufficient evidence"},
    )
    _install_fake(step, fake)

    with pytest.raises(StepRejected) as exc_info:
        await step.process({"proposals": []}, run_id="r", step_id="s")

    assert exc_info.value.reason == "insufficient evidence"
    assert exc_info.value.approval_id == "1"


@pytest.mark.asyncio
async def test_rejected_without_comment_uses_default_reason(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    # No comment override — comment stays None in the record.
    fake = FakeControlPlane(status_script=["rejected"])
    _install_fake(step, fake)

    with pytest.raises(StepRejected) as exc_info:
        await step.process({"proposals": []}, run_id="r", step_id="s")

    assert exc_info.value.reason == "rejected by user"


# ---------------------------------------------------------------------------
# AC5 — soft-gate timeout honors on_timeout policy LOCALLY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_gate_timeout_auto_approve_returns_input(tmp_path: Path) -> None:
    step = _make_step(
        tmp_path,
        kind="soft",
        timeout_seconds=0.05,
        on_timeout="auto_approve",
        poll_interval_seconds=0.01,
    )
    # Script stays pending forever — forces the local timeout.
    fake = FakeControlPlane(status_script=["pending"])
    _install_fake(step, fake)

    input_data = {"proposals": ["x", "y"]}
    result = await step.process(input_data, run_id="r", step_id="s")

    assert result == input_data


@pytest.mark.asyncio
async def test_soft_gate_timeout_reject_raises_step_rejected(tmp_path: Path) -> None:
    step = _make_step(
        tmp_path,
        kind="soft",
        timeout_seconds=0.05,
        on_timeout="reject",
        poll_interval_seconds=0.01,
    )
    fake = FakeControlPlane(status_script=["pending"])
    _install_fake(step, fake)

    with pytest.raises(StepRejected) as exc_info:
        await step.process({"proposals": []}, run_id="r", step_id="s")

    # The local-timeout synthetic decision carries an explanatory comment that
    # the step surfaces as the rejection reason.
    assert "soft gate timed out" in exc_info.value.reason


# ---------------------------------------------------------------------------
# AC6 — resume_approval_id skips the POST and polls the existing approval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_approval_id_skips_post_and_polls_existing(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    fake = FakeControlPlane(status_script=["approved"])
    fake.seed("existing-approval-123")
    _install_fake(step, fake)

    result = await step.process(
        {"proposals": ["pre-existing"]},
        run_id="r",
        step_id="s",
        resume_approval_id="existing-approval-123",
    )

    assert result == {"proposals": ["pre-existing"]}
    # No POSTs — we resumed an existing approval.
    assert fake.posts == []
    # And we polled the one that was seeded, not a new id.
    assert fake.get_count("existing-approval-123") >= 1


# ---------------------------------------------------------------------------
# Guardrails — required kwargs & config validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_requires_run_id_and_step_id(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    fake = FakeControlPlane(status_script=["approved"])
    _install_fake(step, fake)

    with pytest.raises(ValueError, match="run_id and step_id"):
        await step.process({"proposals": []}, run_id="r")  # missing step_id
    with pytest.raises(ValueError, match="run_id and step_id"):
        await step.process({"proposals": []}, step_id="s")  # missing run_id


def test_invalid_gate_kind_rejected_at_init(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gate_policy.kind"):
        _make_step(tmp_path, kind="no_such_kind")


def test_invalid_on_timeout_rejected_at_init(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gate_policy.on_timeout"):
        _make_step(tmp_path, kind="soft", timeout_seconds=1.0, on_timeout="maybe")


def test_negative_timeout_rejected_at_init(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gate_policy.timeout_seconds"):
        _make_step(tmp_path, kind="soft", timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# Framework compliance — authored step must not silently override execute()
# ---------------------------------------------------------------------------


def test_approval_step_does_not_override_execute() -> None:
    """If ApprovalStep ever overrides execute, swap the assertion below and
    update the skill — the framework validator would still let this pass
    statically, so this is a belt-and-suspenders check.
    """
    assert "execute" not in ApprovalStep.__dict__, (
        "ApprovalStep must not override execute(); the framework owns it. "
        "See nanobrain-step-authoring skill."
    )


def test_approval_step_process_is_async() -> None:
    import inspect
    assert inspect.iscoroutinefunction(ApprovalStep.process), (
        "ApprovalStep.process must be async def. FAIL-FAST would otherwise trigger "
        "at step initialization."
    )


# ---------------------------------------------------------------------------
# Summary formatter — default truncates at SUMMARY_MAX_CHARS
# ---------------------------------------------------------------------------


def test_default_summary_truncates_long_input(tmp_path: Path) -> None:
    step = _make_step(tmp_path, kind="hard")
    big_input = {"payload": "x" * 5000}
    summary = step._format_summary(big_input)
    assert len(summary) <= ApprovalStep.SUMMARY_MAX_CHARS
