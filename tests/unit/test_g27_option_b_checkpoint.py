"""G27 Option B — opt-in G5 checkpoint integration tests.

Pins the Option B contract added to DeferredHITLStep on 2026-05-11:

  * When ``checkpoint_dir`` is configured on the step, ``process()``
    writes a G5-compatible manifest of ``input_data`` to
    ``<checkpoint_dir>/<approval_id>.manifest.json`` BEFORE raising
    ``ApprovalPendingError``.
  * The exception carries ``checkpoint_manifest_handle`` (manifest
    path). When ``checkpoint_dir`` is absent, the field is ``None``
    and Option A behavior is preserved exactly.
  * WorkflowRunner forwards the manifest handle into
    ``suspension_info["checkpoint_manifest_handle"]`` and back into
    the resumed payload via the reserved key
    ``__resume_checkpoint_handle__``.
  * The manifest is readable by ``ResumeStep`` — round-trip works
    against the same file format. This is the load-bearing
    composition contract: any workflow that wants no-re-run
    semantics threads the resume handle into a ResumeStep input.
  * Idempotency: multi-cycle suspensions (multi-gate workflows) on
    the same approval_id reuse the same manifest path. The
    deterministic approval_id strategy is what makes this work.

Source: design doc ``nanobrain/docs/g27_g21_wiring_design.md``
("Option B evaluation framework") + the step's docstring.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from nanobrain.library.runtime.approval_store import InMemoryApprovalStore
from nanobrain.library.runtime.workflow_runner import WorkflowRunner
from nanobrain.library.steps.deferred_hitl_step import (
    ApprovalPendingError,
    DeferredHITLStep,
)


# ---------------------------------------------------------------------------
# Helpers (mirror canonical pattern in test_g27_g21_wiring.py)
# ---------------------------------------------------------------------------


def _build_runner() -> WorkflowRunner:
    import yaml as _yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        _yaml.safe_dump(
            {"name": "g27_optb_runner", "task_store_backend": "in_memory"},
            f,
        )
        path = f.name
    return WorkflowRunner.from_config(path)


def _make_step(
    approval_store, *, name: str = "approve_step", **cfg_kwargs
) -> DeferredHITLStep:
    import yaml

    cfg_kwargs.setdefault("name", name)
    cfg_kwargs.setdefault("prompt_template", "approve this?")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        yaml.safe_dump(cfg_kwargs, f)
        path = f.name
    return DeferredHITLStep.from_config(
        path, approval_store=approval_store
    )


# ---------------------------------------------------------------------------
# Option A preservation: checkpoint_dir absent => no manifest written
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_option_a_unchanged_when_no_checkpoint_dir(tmp_path):
    """Step without checkpoint_dir raises ApprovalPendingError with
    ``checkpoint_manifest_handle=None``. No files created. This is the
    bit-for-bit Option A behavior — the new field is opt-in."""
    store = InMemoryApprovalStore()
    step = _make_step(store)

    with pytest.raises(ApprovalPendingError) as exc_info:
        await step.process({"k": "v"})

    assert exc_info.value.checkpoint_manifest_handle is None
    # No files created anywhere — confirm we didn't accidentally
    # mkdir a default location.
    # (We can only confirm tmp_path is empty; a global default would
    # be a regression but is hard to assert against in unit scope.)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Option B: checkpoint_dir present => manifest written, handle on exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_option_b_writes_manifest_and_carries_handle(tmp_path):
    store = InMemoryApprovalStore()
    ckpt_dir = tmp_path / "hitl_ckpts"
    step = _make_step(store, checkpoint_dir=str(ckpt_dir))

    input_data = {
        "query": "find the answer",
        "retrieved_docs": [{"id": 1, "text": "doc one"}],
        "scratchpad": {"step": 3},
    }

    with pytest.raises(ApprovalPendingError) as exc_info:
        await step.process(input_data)

    handle = exc_info.value.checkpoint_manifest_handle
    assert handle is not None, "Option B must carry a manifest handle"
    manifest_path = Path(handle)
    assert manifest_path.is_file()
    assert manifest_path.parent == ckpt_dir

    # Manifest is a JSON file with the G5 schema fields.
    manifest = json.loads(manifest_path.read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["backend"] == "filesystem"
    assert manifest["step_name"] == step.name
    assert set(manifest["captured"]) == set(input_data.keys())
    # G27-specific provenance keyed under g27_source.
    assert manifest["g27_source"]["approval_id"] == exc_info.value.approval_id
    assert manifest["g27_source"]["step_name"] == step.name


@pytest.mark.asyncio
async def test_option_b_resume_step_can_load_the_manifest(tmp_path):
    """The composition contract: Option B's manifest MUST be loadable
    via ResumeStep, since that is the framework-native primitive
    workflow authors use to rehydrate input_data on resume."""
    from nanobrain.library.steps.checkpoint_resume import ResumeStep

    store = InMemoryApprovalStore()
    ckpt_dir = tmp_path / "hitl_ckpts"
    step = _make_step(store, checkpoint_dir=str(ckpt_dir))

    payload_before_suspend = {
        "query": "load-bearing question",
        "retrieved_docs": [{"id": 42, "text": "the answer"}],
        "tokens_used": 1234,
    }
    with pytest.raises(ApprovalPendingError) as exc_info:
        await step.process(payload_before_suspend)
    manifest_path = exc_info.value.checkpoint_manifest_handle
    assert manifest_path is not None

    # Build a ResumeStep via tmp YAML and load the manifest.
    import yaml as _yaml
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as f:
        _yaml.safe_dump({"name": "resume_step"}, f)
        resume_path = f.name
    resume_step = ResumeStep.from_config(resume_path)

    restored = await resume_step.process({"manifest_path": manifest_path})

    # Every original key is restored with its original value.
    for k, v in payload_before_suspend.items():
        assert restored[k] == v, f"key {k!r} did not round-trip"
    # Bookkeeping keys ResumeStep adds.
    assert restored["_resumed_from_manifest"] == manifest_path
    assert "_resumed_at" in restored


@pytest.mark.asyncio
async def test_option_b_idempotent_manifest_path_on_resuspend(tmp_path):
    """Multi-cycle suspension on the SAME approval_id reuses the SAME
    manifest path. This is what makes Option B safe under retries —
    no exploding-checkpoint-dir as a workflow re-enters the gate."""
    store = InMemoryApprovalStore()
    ckpt_dir = tmp_path / "hitl_ckpts"
    step = _make_step(store, checkpoint_dir=str(ckpt_dir))

    input_data = {"x": 1}

    with pytest.raises(ApprovalPendingError) as exc1:
        await step.process(input_data)
    with pytest.raises(ApprovalPendingError) as exc2:
        await step.process(input_data)

    assert (
        exc1.value.checkpoint_manifest_handle
        == exc2.value.checkpoint_manifest_handle
    )
    # Exactly one manifest file in the dir.
    manifests = list(ckpt_dir.glob("*.manifest.json"))
    assert len(manifests) == 1


@pytest.mark.asyncio
async def test_option_b_resolved_path_does_not_rewrite_manifest(tmp_path):
    """Once the approval is resolved, process() returns the decision
    payload — there's no second checkpoint write, the manifest from
    the suspended cycle remains untouched. Resolution path is
    short-circuit (the existing manifest is already what the resume
    needs)."""
    store = InMemoryApprovalStore()
    ckpt_dir = tmp_path / "hitl_ckpts"
    step = _make_step(store, checkpoint_dir=str(ckpt_dir))

    with pytest.raises(ApprovalPendingError) as exc:
        await step.process({"x": 1})
    manifest_path = Path(exc.value.checkpoint_manifest_handle)
    bytes_before = manifest_path.read_bytes()

    # Operator resolves.
    store.resolve(
        exc.value.approval_id,
        decision="approved",
        decided_by="op",
        decision_payload={"ok": True},
    )

    # Re-invoke — returns the approved payload.
    result = await step.process({"x": 1})
    assert result["decision"] == "approved"
    # And the manifest file is unchanged.
    assert manifest_path.read_bytes() == bytes_before


# ---------------------------------------------------------------------------
# WorkflowRunner integration: suspension_info + resume_suspended carry the
# handle into the resumed workflow's payload.
# ---------------------------------------------------------------------------


def test_runner_forwards_handle_to_suspension_info_and_resumed_payload(
    tmp_path,
):
    """End-to-end: detached run hits the HITL gate, status flips to
    suspended, suspension_info carries the checkpoint handle, AND the
    resumed payload contains ``__resume_checkpoint_handle__``."""

    async def _scenario():
        runner = _build_runner()
        store = InMemoryApprovalStore()
        ckpt_dir = tmp_path / "hitl_ckpts"
        step = _make_step(store, checkpoint_dir=str(ckpt_dir))

        # Workflow callable: capture the payload it actually receives
        # on resume so we can assert on the reserved key.
        received_payloads = []

        async def _workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
            received_payloads.append(dict(payload))
            return await step.process(payload)

        original_payload = {"q": "ping"}
        await runner.run_detached(
            _workflow, "task-optb", original_payload
        )

        # Wait for suspension.
        for _ in range(50):
            h = await runner.get_handle("task-optb")
            if h.status == "suspended":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError("did not suspend")

        # suspension_info carries the manifest handle.
        manifest_handle = h.suspension_info.get("checkpoint_manifest_handle")
        assert manifest_handle is not None
        assert Path(manifest_handle).is_file()

        # Resolve and resume.
        store.resolve(
            h.suspension_info["approval_id"],
            decision="approved",
            decided_by="op",
            decision_payload={"ok": True},
        )
        await runner.resume_suspended("task-optb")
        await runner.await_completion("task-optb", timeout=2.0)

        # Workflow saw the resumed payload with the reserved key.
        # First entry is the original (pre-suspend); second is the
        # resume entry — it MUST carry __resume_checkpoint_handle__.
        assert len(received_payloads) == 2
        assert (
            received_payloads[0].get("__resume_checkpoint_handle__")
            is None
        )
        assert received_payloads[1][
            "__resume_checkpoint_handle__"
        ] == manifest_handle
        # Other payload keys preserved bit-for-bit.
        assert received_payloads[1]["q"] == "ping"

        # And the original payload object stored in the runner was NOT
        # mutated — defensive copy.
        assert "__resume_checkpoint_handle__" not in original_payload

    asyncio.run(_scenario())


def test_runner_option_a_path_no_resume_key_in_payload(tmp_path):
    """When the step ran in Option A mode (no checkpoint_dir),
    suspension_info.checkpoint_manifest_handle is None and the resumed
    payload does NOT carry ``__resume_checkpoint_handle__``. Option A
    behavior preserved bit-for-bit through the runner integration."""

    async def _scenario():
        runner = _build_runner()
        store = InMemoryApprovalStore()
        step = _make_step(store)  # no checkpoint_dir

        received_payloads = []

        async def _workflow(payload: Dict[str, Any]) -> Dict[str, Any]:
            received_payloads.append(dict(payload))
            return await step.process(payload)

        await runner.run_detached(_workflow, "task-opta", {"q": "x"})
        for _ in range(50):
            h = await runner.get_handle("task-opta")
            if h.status == "suspended":
                break
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError("did not suspend")

        assert h.suspension_info["checkpoint_manifest_handle"] is None

        store.resolve(
            h.suspension_info["approval_id"],
            decision="approved",
            decided_by="op",
        )
        await runner.resume_suspended("task-opta")
        await runner.await_completion("task-opta", timeout=2.0)

        assert (
            "__resume_checkpoint_handle__" not in received_payloads[1]
        )

    asyncio.run(_scenario())


@pytest.mark.asyncio
async def test_option_b_rejects_stream_shaped_value(tmp_path):
    """G5's filesystem backend (which Option B reuses) FAIL-FASTs
    on async-iterator values per the G5 spec — streams are not
    snapshottable. The G27 step inherits this rejection so an
    operator who accidentally hands a stream to a HITL gate sees a
    clear error rather than a silently-truncated manifest."""
    store = InMemoryApprovalStore()
    ckpt_dir = tmp_path / "hitl_ckpts"
    step = _make_step(store, checkpoint_dir=str(ckpt_dir))

    class _StreamLike:
        def __aiter__(self):  # noqa: D401 - protocol shape
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    with pytest.raises(Exception) as exc:
        await step.process({"feed": _StreamLike()})
    msg = str(exc.value)
    assert "stream" in msg.lower(), (
        f"Expected a stream-rejection error; got {msg!r}"
    )
