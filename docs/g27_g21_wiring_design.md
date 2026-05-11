# G27 ↔ G21 wiring — design doc (deferred wiring)

**Status:** **DEFERRED with explicit design recorded.** The G27 primitive
(``DeferredHITLStep`` + ``ApprovalStore``) shipped 2026-05-09 in
nanobrain commits leading up to ``c3b4b86``. The runner-side wiring
that auto-suspends a detached run on ``ApprovalPendingError`` and
auto-resumes on ``ApprovalStore.resolve(...)`` is recorded here but
NOT implemented in this chain.

**Why deferred:** the wiring requires choosing between two materially
different resume semantics with different infrastructure costs.
Solo-implementing one without recording the choice would commit the
project to that semantic before review. This doc surfaces the choice
so it can be made deliberately.

---

## Problem statement

A workflow run inside ``WorkflowRunner.run_detached(...)`` may invoke
a ``DeferredHITLStep`` mid-run. When it does, the step raises
``ApprovalPendingError`` carrying ``approval_id`` + ``step_name`` +
``prompt``. Today the runner's exception path treats this as a normal
``failed`` lifecycle transition: the task lifetime ends, the operator
sees a "failed" status, and the workflow does not resume on
resolution.

The desired behavior is **soft-suspend**: the runner sees
``ApprovalPendingError``, transitions the task to a new
``"suspended"`` state (NOT ``"failed"``), persists enough state to
correlate the suspension with an external resolution event, and
when the resolution arrives, transitions back to ``"running"`` and
re-invokes the workflow.

---

## Two resume semantics

### Option A — Resume-from-start (deterministic re-run)

The runner's resume is a literal re-invocation of the original
workflow with the original payload. The workflow runs deterministically
to the same ``DeferredHITLStep`` invocation, the step finds its
existing ``Approval`` in the store (deterministic ``approval_id`` per
G27 P6+a default), sees it RESOLVED, and returns the decision.

**Trade-offs:**
- ✅ No new framework state. The ApprovalStore already holds
  everything. The runner's `_run_until_done` just runs the
  workflow callable again.
- ✅ Idempotent by construction: the deterministic ``approval_id``
  hash means re-runs hit the same record.
- ❌ Steps BEFORE the deferred-HITL step run twice. Pure-compute
  steps are fine; LLM-bound steps re-charge cost; side-effecting
  steps (writes to a DB, posts to an API) double-fire.
- ❌ Re-running a long pre-HITL pipeline is expensive — a workflow
  that did 20 minutes of retrieval before hitting an approval gate
  pays that 20 minutes again on every resume.

**Required mitigation:** workflows that use deferred-HITL must
gate side-effecting steps behind G5 checkpoints OR keep them
post-approval. This is workflow-author discipline, not framework
enforcement.

### Option B — Resume-from-step (G5 checkpoint integration)

The runner's resume continues from the step that suspended. Requires:

1. The ``DeferredHITLStep`` write a G5 ``WorkflowCheckpoint`` immediately
   before raising ``ApprovalPendingError`` (capturing all upstream data
   units).
2. The runner persist the suspension marker (approval_id ↔ checkpoint
   handle) in a durable store (Postgres-backed; G21 Step 4 ships the
   ``PostgresTaskStore``).
3. On resume, the runner re-creates the workflow from the checkpoint
   (G5's ``ResumeStep`` machinery), positions execution at the
   suspended step, and re-invokes only that step (which now finds the
   resolved approval and returns).

**Trade-offs:**
- ✅ No re-run of pre-HITL steps. Side-effecting steps fire once.
  Long pipelines resume in seconds, not minutes.
- ❌ Significant new state surface: every suspension produces a
  checkpoint manifest that must be findable + valid + non-stale.
- ❌ Cross-process resume: a task suspended in process A and
  resumed in process B requires the checkpoint to live in shared
  storage (Redis/Postgres) and the workflow class to be importable
  in B (G5 already requires this; no new constraint).
- ❌ "What if the workflow YAML changed between suspension and
  resume?" is now a real concern — the checkpoint's content_hash
  pins the workflow's identity, and a hash mismatch on resume is
  fail-fast.

---

## Recommendation (pending operator review)

**Ship Option A first** as the v1 wiring. Operators who want
Option B's no-re-run semantic compose ``DeferredHITLStep`` with
``CheckpointStep`` + ``ResumeStep`` (already shipped) by hand:

```yaml
steps:
  big_retrieval:
    class: my.RetrieveStep
    config: {...}
  checkpoint_before_approval:
    class: nanobrain.library.steps.CheckpointStep
    config: {...}
  hitl_gate:
    class: nanobrain.library.steps.DeferredHITLStep
    config: {...}
  apply_decision:
    class: my.ApplyDecisionStep
    config: {...}
```

A future Option B framework-side wiring can land as G27.2 after
operator deployment data tells us re-run cost is the dominant
pain point.

## Concrete v1 (Option A) implementation sketch

In ``nanobrain/library/runtime/workflow_runner.py``:

1. Add lifecycle state ``"suspended"`` to ``_STATUS_VALID`` AND
   ``_STATUS_ACTIVE`` tuples.

2. In ``_run_workflow_to_completion`` (or wherever the asyncio
   task body lives), wrap the workflow invocation:

   ```python
   try:
       result = await workflow_callable(payload)
   except ApprovalPendingError as exc:
       # Soft-suspend: do NOT mark failed.
       await self._task_store.update(
           task_id,
           status="suspended",
           extra={
               "suspension_kind": "deferred_hitl",
               "approval_id": exc.approval_id,
               "step_name": exc.step_name,
               "prompt": exc.prompt,
           },
       )
       return  # exit asyncio task; the runner will re-spawn on resolve
   ```

3. Add ``WorkflowRunner.resume(task_id)``:

   ```python
   def resume(self, task_id: str) -> None:
       handle = self._handles[task_id]
       if handle.status != "suspended":
           raise ValueError(...)
       # Re-invoke run_detached with the original payload. The
       # ApprovalStore now has the resolved approval; the
       # DeferredHITLStep will find it and return.
       self._spawn_task(task_id, handle.workflow_callable, handle.payload)
   ```

4. Add an optional ``approval_store`` kwarg on ``run_detached`` so
   the runner can subscribe to resolution events (or operators can
   subscribe externally and call ``runner.resume(task_id)``).

5. Tests:
   - Suspend-on-pending: assert task status transitions to
     ``"suspended"``, not ``"failed"``.
   - Resume-after-resolve: asserts the post-approval payload
     reaches the asyncio task's return.
   - Re-run idempotency: side-effecting step in a pre-HITL position
     fires twice (and the test names this as expected behavior under
     Option A).

---

## Cross-references

- ``nanobrain/library/steps/deferred_hitl_step.py`` — G27 primitive (shipped)
- ``nanobrain/library/runtime/approval_store.py`` — ApprovalStore protocol + 2 backends
- ``nanobrain/library/runtime/workflow_runner.py`` — G21 WorkflowRunner (no G27 hooks yet)
- ``nanobrain/library/steps/checkpoint_resume.py`` — G5 CheckpointStep + ResumeStep
- ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md`` Round 3 G27
- ``apecx-mcp-integration/docs/development_roadmap.md`` §8.8 (P6+a)

## Decision needed

Operator picks one:
1. Ship Option A v1 (recommended; minimal new state). **DONE
   2026-05-11** — commit `abd41b5` (in-memory) + commit `3cb87c5`
   (SQLite + Postgres serialization).
2. Wait for Option B (full G5-checkpoint integration); ship neither
   until then. **REJECTED** in favor of Option A v1.
3. Both: A first, B as G27.2. **CURRENT STATE** — A v1 is live;
   B is evaluated below.

The G27 primitive is already shipped + tested; Option A v1 is now
also shipped + tested.

## Option B evaluation framework (instrumentation, 2026-05-11)

Premature implementation of Option B (G5-checkpoint integration) is
speculative — operators decide whether Option A's re-run-from-start
semantic is acceptable based on deployment data, not on theoretical
re-run cost.

This commit ships **instrumentation only**:

### DetachedTaskHandle.resume_count

A new integer field on the handle, incremented each time
``WorkflowRunner.resume_suspended(task_id)`` re-spawns the workflow.
Persisted in SQLite + Postgres TaskStores via the same migration
that landed ``suspension_info_json``.

### How operators measure Option B's value

Combine ``resume_count`` with ``cost_actual`` (G26 tracking) to
compute the load-bearing decision metric:

```python
# Pseudo-SQL against the Postgres task store
SELECT
  task_id,
  resume_count,
  cost_actual_json,
  EXTRACT(EPOCH FROM (completed_at - created_at)) AS total_seconds
FROM nanobrain_detached_tasks
WHERE resume_count > 0
ORDER BY resume_count DESC;
```

Per-task re-run cost is `resume_count × pre_hitl_step_cost`. The
fraction of cumulative cost spent on re-runs vs. real work tells
operators whether Option B's no-re-run semantic would pay back the
G5 checkpoint integration's complexity cost.

### Decision rule of thumb (recommendation)

  * `resume_count × pre_hitl_step_seconds < 30s` per task: stick
    with Option A. The re-run cost is dwarfed by other workflow
    overhead.
  * `resume_count × pre_hitl_step_seconds > 5 minutes` per task,
    OR a meaningful fraction of users hit multi-resume cycles:
    promote Option B to the next chain. The G5 checkpoint integration
    is justified.
  * In between: re-evaluate quarterly. Operators may discover the
    pain point only after a particular workflow shape becomes
    popular.

### What Option B implementation would entail (for reference)

When the operator chooses to implement Option B:

  1. ``DeferredHITLStep`` writes a G5 ``WorkflowCheckpoint`` (via
     CheckpointStep) IMMEDIATELY before raising ApprovalPendingError.
     Captures all upstream DataUnits + the step's input.
  2. ``WorkflowRunner.run_detached`` records the checkpoint manifest
     handle in ``suspension_info["checkpoint_handle"]``.
  3. ``resume_suspended`` re-creates the workflow from the manifest
     (G5's ResumeStep machinery) at the suspended step boundary,
     re-invokes ONLY that step (which now finds the resolved approval
     and returns).
  4. Workflow content_hash pin: if the workflow YAML changed between
     suspend + resume, FAIL-FAST (already a G5 contract — the manifest
     pins the workflow's identity).
  5. New tests: a multi-step workflow with a non-trivial pre-HITL
     pipeline. Assert pre-HITL steps fire exactly once across the
     suspend + resume cycle.

The G5 checkpoint primitive (``CheckpointStep`` +
``ResumeStep``) is already shipped (commit `c84b510` lineage).
Option B is composition work, not new framework primitives.
