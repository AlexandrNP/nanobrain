"""G115 — workflow-scoped trigger-cascade drain (2026-05-18).

Pins the per-workflow task-scoping behavior added to
``AsyncTriggerExecutor.wait_for_all_tasks``. Without this scoping,
nested ``Workflow.run()`` calls deadlock because the inner cascade
cannot tell its own tasks from the outer's still-awaiting task.

These tests exercise the scoping primitives DIRECTLY on the
executor + ContextVar, NOT through a full Workflow.run(). The
G110 integration tests are the end-to-end check; these unit tests
isolate the mechanism so regressions surface clearly.
"""

from __future__ import annotations

import asyncio
import pytest

from nanobrain.core.trigger import (
    AsyncTriggerExecutor,
    _active_workflow_id,
    _current_workflow_id,
    _tag_task_with_workflow,
)


def _fresh_executor() -> AsyncTriggerExecutor:
    """A fresh AsyncTriggerExecutor that does NOT share the singleton's
    state (so prior tests don't leak background_tasks into ours).

    Constructed directly — the singleton accessor `get_instance` is a
    convenience for production code; tests own their own instance to
    keep `background_tasks` clean."""
    return AsyncTriggerExecutor()


async def _sleep_then_finish(seconds: float = 0.05) -> None:
    """Trivial async work for task fixtures."""
    await asyncio.sleep(seconds)


def test_contextvar_default_is_none():
    """The ContextVar defaults to None — no workflow active by default."""
    assert _current_workflow_id() is None


def test_contextvar_set_and_reset():
    """set() + reset() restore the prior value (supports nesting)."""
    assert _current_workflow_id() is None
    outer_token = _active_workflow_id.set("outer#1")
    try:
        assert _current_workflow_id() == "outer#1"
        inner_token = _active_workflow_id.set("inner#1")
        try:
            assert _current_workflow_id() == "inner#1"
        finally:
            _active_workflow_id.reset(inner_token)
        assert _current_workflow_id() == "outer#1"
    finally:
        _active_workflow_id.reset(outer_token)
    assert _current_workflow_id() is None


@pytest.mark.asyncio
async def test_tag_task_no_op_when_no_active_workflow():
    """When no workflow is active, _tag_task_with_workflow does not
    attach an attribute. The task remains untagged."""
    task = asyncio.create_task(_sleep_then_finish())
    try:
        _tag_task_with_workflow(task)
        assert not hasattr(task, "_nb_workflow_id")
    finally:
        await task


@pytest.mark.asyncio
async def test_tag_task_stamps_active_workflow_id():
    """When a workflow is active, the tag is applied."""
    token = _active_workflow_id.set("wfA#1")
    try:
        task = asyncio.create_task(_sleep_then_finish())
        try:
            _tag_task_with_workflow(task)
            assert getattr(task, "_nb_workflow_id", None) == "wfA#1"
        finally:
            await task
    finally:
        _active_workflow_id.reset(token)


@pytest.mark.asyncio
async def test_wait_for_all_tasks_without_workflow_id_drains_everything():
    executor = _fresh_executor()
    """Legacy behavior (workflow_id=None): wait for ALL tasks in
    background_tasks regardless of tagging. Preserves pre-G115
    semantics for callers that don't pass workflow_id."""
    t1 = asyncio.create_task(_sleep_then_finish())
    t2 = asyncio.create_task(_sleep_then_finish())
    t1._nb_workflow_id = "wfA"  # tagged
    # t2 untagged
    executor.background_tasks.add(t1)
    executor.background_tasks.add(t2)
    t1.add_done_callback(executor.background_tasks.discard)
    t2.add_done_callback(executor.background_tasks.discard)

    drained = await executor.wait_for_all_tasks(timeout=2.0, settle_ms=20)
    assert drained is True
    assert not executor.background_tasks


@pytest.mark.asyncio
async def test_wait_for_workflow_id_only_drains_matching_tasks():
    executor = _fresh_executor()
    """When workflow_id is set, only tasks tagged with that id are
    considered for the drain. Other workflows' tasks (and untagged
    tasks) are ignored — they are not this scope's responsibility."""
    t_a = asyncio.create_task(_sleep_then_finish(0.02))
    t_b_long = asyncio.create_task(asyncio.sleep(10.0))  # would block legacy
    t_a._nb_workflow_id = "wfA"
    t_b_long._nb_workflow_id = "wfB"
    executor.background_tasks.add(t_a)
    executor.background_tasks.add(t_b_long)
    t_a.add_done_callback(executor.background_tasks.discard)
    t_b_long.add_done_callback(executor.background_tasks.discard)

    # Scoped wait for wfA: returns True quickly because wfB's
    # long-sleep task is OUT of scope and not awaited.
    drained_a = await executor.wait_for_all_tasks(
        timeout=1.0, settle_ms=20, workflow_id="wfA"
    )
    assert drained_a is True
    assert t_a.done()
    assert not t_b_long.done()  # wfB still running, NOT drained by scope

    t_b_long.cancel()
    try:
        await t_b_long
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_wait_for_workflow_id_drains_when_only_tagged_tasks_exist():
    executor = _fresh_executor()
    """When the only tasks present are tagged with the requested
    workflow_id, the scoped wait drains them just like legacy."""
    t1 = asyncio.create_task(_sleep_then_finish(0.02))
    t2 = asyncio.create_task(_sleep_then_finish(0.02))
    t1._nb_workflow_id = "wfA"
    t2._nb_workflow_id = "wfA"
    executor.background_tasks.add(t1)
    executor.background_tasks.add(t2)
    t1.add_done_callback(executor.background_tasks.discard)
    t2.add_done_callback(executor.background_tasks.discard)

    drained = await executor.wait_for_all_tasks(
        timeout=1.0, settle_ms=20, workflow_id="wfA"
    )
    assert drained is True
    assert not executor.background_tasks


@pytest.mark.asyncio
async def test_wait_for_workflow_id_returns_true_when_no_matching_tasks():
    executor = _fresh_executor()
    """If NO tasks in the set match the requested workflow_id, the
    scoped wait returns True immediately — there's nothing in scope
    to drain. This is the load-bearing nested-workflow case: inner
    scope sees only the outer's tagged task (different id), so
    inner's wait completes without blocking on it."""
    outer_task = asyncio.create_task(asyncio.sleep(10.0))
    outer_task._nb_workflow_id = "outer"
    executor.background_tasks.add(outer_task)
    outer_task.add_done_callback(executor.background_tasks.discard)

    # Inner scope's wait — should NOT block on the outer's task.
    drained = await executor.wait_for_all_tasks(
        timeout=1.0, settle_ms=20, workflow_id="inner"
    )
    assert drained is True

    outer_task.cancel()
    try:
        await outer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_wait_for_workflow_id_excludes_untagged_tasks():
    executor = _fresh_executor()
    """Untagged tasks (created outside any Workflow.run scope) are
    EXCLUDED from a scoped wait. This is conservative: a scoped
    caller is responsible only for their own workflow's tasks, not
    for foreign tasks created via direct asyncio code."""
    untagged = asyncio.create_task(asyncio.sleep(5.0))
    executor.background_tasks.add(untagged)
    untagged.add_done_callback(executor.background_tasks.discard)

    drained = await executor.wait_for_all_tasks(
        timeout=0.5, settle_ms=20, workflow_id="my_scope"
    )
    assert drained is True
    assert not untagged.done()

    untagged.cancel()
    try:
        await untagged
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_legacy_wait_still_includes_untagged_tasks():
    executor = _fresh_executor()
    """Confirm that the legacy (workflow_id=None) path still drains
    untagged tasks — preserves backward compatibility for callers
    that don't know about G115."""
    untagged = asyncio.create_task(_sleep_then_finish(0.02))
    executor.background_tasks.add(untagged)
    untagged.add_done_callback(executor.background_tasks.discard)

    drained = await executor.wait_for_all_tasks(timeout=1.0, settle_ms=20)
    assert drained is True
    assert untagged.done()
