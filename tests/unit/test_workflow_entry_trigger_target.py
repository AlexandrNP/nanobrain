"""Tests for G22 Step 2 — target_workflow dotted-path resolution.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G22``.

Coverage:
1. target_workflow resolves to an async callable.
2. target_workflow resolves to an instance with a .run method.
3. target_workflow + workflow_callable kwarg: kwarg wins.
4. Neither set: FAIL-FAST.
5. Class as target rejected (FAIL-FAST with explicit instantiation hint).
6. Module not importable: FAIL-FAST.
7. Attribute not found: FAIL-FAST.
8. Resolved value not callable and not .run-bearing: FAIL-FAST.
9. End-to-end: target_workflow path drives a detached run.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.trigger import (
    EventTrigger,
    TriggerConfig,
    TriggerType,
)
from nanobrain.library.runtime import WorkflowEntryTrigger, WorkflowRunner
from nanobrain.library.runtime.entry_triggers import _resolve_workflow_target


# ---------------------------------------------------------------------------
# Module-level fixtures (must be importable for dotted-path resolution)
# ---------------------------------------------------------------------------

async def _async_callable_target(payload):
    return {"resolved": "async_callable", "payload": payload}


def _sync_callable_target(payload):
    return {"resolved": "sync_callable", "payload": payload}


class _FakeWorkflowInstance:
    """Quacks like a Workflow: has a .run(payload) async method."""

    async def run(self, payload):
        return {"resolved": "instance.run", "payload": payload}


_workflow_instance = _FakeWorkflowInstance()


class _SomeClass:
    """Class shape — should be rejected (no ad-hoc instantiation)."""
    pass


_not_callable_value = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path) -> WorkflowRunner:
    yml = tmp / "runner.yml"
    yml.write_text(yaml.safe_dump({
        "name": "r", "task_store_backend": "in_memory",
    }))
    return WorkflowRunner.from_config(str(yml))


def _build_event_trigger() -> EventTrigger:
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(
            name="ev", trigger_type=TriggerType.EVENT,
            debounce_ms=0, max_frequency_hz=1000.0,
        )
    finally:
        TriggerConfig._allow_direct_instantiation = False
    return EventTrigger.from_config(cfg)


def _build_entry(tmp: Path, *, target_workflow=None,
                 workflow_callable=None, runner=None, inner=None,
                 on_launch=None):
    cfg = {"name": "entry"}
    if target_workflow:
        cfg["target_workflow"] = target_workflow
    yml = tmp / "entry.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return WorkflowEntryTrigger.from_config(
        str(yml),
        runner=runner,
        inner_trigger=inner,
        workflow_callable=workflow_callable,
        on_launch=on_launch,
    )


# ---------------------------------------------------------------------------
# 1-2. Resolution shapes
# ---------------------------------------------------------------------------

class TestResolutionShapes:

    def test_async_callable(self):
        fn = _resolve_workflow_target(__name__ + "._async_callable_target")
        assert fn is _async_callable_target

    def test_sync_callable(self):
        """A sync callable is also accepted — the framework awaits it
        only if it returns a coroutine (caller responsibility). v1 does
        not enforce async; this is intentional to avoid coupling to the
        callable's signature."""
        fn = _resolve_workflow_target(__name__ + "._sync_callable_target")
        assert fn is _sync_callable_target

    def test_instance_with_run_method(self):
        fn = _resolve_workflow_target(__name__ + "._workflow_instance")
        # The bound method should be returned, not the instance
        assert fn == _workflow_instance.run
        assert callable(fn)


# ---------------------------------------------------------------------------
# 3-4. Precedence + missing config
# ---------------------------------------------------------------------------

class TestPrecedence:

    def test_kwarg_wins_over_target_workflow(self):
        """When both workflow_callable kwarg AND target_workflow YAML
        are set, the kwarg wins (programmatic > YAML). This matches the
        framework's general principle that explicit > implicit."""
        async def kwarg_callable(payload):
            return {"from": "kwarg"}

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            inner = _build_event_trigger()
            entry = _build_entry(
                tmp, runner=runner, inner=inner,
                target_workflow=__name__ + "._async_callable_target",
                workflow_callable=kwarg_callable,
            )
            assert entry._workflow_callable is kwarg_callable

    def test_neither_set_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            inner = _build_event_trigger()
            with pytest.raises(ComponentConfigurationError) as exc_info:
                _build_entry(tmp, runner=runner, inner=inner)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "workflow_callable" in str(exc_info.value)
            assert "target_workflow" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Class rejection
# ---------------------------------------------------------------------------

class TestClassRejection:

    def test_class_target_rejected_with_hint(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _resolve_workflow_target(__name__ + "._SomeClass")
        msg = str(exc_info.value)
        assert "FAIL-FAST" in msg
        assert "class" in msg
        assert "from_config" in msg, "Error must hint at the from_config fix"


# ---------------------------------------------------------------------------
# 6-8. Resolution failure modes
# ---------------------------------------------------------------------------

class TestResolutionFailures:

    def test_module_not_importable(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _resolve_workflow_target("nonexistent_pkg_q9.thing")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "not importable" in str(exc_info.value)

    def test_attribute_not_found(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _resolve_workflow_target(__name__ + ".nonexistent_attr")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_resolved_to_non_callable_no_run(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _resolve_workflow_target(__name__ + "._not_callable_value")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "neither callable nor an instance with a callable .run" in str(exc_info.value)

    def test_bad_spec_no_dot(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _resolve_workflow_target("no_dot_here")
        assert "FAIL-FAST" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 9. End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_target_workflow_drives_run_detached(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(
                    tmp, runner=runner, inner=inner,
                    target_workflow=__name__ + "._async_callable_target",
                    on_launch=on_launch,
                )
                await entry.start()
                await inner.fire_event({"kind": "x"})
                await asyncio.sleep(0.1)
                h = await runner.await_completion(launches[0], timeout=5)
                assert h.status == "completed"
                assert h.result["resolved"] == "async_callable"
        asyncio.run(run())

    def test_target_workflow_instance_run_method(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(
                    tmp, runner=runner, inner=inner,
                    target_workflow=__name__ + "._workflow_instance",
                    on_launch=on_launch,
                )
                await entry.start()
                await inner.fire_event({"kind": "x"})
                await asyncio.sleep(0.1)
                h = await runner.await_completion(launches[0], timeout=5)
                assert h.status == "completed"
                assert h.result["resolved"] == "instance.run"
        asyncio.run(run())
