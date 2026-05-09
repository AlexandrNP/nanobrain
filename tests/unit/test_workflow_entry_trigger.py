"""Tests for G22 — WorkflowEntryTrigger (the workflow-start half).

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G22``.

Coverage:
1. WorkflowEntryTriggerConfig validation.
2. from_config requires runner / inner_trigger / workflow_callable kwargs.
3. End-to-end: EventTrigger -> WorkflowEntryTrigger -> WorkflowRunner.run_detached.
4. Payload pass-through for dict event bodies.
5. Payload synthesis for non-dict event bodies ({event_body: ...}).
6. payload_factory dotted-path resolution.
7. autonomy_level + cost_envelope_template forwarded into payload metadata.
8. Multiple inner-fire events produce distinct task IDs.
9. on_launch callback invoked with each handle.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.trigger import (
    EventTrigger,
    TriggerConfig,
    TriggerType,
)
from nanobrain.library.runtime import (
    WorkflowEntryTrigger,
    WorkflowEntryTriggerConfig,
    WorkflowRunner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path) -> WorkflowRunner:
    yml = tmp / "runner.yml"
    yml.write_text(yaml.safe_dump({
        "name": "r", "task_store_backend": "in_memory",
    }))
    return WorkflowRunner.from_config(str(yml))


def _build_event_trigger(name: str = "ev") -> EventTrigger:
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(
            name=name,
            trigger_type=TriggerType.EVENT,
            debounce_ms=0,
            max_frequency_hz=1000.0,
        )
    finally:
        TriggerConfig._allow_direct_instantiation = False
    return EventTrigger.from_config(cfg)


async def _echo_workflow(payload: dict) -> dict:
    await asyncio.sleep(0.01)
    return {"echoed": payload}


def _build_entry(tmp: Path, runner, inner, workflow_callable,
                 *, payload_factory=None, autonomy_level="strict_hitl",
                 cost_envelope_template=None, on_launch=None):
    cfg = {"name": "entry", "autonomy_level": autonomy_level}
    if payload_factory:
        cfg["payload_factory"] = payload_factory
    if cost_envelope_template:
        cfg["cost_envelope_template"] = cost_envelope_template
    yml = tmp / "entry.yml"
    yml.write_text(yaml.safe_dump(cfg))
    return WorkflowEntryTrigger.from_config(
        str(yml),
        runner=runner,
        inner_trigger=inner,
        workflow_callable=workflow_callable,
        on_launch=on_launch,
    )


# Module-level callable for payload_factory dotted-path resolution test.
def _factory_for_test(event_body: Any) -> dict:
    if isinstance(event_body, dict):
        return {"factory_built": True, **event_body}
    return {"factory_built": True, "raw": event_body}


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestEntryConfig:

    def _build(self, **kwargs):
        WorkflowEntryTriggerConfig._allow_direct_instantiation = True
        try:
            return WorkflowEntryTriggerConfig(**kwargs)
        finally:
            WorkflowEntryTriggerConfig._allow_direct_instantiation = False

    def test_defaults(self):
        cfg = self._build(name="t")
        assert cfg.autonomy_level == "strict_hitl"
        assert cfg.payload_factory is None

    def test_invalid_autonomy_level_rejected(self):
        with pytest.raises(Exception):
            self._build(name="t", autonomy_level="invalid")


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_missing_runner_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inner = _build_event_trigger()
            with pytest.raises(ComponentConfigurationError) as exc_info:
                _build_entry(tmp, runner=None, inner=inner,
                             workflow_callable=_echo_workflow)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "runner" in str(exc_info.value)

    def test_missing_inner_trigger_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                _build_entry(tmp, runner=runner, inner=None,
                             workflow_callable=_echo_workflow)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "inner_trigger" in str(exc_info.value)

    def test_missing_workflow_callable_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            inner = _build_event_trigger()
            with pytest.raises(ComponentConfigurationError) as exc_info:
                _build_entry(tmp, runner=runner, inner=inner,
                             workflow_callable=None)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "workflow_callable" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3-5. End-to-end + payload handling
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_event_dict_payload_pass_through(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(tmp, runner, inner, _echo_workflow,
                                     on_launch=on_launch)
                await entry.start()

                await inner.fire_event({"kind": "novel"})
                await asyncio.sleep(0.1)

                assert len(launches) == 1
                final = await runner.await_completion(launches[0], timeout=5)
                assert final.status == "completed"
                # Dict event body passed through (plus autonomy metadata):
                assert final.result == {"echoed": {
                    "kind": "novel",
                    "__autonomy_level__": "strict_hitl",
                }}
        asyncio.run(run())

    def test_non_dict_event_body_wrapped(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(tmp, runner, inner, _echo_workflow,
                                     on_launch=on_launch)
                await entry.start()
                await inner.fire_event("a_string_event")
                await asyncio.sleep(0.1)

                final = await runner.await_completion(launches[0], timeout=5)
                assert final.status == "completed"
                # Non-dict event body wrapped under 'event_body':
                payload = final.result["echoed"]
                assert payload["event_body"] == "a_string_event"
                assert payload["__autonomy_level__"] == "strict_hitl"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. payload_factory dotted-path resolution
# ---------------------------------------------------------------------------

class TestPayloadFactory:

    def test_dotted_path_factory_resolved_and_called(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(
                    tmp, runner, inner, _echo_workflow,
                    payload_factory=__name__ + "._factory_for_test",
                    on_launch=on_launch,
                )
                await entry.start()
                await inner.fire_event({"event_id": "abc"})
                await asyncio.sleep(0.1)

                final = await runner.await_completion(launches[0], timeout=5)
                payload = final.result["echoed"]
                assert payload["factory_built"] is True
                assert payload["event_id"] == "abc"
                assert payload["__autonomy_level__"] == "strict_hitl"
        asyncio.run(run())

    def test_unresolvable_factory_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            inner = _build_event_trigger()
            with pytest.raises(ComponentConfigurationError) as exc_info:
                _build_entry(
                    tmp, runner, inner, _echo_workflow,
                    payload_factory="nonexistent.pkg.fn",
                )
            assert "FAIL-FAST" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Forward-compat metadata
# ---------------------------------------------------------------------------

class TestForwardCompatMetadata:

    def test_autonomy_and_cost_template_in_payload(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(
                    tmp, runner, inner, _echo_workflow,
                    autonomy_level="pure_autonomous",
                    cost_envelope_template="weekly_digest_default",
                    on_launch=on_launch,
                )
                await entry.start()
                await inner.fire_event({"kind": "x"})
                await asyncio.sleep(0.1)

                final = await runner.await_completion(launches[0], timeout=5)
                payload = final.result["echoed"]
                assert payload["__autonomy_level__"] == "pure_autonomous"
                assert payload["__cost_envelope_template__"] == "weekly_digest_default"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 8. Multiple events
# ---------------------------------------------------------------------------

class TestMultipleEvents:

    def test_distinct_task_ids_per_event(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_event_trigger()
                launches = []
                async def on_launch(h):
                    launches.append(h.task_id)
                entry = _build_entry(tmp, runner, inner, _echo_workflow,
                                     on_launch=on_launch)
                await entry.start()
                await inner.fire_event({"i": 1})
                await asyncio.sleep(0.05)
                await inner.fire_event({"i": 2})
                await asyncio.sleep(0.05)
                await inner.fire_event({"i": 3})
                await asyncio.sleep(0.2)

                assert len(launches) == 3
                assert len(set(launches)) == 3, "Task IDs collided"
                for tid in launches:
                    h = await runner.await_completion(tid, timeout=5)
                    assert h.status == "completed"
        asyncio.run(run())
