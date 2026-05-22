"""G125 (2026-05-18) — ``Workflow.process()`` tags listener tasks with the
workflow's ``_active_workflow_id`` ContextVar.

ROOT-CAUSE NARRATIVE
====================

The earlier G124 chain (commit fb3c89e) added a ``settle_ms`` safe-floor
of 500ms because the manual ``wf.process() + wf.wait_for_cascade()``
pattern empirically returned EMPTY outputs against real-LLM latency.
The G124 fix was a band-aid: it made drain wait long enough that the
cascade USUALLY finished anyway. The underlying race was untouched.

The actual root cause (diagnosed 2026-05-18, this commit fixes it):

* ``Workflow.run()`` (line 2528) sets the G115 ``_active_workflow_id``
  ContextVar via ``_g115_cv.set(self._g115_workflow_id())`` so listener
  tasks born during ``run()`` inherit the workflow's id.
* ``Workflow.process()`` (the standalone shape ``wf.process() +
  wf.wait_for_cascade()``) did NOT set the ContextVar.
* Listener tasks born during ``process()`` therefore lacked the
  ``_nb_workflow_id`` tag (``getattr(task, '_nb_workflow_id', None)``
  returned None / ``'<no attr>'``).
* ``wait_for_cascade()`` calls ``wait_for_all_tasks(workflow_id=self._g115_workflow_id())``.
* The ``_scoped`` filter inside ``wait_for_all_tasks`` excludes tasks
  whose ``_nb_workflow_id`` doesn't match the requested workflow_id.
* Result: ``scoped`` was empty even though listener tasks were
  in-flight. Drain returned True instantly. Output DUs were never
  populated. The reader saw ``None``.

G125 closes the race by making ``process()`` set the ContextVar around
its body, symmetric with ``run()``. After G125, ``settle_ms=50`` works
correctly even for slow cascades, and the G124 safe-floor warning
becomes a defensive heuristic rather than a load-bearing mitigation.

THIS TEST PINS:
===============

1. ``Workflow.process()`` sets ``_active_workflow_id`` to the workflow's
   id during execution.
2. Listener tasks created during ``process()`` carry the
   ``_nb_workflow_id`` tag matching the workflow.
3. ``wait_for_cascade()`` after ``process()`` correctly awaits the
   tagged listener tasks (no false-empty drain).
4. End-to-end: the standalone ``wf.process() + wf.wait_for_cascade(settle_ms=50)``
   pattern produces non-empty output DUs for a workflow with a
   real-latency step body. (Before G125, this returned empty.)
5. The ContextVar is properly reset on return (no leakage to outer
   scope).
"""

# ALLOWED_WAIT_FOR_CASCADE: this test pins the wait_for_cascade
# behavior itself + the race-window fix. Calling it directly is
# the SUT, not a misuse of the primitive.

from __future__ import annotations

import asyncio
import os
import tempfile
import textwrap
from pathlib import Path

import pytest

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.trigger import _active_workflow_id
from nanobrain.core.workflow import Workflow


class _SlowEchoConfig(StepConfig):
    pass


class _SlowEcho(BaseStep):
    """Step whose process() takes 200ms — enough that a settle_ms=50
    drain would return BEFORE the cascade completes if the workflow_id
    tag is wrong (which is the G125 bug)."""

    COMPONENT_TYPE = "test_slow_echo"

    @classmethod
    def _get_config_class(cls):
        return _SlowEchoConfig

    async def process(self, input_data, **kw):
        await asyncio.sleep(0.2)
        return {"echo": input_data}


# Make the class importable from the test file's __main__ namespace
# so the YAML can reference it via ``class: __main__._SlowEcho``.
import __main__ as _m  # noqa: E402

_m._SlowEcho = _SlowEcho


def _build_workflow_yaml(tmp_path: Path) -> Path:
    # Steps must reference their config by FILE PATH, not an inline dict —
    # G121 inline-step-config support was reverted (see
    # ConfigBase._is_inline_config_supported, which now excludes BaseStep).
    echo_yml = tmp_path / "echo_step.yml"
    echo_yml.write_text(textwrap.dedent("""
        name: echo
        input_data_units:
          echo_in: {class: nanobrain.core.data_unit.DataUnitMemory, name: echo_in}
        output_data_units:
          echo_out: {class: nanobrain.core.data_unit.DataUnitMemory, name: echo_out}
        triggers:
          - {class: nanobrain.core.trigger.DataUnitChangeTrigger, data_unit: echo_in}
    """).strip())

    wf_yml = tmp_path / "wf.yml"
    wf_yml.write_text(textwrap.dedent(f"""
        name: g125_test_wf
        config_version: 2
        input_data_units:
          wf_in: {{class: nanobrain.core.data_unit.DataUnitMemory, name: wf_in}}
        output_data_units:
          wf_out: {{class: nanobrain.core.data_unit.DataUnitMemory, name: wf_out}}
        steps:
          echo:
            class: __main__._SlowEcho
            config: {echo_yml}
        links:
          inp: {{class: nanobrain.core.link.DirectLink, config: {{link_type: direct, source: wf_in, target: echo.echo_in, auto_transfer: true}}}}
          out: {{class: nanobrain.core.link.DirectLink, config: {{link_type: direct, source: echo.echo_out, target: wf_out, auto_transfer: true}}}}
    """).strip())
    return wf_yml


@pytest.mark.asyncio
async def test_g125_process_sets_workflow_id_contextvar():
    """G125 contract #1: process() sets _active_workflow_id to the
    workflow's id during its execution."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = Workflow.from_config(str(_build_workflow_yaml(Path(tmp))))
        await wf.initialize()
        captured = {}

        class _ProbeConfig(StepConfig):
            pass

        class _ProbeStep(BaseStep):
            COMPONENT_TYPE = "g125_probe"

            @classmethod
            def _get_config_class(cls):
                return _ProbeConfig

            async def process(self, input_data, **kw):
                captured["seen_during_process"] = _active_workflow_id.get(None)
                return {"echo": input_data}

        # Replace the echo step's process to probe the contextvar from
        # INSIDE the cascade — at the moment listener tasks fire.
        echo_step = wf.child_steps["echo"]
        echo_step.process = _ProbeStep.process.__get__(echo_step)

        await wf.process({"echo_in": {"x": 1}})
        await wf.wait_for_cascade(timeout=5.0, settle_ms=50)

        # The probe should have seen the workflow's id while running.
        assert captured["seen_during_process"] == wf._g115_workflow_id(), (
            f"G125 regression: workflow_id not propagated to listener-spawned "
            f"step body. Got: {captured['seen_during_process']!r}; "
            f"expected: {wf._g115_workflow_id()!r}"
        )


@pytest.mark.asyncio
async def test_g125_drain_after_process_returns_populated_output():
    """G125 contract #4: the standalone ``wf.process() + wf.wait_for_cascade(settle_ms=50)``
    pattern produces a non-empty output. Before G125, this returned
    empty because drain filtered out the (untagged) listener tasks."""
    os.environ["NANOBRAIN_ALLOW_SHORT_SETTLE_MS"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            wf = Workflow.from_config(str(_build_workflow_yaml(Path(tmp))))
            await wf.initialize()
            await wf.process({"echo_in": {"msg": "hello"}})
            drained = await wf.wait_for_cascade(timeout=5.0, settle_ms=50)
            assert drained is True, "cascade reported not drained"
            echo_out = await wf.child_steps["echo"].step_output_data_units["echo_out"].get()
            assert echo_out is not None, (
                "G125 regression: echo_out is None after wait_for_cascade "
                "returned drained=True. This is the pre-G125 silent-failure "
                "shape — see test_g125_process_workflow_id_tag.py docstring."
            )
            # The step sees the trigger envelope {"echo_in": {...}} (it
            # doesn't unwrap), so the echo wraps the envelope. Either
            # nested shape proves the input made it through.
            assert "echo" in echo_out and echo_out["echo"] == {"echo_in": {"msg": "hello"}}, (
                f"echo_out content unexpected: {echo_out!r}"
            )
    finally:
        os.environ.pop("NANOBRAIN_ALLOW_SHORT_SETTLE_MS", None)


@pytest.mark.asyncio
async def test_g125_contextvar_restored_on_return():
    """G125 contract #5: the ContextVar is reset after process() returns,
    so an outer scope (e.g., nested workflow runs) sees its prior value."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = Workflow.from_config(str(_build_workflow_yaml(Path(tmp))))
        await wf.initialize()

        # Simulate an outer scope by pre-setting the contextvar.
        outer_token = _active_workflow_id.set("outer-scope-id")
        try:
            await wf.process({"echo_in": {"x": 1}})
            await wf.wait_for_cascade(timeout=5.0, settle_ms=200)

            # The contextvar should be back to "outer-scope-id" after
            # process() + drain — confirming the reset-token semantics.
            assert _active_workflow_id.get(None) == "outer-scope-id", (
                f"G125 regression: process() leaked workflow_id into outer "
                f"scope. Got: {_active_workflow_id.get(None)!r}; "
                f"expected: 'outer-scope-id'"
            )
        finally:
            _active_workflow_id.reset(outer_token)


@pytest.mark.asyncio
async def test_g125_process_does_not_break_run_path():
    """Regression guard: ``Workflow.run()`` (which also sets the ContextVar)
    still works after G125. Two callers setting + resetting the same
    ContextVar must compose cleanly via the reset-token contract."""
    with tempfile.TemporaryDirectory() as tmp:
        wf = Workflow.from_config(str(_build_workflow_yaml(Path(tmp))))
        outputs = await wf.run(
            {"echo_in": {"x": 42}},
            timeout=5.0,
            settle_ms=500,
            raise_on_cascade_timeout=False,
        )
        assert isinstance(outputs, dict)
        assert outputs.get("status") == "completed"
