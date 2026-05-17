"""Unit tests for G110 RecursiveSubworkflowStep.

The interesting testable surface:
  * Depth threading: envelope's _recursion_depth increments per call
  * Depth cap: terminal envelope emitted at max_recursion_depth
  * Path resolution: FAIL-FAST on missing inner workflow path
  * Lazy inner-workflow load: self-reference doesn't crash at init

End-to-end recursion against a real recursive workflow YAML would
require a non-trivial domain example; covered separately when an
HD-RSS-as-YAML port lands. These unit tests pin the primitive's
mechanics in isolation.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps.recursive_subworkflow_step import (
    RecursiveSubworkflowStep,
)


def _make_minimal_inner_workflow_yaml() -> str:
    """Build a trivial workflow YAML on disk that the step can point
    to. Doesn't need to be a real recursive workflow — the tests
    mock the workflow load."""
    # We don't actually need this to be runnable for most tests —
    # path resolution just checks the file exists. The mocks intercept
    # the actual Workflow.from_config call.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("name: dummy\nsteps: {}\nlinks: {}\n")
        return f.name


def _build_step(*, max_depth: int = 3, inner_path: str | None = None) -> RecursiveSubworkflowStep:
    """Build a RecursiveSubworkflowStep with sensible defaults."""
    if inner_path is None:
        inner_path = _make_minimal_inner_workflow_yaml()
    cfg = {
        "name": "test_recursive",
        "description": "Test",
        "inner_workflow_path": inner_path,
        "max_recursion_depth": max_depth,
        "input_data_unit_name": "iter_input",
        "output_data_unit_name": "iter_output",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        config_path = f.name
    return RecursiveSubworkflowStep.from_config(config_path)


class TestPathResolution:
    """The Step resolves the inner workflow path at INIT time (FAIL-FAST
    on missing) but doesn't LOAD the workflow until process()."""

    def test_missing_inner_path_raises_at_init(self):
        with pytest.raises(ComponentConfigurationError, match="does not exist"):
            _build_step(inner_path="/absolutely/does/not/exist.yml")

    def test_existing_inner_path_loads_at_init(self):
        step = _build_step()
        assert step._inner_workflow_path.is_file()

    def test_inner_workflow_not_loaded_at_init(self):
        """The lazy-load is the KEY feature — enables self-reference.
        Verify Workflow.from_config is NOT called during init."""
        # The Step's _init_from_config only resolves the path, not loads.
        # If init triggered a load, a self-referencing path would recurse
        # at config time. We test this via the absence of an _inner_workflow
        # attribute (which SubworkflowStep DOES set at init).
        step = _build_step()
        assert not hasattr(step, "_inner_workflow"), (
            "RecursiveSubworkflowStep should NOT pre-load the inner workflow "
            "at init time — that's what makes self-reference safe."
        )


class TestDepthCap:
    """When incoming depth >= max_recursion_depth, emit terminal envelope
    instead of recursing. This is the bounded-recursion guarantee."""

    def test_depth_cap_emits_terminal_envelope(self):
        step = _build_step(max_depth=2)
        # Input at depth 2 should hit the cap.
        result = asyncio.run(step.process({
            "code_spec": "x",
            "_recursion_depth": 2,
        }))
        assert result["_recursion_terminated"] is True
        # Other input fields preserved.
        assert result["code_spec"] == "x"
        # Depth NOT incremented in terminal output.
        assert result["_recursion_depth"] == 2

    def test_depth_zero_recurses_into_inner(self):
        """Below cap, the step should load + run the inner workflow.
        We mock Workflow.from_config to verify it's called per process()
        invocation."""
        step = _build_step(max_depth=3)
        mock_workflow = mock.MagicMock()

        async def fake_run(*args, **kwargs):
            return {"iter_output": {"result": "recursed"}}

        mock_workflow.run = fake_run

        with mock.patch(
            "nanobrain.library.steps.recursive_subworkflow_step.Workflow.from_config",
            return_value=mock_workflow,
        ) as mock_from_config:
            result = asyncio.run(step.process({"code_spec": "x", "_recursion_depth": 0}))

        # Workflow.from_config WAS called (lazy load fires on process()).
        mock_from_config.assert_called_once()
        # Result is the inner workflow's output unit value.
        assert result == {"result": "recursed"}

    def test_no_explicit_depth_treated_as_zero(self):
        """A first call without _recursion_depth should treat depth=0."""
        step = _build_step(max_depth=3)
        mock_workflow = mock.MagicMock()

        captured_call: dict = {}

        async def fake_run(payload, **kwargs):
            captured_call["payload"] = payload
            return {"iter_output": {"result": "ok"}}

        mock_workflow.run = fake_run

        with mock.patch(
            "nanobrain.library.steps.recursive_subworkflow_step.Workflow.from_config",
            return_value=mock_workflow,
        ):
            asyncio.run(step.process({"code_spec": "x"}))

        # The recursive call's envelope should have depth=1 (i.e. 0+1).
        envelope = captured_call["payload"]["iter_input"]
        assert envelope["_recursion_depth"] == 1


class TestEnvelopeFieldNames:
    """The depth + terminal field names are configurable. Verify
    that customizing them works end-to-end."""

    def test_custom_depth_field_name(self):
        inner = _make_minimal_inner_workflow_yaml()
        cfg = {
            "name": "t",
            "inner_workflow_path": inner,
            "max_recursion_depth": 1,
            "depth_field_name": "level",
            "terminal_marker_field": "stopped",
            "input_data_unit_name": "in",
            "output_data_unit_name": "out",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            step = RecursiveSubworkflowStep.from_config(f.name)

        # At depth 1, hits cap (max_recursion_depth=1).
        result = asyncio.run(step.process({"x": 1, "level": 1}))
        assert result["stopped"] is True
        assert result["level"] == 1


class TestInnerWorkflowRunErrors:
    def test_inner_returns_none_raises(self):
        step = _build_step(max_depth=3)
        mock_workflow = mock.MagicMock()

        async def fake_run(*args, **kwargs):
            return None

        mock_workflow.run = fake_run

        with mock.patch(
            "nanobrain.library.steps.recursive_subworkflow_step.Workflow.from_config",
            return_value=mock_workflow,
        ):
            with pytest.raises(RuntimeError, match="returned None"):
                asyncio.run(step.process({"code_spec": "x", "_recursion_depth": 0}))

    def test_inner_missing_output_unit_raises(self):
        step = _build_step(max_depth=3)
        mock_workflow = mock.MagicMock()

        async def fake_run(*args, **kwargs):
            return {"wrong_output_name": {"result": "x"}}  # not iter_output

        mock_workflow.run = fake_run

        with mock.patch(
            "nanobrain.library.steps.recursive_subworkflow_step.Workflow.from_config",
            return_value=mock_workflow,
        ):
            with pytest.raises(RuntimeError, match="did not produce output"):
                asyncio.run(step.process({"code_spec": "x"}))


class TestNonDictResultWrap:
    """When the inner workflow's output is a scalar (rare but valid),
    wrap it in {value: ...} so downstream consumers always get a dict."""

    def test_scalar_result_wrapped(self):
        step = _build_step()
        mock_workflow = mock.MagicMock()

        async def fake_run(*args, **kwargs):
            return {"iter_output": 42}  # scalar, not dict

        mock_workflow.run = fake_run

        with mock.patch(
            "nanobrain.library.steps.recursive_subworkflow_step.Workflow.from_config",
            return_value=mock_workflow,
        ):
            result = asyncio.run(step.process({"code_spec": "x"}))

        assert result == {"value": 42}
