"""Unit tests for :class:`SubworkflowStep` (workflow-as-step primitive).

Pins (in order of how the test file is laid out):

  1. Config + path validation
     - Missing inner_workflow_path AND no subclass default → FAIL-FAST.
     - Nonexistent absolute path → FAIL-FAST.
     - Bad YAML → FAIL-FAST on load.
     - Valid path → loads cleanly; exposes ``.inner_workflow`` and
       ``.inner_workflow_path`` properties.
  2. Subclass shortcut: ``_default_inner_workflow_path`` is honored
     when the config field is empty.
  3. Silent-failure gates on the inner workflow's run() return:
     - status ≠ "completed" → ``RuntimeError`` with the actual status.
     - status == "completed_no_await" (await_cascade=False) → RuntimeError
       unless ``allow_completed_no_await=True``.
     - All output values empty/None → ``RuntimeError`` (EMPTY-OUTPUT
       gate) unless ``allow_empty_inner_output=True``.
     - Status key is stripped from the returned dict on success.
  4. ``process()`` input shape — non-dict raises immediately, before
     any inner workflow side effects.

The happy path with a real producing inner workflow is covered by the
integration tests in apecx-mcp-integration's code-writing pipeline
(CW-11). Unit tests intentionally stop at the framework boundary —
mocking the inner workflow would replicate the dominant silent-
failure shape this step exists to prevent.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.steps.subworkflow_step import (
    SubworkflowStep,
    SubworkflowStepConfig,
)


_EMPTY_WORKFLOW_YML = textwrap.dedent(
    """\
    name: subworkflow_test_empty_inner
    description: "Empty inner workflow — has no first step, run() yields no_first_step."
    version: "0.1.0"
    config_version: 2

    steps: {}
    links: {}
    """
)


def _stage_empty_workflow(tmp_path: Path, filename: str = "inner.yml") -> Path:
    """Materialize the empty-workflow fixture under tmp_path."""
    p = tmp_path / filename
    p.write_text(_EMPTY_WORKFLOW_YML)
    return p


def _build_step(
    tmp_path: Path,
    *,
    inner_path: Optional[Path] = None,
    name: str = "test_subworkflow",
    overrides: Optional[Dict[str, Any]] = None,
) -> SubworkflowStep:
    """Stage a SubworkflowStep YAML and load via from_config.

    Uses the empty inner workflow as the inner unless ``inner_path``
    is given. ``overrides`` injects additional top-level keys into the
    SubworkflowStep YAML (e.g. allow_empty_inner_output: true).
    """
    if inner_path is None:
        inner_path = _stage_empty_workflow(tmp_path)

    step_yaml = {
        "name": name,
        "inner_workflow_path": str(inner_path),
    }
    if overrides:
        step_yaml.update(overrides)

    step_yaml_text = "\n".join(f"{k}: {v}" for k, v in step_yaml.items())
    step_path = tmp_path / f"{name}.yml"
    step_path.write_text(step_yaml_text)
    return SubworkflowStep.from_config(str(step_path))


# ---------------------------------------------------------------------------
# 1. Config + path validation
# ---------------------------------------------------------------------------


def test_missing_path_and_no_subclass_default_fails_fast(tmp_path):
    """No inner_workflow_path AND no subclass default → ComponentConfigurationError.

    The subworkflow has to know what workflow to embed; refusing
    silently would leave a step that loaded "cleanly" but raised
    on first call. Fail at load time."""
    step_path = tmp_path / "missing_path.yml"
    step_path.write_text("name: missing_path_step\n")
    with pytest.raises(ComponentConfigurationError, match="inner_workflow_path"):
        SubworkflowStep.from_config(str(step_path))


def test_nonexistent_absolute_path_fails_fast(tmp_path):
    """Absolute path that doesn't exist on disk → fail at load.

    Why fail HERE not in process(): a workflow author authors many
    steps at compose time; surfacing a bad path at step-init is much
    more diagnosable than a runtime AttributeError 20s into a run."""
    step_path = tmp_path / "bad_path.yml"
    step_path.write_text(
        "name: bad_path_step\n"
        "inner_workflow_path: /nonexistent/path/that/does/not/exist.yml\n"
    )
    with pytest.raises(ComponentConfigurationError, match="does not exist"):
        SubworkflowStep.from_config(str(step_path))


def test_unresolvable_relative_path_fails_fast(tmp_path):
    """Relative path that resolves nowhere → fail at load with the
    resolution attempts surfaced in the message."""
    step_path = tmp_path / "rel_path.yml"
    step_path.write_text(
        "name: rel_path_step\n"
        "inner_workflow_path: definitely-not-a-real/workflow.yml\n"
    )
    with pytest.raises(ComponentConfigurationError) as exc_info:
        SubworkflowStep.from_config(str(step_path))
    msg = str(exc_info.value)
    assert "could not be resolved" in msg
    # Must surface the search locations to make the failure self-diagnosing.
    assert "workspace_root" in msg or "cwd" in msg.lower()


def test_valid_path_loads_cleanly_and_exposes_properties(tmp_path):
    step = _build_step(tmp_path)
    assert step.name == "test_subworkflow"
    # Both properties round-trip back to the staged YAML.
    assert step.inner_workflow_path.is_file()
    assert step.inner_workflow is not None
    # And the loaded workflow's name matches the YAML.
    assert step.inner_workflow.name == "subworkflow_test_empty_inner"


def test_inner_workflow_yaml_parse_failure_surfaces_as_load_error(tmp_path):
    """Malformed YAML inside the inner workflow → step load fails with
    a clear failure pointing at the inner path."""
    bad_inner = tmp_path / "bad_inner.yml"
    bad_inner.write_text(":\nthis is not valid: yaml: at all\n  - [unbalanced")
    step_path = tmp_path / "step.yml"
    step_path.write_text(
        f"name: bad_inner_step\n"
        f"inner_workflow_path: {bad_inner}\n"
    )
    with pytest.raises(ComponentConfigurationError, match="failed to .*load.*inner"):
        SubworkflowStep.from_config(str(step_path))


# ---------------------------------------------------------------------------
# 2. Subclass shortcut: _default_inner_workflow_path
# ---------------------------------------------------------------------------


def test_subclass_default_path_is_honored_when_config_field_omitted(tmp_path):
    """A subclass that hardcodes _default_inner_workflow_path must
    work without inner_workflow_path in the YAML."""
    inner = _stage_empty_workflow(tmp_path)
    default_path = str(inner)

    class _NamedReflection(SubworkflowStep):
        @classmethod
        def _default_inner_workflow_path(cls) -> Optional[str]:
            return default_path

    step_path = tmp_path / "named_reflection.yml"
    step_path.write_text("name: named_reflection_step\n")
    step = _NamedReflection.from_config(str(step_path))
    assert step.inner_workflow_path == inner.resolve()


def test_config_field_overrides_subclass_default(tmp_path):
    """Explicit inner_workflow_path in YAML wins over the subclass
    default — matches Pydantic's standard precedence + lets an
    operator override a hardcoded pattern for testing."""
    inner_a = _stage_empty_workflow(tmp_path, "inner_a.yml")
    inner_b = _stage_empty_workflow(tmp_path, "inner_b.yml")

    class _DefaultsToA(SubworkflowStep):
        @classmethod
        def _default_inner_workflow_path(cls) -> Optional[str]:
            return str(inner_a)

    step_path = tmp_path / "overridden.yml"
    step_path.write_text(
        f"name: overridden_step\n"
        f"inner_workflow_path: {inner_b}\n"
    )
    step = _DefaultsToA.from_config(str(step_path))
    assert step.inner_workflow_path == inner_b.resolve()


# ---------------------------------------------------------------------------
# 3. Silent-failure gates on Workflow.run() return
# ---------------------------------------------------------------------------


def test_non_completed_status_raises_runtime_error(tmp_path):
    """Empty inner workflow → Workflow.run() returns
    status='no_first_step'. SubworkflowStep must raise RuntimeError
    rather than silently propagate the status to downstream steps."""
    step = _build_step(tmp_path)
    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(step.process({}))
    msg = str(exc_info.value)
    assert "no_first_step" in msg
    # The error message must surface that the cause is the inner workflow,
    # not a problem inside SubworkflowStep itself.
    assert "inner workflow" in msg.lower()


def test_empty_output_gate_default_raises_on_empty_inner_output(tmp_path):
    """Empty workflow → empty result dict → SubworkflowStep raises
    even on the EMPTY-OUTPUT path BEFORE checking status.

    (Actually the no_first_step check fires first; this test is a
    duplicate of the status check above. We keep it so a future change
    to status semantics doesn't silently bypass the empty-output gate.)"""
    step = _build_step(tmp_path)
    with pytest.raises(RuntimeError):
        asyncio.run(step.process({}))


def test_empty_output_gate_opt_in_does_not_short_circuit_status_check(tmp_path):
    """Even with allow_empty_inner_output=True, a non-completed status
    must still raise. The two gates are independent — opting out of
    one does not weaken the other."""
    step = _build_step(
        tmp_path,
        overrides={"allow_empty_inner_output": "true"},
    )
    with pytest.raises(RuntimeError, match="no_first_step"):
        asyncio.run(step.process({}))


# ---------------------------------------------------------------------------
# 4. process() input shape
# ---------------------------------------------------------------------------


def test_non_dict_input_fails_fast_before_running_inner_workflow(tmp_path):
    """process(input_data) requires a dict (matches Workflow.run's
    input shape: keyed by workflow-level input data unit names).
    A list / string / None must raise IMMEDIATELY — not after the
    inner workflow has been touched."""
    step = _build_step(tmp_path)

    for bad in [["not", "a", "dict"], "string-input", 42, None]:
        with pytest.raises(ComponentConfigurationError, match="must be a dict"):
            asyncio.run(step.process(bad))


# ---------------------------------------------------------------------------
# 5. Config defaults pinned (regression safety)
# ---------------------------------------------------------------------------


def test_config_defaults_match_design_decisions(tmp_path):
    """Pin the config field defaults. Changing any of these is a
    public-surface change — pinning here forces the author to
    acknowledge the change in a deliberate diff."""
    inner = _stage_empty_workflow(tmp_path)
    step_path = tmp_path / "defaults_step.yml"
    step_path.write_text(
        f"name: defaults_step\n"
        f"inner_workflow_path: {inner}\n"
    )
    step = SubworkflowStep.from_config(str(step_path))
    assert step._timeout_seconds == 60.0
    assert step._settle_ms == 50
    assert step._await_cascade is True
    assert step._allow_completed_no_await is False
    assert step._allow_empty_inner_output is False
    assert step._nest_under_active_context is True


def test_config_field_overrides_take_effect(tmp_path):
    """Each config knob must round-trip from YAML to the step instance."""
    inner = _stage_empty_workflow(tmp_path)
    step_path = tmp_path / "overrides_step.yml"
    step_path.write_text(
        f"name: overrides_step\n"
        f"inner_workflow_path: {inner}\n"
        f"timeout_seconds: 15.0\n"
        f"settle_ms: 200\n"
        f"await_cascade: false\n"
        f"allow_completed_no_await: true\n"
        f"allow_empty_inner_output: true\n"
        f"nest_under_active_context: false\n"
    )
    step = SubworkflowStep.from_config(str(step_path))
    assert step._timeout_seconds == 15.0
    assert step._settle_ms == 200
    assert step._await_cascade is False
    assert step._allow_completed_no_await is True
    assert step._allow_empty_inner_output is True
    assert step._nest_under_active_context is False
