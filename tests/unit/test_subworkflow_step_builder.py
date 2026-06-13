"""Regression tests for ``SubworkflowStep`` inner_workflow_builder seam.

Task E2-F1. The motivating gap: a workflow that exists ONLY as a
programmatic lightweight-``WorkflowBuilder`` builder (apecx's
``viral_conserved_sites``) has no static YAML on disk, yet
``SubworkflowStep`` historically required an ``inner_workflow_path``.
The ``inner_workflow_builder`` field closes that gap: a dotted-path to
a NO-ARG callable that returns a fully-loaded ``Workflow`` instance.

Test layers (per the G99 cycle-bearing rule — load/smoke is necessary
but NOT sufficient; assert on a concrete OUTPUT VALUE driven by a real
``Workflow.run()``):

  1. Config validation FAIL-FAST surfaces (both-set / neither-set /
     bad dotted path / non-callable / wrong return type). Cheap, no run.
  2. **End-to-end**: an OUTER workflow whose single stage is a
     ``SubworkflowStep`` built from a module-level builder callable,
     driven via ``Workflow.run()``. Asserts the inner workflow's output
     VALUE flows all the way through to the outer workflow-level output
     data unit. The inner workflow is REAL (built + loaded via the
     lightweight builder, real DataUnitChangeTrigger cascade, real
     DirectLinks) — only the domain step is a trivial deterministic
     marker, which is the framework-test analogue the task permits.

G117 is honored structurally: the outer step's own input DU
(``outer_in``) is DELIBERATELY named differently from the inner
workflow's first-step input DU (``inner_in``), so the trigger-envelope
unwrap + re-wrap in ``SubworkflowStep._route_input_to_first_step_du``
is actually exercised (not bypassed by accidental name collision).
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import Workflow
from nanobrain.lightweight.workflow_builder import WorkflowBuilder
from nanobrain.library.steps.subworkflow_step import SubworkflowStep


_DU = "nanobrain.core.data_unit.DataUnitMemory"
_TRIGGER = "nanobrain.core.trigger.DataUnitChangeTrigger"


# ---------------------------------------------------------------------------
# Real inner-workflow domain step + builder (module-level so the dotted-path
# resolver can import them).
# ---------------------------------------------------------------------------


class _InnerMarkStep(BaseStep):
    """Deterministic inner-workflow step: stamps a marker on whatever it
    receives. Single output DU → the framework's single-output fallback
    writes the whole return dict to ``inner_out``."""

    COMPONENT_TYPE = "test_inner_mark_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        return {"marked": True, "seen": input_data}


def _build_inner_test_workflow() -> Workflow:
    """NO-ARG builder: build + load a real inner workflow via the
    lightweight WorkflowBuilder. This is the exact shape apecx's
    ``build_viral_conserved_sites_workflow`` has — a no-arg callable that
    returns ``builder.load()`` (a Workflow), with no YAML on disk."""
    b = WorkflowBuilder("inner_builder_wf", "real inner workflow built programmatically")
    b.add_input("inner_wf_in", "DataUnitMemory")
    b.add_output("inner_wf_out", "DataUnitMemory")
    b.add_step(
        "transform",
        f"{__name__}._InnerMarkStep",
        input_data_units={"inner_in": {"class": _DU, "name": "inner_in"}},
        output_data_units={"inner_out": {"class": _DU, "name": "inner_out"}},
        triggers=[{"class": _TRIGGER, "data_unit": "inner_in"}],
    )
    b.add_link("inner_wf_in", "transform.inner_in", link_type="direct")
    b.add_link("transform.inner_out", "inner_wf_out", link_type="direct")
    return b.load()


# A module-level non-callable, for the "resolved to non-callable" FAIL-FAST.
_INNER_WORKFLOW_INSTANCE = "i am a string, not a callable"


def _builder_returns_wrong_type():
    """A 'builder' that returns the WRONG type (a dict, not a Workflow)."""
    return {"not": "a workflow"}


# ---------------------------------------------------------------------------
# 1. Config validation FAIL-FAST surfaces.
# ---------------------------------------------------------------------------


def _write_step_yaml(tmp_path, body: str, name: str = "step") -> str:
    p = tmp_path / f"{name}.yml"
    p.write_text(body)
    return str(p)


def test_both_path_and_builder_set_fails_fast(tmp_path):
    """Declaring both inner_workflow_path and inner_workflow_builder is
    ambiguous → FAIL-FAST at load (not a silent pick-one)."""
    inner = tmp_path / "inner.yml"
    inner.write_text("name: x\nconfig_version: 2\nsteps: {}\nlinks: {}\n")
    yml = _write_step_yaml(
        tmp_path,
        f"name: both_step\n"
        f"inner_workflow_path: {inner}\n"
        f"inner_workflow_builder: {__name__}._build_inner_test_workflow\n",
    )
    with pytest.raises(ComponentConfigurationError, match="mutually exclusive"):
        SubworkflowStep.from_config(yml)


def test_neither_path_nor_builder_fails_fast(tmp_path):
    yml = _write_step_yaml(tmp_path, "name: neither_step\n")
    with pytest.raises(
        ComponentConfigurationError,
        match="inner_workflow_path OR an inner_workflow_builder",
    ):
        SubworkflowStep.from_config(yml)


def test_builder_not_dotted_path_fails_fast(tmp_path):
    yml = _write_step_yaml(
        tmp_path,
        "name: bad_spec_step\ninner_workflow_builder: notdotted\n",
    )
    with pytest.raises(ComponentConfigurationError, match="dotted-path string"):
        SubworkflowStep.from_config(yml)


def test_builder_unimportable_module_fails_fast(tmp_path):
    yml = _write_step_yaml(
        tmp_path,
        "name: bad_mod_step\n"
        "inner_workflow_builder: nanobrain.no_such_module.build\n",
    )
    with pytest.raises(ComponentConfigurationError, match="not importable"):
        SubworkflowStep.from_config(yml)


def test_builder_missing_attr_fails_fast(tmp_path):
    yml = _write_step_yaml(
        tmp_path,
        f"name: bad_attr_step\n"
        f"inner_workflow_builder: {__name__}.no_such_builder_func\n",
    )
    with pytest.raises(ComponentConfigurationError, match="not\\s+found"):
        SubworkflowStep.from_config(yml)


def test_builder_resolves_to_non_callable_fails_fast(tmp_path):
    yml = _write_step_yaml(
        tmp_path,
        f"name: noncallable_step\n"
        f"inner_workflow_builder: {__name__}._INNER_WORKFLOW_INSTANCE\n",
    )
    with pytest.raises(ComponentConfigurationError, match="not callable"):
        SubworkflowStep.from_config(yml)


def test_builder_resolves_to_class_fails_fast(tmp_path):
    yml = _write_step_yaml(
        tmp_path,
        f"name: class_step\n"
        f"inner_workflow_builder: {__name__}._InnerMarkStep\n",
    )
    with pytest.raises(ComponentConfigurationError, match="resolved to a "):
        SubworkflowStep.from_config(yml)


def test_builder_returns_wrong_type_fails_fast(tmp_path):
    yml = _write_step_yaml(
        tmp_path,
        f"name: wrong_type_step\n"
        f"inner_workflow_builder: {__name__}._builder_returns_wrong_type\n",
    )
    with pytest.raises(ComponentConfigurationError, match="expected a Workflow"):
        SubworkflowStep.from_config(yml)


def test_builder_loads_cleanly_and_caches_workflow(tmp_path):
    """Happy path at load time: the builder is invoked once, the
    resulting Workflow is cached, and inner_workflow_path is None."""
    yml = _write_step_yaml(
        tmp_path,
        f"name: ok_builder_step\n"
        f"inner_workflow_builder: {__name__}._build_inner_test_workflow\n",
    )
    step = SubworkflowStep.from_config(yml)
    assert isinstance(step.inner_workflow, Workflow)
    assert step.inner_workflow.name == "inner_builder_wf"
    # Path is None for the builder branch — the inner came from a callable.
    assert step.inner_workflow_path is None


# ---------------------------------------------------------------------------
# 2. End-to-end: OUTER Workflow.run() drives a builder-sourced inner workflow.
# ---------------------------------------------------------------------------


def _build_outer_workflow() -> Workflow:
    """OUTER workflow: one SubworkflowStep stage whose inner workflow is
    sourced from the module-level builder callable. G117: the step's own
    input DU (outer_in) differs from the inner's first-step input DU
    (inner_in)."""
    b = WorkflowBuilder("outer_nesting_wf", "outer workflow nesting a builder workflow")
    b.add_input("wf_in", "DataUnitMemory")
    b.add_output("wf_out", "DataUnitMemory")
    b.add_step(
        "nested",
        "nanobrain.library.steps.subworkflow_step.SubworkflowStep",
        inner_workflow_builder=f"{__name__}._build_inner_test_workflow",
        input_data_units={"outer_in": {"class": _DU, "name": "outer_in"}},
        output_data_units={"outer_out": {"class": _DU, "name": "outer_out"}},
        triggers=[{"class": _TRIGGER, "data_unit": "outer_in"}],
    )
    b.add_link("wf_in", "nested.outer_in", link_type="direct")
    b.add_link("nested.outer_out", "wf_out", link_type="direct")
    return b.load()


def test_outer_run_flows_inner_builder_output_through():
    """The OUTER cascade must drive the builder-sourced inner workflow
    and propagate the inner output VALUE to the outer workflow-level
    output. Asserting on the concrete value (not status) per G99."""
    wf = _build_outer_workflow()

    async def _run():
        return await wf.run(
            {"wf_in": {"q": "hello"}},
            timeout=30.0,
            settle_ms=500,
            raise_on_cascade_timeout=False,
        )

    out = asyncio.run(_run())
    assert isinstance(out, dict)
    assert out.get("status") == "completed", f"outer cascade did not complete: {out}"

    wf_out = out.get("wf_out")
    assert isinstance(wf_out, dict), (
        f"inner builder output did not propagate to the outer workflow "
        f"output data unit; got {wf_out!r} (full run result: {out})"
    )
    # The inner _InnerMarkStep stamps marked=True; that VALUE must survive
    # the full nested cascade (inner run → SubworkflowStep collect →
    # outer output DU → outer workflow-level output).
    assert wf_out.get("marked") is True, (
        f"inner workflow marker lost in transit; wf_out={wf_out!r}"
    )
