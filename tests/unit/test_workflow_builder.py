"""Tests for the lightweight ``WorkflowBuilder``.

The lightweight builder is the alternative-to-YAML programmatic path
for composing workflows. Per the workspace policy, "explore multiple
legit ways of creating workflows including lightweight nanobrain" —
this test suite is the lightweight side's regression seam.

Coverage:
1. Builder constructs with config_version: 2 and an empty triggers list.
2. The legacy dead ``version`` field is GONE (silent-confusion shape).
3. add_link supports each link_type via the framework-known map
   (independent of the discovery layer's YAML coverage).
4. add_link FAIL-FAST surfaces:
   - unknown link_type
   - conditional without condition
   - transform without transform_function
5. add_trigger supports each trigger_type, both at workflow level
   and on a named step.
6. add_trigger FAIL-FAST when step_name is unknown.
7. _resolve_class_path falls back to discovery for non-framework names.
8. End-to-end: load() consumes the generated dict via
   Workflow.from_config and the v2 mutators stamp auto_transfer +
   gate_semantics where appropriate.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from nanobrain.core.step import BaseStep, StepConfig
from nanobrain.core.workflow import WorkflowConfig
from nanobrain.lightweight import WorkflowBuilder
from nanobrain.lightweight.workflow_builder import (
    _FRAMEWORK_LINK_CLASS_PATHS,
    _FRAMEWORK_TRIGGER_CLASS_PATHS,
)


class _EchoStep(BaseStep):
    """Module-level step so ``from_config`` can resolve it by dotted path
    (``<this module>._EchoStep``) during the end-to-end load+run test."""

    COMPONENT_TYPE = "test_builder_echo"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kw):
        return {"echo_out": f"echoed:{input_data}"}


# ---------------------------------------------------------------------------
# 1-2. Initial config shape
# ---------------------------------------------------------------------------

class TestInitialShape:

    def test_config_version_is_2(self):
        b = WorkflowBuilder("t")
        cfg = b.get_config()
        assert cfg["config_version"] == 2

    def test_triggers_list_initialized(self):
        b = WorkflowBuilder("t")
        assert b.get_config()["triggers"] == []

    def test_dead_version_field_absent(self):
        """Pre-G21-Step-5 chain, the builder set ``version: '2.0'`` —
        the framework loader reads ``config_version``, NOT ``version``,
        so the field was DEAD and a silent-confusion shape. The
        hardening removed it; this test enforces it stays gone."""
        b = WorkflowBuilder("t")
        assert "version" not in b.get_config(), \
            "dead 'version' field reintroduced; framework reads " \
            "'config_version', not 'version'"


# ---------------------------------------------------------------------------
# 3. add_link with framework-known classes
# ---------------------------------------------------------------------------

class TestAddLink:

    def test_direct_link(self):
        b = WorkflowBuilder("t")
        b.add_link("a.x", "b.x", link_type="direct")
        link = b.get_config()["links"]["link_0"]
        # Top-level keeps name + class (the loader's `class:` resolves
        # the subclass). Source / target / link_type / etc. live in the
        # nested `config:` block — same shape as hand-authored YAML.
        # Pre-2026-05-15 the builder emitted these at the TOP level
        # which loaded cleanly through WorkflowConfig but was silently
        # dropped by LinkBase.from_config during graph construction.
        assert link["class"] == "nanobrain.core.link.DirectLink"
        assert link["config"]["link_type"] == "direct"
        assert link["config"]["source"] == "a.x"
        assert link["config"]["target"] == "b.x"

    def test_conditional_link(self):
        b = WorkflowBuilder("t")
        b.add_link(
            "a.x", "b.x", link_type="conditional",
            condition={"op": "exists", "field": "val"},
        )
        link = b.get_config()["links"]["link_0"]
        assert link["class"] == "nanobrain.core.link.ConditionalLink"
        assert link["config"]["condition"] == {"op": "exists", "field": "val"}

    def test_transform_link(self):
        b = WorkflowBuilder("t")
        b.add_link(
            "a.x", "b.x", link_type="transform",
            transform_function="json.dumps",
        )
        link = b.get_config()["links"]["link_0"]
        assert link["class"] == "nanobrain.core.link.TransformLink"
        assert link["config"]["transform_function"] == "json.dumps"

    def test_file_link(self):
        b = WorkflowBuilder("t")
        b.add_link("a.x", "b.x", link_type="file")
        assert b.get_config()["links"]["link_0"]["class"] == \
            "nanobrain.core.link.FileLink"

    def test_per_link_gate_semantics_override(self):
        b = WorkflowBuilder("t")
        b.add_link(
            "a.x", "b.x", link_type="conditional",
            condition={"op": "exists", "field": "v"},
            gate_semantics="gate_to_bottom",
        )
        link = b.get_config()["links"]["link_0"]
        assert link["config"]["gate_semantics"] == "gate_to_bottom"

    def test_explicit_link_name(self):
        b = WorkflowBuilder("t")
        b.add_link("a.x", "b.x", link_type="direct", link_name="my_link")
        assert "my_link" in b.get_config()["links"]


# ---------------------------------------------------------------------------
# 4. add_link FAIL-FAST surfaces
# ---------------------------------------------------------------------------

class TestAddLinkFailures:

    def test_unknown_link_type_fails_fast(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError) as exc_info:
            b.add_link("a", "b", link_type="ghost")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "ghost" in str(exc_info.value)

    def test_conditional_without_condition_fails_fast(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError) as exc_info:
            b.add_link("a", "b", link_type="conditional")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "condition" in str(exc_info.value)

    def test_transform_without_function_fails_fast(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError) as exc_info:
            b.add_link("a", "b", link_type="transform")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "transform_function" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. add_trigger
# ---------------------------------------------------------------------------

class TestAddTrigger:

    def test_workflow_level_timer_trigger(self):
        b = WorkflowBuilder("t")
        b.add_trigger(trigger_type="timer", timer_interval_ms=1000)
        trigs = b.get_config()["triggers"]
        assert len(trigs) == 1
        assert trigs[0]["class"] == "nanobrain.core.trigger.TimerTrigger"
        assert trigs[0]["timer_interval_ms"] == 1000

    def test_event_trigger_with_filter(self):
        b = WorkflowBuilder("t")
        b.add_trigger(
            trigger_type="event",
            event_filter={"op": "eq", "field": "kind", "value": "x"},
        )
        trig = b.get_config()["triggers"][0]
        assert trig["class"] == "nanobrain.core.trigger.EventTrigger"
        assert trig["event_filter"] == {"op": "eq", "field": "kind", "value": "x"}

    def test_step_level_trigger_attachment(self):
        b = WorkflowBuilder("t")
        # Need a step first
        b.workflow_config["steps"]["my_step"] = {
            "name": "my_step", "class": "demos.x.MyStep",
        }
        b.add_trigger(
            step_name="my_step", trigger_type="all_data_received",
        )
        cfg = b.get_config()
        assert cfg["triggers"] == []  # workflow-level still empty
        step_triggers = cfg["steps"]["my_step"]["triggers"]
        assert len(step_triggers) == 1
        assert step_triggers[0]["class"] == \
            "nanobrain.core.trigger.AllDataReceivedTrigger"

    def test_trigger_naming_uses_global_index(self):
        b = WorkflowBuilder("t")
        b.workflow_config["steps"]["s1"] = {"name": "s1", "class": "x"}
        b.add_trigger(trigger_type="timer", timer_interval_ms=1000)
        b.add_trigger(step_name="s1", trigger_type="data_updated")
        b.add_trigger(trigger_type="manual")
        # Names must be distinct across workflow + step buckets
        names = [
            b.get_config()["triggers"][0]["name"],
            b.get_config()["steps"]["s1"]["triggers"][0]["name"],
            b.get_config()["triggers"][1]["name"],
        ]
        assert len(set(names)) == 3, f"trigger names collided: {names}"


# ---------------------------------------------------------------------------
# 6. add_trigger FAIL-FAST surfaces
# ---------------------------------------------------------------------------

class TestAddTriggerFailures:

    def test_unknown_trigger_type_fails_fast(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError) as exc_info:
            b.add_trigger(trigger_type="explode")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "explode" in str(exc_info.value)

    def test_unknown_step_fails_fast(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError) as exc_info:
            b.add_trigger(step_name="ghost", trigger_type="data_updated")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "ghost" in str(exc_info.value)


# ---------------------------------------------------------------------------
# add_step dotted-path fallback (framework expansion 2026-05-12)
# ---------------------------------------------------------------------------


class TestAddStepDottedPathFallback:
    """Pins the dotted-path fallback in ``add_step``.

    The builder previously rejected component_class strings that
    weren't in the YAML-scanned discovery set. That blocked
    programmatic workflow construction with new custom step classes
    until at least one example YAML referenced them. The fallback
    accepts any string containing a ``.`` as a fully-qualified
    dotted import path; resolution happens at Workflow.from_config
    time, not at add_step time.
    """

    def test_dotted_path_accepted_as_fully_qualified_class(self):
        b = WorkflowBuilder("dotted_path_test")
        b.add_step(
            "my_custom_step",
            "my.package.steps.MyCustomStep",
        )
        steps = b.get_config()["steps"]
        assert steps["my_custom_step"]["class"] == "my.package.steps.MyCustomStep"

    def test_short_name_unknown_AND_no_dot_still_fails_fast(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError, match="Unknown component"):
            b.add_step("x", "DefinitelyNotARealStep")

    def test_discovered_short_name_takes_precedence_over_dotted_fallback(self):
        """If the short name IS in discovery, use it. Don't accidentally
        treat 'Foo.Bar' as a dotted path when 'Foo.Bar' happens to be
        a discovered short name (no real-world clash; pin it anyway)."""
        b = WorkflowBuilder("t")
        # Pick any discovered step class for the pin.
        discovered = list(b.discovered_classes.keys())
        if not discovered:
            pytest.skip("no discovered classes available for this pin")
        name = discovered[0]
        expected_path = b.discovered_classes[name]["class_path"]
        b.add_step("via_discovery", name)
        assert b.get_config()["steps"]["via_discovery"]["class"] == expected_path


# ---------------------------------------------------------------------------
# 7. _resolve_class_path discovery fallback
# ---------------------------------------------------------------------------

class TestClassPathResolution:

    def test_framework_map_first(self):
        """Framework-known names resolve via the static map even if
        discovery happens to also find them."""
        b = WorkflowBuilder("t")
        path = b._resolve_class_path(
            "DirectLink", _FRAMEWORK_LINK_CLASS_PATHS, "link",
        )
        assert path == "nanobrain.core.link.DirectLink"

    def test_unknown_class_fails_fast_with_both_lists(self):
        b = WorkflowBuilder("t")
        with pytest.raises(ValueError) as exc_info:
            b._resolve_class_path(
                "MyCustomLink", _FRAMEWORK_LINK_CLASS_PATHS, "link",
            )
        msg = str(exc_info.value)
        assert "FAIL-FAST" in msg
        assert "framework-known" in msg
        assert "discovered" in msg


# ---------------------------------------------------------------------------
# 8. End-to-end: get_config dict → WorkflowConfig validation
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_dict_passes_workflowconfig_v2_mutators(self):
        """The dict the builder generates feeds Workflow.from_config and
        comes out with v2 auto_transfer + gate_semantics injected. We
        validate at the WorkflowConfig layer (cheaper than constructing
        a full Workflow with executor + steps + ...)."""
        b = WorkflowBuilder("t")
        b.add_link("a.x", "b.x", link_type="direct")
        b.add_link(
            "c.x", "d.x", link_type="conditional",
            condition={"op": "exists", "field": "v"},
        )
        # workflow-level gate_semantics is propagated into the
        # ConditionalLink:
        cfg = b.get_config()
        cfg["gate_semantics"] = "gate_to_bottom"

        WorkflowConfig._allow_direct_instantiation = True
        try:
            wcfg = WorkflowConfig(**cfg)
        finally:
            WorkflowConfig._allow_direct_instantiation = False

        # As of 2026-05-15 the builder emits NESTED-shape link entries
        # ({name, class, config: {...}}). The G7 auto_transfer and G10
        # gate_semantics mutators handle both shapes and deposit their
        # values inside ``config`` for nested entries.
        # G7 Step 3: auto_transfer-True injected on every inline link
        assert wcfg.links["link_0"]["config"]["auto_transfer"] is True
        assert wcfg.links["link_1"]["config"]["auto_transfer"] is True
        # G10 Step 2: gate_semantics injected on the ConditionalLink
        # but NOT on the DirectLink (only ConditionalLink reads it).
        assert wcfg.links["link_1"]["config"]["gate_semantics"] == "gate_to_bottom"
        assert "gate_semantics" not in wcfg.links["link_0"]["config"]

    def test_save_and_reload(self):
        """The dict round-trips through json.dump → json.load → builder
        (operator might inspect the JSON or commit it for review)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            b = WorkflowBuilder("t")
            b.add_link("a.x", "b.x")
            saved_path = b.save_config(str(tmp / "wf.json"))
            assert Path(saved_path).is_file()
            reloaded = json.loads(Path(saved_path).read_text())
            assert reloaded["config_version"] == 2
            assert "version" not in reloaded
            assert reloaded["links"]["link_0"]["class"] == \
                "nanobrain.core.link.DirectLink"


# ---------------------------------------------------------------------------
# 7. End-to-end load() + run() — guards the "0 child steps, no_first_step"
#    silent-failure shape (a flat step entry with no `config:` key is
#    silently skipped at load; the workflow runs but does nothing).
# ---------------------------------------------------------------------------
class TestBuilderLoadAndRun:
    def _build(self) -> WorkflowBuilder:
        b = WorkflowBuilder("e2e_wf", "builder load+run regression")
        b.add_input("wf_in", "DataUnitMemory")
        b.add_output("wf_out", "DataUnitMemory")
        b.add_step(
            "echo",
            f"{__name__}._EchoStep",
            input_data_units={
                "echo_in": {
                    "class": "nanobrain.core.data_unit.DataUnitMemory",
                    "name": "echo_in",
                }
            },
            output_data_units={
                "echo_out": {
                    "class": "nanobrain.core.data_unit.DataUnitMemory",
                    "name": "echo_out",
                }
            },
            triggers=[
                {
                    "class": "nanobrain.core.trigger.DataUnitChangeTrigger",
                    "data_unit": "echo_in",
                }
            ],
        )
        b.add_link("wf_in", "echo.echo_in", link_type="direct")
        b.add_link("echo.echo_out", "wf_out", link_type="direct")
        return b

    def test_load_materializes_child_steps(self):
        """The step's fields must survive load() — NOT be dropped because
        the in-memory entry is flat (no `config:` key)."""
        wf = self._build().load()
        assert list(wf.child_steps.keys()) == ["echo"], (
            "builder.load() produced a workflow with no child steps — the "
            "flat step entry was silently skipped (regression)"
        )
        echo = wf.child_steps["echo"]
        assert list(echo.step_input_data_units.keys()) == ["echo_in"]
        assert list(echo.step_output_data_units.keys()) == ["echo_out"]

    def test_run_drives_cascade_end_to_end(self):
        """The composed workflow must actually move data — not return
        {'status': 'no_first_step'}."""
        wf = self._build().load()

        async def _run():
            return await wf.run(
                {"wf_in": "hello"},
                timeout=20.0,
                settle_ms=500,
                raise_on_cascade_timeout=False,
            )

        out = asyncio.run(_run())
        assert isinstance(out, dict)
        assert out.get("status") == "completed", f"cascade did not complete: {out}"
        assert out.get("wf_out") == "echoed:{'echo_in': 'hello'}", (
            f"workflow output did not propagate: {out}"
        )
