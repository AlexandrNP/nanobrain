"""Tests for G10 Step 2 — workflow-level gate_semantics propagation.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G10``.

Coverage:
1. WorkflowConfig.gate_semantics field is optional, defaults to None.
2. When None, no propagation happens.
3. When set, gate_semantics is injected into ConditionalLink entries
   (flat + nested + path-reference behavior).
4. When set, gate_semantics is injected into AllDataReceivedTrigger
   entries in workflow.triggers and steps[*].triggers.
5. Explicit per-component values are NEVER overridden.
6. Non-gate-aware classes (DirectLink, TimerTrigger, etc.) are skipped.
7. gate_to_bottom + publish_empty both round-trip.
8. Path-reference triggers/links are NOT mutated (parity with G7 Step 3).
"""

from __future__ import annotations

import pytest

from nanobrain.core.workflow import (
    WorkflowConfig,
    _link_class_needs_gate_semantics_check,
    _trigger_class_needs_gate_semantics_check,
    _set_inline_default_in_entry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(**kwargs) -> WorkflowConfig:
    WorkflowConfig._allow_direct_instantiation = True
    try:
        return WorkflowConfig(**kwargs)
    finally:
        WorkflowConfig._allow_direct_instantiation = False


# ---------------------------------------------------------------------------
# 1. Field defaults
# ---------------------------------------------------------------------------

class TestFieldDefaults:

    def test_gate_semantics_default_is_none(self):
        cfg = _build(name="t")
        assert cfg.gate_semantics is None

    def test_invalid_value_rejected(self):
        with pytest.raises(Exception):
            _build(name="t", gate_semantics="not_a_real_mode")


# ---------------------------------------------------------------------------
# 2. No-op when None
# ---------------------------------------------------------------------------

class TestNoPropagationWhenNone:

    def test_links_untouched(self):
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "source": "a.x", "target": "b.x", "condition": "true_only",
        }
        cfg = _build(name="t", links={"l": link_dict})
        assert "gate_semantics" not in cfg.links["l"]


# ---------------------------------------------------------------------------
# 3. Link injection
# ---------------------------------------------------------------------------

class TestLinkPropagation:

    def test_flat_inline_conditional_link(self):
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "source": "a.x", "target": "b.x", "condition": "true_only",
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        assert cfg.links["l"]["gate_semantics"] == "gate_to_bottom"

    def test_nested_inline_conditional_link(self):
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "config": {"source": "a.x", "target": "b.x",
                       "condition": "true_only"},
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        assert cfg.links["l"]["config"]["gate_semantics"] == "gate_to_bottom"

    def test_explicit_value_preserved_per_link(self):
        """The non-obvious case: an author wrote gate_semantics:
        publish_empty intentionally on ONE link in a gate_to_bottom
        workflow. setdefault must not override that opt-out."""
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "source": "a.x", "target": "b.x", "condition": "true_only",
            "gate_semantics": "publish_empty",
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        assert cfg.links["l"]["gate_semantics"] == "publish_empty"

    def test_directlink_not_touched(self):
        """DirectLink doesn't read gate_semantics; the propagation must
        skip it even if the workflow declares the workflow-level default."""
        link_dict = {
            "class": "nanobrain.core.link.DirectLink",
            "source": "a.x", "target": "b.x",
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        assert "gate_semantics" not in cfg.links["l"]

    def test_path_reference_rewritten_in_v2(self, tmp_path):
        """G7 Step 4 — path-reference link configs are LOADED and
        REWRITTEN to nested-inline form when config_version >= 2 and
        a workflow-level gate_semantics is set."""
        ext = tmp_path / "some_link.yml"
        ext.write_text(
            "source: a.x\n"
            "target: b.x\n"
            "condition: true_only\n"
        )
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "config": str(ext),
        }
        # Default config_version is now 2 (post-G7 Step 4)
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        assert isinstance(cfg.links["l"]["config"], dict)
        assert cfg.links["l"]["config"]["gate_semantics"] == "gate_to_bottom"

    def test_path_reference_skipped_in_v1(self, tmp_path):
        """v1 still skips path-reference configs (Step 4 is v2-only)."""
        ext = tmp_path / "some_link.yml"
        ext.write_text(
            "source: a.x\n"
            "target: b.x\n"
            "condition: true_only\n"
        )
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "config": str(ext),
        }
        cfg = _build(name="t", config_version=1,
                     gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        # Still a string; not loaded.
        assert cfg.links["l"]["config"] == str(ext)


# ---------------------------------------------------------------------------
# 4. Trigger injection
# ---------------------------------------------------------------------------

class TestTriggerPropagation:

    def test_workflow_level_triggers_list(self):
        trig = {
            "class": "nanobrain.core.trigger.AllDataReceivedTrigger",
            "name": "t1",
        }
        cfg = _build(
            name="t", gate_semantics="gate_to_bottom",
            triggers=[trig],
        )
        assert cfg.triggers[0]["gate_semantics"] == "gate_to_bottom"

    def test_step_level_triggers_flat_shape(self):
        step = {
            "class": "demos.x.MyStep",
            "triggers": [
                {
                    "class": "nanobrain.core.trigger.AllDataReceivedTrigger",
                    "name": "t1",
                },
            ],
        }
        cfg = _build(
            name="t", gate_semantics="gate_to_bottom",
            steps={"s1": step},
        )
        assert cfg.steps["s1"]["triggers"][0]["gate_semantics"] == "gate_to_bottom"

    def test_step_level_triggers_nested_under_config(self):
        step = {
            "class": "demos.x.MyStep",
            "config": {
                "name": "s1",
                "triggers": [
                    {
                        "class": "nanobrain.core.trigger.AllDataReceivedTrigger",
                        "name": "t1",
                    },
                ],
            },
        }
        cfg = _build(
            name="t", gate_semantics="gate_to_bottom",
            steps={"s1": step},
        )
        triggers = cfg.steps["s1"]["config"]["triggers"]
        assert triggers[0]["gate_semantics"] == "gate_to_bottom"

    def test_timer_trigger_not_touched(self):
        trig = {
            "class": "nanobrain.core.trigger.TimerTrigger",
            "name": "ticker",
            "timer_interval_ms": 1000,
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     triggers=[trig])
        assert "gate_semantics" not in cfg.triggers[0]

    def test_explicit_value_preserved_per_trigger(self):
        trig = {
            "class": "nanobrain.core.trigger.AllDataReceivedTrigger",
            "name": "t1",
            "gate_semantics": "publish_empty",
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     triggers=[trig])
        assert cfg.triggers[0]["gate_semantics"] == "publish_empty"


# ---------------------------------------------------------------------------
# 5. Both modes round-trip
# ---------------------------------------------------------------------------

class TestBothModes:

    def test_publish_empty_workflow_default(self):
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "source": "a", "target": "b", "condition": "true_only",
        }
        cfg = _build(name="t", gate_semantics="publish_empty",
                     links={"l": link_dict})
        assert cfg.links["l"]["gate_semantics"] == "publish_empty"

    def test_gate_to_bottom_workflow_default(self):
        link_dict = {
            "class": "nanobrain.core.link.ConditionalLink",
            "source": "a", "target": "b", "condition": "true_only",
        }
        cfg = _build(name="t", gate_semantics="gate_to_bottom",
                     links={"l": link_dict})
        assert cfg.links["l"]["gate_semantics"] == "gate_to_bottom"


# ---------------------------------------------------------------------------
# 6. Mixed workflow with multiple components
# ---------------------------------------------------------------------------

class TestMixedWorkflow:

    def test_multi_link_multi_trigger(self):
        cfg = _build(
            name="t", gate_semantics="gate_to_bottom",
            links={
                "cond": {
                    "class": "nanobrain.core.link.ConditionalLink",
                    "source": "a.x", "target": "b.x",
                    "condition": "true_only",
                },
                "direct": {
                    "class": "nanobrain.core.link.DirectLink",
                    "source": "c.x", "target": "d.x",
                },
            },
            triggers=[
                {
                    "class": "nanobrain.core.trigger.AllDataReceivedTrigger",
                    "name": "fan_in",
                },
                {
                    "class": "nanobrain.core.trigger.TimerTrigger",
                    "name": "ticker",
                    "timer_interval_ms": 1000,
                },
            ],
        )
        assert cfg.links["cond"]["gate_semantics"] == "gate_to_bottom"
        assert "gate_semantics" not in cfg.links["direct"]
        assert cfg.triggers[0]["gate_semantics"] == "gate_to_bottom"
        assert "gate_semantics" not in cfg.triggers[1]


# ---------------------------------------------------------------------------
# Helper functions exported for reuse
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_link_gate_aware_check(self):
        assert _link_class_needs_gate_semantics_check(
            "nanobrain.core.link.ConditionalLink"
        ) is True
        assert _link_class_needs_gate_semantics_check(
            "nanobrain.core.link.DirectLink"
        ) is False
        assert _link_class_needs_gate_semantics_check("") is False
        assert _link_class_needs_gate_semantics_check(None) is False

    def test_trigger_gate_aware_check(self):
        assert _trigger_class_needs_gate_semantics_check(
            "nanobrain.core.trigger.AllDataReceivedTrigger"
        ) is True
        assert _trigger_class_needs_gate_semantics_check(
            "nanobrain.core.trigger.TimerTrigger"
        ) is False

    def test_set_inline_default_flat(self):
        d = {"class": "X"}
        assert _set_inline_default_in_entry(d, "k", "v") is True
        assert d["k"] == "v"
        # Second call no-ops:
        assert _set_inline_default_in_entry(d, "k", "different") is False
        assert d["k"] == "v"

    def test_set_inline_default_nested(self):
        d = {"class": "X", "config": {"a": 1}}
        assert _set_inline_default_in_entry(d, "k", "v") is True
        assert d["config"]["k"] == "v"
        assert "k" not in d  # only inner mutated

    def test_set_inline_default_path_reference(self):
        d = {"class": "X", "config": "path.yml"}
        assert _set_inline_default_in_entry(d, "k", "v") is False
        assert d["config"] == "path.yml"
