"""Tests for G2 — dynamic AllDataReceivedTrigger expected-set narrowing.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G2``: a trigger
that supports `expected_set_source` (a workflow-level data unit reference),
`expected_set_field` (a dotted projection path reusing G1's resolver), and
`expected_set_naming` (a str.format template for canonical naming).

These tests cover the standalone resolver function (``_resolve_expected_set``)
in isolation. End-to-end integration with the workflow runtime — which threads
the source data unit into trigger activation — is a follow-up; this commit
ships the primitive pieces so apecx-mcp can adopt them in custom workflow
code while the framework integration is incremental.
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.trigger import AllDataReceivedTrigger, TriggerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDataUnit:
    """Minimal stub that supports the .get() async API the resolver needs."""

    def __init__(self, payload):
        self._payload = payload

    async def get(self):
        return self._payload


def _build_trigger(**cfg_kwargs) -> AllDataReceivedTrigger:
    """Build an AllDataReceivedTrigger with the given config + empty data_units."""
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(**cfg_kwargs)
    finally:
        TriggerConfig._allow_direct_instantiation = False
    # AllDataReceivedTrigger.from_config takes a TriggerConfig directly:
    return AllDataReceivedTrigger.from_config(cfg, data_units=[])


# ---------------------------------------------------------------------------
# 1. Static-list path (backward compatibility)
# ---------------------------------------------------------------------------

class TestStaticListUnchanged:

    def test_no_dynamic_source_returns_static_inputs(self):
        async def run():
            trig = _build_trigger(name="t1", trigger_type="all_data_received")
            result = await trig._resolve_expected_set(
                source_data_unit=None,
                static_inputs=["a", "b", "c"],
            )
            assert result == {"a", "b", "c"}
        asyncio.run(run())

    def test_no_dynamic_source_caches(self):
        """Repeated calls without a source return the cached set."""
        async def run():
            trig = _build_trigger(name="t1", trigger_type="all_data_received")
            r1 = await trig._resolve_expected_set(None, ["a", "b"])
            r2 = await trig._resolve_expected_set(None, ["a", "b"])
            assert r1 is r2  # same object — cache hit
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 2. Dynamic projection happy paths
# ---------------------------------------------------------------------------

class TestDynamicProjection:

    def test_projects_active_layers(self):
        """The motivating example from the gap proposal: project
        `active_layers` from an ExecutionPlan-shaped payload."""
        async def run():
            trig = _build_trigger(
                name="evidence",
                trigger_type="all_data_received",
                expected_set_source="workflow.execution_plan",
                expected_set_field="active_layers",
                expected_set_naming="{value}_layer.layer_result_output",
            )
            source = _FakeDataUnit({
                "active_layers": ["sequence", "structural"],
            })
            result = await trig._resolve_expected_set(
                source,
                static_inputs=[
                    "sequence_layer.layer_result_output",
                    "structural_layer.layer_result_output",
                    "functional_layer.layer_result_output",
                    "cross_source_layer.layer_result_output",
                ],
            )
            assert result == {
                "sequence_layer.layer_result_output",
                "structural_layer.layer_result_output",
            }
        asyncio.run(run())

    def test_identity_naming_template(self):
        """Default template is identity — projected values used as-is."""
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
            )
            source = _FakeDataUnit({"items": ["a", "b"]})
            result = await trig._resolve_expected_set(source, ["a", "b", "c"])
            assert result == {"a", "b"}
        asyncio.run(run())

    def test_caches_after_first_resolve(self):
        """Per spec — resolver is one-shot per activation. Subsequent calls
        return the cached set without re-reading the source."""
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
            )
            source = _FakeDataUnit({"items": ["a"]})
            r1 = await trig._resolve_expected_set(source, ["a", "b"])
            # Mutate source — cached result should NOT change.
            source._payload = {"items": ["a", "b"]}
            r2 = await trig._resolve_expected_set(source, ["a", "b"])
            assert r1 is r2
            assert r2 == {"a"}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3. FAIL-FAST paths
# ---------------------------------------------------------------------------

class TestResolverFailFast:

    def test_missing_source_when_required(self):
        """expected_set_source set + None source ⇒ FAIL-FAST."""
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
            )
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await trig._resolve_expected_set(None, ["a"])
            assert "FAIL-FAST" in str(exc_info.value)
            assert "expected_set_source" in str(exc_info.value)
        asyncio.run(run())

    def test_field_path_missing(self):
        """Source payload doesn't contain the projected field."""
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="active_layers",
            )
            source = _FakeDataUnit({"unrelated": "x"})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await trig._resolve_expected_set(source, ["a"])
            assert "FAIL-FAST" in str(exc_info.value)
            assert "missing in source data unit payload" in str(exc_info.value)
        asyncio.run(run())

    def test_projection_not_a_list(self):
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
            )
            source = _FakeDataUnit({"items": "not-a-list"})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await trig._resolve_expected_set(source, ["a"])
            assert "FAIL-FAST" in str(exc_info.value)
            assert "expected list[str]" in str(exc_info.value)
        asyncio.run(run())

    def test_projection_contains_non_strings(self):
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
            )
            source = _FakeDataUnit({"items": ["a", 42, "b"]})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await trig._resolve_expected_set(source, ["a"])
            assert "FAIL-FAST" in str(exc_info.value)
            assert "non-string elements" in str(exc_info.value)
        asyncio.run(run())

    def test_projected_names_not_subset_of_inputs(self):
        """The projection MUST be a subset of static inputs — names not
        in the static set are off-DAG and the trigger would deadlock
        waiting for them."""
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
            )
            source = _FakeDataUnit({"items": ["a", "ghost"]})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await trig._resolve_expected_set(source, ["a", "b"])
            assert "FAIL-FAST" in str(exc_info.value)
            assert "off-DAG names" in str(exc_info.value)
            assert "ghost" in str(exc_info.value)
        asyncio.run(run())

    def test_naming_template_format_failure(self):
        """A naming template that references unknown placeholders FAIL-FASTs."""
        async def run():
            trig = _build_trigger(
                name="t",
                trigger_type="all_data_received",
                expected_set_source="workflow.plan",
                expected_set_field="items",
                expected_set_naming="{undefined_placeholder}",  # not 'value'
            )
            source = _FakeDataUnit({"items": ["a"]})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await trig._resolve_expected_set(source, ["a"])
            assert "FAIL-FAST" in str(exc_info.value)
            assert "expected_set_naming" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. TriggerConfig field defaults
# ---------------------------------------------------------------------------

class TestTriggerConfigDefaults:

    def test_default_naming_is_identity(self):
        TriggerConfig._allow_direct_instantiation = True
        try:
            cfg = TriggerConfig()
        finally:
            TriggerConfig._allow_direct_instantiation = False
        assert cfg.expected_set_source is None
        assert cfg.expected_set_field is None
        assert cfg.expected_set_naming == "{value}"
