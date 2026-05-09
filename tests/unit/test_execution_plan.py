"""Tests for G16 — ExecutionPlanConfig + ExecutionPlanDataUnit.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G16`` and
``apecx-mcp-integration/docs/agent_workflow_authoring.md §3.1``: the
typed Phase-0 ExecutionPlan schema and its DataUnit carrier.

Tests cover:
1. ExecutionPlanConfig minimal valid plan + extra='forbid' enforcement
2. Per-field validation (skeleton_version, resource_envelope bounds)
3. Strategy A vs B field interactions (skeleton_refs / inter_skeleton_links)
4. ExecutionPlanLayer / ToolInvocation / ResourceEnvelope sub-models
5. Round-trip through ExecutionPlanDataUnit
"""

from __future__ import annotations

import asyncio

import pytest

from nanobrain.library.orchestration import (
    ExecutionPlanConfig,
    ExecutionPlanDataUnit,
    ExecutionPlanInterSkeletonLink,
    ExecutionPlanLayer,
    ExecutionPlanProvenanceSeed,
    ExecutionPlanResourceEnvelope,
    ExecutionPlanSkeletonRef,
    ExecutionPlanToolInvocation,
)


# ---------------------------------------------------------------------------
# Helpers — bypass FromConfigBase prohibition for test construction
# ---------------------------------------------------------------------------

def _build(cls, **kwargs):
    cls._allow_direct_instantiation = True
    try:
        return cls(**kwargs)
    finally:
        cls._allow_direct_instantiation = False


def _minimal_envelope() -> ExecutionPlanResourceEnvelope:
    return _build(
        ExecutionPlanResourceEnvelope,
        executor="LocalExecutor",
        walltime_minutes=10,
        estimated_cost_usd=0.5,
        hpc_eligible=False,
    )


def _minimal_provenance() -> ExecutionPlanProvenanceSeed:
    return _build(
        ExecutionPlanProvenanceSeed,
        session_id="sess-test",
        user_id="user-test",
        intent="test_intent",
        phase0_evidence_refs=[],
    )


def _minimal_plan(**overrides) -> ExecutionPlanConfig:
    base = dict(
        strategy="A",
        skeleton_id="multi_source_discovery",
        skeleton_version="a7c2e4f91b03",
        active_layers=["sequence", "structural"],
        resource_envelope=_minimal_envelope(),
        provenance_seed=_minimal_provenance(),
    )
    base.update(overrides)
    return _build(ExecutionPlanConfig, **base)


# ---------------------------------------------------------------------------
# 1. Minimal valid plan + extra='forbid'
# ---------------------------------------------------------------------------

class TestMinimalPlanShape:

    def test_minimal_strategy_a_plan(self):
        plan = _minimal_plan()
        assert plan.plan_version == "1"
        assert plan.strategy == "A"
        assert plan.skeleton_id == "multi_source_discovery"
        assert plan.active_layers == ["sequence", "structural"]
        assert plan.parameter_bindings == {}
        assert plan.tool_invocations == []
        assert plan.skeleton_refs is None
        assert plan.inter_skeleton_links is None

    def test_extra_field_rejected_at_top_level(self):
        with pytest.raises(Exception) as exc_info:
            _build(
                ExecutionPlanConfig,
                strategy="A",
                skeleton_id="x",
                skeleton_version="abcdef012345",
                active_layers=[],
                resource_envelope=_minimal_envelope(),
                provenance_seed=_minimal_provenance(),
                ghost_field="bad",
            )
        assert "ghost_field" in str(exc_info.value).lower() or "extra" in str(exc_info.value).lower()

    def test_extra_field_rejected_in_layer(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanLayer,
                layer_id="seq",
                layer_type="sequence",
                ghost="bad",
            )

    def test_extra_field_rejected_in_resource_envelope(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanResourceEnvelope,
                executor="LocalExecutor",
                walltime_minutes=10,
                estimated_cost_usd=0.5,
                hpc_eligible=False,
                ghost="bad",
            )

    def test_extra_field_rejected_in_provenance_seed(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanProvenanceSeed,
                session_id="x",
                user_id="y",
                intent="z",
                ghost="bad",
            )


# ---------------------------------------------------------------------------
# 2. Per-field validation
# ---------------------------------------------------------------------------

class TestFieldValidation:

    def test_strategy_must_be_abc(self):
        with pytest.raises(Exception):
            _minimal_plan(strategy="D")
        with pytest.raises(Exception):
            _minimal_plan(strategy="X")

    def test_skeleton_version_empty_rejected(self):
        with pytest.raises(Exception) as exc_info:
            _minimal_plan(skeleton_version="")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_skeleton_version_whitespace_rejected(self):
        with pytest.raises(Exception):
            _minimal_plan(skeleton_version="   ")
        with pytest.raises(Exception):
            _minimal_plan(skeleton_version="abc def")

    def test_skeleton_version_hex_prefix_accepted(self):
        plan = _minimal_plan(skeleton_version="a7c2e4f91b03")
        assert plan.skeleton_version == "a7c2e4f91b03"

    def test_skeleton_version_semver_tag_accepted(self):
        plan = _minimal_plan(skeleton_version="1.2.3-beta.1")
        assert plan.skeleton_version == "1.2.3-beta.1"

    def test_walltime_min_bound(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanResourceEnvelope,
                executor="LocalExecutor",
                walltime_minutes=0,
                estimated_cost_usd=0.0,
                hpc_eligible=False,
            )

    def test_walltime_max_bound(self):
        # 1440 = 24h cap per spec.
        env = _build(
            ExecutionPlanResourceEnvelope,
            executor="LocalExecutor",
            walltime_minutes=1440,
            estimated_cost_usd=0.0,
            hpc_eligible=False,
        )
        assert env.walltime_minutes == 1440
        with pytest.raises(Exception):
            _build(
                ExecutionPlanResourceEnvelope,
                executor="LocalExecutor",
                walltime_minutes=1441,
                estimated_cost_usd=0.0,
                hpc_eligible=False,
            )

    def test_cost_negative_rejected(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanResourceEnvelope,
                executor="LocalExecutor",
                walltime_minutes=1,
                estimated_cost_usd=-1.0,
                hpc_eligible=False,
            )

    def test_layer_type_validated(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanLayer,
                layer_id="x",
                layer_type="ghost_layer",
            )

    def test_active_layers_only_accepts_known_types(self):
        with pytest.raises(Exception):
            _minimal_plan(active_layers=["sequence", "made_up"])


# ---------------------------------------------------------------------------
# 3. Strategy B fields (skeleton_refs + inter_skeleton_links)
# ---------------------------------------------------------------------------

class TestStrategyBFields:

    def test_skeleton_ref_minimal(self):
        ref = _build(
            ExecutionPlanSkeletonRef,
            skeleton_id="other_skel",
            skeleton_version="abcdef012345",
            alias="other",
        )
        assert ref.alias == "other"

    def test_inter_skeleton_link_uses_from_alias(self):
        """The 'from' field uses Field(alias='from') because Python
        reserves 'from' as a keyword. Verify both spellings work."""
        link = ExecutionPlanInterSkeletonLink._allow_direct_instantiation = True
        try:
            link = ExecutionPlanInterSkeletonLink(**{
                "from": "alpha.output",
                "to": "beta.input",
                "link_class": "DirectLink",
            })
        finally:
            ExecutionPlanInterSkeletonLink._allow_direct_instantiation = False
        assert link.from_ == "alpha.output"
        assert link.to == "beta.input"
        assert link.link_class == "DirectLink"

    def test_inter_skeleton_link_with_g1_predicate(self):
        """ConditionalLink-class boundary link carries a predicate dict
        in the G1 PredicateConfig shape."""
        ExecutionPlanInterSkeletonLink._allow_direct_instantiation = True
        try:
            link = ExecutionPlanInterSkeletonLink(**{
                "from": "alpha.output",
                "to": "beta.input",
                "link_class": "ConditionalLink",
                "predicate": {
                    "op": "contains",
                    "field": "active_layers",
                    "value": "structural",
                },
            })
        finally:
            ExecutionPlanInterSkeletonLink._allow_direct_instantiation = False
        assert link.predicate["op"] == "contains"

    def test_strategy_b_plan_with_refs_and_links(self):
        ref = _build(
            ExecutionPlanSkeletonRef,
            skeleton_id="other_skel",
            skeleton_version="cafe1234abcd",
            alias="other",
        )
        ExecutionPlanInterSkeletonLink._allow_direct_instantiation = True
        try:
            link = ExecutionPlanInterSkeletonLink(**{
                "from": "primary.output",
                "to": "other.input",
                "link_class": "DirectLink",
            })
        finally:
            ExecutionPlanInterSkeletonLink._allow_direct_instantiation = False

        plan = _minimal_plan(
            strategy="B",
            skeleton_refs=[ref],
            inter_skeleton_links=[link],
        )
        assert plan.strategy == "B"
        assert len(plan.skeleton_refs) == 1
        assert len(plan.inter_skeleton_links) == 1


# ---------------------------------------------------------------------------
# 4. Tool invocations
# ---------------------------------------------------------------------------

class TestToolInvocations:

    def test_minimal_tool_invocation(self):
        ti = _build(
            ExecutionPlanToolInvocation,
            slot_id="alignment",
            tool_descriptor_ref="rhea:muscle.align@5.1.0",
        )
        assert ti.input_bindings == {}

    def test_with_bindings(self):
        ti = _build(
            ExecutionPlanToolInvocation,
            slot_id="alignment",
            tool_descriptor_ref="rhea:muscle.align@5.1.0",
            input_bindings={
                "sequences": "{{sequence_layer.output.sequences}}",
                "alphabet": "protein",
            },
        )
        assert ti.input_bindings["alphabet"] == "protein"

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            _build(
                ExecutionPlanToolInvocation,
                slot_id="x",
                tool_descriptor_ref="y",
                ghost="bad",
            )


# ---------------------------------------------------------------------------
# 5. ExecutionPlanDataUnit round-trip
# ---------------------------------------------------------------------------

class TestExecutionPlanDataUnit:

    def test_set_and_get_via_data_unit(self):
        async def run():
            plan = _minimal_plan()
            du = ExecutionPlanDataUnit.from_config({
                "class": "nanobrain.library.orchestration.execution_plan.ExecutionPlanDataUnit",
                "name": "execution_plan",
            })
            await du.set(plan)
            roundtripped = await du.get()
            assert roundtripped is plan
            assert roundtripped.skeleton_id == "multi_source_discovery"
        asyncio.run(run())

    def test_data_unit_class_inherits_data_unit_memory_semantics(self):
        """ExecutionPlanDataUnit IS a DataUnitMemory subclass and lives
        at `nanobrain.library.orchestration.execution_plan.ExecutionPlanDataUnit`,
        which the DataUnitConfig.class_field validator now admits (the
        validator's allowlist was widened 2026-05-09 to include
        `nanobrain.library.*` paths so framework-shipped library data
        units like this one are loadable through the standard path)."""
        from nanobrain.core.data_unit import DataUnitMemory
        assert issubclass(ExecutionPlanDataUnit, DataUnitMemory)
