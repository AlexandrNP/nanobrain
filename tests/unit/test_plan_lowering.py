"""Tests for G17 — PlanLoweringStep + SkeletonLoaderStep.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G17`` and
``apecx-mcp-integration/docs/agent_workflow_authoring.md §5``.

Tests cover:
1. SkeletonLoaderStep — registry resolution, error paths
2. PlanLoweringStep — Step 2 (binding validation FAIL-FAST shapes)
3. PlanLoweringStep — Step 3 (hole substitution preserves YAML)
4. PlanLoweringStep — Step 5 (tool descriptor recording)
5. PlanLoweringStep — Step 6 (provenance seed injection)
6. PlanLoweringStep — Step 7 (lowered_yaml_hash determinism)
7. End-to-end: skeleton + plan → lowered YAML that parses
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.orchestration import (
    PlanLoweringStep,
    Skeleton,
    SkeletonLoaderStep,
    SkeletonRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_step_from_dict(cls, config_dict: Dict[str, Any], **kwargs):
    """Build a step via temp YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml.safe_dump(config_dict))
        path = f.name
    return cls.from_config(path, **kwargs)


def _build_skel(**overrides):
    base = dict(
        skeleton_id="test_skel",
        skeleton_version="1.0.0",
        description="test",
        body=(
            "class: nanobrain.core.workflow.Workflow\n"
            "config:\n"
            "  name: test\n"
            "  param: {{the_param: string}}\n"
            "  count: {{the_count: integer}}\n"
        ),
        holes={
            "the_param": {"type": "string", "required": True},
            "the_count": {"type": "integer", "required": False, "default": 10},
        },
    )
    base.update(overrides)
    return Skeleton.from_config(base)


# ---------------------------------------------------------------------------
# 1. SkeletonLoaderStep
# ---------------------------------------------------------------------------

class TestSkeletonLoaderStep:

    def test_loader_requires_registry(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _build_step_from_dict(SkeletonLoaderStep, {"name": "x"})
        assert "FAIL-FAST" in str(exc_info.value)
        assert "skeleton_registry" in str(exc_info.value)

    def test_loader_rejects_non_registry(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _build_step_from_dict(
                SkeletonLoaderStep, {"name": "x"},
                skeleton_registry="not a registry",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "SkeletonRegistry" in str(exc_info.value)

    def test_resolves_by_semver(self):
        async def run():
            reg = SkeletonRegistry()
            sk = _build_skel(skeleton_version="1.0.0")
            reg.register(sk)
            loader = _build_step_from_dict(
                SkeletonLoaderStep, {"name": "ld"}, skeleton_registry=reg)
            result = await loader.process({
                "skeleton_id": "test_skel",
                "skeleton_version": "1.0.0",
            })
            assert result["skeleton"] is sk
            # The resolved version is ALWAYS the content_hash, even if
            # the input was a semver tag — this is the spec.
            assert result["skeleton_version"] == sk.content_hash
            assert result["content_hash"] == sk.content_hash
        asyncio.run(run())

    def test_resolves_by_12_char_prefix(self):
        async def run():
            reg = SkeletonRegistry()
            sk = _build_skel()
            reg.register(sk)
            loader = _build_step_from_dict(
                SkeletonLoaderStep, {"name": "ld"}, skeleton_registry=reg)
            result = await loader.process({
                "skeleton_id": "test_skel",
                "skeleton_version": sk.content_hash[:12],
            })
            assert result["skeleton"] is sk
        asyncio.run(run())

    def test_unknown_skeleton_fails_fast(self):
        async def run():
            reg = SkeletonRegistry()
            loader = _build_step_from_dict(
                SkeletonLoaderStep, {"name": "ld"}, skeleton_registry=reg)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await loader.process({
                    "skeleton_id": "ghost", "skeleton_version": "1.0.0"})
            assert "FAIL-FAST" in str(exc_info.value)
            assert "ghost" in str(exc_info.value)
        asyncio.run(run())

    def test_missing_input_keys_fails_fast(self):
        async def run():
            reg = SkeletonRegistry()
            loader = _build_step_from_dict(
                SkeletonLoaderStep, {"name": "ld"}, skeleton_registry=reg)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await loader.process({"skeleton_id": "x"})  # missing version
            assert "FAIL-FAST" in str(exc_info.value)
            assert "missing required keys" in str(exc_info.value)
        asyncio.run(run())

    def test_non_dict_input_fails_fast(self):
        async def run():
            reg = SkeletonRegistry()
            loader = _build_step_from_dict(
                SkeletonLoaderStep, {"name": "ld"}, skeleton_registry=reg)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await loader.process("not a dict")
            assert "FAIL-FAST" in str(exc_info.value)
            assert "must be a dict" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 2. PlanLoweringStep — Step 2 binding validation
# ---------------------------------------------------------------------------

class TestBindingValidation:

    def test_missing_required_hole_fails_fast(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await lowering.process({
                    "skeleton": sk,
                    "execution_plan": {
                        "parameter_bindings": {},  # missing the_param
                    },
                })
            assert "FAIL-FAST" in str(exc_info.value)
            assert "missing required holes" in str(exc_info.value)
            assert "the_param" in str(exc_info.value)
        asyncio.run(run())

    def test_extra_binding_key_fails_fast(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await lowering.process({
                    "skeleton": sk,
                    "execution_plan": {
                        "parameter_bindings": {
                            "the_param": "x",
                            "ghost_key": "should_not_be_here",
                        },
                    },
                })
            assert "FAIL-FAST" in str(exc_info.value)
            assert "extra binding keys" in str(exc_info.value)
            assert "ghost_key" in str(exc_info.value)
        asyncio.run(run())

    def test_optional_default_applied(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                },
            })
            # the_count's default (10) was applied:
            summary = result["binding_summary"]
            assert summary["the_count"]["value"] == 10
            assert summary["the_count"]["source"] == "default"
            assert summary["the_param"]["source"] == "binding"
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3. PlanLoweringStep — Step 3 hole substitution
# ---------------------------------------------------------------------------

class TestHoleSubstitution:

    def test_string_substitution(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "ENTITY_X"},
                },
            })
            body = result["lowered_yaml"]
            # Substituted as JSON-quoted string (valid YAML):
            assert '"ENTITY_X"' in body
            # Original hole token is GONE:
            assert "{{the_param" not in body
        asyncio.run(run())

    def test_substitution_produces_parseable_yaml(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {
                        "the_param": "test_value",
                        "the_count": 42,
                    },
                },
            })
            body = result["lowered_yaml"]
            # Strip provenance comment header for YAML parsing:
            parsed = yaml.safe_load(body)
            assert parsed["config"]["param"] == "test_value"
            assert parsed["config"]["count"] == 42
        asyncio.run(run())

    def test_array_substitution(self):
        async def run():
            sk = Skeleton.from_config({
                "skeleton_id": "arr",
                "skeleton_version": "1.0.0",
                "body": "config:\n  items: {{my_list: array}}\n",
                "holes": {"my_list": {"type": "array", "required": True}},
            })
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"my_list": ["a", "b", "c"]},
                },
            })
            parsed = yaml.safe_load(result["lowered_yaml"])
            assert parsed["config"]["items"] == ["a", "b", "c"]
        asyncio.run(run())

    def test_boolean_substitution(self):
        async def run():
            sk = Skeleton.from_config({
                "skeleton_id": "bool",
                "skeleton_version": "1.0.0",
                "body": "config:\n  flag: {{my_flag: boolean}}\n",
                "holes": {"my_flag": {"type": "boolean", "required": True}},
            })
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"my_flag": True},
                },
            })
            parsed = yaml.safe_load(result["lowered_yaml"])
            assert parsed["config"]["flag"] is True
        asyncio.run(run())

    def test_null_substitution(self):
        async def run():
            sk = Skeleton.from_config({
                "skeleton_id": "nullable",
                "skeleton_version": "1.0.0",
                "body": "config:\n  optional_field: {{my_field: any}}\n",
                "holes": {"my_field": {"type": "any", "required": False}},
            })
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {},  # optional, no value
                },
            })
            parsed = yaml.safe_load(result["lowered_yaml"])
            assert parsed["config"]["optional_field"] is None
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. Tool descriptor recording
# ---------------------------------------------------------------------------

class TestToolDescriptorRecording:

    def test_tool_invocations_recorded_in_body(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                    "tool_invocations": [
                        {"slot_id": "alignment", "tool_descriptor_ref": "rhea:muscle@5.1.0"},
                        {"slot_id": "search", "tool_descriptor_ref": "rhea:blast@2.14.0"},
                    ],
                },
            })
            body = result["lowered_yaml"]
            assert "tool invocations" in body
            assert "alignment" in body
            assert "rhea:muscle@5.1.0" in body
            assert "search" in body
            assert "rhea:blast@2.14.0" in body
        asyncio.run(run())

    def test_no_tool_invocations_no_comment(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                },
            })
            assert "tool invocations" not in result["lowered_yaml"]
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Provenance seed injection
# ---------------------------------------------------------------------------

class TestProvenanceSeedInjection:

    def test_seed_injected_as_comment_header(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                    "provenance_seed": {
                        "session_id": "sess-xyz-001",
                        "user_id": "scientist-42",
                        "intent": "discovery",
                        "phase0_evidence_refs": ["ref-a", "ref-b"],
                    },
                },
            })
            body = result["lowered_yaml"]
            assert "provenance header" in body
            assert "session_id" in body
            assert "sess-xyz-001" in body
            assert "scientist-42" in body
            assert "skeleton_content_hash" in body
            assert sk.content_hash in body
            # Header is a YAML comment (starts with #) — doesn't affect parse:
            parsed = yaml.safe_load(body)
            assert parsed["config"]["param"] == "x"
        asyncio.run(run())

    def test_empty_seed_still_injects_skeleton_hash(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            result = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                    "provenance_seed": {},
                },
            })
            body = result["lowered_yaml"]
            assert "skeleton_content_hash" in body
            assert sk.content_hash in body
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. lowered_yaml_hash determinism
# ---------------------------------------------------------------------------

class TestLoweredHashDeterminism:

    def test_same_inputs_same_hash(self):
        async def run():
            sk = _build_skel()
            lowering1 = _build_step_from_dict(
                PlanLoweringStep, {"name": "low1"})
            lowering2 = _build_step_from_dict(
                PlanLoweringStep, {"name": "low2"})
            inputs = {
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                    "provenance_seed": {"session_id": "s1"},
                },
            }
            r1 = await lowering1.process(inputs)
            r2 = await lowering2.process(inputs)
            assert r1["lowered_yaml_hash"] == r2["lowered_yaml_hash"]
            assert len(r1["lowered_yaml_hash"]) == 64
        asyncio.run(run())

    def test_different_bindings_different_hash(self):
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            r1 = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "alpha"},
                },
            })
            r2 = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "beta"},
                },
            })
            assert r1["lowered_yaml_hash"] != r2["lowered_yaml_hash"]
        asyncio.run(run())

    def test_different_provenance_changes_hash(self):
        """Per spec, the provenance seed IS part of the lowered body
        (as a comment header) — different seeds produce different hashes.
        This is intentional: the lowered_yaml_hash identifies the EXACT
        bytes that will execute, including provenance-bearing metadata."""
        async def run():
            sk = _build_skel()
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})
            r1 = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                    "provenance_seed": {"session_id": "s1"},
                },
            })
            r2 = await lowering.process({
                "skeleton": sk,
                "execution_plan": {
                    "parameter_bindings": {"the_param": "x"},
                    "provenance_seed": {"session_id": "s2"},
                },
            })
            assert r1["lowered_yaml_hash"] != r2["lowered_yaml_hash"]
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. End-to-end: skeleton → plan → lowered YAML
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self):
        async def run():
            # Build skeleton + register:
            sk = _build_skel(skeleton_id="e2e", skeleton_version="1.0.0")
            reg = SkeletonRegistry()
            reg.register(sk)

            # Loader → Lowering chain:
            loader = _build_step_from_dict(
                SkeletonLoaderStep, {"name": "ld"}, skeleton_registry=reg)
            lowering = _build_step_from_dict(
                PlanLoweringStep, {"name": "low"})

            loaded = await loader.process({
                "skeleton_id": "e2e",
                "skeleton_version": "1.0.0",
            })
            result = await lowering.process({
                "skeleton": loaded["skeleton"],
                "execution_plan": {
                    "parameter_bindings": {"the_param": "TARGET"},
                    "provenance_seed": {"session_id": "abc"},
                    "tool_invocations": [
                        {"slot_id": "tool1", "tool_descriptor_ref": "rhea:t@1.0"}
                    ],
                },
            })

            # Verify the lowered YAML body parses and contains substituted values:
            body = result["lowered_yaml"]
            parsed = yaml.safe_load(body)
            assert parsed["config"]["name"] == "test"
            assert parsed["config"]["param"] == "TARGET"
            # Provenance header present:
            assert "session_id" in body
            # Tool invocation recorded:
            assert "rhea:t@1.0" in body
            # Reproducibility hash:
            assert len(result["lowered_yaml_hash"]) == 64
        asyncio.run(run())
