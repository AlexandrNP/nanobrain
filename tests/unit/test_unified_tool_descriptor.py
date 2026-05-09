"""Tests for G15 — UnifiedToolDescriptor primitive.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G15`` and
``apecx-mcp-integration/docs/tool_descriptor_contract.md §2``: the typed
tool-card primitive that replaces the free-form ToolConfig.tool_card dict.

Tests cover:
1. UTD shape validation + extra='forbid'
2. descriptor_id grammar
3. Auto-resolution of descriptor_hash
4. compute_descriptor_hash determinism + canonical ordering
5. ToolBase.from_descriptor — happy path + capability check + errors
"""

from __future__ import annotations

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import (
    UnifiedToolDescriptor,
    UTDInputSpec,
    UTDOutputSpec,
    UTDCostEstimate,
    UTDFailureMode,
    UTDProvenancePin,
    UTDVersionEntry,
    ToolBase,
    compute_descriptor_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(cls, **kwargs):
    cls._allow_direct_instantiation = True
    try:
        return cls(**kwargs)
    finally:
        cls._allow_direct_instantiation = False


def _minimal_pin(class_path: str = "nanobrain.core.tool.FunctionTool") -> UTDProvenancePin:
    return _build(UTDProvenancePin, class_path=class_path)


def _minimal_utd(**overrides) -> UnifiedToolDescriptor:
    base = dict(
        descriptor_id="rhea:muscle.align@5.1.0",
        display_name="MUSCLE Align",
        summary="Multiple sequence alignment via MUSCLE.",
        provenance_pin=_minimal_pin(),
    )
    base.update(overrides)
    return _build(UnifiedToolDescriptor, **base)


# ---------------------------------------------------------------------------
# 1. UTD shape validation
# ---------------------------------------------------------------------------

class TestUTDShape:

    def test_minimal_utd(self):
        utd = _minimal_utd()
        assert utd.descriptor_id == "rhea:muscle.align@5.1.0"
        assert utd.display_name == "MUSCLE Align"
        assert utd.side_effects == "none"
        assert utd.determinism == "R3"
        assert utd.resource_class == "cpu_light"
        assert utd.descriptor_hash is not None
        assert len(utd.descriptor_hash) == 64

    def test_full_utd(self):
        utd = _build(
            UnifiedToolDescriptor,
            descriptor_id="native:my_tool@1.0.0",
            display_name="My Tool",
            summary="Does a thing.",
            long_description="More detail here.",
            inputs=[
                _build(UTDInputSpec, name="x", type="string", required=True),
                _build(UTDInputSpec, name="y", type="integer", required=False, default=10),
            ],
            outputs=[
                _build(UTDOutputSpec, name="result", type="string"),
            ],
            side_effects="filesystem_write",
            determinism="R2",
            resource_class="cpu_medium",
            cost_estimate=_build(
                UTDCostEstimate,
                estimated_seconds=12.5,
                estimated_usd=0.01,
                confidence="medium",
            ),
            failure_modes=[
                _build(
                    UTDFailureMode,
                    code="input_too_large",
                    detail="x exceeded 1MB",
                    detection_signal="check input size before call",
                    recovery="split input into chunks",
                ),
            ],
            provenance_pin=_minimal_pin(),
            requires_capability=["read_filesystem"],
            version_history=[
                _build(
                    UTDVersionEntry,
                    version="1.0.0",
                    published_at="2026-05-09T00:00:00Z",
                    deprecated=False,
                ),
            ],
        )
        assert utd.cost_estimate.estimated_seconds == 12.5
        assert len(utd.failure_modes) == 1
        assert utd.failure_modes[0].code == "input_too_large"
        assert utd.requires_capability == ["read_filesystem"]

    def test_extra_field_rejected_at_top_level(self):
        with pytest.raises(Exception):
            _minimal_utd(ghost_field="bad")

    def test_extra_field_rejected_in_input(self):
        with pytest.raises(Exception):
            _build(UTDInputSpec, name="x", type="string", ghost="bad")

    def test_extra_field_rejected_in_provenance_pin(self):
        with pytest.raises(Exception):
            _build(UTDProvenancePin, class_path="x.y.Z", ghost="bad")


# ---------------------------------------------------------------------------
# 2. descriptor_id grammar
# ---------------------------------------------------------------------------

class TestDescriptorIdGrammar:

    def test_canonical_id(self):
        utd = _minimal_utd(descriptor_id="rhea:muscle.align@5.1.0")
        assert utd.descriptor_backend == "rhea"
        assert utd.descriptor_tool_id == "muscle.align"
        assert utd.descriptor_version == "5.1.0"

    def test_native_backend(self):
        utd = _minimal_utd(descriptor_id="native:domain.lookup@1.0.0")
        assert utd.descriptor_backend == "native"
        assert utd.descriptor_tool_id == "domain.lookup"

    def test_galaxy_backend(self):
        utd = _minimal_utd(descriptor_id="galaxy:fastqc@0.12.1")
        assert utd.descriptor_backend == "galaxy"

    def test_loose_version_accepted(self):
        utd = _minimal_utd(descriptor_id="rhea:tool@1.0.0-rc.1+build")
        assert utd.descriptor_version == "1.0.0-rc.1+build"

    def test_uppercase_backend_rejected(self):
        with pytest.raises(Exception) as exc_info:
            _minimal_utd(descriptor_id="Rhea:tool@1.0.0")
        assert "FAIL-FAST" in str(exc_info.value)

    def test_missing_backend_rejected(self):
        with pytest.raises(Exception):
            _minimal_utd(descriptor_id="tool@1.0.0")

    def test_missing_version_rejected(self):
        with pytest.raises(Exception):
            _minimal_utd(descriptor_id="rhea:tool")


# ---------------------------------------------------------------------------
# 3. Auto-resolution of descriptor_hash
# ---------------------------------------------------------------------------

class TestDescriptorHashResolution:

    def test_hash_auto_computed_when_empty(self):
        utd = _minimal_utd(descriptor_hash=None)
        assert utd.descriptor_hash is not None
        assert len(utd.descriptor_hash) == 64

    def test_hash_auto_computed_on_sentinel(self):
        utd = _minimal_utd(descriptor_hash="<computed-at-load>")
        assert utd.descriptor_hash != "<computed-at-load>"
        assert len(utd.descriptor_hash) == 64

    def test_explicit_hash_preserved(self):
        utd = _minimal_utd(descriptor_hash="abc123")
        assert utd.descriptor_hash == "abc123"

    def test_changing_inputs_changes_auto_hash(self):
        utd1 = _minimal_utd(inputs=[
            _build(UTDInputSpec, name="x", type="string"),
        ])
        utd2 = _minimal_utd(inputs=[
            _build(UTDInputSpec, name="x", type="integer"),
        ])
        assert utd1.descriptor_hash != utd2.descriptor_hash

    def test_changing_summary_does_not_change_hash(self):
        """Descriptive fields (summary, long_description, display_name)
        do NOT affect the contract hash. Hash covers behavior only."""
        utd1 = _minimal_utd(summary="a", display_name="A")
        utd2 = _minimal_utd(summary="b", display_name="B")
        assert utd1.descriptor_hash == utd2.descriptor_hash


# ---------------------------------------------------------------------------
# 4. compute_descriptor_hash determinism
# ---------------------------------------------------------------------------

class TestComputeDescriptorHashDeterminism:

    def test_same_inputs_same_hash(self):
        h1 = compute_descriptor_hash(
            descriptor_id="x:y@1.0", inputs=[{"name": "a"}], outputs=[],
            side_effects="none", determinism="R3", resource_class="cpu_light",
        )
        h2 = compute_descriptor_hash(
            descriptor_id="x:y@1.0", inputs=[{"name": "a"}], outputs=[],
            side_effects="none", determinism="R3", resource_class="cpu_light",
        )
        assert h1 == h2

    def test_input_order_does_not_affect_hash(self):
        h1 = compute_descriptor_hash(
            descriptor_id="x:y@1.0",
            inputs=[{"name": "a"}, {"name": "b"}],
            outputs=[], side_effects="none",
            determinism="R3", resource_class="cpu_light",
        )
        h2 = compute_descriptor_hash(
            descriptor_id="x:y@1.0",
            inputs=[{"name": "b"}, {"name": "a"}],
            outputs=[], side_effects="none",
            determinism="R3", resource_class="cpu_light",
        )
        assert h1 == h2

    def test_different_determinism_different_hash(self):
        h1 = compute_descriptor_hash(
            descriptor_id="x:y@1.0", inputs=[], outputs=[],
            side_effects="none", determinism="R1", resource_class="cpu_light",
        )
        h2 = compute_descriptor_hash(
            descriptor_id="x:y@1.0", inputs=[], outputs=[],
            side_effects="none", determinism="R3", resource_class="cpu_light",
        )
        assert h1 != h2


# ---------------------------------------------------------------------------
# 5. ToolBase.from_descriptor
# ---------------------------------------------------------------------------

class TestToolBaseFromDescriptor:

    def test_descriptor_input_validation(self):
        """A non-UTD non-dict argument FAIL-FASTs."""
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor("not a utd")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "expects UnifiedToolDescriptor" in str(exc_info.value)

    def test_dict_with_bad_shape_fails_fast(self):
        """Dict that doesn't match UTD schema FAIL-FASTs."""
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor({"missing": "fields"})
        assert "FAIL-FAST" in str(exc_info.value)
        assert "failed UTD shape" in str(exc_info.value)

    def test_unimportable_class_path_fails_fast(self):
        utd = _minimal_utd(
            provenance_pin=_minimal_pin(class_path="nonexistent.module.NoTool"),
        )
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(utd)
        assert "FAIL-FAST" in str(exc_info.value)
        assert "could not import" in str(exc_info.value)

    def test_non_toolbase_class_fails_fast(self):
        utd = _minimal_utd(
            provenance_pin=_minimal_pin(class_path="builtins.dict"),
        )
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(utd)
        assert "FAIL-FAST" in str(exc_info.value)
        assert "is not a ToolBase subclass" in str(exc_info.value)

    def test_capability_check_passes_when_held(self):
        """When user_capabilities is provided AND covers requires_capability,
        the check passes (the call may still fail later for unrelated
        reasons, but the capability gate doesn't block)."""
        utd = _minimal_utd(
            requires_capability=["cap_a", "cap_b"],
            provenance_pin=_minimal_pin(class_path="builtins.dict"),  # downstream
        )
        # We expect downstream failure (builtins.dict is not ToolBase)
        # but the failure should NOT be a capability error.
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(
                utd, user_capabilities=["cap_a", "cap_b", "extra"])
        assert "is not a ToolBase subclass" in str(exc_info.value)
        assert "capabilities" not in str(exc_info.value).lower()

    def test_capability_check_fails_fast_when_missing(self):
        utd = _minimal_utd(
            requires_capability=["read_phi", "write_audit"],
            provenance_pin=_minimal_pin(class_path="builtins.dict"),
        )
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(
                utd, user_capabilities=["only_one"])
        assert "FAIL-FAST" in str(exc_info.value)
        assert "requires capabilities" in str(exc_info.value)
        assert "read_phi" in str(exc_info.value)
        assert "write_audit" in str(exc_info.value)

    def test_capability_check_skipped_when_not_provided(self):
        """When user_capabilities kwarg is absent, no capability check
        runs — the orchestrator's HITL gate (GATE-A2) handles that."""
        utd = _minimal_utd(
            requires_capability=["sensitive"],
            provenance_pin=_minimal_pin(class_path="builtins.dict"),
        )
        # Should fail on the class-resolution path, NOT capabilities.
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBase.from_descriptor(utd)  # no user_capabilities kwarg
        assert "is not a ToolBase subclass" in str(exc_info.value)
        assert "capabilities" not in str(exc_info.value).lower()
