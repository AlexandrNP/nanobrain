"""Tests for G12 — declarative resource envelope on Step + workflow rollup.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G12``.

Tests cover:
1. ResourceEnvelope shape + extra='forbid' + non-negative bounds
2. aggregate_resource_envelopes per-field rules (sum / max / union)
3. Edge cases: empty input, all-None fields, mixed None + value
4. StepConfig.resource_envelope additive (default None)
"""

from __future__ import annotations

import pytest

from nanobrain.core.step import (
    ResourceEnvelope,
    StepConfig,
    aggregate_resource_envelopes,
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


# ---------------------------------------------------------------------------
# 1. ResourceEnvelope shape
# ---------------------------------------------------------------------------

class TestResourceEnvelopeShape:

    def test_empty(self):
        e = _build(ResourceEnvelope)
        assert e.walltime_minutes is None
        assert e.cpu_cores is None
        assert e.memory_gb is None
        assert e.capability_tokens == []
        assert e.cost_units is None

    def test_full(self):
        e = _build(
            ResourceEnvelope,
            walltime_minutes=10.0,
            cpu_cores=4.0,
            memory_gb=8.0,
            capability_tokens=["read_phi", "write_audit"],
            cost_units=2.5,
        )
        assert e.walltime_minutes == 10.0
        assert e.cpu_cores == 4.0
        assert e.cost_units == 2.5

    def test_negative_walltime_rejected(self):
        with pytest.raises(Exception):
            _build(ResourceEnvelope, walltime_minutes=-1.0)

    def test_negative_cpu_rejected(self):
        with pytest.raises(Exception):
            _build(ResourceEnvelope, cpu_cores=-0.5)

    def test_negative_cost_rejected(self):
        with pytest.raises(Exception):
            _build(ResourceEnvelope, cost_units=-1.0)

    def test_zero_walltime_accepted(self):
        # Zero is the "no-op step" case — legitimate.
        e = _build(ResourceEnvelope, walltime_minutes=0.0)
        assert e.walltime_minutes == 0.0

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            _build(ResourceEnvelope, walltime_minutes=1.0, ghost="bad")


# ---------------------------------------------------------------------------
# 2. aggregate_resource_envelopes — per-field rules
# ---------------------------------------------------------------------------

class TestAggregationRules:

    def test_walltime_summed(self):
        e1 = _build(ResourceEnvelope, walltime_minutes=3.0)
        e2 = _build(ResourceEnvelope, walltime_minutes=5.0)
        e3 = _build(ResourceEnvelope, walltime_minutes=2.0)
        agg = aggregate_resource_envelopes([e1, e2, e3])
        assert agg.walltime_minutes == 10.0

    def test_cost_summed(self):
        e1 = _build(ResourceEnvelope, cost_units=0.10)
        e2 = _build(ResourceEnvelope, cost_units=0.30)
        agg = aggregate_resource_envelopes([e1, e2])
        assert agg.cost_units == pytest.approx(0.40)

    def test_cpu_max(self):
        e1 = _build(ResourceEnvelope, cpu_cores=2.0)
        e2 = _build(ResourceEnvelope, cpu_cores=8.0)
        e3 = _build(ResourceEnvelope, cpu_cores=4.0)
        agg = aggregate_resource_envelopes([e1, e2, e3])
        assert agg.cpu_cores == 8.0

    def test_memory_max(self):
        e1 = _build(ResourceEnvelope, memory_gb=1.0)
        e2 = _build(ResourceEnvelope, memory_gb=16.0)
        agg = aggregate_resource_envelopes([e1, e2])
        assert agg.memory_gb == 16.0

    def test_capability_tokens_union(self):
        e1 = _build(ResourceEnvelope, capability_tokens=["a", "b"])
        e2 = _build(ResourceEnvelope, capability_tokens=["b", "c"])
        e3 = _build(ResourceEnvelope, capability_tokens=["a"])
        agg = aggregate_resource_envelopes([e1, e2, e3])
        assert agg.capability_tokens == ["a", "b", "c"]  # sorted

    def test_capability_tokens_empty_remains_empty(self):
        e1 = _build(ResourceEnvelope)
        e2 = _build(ResourceEnvelope)
        agg = aggregate_resource_envelopes([e1, e2])
        assert agg.capability_tokens == []


# ---------------------------------------------------------------------------
# 3. Edge cases
# ---------------------------------------------------------------------------

class TestAggregationEdgeCases:

    def test_empty_list_returns_empty_envelope(self):
        agg = aggregate_resource_envelopes([])
        assert agg.walltime_minutes is None
        assert agg.cost_units is None
        assert agg.cpu_cores is None
        assert agg.memory_gb is None
        assert agg.capability_tokens == []

    def test_all_none_walltime_remains_none(self):
        e1 = _build(ResourceEnvelope, cpu_cores=2.0)
        e2 = _build(ResourceEnvelope, cpu_cores=4.0)
        agg = aggregate_resource_envelopes([e1, e2])
        assert agg.walltime_minutes is None
        assert agg.cost_units is None

    def test_partial_none_skips_none(self):
        """Mix of None + value: aggregate considers only the non-None
        contributions for that field."""
        e1 = _build(ResourceEnvelope, walltime_minutes=10.0)
        e2 = _build(ResourceEnvelope)  # no walltime declared
        e3 = _build(ResourceEnvelope, walltime_minutes=5.0)
        agg = aggregate_resource_envelopes([e1, e2, e3])
        # Sum of declared values; un-declared step doesn't contribute.
        assert agg.walltime_minutes == 15.0

    def test_single_envelope_passes_through(self):
        e = _build(
            ResourceEnvelope,
            walltime_minutes=10.0, cpu_cores=4.0,
            memory_gb=8.0, cost_units=2.0,
            capability_tokens=["x"],
        )
        agg = aggregate_resource_envelopes([e])
        assert agg.walltime_minutes == 10.0
        assert agg.cpu_cores == 4.0
        assert agg.memory_gb == 8.0
        assert agg.cost_units == 2.0
        assert agg.capability_tokens == ["x"]


# ---------------------------------------------------------------------------
# 4. StepConfig integration — additive default-None
# ---------------------------------------------------------------------------

class TestStepConfigIntegration:

    def test_default_resource_envelope_is_none(self):
        cfg = _build(StepConfig, name="t")
        assert cfg.resource_envelope is None

    def test_explicit_dict_envelope(self):
        cfg = _build(
            StepConfig,
            name="t",
            resource_envelope={
                "walltime_minutes": 10.0,
                "cpu_cores": 2.0,
                "capability_tokens": ["read_phi"],
            },
        )
        # The dict is preserved as-is (not coerced to ResourceEnvelope here);
        # the workflow-side aggregator handles dict → ResourceEnvelope.
        assert isinstance(cfg.resource_envelope, dict)
        assert cfg.resource_envelope["walltime_minutes"] == 10.0
