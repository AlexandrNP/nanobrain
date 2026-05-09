"""Tests for G1 — declarative ConditionalLink predicate DSL.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G1``: a typed
``PredicateConfig`` Pydantic model with the fixed vocabulary
``eq | ne | in | contains | exists | all | any | not``, evaluated by
``evaluate_predicate``. The evaluator FAIL-FASTs on dotted-path miss when
``op != "exists"`` — silent ``None`` returns from path-miss are forbidden
(workspace policy: real failures, not silent passes).

These tests cover:

1. ``PredicateConfig`` shape validation — leaf vs combinator field rules.
2. ``evaluate_predicate`` correctness across the leaf op vocabulary.
3. ``evaluate_predicate`` FAIL-FAST behavior on missing fields.
4. Combinator semantics (``all`` / ``any`` / ``not``).
5. Backwards compatibility — legacy ``field/operator/value`` shape still works
   and emits a deprecation warning.
6. ``parse_condition_from_config`` correctly routes between the new and
   legacy paths based on the presence of the ``op`` key.
7. Pydantic-model payload resolution (not just dicts).
"""

from __future__ import annotations

import logging
from typing import Optional

import pytest
from pydantic import BaseModel

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.link import (
    PredicateConfig,
    evaluate_predicate,
    get_nested_value_strict,
    parse_condition_from_config,
    _PATH_MISS,
)


# ---------------------------------------------------------------------------
# 1. PredicateConfig shape validation
# ---------------------------------------------------------------------------

class TestPredicateConfigValidation:
    """The Pydantic validator must catch invalid shapes at config load time,
    not at evaluation time. This is the FAIL-FAST contract."""

    def _build(self, **kwargs):
        """Helper: bypass FromConfigBase prohibition for test construction."""
        PredicateConfig._allow_direct_instantiation = True
        try:
            return PredicateConfig(**kwargs)
        finally:
            PredicateConfig._allow_direct_instantiation = False

    def test_leaf_eq_minimal(self):
        """eq with field+value is the minimal leaf."""
        pred = self._build(op="eq", field="x", value=1)
        assert pred.op == "eq"
        assert pred.field == "x"
        assert pred.value == 1

    def test_leaf_eq_rejects_of(self):
        """A leaf op cannot have 'of' (combinator field)."""
        with pytest.raises(Exception) as exc_info:
            self._build(op="eq", field="x", value=1, of=[])
        assert "leaf op" in str(exc_info.value)
        assert "'of' must not be set" in str(exc_info.value)

    def test_leaf_requires_field(self):
        """Non-exists leaf ops require 'field'."""
        with pytest.raises(Exception) as exc_info:
            self._build(op="eq", value=1)
        assert "requires 'field'" in str(exc_info.value)

    def test_exists_requires_field(self):
        with pytest.raises(Exception) as exc_info:
            self._build(op="exists")
        assert "requires 'field'" in str(exc_info.value)

    def test_exists_rejects_value(self):
        with pytest.raises(Exception) as exc_info:
            self._build(op="exists", field="x", value=1)
        assert "must not set 'value'" in str(exc_info.value)

    def test_combinator_all_minimal(self):
        sub = self._build(op="eq", field="x", value=1)
        pred = self._build(op="all", of=[sub])
        assert pred.op == "all"
        assert len(pred.of) == 1

    def test_combinator_rejects_field(self):
        with pytest.raises(Exception) as exc_info:
            self._build(op="all", field="x", of=[])
        assert "combinator" in str(exc_info.value)
        assert "'field' and 'value' must not be set" in str(exc_info.value)

    def test_combinator_requires_of(self):
        with pytest.raises(Exception) as exc_info:
            self._build(op="all")
        assert "requires 'of'" in str(exc_info.value)

    def test_combinator_requires_nonempty_of(self):
        with pytest.raises(Exception) as exc_info:
            self._build(op="all", of=[])
        assert "at least one sub-predicate" in str(exc_info.value)

    def test_not_requires_exactly_one(self):
        a = self._build(op="eq", field="x", value=1)
        b = self._build(op="eq", field="y", value=2)
        with pytest.raises(Exception) as exc_info:
            self._build(op="not", of=[a, b])
        assert "exactly one" in str(exc_info.value)

    def test_extra_fields_forbidden(self):
        """ConfigBase extra='forbid' must catch typos."""
        with pytest.raises(Exception):
            self._build(op="eq", field="x", value=1, fielld="typo")

    def test_unknown_op_rejected_by_literal(self):
        """Pydantic's Literal type rejects unknown ops at parse time."""
        with pytest.raises(Exception):
            self._build(op="lt", field="x", value=1)


# ---------------------------------------------------------------------------
# 2-3. evaluate_predicate — leaf ops + FAIL-FAST on miss
# ---------------------------------------------------------------------------

class TestEvaluatePredicateLeaf:

    def _build(self, **kwargs):
        PredicateConfig._allow_direct_instantiation = True
        try:
            return PredicateConfig(**kwargs)
        finally:
            PredicateConfig._allow_direct_instantiation = False

    def test_eq_match(self):
        pred = self._build(op="eq", field="x", value=42)
        assert evaluate_predicate({"x": 42}, pred) is True

    def test_eq_mismatch(self):
        pred = self._build(op="eq", field="x", value=42)
        assert evaluate_predicate({"x": 7}, pred) is False

    def test_ne_match(self):
        pred = self._build(op="ne", field="x", value=42)
        assert evaluate_predicate({"x": 7}, pred) is True

    def test_in_match(self):
        # 'in' = resolved IN value (value is the container)
        pred = self._build(op="in", field="x", value=["a", "b", "c"])
        assert evaluate_predicate({"x": "b"}, pred) is True

    def test_in_mismatch(self):
        pred = self._build(op="in", field="x", value=["a", "b"])
        assert evaluate_predicate({"x": "z"}, pred) is False

    def test_in_non_iterable_value_fails_fast(self):
        pred = self._build(op="in", field="x", value=42)  # int isn't iterable
        with pytest.raises(ComponentConfigurationError) as exc_info:
            evaluate_predicate({"x": 1}, pred)
        assert "must be iterable" in str(exc_info.value)

    def test_contains_match(self):
        # 'contains' = value IN resolved (resolved is the container)
        pred = self._build(op="contains", field="layers", value="structural")
        assert evaluate_predicate({"layers": ["sequence", "structural"]}, pred) is True

    def test_contains_mismatch(self):
        pred = self._build(op="contains", field="layers", value="design")
        assert evaluate_predicate({"layers": ["sequence", "structural"]}, pred) is False

    def test_contains_non_container_fails_fast(self):
        pred = self._build(op="contains", field="x", value="needle")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            evaluate_predicate({"x": 42}, pred)  # int isn't a container for 'in'
        assert "non-container" in str(exc_info.value)

    def test_exists_present(self):
        pred = self._build(op="exists", field="x")
        assert evaluate_predicate({"x": None}, pred) is True

    def test_exists_absent(self):
        pred = self._build(op="exists", field="missing")
        assert evaluate_predicate({"x": 1}, pred) is False

    def test_exists_nested_present(self):
        pred = self._build(op="exists", field="a.b.c")
        assert evaluate_predicate({"a": {"b": {"c": 1}}}, pred) is True

    def test_exists_nested_partial_absent(self):
        pred = self._build(op="exists", field="a.b.c")
        assert evaluate_predicate({"a": {"b": {}}}, pred) is False

    def test_eq_fails_fast_on_missing_field(self):
        """The G1 contract: ``op != 'exists'`` MUST FAIL-FAST on path miss
        rather than silently returning False — that's the silent-failure shape
        the design package was built to eliminate."""
        pred = self._build(op="eq", field="missing.path", value=1)
        with pytest.raises(ComponentConfigurationError) as exc_info:
            evaluate_predicate({"x": 1}, pred)
        assert "FAIL-FAST" in str(exc_info.value)
        assert "missing in payload" in str(exc_info.value)
        assert "use op='exists'" in str(exc_info.value)

    def test_dotted_path_traverses_dicts(self):
        pred = self._build(op="eq", field="a.b.c", value="deep")
        assert evaluate_predicate({"a": {"b": {"c": "deep"}}}, pred) is True

    def test_dotted_path_traverses_pydantic_models(self):
        """Per G1 spec: dotted paths resolve via __getattr__ on objects too."""
        class Inner(BaseModel):
            value: int = 5

        class Outer(BaseModel):
            inner: Inner = Inner()

        pred = self._build(op="eq", field="inner.value", value=5)
        assert evaluate_predicate(Outer(), pred) is True

    def test_mixed_dict_and_object_traversal(self):
        """Dotted path traverses dicts AND objects in one walk."""
        class Box(BaseModel):
            payload: dict = {"k": 7}

        pred = self._build(op="eq", field="payload.k", value=7)
        assert evaluate_predicate(Box(), pred) is True


# ---------------------------------------------------------------------------
# 4. Combinator semantics
# ---------------------------------------------------------------------------

class TestEvaluatePredicateCombinator:

    def _build(self, **kwargs):
        PredicateConfig._allow_direct_instantiation = True
        try:
            return PredicateConfig(**kwargs)
        finally:
            PredicateConfig._allow_direct_instantiation = False

    def test_all_true(self):
        sub_a = self._build(op="eq", field="x", value=1)
        sub_b = self._build(op="eq", field="y", value=2)
        pred = self._build(op="all", of=[sub_a, sub_b])
        assert evaluate_predicate({"x": 1, "y": 2}, pred) is True

    def test_all_short_circuits_false(self):
        sub_a = self._build(op="eq", field="x", value=1)
        sub_b = self._build(op="eq", field="y", value=99)
        pred = self._build(op="all", of=[sub_a, sub_b])
        assert evaluate_predicate({"x": 1, "y": 2}, pred) is False

    def test_any_one_true(self):
        sub_a = self._build(op="eq", field="x", value=99)
        sub_b = self._build(op="eq", field="y", value=2)
        pred = self._build(op="any", of=[sub_a, sub_b])
        assert evaluate_predicate({"x": 1, "y": 2}, pred) is True

    def test_any_all_false(self):
        sub_a = self._build(op="eq", field="x", value=99)
        sub_b = self._build(op="eq", field="y", value=99)
        pred = self._build(op="any", of=[sub_a, sub_b])
        assert evaluate_predicate({"x": 1, "y": 2}, pred) is False

    def test_not_inverts(self):
        sub = self._build(op="eq", field="x", value=99)
        pred = self._build(op="not", of=[sub])
        assert evaluate_predicate({"x": 1}, pred) is True

    def test_combinator_propagates_fail_fast(self):
        """A combinator's sub-predicate's FAIL-FAST must surface."""
        sub_a = self._build(op="eq", field="x", value=1)
        sub_b = self._build(op="eq", field="missing", value=1)
        pred = self._build(op="all", of=[sub_a, sub_b])
        with pytest.raises(ComponentConfigurationError):
            evaluate_predicate({"x": 1}, pred)

    def test_realistic_layered_reasoning_gate(self):
        """The motivating example from `nanobrain_capability_gaps.md G1`:
        gate the structural layer when Phase 0 selected it AND the user
        opted into the heavier accessibility computation."""
        sub_layer = self._build(op="contains", field="active_layers", value="structural")
        sub_opt = self._build(
            op="eq",
            field="layer_options.structural.compute_accessibility",
            value=True,
        )
        pred = self._build(op="all", of=[sub_layer, sub_opt])

        # Both true:
        plan_active = {
            "active_layers": ["sequence", "structural"],
            "layer_options": {"structural": {"compute_accessibility": True}},
        }
        assert evaluate_predicate(plan_active, pred) is True

        # Layer present but option false:
        plan_layer_only = {
            "active_layers": ["structural"],
            "layer_options": {"structural": {"compute_accessibility": False}},
        }
        assert evaluate_predicate(plan_layer_only, pred) is False

        # Layer absent — short-circuits to False before evaluating the opt:
        plan_no_layer = {
            "active_layers": ["sequence"],
            "layer_options": {"structural": {"compute_accessibility": True}},
        }
        assert evaluate_predicate(plan_no_layer, pred) is False


# ---------------------------------------------------------------------------
# 5-6. parse_condition_from_config — routing + backward compatibility
# ---------------------------------------------------------------------------

class TestParseConditionRouting:
    """``parse_condition_from_config`` is the public entry point used by
    ``ConditionalLink.resolve_dependencies``. It must route to the new
    G1 evaluator when the config has ``op:`` and otherwise preserve
    legacy behavior (with a deprecation warning)."""

    def test_routes_g1_dict_to_new_evaluator(self):
        cond = {"op": "eq", "field": "x", "value": 42}
        func = parse_condition_from_config(cond)
        assert func({"x": 42}) is True
        assert func({"x": 7}) is False

    def test_g1_dict_validation_error_fails_fast(self):
        """A G1-shaped dict that fails validation must FAIL-FAST,
        not fall through to the legacy path."""
        with pytest.raises(ComponentConfigurationError) as exc_info:
            parse_condition_from_config({"op": "eq", "value": 42})  # missing 'field'
        assert "FAIL-FAST" in str(exc_info.value)

    def test_routes_legacy_dict_to_legacy_evaluator(self, caplog):
        """Legacy field/operator/value shape still works, with a
        deprecation warning."""
        cond = {"field": "x", "operator": "equals", "value": 42}
        with caplog.at_level(logging.WARNING):
            func = parse_condition_from_config(cond)
        assert any("deprecated" in rec.message.lower() for rec in caplog.records)
        assert func({"x": 42}) is True
        assert func({"x": 7}) is False

    def test_routes_legacy_string_to_legacy_evaluator(self, caplog):
        with caplog.at_level(logging.WARNING):
            func = parse_condition_from_config("hello")
        assert any("deprecated" in rec.message.lower() for rec in caplog.records)
        assert func("hello world") is True
        assert func("nope") is False

    def test_accepts_prebuilt_predicate_config(self):
        PredicateConfig._allow_direct_instantiation = True
        try:
            pred = PredicateConfig(op="eq", field="x", value=1)
        finally:
            PredicateConfig._allow_direct_instantiation = False
        func = parse_condition_from_config(pred)
        assert func({"x": 1}) is True


# ---------------------------------------------------------------------------
# 7. get_nested_value_strict — the underlying resolver
# ---------------------------------------------------------------------------

class TestGetNestedValueStrict:

    def test_dict_path(self):
        assert get_nested_value_strict({"a": {"b": 1}}, "a.b") == 1

    def test_object_attr_path(self):
        class O:
            x = 5
        assert get_nested_value_strict(O(), "x") == 5

    def test_dict_miss_returns_sentinel(self):
        assert get_nested_value_strict({"a": 1}, "missing") is _PATH_MISS

    def test_object_miss_returns_sentinel(self):
        class O:
            pass
        assert get_nested_value_strict(O(), "missing") is _PATH_MISS

    def test_partial_dict_miss(self):
        assert get_nested_value_strict({"a": {"b": 1}}, "a.missing") is _PATH_MISS

    def test_empty_path_returns_root(self):
        assert get_nested_value_strict({"a": 1}, "") == {"a": 1}

    def test_resolved_none_is_not_miss(self):
        """A field that is present and explicitly None must be distinguishable
        from a missing field. This is the core motivation for the sentinel."""
        result = get_nested_value_strict({"a": None}, "a")
        assert result is None
        assert result is not _PATH_MISS
