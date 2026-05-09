"""Tests for G6 — typed step input/output schemas + reserved-fields escape valve.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G6``: every
step's `process()` may declare an input + output schema. The framework
validates payloads at the wire boundary. Reserved fields (`errors`,
`partial`) are admitted alongside the typed payload per the escape valve.

These tests cover:
1. SchemaRef validation (one-of class/json_schema, extra='forbid')
2. validate_payload_against_schema with Pydantic class path
3. validate_payload_against_schema with inline JSON Schema
4. Reserved fields escape valve (errors + partial admitted)
5. FAIL-FAST on schema mismatch (input + output)
6. End-to-end through a minimal BaseStep subclass
"""

from __future__ import annotations

from typing import Optional

import pytest
from pydantic import BaseModel

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.step import (
    SchemaRef,
    StepError,
    validate_payload_against_schema,
    RESERVED_OUTPUT_FIELDS,
)


# ---------------------------------------------------------------------------
# Module-level Pydantic models used as schema targets.
# Defined at module scope so import_class_from_path can find them.
# ---------------------------------------------------------------------------

class _MyInputSchema(BaseModel):
    """Minimal input schema for tests."""
    model_config = {"extra": "forbid"}
    query: str
    limit: int = 10


class _MyOutputSchema(BaseModel):
    """Minimal output schema for tests."""
    model_config = {"extra": "forbid"}
    answer: str
    score: float


# ---------------------------------------------------------------------------
# 1. SchemaRef validation
# ---------------------------------------------------------------------------

class TestSchemaRefValidation:

    def _build(self, **kwargs):
        SchemaRef._allow_direct_instantiation = True
        try:
            return SchemaRef(**kwargs)
        finally:
            SchemaRef._allow_direct_instantiation = False

    def test_class_path_only(self):
        ref = self._build(**{"class": "x.y.Z"})
        assert ref.class_field == "x.y.Z"
        assert ref.json_schema is None

    def test_json_schema_only(self):
        ref = self._build(json_schema={"type": "object"})
        assert ref.json_schema == {"type": "object"}
        assert ref.class_field is None

    def test_neither_field_fails(self):
        with pytest.raises(Exception) as exc_info:
            self._build()
        assert "EXACTLY ONE" in str(exc_info.value)

    def test_both_fields_fail(self):
        with pytest.raises(Exception) as exc_info:
            self._build(**{"class": "x.y.Z", "json_schema": {"type": "object"}})
        assert "EXACTLY ONE" in str(exc_info.value)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            self._build(**{"class": "x.y.Z", "extra_typo": "bad"})

    def test_validate_on_set_default_false(self):
        ref = self._build(**{"class": "x.y.Z"})
        assert ref.validate_on_set is False


# ---------------------------------------------------------------------------
# 2. Pydantic class path validation
# ---------------------------------------------------------------------------

class TestPydanticClassValidation:

    def _ref(self, dotted_path: str) -> SchemaRef:
        SchemaRef._allow_direct_instantiation = True
        try:
            return SchemaRef(**{"class": dotted_path})
        finally:
            SchemaRef._allow_direct_instantiation = False

    def test_valid_input_passes(self):
        ref = self._ref("tests.unit.test_step_schemas._MyInputSchema")
        validate_payload_against_schema(
            {"query": "x", "limit": 5}, ref,
            component_name="test_step", direction="input",
        )

    def test_extra_field_rejected(self):
        ref = self._ref("tests.unit.test_step_schemas._MyInputSchema")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"query": "x", "limit": 5, "ghost": True}, ref,
                component_name="test_step", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "test_step" in str(exc_info.value)
        assert "input" in str(exc_info.value)

    def test_missing_required_rejected(self):
        ref = self._ref("tests.unit.test_step_schemas._MyInputSchema")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"limit": 5}, ref,
                component_name="test_step", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)

    def test_unimportable_class_fails_fast(self):
        ref = self._ref("nonexistent.module.NoSuchClass")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"x": 1}, ref,
                component_name="test_step", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "failed to import" in str(exc_info.value)

    def test_non_pydantic_class_fails_fast(self):
        # Use a builtin that's not a Pydantic model:
        ref = self._ref("builtins.dict")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"x": 1}, ref,
                component_name="test_step", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "not a Pydantic model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Inline JSON Schema validation
# ---------------------------------------------------------------------------

class TestJsonSchemaValidation:

    def _ref(self, json_schema: dict) -> SchemaRef:
        SchemaRef._allow_direct_instantiation = True
        try:
            return SchemaRef(json_schema=json_schema)
        finally:
            SchemaRef._allow_direct_instantiation = False

    def test_valid_passes(self):
        # jsonschema may not be installed — skip gracefully if so.
        pytest.importorskip("jsonschema")
        ref = self._ref({
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        })
        validate_payload_against_schema(
            {"query": "x"}, ref,
            component_name="t", direction="input",
        )

    def test_missing_required_rejected(self):
        pytest.importorskip("jsonschema")
        ref = self._ref({
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string"}},
        })
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {}, ref,
                component_name="t", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)

    def test_jsonschema_missing_explains_install(self, monkeypatch):
        """When jsonschema package is unavailable, the validator must
        emit a helpful FAIL-FAST naming the install command. We simulate
        the missing package by stubbing the import."""
        import builtins
        original_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("simulated missing jsonschema")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", bad_import)

        ref = self._ref({"type": "object"})
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {}, ref,
                component_name="t", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "pip install jsonschema" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Reserved fields escape valve (errors + partial)
# ---------------------------------------------------------------------------

class TestReservedFieldsEscapeValve:

    def _ref_class(self, dotted_path: str) -> SchemaRef:
        SchemaRef._allow_direct_instantiation = True
        try:
            return SchemaRef(**{"class": dotted_path})
        finally:
            SchemaRef._allow_direct_instantiation = False

    def test_errors_field_admitted_in_output(self):
        """A step whose schema is _MyOutputSchema may still return
        `errors: list[StepError]` alongside the typed payload."""
        ref = self._ref_class("tests.unit.test_step_schemas._MyOutputSchema")
        validate_payload_against_schema(
            {
                "answer": "ok",
                "score": 0.9,
                "errors": [{"code": "WARN", "detail": "minor", "source": "x"}],
            },
            ref,
            component_name="t", direction="output",
        )

    def test_partial_field_admitted_in_output(self):
        ref = self._ref_class("tests.unit.test_step_schemas._MyOutputSchema")
        validate_payload_against_schema(
            {"answer": "ok", "score": 0.9, "partial": True},
            ref,
            component_name="t", direction="output",
        )

    def test_errors_field_validated_against_steperror_shape(self):
        ref = self._ref_class("tests.unit.test_step_schemas._MyOutputSchema")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {
                    "answer": "ok", "score": 0.9,
                    "errors": [{"code": "X", "extra_typo": "bad"}],  # no 'detail'
                },
                ref,
                component_name="t", direction="output",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "errors[0]" in str(exc_info.value)

    def test_errors_field_must_be_list(self):
        ref = self._ref_class("tests.unit.test_step_schemas._MyOutputSchema")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"answer": "ok", "score": 0.9, "errors": "not-a-list"},
                ref,
                component_name="t", direction="output",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "must be a list" in str(exc_info.value)

    def test_partial_must_be_bool(self):
        ref = self._ref_class("tests.unit.test_step_schemas._MyOutputSchema")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"answer": "ok", "score": 0.9, "partial": "yes"},
                ref,
                component_name="t", direction="output",
            )
        assert "FAIL-FAST" in str(exc_info.value)
        assert "must be bool" in str(exc_info.value)

    def test_reserved_fields_NOT_admitted_in_input(self):
        """Input direction does NOT have the escape valve. A schema with
        extra='forbid' rejects 'errors' and 'partial' on input."""
        ref = self._ref_class("tests.unit.test_step_schemas._MyInputSchema")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            validate_payload_against_schema(
                {"query": "x", "errors": []},
                ref,
                component_name="t", direction="input",
            )
        assert "FAIL-FAST" in str(exc_info.value)

    def test_errors_only_output_skips_user_schema(self):
        """A step that returns ONLY reserved fields (e.g., total failure
        with no typed payload) must still validate cleanly."""
        ref = self._ref_class("tests.unit.test_step_schemas._MyOutputSchema")
        validate_payload_against_schema(
            {"errors": [{"code": "FAIL", "detail": "total"}], "partial": True},
            ref,
            component_name="t", direction="output",
        )


# ---------------------------------------------------------------------------
# 5. End-to-end: BaseStep subclass with declared schemas
# ---------------------------------------------------------------------------

class _MinimalStepBase:
    """Minimal stand-in for BaseStep's _execute_process call shape — we
    don't want to spin up a full step with executor + data units for
    schema tests. The validator is a pure function; we test it directly."""


class TestEndToEndValidation:
    """Sanity-check that the validator integrates with the wire boundary
    in the same shape that BaseStep._execute_process uses."""

    def test_input_then_output_round_trip(self):
        SchemaRef._allow_direct_instantiation = True
        try:
            in_ref = SchemaRef(**{
                "class": "tests.unit.test_step_schemas._MyInputSchema"})
            out_ref = SchemaRef(**{
                "class": "tests.unit.test_step_schemas._MyOutputSchema"})
        finally:
            SchemaRef._allow_direct_instantiation = False

        # Simulate a step's full validation lifecycle:
        input_payload = {"query": "hello", "limit": 5}
        validate_payload_against_schema(
            input_payload, in_ref, component_name="t", direction="input")

        # Step "produces" its output:
        output_payload = {"answer": "hi", "score": 0.95}
        validate_payload_against_schema(
            output_payload, out_ref, component_name="t", direction="output")

    def test_output_with_partial_and_errors(self):
        """Realistic shape: a partial result with an errors list."""
        SchemaRef._allow_direct_instantiation = True
        try:
            out_ref = SchemaRef(**{
                "class": "tests.unit.test_step_schemas._MyOutputSchema"})
        finally:
            SchemaRef._allow_direct_instantiation = False

        validate_payload_against_schema(
            {
                "answer": "best-effort",
                "score": 0.42,
                "partial": True,
                "errors": [
                    {"code": "TIMEOUT", "detail": "tool A timed out", "source": "tool_a"},
                    {"code": "RATE_LIMIT", "detail": "tool B rate-limited", "source": "tool_b"},
                ],
            },
            out_ref,
            component_name="real_step", direction="output",
        )
