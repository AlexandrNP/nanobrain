"""Tests for G11 — ToolExecutionStep tool-step taxonomy.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G11``:
ToolExecutionStep is a BaseStep that consumes a UTD reference (G15) and
dispatches to a registered backend adapter.

Tests cover:
1. ToolBackendRegistry — register, lookup, unregister, list, double-register
2. ToolExecutionStepConfig — exactly-one-of validation
3. ToolExecutionStep — UTD resolution at init (inline + path)
4. ToolExecutionStep — process() dispatches to the right adapter
5. ToolExecutionStep — G13 run-context namespace propagation
6. ToolExecutionStep — backend kwargs merging
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.orchestration import WorkflowRunContext
from nanobrain.library.steps import (
    ToolBackendAdapter,
    ToolBackendRegistry,
    ToolExecutionStep,
    ToolExecutionStepConfig,
)


# ---------------------------------------------------------------------------
# Test backend adapter — captures invocation arguments for assertions.
# ---------------------------------------------------------------------------

class _RecordingBackend(ToolBackendAdapter):
    """Test adapter that records every invoke() call and returns a fixed
    result. Each test instantiates with a unique BACKEND_NAME (uuid hex
    suffix) so concurrent tests don't trample the global registry."""

    def __init__(self, name: str, fixed_result: Dict[str, Any] | None = None):
        self.BACKEND_NAME = name
        self._fixed_result = fixed_result if fixed_result is not None else {"result": "ok"}
        self.calls: list[Dict[str, Any]] = []

    async def invoke(self, utd, inputs, *, run_context_namespace="", **kwargs):
        self.calls.append({
            "utd_id": utd.descriptor_id,
            "inputs": inputs,
            "run_context_namespace": run_context_namespace,
            "kwargs": kwargs,
        })
        return dict(self._fixed_result)


def _unique_backend_name() -> str:
    return f"testbackend_{uuid.uuid4().hex[:8]}"


def _build_minimal_utd_dict(backend: str) -> Dict[str, Any]:
    return {
        "descriptor_id": f"{backend}:test_tool@1.0.0",
        "display_name": "Test Tool",
        "summary": "test only",
        "provenance_pin": {"class_path": "builtins.dict"},
    }


# ---------------------------------------------------------------------------
# 1. ToolBackendRegistry
# ---------------------------------------------------------------------------

class TestToolBackendRegistry:

    def test_register_and_lookup(self):
        backend_name = _unique_backend_name()
        adapter = _RecordingBackend(backend_name)
        ToolBackendRegistry.register(adapter)
        try:
            assert ToolBackendRegistry.get(backend_name) is adapter
            assert backend_name in ToolBackendRegistry.list_backends()
        finally:
            ToolBackendRegistry.unregister(backend_name)

    def test_lookup_missing_raises_keyerror(self):
        with pytest.raises(KeyError) as exc_info:
            ToolBackendRegistry.get("absolutely_does_not_exist_xyz_123")
        assert "FAIL-FAST" in str(exc_info.value)
        assert "no ToolBackendAdapter" in str(exc_info.value)

    def test_register_blank_name_fails_fast(self):
        adapter = _RecordingBackend("")
        with pytest.raises(ComponentConfigurationError) as exc_info:
            ToolBackendRegistry.register(adapter)
        assert "FAIL-FAST" in str(exc_info.value)
        assert "BACKEND_NAME" in str(exc_info.value)

    def test_register_same_name_different_adapter_fails_fast(self):
        backend_name = _unique_backend_name()
        a1 = _RecordingBackend(backend_name)
        a2 = _RecordingBackend(backend_name)
        ToolBackendRegistry.register(a1)
        try:
            with pytest.raises(ComponentConfigurationError) as exc_info:
                ToolBackendRegistry.register(a2)
            assert "FAIL-FAST" in str(exc_info.value)
            assert "already registered" in str(exc_info.value)
        finally:
            ToolBackendRegistry.unregister(backend_name)

    def test_register_same_adapter_idempotent(self):
        backend_name = _unique_backend_name()
        adapter = _RecordingBackend(backend_name)
        ToolBackendRegistry.register(adapter)
        try:
            # Second registration of the SAME instance is OK.
            ToolBackendRegistry.register(adapter)
            assert ToolBackendRegistry.get(backend_name) is adapter
        finally:
            ToolBackendRegistry.unregister(backend_name)


# ---------------------------------------------------------------------------
# 2. ToolExecutionStepConfig
# ---------------------------------------------------------------------------

class TestToolExecutionStepConfig:

    def _build(self, **kwargs):
        ToolExecutionStepConfig._allow_direct_instantiation = True
        try:
            return ToolExecutionStepConfig(**kwargs)
        finally:
            ToolExecutionStepConfig._allow_direct_instantiation = False

    def test_inline_descriptor_only_accepted(self):
        cfg = self._build(
            name="t",
            tool_descriptor=_build_minimal_utd_dict("testbackend"),
        )
        assert cfg.tool_descriptor is not None
        assert cfg.tool_descriptor_path is None

    def test_path_only_accepted(self):
        cfg = self._build(
            name="t",
            tool_descriptor_path="/tmp/some.yml",
        )
        assert cfg.tool_descriptor is None
        assert cfg.tool_descriptor_path == "/tmp/some.yml"

    def test_neither_fails(self):
        with pytest.raises(Exception) as exc_info:
            self._build(name="t")
        assert "EXACTLY ONE" in str(exc_info.value)

    def test_both_fails(self):
        with pytest.raises(Exception) as exc_info:
            self._build(
                name="t",
                tool_descriptor=_build_minimal_utd_dict("x"),
                tool_descriptor_path="/tmp/some.yml",
            )
        assert "EXACTLY ONE" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. ToolExecutionStep — UTD resolution at init
# ---------------------------------------------------------------------------

def _build_step(yaml_dict: Dict[str, Any]) -> ToolExecutionStep:
    """Write to tmp YAML and load via from_config."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write(yaml.safe_dump(yaml_dict))
        path = f.name
    return ToolExecutionStep.from_config(path)


class TestUTDResolutionAtInit:

    def test_inline_utd(self):
        step = _build_step({
            "name": "step_inline",
            "tool_descriptor": _build_minimal_utd_dict("inline_test"),
        })
        assert step.utd.descriptor_id == "inline_test:test_tool@1.0.0"
        assert step.backend_name == "inline_test"

    def test_path_utd(self):
        # Write UTD YAML to a tmp file, then build step pointing at it.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yaml.safe_dump(_build_minimal_utd_dict("pathtest")))
            utd_path = f.name
        step = _build_step({
            "name": "step_path",
            "tool_descriptor_path": utd_path,
        })
        assert step.utd.descriptor_id == "pathtest:test_tool@1.0.0"

    def test_inline_utd_bad_shape_fails_fast(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _build_step({
                "name": "bad",
                "tool_descriptor": {"missing": "fields"},
            })
        assert "FAIL-FAST" in str(exc_info.value)
        assert "failed UTD shape" in str(exc_info.value)

    def test_path_utd_missing_file_fails_fast(self):
        with pytest.raises(ComponentConfigurationError) as exc_info:
            _build_step({
                "name": "bad",
                "tool_descriptor_path": "/nonexistent/path/to/utd.yml",
            })
        assert "FAIL-FAST" in str(exc_info.value)
        assert "not found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. ToolExecutionStep — process() dispatches to the right adapter
# ---------------------------------------------------------------------------

class TestProcessDispatch:

    def test_basic_dispatch(self):
        async def run():
            backend_name = _unique_backend_name()
            adapter = _RecordingBackend(
                backend_name,
                fixed_result={"out_a": "value_a", "out_b": 42},
            )
            ToolBackendRegistry.register(adapter)
            try:
                step = _build_step({
                    "name": "dispatch_test",
                    "tool_descriptor": _build_minimal_utd_dict(backend_name),
                })
                result = await step.process({"in_x": "v1", "in_y": 10})
                assert result == {"out_a": "value_a", "out_b": 42}
                assert len(adapter.calls) == 1
                call = adapter.calls[0]
                assert call["inputs"] == {"in_x": "v1", "in_y": 10}
                assert call["utd_id"] == f"{backend_name}:test_tool@1.0.0"
            finally:
                ToolBackendRegistry.unregister(backend_name)
        asyncio.run(run())

    def test_dispatch_to_unregistered_backend_raises(self):
        async def run():
            # Build step pointing at a backend that does NOT exist:
            step = _build_step({
                "name": "no_backend",
                "tool_descriptor": _build_minimal_utd_dict("definitely_not_registered"),
            })
            with pytest.raises(KeyError) as exc_info:
                await step.process({"x": 1})
            assert "FAIL-FAST" in str(exc_info.value)
            assert "no ToolBackendAdapter" in str(exc_info.value)
        asyncio.run(run())

    def test_non_dict_input_fails_fast(self):
        async def run():
            backend_name = _unique_backend_name()
            adapter = _RecordingBackend(backend_name)
            ToolBackendRegistry.register(adapter)
            try:
                step = _build_step({
                    "name": "non_dict_input",
                    "tool_descriptor": _build_minimal_utd_dict(backend_name),
                })
                with pytest.raises(ComponentConfigurationError) as exc_info:
                    await step.process("not a dict")
                assert "FAIL-FAST" in str(exc_info.value)
                assert "must be dict" in str(exc_info.value)
            finally:
                ToolBackendRegistry.unregister(backend_name)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. G13 run-context namespace propagation
# ---------------------------------------------------------------------------

class TestRunContextNamespacePropagation:

    def test_no_active_context_empty_namespace(self):
        async def run():
            backend_name = _unique_backend_name()
            adapter = _RecordingBackend(backend_name)
            ToolBackendRegistry.register(adapter)
            try:
                step = _build_step({
                    "name": "ns_test_no_ctx",
                    "tool_descriptor": _build_minimal_utd_dict(backend_name),
                })
                await step.process({"x": 1})
                assert adapter.calls[0]["run_context_namespace"] == ""
            finally:
                ToolBackendRegistry.unregister(backend_name)
        asyncio.run(run())

    def test_active_context_namespace_propagates(self):
        async def run():
            backend_name = _unique_backend_name()
            adapter = _RecordingBackend(backend_name)
            ToolBackendRegistry.register(adapter)
            try:
                step = _build_step({
                    "name": "ns_test_with_ctx",
                    "tool_descriptor": _build_minimal_utd_dict(backend_name),
                })
                ctx = WorkflowRunContext.from_config({"run_id": "tenant_xyz"})
                with ctx.activate():
                    await step.process({"x": 1})
                assert adapter.calls[0]["run_context_namespace"] == "run_tenant_xyz"
            finally:
                ToolBackendRegistry.unregister(backend_name)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Backend kwargs merging
# ---------------------------------------------------------------------------

class TestBackendKwargsMerging:

    def test_step_level_kwargs_passed(self):
        async def run():
            backend_name = _unique_backend_name()
            adapter = _RecordingBackend(backend_name)
            ToolBackendRegistry.register(adapter)
            try:
                step = _build_step({
                    "name": "kwargs_test",
                    "tool_descriptor": _build_minimal_utd_dict(backend_name),
                    "backend_kwargs": {"timeout_seconds": 60, "retries": 3},
                })
                await step.process({"x": 1})
                kwargs = adapter.calls[0]["kwargs"]
                assert kwargs["timeout_seconds"] == 60
                assert kwargs["retries"] == 3
            finally:
                ToolBackendRegistry.unregister(backend_name)
        asyncio.run(run())

    def test_per_call_kwargs_override_step_level(self):
        async def run():
            backend_name = _unique_backend_name()
            adapter = _RecordingBackend(backend_name)
            ToolBackendRegistry.register(adapter)
            try:
                step = _build_step({
                    "name": "kwargs_override",
                    "tool_descriptor": _build_minimal_utd_dict(backend_name),
                    "backend_kwargs": {"timeout_seconds": 60},
                })
                # Per-call kwargs should win:
                await step.process({"x": 1}, timeout_seconds=10, extra="added")
                kwargs = adapter.calls[0]["kwargs"]
                assert kwargs["timeout_seconds"] == 10
                assert kwargs["extra"] == "added"
            finally:
                ToolBackendRegistry.unregister(backend_name)
        asyncio.run(run())
