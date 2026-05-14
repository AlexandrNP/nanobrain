"""Unit tests for GlobusComputeExecutor (G22 / G24).

Unconditional — no network, no real Globus endpoint, no auth. FAIL-LOUD
paths are exercised by monkeypatching the ``globus_compute_sdk`` import and
by feeding malformed configs; the dispatch approach (B) is exercised
IN-PROCESS via the module-level ``_run_step_on_endpoint`` worker against a
real trivial step — proving config-reconstruction works without Globus.

Covers:
  * ``GlobusComputeConfig`` validates; ``extra='forbid'`` rejects typos;
    ``endpoint_id`` is required.
  * ``GlobusComputeExecutor`` loads via ``from_config`` (path + object).
  * FAIL-LOUD: missing ``globus_compute`` block.
  * FAIL-LOUD: missing ``globus_compute_sdk`` at ``initialize()``.
  * FAIL-LOUD: ``execute()`` with a non-step closure / missing config path.
  * approach (B): ``_run_step_on_endpoint`` reconstructs a real step from
    config and runs it; a bad class/path is reported in the error envelope.
"""

from __future__ import annotations

import asyncio
import builtins
import sys
import tempfile
from pathlib import Path

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.distributed.globus_compute_executor import (
    GlobusComputeConfig,
    GlobusComputeExecutor,
    _run_step_on_endpoint,
)
from nanobrain.core.executor import ExecutorConfig, ExecutorType

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
_ECHO_STEP_YML = str(_FIXTURE_DIR / "trivial_echo_step.yml")
_ECHO_STEP_CLASS = "tests.fixtures.trivial_echo_step.TrivialEchoStep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_executor_yaml(globus_compute_block: dict | None) -> str:
    """Write an ExecutorConfig YAML with a globus_compute block; return path."""
    lines = ["executor_type: globus_compute", "name: gce_test"]
    if globus_compute_block is not None:
        lines.append("globus_compute:")
        for k, v in globus_compute_block.items():
            lines.append(f"  {k}: {v!r}")
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".yml", delete=False, encoding="utf-8"
    )
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# GlobusComputeConfig validation
# ---------------------------------------------------------------------------
def test_config_validates_minimal():
    cfg = GlobusComputeConfig(endpoint_id="abc-123")
    assert cfg.endpoint_id == "abc-123"
    assert cfg.auth_mode == "client_credentials"
    assert cfg.task_timeout_seconds == 3600.0


def test_config_requires_endpoint_id():
    with pytest.raises(Exception):  # pydantic ValidationError
        GlobusComputeConfig()


def test_config_extra_forbid_rejects_typo():
    with pytest.raises(Exception) as exc:
        GlobusComputeConfig(endpoint_id="x", endpoint_idd="typo")
    # pydantic flags the unknown field
    assert "endpoint_idd" in str(exc.value) or "extra" in str(exc.value).lower()


def test_config_rejects_bad_auth_mode():
    with pytest.raises(Exception):
        GlobusComputeConfig(endpoint_id="x", auth_mode="not_a_mode")


def test_config_from_executor_config_missing_block_fails_loud():
    exec_cfg = ExecutorConfig.from_config(_write_executor_yaml(None))
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusComputeConfig.from_executor_config(exec_cfg)
    assert "FAIL-FAST" in str(exc.value)
    assert "globus_compute" in str(exc.value)


def test_config_from_executor_config_invalid_block_fails_loud():
    exec_cfg = ExecutorConfig.from_config(
        _write_executor_yaml({"bogus_field": "v"})
    )
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusComputeConfig.from_executor_config(exec_cfg)
    assert "FAIL-FAST" in str(exc.value)


# ---------------------------------------------------------------------------
# GlobusComputeExecutor.from_config
# ---------------------------------------------------------------------------
def test_executor_loads_from_config_path():
    path = _write_executor_yaml({"endpoint_id": "endpoint-uuid-1"})
    executor = GlobusComputeExecutor.from_config(path)
    assert isinstance(executor, GlobusComputeExecutor)
    assert executor.globus_compute_config.endpoint_id == "endpoint-uuid-1"
    assert not executor.is_initialized  # no network at construction


def test_executor_loads_from_executor_config_object():
    path = _write_executor_yaml({"endpoint_id": "endpoint-uuid-2"})
    exec_cfg = ExecutorConfig.from_config(path)
    executor = GlobusComputeExecutor.from_config(exec_cfg)
    assert executor.globus_compute_config.endpoint_id == "endpoint-uuid-2"


def test_executor_direct_construction_forbidden():
    with pytest.raises(RuntimeError) as exc:
        GlobusComputeExecutor()
    assert "Direct instantiation" in str(exc.value)


def test_executor_missing_globus_compute_block_fails_loud():
    path = _write_executor_yaml(None)
    with pytest.raises(ComponentConfigurationError) as exc:
        GlobusComputeExecutor.from_config(path)
    assert "FAIL-FAST" in str(exc.value)


def test_executor_bad_config_type_fails():
    with pytest.raises(ValueError) as exc:
        GlobusComputeExecutor.from_config(12345)
    assert "Invalid config type" in str(exc.value)


# ---------------------------------------------------------------------------
# initialize() FAIL-LOUD: globus_compute_sdk missing
# ---------------------------------------------------------------------------
def test_initialize_missing_globus_compute_sdk_fails_loud(monkeypatch):
    path = _write_executor_yaml({"endpoint_id": "endpoint-uuid-3"})
    executor = GlobusComputeExecutor.from_config(path)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "globus_compute_sdk" or name.startswith("globus_compute_sdk."):
            raise ImportError("simulated: globus_compute_sdk not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ComponentConfigurationError) as exc:
        asyncio.run(executor.initialize())
    msg = str(exc.value)
    assert "FAIL-FAST" in msg
    assert "globus_compute_sdk" in msg
    assert "pip install" in msg


# ---------------------------------------------------------------------------
# execute() FAIL-LOUD: non-step closure / missing config path
# ---------------------------------------------------------------------------
def test_execute_non_step_closure_fails_loud(monkeypatch):
    path = _write_executor_yaml({"endpoint_id": "endpoint-uuid-4"})
    executor = GlobusComputeExecutor.from_config(path)
    # Pretend it is initialized so execute() reaches the closure-introspect.
    executor._is_initialized = True

    def _arbitrary_callable():
        return 42

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(executor.execute(_arbitrary_callable))
    assert "cannot extract a step instance" in str(exc.value)


def test_execute_step_without_config_path_fails_loud():
    path = _write_executor_yaml({"endpoint_id": "endpoint-uuid-5"})
    executor = GlobusComputeExecutor.from_config(path)
    executor._is_initialized = True

    # Build a fake "step" object: has .config and .process but no
    # _config_path and config has no source_path.
    class _FakeConfig:
        source_path = None

    class _FakeStep:
        config = _FakeConfig()

        def process(self, input_data):  # noqa: D401 - test stub
            return input_data

    fake_step = _FakeStep()
    input_data = {"k": "v"}

    def _wrapper():
        # closure captures fake_step + input_data, like execute_wrapper
        return (fake_step, input_data)

    with pytest.raises(RuntimeError) as exc:
        asyncio.run(executor.execute(_wrapper))
    msg = str(exc.value)
    assert "_config_path" in msg
    assert "source_path" in msg


def test_extract_step_and_input_finds_both():
    """The closure-introspection mirrors ParslExecutor's __closure__ walk."""

    class _FakeConfig:
        source_path = "/some/path.yml"

    class _FakeStep:
        config = _FakeConfig()

        def process(self, input_data):
            return input_data

    step = _FakeStep()
    payload = {"hello": "world"}

    def _wrapper():
        return (step, payload)

    found_step, found_input = GlobusComputeExecutor._extract_step_and_input(
        _wrapper
    )
    assert found_step is step
    assert found_input == payload


# ---------------------------------------------------------------------------
# Dispatch approach (B): _run_step_on_endpoint reconstructs from config.
# This is the load-bearing proof that approach (B) works WITHOUT Globus.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _ensure_tests_on_path():
    """The fixture step is imported as ``tests.fixtures.trivial_echo_step``."""
    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    yield


def test_run_step_on_endpoint_reconstructs_and_runs_real_step():
    """approach (B): worker reconstructs a real step from config + runs it."""
    envelope = _run_step_on_endpoint(
        _ECHO_STEP_YML, _ECHO_STEP_CLASS, {"value": "hello"}
    )
    assert envelope["status"] == "success"
    # TrivialEchoStep uppercases a string payload, unwrapping a 1-key dict.
    assert envelope["result"] == {"echoed": "HELLO"}
    assert "worker_node" in envelope
    assert "worker_pid" in envelope


def test_run_step_on_endpoint_bad_class_reports_error_envelope():
    envelope = _run_step_on_endpoint(
        _ECHO_STEP_YML, "tests.fixtures.trivial_echo_step.NoSuchStep", {}
    )
    assert envelope["status"] == "error"
    assert "traceback" in envelope
    assert envelope["error"]  # non-empty


def test_run_step_on_endpoint_bad_config_path_reports_error_envelope():
    envelope = _run_step_on_endpoint(
        "/nonexistent/path/to/step.yml", _ECHO_STEP_CLASS, {}
    )
    assert envelope["status"] == "error"
    assert "FileNotFoundError" in envelope["error"] or "not found" in envelope[
        "error"
    ].lower()


# ---------------------------------------------------------------------------
# Enum / ExecutorConfig field wiring
# ---------------------------------------------------------------------------
def test_executor_type_enum_has_globus_compute():
    assert ExecutorType.GLOBUS_COMPUTE.value == "globus_compute"


def test_executor_config_has_globus_compute_field():
    assert "globus_compute" in ExecutorConfig.model_fields
