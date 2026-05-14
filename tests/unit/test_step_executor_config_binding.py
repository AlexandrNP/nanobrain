"""Unit tests for step-level ``executor_config`` binding (framework fix).

Before this fix, ``StepConfig.executor_config`` was a declared, Pydantic-parsed
field that ``BaseStep.resolve_dependencies`` never consumed — a step YAML had
NO way to declare a non-local executor. This suite pins the wiring:

  * a step YAML whose ``executor_config:`` references a thread ExecutorConfig
    builds a ``ThreadExecutor`` and binds it as ``step.executor``;
  * a step YAML whose ``executor_config:`` references a globus_compute
    ExecutorConfig builds a ``GlobusComputeExecutor`` with the config threaded
    through (no network — just isinstance + config assertions);
  * a step with NO ``executor_config`` still gets the default ``LocalExecutor``
    (regression guard for the hot path);
  * an unknown ``executor_type`` FAIL-LOUDs at step-build time.

Note on config shape: ``StepConfig.executor_config`` is typed
``Optional[ExecutorConfig]``. ``ExecutorConfig`` is a ``ConfigBase`` subclass
which FORBIDS inline-dict construction — so the step YAML must reference the
executor config via the ``class:`` + ``config:`` indirection pattern (a
separate ExecutorConfig YAML file), NOT an inline dict block. This is exercised
exactly as a real step YAML would declare it.

Unconditional — no network, no real endpoint, no auth.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.executor import (
    ExecutorConfig,
    LocalExecutor,
    ThreadExecutor,
    build_executor_from_config,
)
from nanobrain.core.distributed.globus_compute_executor import GlobusComputeExecutor
from nanobrain.core.step import BaseStep, StepConfig

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
_ECHO_STEP_CLASS_YML = str(_FIXTURE_DIR / "trivial_echo_step.yml")


class _PlainStep(BaseStep):
    """Minimal real step used to exercise resolve_dependencies end-to-end."""

    COMPONENT_TYPE = "plain_executor_binding_test_step"

    @classmethod
    def _get_config_class(cls):
        return StepConfig

    async def process(self, input_data, **kwargs):  # noqa: D102
        return {"out": input_data}


def _write(text: str, suffix: str = ".yml") -> str:
    f = tempfile.NamedTemporaryFile(
        "w", suffix=suffix, delete=False, encoding="utf-8"
    )
    f.write(text)
    f.close()
    return f.name


def _write_step_yaml(executor_config_block: str = "") -> str:
    """Write a minimal step YAML (no top-level `class:` so the called class loads it)."""
    return _write(
        "name: executor_binding_test_step\n"
        "description: 'step-level executor_config binding test'\n"
        + executor_config_block
    )


# ---------------------------------------------------------------------------
# Priority 2 — step's own executor_config builds the declared executor.
# ---------------------------------------------------------------------------
def test_step_executor_config_thread_builds_thread_executor():
    """A step YAML referencing a thread ExecutorConfig -> ThreadExecutor."""
    exec_yml = _write(
        "executor_type: thread\nname: step_thread_executor\nmax_workers: 3\n"
    )
    step_yml = _write_step_yaml(
        "executor_config:\n"
        "  class: 'nanobrain.core.executor.ExecutorConfig'\n"
        f"  config: {exec_yml!r}\n"
    )
    step = _PlainStep.from_config(step_yml)
    assert isinstance(step.executor, ThreadExecutor)


def test_step_executor_config_globus_compute_builds_globus_executor():
    """A step YAML referencing a globus_compute ExecutorConfig -> GlobusComputeExecutor.

    No network: we assert isinstance and that the globus_compute block
    threaded all the way through to the executor's validated config.
    """
    exec_yml = _write(
        "executor_type: globus_compute\n"
        "name: step_gce\n"
        "globus_compute:\n"
        "  endpoint_id: 'deadbeef-0000-1111-2222-333344445555'\n"
        "  auth_mode: client_credentials\n"
        "  task_timeout_seconds: 1800.0\n"
    )
    step_yml = _write_step_yaml(
        "executor_config:\n"
        "  class: 'nanobrain.core.executor.ExecutorConfig'\n"
        f"  config: {exec_yml!r}\n"
    )
    step = _PlainStep.from_config(step_yml)
    assert isinstance(step.executor, GlobusComputeExecutor)
    # Config threaded through to the validated GlobusComputeConfig.
    gc_cfg = step.executor.globus_compute_config
    assert gc_cfg.endpoint_id == "deadbeef-0000-1111-2222-333344445555"
    assert gc_cfg.auth_mode == "client_credentials"
    assert gc_cfg.task_timeout_seconds == 1800.0


# ---------------------------------------------------------------------------
# Priority 4 — no executor_config -> default LocalExecutor (regression guard).
# ---------------------------------------------------------------------------
def test_step_without_executor_config_gets_local_executor():
    """A step with no executor_config and no pre-built executor -> LocalExecutor.

    This is the hot path for EVERY step that doesn't declare an executor;
    behavior must be bit-for-bit unchanged.
    """
    step = _PlainStep.from_config(_write_step_yaml(""))
    assert isinstance(step.executor, LocalExecutor)


def test_trivial_echo_step_fixture_still_gets_local_executor():
    """The existing dispatch fixture (no executor_config) is unaffected."""
    from tests.fixtures.trivial_echo_step import TrivialEchoStep

    step = TrivialEchoStep.from_config(_ECHO_STEP_CLASS_YML)
    assert isinstance(step.executor, LocalExecutor)


# ---------------------------------------------------------------------------
# FAIL-LOUD — unknown executor_type at step-build time.
# ---------------------------------------------------------------------------
def test_step_unknown_executor_type_fails_loud():
    """An executor_config with an unknown executor_type FAIL-LOUDs.

    The unknown value is rejected by the ExecutorType enum when the
    referenced ExecutorConfig is loaded; the step build raises rather than
    silently falling back to LocalExecutor.
    """
    exec_yml = _write("executor_type: quantum_warp_drive\nname: bogus\n")
    step_yml = _write_step_yaml(
        "executor_config:\n"
        "  class: 'nanobrain.core.executor.ExecutorConfig'\n"
        f"  config: {exec_yml!r}\n"
    )
    with pytest.raises(Exception) as exc_info:
        _PlainStep.from_config(step_yml)
    assert "quantum_warp_drive" in str(exc_info.value) or "executor_type" in str(
        exc_info.value
    )


def test_step_globus_compute_missing_block_fails_loud():
    """globus_compute executor_type with no globus_compute block FAIL-LOUDs."""
    exec_yml = _write("executor_type: globus_compute\nname: gce_no_block\n")
    step_yml = _write_step_yaml(
        "executor_config:\n"
        "  class: 'nanobrain.core.executor.ExecutorConfig'\n"
        f"  config: {exec_yml!r}\n"
    )
    with pytest.raises(ComponentConfigurationError) as exc_info:
        _PlainStep.from_config(step_yml)
    assert "globus_compute" in str(exc_info.value)


# ---------------------------------------------------------------------------
# The shared dispatch helper itself.
# ---------------------------------------------------------------------------
def test_build_executor_from_config_unknown_type_fails_loud():
    """build_executor_from_config FAIL-LOUDs on an unknown executor_type."""

    class _FakeType:
        value = "not_a_real_executor"

    class _FakeExecutorConfig:
        executor_type = _FakeType()

    with pytest.raises(ComponentConfigurationError) as exc_info:
        build_executor_from_config(_FakeExecutorConfig())
    assert "not_a_real_executor" in str(exc_info.value)


def test_build_executor_from_config_thread_roundtrip():
    """build_executor_from_config(parsed ExecutorConfig) -> ThreadExecutor."""
    cfg = ExecutorConfig.from_config(
        _write("executor_type: thread\nname: helper_thread\nmax_workers: 2\n")
    )
    executor = build_executor_from_config(cfg)
    assert isinstance(executor, ThreadExecutor)
