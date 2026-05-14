"""Unit tests for workflow-level ``executor_config`` binding (silent-failure fix).

Before this fix, ``Workflow.resolve_dependencies`` built the workflow-level
executor inside a ``try/except Exception`` that, on ANY failure, logged a
warning, set ``executor = None``, and fell through to the parent default
``LocalExecutor``. That is a catastrophic silent failure for remote execution:
a workflow declaring ``executor_type: parsl`` / ``globus_compute`` that could
not build the executor (missing dependency, bad endpoint, malformed YAML,
unknown ``executor_type``) would *silently run on the local machine* — tests
pass, the workflow "runs", but it ran in the wrong place.

This suite pins the corrected contract:

  * no ``executor_config`` declared -> parent default ``LocalExecutor``
    (regression guard for the hot path — unchanged behavior);
  * an ``executor_config`` declared but unbuildable -> FAIL-LOUD with
    ``ComponentConfigurationError`` naming the config path (previously it
    silently became Local);
  * a pre-built ``executor`` passed programmatically still wins.

``Workflow.resolve_dependencies`` is a classmethod taking a
``component_config`` dict, so it is exercised directly — no full workflow
graph needed. Unconditional — no network, no real endpoint.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.executor import LocalExecutor
from nanobrain.core.workflow import Workflow


def _write(text: str, suffix: str = ".yml") -> str:
    f = tempfile.NamedTemporaryFile(
        "w", suffix=suffix, delete=False, encoding="utf-8"
    )
    f.write(text)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# No executor_config -> parent default LocalExecutor (regression guard).
# ---------------------------------------------------------------------------
def test_workflow_without_executor_config_defaults_to_local():
    """A workflow with no executor_config falls through to LocalExecutor.

    This is correct: a workflow that asks for nothing gets Local. Behavior
    must be bit-for-bit unchanged by the silent-failure fix.
    """
    deps = Workflow.resolve_dependencies({})
    assert isinstance(deps["executor"], LocalExecutor)

    deps_explicit_none = Workflow.resolve_dependencies({"executor_config": None})
    assert isinstance(deps_explicit_none["executor"], LocalExecutor)


# ---------------------------------------------------------------------------
# executor_config declared AND builds -> use it (via the shared helper).
# ---------------------------------------------------------------------------
def test_workflow_executor_config_thread_builds_thread_executor():
    """A workflow executor_config referencing a thread ExecutorConfig -> ThreadExecutor."""
    from nanobrain.core.executor import ThreadExecutor

    exec_yml = _write(
        "executor_type: thread\nname: wf_thread_executor\nmax_workers: 2\n"
    )
    deps = Workflow.resolve_dependencies({"executor_config": exec_yml})
    assert isinstance(deps["executor"], ThreadExecutor)


# ---------------------------------------------------------------------------
# executor_config declared but UNBUILDABLE -> FAIL-LOUD (the core fix).
# ---------------------------------------------------------------------------
def test_workflow_unknown_executor_type_fails_loud():
    """An executor_config with an unknown executor_type FAIL-LOUDs.

    Previously this silently fell back to LocalExecutor — the exact
    silent-relocation bug this fix closes.
    """
    exec_yml = _write("executor_type: quantum_warp_drive\nname: bogus\n")
    with pytest.raises(ComponentConfigurationError) as exc_info:
        Workflow.resolve_dependencies({"executor_config": exec_yml})
    msg = str(exc_info.value)
    # Message names the offending config path and the underlying cause.
    assert exec_yml in msg
    assert "quantum_warp_drive" in msg or "executor_type" in msg
    # It must NOT have silently produced a LocalExecutor.


def test_workflow_missing_executor_config_file_fails_loud():
    """An executor_config path that does not resolve FAIL-LOUDs, not falls back."""
    bogus_path = "/nonexistent/dir/does_not_exist_executor.yml"
    with pytest.raises(ComponentConfigurationError) as exc_info:
        Workflow.resolve_dependencies({"executor_config": bogus_path})
    assert bogus_path in str(exc_info.value)


def test_workflow_malformed_executor_config_fails_loud():
    """A malformed executor_config YAML FAIL-LOUDs rather than silently going Local."""
    # globus_compute executor_type with no globus_compute block is rejected
    # downstream by GlobusComputeExecutor — a real "declared but unbuildable".
    exec_yml = _write("executor_type: globus_compute\nname: gce_no_block\n")
    with pytest.raises(ComponentConfigurationError) as exc_info:
        Workflow.resolve_dependencies({"executor_config": exec_yml})
    assert exec_yml in str(exc_info.value)


# ---------------------------------------------------------------------------
# A pre-built executor passed programmatically still wins.
# ---------------------------------------------------------------------------
def test_workflow_prebuilt_executor_kwarg_wins():
    """A pre-built `executor` kwarg takes precedence over executor_config."""
    local_yml = _write("executor_type: local\nname: prebuilt_local\n")
    prebuilt = LocalExecutor.from_config(local_yml)
    # executor_config is bogus, but the pre-built executor kwarg short-circuits
    # before it is ever consulted -> no raise.
    deps = Workflow.resolve_dependencies(
        {"executor_config": "/nonexistent/bogus.yml"}, executor=prebuilt
    )
    assert deps["executor"] is prebuilt
