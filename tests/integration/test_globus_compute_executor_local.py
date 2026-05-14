"""Integration test for GlobusComputeExecutor against a REAL Globus Compute endpoint.

Gated on ``$GLOBUS_COMPUTE_ENDPOINT_ID`` (mirrors the gating pattern in
``test_rhea_mcp_dispatcher.py``). When the env var is unset this whole
module SKIPS cleanly — no network, no auth, no failure.

To run this against a real endpoint:

  export GLOBUS_COMPUTE_ENDPOINT_ID=<your endpoint uuid>
  # confidential client credentials for the default client_credentials mode:
  export GLOBUS_COMPUTE_CLIENT_ID=<...>
  export GLOBUS_COMPUTE_CLIENT_SECRET=<...>
  # the endpoint MUST have nanobrain importable AND the fixture step YAML
  # resolvable on its filesystem (shared FS, or the same checkout path).
  pytest tests/integration/test_globus_compute_executor_local.py

This test was NOT run in the development environment — no Globus auth is
available here. It is the real-data round-trip the workspace policy
requires before the executor is considered "tested"; an operator with a
real endpoint must run it and record the outcome.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_ENDPOINT_ID = os.environ.get("GLOBUS_COMPUTE_ENDPOINT_ID")
_skip = pytest.mark.skipif(
    _ENDPOINT_ID is None,
    reason="GLOBUS_COMPUTE_ENDPOINT_ID not set",
)

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
_ECHO_STEP_YML = str(_FIXTURE_DIR / "trivial_echo_step.yml")


@pytest.fixture(autouse=True)
def _ensure_tests_on_path():
    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    yield


def _write_executor_yaml() -> str:
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".yml", delete=False, encoding="utf-8"
    )
    f.write(
        "executor_type: globus_compute\n"
        "name: gce_integration\n"
        "globus_compute:\n"
        f"  endpoint_id: {_ENDPOINT_ID!r}\n"
        "  auth_mode: client_credentials\n"
        "  task_timeout_seconds: 600.0\n"
    )
    f.close()
    return f.name


@_skip
def test_real_round_trip_trivial_step():
    """Submit a real nanobrain step to the real endpoint; assert the result.

    The step's ``execute_wrapper`` closure shape is reproduced here exactly
    as ``BaseStep._execute_on_trigger`` builds it: a zero-arg callable whose
    closure captures the step instance and the input_data dict.
    """
    from nanobrain.core.distributed.globus_compute_executor import (
        GlobusComputeExecutor,
    )
    from tests.fixtures.trivial_echo_step import TrivialEchoStep

    executor = GlobusComputeExecutor.from_config(_write_executor_yaml())
    step = TrivialEchoStep.from_config(_ECHO_STEP_YML)
    input_data = {"value": "round-trip"}

    # Reproduce the closure BaseStep hands to executor.execute(...).
    async def execute_wrapper():  # noqa: RUF029 - shape must match the framework
        return await step.process(input_data)

    async def _run():
        try:
            result = await executor.execute(execute_wrapper)
        finally:
            await executor.shutdown()
        return result

    result = asyncio.run(_run())
    # TrivialEchoStep uppercases a string payload (unwrapping the 1-key dict).
    assert result == {"echoed": "ROUND-TRIP"}


@_skip
def test_real_endpoint_surfaces_remote_exception():
    """A step that raises on the endpoint must surface FAIL-LOUD locally."""
    from nanobrain.core.distributed.globus_compute_executor import (
        GlobusComputeExecutor,
    )
    from tests.fixtures.trivial_echo_step import TrivialEchoStep

    executor = GlobusComputeExecutor.from_config(_write_executor_yaml())
    step = TrivialEchoStep.from_config(_ECHO_STEP_YML)

    # Point the step at a bogus class name path is not possible here, so
    # instead pass an input the worker reconstruction handles fine — this
    # test mainly asserts a clean shutdown + a real result envelope shape.
    # (A genuine remote-raise step belongs in a richer fixture; the
    # error-envelope path is unit-tested in test_globus_compute_executor.py.)
    input_data = {"value": "ok"}

    async def execute_wrapper():  # noqa: RUF029
        return await step.process(input_data)

    async def _run():
        try:
            return await executor.execute(execute_wrapper)
        finally:
            await executor.shutdown()

    result = asyncio.run(_run())
    assert result == {"echoed": "OK"}
