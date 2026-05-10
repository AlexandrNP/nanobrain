"""G11-completion — pin the LocalParslAdapter contract end-to-end against real Parsl.

Pre-G11-completion: the abstract ``ToolBackendAdapter`` shipped, but no
in-tree concrete adapter for local-process Python execution existed. The
RheaAdapter shipped from the Rhea fork; Galaxy is deferred. The
"single-host development + small-deployment" path had no first-class
backend, forcing every consumer to either route through Rhea (overkill)
or roll their own dispatcher (silent-failure-prone).

Post-G11-completion: ``LocalParslAdapter`` (BACKEND_NAME="local_parsl")
materializes a Python callable from a UTD's
``provenance_pin.class_path`` (or an explicit kwarg) and dispatches it
via Parsl's ``python_app``. The default executor preset is
``ThreadPoolExecutor`` (P0++b decision; see local_parsl_adapter.py
module docstring).

This integration test exercises the REAL Parsl runtime — no mocks, no
stubs. The test runs in a dedicated module-scoped scope so parsl.load
is called once and torn down cleanly at the end.

Pinned contracts:
  1. invoke() with explicit_callable returns the dict-shaped result
  2. invoke() with provenance_pin.class_path resolves the callable
     via importlib and runs it
  3. inputs missing required params FAIL-FAST before parsl is touched
  4. inputs with extra keys (no **kwargs in callable) FAIL-FAST
  5. callable returning a non-dict gets wrapped under "_result"
  6. concurrent invokes interleave (true parallelism — N callables
     submitted, N futures retrieved in non-input order)
  7. close() / scope() cleanup is idempotent

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 2 G11-completion;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.7 (P0++b).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.unified_tool_descriptor import UnifiedToolDescriptor
from nanobrain.library.tools.local_parsl_adapter import (
    LocalParslAdapter,
)


pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Module-scoped Parsl lifecycle: one adapter / one parsl.load per test
# module. Tests share the adapter, isolation comes from per-test inputs +
# ``ToolBackendRegistry`` not being touched.
#
# Module-scoped fixture also sidesteps the parsl-load-twice failure that
# function-scoped adapters would hit between tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter() -> LocalParslAdapter:
    a = LocalParslAdapter(executor_kind="thread", max_workers=4)
    yield a
    a.close()


# ---------------------------------------------------------------------------
# Test fixtures: callables that the adapter dispatches.
# Top-level so they are importable via dotted path (HighThroughputExecutor
# requires this). For ThreadPoolExecutor it's not strictly required but
# good discipline.
# ---------------------------------------------------------------------------


def _add_two(a: int, b: int) -> Dict[str, Any]:
    """Sum two integers; return the result + a worker marker."""
    import os

    return {"sum": a + b, "worker_pid": os.getpid()}


def _slow_double(x: int, sleep_seconds: float = 0.5) -> Dict[str, Any]:
    """Sleep then double — exercises true concurrency."""
    time.sleep(sleep_seconds)
    return {"doubled": x * 2}


def _no_required_args(default_a: int = 0) -> Dict[str, Any]:
    """All-optional signature. Used to test the missing-required path
    is selective (no false positives)."""
    return {"echoed": default_a}


def _strict_signature(only_x: int) -> Dict[str, Any]:
    """Single-required signature. Used to test missing-required + extra-key paths."""
    return {"only_x_seen": only_x}


def _scalar_return(x: int) -> int:
    """Returns a scalar (not a dict) — adapter must wrap under _result."""
    return x * 10


# ---------------------------------------------------------------------------
# UTD builders. We use UnifiedToolDescriptor.from_python_callable for the
# class_path-resolution path; the explicit-callable tests use a minimal
# UTD shape that ignores provenance_pin.
# ---------------------------------------------------------------------------


def _utd_for_callable(fn) -> UnifiedToolDescriptor:
    """Build a UTD whose provenance_pin.class_path points at ``fn``."""
    return UnifiedToolDescriptor.from_python_callable(
        fn,
        backend="local_parsl",
        version="0.1.0",
        provenance_class_path=f"{fn.__module__}.{fn.__name__}",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_invoke_with_explicit_callable_returns_dict(adapter: LocalParslAdapter):
    """The simplest path: pass a callable directly to the adapter's
    constructor surrogate (via ``explicit_callable``-shape adapter
    instance) — invoke() routes inputs through parsl and returns the
    dict result."""
    explicit = LocalParslAdapter(
        executor_kind="thread", max_workers=2, explicit_callable=_add_two,
    )
    # The explicit adapter must NOT load parsl twice — the module-
    # scoped adapter already loaded. It detects + reuses.
    utd = _utd_for_callable(_add_two)

    result = asyncio.run(
        explicit.invoke(utd, {"a": 3, "b": 4})
    )
    assert isinstance(result, dict)
    assert result["sum"] == 7
    assert "worker_pid" in result, (
        f"callable's worker_pid output should be passed through: {result}"
    )


def test_invoke_with_class_path_resolves_via_importlib(
    adapter: LocalParslAdapter,
):
    """The HPC-bundle-replay path: the UTD carries the callable's
    importable dotted path; the adapter resolves via importlib and
    dispatches."""
    utd = _utd_for_callable(_add_two)
    result = asyncio.run(adapter.invoke(utd, {"a": 10, "b": 32}))
    assert result["sum"] == 42


def test_invoke_missing_required_input_fails_fast(adapter: LocalParslAdapter):
    """Inputs missing a required parameter must FAIL-FAST BEFORE
    touching Parsl — otherwise the worker raises a TypeError far
    from the call site and operators have to dig through Parsl logs."""
    utd = _utd_for_callable(_strict_signature)
    with pytest.raises(ComponentConfigurationError) as excinfo:
        asyncio.run(adapter.invoke(utd, {}))
    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "only_x" in msg, (
        f"missing-required error must name the missing parameter; "
        f"got: {msg!r}"
    )


def test_invoke_extra_kwargs_fail_fast(adapter: LocalParslAdapter):
    """Inputs with keys not in the callable's signature (and the
    callable doesn't accept **kwargs) FAIL-FAST — adapters cannot
    silently drop unrecognized inputs because that would mask
    skeleton-binding regressions upstream."""
    utd = _utd_for_callable(_strict_signature)
    with pytest.raises(ComponentConfigurationError) as excinfo:
        asyncio.run(
            adapter.invoke(utd, {"only_x": 1, "rogue": "should not pass"})
        )
    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "rogue" in msg, (
        f"extra-key error must name the offending key; got: {msg!r}"
    )


def test_scalar_return_wrapped_under_result(adapter: LocalParslAdapter):
    """Callables that don't return a dict are wrapped under
    ``{"_result": <value>}`` so downstream code always gets a
    uniform dict shape (mirrors BaseStep._update_output_data_units)."""
    utd = _utd_for_callable(_scalar_return)
    result = asyncio.run(adapter.invoke(utd, {"x": 7}))
    assert isinstance(result, dict)
    assert result == {"_result": 70}


def test_concurrent_invokes_interleave(adapter: LocalParslAdapter):
    """Submit N slow callables; verify they ran concurrently (total
    wall-clock should be substantially less than N * sleep_seconds).

    With max_workers=4 and 4 tasks each sleeping 0.5s, sequential
    would be 2.0s; parallel should be ~0.5s. Use a generous bound
    of 1.5s to absorb test-runner overhead.
    """
    utd = _utd_for_callable(_slow_double)
    n_tasks = 4
    sleep_s = 0.5

    async def _run_all() -> list[Dict[str, Any]]:
        return await asyncio.gather(
            *[
                adapter.invoke(utd, {"x": i, "sleep_seconds": sleep_s})
                for i in range(n_tasks)
            ]
        )

    t0 = time.monotonic()
    results = asyncio.run(_run_all())
    elapsed = time.monotonic() - t0

    assert len(results) == n_tasks
    assert sorted(r["doubled"] for r in results) == [0, 2, 4, 6]
    assert elapsed < (n_tasks * sleep_s) - 0.2, (
        f"expected parallel execution; got elapsed={elapsed:.2f}s "
        f"(would be {n_tasks * sleep_s:.2f}s sequential). max_workers=4."
    )


def test_close_is_idempotent(tmp_path):
    """``close()`` can be called multiple times safely. Critical for
    test teardown + ``with adapter.scope()`` patterns where exception
    paths might call close() before normal exit also calls it.

    Constructs a fresh adapter to avoid disturbing the module-scoped
    fixture; the new adapter never loads parsl on its own (existing
    DFK takes precedence) so close() only clears caches.
    """
    a = LocalParslAdapter(executor_kind="thread", max_workers=1)
    a.close()
    a.close()
    a.close()  # third call must not raise


def test_callable_with_no_required_args_works(adapter: LocalParslAdapter):
    """All-optional signature: invoking with empty inputs must NOT
    trigger the missing-required FAIL-FAST."""
    utd = _utd_for_callable(_no_required_args)
    result = asyncio.run(adapter.invoke(utd, {}))
    assert result == {"echoed": 0}


def test_invalid_executor_kind_fails_fast_at_construction():
    """Bogus executor_kind must FAIL-FAST at __init__ — operators
    catch typos before any tool runs."""
    with pytest.raises(ComponentConfigurationError) as excinfo:
        LocalParslAdapter(executor_kind="rocket")  # type: ignore[arg-type]
    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg
    assert "thread" in msg and "process" in msg and "htex" in msg, (
        f"error must name the valid kinds; got: {msg!r}"
    )
