"""G44 — pin the DataUnitProxyRef.namespace() unscoped-fallback contract.

Pre-G44 ``DataUnitProxyRef.namespace()`` silently returned ``""`` when no
``WorkflowRunContext`` was active, leaving the proxystore key under a
GLOBAL identity instead of per-run isolation. Multi-tenant workflows
running in a shared ProxyStore Redis would collide on equality / hashing
without any visible signal — the same shape as G7's ``auto_transfer=False``
silent failure.

Post-G44:

  * default mode emits a WARNING once per DataUnitProxyRef instance,
    naming the data-unit + the reason (no run context vs. library
    import failure)
  * ``NANOBRAIN_STRICT_NAMESPACE=1`` flips the WARNING into a
    ``ComponentConfigurationError`` so any deployment where multi-tenant
    isolation MUST hold can FAIL-FAST

This test pins:

  1. unscoped fallback emits exactly one WARNING per instance
  2. WARNING message names the data unit + the reason
  3. WARNING is rate-limited (a second namespace() call does NOT
     re-warn)
  4. ``NANOBRAIN_STRICT_NAMESPACE=1`` raises ComponentConfigurationError
  5. an explicit ``proxystore_namespace_prefix`` suppresses the warning
     entirely
  6. an active WorkflowRunContext suppresses the warning entirely

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 5 G44; ``apecx-mcp-integration/docs/development_roadmap.md`` 8.6.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from nanobrain.core.component_base import ComponentConfigurationError


def _make_unscoped_proxy_ref():
    """Build a minimal DataUnitProxyRef without invoking from_config —
    we monkey-patch in only the state ``namespace()`` reads, so the test
    does not require a live ProxyStore.

    Direct instantiation is normally forbidden by FromConfigBase; we use
    ``object.__new__`` to bypass the constructor guard for this targeted
    state-machine test, then attach the minimum attributes.
    """
    from nanobrain.core.data_unit import DataUnitProxyRef

    ref = object.__new__(DataUnitProxyRef)
    ref.name = "test_unscoped_du"
    # No explicit prefix — falls through to context lookup, then to "".
    ref._namespace_prefix = ""
    ref._unscoped_warning_emitted = False
    return ref


def test_unscoped_namespace_emits_one_warning(caplog: pytest.LogCaptureFixture):
    """First namespace() call with no run context AND no prefix
    emits a WARNING; the resolved value is still ``""``."""
    ref = _make_unscoped_proxy_ref()

    caplog.set_level(logging.WARNING)
    with patch(
        "nanobrain.library.orchestration.run_context.current_run_context",
        return_value=None,
    ):
        result = ref.namespace()

    assert result == "", (
        f"unscoped fallback must still resolve to '' (G13 contract); "
        f"got {result!r}"
    )
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "test_unscoped_du" in r.message
    ]
    assert len(warnings) == 1, (
        f"expected exactly one G44 WARNING for the unscoped data unit; "
        f"got {len(warnings)} (records: {[r.message for r in caplog.records]})"
    )


def test_unscoped_warning_names_du_and_reason(caplog: pytest.LogCaptureFixture):
    """Operator must see WHICH data unit was unscoped and WHY."""
    ref = _make_unscoped_proxy_ref()

    caplog.set_level(logging.WARNING)
    with patch(
        "nanobrain.library.orchestration.run_context.current_run_context",
        return_value=None,
    ):
        ref.namespace()

    relevant = [r for r in caplog.records if "test_unscoped_du" in r.message]
    assert relevant, "no warning mentioned the data unit name"
    msg = relevant[0].message
    assert "no active WorkflowRunContext" in msg, (
        f"WARNING must name the reason; got: {msg!r}"
    )
    assert "NANOBRAIN_STRICT_NAMESPACE" in msg, (
        f"WARNING must hint at the strict-mode opt-out; got: {msg!r}"
    )


def test_unscoped_warning_is_rate_limited_per_instance(
    caplog: pytest.LogCaptureFixture,
):
    """Second + Nth namespace() calls do NOT re-warn — namespace() is
    called many times per run (set / get / equality / hash); spamming
    the log would be its own usability bug."""
    ref = _make_unscoped_proxy_ref()

    caplog.set_level(logging.WARNING)
    with patch(
        "nanobrain.library.orchestration.run_context.current_run_context",
        return_value=None,
    ):
        ref.namespace()
        ref.namespace()
        ref.namespace()

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "test_unscoped_du" in r.message
    ]
    assert len(warnings) == 1, (
        f"namespace() WARNING must rate-limit to once per instance; "
        f"got {len(warnings)} warnings."
    )


def test_strict_mode_raises_instead_of_warning(monkeypatch):
    """``NANOBRAIN_STRICT_NAMESPACE=1`` flips WARNING into FAIL-FAST.

    Deployments where multi-tenant isolation MUST hold (HPC bundles,
    shared-Redis ProxyStore) opt into this mode so a missing
    WorkflowRunContext breaks the run loudly instead of silently
    cross-pollinating tenants' keys.
    """
    monkeypatch.setenv("NANOBRAIN_STRICT_NAMESPACE", "1")
    ref = _make_unscoped_proxy_ref()

    with patch(
        "nanobrain.library.orchestration.run_context.current_run_context",
        return_value=None,
    ):
        with pytest.raises(ComponentConfigurationError) as excinfo:
            ref.namespace()

    msg = str(excinfo.value)
    assert "FAIL-FAST" in msg, (
        f"strict-mode error must mention FAIL-FAST; got: {msg!r}"
    )
    assert "test_unscoped_du" in msg, (
        f"strict-mode error must name the offending data unit; got: {msg!r}"
    )


def test_explicit_prefix_suppresses_warning(caplog: pytest.LogCaptureFixture):
    """When the data unit declares a ``proxystore_namespace_prefix``,
    the silent-failure shape does not apply (the prefix wins per the
    G13 resolution order). No warning should fire."""
    ref = _make_unscoped_proxy_ref()
    ref._namespace_prefix = "explicit-prefix-tenant-7"

    caplog.set_level(logging.WARNING)
    # Even with no run context, the explicit prefix wins.
    with patch(
        "nanobrain.library.orchestration.run_context.current_run_context",
        return_value=None,
    ):
        result = ref.namespace()

    assert result == "explicit-prefix-tenant-7"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "test_unscoped_du" in r.message
    ]
    assert not warnings, (
        f"explicit prefix must suppress G44 warning; got {len(warnings)}"
    )


def test_active_run_context_suppresses_warning(caplog: pytest.LogCaptureFixture):
    """When a WorkflowRunContext is active, namespace() returns its
    proxystore_namespace and does NOT warn — the run context is the
    canonical source of multi-tenant isolation."""
    ref = _make_unscoped_proxy_ref()

    class _FakeRunCtx:
        proxystore_namespace = "run-id-42"

    caplog.set_level(logging.WARNING)
    with patch(
        "nanobrain.library.orchestration.run_context.current_run_context",
        return_value=_FakeRunCtx(),
    ):
        result = ref.namespace()

    assert result == "run-id-42"
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "test_unscoped_du" in r.message
    ]
    assert not warnings, (
        f"active run context must suppress G44 warning; got {len(warnings)}"
    )
