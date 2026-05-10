"""G26 — pin the CostEnvelope / CostTracker enforcement contract.

eval_03 Round 3 G26: pre-G26 the framework had ResourceEnvelope
declarations (G12) but no enforcement primitive. GATE-R1 in
hitl_safety_gates.md §8 describes "halt the task when its cumulative
cost exceeds the cap"; that halt did not exist. Without it the
autonomy mode cannot run safely — a single runaway long-running
LLM call could drain the deployment budget.

Post-G26: CostEnvelope (declared cap) + CostTracker (cumulative ledger
with check-and-raise enforcement) + current_cost_tracker() (contextvar
accessor).

This test pins:
  1. record() under the cap returns the new cumulative
  2. record() at the cap is allowed (boundary inclusive)
  3. record() above the cap raises CostEnvelopeBreach
  4. multiple kinds enforced independently
  5. ledger is failure-atomic — breach leaves ledger in pre-record state
  6. negative amount FAIL-FAST
  7. custom kinds (not usd/tokens/walltime) recorded but not enforced
  8. activate() context installs the tracker for nested code
  9. record_cost() helper is a no-op when no tracker is active
 10. record_cost() helper enforces against the active tracker
 11. multiple sequential records accumulate correctly
 12. concurrent records (under lock) cannot bypass the cap
 13. snapshot() returns a copy (mutation does not bleed)
 14. activate() context cleans up on exit (nested tracker contexts work)

Source: ``apecx-mcp-integration/eval_03_nanobrain_gap_inventory.md``
Round 3 G26;
``apecx-mcp-integration/docs/development_roadmap.md`` 8.8 (P6+b).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from nanobrain.core.cost_envelope import (
    CostEnvelope,
    CostEnvelopeBreach,
    CostTracker,
    current_cost_tracker,
    record_cost,
)


def test_record_under_cap_returns_new_cumulative():
    tracker = CostTracker(CostEnvelope(usd=10.0))
    assert tracker.record("usd", 3.0) == 3.0
    assert tracker.record("usd", 4.0) == 7.0


def test_record_at_cap_is_allowed_boundary_inclusive():
    """Recording exactly at the cap must NOT raise — operators
    declare caps as 'allow up to and including this'. Going one
    past would be a different test."""
    tracker = CostTracker(CostEnvelope(tokens=1000))
    tracker.record("tokens", 500)
    # Second record reaches exactly the cap.
    assert tracker.record("tokens", 500) == 1000.0


def test_record_above_cap_raises_breach():
    tracker = CostTracker(CostEnvelope(usd=10.0))
    tracker.record("usd", 9.0)
    with pytest.raises(CostEnvelopeBreach) as excinfo:
        tracker.record("usd", 1.5)  # would land at 10.5
    err = excinfo.value
    assert err.kind == "usd"
    assert err.cap == 10.0
    assert err.previous == 9.0
    # attempted is the would-be cumulative if we'd allowed it.
    assert err.attempted == 10.5


def test_multiple_kinds_enforced_independently():
    """An overrun on tokens must not affect the usd ledger; each
    kind has its own cap + ledger entry."""
    tracker = CostTracker(CostEnvelope(usd=10.0, tokens=100))
    tracker.record("usd", 5.0)
    with pytest.raises(CostEnvelopeBreach):
        tracker.record("tokens", 200)
    # usd ledger is unaffected by the token breach.
    assert tracker.cumulative("usd") == 5.0
    # tokens ledger is unchanged from the failed record.
    assert tracker.cumulative("tokens") == 0.0


def test_ledger_is_failure_atomic_on_breach():
    """A breached record() must leave the ledger in its pre-record
    state — otherwise a retry would charge twice."""
    tracker = CostTracker(CostEnvelope(usd=10.0))
    tracker.record("usd", 9.5)
    with pytest.raises(CostEnvelopeBreach):
        tracker.record("usd", 1.0)
    # Cumulative still 9.5, NOT 10.5.
    assert tracker.cumulative("usd") == 9.5
    # And a smaller follow-up should still fit.
    tracker.record("usd", 0.4)
    assert tracker.cumulative("usd") == 9.9


def test_negative_amount_fails_fast():
    """Negative cost is nonsense — operators almost always passed
    something they meant as a refund or a sentinel. FAIL-FAST so
    they fix the call site instead of polluting the ledger."""
    tracker = CostTracker(CostEnvelope(usd=10.0))
    with pytest.raises(ValueError, match="amount must be >= 0"):
        tracker.record("usd", -1.0)


def test_custom_kinds_recorded_but_not_enforced():
    """Operators may record arbitrary kinds (e.g., 'gpu_minutes')
    against a tracker; only the canonical three (usd / tokens /
    walltime_seconds) are enforced because only they are declared
    on the CostEnvelope schema."""
    tracker = CostTracker(CostEnvelope(usd=10.0))
    # Custom kind: no cap, no enforcement.
    tracker.record("gpu_minutes", 9999.0)
    tracker.record("gpu_minutes", 1.0)
    assert tracker.cumulative("gpu_minutes") == 10000.0


def test_activate_context_installs_and_clears():
    tracker = CostTracker(CostEnvelope(usd=5.0))
    assert current_cost_tracker() is None
    with tracker.activate():
        assert current_cost_tracker() is tracker
    assert current_cost_tracker() is None


def test_record_cost_helper_is_noop_without_tracker():
    """When no tracker is active, record_cost() is a fast no-op;
    callers don't have to hand-check the contextvar at every site."""
    # Note: Pytest tests run with a clean contextvar default.
    assert record_cost("usd", 99.0) is None


def test_record_cost_helper_enforces_active_tracker():
    tracker = CostTracker(CostEnvelope(tokens=100))
    with tracker.activate():
        assert record_cost("tokens", 50) == 50.0
        with pytest.raises(CostEnvelopeBreach):
            record_cost("tokens", 100)


def test_concurrent_records_cannot_bypass_cap():
    """Two threads racing on record() under one tracker MUST NOT
    both succeed past the cap. The lock is the contract; this test
    exercises it.

    Construct: cap=100, run 1000 record(1) calls across 8 threads.
    The first 100 succeed; the rest raise CostEnvelopeBreach. The
    final cumulative is exactly 100.
    """
    tracker = CostTracker(CostEnvelope(usd=100.0))
    breaches = []
    breaches_lock = threading.Lock()

    def _record_one() -> None:
        try:
            tracker.record("usd", 1.0)
        except CostEnvelopeBreach as e:
            with breaches_lock:
                breaches.append(e)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_record_one) for _ in range(1000)]
        for f in futures:
            f.result()

    assert tracker.cumulative("usd") == 100.0, (
        f"concurrent records bypassed the cap; cumulative="
        f"{tracker.cumulative('usd')}"
    )
    # 900 attempts breached.
    assert len(breaches) == 900


def test_snapshot_returns_copy():
    """Mutating a snapshot dict must NOT change the tracker's ledger.
    Critical because callers pass snapshots into provenance records
    and may modify them downstream."""
    tracker = CostTracker(CostEnvelope(usd=100.0))
    tracker.record("usd", 5.0)
    snap = tracker.snapshot()
    snap["usd"] = 999.0  # mutate the copy
    assert tracker.cumulative("usd") == 5.0


def test_no_envelope_means_no_enforcement():
    """A tracker with all-None caps records but never breaches —
    the operator wanted accounting without enforcement."""
    tracker = CostTracker(CostEnvelope())  # all None
    tracker.record("usd", 1e9)
    tracker.record("tokens", int(1e9))
    assert tracker.cumulative("usd") == 1e9


def test_nested_activate_contexts_restore_outer():
    """activate() is contextvar-based; nested activations should
    restore the outer tracker on exit (PEP 567 behavior)."""
    outer = CostTracker(CostEnvelope(usd=100.0))
    inner = CostTracker(CostEnvelope(usd=10.0))
    with outer.activate():
        assert current_cost_tracker() is outer
        with inner.activate():
            assert current_cost_tracker() is inner
        # Outer restored.
        assert current_cost_tracker() is outer
    assert current_cost_tracker() is None


def test_breach_message_carries_context():
    """The exception's __str__ must surface kind + cap + attempted +
    previous so operators can act without inspecting attributes."""
    tracker = CostTracker(CostEnvelope(usd=10.0))
    tracker.record("usd", 8.0)
    try:
        tracker.record("usd", 5.0)
    except CostEnvelopeBreach as e:
        s = str(e)
        assert "usd" in s
        assert "10" in s  # cap
        assert "13" in s  # attempted
        assert "8" in s  # previous
        assert "cost_envelope_breach" in s, (
            f"breach message must hint at the runner-side reason "
            f"string; got: {s!r}"
        )
        return
    pytest.fail("expected CostEnvelopeBreach")
