"""Tests for G22 Step 3 — missed-schedule policy.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G22``.

Coverage:
1. TriggerConfig.on_missed default = 'skip'; values validated.
2. TimerTrigger.replay_missed_fires('skip') returns 0; no callbacks.
3. TimerTrigger.replay_missed_fires('catch_up') fires N times.
4. TimerTrigger.replay_missed_fires('merge') fires once.
5. zero-elapsed window: returns 0 for all policies.
6. Past-into-future timestamp: FAIL-FAST.
7. integer-millisecond rounding: 1.0s / 100ms = exactly 10 (no off-by-one).
8. WorkflowEntryTrigger.replay_missed_fires delegates to inner.
9. Wrapper-level on_missed overrides inner's on_missed for the replay.
10. Non-cadenced inner trigger (EventTrigger): wrapper replay no-ops.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
import yaml

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.trigger import (
    EventTrigger,
    TimerTrigger,
    TriggerConfig,
    TriggerType,
)
from nanobrain.library.runtime import (
    WorkflowEntryTrigger,
    WorkflowRunner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_timer(on_missed: str = "skip", interval_ms: int = 100) -> TimerTrigger:
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(
            name=f"t_{on_missed}",
            trigger_type=TriggerType.TIMER,
            timer_interval_ms=interval_ms,
            on_missed=on_missed,
            debounce_ms=0,
            max_frequency_hz=1000.0,
        )
    finally:
        TriggerConfig._allow_direct_instantiation = False
    return TimerTrigger.from_config(cfg)


# ---------------------------------------------------------------------------
# 1. TriggerConfig field
# ---------------------------------------------------------------------------

class TestOnMissedField:

    def test_default_skip(self):
        TriggerConfig._allow_direct_instantiation = True
        try:
            cfg = TriggerConfig(name="t", trigger_type=TriggerType.TIMER)
        finally:
            TriggerConfig._allow_direct_instantiation = False
        assert cfg.on_missed == "skip"

    def test_invalid_policy_rejected(self):
        TriggerConfig._allow_direct_instantiation = True
        try:
            with pytest.raises(Exception):
                TriggerConfig(
                    name="t", trigger_type=TriggerType.TIMER,
                    on_missed="explode",
                )
        finally:
            TriggerConfig._allow_direct_instantiation = False


# ---------------------------------------------------------------------------
# 2-4. TimerTrigger.replay_missed_fires policies
# ---------------------------------------------------------------------------

class TestReplayPolicies:

    def test_skip_returns_zero(self):
        async def run():
            t = _build_timer("skip", interval_ms=100)
            fires = []
            async def cb(_=None):
                fires.append(True)
            await t.add_callback(cb)
            n = await t.replay_missed_fires(0.0, 1.0)
            assert n == 0
            assert fires == []
        asyncio.run(run())

    def test_catch_up_fires_n_times(self):
        async def run():
            t = _build_timer("catch_up", interval_ms=100)
            fires = []
            async def cb(_=None):
                fires.append(True)
            await t.add_callback(cb)
            n = await t.replay_missed_fires(0.0, 1.0)
            assert n == 10
            assert len(fires) == 10
        asyncio.run(run())

    def test_merge_fires_once(self):
        async def run():
            t = _build_timer("merge", interval_ms=100)
            fires = []
            async def cb(_=None):
                fires.append(True)
            await t.add_callback(cb)
            n = await t.replay_missed_fires(0.0, 1.0)
            assert n == 1
            assert len(fires) == 1
        asyncio.run(run())

    def test_merge_fires_zero_when_no_intervals_missed(self):
        async def run():
            t = _build_timer("merge", interval_ms=100)
            fires = []
            async def cb(_=None):
                fires.append(True)
            await t.add_callback(cb)
            # 50ms elapsed, interval 100ms → 0 missed
            n = await t.replay_missed_fires(0.0, 0.05)
            assert n == 0
            assert fires == []
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5-7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_zero_elapsed_returns_zero_all_policies(self):
        async def run():
            for policy in ("skip", "catch_up", "merge"):
                t = _build_timer(policy, interval_ms=100)
                fires = []
                async def cb(_=None):
                    fires.append(True)
                await t.add_callback(cb)
                n = await t.replay_missed_fires(1.0, 1.0)
                assert n == 0, f"{policy} fired with no elapsed time"
                assert fires == []
        asyncio.run(run())

    def test_future_last_fire_fails_fast(self):
        async def run():
            t = _build_timer("catch_up")
            with pytest.raises(ValueError) as exc_info:
                await t.replay_missed_fires(2.0, 1.0)  # last > now
            assert "FAIL-FAST" in str(exc_info.value)
            assert "future" in str(exc_info.value)
        asyncio.run(run())

    def test_exact_boundary_is_n_not_n_minus_1(self):
        """The integer-ms rounding fix: elapsed=1.0s, interval=100ms
        must give exactly 10 missed fires, not 9 (the original
        floating-point bug)."""
        async def run():
            t = _build_timer("catch_up", interval_ms=100)
            fires = []
            async def cb(_=None):
                fires.append(True)
            await t.add_callback(cb)
            n = await t.replay_missed_fires(0.0, 1.0)
            assert n == 10
            assert len(fires) == 10
        asyncio.run(run())

    def test_zero_interval_returns_zero(self):
        async def run():
            t = _build_timer("catch_up", interval_ms=1)  # smallest allowed
            t.interval_ms = 0  # manual override for the edge case
            fires = []
            async def cb(_=None):
                fires.append(True)
            await t.add_callback(cb)
            n = await t.replay_missed_fires(0.0, 10.0)
            assert n == 0
            assert fires == []
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 8-10. WorkflowEntryTrigger wrapper
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path) -> WorkflowRunner:
    yml = tmp / "runner.yml"
    yml.write_text(yaml.safe_dump({
        "name": "r", "task_store_backend": "in_memory",
    }))
    return WorkflowRunner.from_config(str(yml))


async def _noop_workflow(payload):
    return {}


def _build_entry(tmp: Path, runner, inner, on_missed: str = "skip"):
    yml = tmp / "entry.yml"
    yml.write_text(yaml.safe_dump({"name": "entry", "on_missed": on_missed}))
    return WorkflowEntryTrigger.from_config(
        str(yml),
        runner=runner,
        inner_trigger=inner,
        workflow_callable=_noop_workflow,
    )


class TestWrapperReplay:

    def test_wrapper_delegates_to_inner_timer(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer("skip", interval_ms=100)
                entry = _build_entry(tmp, runner, inner, on_missed="catch_up")
                fires = []
                async def cb(_=None):
                    fires.append(True)
                await inner.add_callback(cb)
                n = await entry.replay_missed_fires(0.0, 1.0)
                assert n == 10
                assert len(fires) == 10
        asyncio.run(run())

    def test_wrapper_overrides_inner_policy(self):
        """Wrapper's on_missed='merge' must override inner's
        on_missed='catch_up' for the duration of the replay window."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer("catch_up", interval_ms=100)
                entry = _build_entry(tmp, runner, inner, on_missed="merge")
                fires = []
                async def cb(_=None):
                    fires.append(True)
                await inner.add_callback(cb)
                n = await entry.replay_missed_fires(0.0, 1.0)
                assert n == 1, "Wrapper-level merge should win over inner catch_up"
                assert len(fires) == 1
                # And inner's policy is restored after replay
                assert inner.on_missed == "catch_up"
        asyncio.run(run())

    def test_wrapper_event_inner_is_noop(self):
        """EventTrigger has no replay_missed_fires (not cadenced).
        The wrapper must return 0, not raise."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                TriggerConfig._allow_direct_instantiation = True
                try:
                    cfg = TriggerConfig(
                        name="ev", trigger_type=TriggerType.EVENT,
                        debounce_ms=0, max_frequency_hz=1000.0,
                    )
                finally:
                    TriggerConfig._allow_direct_instantiation = False
                inner = EventTrigger.from_config(cfg)
                entry = _build_entry(tmp, runner, inner, on_missed="catch_up")
                n = await entry.replay_missed_fires(0.0, 1.0)
                assert n == 0  # no schedule to miss
        asyncio.run(run())
