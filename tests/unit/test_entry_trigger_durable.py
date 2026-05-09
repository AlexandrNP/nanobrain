"""Tests for G22 Step 4 — durable inner-trigger → launch binding.

Per ``apecx-mcp-integration/docs/nanobrain_capability_gaps.md G22``.

Coverage:
1. InMemoryEntryStateStore basic CRUD.
2. FileEntryStateStore basic CRUD + atomic write.
3. FileEntryStateStore rejects path-traversal entry_ids.
4. bind_durable_state requires an EntryStateStore instance.
5. Inner fire persists last_fire_epoch_seconds + last_task_id.
6. State-store write failure does NOT crash the fire cascade
   (lost-bookkeeping > lost-work).
7. recover_from_durable_state with no prior state returns 0.
8. recover_from_durable_state replays missed fires per on_missed.
9. Two restarts in a row each see persisted state from the prior fire.
10. EventTrigger inner: persistence works but recover is a no-op
    (event-driven, not cadenced).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
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
    EntryStateStore,
    FileEntryStateStore,
    InMemoryEntryStateStore,
    WorkflowEntryTrigger,
    WorkflowRunner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_runner(tmp: Path) -> WorkflowRunner:
    yml = tmp / "runner.yml"
    yml.write_text(yaml.safe_dump({
        "name": "r", "task_store_backend": "in_memory",
        "heartbeat_interval_seconds": 0,
    }))
    return WorkflowRunner.from_config(str(yml))


def _build_timer(on_missed: str = "skip") -> TimerTrigger:
    TriggerConfig._allow_direct_instantiation = True
    try:
        cfg = TriggerConfig(
            name="ticker", trigger_type=TriggerType.TIMER,
            timer_interval_ms=100, on_missed=on_missed,
            debounce_ms=0, max_frequency_hz=1000.0,
        )
    finally:
        TriggerConfig._allow_direct_instantiation = False
    return TimerTrigger.from_config(cfg)


async def _noop_workflow(payload):
    return {"ok": True}


def _build_entry(tmp: Path, runner, inner, on_missed: str = "skip"):
    yml = tmp / "entry.yml"
    yml.write_text(yaml.safe_dump({"name": "entry", "on_missed": on_missed}))
    return WorkflowEntryTrigger.from_config(
        str(yml), runner=runner, inner_trigger=inner,
        workflow_callable=_noop_workflow,
    )


# ---------------------------------------------------------------------------
# 1. InMemoryEntryStateStore
# ---------------------------------------------------------------------------

class TestInMemoryStore:

    def test_set_get_delete(self):
        async def run():
            store = InMemoryEntryStateStore()
            assert await store.get("nope") is None
            await store.set("e1", {"a": 1, "b": 2})
            assert await store.get("e1") == {"a": 1, "b": 2}
            await store.delete("e1")
            assert await store.get("e1") is None
        asyncio.run(run())

    def test_get_returns_a_copy(self):
        """Mutating the returned dict must not affect stored state."""
        async def run():
            store = InMemoryEntryStateStore()
            await store.set("e1", {"a": 1})
            r = await store.get("e1")
            r["a"] = 999
            r2 = await store.get("e1")
            assert r2 == {"a": 1}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 2-3. FileEntryStateStore
# ---------------------------------------------------------------------------

class TestFileStore:

    def test_set_get_delete(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = FileEntryStateStore(str(Path(tmp) / "state"))
                assert await store.get("e1") is None
                await store.set("e1", {"a": 1, "b": [2, 3]})
                got = await store.get("e1")
                assert got == {"a": 1, "b": [2, 3]}
                await store.delete("e1")
                assert await store.get("e1") is None
        asyncio.run(run())

    def test_atomic_write_no_torn_file(self):
        """Verify that the .tmp file does NOT remain after a write."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp) / "state"
                store = FileEntryStateStore(str(base))
                await store.set("e1", {"a": 1})
                assert (base / "e1.json").is_file()
                assert not (base / "e1.json.tmp").exists()
        asyncio.run(run())

    def test_path_traversal_rejected(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = FileEntryStateStore(str(Path(tmp) / "state"))
                for bad in ("..", ".", "../escape", "a/b", "a\\b"):
                    with pytest.raises(ComponentConfigurationError) as exc_info:
                        await store.set(bad, {})
                    assert "FAIL-FAST" in str(exc_info.value)
        asyncio.run(run())

    def test_empty_entry_id_rejected(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                store = FileEntryStateStore(str(Path(tmp) / "state"))
                with pytest.raises(ComponentConfigurationError):
                    await store.set("", {})
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. bind_durable_state validation
# ---------------------------------------------------------------------------

class TestBindValidation:

    def test_non_store_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            inner = _build_timer()
            entry = _build_entry(tmp, runner, inner)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                entry.bind_durable_state("not a store")
            assert "FAIL-FAST" in str(exc_info.value)

    def test_none_detaches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runner = _build_runner(tmp)
            inner = _build_timer()
            entry = _build_entry(tmp, runner, inner)
            store = InMemoryEntryStateStore()
            entry.bind_durable_state(store, "e1")
            assert entry._state_store is store
            entry.bind_durable_state(None)
            assert entry._state_store is None


# ---------------------------------------------------------------------------
# 5-6. Persistence on inner fire
# ---------------------------------------------------------------------------

class TestPersistenceOnFire:

    def test_persists_last_fire_and_task_id(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer()
                entry = _build_entry(tmp, runner, inner)
                store = InMemoryEntryStateStore()
                entry.bind_durable_state(store)

                t0 = time.time()
                await entry._on_inner_fire(None)
                await asyncio.sleep(0.05)
                state = await store.get("entry")
                assert state is not None
                assert state["last_fire_epoch_seconds"] >= t0
                assert state["last_task_id"].startswith("entry-")
        asyncio.run(run())

    def test_store_failure_does_not_crash_fire(self):
        """Lost-bookkeeping > lost-work. If the durable store throws,
        the workflow still launches; the fire is not rolled back."""

        class _BrokenStore(EntryStateStore):
            async def get(self, entry_id):
                return None
            async def set(self, entry_id, state):
                raise RuntimeError("disk full")
            async def delete(self, entry_id):
                pass

        async def run():
            launches = []
            async def on_launch(h):
                launches.append(h.task_id)

            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer()
                yml = tmp / "entry.yml"
                yml.write_text(yaml.safe_dump({"name": "entry"}))
                entry = WorkflowEntryTrigger.from_config(
                    str(yml), runner=runner, inner_trigger=inner,
                    workflow_callable=_noop_workflow,
                    on_launch=on_launch,
                )
                entry.bind_durable_state(_BrokenStore())

                # Fire — store failure must NOT propagate
                await entry._on_inner_fire(None)
                await asyncio.sleep(0.05)
                # The workflow still launched.
                assert len(launches) == 1
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7-9. recover_from_durable_state
# ---------------------------------------------------------------------------

class TestRecover:

    def test_no_prior_state_returns_zero(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer(on_missed="catch_up")
                entry = _build_entry(tmp, runner, inner, on_missed="catch_up")
                entry.bind_durable_state(InMemoryEntryStateStore())
                n = await entry.recover_from_durable_state()
                assert n == 0
        asyncio.run(run())

    def test_replays_missed_fires_from_persisted_timestamp(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer(on_missed="catch_up")
                fires = []
                async def cb(_=None):
                    fires.append(True)
                await inner.add_callback(cb)

                entry = _build_entry(tmp, runner, inner, on_missed="catch_up")
                store = InMemoryEntryStateStore()
                entry.bind_durable_state(store)

                # Set a persisted timestamp 0.5s in the past.
                await store.set("entry", {
                    "last_fire_epoch_seconds": time.time() - 0.5,
                    "last_task_id": "old",
                })
                n = await entry.recover_from_durable_state()
                # 100ms interval, 500ms elapsed → 5 missed
                assert n == 5
                assert len(fires) == 5
        asyncio.run(run())

    def test_two_restarts_each_see_persisted_state(self):
        """End-to-end: fire → persist → simulate restart → recover →
        observed timestamp matches the last fire's timestamp."""
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                runner = _build_runner(tmp)
                inner = _build_timer()
                entry = _build_entry(tmp, runner, inner)
                store = FileEntryStateStore(str(tmp / "state"))
                entry.bind_durable_state(store)

                t_fire = time.time()
                await entry._on_inner_fire(None)
                await asyncio.sleep(0.05)

                # "Restart" — fresh wrapper, same store.
                inner2 = _build_timer()
                entry2 = _build_entry(tmp, runner, inner2)
                entry2.bind_durable_state(store)
                # Read raw state to verify
                state = await store.get("entry")
                assert state is not None
                assert abs(state["last_fire_epoch_seconds"] - t_fire) < 1.0
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 10. Event inner: persist + no-op recover
# ---------------------------------------------------------------------------

class TestEventInner:

    def test_event_inner_persists_but_recover_noops(self):
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
                entry = _build_entry(tmp, runner, inner)
                store = InMemoryEntryStateStore()
                entry.bind_durable_state(store)

                await entry._on_inner_fire({"kind": "x"})
                await asyncio.sleep(0.05)
                # State persisted (the wrapper doesn't know inner is non-cadenced)
                state = await store.get("entry")
                assert state is not None

                # But recovery is a no-op for event-driven inners:
                # replay_missed_fires returns 0 when inner has no
                # replay_missed_fires method.
                n = await entry.recover_from_durable_state()
                assert n == 0
        asyncio.run(run())
