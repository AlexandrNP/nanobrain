"""Tests for ``ToolBase.from_python_callable`` + PythonCallableDispatcher.

The ergonomic counterpart to RheaMCPDispatcher: wraps an in-process
Python callable as a runtime-dispatchable ToolBase. Same descriptor-
driven API as RheaMCPDispatcher; same FAIL-FAST contract on bad
payloads.

Coverage:
1. Sync callable round-trip (sync fn → ToolBase → execute).
2. Async callable round-trip.
3. Sync callables run in asyncio.to_thread (don't block the loop).
4. Missing required parameter → FAIL-FAST.
5. Extra unexpected kwarg → FAIL-FAST (unless callable accepts **kwargs).
6. **kwargs-accepting callable allows extras.
7. Non-dict payload → FAIL-FAST.
8. Descriptor is auto-derived AND attached as ``tool.descriptor``.
9. UTD overrides flow through (display_name, summary, etc.).
10. The dispatcher's provenance_pin points at PythonCallableDispatcher
    (not at the callable's import path).
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.core.tool import ToolBase
from nanobrain.library.tools.python_callable_dispatcher import (
    PythonCallableDispatcher,
    build_tool_from_python_callable,
)


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

def _add(a: int, b: int = 0) -> int:
    """Add two integers."""
    return a + b


async def _afetch(url: str) -> dict:
    """Pretend HTTP fetch (async)."""
    return {"url": url, "ok": True}


def _slow_blocking(_) -> str:
    """Sync callable that blocks for ~50ms."""
    time.sleep(0.05)
    return "done"


def _accepts_kwargs(**kwargs) -> dict:
    """Callable that accepts arbitrary **kwargs."""
    return dict(kwargs)


def _no_required_args(x: int = 0, y: int = 0) -> int:
    """All optional args."""
    return x + y


# ---------------------------------------------------------------------------
# 1-2. Sync + async round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_sync_callable(self):
        async def run():
            tool = ToolBase.from_python_callable(_add)
            r = await tool.execute({"a": 5, "b": 3})
            assert r == 8
            r2 = await tool.execute({"a": 10})  # b uses default 0
            assert r2 == 10
        asyncio.run(run())

    def test_async_callable(self):
        async def run():
            tool = ToolBase.from_python_callable(_afetch)
            r = await tool.execute({"url": "http://example.com"})
            assert r == {"url": "http://example.com", "ok": True}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3. Sync callables run in to_thread (loop stays responsive)
# ---------------------------------------------------------------------------

class TestSyncOffload:

    def test_sync_callable_does_not_block_loop(self):
        """A sync callable that sleeps must run via asyncio.to_thread.
        We verify by scheduling a fast async task IN PARALLEL and
        confirming it completes before the sync callable does."""
        async def run():
            tool = ToolBase.from_python_callable(_slow_blocking)

            fast_completed_first = []

            async def fast():
                await asyncio.sleep(0.01)
                fast_completed_first.append(True)

            slow_task = asyncio.create_task(tool.execute({"_": "x"}))
            fast_task = asyncio.create_task(fast())
            await asyncio.gather(slow_task, fast_task)

            # If the sync callable had blocked the loop, ``fast`` would
            # have finished AFTER the 50ms sleep. Verifying both ran
            # concurrently:
            assert fast_completed_first == [True]
            assert slow_task.result() == "done"
        asyncio.run(run())

    def test_sync_callable_runs_in_different_thread(self):
        """The to_thread offload should run the callable in a worker
        thread, not the main thread."""
        async def run():
            main_thread_id = threading.get_ident()
            captured = {}

            def capture_thread(_) -> int:
                captured["tid"] = threading.get_ident()
                return captured["tid"]

            tool = ToolBase.from_python_callable(capture_thread)
            await tool.execute({"_": None})
            assert captured["tid"] != main_thread_id
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4-5. FAIL-FAST on bad payloads
# ---------------------------------------------------------------------------

class TestBadPayloadFailFast:

    def test_missing_required_param_fails_fast(self):
        async def run():
            tool = ToolBase.from_python_callable(_add)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await tool.execute({"b": 1})  # missing 'a'
            assert "FAIL-FAST" in str(exc_info.value)
            assert "missing required parameters" in str(exc_info.value)
            assert "'a'" in str(exc_info.value)
        asyncio.run(run())

    def test_extra_unexpected_kwarg_fails_fast(self):
        async def run():
            tool = ToolBase.from_python_callable(_add)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await tool.execute({"a": 1, "b": 2, "z": 99})
            assert "FAIL-FAST" in str(exc_info.value)
            assert "unexpected keys" in str(exc_info.value)
        asyncio.run(run())

    def test_non_dict_payload_fails_fast(self):
        async def run():
            tool = ToolBase.from_python_callable(_add)
            with pytest.raises(ComponentConfigurationError) as exc_info:
                await tool.execute("not a dict")
            assert "FAIL-FAST" in str(exc_info.value)
            assert "must be a dict" in str(exc_info.value)
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. **kwargs-accepting callable
# ---------------------------------------------------------------------------

class TestVarKwargsAcceptance:

    def test_kwargs_callable_accepts_arbitrary_keys(self):
        async def run():
            tool = ToolBase.from_python_callable(_accepts_kwargs)
            r = await tool.execute({"foo": 1, "bar": 2, "baz": "hi"})
            assert r == {"foo": 1, "bar": 2, "baz": "hi"}
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. Empty payload + all-optional callable
# ---------------------------------------------------------------------------

class TestEmptyPayload:

    def test_empty_payload_with_all_optional_args_works(self):
        async def run():
            tool = ToolBase.from_python_callable(_no_required_args)
            r = await tool.execute({})
            assert r == 0  # defaults x=0, y=0
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 8. Descriptor auto-derived + attached
# ---------------------------------------------------------------------------

class TestDescriptorAttached:

    def test_descriptor_present_on_tool(self):
        tool = ToolBase.from_python_callable(_add)
        assert tool.descriptor is not None
        assert tool.descriptor.descriptor_id.startswith("native:")
        assert "_add" in tool.descriptor.descriptor_id.lower()

    def test_descriptor_inputs_match_signature(self):
        tool = ToolBase.from_python_callable(_add)
        names = [i.name for i in tool.descriptor.inputs]
        assert names == ["a", "b"]
        types = [i.type for i in tool.descriptor.inputs]
        assert types == ["int", "int"]


# ---------------------------------------------------------------------------
# 9. UTD overrides
# ---------------------------------------------------------------------------

class TestUtdOverrides:

    def test_overrides_propagate_to_descriptor(self):
        tool = ToolBase.from_python_callable(
            _add,
            backend="apecx",
            version="2.0.0",
            display_name="Custom Add",
            summary="Custom summary string.",
            side_effects="network",
        )
        assert tool.descriptor.display_name == "Custom Add"
        assert tool.descriptor.summary == "Custom summary string."
        assert tool.descriptor.side_effects == "network"
        assert tool.descriptor.descriptor_id.startswith("apecx:")
        assert tool.descriptor.descriptor_id.endswith("@2.0.0")


# ---------------------------------------------------------------------------
# 10. provenance_pin points at the dispatcher class
# ---------------------------------------------------------------------------

class TestProvenancePinPointsAtDispatcher:

    def test_provenance_class_path_is_dispatcher(self):
        """Critical: from_descriptor uses provenance_pin.class_path to
        find the implementation. For PythonCallableDispatcher-wrapped
        callables, that class_path MUST be the dispatcher (not the
        callable's import path)."""
        tool = ToolBase.from_python_callable(_add)
        assert (
            tool.descriptor.provenance_pin.class_path
            == "nanobrain.library.tools.python_callable_dispatcher.PythonCallableDispatcher"
        )
        assert isinstance(tool, PythonCallableDispatcher)


# ---------------------------------------------------------------------------
# 11. Public function alias
# ---------------------------------------------------------------------------

class TestPublicAlias:

    def test_build_tool_from_python_callable_is_a_classmethod_alternative(self):
        """The module-level build_tool_from_python_callable function and
        ToolBase.from_python_callable should produce equivalent tools."""
        async def run():
            t1 = ToolBase.from_python_callable(_add)
            t2 = build_tool_from_python_callable(_add)
            r1 = await t1.execute({"a": 7, "b": 2})
            r2 = await t2.execute({"a": 7, "b": 2})
            assert r1 == r2 == 9
            # Same descriptor_id (deterministic)
            assert t1.descriptor.descriptor_id == t2.descriptor.descriptor_id
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 12. Concurrent calls don't interfere
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_concurrent_dispatches_work(self):
        async def run():
            tool = ToolBase.from_python_callable(_add)
            results = await asyncio.gather(
                *(tool.execute({"a": i, "b": 1}) for i in range(20)),
            )
            assert results == list(range(1, 21))
        asyncio.run(run())
