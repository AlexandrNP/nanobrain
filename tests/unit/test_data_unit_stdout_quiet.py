"""DataUnitMemory must write nothing to stdout during normal operations.

Regression for the 5 hard-wired ``print(f"🔗 BRUTAL TRUTH: ...")`` debug
statements that lived in ``data_unit.py`` until 2026-06-09. They were
explicitly designed to "bypass logging" and dump to ``sys.stdout`` —
which corrupts the MCP stdio JSON-RPC transport (every line that isn't
a well-formed JSON-RPC envelope makes the client disconnect).

This test pins the contract that production code paths in ``data_unit.py``
do not write to ``sys.stdout``. Logging is fine — that flows through the
nanobrain logging system to stderr/file handlers. Only raw ``print()``
and ``sys.stdout.write()`` are forbidden.

Sibling-fix history (same family of MCP-wire-corruption bug):
- G43 (2026-05-09) stripped 14 ``print(f"DEBUG: ...")`` from
  ``mcp_support.py``.
- ResilientStreamHandler (2026-05-22) protected ``logging`` against
  closed MCP stdio handles.
- This test guards the data-unit hot path (every workflow run touches it).
"""
from __future__ import annotations

import asyncio
import io
import sys
from contextlib import redirect_stdout

import pytest

from nanobrain.core.data_unit import DataUnitMemory


def _memory_config(name: str) -> dict:
    return {
        "class": "nanobrain.core.data_unit.DataUnitMemory",
        "name": name,
        "persistent": False,
        "enable_logging": False,
    }


@pytest.mark.asyncio
async def test_set_writes_nothing_to_stdout():
    """``DataUnitMemory.set()`` must not touch stdout — that path is the
    MCP JSON-RPC channel under stdio transport."""
    du = DataUnitMemory.from_config(_memory_config("test_set_quiet"))
    await du.initialize()

    buf = io.StringIO()
    with redirect_stdout(buf):
        await du.set({"hello": "world"})

    captured = buf.getvalue()
    assert captured == "", (
        f"DataUnitMemory.set() leaked to stdout: {captured!r}. "
        "This corrupts the MCP stdio JSON-RPC wire — see the file "
        "docstring for the failure mode."
    )


@pytest.mark.asyncio
async def test_multiple_sets_write_nothing_to_stdout():
    """N consecutive ``set()`` calls (the cascade pattern) stay quiet."""
    du = DataUnitMemory.from_config(_memory_config("test_multi_set_quiet"))
    await du.initialize()

    buf = io.StringIO()
    with redirect_stdout(buf):
        for i in range(10):
            await du.set({"i": i, "marker": "cascade-test"})

    captured = buf.getvalue()
    assert captured == "", (
        f"After 10 set() calls, DataUnitMemory leaked {len(captured)} "
        f"chars to stdout: {captured[:200]!r}..."
    )


@pytest.mark.asyncio
async def test_set_with_change_listener_writes_nothing_to_stdout():
    """The change-listener notification path was the worst offender —
    ``_notify_change_listeners`` previously emitted a print() per call."""
    du = DataUnitMemory.from_config(
        _memory_config("test_listener_quiet")
    )
    await du.initialize()

    received = []

    async def _listener(event):
        received.append(event)

    du.register_change_listener(_listener)

    buf = io.StringIO()
    with redirect_stdout(buf):
        await du.set({"first": 1})
        await du.set({"second": 2})
        # Give async listener tasks a tick to fire.
        await asyncio.sleep(0.05)

    captured = buf.getvalue()
    assert captured == "", (
        f"With a registered change listener, DataUnitMemory leaked to "
        f"stdout: {captured!r}"
    )
    # Sanity: the listener actually ran (we're not silently no-op-ing the test).
    assert len(received) >= 1, (
        "change listener never fired — test would have passed for the "
        "wrong reason"
    )
