"""BaseStep.run_blocking — run sync work off the event loop so streaming isn't starved.

Pins that a sync call offloaded via run_blocking does NOT block the loop (the 2110s-gap fix:
a sync globus_sdk call under LocalExecutor froze the loop, starving progress + keepalive).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


def _make_step():
    from nanobrain.core.step import BaseStep

    class _S(BaseStep):
        async def process(self, input_data: dict[str, Any], **kwargs: Any) -> Any:
            return {}

    _S._allow_direct_instantiation = True
    try:
        step = object.__new__(_S)
    finally:
        _S._allow_direct_instantiation = False
    step.name = "s"
    step._step_config_object = None
    step.config = None
    return step


def test_run_blocking_returns_value_and_propagates_exception():
    step = _make_step()

    async def _go():
        assert await step.run_blocking(lambda a, b: a + b, 2, 3) == 5
        raised = False
        try:
            await step.run_blocking(lambda: (_ for _ in ()).throw(ValueError("boom")))
        except ValueError as exc:
            raised = "boom" in str(exc)
        assert raised, "run_blocking must propagate the sync callable's exception"

    asyncio.run(_go())


def test_run_blocking_does_not_block_the_loop():
    """While a 0.3s blocking sleep runs via run_blocking, a concurrent async ticker keeps
    ticking — proving the event loop is free (a direct time.sleep would freeze it)."""
    step = _make_step()

    async def _go():
        ticks = 0

        async def _ticker():
            nonlocal ticks
            for _ in range(100):
                await asyncio.sleep(0.02)
                ticks += 1

        t = asyncio.create_task(_ticker())
        await step.run_blocking(time.sleep, 0.3)
        t.cancel()
        return ticks

    ticks = asyncio.run(_go())
    assert ticks >= 4, f"loop was blocked during run_blocking (only {ticks} ticks)"
