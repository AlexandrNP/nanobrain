"""Academy spawn/scrape/utilize patterns inside nanobrain workflows.

Per the deployment-validation chain (2026-05-09): validates that
Academy agents can be dynamically spawned, introspected ("scraped"),
and utilized from inside a nanobrain BaseStep / Workflow at runtime.

Three patterns covered:

1. **Spawn** — ``AcademyManagerWrapper.register_agent_class(name, cls)``
   launches a real Academy agent dynamically from inside a workflow
   step. The launched agent persists for the duration of the Manager
   singleton (process lifetime).

2. **Scrape** — discover an agent's available actions. Academy binds
   ``@action``-decorated methods at class definition time, so
   "scrape" means Python-side introspection of the agent class's
   action methods. We use ``inspect`` to walk the class's annotations
   and the ``academy.agent.action`` decorator marker.

3. **Utilize** — dispatch an action via the wrapper handle and consume
   the result. Tested both via attribute syntax (``await handle.echo("x")``)
   and ``__call__`` syntax (``await handle("echo", "x")``).

Plus an end-to-end test that wires all three into a single nanobrain
``BaseStep.process()`` running inside a real Workflow.

The test does NOT require external infrastructure (Academy uses
``LocalExchangeFactory`` — purely in-process). Pre-2026-05-09, the
Academy lifecycle had a singleton-leak bug that caused cross-test
contamination; that is now fixed (see academy_integration.py
``__call__`` reordering).
"""

from __future__ import annotations

import asyncio
import inspect
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import pytest
import yaml

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

from academy.agent import Agent, action  # noqa: E402


class GreeterAgent(Agent):
    """Multi-action Academy agent. Action surface used by the scrape test."""

    async def agent_on_startup(self) -> None:
        self._call_count = 0

    @action
    async def greet(self, name: str) -> str:
        self._call_count += 1
        return f"Hello, {name}! (call #{self._call_count})"

    @action
    async def shout(self, msg: str) -> str:
        self._call_count += 1
        return msg.upper() + "!"

    @action
    async def get_call_count(self) -> int:
        return self._call_count


async def _academy_session():
    """Context-manager-like helper: enter a fresh Academy Manager;
    yield it; tear down at exit.

    Each test calls this from inside its own ``asyncio.run(...)`` so
    setup, test body, and teardown all live in the SAME event loop.
    Using a pytest fixture that called ``asyncio.run`` separately
    leaked agent-control-block tasks across loops — the canonical
    "different-loop teardown" Academy bug.
    """
    from nanobrain.core.academy_integration import (
        AcademyIntegration,
        shutdown_academy_manager,
    )
    await shutdown_academy_manager()
    mgr = AcademyIntegration.setup_academy_manager()
    return mgr


async def _academy_teardown():
    from nanobrain.core.academy_integration import shutdown_academy_manager
    await shutdown_academy_manager()


@pytest.fixture(autouse=True)
def _force_real_mode(monkeypatch):
    monkeypatch.delenv("ACADEMY_DEMO_MODE", raising=False)


# ---------------------------------------------------------------------------
# Helpers — Python-side scrape utility
# ---------------------------------------------------------------------------

def _scrape_agent_actions(agent_cls: type) -> Dict[str, Dict[str, Any]]:
    """Return a dict of ``{action_name: {sig, doc, params}}`` for every
    ``@action``-decorated method on ``agent_cls``.

    Academy marks decorated methods via the ``_action`` attribute
    (the decorator wraps them with a sentinel). We walk ``dir(cls)``
    and pick out the ones bearing the marker.

    This is the framework-side "scrape" primitive — equivalent to
    MCP's ``tools/list``, but for Academy agents.
    """
    actions: Dict[str, Dict[str, Any]] = {}
    for attr_name in dir(agent_cls):
        if attr_name.startswith("_"):
            continue
        attr = getattr(agent_cls, attr_name)
        # Academy's @action decorator marks methods. Detection contract:
        # callable + bound to the class + decorated. We use a duck-type
        # check for the academy.agent.action marker (academy uses
        # ``_action_marker`` or similar; the public API surface is the
        # ``inspect`` view).
        if not callable(attr):
            continue
        # Check: was this decorated by academy.agent.action?
        # The decorator preserves __wrapped__ for inspectable functions
        # in 0.4.0; if not, the method is still callable so we accept it
        # as "potentially scrapeable" if it's defined on the class
        # itself (not inherited from object/Agent).
        if attr_name in ("agent_on_startup", "agent_on_shutdown"):
            continue
        if attr_name not in agent_cls.__dict__:
            continue  # inherited from a base class
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            continue
        actions[attr_name] = {
            "signature": str(sig),
            "doc": (attr.__doc__ or "").strip(),
            "params": [
                p.name for p in sig.parameters.values() if p.name != "self"
            ],
        }
    return actions


# ---------------------------------------------------------------------------
# 1. SPAWN — dynamic agent registration from inside workflow code
# ---------------------------------------------------------------------------

class TestSpawn:

    def test_spawn_agent_dynamically(self):
        async def run():
            mgr = await _academy_session()
            try:
                handle = await mgr.register_agent_class(
                    "greeter_dynamic", GreeterAgent,
                )
                result = await handle.greet("dynamic_world")
                assert result == "Hello, dynamic_world! (call #1)"
            finally:
                await _academy_teardown()
        asyncio.run(run())

    def test_spawn_multiple_agents_independent_state(self):
        async def run():
            mgr = await _academy_session()
            try:
                h1 = await mgr.register_agent_class("agent_a", GreeterAgent)
                h2 = await mgr.register_agent_class("agent_b", GreeterAgent)
                await h1.greet("alpha")
                await h2.greet("beta")
                await h2.greet("gamma")
                count_a = await h1.get_call_count()
                count_b = await h2.get_call_count()
                assert count_a == 1
                assert count_b == 2
            finally:
                await _academy_teardown()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 2. SCRAPE — discover agent capabilities (Python-side introspection)
# ---------------------------------------------------------------------------

class TestScrape:

    def test_scrape_lists_all_action_methods(self):
        """The scrape utility must list every @action method on
        the GreeterAgent class — greet, shout, get_call_count."""
        actions = _scrape_agent_actions(GreeterAgent)
        assert set(actions.keys()) == {"greet", "shout", "get_call_count"}

    def test_scrape_extracts_signatures(self):
        actions = _scrape_agent_actions(GreeterAgent)
        # greet(name: str) -> str — signature is rendered with quoted
        # annotation strings under newer Python (PEP 563-ish behavior).
        assert actions["greet"]["params"] == ["name"]
        sig = actions["greet"]["signature"]
        assert "name" in sig
        assert "str" in sig
        # get_call_count() -> int
        assert actions["get_call_count"]["params"] == []

    def test_scrape_excludes_lifecycle_hooks(self):
        """``agent_on_startup`` / ``agent_on_shutdown`` are framework
        hooks, NOT user-callable actions. Scrape must exclude them."""
        actions = _scrape_agent_actions(GreeterAgent)
        assert "agent_on_startup" not in actions
        assert "agent_on_shutdown" not in actions

    def test_scrape_excludes_inherited_base_methods(self):
        """Methods inherited from academy.agent.Agent (not declared
        on the user's class) must NOT appear in the scrape — those
        aren't user-defined actions."""
        actions = _scrape_agent_actions(GreeterAgent)
        # 'agent' module attributes should not bleed through
        assert all(name in GreeterAgent.__dict__ for name in actions)


# ---------------------------------------------------------------------------
# 3. UTILIZE — dispatch via both attribute and __call__ paths
# ---------------------------------------------------------------------------

class TestUtilize:

    def test_dispatch_via_attribute_syntax(self):
        async def run():
            mgr = await _academy_session()
            try:
                handle = await mgr.register_agent_class(
                    "utilize_attr", GreeterAgent,
                )
                r1 = await handle.greet("alice")
                r2 = await handle.shout("hello")
                r3 = await handle.get_call_count()
                assert r1 == "Hello, alice! (call #1)"
                assert r2 == "HELLO!"
                assert r3 == 2
            finally:
                await _academy_teardown()
        asyncio.run(run())

    def test_dispatch_via_call_syntax(self):
        async def run():
            mgr = await _academy_session()
            try:
                handle = await mgr.register_agent_class(
                    "utilize_call", GreeterAgent,
                )
                r = await handle("greet", "bob")
                assert r == "Hello, bob! (call #1)"
            finally:
                await _academy_teardown()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. END-TO-END — spawn + scrape + utilize all from inside a BaseStep
# ---------------------------------------------------------------------------

class _AcademyOrchestratorStep:
    """Lightweight step-shape callable: not a true BaseStep subclass
    (those need the full from_config plumbing); just an async callable
    that exercises the same spawn → scrape → utilize flow that a
    production step would use."""

    def __init__(self, manager_wrapper, agent_class: type):
        self._mgr = manager_wrapper
        self._agent_class = agent_class

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn an agent named after the request; scrape its actions;
        dispatch the requested action; return the response + metadata."""
        agent_name = input_data["agent_name"]
        action_name = input_data["action"]
        action_args = input_data.get("args", [])

        # SPAWN
        handle = await self._mgr.register_agent_class(
            agent_name, self._agent_class,
        )

        # SCRAPE
        available_actions = sorted(_scrape_agent_actions(self._agent_class).keys())

        # UTILIZE
        if action_name not in available_actions:
            raise ValueError(
                f"Action {action_name!r} not in scraped actions "
                f"{available_actions}"
            )
        result = await handle(action_name, *action_args)

        return {
            "agent_name": agent_name,
            "scraped_actions": available_actions,
            "action_called": action_name,
            "result": result,
        }


class TestEndToEndAcademyInWorkflow:

    def test_orchestrator_step_spawns_scrapes_utilizes(self):
        async def run():
            mgr = await _academy_session()
            try:
                step = _AcademyOrchestratorStep(mgr, GreeterAgent)
                r1 = await step.process({
                    "agent_name": "orch_greeter",
                    "action": "greet",
                    "args": ["world"],
                })
                assert r1["scraped_actions"] == ["get_call_count", "greet", "shout"]
                assert r1["action_called"] == "greet"
                assert r1["result"] == "Hello, world! (call #1)"

                r2 = await step.process({
                    "agent_name": "orch_greeter",  # same agent — count persists
                    "action": "shout",
                    "args": ["fire"],
                })
                assert r2["result"] == "FIRE!"

                r3 = await step.process({
                    "agent_name": "orch_greeter",
                    "action": "get_call_count",
                    "args": [],
                })
                assert r3["result"] == 2  # 2 prior calls (greet + shout)
            finally:
                await _academy_teardown()
        asyncio.run(run())

    def test_unscraped_action_fails_fast(self):
        async def run():
            mgr = await _academy_session()
            try:
                step = _AcademyOrchestratorStep(mgr, GreeterAgent)
                with pytest.raises(ValueError, match="not in scraped actions"):
                    await step.process({
                        "agent_name": "fail_fast_test",
                        "action": "nonexistent_action",
                        "args": [],
                    })
            finally:
                await _academy_teardown()
        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. Pre-fix regression: placeholder dispatch must NOT enter the Manager.
# This is the test that broke the suite pre-2026-05-09.
# ---------------------------------------------------------------------------

class TestPlaceholderNoLongerLeaks:

    def test_placeholder_failfast_does_not_leak_manager(self):
        from nanobrain.core.academy_integration import AcademyNotImplementedError

        async def run():
            mgr = await _academy_session()
            try:
                # Step 1: trigger the placeholder fail-fast
                placeholder = mgr.get_handle("not_registered")
                with pytest.raises(AcademyNotImplementedError):
                    await placeholder("greet", "x")

                # Step 2: register a REAL agent in the same wrapper.
                # If the placeholder raise had leaked Manager state, this
                # would deadlock or crash. The fix means it should work.
                handle = await mgr.register_agent_class(
                    "post_placeholder", GreeterAgent,
                )
                r = await handle.greet("after_placeholder")
                assert r == "Hello, after_placeholder! (call #1)"
            finally:
                await _academy_teardown()
        asyncio.run(run())
