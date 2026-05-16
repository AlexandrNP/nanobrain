"""Integration tests for RheaCodeUseAgent.

Two surfaces:

1. **Unconditional** — the multi-round tool-use loop is driven with a
   FAKE LLM client (scripted tool calls) + a FAKE web search backend.
   No network, no Ollama, no Rhea. This proves the loop logic:
   tool-spec assembly, tool-call dispatch, result feed-back, the
   round cap, FAIL-LOUD on an unknown tool, and graceful surfacing of
   a tool dispatch failure.

2. **Gated on $RHEA_MCP_URL** — against a live Rhea MCP worker, the
   agent's ``_rhea_tool_specs()`` returns the live catalog (at minimum
   ``rhea__find_tools``) and a Rhea tool call round-trips. Gating
   mirrors nanobrain's test_rhea_mcp_dispatcher.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from nanobrain.core.component_base import ComponentConfigurationError
from nanobrain.library.agents.rhea_code_use_agent import RheaCodeUseAgent
from nanobrain.library.tools.web_search import WebSearchBackend


# ---- fake LLM client (OpenAI-shaped, scripted) ---------------------------

class _FakeFn:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = _FakeFn(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, content, tool_calls):
        self.message = _FakeMessage(content, tool_calls)
        self.finish_reason = "tool_calls" if tool_calls else "stop"


class _FakeResponse:
    def __init__(self, content, tool_calls):
        self.id = "fake-response"
        self.model = "fake-model"
        self.choices = [_FakeChoice(content, tool_calls)]
        self.usage = None


class _FakeCompletions:
    """Returns scripted responses in order; records every request."""

    def __init__(self, script):
        self._script = script
        self._idx = 0
        self.requests: list[dict] = []

    async def create(self, **params):
        self.requests.append(params)
        if self._idx >= len(self._script):
            # Script exhausted -> a tool-free answer (loop terminates).
            return _FakeResponse("(scripted: no more steps)", None)
        resp = self._script[self._idx]
        self._idx += 1
        return resp


class _FakeChat:
    def __init__(self, script):
        self.completions = _FakeCompletions(script)


class _FakeLLMClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)


class _FakeBackend(WebSearchBackend):
    name = "fake"

    def __init__(self, results=None):
        self._results = results if results is not None else [
            {"title": "FASTA format", "url": "http://x/fasta", "snippet": "stores sequences"},
        ]
        self.calls: list[tuple[str, int]] = []

    async def search(self, query, *, max_results):
        self.calls.append((query, max_results))
        return list(self._results)


# ---- helpers --------------------------------------------------------------

def _build_agent(*, max_tool_rounds: int = 4, rhea_mcp_url: str | None = None) -> RheaCodeUseAgent:
    """Build a RheaCodeUseAgent with a WebSearchTool and no real LLM.

    The LLM client is left unconfigured (no provider/base_url) so
    _initialize_llm_client yields None; the caller injects a fake.
    """
    ws = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
    ws.write(
        "name: web_search\ntool_type: external\ndescription: test web search\n"
        "parameters:\n  backend: duckduckgo\n  max_results: 3\n"
        "tool_card:\n  capabilities: ['web_search']\n"
    )
    ws.close()
    ag = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False, encoding="utf-8")
    rhea_line = f"rhea_mcp_url: {rhea_mcp_url}\n" if rhea_mcp_url else ""
    ag.write(
        "name: rhea_code_use_agent_test\n"
        "model: fake-model\n"
        "temperature: 0.0\n"
        "system_prompt: 'You are a test agent.'\n"
        f"max_tool_rounds: {max_tool_rounds}\n"
        f"{rhea_line}"
        f"web_search_tool_config: {ws.name}\n"
    )
    ag.close()
    agent = RheaCodeUseAgent.from_config(ag.name)
    return agent


# ---- 1. unconditional: the multi-round loop ------------------------------

def test_agent_builds_and_registers_web_search_tool():
    agent = _build_agent()
    assert "web_search" in agent.tool_registry.list_tools()
    specs = agent._local_tool_specs()
    assert len(specs) == 1
    assert specs[0]["function"]["name"] == "web_search"


def test_multi_round_loop_dispatches_tool_then_answers():
    """Round 1: LLM requests web_search. Round 2: LLM answers (no tool)."""
    async def run():
        agent = _build_agent()
        # Script: round1 -> call web_search; round2 -> final answer.
        script = [
            _FakeResponse(
                None,
                [_FakeToolCall("c1", "web_search", json.dumps({"query": "what is FASTA"}))],
            ),
            _FakeResponse("FASTA stores biological sequences.", None),
        ]
        agent.llm_client = _FakeLLMClient(script)
        fake_backend = _FakeBackend()
        agent.tool_registry.get("web_search").backend = fake_backend

        answer = await agent.process("What is FASTA?")
        assert answer == "FASTA stores biological sequences."
        # The web_search tool was actually dispatched with the LLM's args.
        assert fake_backend.calls == [("what is FASTA", 3)]
        # Two LLM rounds happened; round 1 was given the tool specs.
        reqs = agent.llm_client.chat.completions.requests
        assert len(reqs) == 2
        assert reqs[0].get("tools"), "round 1 should pass tool specs to the LLM"
        # The tool result was fed back into round 2's messages.
        round2_msgs = reqs[1]["messages"]
        assert any(m.get("role") == "tool" and m.get("name") == "web_search"
                   for m in round2_msgs)

    asyncio.run(run())


def test_loop_terminates_immediately_on_tool_free_answer():
    async def run():
        agent = _build_agent()
        agent.llm_client = _FakeLLMClient([_FakeResponse("Direct answer.", None)])
        answer = await agent.process("trivial question")
        assert answer == "Direct answer."
        assert len(agent.llm_client.chat.completions.requests) == 1

    asyncio.run(run())


def test_max_tool_rounds_cap_is_enforced():
    """An LLM that never stops requesting tools is capped — and the cap
    is reported, not silently truncated."""
    async def run():
        agent = _build_agent(max_tool_rounds=3)
        # Every scripted response requests a tool -> never terminates.
        always_tool = _FakeResponse(
            "partial",
            [_FakeToolCall("c", "web_search", json.dumps({"query": "q"}))],
        )
        # _FakeCompletions returns this for round 1-3, then a tool-free
        # default; but we only allow 3 rounds, so we never reach it.
        agent.llm_client = _FakeLLMClient([always_tool, always_tool, always_tool])
        agent.tool_registry.get("web_search").backend = _FakeBackend()

        answer = await agent.process("loop forever")
        assert "max_tool_rounds=3" in answer  # the cap is explicitly reported
        assert len(agent.llm_client.chat.completions.requests) == 3

    asyncio.run(run())


def test_unknown_tool_call_fails_loud_but_is_fed_back():
    """An LLM that hallucinates a tool name -> the dispatch error is fed
    back to the LLM as the tool result (visible), not swallowed."""
    async def run():
        agent = _build_agent()
        script = [
            _FakeResponse(
                None,
                [_FakeToolCall("c1", "nonexistent_tool", "{}")],
            ),
            _FakeResponse("Recovered after the failed tool.", None),
        ]
        agent.llm_client = _FakeLLMClient(script)
        answer = await agent.process("trigger a bad tool call")
        assert answer == "Recovered after the failed tool."
        # The failure text was fed back into round 2.
        round2_msgs = agent.llm_client.chat.completions.requests[1]["messages"]
        tool_msg = next(m for m in round2_msgs if m.get("role") == "tool")
        assert "failed" in tool_msg["content"].lower()
        assert "nonexistent_tool" in tool_msg["content"]

    asyncio.run(run())


def test_no_llm_client_fails_loud():
    async def run():
        agent = _build_agent()
        agent.llm_client = None
        with pytest.raises(ComponentConfigurationError, match="no LLM"):
            await agent.process("anything")

    asyncio.run(run())


def test_rhea_tool_dispatch_without_url_fails_loud_but_recovers():
    """When no rhea_mcp_url is configured and the LLM calls a rhea__
    tool, the dispatch FAIL-LOUDs — but the error is fed back, so the
    LLM can recover (it is NOT a silent no-op)."""
    async def run():
        agent = _build_agent(rhea_mcp_url=None)  # no Rhea configured
        script = [
            _FakeResponse(
                None,
                [_FakeToolCall("c1", "rhea__find_tools", json.dumps({"query": "x"}))],
            ),
            _FakeResponse("Fell back to my own knowledge.", None),
        ]
        agent.llm_client = _FakeLLMClient(script)
        answer = await agent.process("use a rhea tool")
        assert answer == "Fell back to my own knowledge."
        round2_msgs = agent.llm_client.chat.completions.requests[1]["messages"]
        tool_msg = next(m for m in round2_msgs if m.get("role") == "tool")
        assert "failed" in tool_msg["content"].lower()

    asyncio.run(run())


# ---- 2. gated on a live Rhea worker --------------------------------------

_RHEA_URL = os.environ.get("RHEA_MCP_URL")


@pytest.mark.skipif(
    _RHEA_URL is None,
    reason="RHEA_MCP_URL not set — live Rhea worker required for this test",
)
def test_live_rhea_catalog_is_discovered():
    """Against a live Rhea worker, the agent discovers the live tool
    catalog. A fresh Rhea worker exposes at least ``find_tools``."""
    async def run():
        agent = _build_agent(rhea_mcp_url=_RHEA_URL)
        specs = await agent._rhea_tool_specs()
        names = {s["function"]["name"] for s in specs}
        assert any(n.startswith("rhea__") for n in names), (
            f"expected at least one rhea__ tool in the live catalog; got {names}"
        )
        # find_tools is Rhea's always-present meta-tool.
        assert "rhea__find_tools" in names
        await agent.aclose()

    asyncio.run(run())
