"""RheaCodeUseAgent — a nanobrain Agent that uses Rhea MCP tools + web search.

The Agent-path counterpart to the Rhea Step components. Where
``ToolExecutionStep`` + ``RheaAdapter`` dispatch ONE pinned Rhea tool
inside a workflow, this agent does open-ended **tool use**: given a
task, it decides which tools to call, calls them, reads the results,
and iterates until it can answer.

It holds two kinds of tool:

* a :class:`~nanobrain.library.tools.web_search.WebSearchTool` (a
  ``ToolBase`` registered in the agent's ``tool_registry``), and
* the **live** Rhea MCP catalog, reached over an ``MCPTransport``.

Why a live MCP client, not per-tool RheaMCPDispatcher
-----------------------------------------------------

Rhea's catalog is **dynamic**. A fresh Rhea worker exposes exactly one
tool, ``find_tools`` — a meta-tool that semantic-searches Rhea's
registry and *populates* relevant tools on demand. A static per-tool
``RheaMCPDispatcher`` (materialized once from a fixed UTD) cannot track
a catalog that grows at runtime. So this agent re-queries Rhea's
``tools/list`` on every tool-use round: after the LLM calls
``find_tools``, the next round's tool list includes whatever Rhea just
surfaced. ``RheaMCPDispatcher`` remains the right choice for a
*known, fixed* Rhea tool; this agent is the right choice for
*discover-then-use*.

Framework-capacity expansion: multi-round tool use
--------------------------------------------------

nanobrain's ``SimpleAgent`` / ``ConversationalAgent`` do a
**single-round** tool loop (call LLM → run tool calls once → call LLM
once more → return). A discover-then-use agent needs MULTIPLE rounds
(``find_tools`` → re-list → call the surfaced tool → synthesize). This
agent implements that multi-round loop on top of the framework's
existing ``_call_llm`` primitive — a clean, native expansion, no
LangChain dependency.

Honest scope
------------

* The agent + its tool-use loop are complete and exercised end-to-end
  against the ``WebSearchTool`` (which is fully functional).
* The Rhea ``find_tools`` tool is *reachable* (the MCP handshake +
  ``tools/list`` + ``tools/call`` all work against a live worker), but
  ``find_tools`` *succeeding* requires the Rhea backend's embedding
  service + a populated Postgres tool registry. Against a
  minimum-viable Rhea worker (server + Redis only), ``find_tools``
  returns an error — and the agent surfaces that honestly to the LLM
  (the tool result is the error text; the LLM can fall back to web
  search). The agent code is correct; the gap is Rhea-backend
  deployment, documented in ``docs/rhea_code_use_agent.md``.
* Tool-selection quality is model-dependent. With a small local model
  (mistral-nemo) the LLM may pick the wrong tool or skip tools — this
  is measured + reported honestly, not papered over.

A FAIL-LOUD note: tool *dispatch* errors (a Rhea HTTP failure, a
WebSearchTool backend error) are caught and fed back to the LLM as the
tool's result text — NOT swallowed. The LLM sees "tool X failed: ..."
and can react. What is never hidden is a tool that silently returns
nothing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from nanobrain.core.agent import Agent, AgentConfig
from nanobrain.core.component_base import ComponentConfigurationError
from pydantic import ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

#: Rhea tool names are namespaced in the LLM's tool list so they never
#: collide with the agent's local ToolBase tools.
_RHEA_TOOL_PREFIX = "rhea__"


class RheaCodeUseAgentConfig(AgentConfig):
    """Configuration for :class:`RheaCodeUseAgent`.

    Inherits every ``AgentConfig`` field (``model``, ``system_prompt``,
    ``temperature``, ``provider``, ``base_url``, ``api_key``, ...) and
    adds the Rhea + tool-loop fields.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    # Injected by the framework's config loader at load time — must be
    # declared so extra="forbid" does not reject the assignment.
    source_path: Optional[str] = Field(default=None)

    rhea_mcp_url: Optional[str] = Field(
        default=None,
        description=(
            "URL of the Rhea MCP worker (e.g. http://localhost:3001/mcp/). "
            "Falls back to $RHEA_MCP_URL when unset. When neither is set, "
            "the agent runs with web search only (Rhea tools unavailable) "
            "and says so — it does NOT silently pretend Rhea is present."
        ),
    )
    max_tool_rounds: int = Field(
        default=6,
        ge=1,
        description=(
            "Cap on tool-use rounds per process() call. A discover-then-"
            "use flow needs several (find_tools -> re-list -> call -> "
            "synthesize); the cap prevents an infinite tool-call loop."
        ),
    )
    web_search_tool_config: Optional[str] = Field(
        default=None,
        description=(
            "Path to a WebSearchTool YAML config. When set, the agent "
            "builds + registers the web search tool at init. When unset, "
            "the caller may still register tools via register_tool()."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _strip_framework_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data.pop("class", None)
        return data


class RheaCodeUseAgent(Agent):
    """An Agent that discovers + uses Rhea MCP tools and a web search tool."""

    COMPONENT_TYPE = "rhea_code_use_agent"
    REQUIRED_CONFIG_FIELDS = ["name"]

    @classmethod
    def _get_config_class(cls):
        return RheaCodeUseAgentConfig

    def _init_from_config(
        self,
        config: RheaCodeUseAgentConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)

        self._rhea_mcp_url: Optional[str] = (
            getattr(config, "rhea_mcp_url", None)
            or os.environ.get("RHEA_MCP_URL")
        )
        self._max_tool_rounds: int = int(getattr(config, "max_tool_rounds", 6))
        self._rhea_transport = None  # lazy MCPTransport

        # Optionally build + register the web search tool now.
        ws_config = getattr(config, "web_search_tool_config", None)
        if ws_config:
            from nanobrain.library.tools.web_search import WebSearchTool  # noqa: PLC0415

            tool = WebSearchTool.from_config(ws_config)
            self.register_tool(tool)
            logger.info(
                "RheaCodeUseAgent %r registered WebSearchTool from %s",
                self.name,
                ws_config,
            )

        logger.info(
            "RheaCodeUseAgent %r initialized: rhea_mcp_url=%s max_tool_rounds=%d",
            self.name,
            self._rhea_mcp_url or "(none — web search only)",
            self._max_tool_rounds,
        )

    # ---- Rhea MCP plumbing ----------------------------------------------

    async def _ensure_rhea_transport(self):
        """Lazily build the MCPTransport to the Rhea worker."""
        if self._rhea_transport is None:
            if not self._rhea_mcp_url:
                return None
            from nanobrain.library.tools._mcp_transport import (  # noqa: PLC0415
                MCPTransport,
            )

            self._rhea_transport = MCPTransport(
                mcp_url=self._rhea_mcp_url,
                client_name=f"rhea-code-use-agent-{self.name}",
            )
        return self._rhea_transport

    async def _rhea_tool_specs(self) -> List[Dict[str, Any]]:
        """Query Rhea's LIVE tool catalog and return OpenAI tool specs.

        Re-queried every round: Rhea's catalog grows at runtime after a
        ``find_tools`` call, and this is how the agent picks that up.
        Rhea provides each tool's ``inputSchema`` directly — it IS the
        OpenAI ``parameters`` shape, so no conversion is needed.

        A Rhea connection failure here is FAIL-LOUD: an agent that
        silently dropped the entire Rhea catalog because the worker
        was unreachable would look like it ran fine while doing half
        the job.
        """
        transport = await self._ensure_rhea_transport()
        if transport is None:
            return []
        result = await transport.call("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        specs: List[Dict[str, Any]] = []
        for tool in tools:
            raw_name = tool.get("name")
            if not raw_name:
                continue
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"{_RHEA_TOOL_PREFIX}{raw_name}",
                        "description": str(tool.get("description", "")),
                        "parameters": tool.get("inputSchema")
                        or {"type": "object", "properties": {}},
                    },
                }
            )
        return specs

    def _local_tool_specs(self) -> List[Dict[str, Any]]:
        """OpenAI specs for the agent's registered ToolBase tools."""
        specs: List[Dict[str, Any]] = []
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get(tool_name)
            if tool is not None and hasattr(tool, "get_schema"):
                specs.append(tool.get_schema())
        return specs

    async def _dispatch_tool_call(
        self, name: str, args: Dict[str, Any]
    ) -> Any:
        """Route one tool call. ``rhea__*`` -> Rhea MCP; else -> registry.

        Raises on a genuine dispatch failure — the caller catches it
        and feeds the failure text back to the LLM (visible, not
        swallowed).
        """
        if name.startswith(_RHEA_TOOL_PREFIX):
            rhea_name = name[len(_RHEA_TOOL_PREFIX) :]
            transport = await self._ensure_rhea_transport()
            if transport is None:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: RheaCodeUseAgent {self.name!r} cannot "
                    f"dispatch Rhea tool {rhea_name!r} — no rhea_mcp_url "
                    f"configured and $RHEA_MCP_URL is unset."
                )
            from nanobrain.library.tools._mcp_transport import (  # noqa: PLC0415
                parse_tool_call_result,
            )

            raw = await transport.call(
                "tools/call", {"name": rhea_name, "arguments": args}
            )
            return parse_tool_call_result(raw, rhea_name)

        tool = self.tool_registry.get(name)
        if tool is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaCodeUseAgent {self.name!r} LLM requested "
                f"unknown tool {name!r}; registered: "
                f"{self.tool_registry.list_tools()}"
            )
        return await tool.execute(**args)

    # ---- Agent contract: process() --------------------------------------

    async def process(self, input_text: str, **kwargs: Any) -> str:
        """Run the multi-round tool-use loop for one task.

        Each round: assemble the combined tool list (local ToolBase
        tools + the LIVE Rhea catalog), call the LLM, dispatch any tool
        calls it requests, feed the results back. Stop when the LLM
        answers without a tool call, or at ``max_tool_rounds``.
        """
        if not self.llm_client:
            raise ComponentConfigurationError(
                f"FAIL-FAST: RheaCodeUseAgent {self.name!r} has no LLM "
                f"client. Configure provider/base_url/model (e.g. "
                f"provider: openai_compatible, base_url: "
                f"http://localhost:11434/v1, model: mistral-nemo:latest)."
            )

        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append(
                {"role": "system", "content": self.config.system_prompt}
            )
        messages.append({"role": "user", "content": input_text})

        local_specs = self._local_tool_specs()
        last_content = ""

        for round_idx in range(self._max_tool_rounds):
            # Re-query Rhea EVERY round — the catalog is dynamic.
            rhea_specs = await self._rhea_tool_specs()
            all_specs = local_specs + rhea_specs

            llm_response = await self._call_llm(
                messages, tools=all_specs or None
            )
            message = llm_response["choices"][0]["message"]
            last_content = message.get("content") or last_content
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # The LLM answered without requesting a tool — done.
                logger.info(
                    "RheaCodeUseAgent %r: finished after %d round(s)",
                    self.name,
                    round_idx + 1,
                )
                return message.get("content") or ""

            # Append the assistant's tool-call message, then dispatch.
            messages.append(message)
            for tool_call in tool_calls:
                fn = tool_call["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args = {}
                    logger.warning(
                        "RheaCodeUseAgent %r: tool %r args were not valid "
                        "JSON (%s); dispatching with {}",
                        self.name,
                        name,
                        e,
                    )
                try:
                    result = await self._dispatch_tool_call(name, args)
                    content = (
                        json.dumps(result, default=str)
                        if not isinstance(result, str)
                        else result
                    )
                except Exception as e:  # noqa: BLE001
                    # Dispatch failure is NOT swallowed — it is fed back
                    # to the LLM verbatim so the model can react (retry,
                    # pick another tool, or report the limitation).
                    content = f"Tool {name!r} failed: {type(e).__name__}: {e}"
                    logger.warning(
                        "RheaCodeUseAgent %r: tool %r dispatch failed: %s",
                        self.name,
                        name,
                        e,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": content,
                    }
                )

        # Hit the round cap with the LLM still wanting tools. Return the
        # last content + an explicit note — never silently truncate.
        logger.warning(
            "RheaCodeUseAgent %r: hit max_tool_rounds=%d without a "
            "tool-free answer",
            self.name,
            self._max_tool_rounds,
        )
        return (
            (last_content or "")
            + f"\n\n[RheaCodeUseAgent: stopped at max_tool_rounds="
            f"{self._max_tool_rounds}; the model was still requesting "
            f"tool calls. The answer above may be incomplete.]"
        )

    async def aclose(self) -> None:
        """Close the Rhea MCP transport. Idempotent."""
        if self._rhea_transport is not None:
            await self._rhea_transport.aclose()
            self._rhea_transport = None


__all__ = ["RheaCodeUseAgent", "RheaCodeUseAgentConfig"]
