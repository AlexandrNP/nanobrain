"""
Agent System for NanoBrain Framework

Provides tool-calling based AI processing with LLM integration.
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

from .executor import ExecutorBase, LocalExecutor, ExecutorConfig
from .tool import ToolBase, ToolRegistry, ToolType, ToolConfig, create_tool
from .logging_system import (
    NanoBrainLogger, get_logger, OperationType, ToolCallLog, 
    AgentConversationLog, trace_function_calls
)

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str
    description: str = ""
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    system_prompt: str = ""
    prompt_templates: Optional[Dict[str, str]] = Field(default=None, description="Templates for different prompt types")
    executor_config: Optional[ExecutorConfig] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    auto_initialize: bool = True
    debug_mode: bool = False
    enable_logging: bool = True
    log_conversations: bool = True
    log_tool_calls: bool = True


class Agent(ABC):
    """
    Base class for Agents that use tool calling for AI processing.
    
    Biological analogy: Prefrontal cortex orchestrating specialized brain regions.
    Justification: Like how the prefrontal cortex coordinates different brain
    regions for complex tasks, agents coordinate different tools for complex
    AI processing tasks.
    """
    
    def __init__(self, config: AgentConfig, executor: Optional[ExecutorBase] = None, **kwargs):
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Initialize logging system
        self.nb_logger = get_logger(f"agent.{self.name}", debug_mode=config.debug_mode)
        self.nb_logger.info(f"Initializing agent: {self.name}", agent_name=self.name, config=config.model_dump())
        
        # Executor for running the agent
        self.executor = executor or LocalExecutor(config.executor_config)
        
        # Tool registry for managing tools
        self.tool_registry = ToolRegistry()
        
        # LLM client (will be set during initialization)
        self.llm_client = None
        
        # State management
        self._is_initialized = False
        self._conversation_history: List[Dict[str, Any]] = []
        self._execution_count = 0
        self._error_count = 0
        self._total_tokens_used = 0
        self._total_llm_calls = 0
        
        # Performance tracking
        self._start_time = time.time()
        self._last_activity_time = time.time()
        
    async def initialize(self) -> None:
        """Initialize the agent and its components."""
        if self._is_initialized:
            self.nb_logger.debug(f"Agent {self.name} already initialized")
            return
        
        async with self.nb_logger.async_execution_context(
            OperationType.AGENT_PROCESS, 
            f"{self.name}.initialize"
        ) as context:
            # Initialize executor
            self.nb_logger.debug(f"Initializing executor for agent {self.name}")
            await self.executor.initialize()
            
            # Initialize LLM client
            self.nb_logger.debug(f"Initializing LLM client for agent {self.name}")
            await self._initialize_llm_client()
            
            # Register tools from configuration
            self.nb_logger.debug(f"Registering {len(self.config.tools)} tools for agent {self.name}")
            for i, tool_config in enumerate(self.config.tools):
                try:
                    await self._register_tool_from_config(tool_config)
                    self.nb_logger.debug(f"Registered tool {i+1}/{len(self.config.tools)}")
                except Exception as e:
                    self.nb_logger.error(f"Failed to register tool {i+1}: {e}", tool_config=tool_config)
            
            # Initialize all tools
            self.nb_logger.debug(f"Initializing all tools for agent {self.name}")
            await self.tool_registry.initialize_all()
            
            self._is_initialized = True
            context.metadata['tools_count'] = len(self.tool_registry.list_tools())
            context.metadata['llm_client_type'] = type(self.llm_client).__name__ if self.llm_client else None
            
        self.nb_logger.info(f"Agent {self.name} initialized successfully", 
                           tools_count=len(self.tool_registry.list_tools()),
                           has_llm_client=self.llm_client is not None)
    
    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        async with self.nb_logger.async_execution_context(
            OperationType.AGENT_PROCESS, 
            f"{self.name}.shutdown"
        ) as context:
            # Log final statistics
            uptime_seconds = time.time() - self._start_time
            self.nb_logger.info(f"Agent {self.name} shutting down", 
                               uptime_seconds=uptime_seconds,
                               execution_count=self._execution_count,
                               error_count=self._error_count,
                               total_tokens_used=self._total_tokens_used,
                               total_llm_calls=self._total_llm_calls)
            
            # Shutdown tools
            await self.tool_registry.shutdown_all()
            
            # Shutdown executor
            await self.executor.shutdown()
            
            self._is_initialized = False
            context.metadata['final_stats'] = {
                'uptime_seconds': uptime_seconds,
                'execution_count': self._execution_count,
                'error_count': self._error_count,
                'total_tokens_used': self._total_tokens_used,
                'total_llm_calls': self._total_llm_calls
            }
    
    async def _initialize_llm_client(self) -> None:
        """Initialize the LLM client."""
        try:
            # Try to import OpenAI client
            from openai import AsyncOpenAI
            import os
            
            # Check if API key is available
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.nb_logger.warning(f"No OpenAI API key found for agent {self.name}. Set OPENAI_API_KEY environment variable.")
                self.llm_client = None
                return
            
            # Create OpenAI client with API key
            self.llm_client = AsyncOpenAI(api_key=api_key)
            self.nb_logger.debug(f"Agent {self.name} initialized with OpenAI client")
            
            # Test the client with a simple call to verify it works
            try:
                # Make a minimal test call to verify the client works
                test_response = await self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                self.nb_logger.debug(f"Agent {self.name} OpenAI client test successful")
            except Exception as e:
                self.nb_logger.warning(f"OpenAI client test failed for agent {self.name}: {e}")
                self.llm_client = None
                
        except ImportError:
            self.nb_logger.warning("OpenAI client not available. Install with: pip install openai")
            # Could add other LLM clients here (Anthropic, etc.)
            self.llm_client = None
        except Exception as e:
            self.nb_logger.error(f"Failed to initialize LLM client for agent {self.name}: {e}")
            self.llm_client = None
    
    async def _register_tool_from_config(self, tool_config: Dict[str, Any]) -> None:
        """Register a tool from configuration."""
        tool_type = ToolType(tool_config.get('tool_type', 'function'))
        
        self.nb_logger.debug(f"Registering tool from config", 
                            tool_type=tool_type.value, 
                            tool_name=tool_config.get('name', 'unnamed'))
        
        # Create tool configuration
        config = ToolConfig(**tool_config)
        
        # Create and register tool based on type
        if tool_type == ToolType.FUNCTION:
            # Function tools need to be provided externally
            self.nb_logger.warning(f"Function tool {config.name} needs to be registered manually")
        elif tool_type == ToolType.AGENT:
            # Agent tools need agent instance
            self.nb_logger.warning(f"Agent tool {config.name} needs to be registered manually")
        else:
            # Other tool types can be created from config
            tool = create_tool(tool_type, config, **tool_config)
            self.tool_registry.register(tool)
            self.nb_logger.debug(f"Successfully registered tool: {config.name}")
    
    def register_tool(self, tool: ToolBase) -> None:
        """Register a tool with the agent."""
        self.tool_registry.register(tool)
        self.nb_logger.info(f"Agent {self.name} registered tool: {tool.name}", 
                           tool_name=tool.name, 
                           tool_type=type(tool).__name__)
    
    def register_function_tool(self, func: callable, name: str, description: str = "", 
                             parameters: Optional[Dict[str, Any]] = None) -> None:
        """Register a function as a tool."""
        self.nb_logger.debug(f"Registering function tool: {name}", 
                            function_name=func.__name__ if hasattr(func, '__name__') else 'unknown')
        
        config = ToolConfig(
            tool_type=ToolType.FUNCTION,
            name=name,
            description=description,
            parameters=parameters or {}
        )
        tool = create_tool(ToolType.FUNCTION, config, func=func)
        self.register_tool(tool)
    
    def register_agent_tool(self, agent: 'Agent', name: Optional[str] = None, 
                           description: Optional[str] = None) -> None:
        """Register another agent as a tool."""
        tool_name = name or agent.name
        self.nb_logger.debug(f"Registering agent tool: {tool_name}", 
                            target_agent=agent.name)
        
        config = ToolConfig(
            tool_type=ToolType.AGENT,
            name=tool_name,
            description=description or agent.description,
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input text for the agent"
                    }
                },
                "required": ["input"]
            }
        )
        tool = create_tool(ToolType.AGENT, config, agent=agent)
        self.register_tool(tool)
    
    @abstractmethod
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input text and return response.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Response text
        """
        pass
    
    async def execute(self, input_text: str, **kwargs) -> str:
        """Execute the agent using the configured executor."""
        if not self._is_initialized:
            await self.initialize()
        
        async with self.nb_logger.async_execution_context(
            OperationType.AGENT_PROCESS, 
            f"{self.name}.execute",
            input_length=len(input_text),
            kwargs_keys=list(kwargs.keys())
        ) as context:
            try:
                # Process using executor
                result = await self.executor.execute(self._execute_process, input_text=input_text, **kwargs)
                
                self._execution_count += 1
                self._last_activity_time = time.time()
                
                context.metadata['result_length'] = len(result) if isinstance(result, str) else None
                context.metadata['execution_count'] = self._execution_count
                
                self.nb_logger.debug(f"Agent {self.name} executed successfully", 
                                   execution_count=self._execution_count,
                                   result_length=len(result) if isinstance(result, str) else None)
                return result
                
            except Exception as e:
                self._error_count += 1
                context.metadata['error_count'] = self._error_count
                self.nb_logger.error(f"Agent {self.name} execution failed: {e}", 
                                   error_type=type(e).__name__,
                                   error_count=self._error_count)
                raise
    
    async def _execute_process(self, input_text: str, **kwargs) -> str:
        """Wrapper for process method to be executed by executor."""
        return await self.process(input_text, **kwargs)
    
    async def _call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Call the LLM with messages and optional tools."""
        if not self.llm_client:
            raise RuntimeError("LLM client not initialized")
        
        async with self.nb_logger.async_execution_context(
            OperationType.LLM_CALL, 
            f"{self.name}.llm_call",
            message_count=len(messages),
            has_tools=tools is not None,
            tool_count=len(tools) if tools else 0
        ) as context:
            try:
                # Prepare the request
                request_params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                }
                
                if self.config.max_tokens:
                    request_params["max_tokens"] = self.config.max_tokens
                
                if tools:
                    request_params["tools"] = tools
                    request_params["tool_choice"] = "auto"
                
                self.nb_logger.debug(f"Making LLM call", 
                                   model=self.config.model,
                                   message_count=len(messages),
                                   tool_count=len(tools) if tools else 0)
                
                # Make the API call
                response = await self.llm_client.chat.completions.create(**request_params)
                
                # Track usage
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = response.usage.total_tokens
                    self._total_tokens_used += tokens_used
                    context.metadata['tokens_used'] = tokens_used
                    context.metadata['total_tokens_used'] = self._total_tokens_used
                
                self._total_llm_calls += 1
                context.metadata['total_llm_calls'] = self._total_llm_calls
                
                # Convert response to dict
                response_dict = {
                    "id": response.id,
                    "model": response.model,
                    "choices": [
                        {
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    } for tc in (choice.message.tool_calls or [])
                                ] if choice.message.tool_calls else None
                            },
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices
                    ],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None
                }
                
                self.nb_logger.debug(f"LLM call completed", 
                                   tokens_used=tokens_used if 'tokens_used' in locals() else None,
                                   finish_reason=response_dict["choices"][0]["finish_reason"] if response_dict["choices"] else None)
                
                return response_dict
                
            except Exception as e:
                self.nb_logger.error(f"LLM call failed: {e}", 
                                   error_type=type(e).__name__)
                raise
    
    async def _execute_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            
            async with self.nb_logger.async_execution_context(
                OperationType.TOOL_CALL, 
                f"{self.name}.tool_call.{tool_name}",
                tool_name=tool_name,
                arguments_length=len(tool_args_str)
            ) as context:
                try:
                    # Parse arguments
                    tool_args = json.loads(tool_args_str)
                    context.metadata['parsed_args'] = tool_args
                    
                    self.nb_logger.debug(f"Executing tool call: {tool_name}", 
                                       tool_name=tool_name,
                                       arguments=tool_args)
                    
                    # Get tool from registry
                    tool = self.tool_registry.get_tool(tool_name)
                    if not tool:
                        raise ValueError(f"Tool '{tool_name}' not found")
                    
                    # Execute tool
                    start_time = time.time()
                    result = await tool.execute(**tool_args)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    # Log tool call
                    if self.config.log_tool_calls:
                        self.nb_logger.log_tool_call(
                            tool_name=tool_name,
                            parameters=tool_args,
                            result=result,
                            duration_ms=duration_ms
                        )
                    
                    context.metadata['result_type'] = type(result).__name__
                    context.metadata['duration_ms'] = duration_ms
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_name,
                        "content": str(result)
                    })
                    
                    self.nb_logger.debug(f"Tool call completed: {tool_name}", 
                                       duration_ms=duration_ms,
                                       result_type=type(result).__name__)
                    
                except Exception as e:
                    error_msg = f"Tool call failed: {e}"
                    context.metadata['error'] = str(e)
                    
                    # Log failed tool call
                    if self.config.log_tool_calls:
                        self.nb_logger.log_tool_call(
                            tool_name=tool_name,
                            parameters=tool_args if 'tool_args' in locals() else {},
                            error=str(e)
                        )
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_name,
                        "content": error_msg
                    })
                    
                    self.nb_logger.error(f"Tool call failed: {tool_name}", 
                                       error=str(e),
                                       error_type=type(e).__name__)
        
        return results
    
    def add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = {"role": role, "content": content, "timestamp": time.time()}
        self._conversation_history.append(message)
        self.nb_logger.debug(f"Added message to conversation", 
                           role=role, 
                           content_length=len(content),
                           conversation_length=len(self._conversation_history))
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        old_length = len(self._conversation_history)
        self._conversation_history.clear()
        self.nb_logger.debug(f"Cleared conversation history", 
                           previous_length=old_length)
    
    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._is_initialized
    
    @property
    def execution_count(self) -> int:
        """Get the number of executions."""
        return self._execution_count
    
    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return self._error_count
    
    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.tool_registry.list_tools()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        uptime_seconds = time.time() - self._start_time
        idle_seconds = time.time() - self._last_activity_time
        
        return {
            "uptime_seconds": uptime_seconds,
            "idle_seconds": idle_seconds,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (self._execution_count - self._error_count) / max(self._execution_count, 1),
            "total_tokens_used": self._total_tokens_used,
            "total_llm_calls": self._total_llm_calls,
            "avg_tokens_per_call": self._total_tokens_used / max(self._total_llm_calls, 1),
            "conversation_length": len(self._conversation_history),
            "available_tools": len(self.available_tools)
        }


class SimpleAgent(Agent):
    """Simple agent that processes input without conversation history."""
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Process input text and return response."""
        async with self.nb_logger.async_execution_context(
            OperationType.AGENT_PROCESS, 
            f"{self.name}.process",
            input_length=len(input_text),
            agent_type="SimpleAgent"
        ) as context:
            if not self.llm_client:
                # Fallback for when LLM is not available
                response = f"Echo from {self.name}: {input_text}"
                self.nb_logger.warning(f"No LLM client available, using echo response")
            else:
                # Prepare messages
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                messages.append({"role": "user", "content": input_text})
                
                # Get available tools
                tools = None
                if self.tool_registry.list_tools():
                    tools = []
                    for tool_name in self.tool_registry.list_tools():
                        tool = self.tool_registry.get_tool(tool_name)
                        if tool and hasattr(tool, 'get_openai_function_schema'):
                            tools.append(tool.get_openai_function_schema())
                
                # Call LLM
                llm_response = await self._call_llm(messages, tools)
                
                # Process response
                choice = llm_response["choices"][0]
                message = choice["message"]
                
                if message.get("tool_calls"):
                    # Execute tool calls
                    tool_results = await self._execute_tool_calls(message["tool_calls"])
                    
                    # Add tool results to messages and call LLM again
                    messages.append(message)
                    messages.extend(tool_results)
                    
                    final_response = await self._call_llm(messages)
                    response = final_response["choices"][0]["message"]["content"]
                else:
                    response = message["content"]
            
            # Log conversation if enabled
            if self.config.log_conversations:
                self.nb_logger.log_agent_conversation(
                    agent_name=self.name,
                    input_text=input_text,
                    response_text=response,
                    llm_calls=self._total_llm_calls,
                    total_tokens=self._total_tokens_used,
                    duration_ms=context.duration_ms
                )
            
            context.metadata['response_length'] = len(response) if response else 0
            return response or ""


class ConversationalAgent(Agent):
    """Agent that maintains conversation history and context."""
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.max_history_length = kwargs.get('max_history_length', 10)
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Process input text with conversation history."""
        async with self.nb_logger.async_execution_context(
            OperationType.AGENT_PROCESS, 
            f"{self.name}.process",
            input_length=len(input_text),
            agent_type="ConversationalAgent",
            history_length=len(self._conversation_history)
        ) as context:
            if not self.llm_client:
                # Fallback for when LLM is not available
                response = f"Conversational echo from {self.name}: {input_text}"
                self.nb_logger.warning(f"No LLM client available, using echo response")
            else:
                # Prepare messages with history
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                
                # Add conversation history (limited)
                history_start = max(0, len(self._conversation_history) - self.max_history_length)
                for msg in self._conversation_history[history_start:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current input
                messages.append({"role": "user", "content": input_text})
                
                # Get available tools
                tools = None
                if self.tool_registry.list_tools():
                    tools = []
                    for tool_name in self.tool_registry.list_tools():
                        tool = self.tool_registry.get_tool(tool_name)
                        if tool and hasattr(tool, 'get_openai_function_schema'):
                            tools.append(tool.get_openai_function_schema())
                
                # Call LLM
                llm_response = await self._call_llm(messages, tools)
                
                # Process response
                choice = llm_response["choices"][0]
                message = choice["message"]
                
                if message.get("tool_calls"):
                    # Execute tool calls
                    tool_results = await self._execute_tool_calls(message["tool_calls"])
                    
                    # Add tool results to messages and call LLM again
                    messages.append(message)
                    messages.extend(tool_results)
                    
                    final_response = await self._call_llm(messages)
                    response = final_response["choices"][0]["message"]["content"]
                else:
                    response = message["content"]
                
                # Update conversation history
                self.add_to_conversation("user", input_text)
                if response:
                    self.add_to_conversation("assistant", response)
            
            # Log conversation if enabled
            if self.config.log_conversations:
                self.nb_logger.log_agent_conversation(
                    agent_name=self.name,
                    input_text=input_text,
                    response_text=response,
                    llm_calls=self._total_llm_calls,
                    total_tokens=self._total_tokens_used,
                    duration_ms=context.duration_ms
                )
            
            context.metadata['response_length'] = len(response) if response else 0
            context.metadata['final_history_length'] = len(self._conversation_history)
            return response or ""


def create_agent(agent_type: str, config: AgentConfig, **kwargs) -> Agent:
    """
    Factory function to create agents of different types.
    
    Args:
        agent_type: Type of agent ('simple' or 'conversational')
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Agent instance
    """
    logger = get_logger("agent.factory")
    logger.info(f"Creating agent: {config.name}", 
               agent_type=agent_type, 
               agent_name=config.name)
    
    if agent_type.lower() == "simple":
        return SimpleAgent(config, **kwargs)
    elif agent_type.lower() == "conversational":
        return ConversationalAgent(config, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 