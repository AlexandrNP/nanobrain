"""
Tool System for NanoBrain Framework

Provides tool interface and adapters for different tool frameworks.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Union
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools."""
    FUNCTION = "function"
    AGENT = "agent"
    STEP = "step"
    EXTERNAL = "external"
    LANGCHAIN = "langchain"


class ToolConfig(BaseModel):
    """Configuration for tools."""
    tool_type: ToolType = ToolType.FUNCTION
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = True
    timeout: Optional[float] = None
    
    class Config:
        use_enum_values = True


class ToolBase(ABC):
    """
    Base class for tools that can be used by Agents.
    
    Biological analogy: Specialized neural circuits for specific functions.
    Justification: Like how the brain has specialized circuits for vision, 
    language, etc., tools provide specialized functionality for agents.
    """
    
    def __init__(self, config: ToolConfig, **kwargs):
        self.config = config
        self.name = config.name
        self.description = config.description
        self._is_initialized = False
        self._call_count = 0
        self._error_count = 0
        
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the tool."""
        if not self._is_initialized:
            self._is_initialized = True
            logger.debug(f"Tool {self.name} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the tool."""
        self._is_initialized = False
        logger.debug(f"Tool {self.name} shutdown")
    
    @property
    def is_initialized(self) -> bool:
        """Check if tool is initialized."""
        return self._is_initialized
    
    @property
    def call_count(self) -> int:
        """Get number of tool calls."""
        return self._call_count
    
    @property
    def error_count(self) -> int:
        """Get number of tool errors."""
        return self._error_count
    
    async def _record_call(self, success: bool = True) -> None:
        """Record tool call statistics."""
        self._call_count += 1
        if not success:
            self._error_count += 1
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.config.parameters
            }
        }


class FunctionTool(ToolBase):
    """
    Tool that wraps a Python function.
    """
    
    def __init__(self, func: Callable, config: ToolConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.func = func
        
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            if self.config.async_execution and asyncio.iscoroutinefunction(self.func):
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        self.func(**kwargs), 
                        timeout=self.config.timeout
                    )
                else:
                    result = await self.func(**kwargs)
            else:
                # Run in thread pool for sync functions
                loop = asyncio.get_event_loop()
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: self.func(**kwargs)),
                        timeout=self.config.timeout
                    )
                else:
                    result = await loop.run_in_executor(None, lambda: self.func(**kwargs))
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"FunctionTool {self.name} execution failed: {e}")
            raise


class AgentTool(ToolBase):
    """
    Tool that wraps another Agent for agent-to-agent interaction.
    """
    
    def __init__(self, agent: Any, config: ToolConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.agent = agent
        
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped agent."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Extract the main input for the agent
            input_text = kwargs.get('input', kwargs.get('query', kwargs.get('text', '')))
            
            if hasattr(self.agent, 'process'):
                result = await self.agent.process(input_text, **kwargs)
            elif hasattr(self.agent, 'execute'):
                result = await self.agent.execute(input_text, **kwargs)
            else:
                raise ValueError(f"Agent {self.agent.name} has no process or execute method")
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"AgentTool {self.name} execution failed: {e}")
            raise


class StepTool(ToolBase):
    """
    Tool that wraps a Step for step-based processing.
    """
    
    def __init__(self, step: Any, config: ToolConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.step = step
        
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped step."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Set input data in step's input data units
            if hasattr(self.step, 'input_data_units') and self.step.input_data_units:
                for i, (key, value) in enumerate(kwargs.items()):
                    if i < len(self.step.input_data_units):
                        await self.step.input_data_units[i].set(value)
            
            # Execute the step
            if hasattr(self.step, 'execute'):
                result = await self.step.execute()
            else:
                raise ValueError(f"Step {self.step.name} has no execute method")
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"StepTool {self.name} execution failed: {e}")
            raise


class LangChainTool(ToolBase):
    """
    Adapter for LangChain tools.
    """
    
    def __init__(self, langchain_tool: Any, config: ToolConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.langchain_tool = langchain_tool
        
    async def execute(self, **kwargs) -> Any:
        """Execute the LangChain tool."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # LangChain tools typically expect a single input string
            input_str = kwargs.get('input', kwargs.get('query', ''))
            
            if hasattr(self.langchain_tool, 'arun'):
                # Async LangChain tool
                result = await self.langchain_tool.arun(input_str)
            elif hasattr(self.langchain_tool, 'run'):
                # Sync LangChain tool - run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.langchain_tool.run(input_str)
                )
            else:
                raise ValueError(f"LangChain tool {self.name} has no run method")
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"LangChainTool {self.name} execution failed: {e}")
            raise
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema from LangChain tool."""
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            # Convert Pydantic schema to function calling schema
            schema = self.langchain_tool.args_schema.schema()
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": schema
                }
            }
        else:
            # Fallback to simple string input
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Input for the tool"
                            }
                        },
                        "required": ["input"]
                    }
                }
            }


class ToolRegistry:
    """
    Registry for managing tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolBase] = {}
        
    def register(self, tool: ToolBase) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
    
    def get(self, name: str) -> Optional[ToolBase]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        
        return await tool.execute(**kwargs)
    
    async def initialize_all(self) -> None:
        """Initialize all registered tools."""
        for tool in self._tools.values():
            if not tool.is_initialized:
                await tool.initialize()
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered tools."""
        for tool in self._tools.values():
            if tool.is_initialized:
                await tool.shutdown()


def create_tool(tool_type: ToolType, config: ToolConfig, **kwargs) -> ToolBase:
    """
    Factory function to create tools.
    
    Args:
        tool_type: Type of tool to create
        config: Tool configuration
        **kwargs: Additional arguments
        
    Returns:
        Configured tool instance
    """
    if tool_type == ToolType.FUNCTION:
        func = kwargs.get('func')
        if not func:
            raise ValueError("func required for FUNCTION tool")
        return FunctionTool(func, config, **kwargs)
    elif tool_type == ToolType.AGENT:
        agent = kwargs.get('agent')
        if not agent:
            raise ValueError("agent required for AGENT tool")
        # Remove agent from kwargs to avoid duplicate parameter
        agent_kwargs = {k: v for k, v in kwargs.items() if k != 'agent'}
        return AgentTool(agent, config, **agent_kwargs)
    elif tool_type == ToolType.STEP:
        step = kwargs.get('step')
        if not step:
            raise ValueError("step required for STEP tool")
        return StepTool(step, config, **kwargs)
    elif tool_type == ToolType.LANGCHAIN:
        langchain_tool = kwargs.get('langchain_tool')
        if not langchain_tool:
            raise ValueError("langchain_tool required for LANGCHAIN tool")
        return LangChainTool(langchain_tool, config, **kwargs)
    else:
        raise ValueError(f"Unknown tool type: {tool_type}")


def function_tool(name: str, description: str = "", parameters: Optional[Dict[str, Any]] = None):
    """
    Decorator to create a function tool.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Parameter schema
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> ToolBase:
        config = ToolConfig(
            tool_type=ToolType.FUNCTION,
            name=name,
            description=description,
            parameters=parameters or {}
        )
        return FunctionTool(func, config)
    
    return decorator 