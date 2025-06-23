# NanoBrain Framework API Reference

This document provides comprehensive API documentation for all core classes in the NanoBrain framework. It is designed to be understandable by both humans and LLMs for automatic code generation.

## ⚡ Pydantic V2 Migration Complete

**Status:** ✅ **FULLY MIGRATED** - NanoBrain is now Pydantic V2 compliant

The NanoBrain framework has been successfully migrated to Pydantic V2 with zero breaking changes. All components now use modern Pydantic patterns:

### New Model Configuration Pattern

All NanoBrain models now use the **ConfigDict** pattern instead of class-based configuration:

```python
from pydantic import BaseModel, Field, ConfigDict

class MyModel(BaseModel):
    """Modern Pydantic V2 model example."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "example_component",
                "description": "Example description",
                "enable_feature": True
            }
        }
    )
    
    name: str = Field(..., description="Component name")
    description: str = Field(default="", description="Component description")
    enable_feature: bool = Field(default=True, description="Enable feature flag")
```

### Migration Benefits

- ✅ **Zero Breaking Changes** - Full backward compatibility maintained
- ✅ **Enhanced API Documentation** - Better OpenAPI schema generation
- ✅ **Future-Ready** - Prepared for Pydantic V3
- ✅ **Performance** - Improved validation performance
- ✅ **Modern Patterns** - Uses latest Pydantic best practices

### For Developers

When creating new components, always use the modern pattern:

```python
# ✅ Correct - Use ConfigDict
model_config = ConfigDict(json_schema_extra={"example": {...}})

# ❌ Deprecated - Don't use class Config
class Config:
    schema_extra = {"example": {...}}
```

## Import Structure

All NanoBrain core components are available through the `nanobrain` package:

```python
# Core components (currently available)
from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.executor import ParslExecutor, LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import get_logger

# Direct imports from nanobrain package
from nanobrain import ConversationalAgent, AgentConfig, DataUnitMemory, DataUnitConfig
from nanobrain import LocalExecutor, ParslExecutor, ExecutorConfig

# Library components (require explicit full paths due to disabled imports)
# Note: Library imports are currently disabled in nanobrain.__init__.py
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.library.workflows.chat_workflow.chat_workflow import ChatWorkflow

# Configuration
from nanobrain.config import get_config_manager
```

**Important Note**: The `nanobrain.library` module is currently disabled in the main package imports. To use library components, you must import them using their full module paths as shown above.

### Recommended Approach: YAML-Based Configuration

The NanoBrain framework follows a **YAML-first configuration philosophy**. Instead of manual configuration, use the component factory:

```python
from nanobrain.config.component_factory import create_component_from_yaml, create_workflow_from_yaml

# Load agent from YAML (recommended)
agent = create_component_from_yaml("docs/simple_agent_config.yml")

# Load complete workflow from YAML
workflow_components = create_workflow_from_yaml(
    "nanobrain/library/workflows/chat_workflow/chat_workflow.yml"
)
```

**Benefits:**
- Consistent configuration across all components
- Built-in validation and schema checking
- Reusable and shareable configurations
- Self-documenting YAML files

## Current Package Structure

The actual package structure is:

```
nanobrain/
├── core/                    # Core framework components (available)
│   ├── agent.py            # Base agent classes
│   ├── data_unit.py        # Data management
│   ├── executor.py         # Execution engines
│   ├── logging_system.py   # Logging and monitoring
│   └── ...
├── library/                # Library components (imports disabled)
│   ├── agents/             # Enhanced agent implementations
│   │   └── conversational/ # EnhancedCollaborativeAgent
│   ├── workflows/          # Workflow implementations
│   │   └── chat_workflow/  # ChatWorkflow class
│   └── infrastructure/     # Infrastructure components
├── config/                 # Configuration management
└── __init__.py            # Main package exports (library disabled)
```

## Table of Contents

1. [Data Units](#data-units)
2. [Triggers](#triggers)
3. [Links](#links)
4. [Steps](#steps)
5. [Agents](#agents)
6. [Executors](#executors)
7. [Tools](#tools)
8. [Logging System](#logging-system)
9. [Protocol Support](#protocol-support)
10. [Usage Patterns](#usage-patterns)

---

## Data Units

Data Units handle data storage and retrieval in the NanoBrain framework. They are the primary mechanism for data flow between components.

### DataUnitBase (Abstract Base Class)

**Purpose**: Base class for all data units that handle data storage and retrieval.

**Constructor**:
```python
DataUnitBase(config: Optional[DataUnitConfig] = None, **kwargs)
```

**Key Methods**:
- `async def get() -> Any`: Get data from the unit
- `async def set(data: Any) -> None`: Set data in the unit
- `async def clear() -> None`: Clear data from the unit
- `async def initialize() -> None`: Initialize the data unit
- `async def shutdown() -> None`: Shutdown the data unit
- `async def read() -> Any`: Alias for get() with logging
- `async def write(data: Any) -> None`: Alias for set() with logging
- `async def set_metadata(key: str, value: Any) -> None`: Set metadata
- `async def get_metadata(key: str, default: Any = None) -> Any`: Get metadata

**Properties**:
- `is_initialized: bool`: Check if data unit is initialized
- `metadata: Dict[str, Any]`: Get metadata dictionary

### DataUnitMemory

**Purpose**: In-memory data unit for fast access.

**Constructor**:
```python
DataUnitMemory(config: Optional[DataUnitConfig] = None, **kwargs)
```

**Usage Example**:
```python
# Create and initialize
config = DataUnitConfig(data_type="memory", name="my_data")
data_unit = DataUnitMemory(config)
await data_unit.initialize()

# Store and retrieve data
await data_unit.set({"message": "Hello World"})
data = await data_unit.get()  # Returns {"message": "Hello World"}

# Clean up
await data_unit.shutdown()
```

### DataUnitFile

**Purpose**: File-based data unit for persistent storage.

**Constructor**:
```python
DataUnitFile(file_path: str, config: Optional[DataUnitConfig] = None, **kwargs)
```

### DataUnitString

**Purpose**: String-based data unit with append capabilities.

**Constructor**:
```python
DataUnitString(initial_value: str = "", config: Optional[DataUnitConfig] = None, **kwargs)
```

**Additional Methods**:
- `async def append(data: str) -> None`: Append string data

### DataUnitStream

**Purpose**: Stream-based data unit with subscription capabilities.

**Constructor**:
```python
DataUnitStream(config: Optional[DataUnitConfig] = None, **kwargs)
```

**Additional Methods**:
- `async def subscribe() -> asyncio.Queue`: Subscribe to data updates
- `async def unsubscribe(queue: asyncio.Queue) -> None`: Unsubscribe from updates

**Important**: Only `DataUnitStream` has `subscribe()` method. Other data units do NOT have this method.

---

## Triggers

Triggers control when Steps execute based on various conditions.

### TriggerBase (Abstract Base Class)

**Purpose**: Base class for triggers that control when Steps execute.

**Constructor**:
```python
TriggerBase(config: Optional[TriggerConfig] = None, **kwargs)
```

**Key Methods**:
- `async def start_monitoring() -> None`: Start monitoring for trigger conditions
- `async def stop_monitoring() -> None`: Stop monitoring for trigger conditions
- `async def add_callback(callback: Callable) -> None`: Add callback to execute when triggered
- `async def remove_callback(callback: Callable) -> None`: Remove callback
- `async def trigger(data: Any = None) -> None`: Execute trigger with rate limiting

**Properties**:
- `is_active: bool`: Check if trigger is actively monitoring

### DataUpdatedTrigger

**Purpose**: Trigger that fires when data units are updated.

**Constructor**:
```python
DataUpdatedTrigger(data_units: List[Any], config: Optional[TriggerConfig] = None, **kwargs)
```

**Usage Example**:
```python
# Create trigger for data unit updates
trigger = DataUpdatedTrigger([data_unit1, data_unit2])
await trigger.add_callback(my_callback_function)
await trigger.start_monitoring()

# Later...
await trigger.stop_monitoring()
```

### AllDataReceivedTrigger

**Purpose**: Trigger that fires when all required data units have data.

**Constructor**:
```python
AllDataReceivedTrigger(data_units: List[Any], config: Optional[TriggerConfig] = None, **kwargs)
```

### TimerTrigger

**Purpose**: Trigger that fires at regular intervals.

**Constructor**:
```python
TimerTrigger(interval_ms: int, config: Optional[TriggerConfig] = None, **kwargs)
```

### ManualTrigger

**Purpose**: Trigger that fires manually when requested.

**Constructor**:
```python
ManualTrigger(config: Optional[TriggerConfig] = None, **kwargs)
```

**Additional Methods**:
- `async def fire(data: Any = None) -> None`: Manually fire the trigger

---

## Links

Links handle data flow between components.

### LinkBase (Abstract Base Class)

**Purpose**: Base class for links that handle data flow between components.

**Constructor**:
```python
LinkBase(source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs)
```

**Key Methods**:
- `async def start() -> None`: Start the link
- `async def stop() -> None`: Stop the link
- `async def transfer(data: Any) -> None`: Transfer data through the link

### DirectLink

**Purpose**: Direct data transfer between components.

**Constructor**:
```python
DirectLink(source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs)
```

**Usage Example**:
```python
# Create direct link between data units
link = DirectLink(source_data_unit, target_data_unit)
await link.start()

# Transfer data manually
await link.transfer(data)

# Stop the link when done
await link.stop()
```

---

## Steps

Steps process data using DataUnits and triggers.

### Step (Abstract Base Class)

**Purpose**: Base class for Steps that process data using DataUnits and triggers.

**Constructor**:
```python
Step(config: StepConfig, executor: Optional[ExecutorBase] = None, **kwargs)
```

**Key Methods**:
- `async def initialize() -> None`: Initialize the step and its components
- `async def shutdown() -> None`: Shutdown the step and cleanup resources
- `async def execute(**kwargs) -> Any`: Execute the step
- `async def process(input_data: Dict[str, Any], **kwargs) -> Any`: Abstract method to implement processing logic
- `async def set_input(data: Any, input_id: str = "input_0") -> None`: Set input data
- `async def get_output() -> Any`: Get output data
- `def register_input_data_unit(input_id: str, data_unit: DataUnitBase) -> None`: Register input data unit
- `def register_output_data_unit(data_unit: DataUnitBase) -> None`: Register output data unit
- `def add_link(link_id: str, link: LinkBase) -> None`: Add link to the step

**Properties**:
- `is_initialized: bool`: Check if step is initialized
- `execution_count: int`: Number of times step has been executed
- `error_count: int`: Number of execution errors
- `is_running: bool`: Check if step is currently running

**Usage Example**:
```python
class MyStep(Step):
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        # Implement your processing logic here
        result = input_data.get('message', '') + ' processed'
        return {'result': result}

# Create and use
config = StepConfig(name="my_step", description="Example step")
step = MyStep(config)
await step.initialize()

# Set input and execute
await step.set_input({'message': 'Hello'})
result = await step.execute()

await step.shutdown()
```

### SimpleStep

**Purpose**: Simple step implementation for basic processing.

**Constructor**:
```python
SimpleStep(config: StepConfig, executor: Optional[ExecutorBase] = None, **kwargs)
```

### TransformStep

**Purpose**: Step that applies a transformation function to input data.

**Constructor**:
```python
TransformStep(config: StepConfig, transform_func: callable = None, **kwargs)
```

---

## Agents

Agents use tool calling for AI processing with LLM integration.

### Agent (Abstract Base Class)

**Purpose**: Base class for Agents that use tool calling for AI processing.

**Constructor**:
```python
Agent(config: AgentConfig, executor: Optional[ExecutorBase] = None, **kwargs)
```

**Key Methods**:
- `async def initialize() -> None`: Initialize the agent and its components
- `async def shutdown() -> None`: Shutdown the agent and cleanup resources
- `async def process(input_text: str, **kwargs) -> str`: Abstract method to implement processing logic
- `async def execute(input_text: str, **kwargs) -> str`: Execute the agent with input text
- `def register_tool(tool: ToolBase) -> None`: Register a tool with the agent
- `def add_to_conversation(role: str, content: str) -> None`: Add message to conversation history
- `def clear_conversation() -> None`: Clear conversation history

**Properties**:
- `is_initialized: bool`: Check if agent is initialized
- `execution_count: int`: Number of times agent has been executed
- `error_count: int`: Number of execution errors
- `available_tools: List[str]`: List of available tool names

### ConversationalAgent

**Purpose**: Agent that maintains conversation context and uses tools.

**Constructor**:
```python
ConversationalAgent(config: AgentConfig, **kwargs)
```

**Usage Example**:
```python
# Create agent configuration
config = AgentConfig(
    name="MyAgent",
    description="Example conversational agent",
    model="gpt-3.5-turbo",
    temperature=0.7,
    system_prompt="You are a helpful assistant."
)

# Create and initialize agent
agent = ConversationalAgent(config)
await agent.initialize()

# Process input
response = await agent.process("Hello, how are you?")
print(response)

# Clean up
await agent.shutdown()
```

### SimpleAgent

**Purpose**: Simple agent implementation for basic AI processing.

**Constructor**:
```python
SimpleAgent(config: AgentConfig, **kwargs)
```

---

## Executors

Executors handle the execution environment for components.

### ExecutorBase (Abstract Base Class)

**Purpose**: Base class for executors that handle component execution.

**Constructor**:
```python
ExecutorBase(config: Optional[ExecutorConfig] = None)
```

**Key Methods**:
- `async def initialize() -> None`: Initialize the executor
- `async def shutdown() -> None`: Shutdown the executor
- `async def execute(func: callable, *args, **kwargs) -> Any`: Execute function

### LocalExecutor

**Purpose**: Local execution in the same process.

**Constructor**:
```python
LocalExecutor(config: Optional[ExecutorConfig] = None)
```

### ThreadExecutor

**Purpose**: Execution using thread pool.

**Constructor**:
```python
ThreadExecutor(config: Optional[ExecutorConfig] = None)
```

### ProcessExecutor

**Purpose**: Execution using process pool.

**Constructor**:
```python
ProcessExecutor(config: Optional[ExecutorConfig] = None)
```

---

## Tools

Tools provide specific functionality that can be used by agents.

### ToolBase (Abstract Base Class)

**Purpose**: Base class for tools that provide specific functionality.

**Constructor**:
```python
ToolBase(config: ToolConfig, **kwargs)
```

**Key Methods**:
- `async def initialize() -> None`: Initialize the tool
- `async def shutdown() -> None`: Shutdown the tool
- `async def execute(*args, **kwargs) -> Any`: Execute the tool

### FunctionTool

**Purpose**: Tool that wraps a Python function.

**Constructor**:
```python
FunctionTool(config: ToolConfig, func: callable, **kwargs)
```

### AgentTool

**Purpose**: Tool that wraps an Agent.

**Constructor**:
```python
AgentTool(config: ToolConfig, agent: Agent, **kwargs)
```

---

## Logging System

The logging system provides comprehensive logging and monitoring capabilities.

### NanoBrainLogger

**Purpose**: Enhanced logger with structured logging and performance tracking.

**Constructor**:
```python
NanoBrainLogger(name: str, log_file: Optional[str] = None, debug_mode: bool = False)
```

**Key Methods**:
- `def info(message: str, **kwargs)`: Log info message
- `def debug(message: str, **kwargs)`: Log debug message
- `def error(message: str, **kwargs)`: Log error message
- `def warning(message: str, **kwargs)`: Log warning message
- `async def async_execution_context(operation_type: OperationType, operation_name: str, **kwargs)`: Async context manager for operations

**Usage Example**:
```python
logger = get_logger("my_component")
logger.info("Component initialized", component_name="MyComponent")

async with logger.async_execution_context(OperationType.STEP_EXECUTE, "my_step") as context:
    # Your code here
    context.metadata['result'] = 'success'
```

---

## Protocol Support

### MCP Support (Model Context Protocol)

**MCPSupportMixin**: Mixin class that adds MCP capabilities to agents.

**Usage**:
```python
class MyAgent(MCPSupportMixin, ConversationalAgent):
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
```

**Key Methods**:
- `async def initialize_mcp(config_path: str = None) -> None`: Initialize MCP support
- `async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any`: Call MCP tool

### A2A Support (Agent-to-Agent Protocol)

**A2ASupportMixin**: Mixin class that adds A2A capabilities to agents.

**Usage**:
```python
class MyAgent(A2ASupportMixin, ConversationalAgent):
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
```

**Key Methods**:
- `async def initialize_a2a(config_path: str = None) -> None`: Initialize A2A support
- `async def call_a2a_agent(agent_name: str, message: str) -> str`: Call A2A agent

---

## Usage Patterns

### Basic Workflow Pattern

```python
# 1. Create data units
input_du = DataUnitMemory(DataUnitConfig(name="input"))
output_du = DataUnitMemory(DataUnitConfig(name="output"))
await input_du.initialize()
await output_du.initialize()

# 2. Create agent
agent_config = AgentConfig(name="MyAgent", model="gpt-3.5-turbo")
agent = ConversationalAgent(agent_config)
await agent.initialize()

# 3. Create step
step_config = StepConfig(name="MyStep")
step = MyCustomStep(step_config, agent)
await step.initialize()

# 4. Create triggers and links
trigger = DataUpdatedTrigger([input_du])
await trigger.add_callback(step.execute)
await trigger.start_monitoring()

link = DirectLink(step.output_data_unit, output_du)
await link.activate()

# 5. Use the workflow
await input_du.set({"message": "Hello"})
# Processing happens automatically via triggers

# 6. Clean up
await trigger.stop_monitoring()
await link.stop()
await step.shutdown()
await agent.shutdown()
await input_du.shutdown()
await output_du.shutdown()
```

### Enhanced Agent Pattern

```python
class EnhancedAgent(A2ASupportMixin, MCPSupportMixin, ConversationalAgent):
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Custom processing with A2A and MCP support
        if self.should_delegate(input_text):
            return await self.call_a2a_agent("specialist", input_text)
        elif self.should_use_tool(input_text):
            return await self.call_mcp_tool("calculator", {"expression": input_text})
        else:
            return await super().process(input_text, **kwargs)
```

### Error Handling Pattern

```python
try:
    await component.initialize()
    result = await component.execute(data)
except Exception as e:
    logger.error(f"Component execution failed: {e}")
    # Handle error appropriately
finally:
    await component.shutdown()
```

---

## Important Notes

1. **Always initialize components** before using them with `await component.initialize()`
2. **Always shutdown components** when done with `await component.shutdown()`
3. **Only DataUnitStream has subscribe() method** - other data units do not
4. **Use async/await** for all component operations
5. **Handle exceptions** appropriately in production code
6. **Use logging** for debugging and monitoring
7. **Follow the factory pattern** for component creation when available
8. **Use mixins** for protocol support (A2A, MCP)
9. **Configure components** using their respective Config classes
10. **Monitor performance** using the built-in logging and metrics systems

This API reference should be used as the authoritative guide for implementing NanoBrain components and workflows. 