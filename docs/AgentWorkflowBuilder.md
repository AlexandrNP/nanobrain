# AgentWorkflowBuilder

## Overview

The `AgentWorkflowBuilder` class is a specialized builder for creating Agent-based workflows. It extends the basic workflow building capabilities with agent-specific features.

**Biological analogy**: Neural circuit formation during development.

**Justification**: Like how neural circuits are formed through a guided process during development, this builder helps construct agent-based workflows with appropriate connections and configurations.

## Recent Changes

The `AgentWorkflowBuilder` class has been refactored to change its relationship with `DataUnitBase` from inheritance to aggregation. This change improves the design by:

1. Following the "composition over inheritance" principle
2. Creating a clearer separation of concerns
3. Enabling more flexible data flow between components

### Before: Inheritance-based Design

Previously, `AgentWorkflowBuilder` inherited from both `Agent` and `DataUnitBase`:

```python
class AgentWorkflowBuilder(Agent, DataUnitBase):
    # Implementation...
```

This design had several limitations:
- It created a complex inheritance hierarchy
- It mixed the responsibilities of data storage and workflow building
- It made it difficult to have separate input and output data units

### After: Aggregation-based Design

Now, `AgentWorkflowBuilder` inherits only from `Agent` and uses aggregation for data units:

```python
class AgentWorkflowBuilder(Agent):
    def __init__(self, executor=None, data=None, persistence_level=0, _debug_mode=False, use_code_writer=True, **kwargs):
        super().__init__(executor=executor)
        
        # Create input and output data units
        self.input = DataUnitString(name="AgentWorkflowBuilderInput", initial_value=data, persistence_level=persistence_level)
        self.output = DataUnitString(name="AgentWorkflowBuilderOutput", persistence_level=persistence_level)
        
        # Create a wrapper for the input data unit to make it compatible with LinkDirect
        self.input_wrapper = InputWrapper(self.input)
        
        # Other initialization...
```

## Key Components

### InputWrapper

A new `InputWrapper` class was created to adapt the input data unit to be compatible with the `LinkDirect` mechanism:

```python
class InputWrapper:
    """
    Wrapper class for input data unit to make it compatible with LinkDirect.
    
    Biological analogy: Dendritic spine that receives signals.
    Justification: Like how dendritic spines form specialized structures to receive
    signals from other neurons, this wrapper adapts the input data unit to be
    compatible with the link mechanism.
    """
    def __init__(self, data_unit):
        self.output = data_unit
```

### Input and Output Data Units

The class now has dedicated input and output data units:

- `input`: A `DataUnitString` instance that stores input queries
- `output`: A `DataUnitString` instance that stores processing results

### Current Agent

The class now holds a fixed instance of `current_agent` that is responsible for writing code based on queries from the input data unit:

```python
# Initialize current_agent
self.current_agent = None

# Schedule the agent initialization
self._schedule_task(self._init_agent(
    model=kwargs.get('model_name', 'gpt-4'),
    use_code_writer=use_code_writer
))
```

## Data Flow

The data flow in the updated design is as follows:

1. Input data is set in the `input` data unit
2. The `current_agent` processes the input data
3. Results are stored in the `output` data unit

This is implemented in the `process` method:

```python
async def process(self, inputs: List[Any], use_code_writer: bool = True) -> Any:
    # ... (input handling)
    
    # Set the input data unit
    self.input.set(input_text)
    
    # ... (processing logic)
    
    # Process with Agent's language model
    response = await self.current_agent.process([input_text])
    
    # Update the output data unit
    self.output.set(response)
    
    # ... (additional processing)
    
    return response
```

## Connection with Agent

The connection between the `input` data unit and the `current_agent` is established in the `_init_agent` method:

```python
async def _init_agent(self, model: str = "gpt-4", use_code_writer: bool = True, **kwargs) -> None:
    # ... (agent initialization)
    
    # Create a direct link between input and current_agent
    if self.current_agent is not None:
        # Register the input as an input source for the agent
        self.current_agent.register_input_source("builder_input", self.input)
        
        if self._debug_mode:
            print("Established direct connection between input and agent")
```

## UML Diagram

The updated UML diagram for the `AgentWorkflowBuilder` class can be found in the [UML.md](UML.md) file, in the "Builder Components" section.

## Usage Example

```python
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.ExecutorFunc import ExecutorFunc

# Create an ExecutorFunc instance
executor = ExecutorFunc()

# Create an AgentWorkflowBuilder with the code writer agent
builder = AgentWorkflowBuilder(
    executor=executor,
    _debug_mode=True,
    use_code_writer=True
)

# Set input data
test_input = "Create a simple DataUnit class for storing text data with decay functionality"
builder.input.set(test_input)

# Process the input
response = await builder.process([test_input], use_code_writer=True)

# Get the output
output = builder.output.get()
print(output)

# Get the generated code
print(builder.get_generated_code())
``` 