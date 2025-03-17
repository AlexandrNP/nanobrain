# AgentWorkflowBuilder

Builder for creating Agent-based workflows.

Biological analogy: Neural circuit formation during development.
Justification: Like how neural circuits are formed through a guided process
during development, this builder helps construct agent-based workflows with
appropriate connections and configurations.

## Class Structure

The `AgentWorkflowBuilder` class inherits from `Agent` and uses aggregation for data units:

```python
class AgentWorkflowBuilder(Agent):
    def __init__(self, executor=None, data=None, persistence_level=0, _debug_mode=False, use_code_writer=True, **kwargs):
        # Implementation...
```

## Key Components

### Input and Output Data Units

- `input`: A `DataUnitString` instance that stores input queries
- `output`: A `DataUnitString` instance that stores processing results

### InputWrapper

A wrapper class that adapts the input data unit to be compatible with the `LinkDirect` mechanism:

```python
class InputWrapper:
    def __init__(self, data_unit):
        self.output = data_unit
```

### Current Agent

The class holds a fixed instance of `current_agent` that is responsible for writing code based on queries from the input data unit.

## Methods

### __init__

```python
def __init__(self, executor: Optional['ExecutorBase'] = None, 
            data: Any = None, 
            persistence_level: int = 0,
            _debug_mode: bool = False,
            use_code_writer: bool = True,
            **kwargs)
```

Initialize the agent workflow builder.

Biological analogy: Neural stem cell differentiation.
Justification: Like a neural stem cell that initializes with the potential to develop into 
various neural components, the AgentWorkflowBuilder initializes with the capability 
to create and connect different components of an agent-based workflow.

### get

```python
def get(self) -> Any
```

Get the current data from the input data unit.

Biological analogy: Memory retrieval.
Justification: Like how the brain retrieves stored memories,
this method retrieves the stored data from the input data unit.

### set

```python
def set(self, data: Any) -> bool
```

Set the data in the input data unit and process it.

Biological analogy: Memory encoding.
Justification: Like how the brain encodes new memories,
this method stores new data in the input data unit and
triggers processing of that data.

### process

```python
async def process(self, inputs: List[Any], use_code_writer: bool = True) -> Any
```

Process inputs using the Agent's language model.

This method processes the input data and updates the output data unit.

### create_agent

```python
async def create_agent(self, model: str = "gpt-4", use_code_writer: bool = True, **kwargs) -> Agent
```

Create a new agent with the specified parameters.

### create_workflow

```python
async def create_workflow(self, name: str, **kwargs) -> Workflow
```

Create a new workflow with the specified parameters.

### get_generated_code

```python
def get_generated_code(self) -> str
```

Get the generated step implementation code.

### get_generated_config

```python
def get_generated_config(self) -> str
```

Get the generated configuration YAML.

### get_generated_tests

```python
def get_generated_tests(self) -> str
```

Get the generated test code.

## Data Flow

The data flow in the `AgentWorkflowBuilder` is as follows:

1. Input data is set in the `input` data unit
2. The `current_agent` processes the input data
3. Results are stored in the `output` data unit

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

## Related Documentation

For more detailed information about the recent changes to the `AgentWorkflowBuilder` class, see the [AgentWorkflowBuilder.md](../AgentWorkflowBuilder.md) documentation. 