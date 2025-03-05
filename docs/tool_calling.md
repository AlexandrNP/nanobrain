# Tool Calling in Agent

This document explains how to use the tool calling capability in the `Agent` class.

## Overview

The `Agent` class now supports tool calling, which allows it to use other `Step` objects as tools to perform specific tasks. This is similar to how humans use tools to extend their capabilities.

There are two approaches to implementing tool calling:

1. **LangChain Tool Binding**: Wrap each `Step` class as a LangChain Tool and register it with the LLM via the `bind_tools` function.
2. **Custom Tool Calling Prompt**: Use a custom-written tool calling prompt that ensures passing the correct arguments and using the tool in the appropriate context.

## Using LangChain Tool Binding

### Creating an Agent with Tools

```python
from Agent import Agent
from Step import Step
from ExecutorBase import ExecutorBase

# Create an executor
executor = ExecutorBase()

# Create tool steps
calculator_step = CalculatorStep(executor=executor)
text_processing_step = TextProcessingStep(executor=executor)

# Create agent with tools
agent = Agent(
    executor=executor,
    model_name="gpt-3.5-turbo",
    tools=[calculator_step, text_processing_step]
)
```

### Processing Input with Tools

```python
# Process input with tools
result = await agent.process_with_tools(["Calculate 5 + 3"])
```

### Adding and Removing Tools

```python
# Add a new tool
new_tool = NewToolStep(executor=executor)
agent.add_tool(new_tool)

# Remove a tool
agent.remove_tool(calculator_step)
```

## Using Custom Tool Calling Prompt

### Creating an Agent with Custom Tool Prompt

```python
# Create agent with tools and custom prompt
agent = Agent(
    executor=executor,
    model_name="gpt-3.5-turbo",
    tools=[calculator_step, text_processing_step],
    use_custom_tool_prompt=True
)
```

### Processing Input with Custom Tool Prompt

```python
# Process input with custom tool prompt
result = await agent.process_with_tools(["Calculate 5 + 3"])
```

### Executing a Tool Directly

```python
# Execute a tool directly by name
result = await agent.execute_tool("CalculatorStep", ["add", "10", "20"])
```

## Creating Tool Steps

To create a step that can be used as a tool, you need to:

1. Subclass `Step`
2. Implement the `process` method with a clear docstring explaining the input format
3. Provide a clear class docstring explaining the purpose of the tool

Example:

```python
class CalculatorStep(Step):
    """
    A step that performs basic arithmetic operations.
    """
    def __init__(self, executor, **kwargs):
        super().__init__(executor, **kwargs)
    
    async def process(self, inputs):
        """
        Process arithmetic operations.
        
        Input format: ["operation", num1, num2]
        Supported operations: add, subtract, multiply, divide
        """
        if len(inputs) < 3:
            return "Error: Not enough inputs. Format should be [operation, num1, num2]"
        
        operation = inputs[0]
        try:
            num1 = float(inputs[1])
            num2 = float(inputs[2])
        except ValueError:
            return "Error: Inputs must be numbers"
        
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return "Error: Cannot divide by zero"
            return num1 / num2
        else:
            return f"Error: Unsupported operation '{operation}'"
```

## Comparison of Approaches

### LangChain Tool Binding

**Advantages:**
- Uses LangChain's built-in tool calling capabilities
- Works with any LLM that supports tool calling
- Handles tool schema generation automatically

**Disadvantages:**
- Requires the LLM to support tool calling
- Less control over the tool calling format

### Custom Tool Calling Prompt

**Advantages:**
- Works with any LLM, even those that don't natively support tool calling
- More control over the tool calling format
- Can be customized for specific use cases

**Disadvantages:**
- Requires more custom code
- May be less reliable with some LLMs

## Biological Analogy

The tool calling capability in the `Agent` class is analogous to how humans use tools to extend their problem-solving capabilities. Just as humans learn to use tools by integrating motor skills with cognitive understanding, the `Agent` integrates `Step` objects as tools for its cognitive processing.

The `_register_tools` method is like tool use acquisition in primates, where the agent learns to use tools by integrating them into its cognitive framework. The `process_with_tools` method is like tool-assisted problem solving, where the agent uses tools to extend its processing capabilities.

## Testing

The tool calling capability is tested in:
- `test/test_agent_tools.py` for LangChain tool binding
- `test/test_agent_custom_tool_prompt.py` for custom tool calling prompt

Run the tests with:

```bash
python -m pytest test/test_agent_tools.py
python -m pytest test/test_agent_custom_tool_prompt.py
``` 