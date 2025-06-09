# YAML Tool Loading Implementation

## Overview

This document describes the implementation of YAML-based tool loading for agents in the NanoBrain framework, replacing the previous programmatic tool registration approach.

## Changes Made

### 1. Extended AgentConfig

**File**: `src/core/agent.py`

Added `tools_config_path` field to `AgentConfig`:

```python
class AgentConfig(BaseModel):
    # ... existing fields ...
    tools_config_path: Optional[str] = Field(default=None, description="Path to YAML file containing tool configurations")
```

### 2. YAML Tool Loading Implementation

**File**: `src/core/agent.py`

Added methods for YAML-based tool loading:

- `_load_tools_from_yaml_config()`: Loads tools from YAML configuration file
- `_resolve_config_path()`: Resolves configuration file paths with multiple search locations
- `_standardize_tool_config()`: Standardizes tool configuration format
- `_register_agent_tool_from_config()`: Creates agent instances from YAML configuration
- `_import_class_from_path()`: Dynamically imports agent classes

### 3. Removed Programmatic Registration

**Files**: `src/core/agent.py`, `src/agents/code_writer.py`

Completely removed programmatic tool registration methods:
- `register_agent_tool()` method from Agent class
- `register_function_tool()` method from Agent class  
- `register_file_writer_tool()` method from CodeWriterAgent class

### 4. YAML Configuration Files

**File**: `src/config/tools.yml`

Created example tools configuration:

```yaml
tools:
  - name: "file_writer"
    tool_type: "agent"
    class: "agents.file_writer.FileWriterAgent"
    description: "Tool for writing files based on natural language descriptions"
    parameters:
      type: "object"
      properties:
        input:
          type: "string"
          description: "Input text for the file writer agent"
      required: ["input"]
    config:
      name: "FileWriterAgent"
      description: "Handles file operations with detailed logging"
      model: "gpt-4"
      temperature: 0.3
```

### 5. Updated Agent Configurations

**File**: `src/agents/config/step_coder.yml`

Added `tools_config_path` to agent configurations:

```yaml
config:
  # ... existing config ...
  tools_config_path: "tools.yml"
```

### 6. Demo and Test Files

Created demonstration and test files:
- `demo/yaml_tool_loading_demo.py`: Comprehensive demo showing YAML-only approach
- `demo/simple_yaml_tool_demo.py`: Simple demo showing basic YAML tool loading
- `tests/test_yaml_tool_loading.py`: Comprehensive test suite for YAML tool loading

## Usage

### YAML-Based Tool Loading (Only Option)

```python
from core.agent import AgentConfig
from agents.code_writer import CodeWriterAgent

# Create agent with YAML tool configuration
config = AgentConfig(
    name="MyAgent",
    description="Agent with YAML tools",
    tools_config_path="tools.yml"  # Load tools from YAML
)

agent = CodeWriterAgent(config)
await agent.initialize()  # Tools are loaded automatically
```

### Programmatic Tool Loading (Removed)

```python
# This approach has been completely removed
# The following methods no longer exist:
# - agent.register_agent_tool()
# - agent.register_function_tool()
# - code_writer.register_file_writer_tool()
```

## Configuration Path Resolution

The system searches for YAML configuration files in the following order:

1. Absolute path (if provided)
2. Relative to current working directory
3. Relative to `src` directory
4. In `config` directory
5. In `src/config` directory
6. In `src/agents/config` directory
7. In `agents/config` directory

## YAML Tool Configuration Format

```yaml
tools:
  - name: "tool_name"                    # Required: Tool name
    tool_type: "agent"                   # Required: Tool type (agent, function, step)
    class: "module.path.ClassName"       # Required: Python class path
    description: "Tool description"      # Required: Tool description
    parameters:                          # Required: Parameter schema for LLM
      type: "object"
      properties:
        input:
          type: "string"
          description: "Input description"
      required: ["input"]
    config:                              # Optional: Agent-specific configuration
      name: "AgentName"
      model: "gpt-4"
      temperature: 0.3
      # ... other agent config fields
```

## Benefits

1. **Declarative Configuration**: Tools are defined in YAML, not code
2. **Clean Agent Interface**: No programmatic tool registration methods cluttering the API
3. **Easy Modification**: Tool configurations can be changed without code changes
4. **Better Separation of Concerns**: Configuration separated from implementation
5. **Dynamic Tool Loading**: Tools are loaded at runtime from configuration
6. **Reusable Configurations**: Tool configurations can be shared across agents
7. **Simplified Codebase**: Eliminates complexity from programmatic tool management

## Migration Guide

### From Programmatic to YAML

1. **Create tools.yml file** with tool configurations
2. **Add tools_config_path** to AgentConfig
3. **Remove all programmatic registration** calls (methods no longer exist)
4. **Test the configuration** to ensure tools load correctly

### Example Migration

**Before (Programmatic - No Longer Supported)**:
```python
# This code will no longer work - methods have been removed
code_writer = CodeWriterAgent(config)
file_writer = FileWriterAgent(file_config)
code_writer.register_file_writer_tool(file_writer)  # Method doesn't exist
```

**After (YAML - Only Option)**:
```python
config = AgentConfig(
    name="CodeWriter",
    tools_config_path="tools.yml"
)
code_writer = CodeWriterAgent(config)
# Tools loaded automatically during initialization
```

## Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_yaml_tool_loading.py -v
```

Run demos to see it in action:

```bash
python demo/simple_yaml_tool_demo.py
python demo/yaml_tool_loading_demo.py
```

## Breaking Changes

- **Programmatic tool registration methods have been completely removed**
- **Existing code using these methods will break and must be migrated**
- **YAML configuration is now the only way to configure agent tools**
- **No backward compatibility for programmatic registration**

### Removed Methods:
- `Agent.register_agent_tool()`
- `Agent.register_function_tool()`
- `CodeWriterAgent.register_file_writer_tool()`

## Future Considerations

- Tool configurations could be extended to support more complex scenarios
- Registry-based tool discovery could be added
- Tool versioning and dependency management could be implemented
- Hot-reloading of tool configurations could be supported 