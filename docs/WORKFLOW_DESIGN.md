# NanoBrain Base Workflow System

## Overview

The NanoBrain base Workflow system provides a foundational architecture for creating graph-based workflow orchestration that extends the existing Step system. This document describes the design, implementation, and usage of the core workflow components.

## Architecture

### Core Components

1. **Workflow** - The main orchestrator class that extends Step
2. **WorkflowConfig** - Configuration class extending StepConfig
3. **WorkflowGraph** - Internal graph representation and analysis
4. **ConfigLoader** - Recursive configuration loading system
5. **ExecutionStrategy** - Enumeration of execution strategies
6. **ErrorHandlingStrategy** - Enumeration of error handling approaches

### Class Hierarchy

```
Step
 └── Workflow
     ├── WorkflowGraph
     ├── ConfigLoader
     └── child_steps: Dict[str, Step]
```

## Design Principles

### 1. Extends Step Architecture

The Workflow class extends the base Step class, making workflows first-class citizens that can be composed hierarchically.

### 2. Graph-Based Structure

Workflows are represented as directed graphs where:
- **Nodes** = Steps (including nested workflows)
- **Edges** = Links (data flow connections)
- **Execution Order** = Topological sort of the graph

### 3. Configuration-Driven

Workflows are defined through YAML configuration files that specify:
- Step definitions (inline or external files)
- Link connections between steps
- Execution parameters and strategies
- Error handling policies

## Implementation Status

✅ **COMPLETED COMPONENTS:**

1. **Base Workflow Class** (`nanobrain/core/workflow.py`)
   - Full implementation with 1113 lines of code
   - Extends Step class with workflow-specific functionality
   - Graph-based execution with topological sorting
   - Comprehensive error handling and monitoring

2. **WorkflowConfig Class**
   - Extends StepConfig with workflow-specific parameters
   - Support for steps, links, execution strategies
   - Validation and error handling configuration

3. **WorkflowGraph Class**
   - Internal graph representation with nodes and edges
   - Cycle detection using DFS algorithm
   - Topological sorting for execution order
   - Graph validation and statistics

4. **ConfigLoader Class**
   - Recursive configuration loading from YAML files
   - Intelligent path resolution across directories
   - Configuration caching to prevent duplicate loading
   - Comprehensive error handling for missing files

5. **Core Integration**
   - Updated `nanobrain/core/__init__.py` to export workflow components
   - Updated `nanobrain/core/step.py` factory to create workflows
   - Updated `nanobrain/core/config/component_factory.py` for workflow support
   - Added WORKFLOW to ComponentType enum

6. **Configuration Support**
   - Example workflow configuration (`config/example_workflow.yaml`)
   - Support for both inline and external step configurations
   - Link definition between workflow steps

7. **Testing Framework**
   - Comprehensive test suite (`tests/test_workflow.py`)
   - Unit tests for all major components
   - Integration tests with existing framework
   - Fixtures for common test scenarios

## Usage Example

```python
from nanobrain.core.workflow import create_workflow

# Create workflow from YAML file
workflow = await create_workflow("config/example_workflow.yaml")

# Execute workflow
result = await workflow.process({"input_data": "example"})

# Get execution statistics
stats = workflow.get_workflow_stats()
print(f"Completed: {stats['completed_steps']} steps")

# Cleanup
await workflow.shutdown()
```

## Configuration Format

```yaml
name: "example_workflow"
description: "A simple example workflow"
execution_strategy: "sequential"
error_handling: "continue"

steps:
  - step_id: "input_step"
    class: "SimpleStep"
    config:
      name: "input_step"
      description: "Input processing step"
  
  - step_id: "processing_step"
    config_file: "processing_step.yaml"  # External config

links:
  - link_id: "input_to_processing"
    source: "input_step"
    target: "processing_step"
    link_type: "direct"
```

## Key Features Implemented

- ✅ Graph-based workflow orchestration
- ✅ Topological execution ordering
- ✅ Cycle detection and validation
- ✅ Recursive configuration loading
- ✅ Dynamic step management (add/remove)
- ✅ Comprehensive error handling
- ✅ Performance monitoring and statistics
- ✅ Integration with existing Step/Link/Executor framework
- ✅ Factory pattern support
- ✅ YAML configuration support
- ✅ Extensive test coverage

## Next Steps for Web Interface Implementation

With the base Workflow class now implemented, we can proceed to create the backend web interface class in `nanobrain/library/interfaces/web` that was originally requested. The workflow system will provide the foundation for orchestrating chat workflows through the REST API.

## Directory Structure

The implementation follows NanoBrain's architectural patterns:

```
nanobrain/core/
├── workflow.py                 # ✅ Base Workflow implementation
├── __init__.py                 # ✅ Updated exports
├── step.py                     # ✅ Updated factory
└── config/
    └── component_factory.py    # ✅ Updated factory

config/
└── example_workflow.yaml      # ✅ Example configuration

tests/
└── test_workflow.py           # ✅ Comprehensive tests

nanobrain/docs/
└── WORKFLOW_DESIGN.md         # ✅ This documentation
```

The base Workflow class is now ready for use throughout the NanoBrain framework, providing a solid foundation for both simple linear workflows and complex multi-step processes while maintaining full compatibility with existing framework components. 