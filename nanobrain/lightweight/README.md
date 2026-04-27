# Nanobrain Lightweight Wrapper

A lightweight, dependency-free wrapper for building nanobrain workflows without requiring the full framework installation.

## Overview

The lightweight wrapper provides:
- **Comprehensive Config Discovery**: Scans all framework config files to find available components
- **Intelligent Priority System**: Automatically selects best configs (Core > Components > Base > Library)
- **Enhanced Workflow Builder**: Build workflows with real components and validation
- **YAML Output**: Generates framework-compatible YAML configurations
- **Zero Dependencies**: Works with Python standard library only

## Quick Start

```python
from nanobrain.lightweight.enhanced_workflow_builder import EnhancedWorkflowBuilder

# Create workflow builder
builder = EnhancedWorkflowBuilder("my_workflow", "Example workflow")

# Add components
builder.add_input("user_query", "DataUnitString")
builder.add_component("agent", "EnhancedCollaborativeAgent", 
                     model="gpt-4", temperature=0.7)
builder.add_output("response", "DataUnitString")

# Connect components
builder.connect("user_query", "agent")
builder.connect("agent", "response")

# Generate and save workflow
builder.save_workflow("my_workflow.yml")
```

## Architecture

### Discovery System
- **ComprehensiveConfigDiscovery**: Scans ALL config files in the framework
- **Real Class References**: Uses explicit `class:` fields from config files
- **Priority-Based Selection**: Intelligent defaults with user override options

### Workflow Builder
- **Component Addition**: Add real framework components with validation
- **Parameter Validation**: Basic type checking against schemas
- **Connection Management**: Link components with DirectLink or custom links
- **YAML Generation**: Framework-compatible output format

## Available Components

The discovery system finds 54+ real framework components including:

### Agents
- `EnhancedCollaborativeAgent`: Advanced conversational agent
- `ViralExpertConversationalAgent`: Domain-specific viral expert
- `QueryAnalysisAgent`: Query understanding and analysis
- `ProteinSynonymAgent`: Protein name resolution

### Steps
- `ConversationalResponseStep`: Generate conversational responses
- `QueryClassificationStep`: Classify user queries
- `AlignmentStep`: Sequence alignment processing
- `ClusteringStep`: Data clustering operations

### Tools
- `BVBRCTool`: Bacterial and viral bioinformatics resource
- `MMseqs2Tool`: Sequence similarity search
- `MUSCLETool`: Multiple sequence alignment

### Data Units
- `DataUnit`: Base data container
- `DataUnitMemory`: In-memory data storage
- `DataUnitString`: String data container
- `DataUnitFile`: File-based data storage

### Executors
- `LocalExecutor`: Local execution environment

## Configuration Priority

When multiple configs exist for a class, priority is:
1. 🥇 **CORE**: `config/core/` - Framework core configs
2. 🥈 **COMPONENTS**: `config/components/` - Standard components
3. 🥉 **BASE**: `config/` - Base configurations
4. 🏅 **LIBRARY**: `library/config/` - Library-specific configs

## Examples

### Basic Chat Workflow
```python
builder = EnhancedWorkflowBuilder("chat_workflow", "Simple chat workflow")

# Add components with intelligent defaults
builder.add_input("user_message", "DataUnitString")
builder.add_component("chat_agent", "EnhancedCollaborativeAgent")
builder.add_output("agent_response", "DataUnitString")

# Connect and save
builder.connect("user_message", "chat_agent")
builder.connect("chat_agent", "agent_response")
builder.save_workflow("chat_workflow.yml")
```

### Advanced Workflow with Custom Config
```python
builder = EnhancedWorkflowBuilder("advanced_workflow", "Advanced processing")

# Use specific config file
builder.add_component("specialized_agent", "SpecializedAgent", 
                     config_choice="viral_conversation_agent.yml",
                     temperature=0.3, max_tokens=1000)

# Add executor with custom parameters
builder.add_component("executor", "LocalExecutor", 
                     max_workers=4, timeout=300)
```

### Workflow Validation
```python
# Build workflow
builder = EnhancedWorkflowBuilder("validation_test", "Test validation")
# ... add components ...

# Validate before saving
validation = builder.validate_workflow()
if validation["valid"]:
    builder.save_workflow("validated_workflow.yml")
else:
    print("Errors:", validation["errors"])
    print("Warnings:", validation["warnings"])
```

## API Reference

### EnhancedWorkflowBuilder

#### Constructor
```python
EnhancedWorkflowBuilder(workflow_name: str, description: str = "")
```

#### Methods
- `add_component(name, class_name, config_choice=None, **params)`: Add component
- `add_input(name, data_unit_class="DataUnitMemory", **params)`: Add input
- `add_output(name, data_unit_class="DataUnitMemory", **params)`: Add output
- `connect(source, target, link_class="DirectLink", **params)`: Connect components
- `save_workflow(file_path, format="yaml")`: Save workflow
- `validate_workflow()`: Validate workflow
- `list_available_components(category=None)`: List available components

### ComprehensiveConfigDiscovery

#### Methods
- `discover_all_configs()`: Run complete discovery
- `get_default_config_for_class(class_name)`: Get highest priority config
- `get_all_configs_for_class(class_name)`: Get all configs for class
- `list_available_classes()`: List all discovered classes

## Requirements

- Python 3.7+
- PyYAML (for YAML output)
- Access to nanobrain framework config files

## Limitations

- **Schema Validation**: Basic type checking only (not full Pydantic validation)
- **Import Dependencies**: Some components require optional dependencies
- **Execution**: This wrapper only generates configs, doesn't execute workflows
- **Real-time Updates**: Discovery results are not automatically updated

## Troubleshooting

### Common Issues

**"Class not available" Error**
- Check if the class name is correct
- Verify the component has config files
- Use `list_available_components()` to see available classes

**"Config not found" Error**
- Check config file path is correct
- Use `get_all_configs_for_class()` to see available configs

**Validation Warnings**
- Check parameter types match expected schema
- Verify required parameters are provided

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Now discovery will show detailed debug information
```

## Contributing

The lightweight wrapper is designed to be:
- **Self-contained**: Minimal external dependencies
- **Framework-agnostic**: Works without full nanobrain installation
- **Extensible**: Easy to add new discovery methods or validation rules

## License

Same as nanobrain framework.
