# NanoBrain

A biologically-inspired framework for building adaptive, resilient systems.

## Framework Organization

### 1. Core Foundation
- **Configuration Management**
  - `ConfigManager`: Core configuration handler (no configuration needed)
  - `DirectoryTracer`: Path management utility (no configuration needed)
  - `ConfigLoader`: Dynamic object creation with configuration loading

### 2. Component Hierarchy
- **Base Components**
  - `PackageBase`: Foundational unit with dependency management
  - `DataUnitBase`: Base memory storage with decay characteristics
  - `LinkBase`: Base connection with signal processing
  - `TriggerBase`: Base activation detector

- **Specialized Components**
  - Memory Units:
    - `DataUnitMemory`: RAM-based volatile storage
    - `DataUnitFile`: File-based persistent storage
  
  - Connection Types:
    - `LinkDirect`: Fast, reliable point-to-point connections
    - `LinkFile`: Persistent connections with storage
  
  - Execution Units:
    - `Runner`: Single-task execution manager
    - `Router`: Multi-target routing manager
    - `ExecutorFunc`: Function-specific executor
    - `ExecutorParallel`: Concurrent task executor

### 3. Component Interaction

#### Data Flow
```
[DataUnit] → [Link] → [Runner/Router] → [Executor] → [Output DataUnit]
```
- DataUnits provide data storage with configurable persistence
- Links manage data transfer with signal processing
- Runners/Routers control execution flow
- Executors perform actual operations

#### Control Flow
```
[Trigger] → [Runner] → [Router] → [Multiple Links] → [Multiple DataUnits]
```
- Triggers detect conditions for activation
- Runners manage single execution paths
- Routers distribute to multiple targets
- Links ensure reliable data transfer

### 4. Configuration System

#### Foundational Classes (No Configuration)
- Core utilities that are directly instantiatable
- Example: `ConfigManager`, `DirectoryTracer`

#### Configurable Components
Each component has:
1. Default Configuration
   - Required parameters (must be provided)
   - Optional parameters with reliable defaults
   ```yaml
   defaults:
     required_param: None
     optional_param: reliable_default_value
   ```

2. Metadata
   - Description and objective
   - Biological analogy and justification
   - Validation rules and constraints

3. Usage Examples
   - Basic usage with default configuration
   - Advanced usage patterns

## Key Features

### 1. Reliability-First Design
- Default configurations prioritize reliable execution
- Components fail gracefully with clear error states
- Built-in circuit breakers prevent cascading failures

### 2. Biological Analogies
- Components mirror biological systems for intuitive understanding
- Neural-inspired processing and adaptation mechanisms
- Homeostatic regulation for system stability

### 3. Adaptive Behavior
- Connection strengths adapt based on usage
- System modulators influence global behavior
- Resource management with automatic recovery

### 4. Resource Management
- Energy-aware execution with recovery mechanisms
- Automatic resource allocation and recovery
- Load balancing in parallel execution

## Usage Examples

### Basic Component Setup
```python
# Create a reliable direct link
link = ConfigLoader.create("LinkDirect",
                        input_data=input_unit,
                        output_data=output_unit)
await link.transfer()  # Guaranteed delivery with default config

# Create a parallel executor
executor = ConfigLoader.create("ExecutorParallel",
                           max_workers=4)
results = await executor.execute_batch(tasks)  # Reliable parallel execution
```

### Advanced Routing
```python
# Create an adaptive router
router = ConfigLoader.create("Router",
                         executor=my_executor,
                         input_source=source,
                         output_sinks=[sink1, sink2],
                         routing_strategy="adaptive")
await router.invoke()  # Intelligent routing with reliability
```

## Best Practices

1. **Configuration Management**
   - Use default configurations for maximum reliability
   - Override only when specific behavior is needed
   - Keep foundational classes unconfigured

2. **Component Design**
   - Follow biological analogies for intuitive design
   - Implement clear error states and recovery
   - Use circuit breakers for fault tolerance

3. **Resource Handling**
   - Monitor resource usage with system modulators
   - Implement recovery mechanisms
   - Use parallel execution for heavy workloads

4. **Testing and Monitoring**
   - Test with various load conditions
   - Monitor system stability and resource usage
   - Use built-in metrics for performance analysis

## Documentation

For detailed technical documentation:

- [UML Diagrams](docs/UML.md) - Comprehensive class diagrams showing framework architecture and relationships
- [Framework Overview](docs/framework_overview.md) - High-level overview of the framework architecture and principles
- [Auto-generated Documentation](docs/auto_generated/index.md) - Detailed documentation for each class, generated from source code and configuration files
- [Tool Calling Documentation](docs/tool_calling.md) - Guide to using the tool calling capabilities

### Documentation Builder

The framework includes a documentation builder that automatically generates documentation from:
1. Python docstrings in source files
2. YAML configuration files
3. Class and function signatures

To build the documentation, run:

```bash
./scripts/build_docs.sh
```

The generated documentation will be available in the `docs/auto_generated` directory.

## LLM Integration

The framework includes robust integration with various Language Model providers:

- **OpenAI**: GPT models (gpt-3.5-turbo, gpt-4, etc.)
- **Anthropic**: Claude models (claude-2, etc.)
- **Google**: Gemini models
- **Meta/Llama**: Llama models
- **Mistral**: Mistral models

The Agent class can work with both chat-based models (BaseChatModel) and completion-based models (BaseLLM), automatically detecting the appropriate model type and using the correct API for each.

## Configuration System

The NanoBrain framework includes a global configuration system that manages API keys, model defaults, and other framework-wide settings.

### Setting Up API Keys

To use external language models, you'll need to set up API keys for the providers you want to use. The framework provides a convenient script to help you set up your API keys:

```bash
./setup_api_keys.py
```

This script will guide you through the process of obtaining and configuring API keys for various LLM providers.

### Configuration File

The global configuration is stored in a YAML file (`config.yml`) in the project root directory. You can also manage the configuration using the command-line interface:

```bash
# List all configuration settings
./nanobrain config list

# Edit a specific configuration setting
./nanobrain config edit --key api_keys.openai --value your_api_key

# Open the configuration file in your default editor
./nanobrain config edit

# Reset the configuration to defaults
./nanobrain config reset
```

### Environment Variables

You can also set configuration values using environment variables. The naming convention is `NANOBRAIN_SECTION_KEY`, for example:

```bash
# Set the OpenAI API key
export NANOBRAIN_API_KEYS_OPENAI=your_api_key

# Set the default model
export NANOBRAIN_MODELS_DEFAULT=gpt-4

# Enable debug mode
export NANOBRAIN_DEVELOPMENT_DEBUG=true
```

Environment variables take precedence over values in the configuration file.

### Testing Mode

When running in testing mode (with the `NANOBRAIN_TESTING=1` environment variable), the framework will use mock implementations of language models and other external dependencies. This allows you to run tests without needing actual API keys.

```bash
# Run in testing mode
NANOBRAIN_TESTING=1 ./nanobrain builder
```