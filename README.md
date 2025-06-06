# NanoBrain Framework v2.0

A modern, async-first framework for orchestrating AI agents, data processing steps, and HPC workflows with unified YAML configuration.

## 🧠 Overview

NanoBrain v2.0 is a complete refactor of the original framework, designed with clean separation of concerns and industrial best practices. The framework provides:

- **Decoupled Architecture**: Clear separation between Agents (tool-calling AI) and Steps (data processing)
- **Agent-to-Agent Interaction**: Agents can use other agents as tools
- **Event-Driven Processing**: Steps use triggers and data units for reactive processing
- **Configurable Execution**: Support for local, threaded, and HPC (Parsl) execution
- **YAML Configuration**: Complete workflow configuration with schema validation
- **Async-First Design**: Built for modern async/await patterns

## 🏗️ Architecture

### Core Components

```
nanobrain/
├── src/
│   ├── core/                    # Core framework components
│   │   ├── agent.py            # Agent system (tool-calling AI)
│   │   ├── step.py             # Step system (data processing)
│   │   ├── executor.py         # Execution backends
│   │   ├── data_unit.py        # Data interfaces
│   │   ├── trigger.py          # Event triggers
│   │   ├── link.py             # Dataflow abstractions
│   │   └── tool.py             # Tool system
│   ├── agents/                  # Specialized agents
│   │   ├── code_writer.py      # Code generation agent
│   │   └── file_writer.py      # File operations agent
│   └── config/                  # Configuration system
│       ├── yaml_config.py      # YAML configuration
│       └── schema_generator.py # Schema generation
├── demo/                        # Demo scripts
│   └── code_writer_advanced.py # Advanced demo
└── requirements.txt             # Dependencies
```

### Key Concepts

#### Agents vs Steps

- **Agents**: AI-powered components that use LLMs and can call tools (including other agents)
- **Steps**: Data processing components that use triggers and data units for reactive processing

#### Data Flow

- **Agents**: Direct tool calling between agents
- **Steps**: Data flows through Links between Steps, triggered by events

#### Configuration

- **YAML-based**: Complete workflow configuration in YAML
- **Schema validation**: JSON schemas for configuration validation
- **Modular**: Separate configuration for agents, steps, executors, etc.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd nanobrain

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from nanobrain.src.agents.code_writer import CodeWriterAgent
from nanobrain.src.agents.file_writer import FileWriterAgent

async def main():
    # Create agents
    file_writer = FileWriterAgent()
    code_writer = CodeWriterAgent()
    
    # Initialize agents
    await file_writer.initialize()
    await code_writer.initialize()
    
    # Register file writer as tool for code writer
    code_writer.register_file_writer_tool(file_writer)
    
    # Generate and save code
    response = await code_writer.process(
        "Create a Python function to calculate fibonacci numbers and save it to fibonacci.py"
    )
    
    print(response)
    
    # Cleanup
    await code_writer.shutdown()
    await file_writer.shutdown()

# Run the example
asyncio.run(main())
```

### Running the Demo

```bash
cd nanobrain
python demo/code_writer_advanced.py
```

The demo showcases:
1. Agent-to-agent interaction
2. Step-based data processing
3. YAML configuration system
4. Mixed agent-step workflows

## 📋 Features

### Agent System

- **Tool Calling**: Agents can use other agents as tools
- **LLM Integration**: Support for OpenAI, Anthropic, and other LLM providers
- **Async Processing**: Full async/await support
- **Specialized Agents**: Pre-built agents for common tasks

### Step System

- **Data Units**: Typed data interfaces (memory, file, stream, database)
- **Triggers**: Event-driven execution (data updates, timers, conditions)
- **Links**: Dataflow abstractions between steps
- **Configurable Executors**: Local, threaded, or HPC execution

### Configuration System

- **YAML Configuration**: Human-readable workflow definitions
- **Schema Generation**: Automatic JSON schema generation
- **Validation**: Configuration validation with detailed error messages
- **Modular**: Separate configuration for different components

### Execution Backends

- **Local Executor**: Single-threaded execution
- **Thread Executor**: Multi-threaded execution
- **Process Executor**: Multi-process execution
- **Parsl Executor**: HPC and distributed execution

## 🔧 Configuration

### YAML Configuration Example

```yaml
name: "example_workflow"
description: "Example NanoBrain workflow"
version: "1.0.0"

# Define executors
executors:
  local:
    executor_type: "local"
    max_workers: 4
  hpc:
    executor_type: "parsl"
    max_workers: 16
    parsl_config:
      provider: "slurm"
      nodes_per_block: 1

# Define agents
agents:
  code_writer:
    name: "code_writer"
    description: "Code generation agent"
    model: "gpt-4"
    tools:
      - name: "file_writer"
        type: "agent"
  
  file_writer:
    name: "file_writer"
    description: "File operations agent"
    model: "gpt-3.5-turbo"

# Define steps
steps:
  data_processor:
    name: "data_processor"
    description: "Process input data"
    executor: "local"
    input_data_units:
      - data_type: "memory"
    output_data_units:
      - data_type: "memory"
    trigger_config:
      trigger_type: "data_updated"

# Define links
links:
  - link_type: "direct"
    source: "data_processor"
    target: "code_generator"
```

### Schema Generation

```python
from nanobrain.src.config.schema_generator import generate_all_schemas

# Generate all schemas
generate_all_schemas("schemas/")
```

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Writing Tests

```python
import pytest
from nanobrain.src.agents.code_writer import CodeWriterAgent

@pytest.mark.asyncio
async def test_code_writer():
    agent = CodeWriterAgent()
    await agent.initialize()
    
    response = await agent.process("Create a hello world function")
    assert "def" in response
    
    await agent.shutdown()
```

## 🔌 Extending the Framework

### Creating Custom Agents

```python
from nanobrain.src.core.agent import Agent, AgentConfig

class CustomAgent(Agent):
    def __init__(self, config=None, **kwargs):
        if config is None:
            config = AgentConfig(
                name="custom_agent",
                description="Custom agent implementation",
                model="gpt-3.5-turbo"
            )
        super().__init__(config, **kwargs)
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Custom processing logic
        return await super().process(input_text, **kwargs)
```

### Creating Custom Steps

```python
from nanobrain.src.core.step import Step, StepConfig

class CustomStep(Step):
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Custom processing logic
        input_data = inputs.get('input_0')
        processed_data = self.custom_processing(input_data)
        return {"output": processed_data}
    
    def custom_processing(self, data):
        # Your custom logic here
        return data
```

### Creating Custom Executors

```python
from nanobrain.src.core.executor import ExecutorBase, ExecutorConfig

class CustomExecutor(ExecutorBase):
    async def execute(self, func: Callable, **kwargs) -> Any:
        # Custom execution logic
        return await func(**kwargs)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by modern workflow orchestration frameworks
- Built with async/await best practices
- Designed for scalability and maintainability

## 📚 Documentation

For detailed documentation, see:
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Examples](examples/)
- [Architecture Guide](docs/architecture.md)

## 🐛 Issues and Support

- Report bugs: [GitHub Issues](https://github.com/your-repo/nanobrain/issues)
- Ask questions: [Discussions](https://github.com/your-repo/nanobrain/discussions)
- Documentation: [Wiki](https://github.com/your-repo/nanobrain/wiki) 