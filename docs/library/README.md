# NanoBrain Library Documentation

Welcome to the NanoBrain Library documentation. This library provides reusable, production-ready components extracted from common patterns found in NanoBrain workflows and demos.

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Components](#components)
- [Examples](#examples)
- [Migration Guide](#migration-guide)
- [Contributing](#contributing)

## Overview

The NanoBrain Library is a collection of high-level, reusable components that implement common patterns found across NanoBrain applications. It provides:

- **Infrastructure Components**: Data management, interfaces, monitoring, and utilities
- **Agent Enhancements**: Protocol support, collaboration, and specialized behaviors
- **Workflow Orchestration**: Pre-built workflows for common use cases including distributed processing
- **Distributed Processing**: Parsl-based workflows for HPC clusters and cloud resources
- **Best Practices**: Production-ready implementations with proper error handling, logging, and performance optimization

### Package Structure (v1.1.0+)

The NanoBrain framework is now organized as a proper Python package:

```python
# Core framework components
from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.executor import ParslExecutor, ExecutorConfig

# Library components
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.library.workflows.chat_workflow import ChatWorkflowOrchestrator
from nanobrain.library.workflows.chat_workflow_parsl import ParslChatWorkflow

# Configuration management
from nanobrain.config import get_config_manager
```

## Architecture

The library follows a layered architecture with clear separation of concerns:

```
nanobrain/library/
├── infrastructure/          # Core infrastructure components
│   ├── data/               # Data abstractions and implementations
│   ├── interfaces/         # External system interfaces
│   ├── steps/              # Specialized step implementations
│   ├── logging/            # Enhanced logging and monitoring
│   ├── load_balancing/     # Load balancing and request management
│   └── monitoring/         # Performance monitoring and health checks
├── agents/                 # Enhanced agent implementations
│   ├── conversational/     # Enhanced conversational agents
│   └── specialized/        # Specialized agent implementations
└── workflows/              # Pre-built workflow orchestrators
    ├── chat_workflow/      # Standard chat workflow components
    └── chat_workflow_parsl/ # Distributed chat workflow with Parsl
```

### Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Dependency Injection**: Components are configurable and testable
3. **Protocol Agnostic**: Interfaces abstract away implementation details
4. **Performance First**: Built-in monitoring, caching, and optimization
5. **Production Ready**: Comprehensive error handling, logging, and recovery

## Quick Start

### Installation

```bash
# The library is part of the NanoBrain framework
cd nanobrain
pip install -e .
```

### Basic Usage

```python
from nanobrain.core.executor import ParslExecutor, ExecutorConfig
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.core.agent import AgentConfig

# Create agents
agents = [
    EnhancedCollaborativeAgent(AgentConfig(name=f"agent_{i}", model="gpt-3.5-turbo"))
    for i in range(3)
]

# Configure parallel processing with Parsl
executor_config = ExecutorConfig(
    executor_type="parsl",
    max_workers=4
)

executor = ParslExecutor(executor_config)
await executor.initialize()

# Initialize agents
for agent in agents:
    await agent.initialize()

# Process requests in parallel
futures = []
for i, message in enumerate(["Hello!", "How are you?", "What's new?"]):
    agent = agents[i % len(agents)]
    future = executor.submit(agent.process, message)
    futures.append(future)

# Collect results
results = []
for future in futures:
    result = await future
    results.append(result)

print("Responses:", results)
```

## Components

### Infrastructure

#### [Data Management](infrastructure/data/README.md)
- **DataUnitBase**: Abstract data unit interface
- **Memory/File/Stream Units**: Concrete implementations
- **ConversationHistory**: Persistent conversation storage
- **SessionManager**: Session lifecycle management

#### [Interfaces](infrastructure/interfaces/README.md)
- **Database Adapters**: MySQL, PostgreSQL, SQLite, MongoDB
- **CLI Components**: Interactive command-line interfaces
- **External Systems**: API, message queue, file system interfaces

#### [Parallel Processing](infrastructure/steps/README.md)
- **ParallelStep**: Generic parallel processing framework
- **ParallelAgentStep**: Agent-specific parallel processing
- **ParallelConversationalAgentStep**: Chat-optimized parallel processing

#### [Load Balancing](infrastructure/load_balancing/README.md)
- **RequestQueue**: Request queuing and prioritization
- **LoadBalancer**: Multiple load balancing strategies
- **CircuitBreaker**: Fault tolerance patterns

#### [Monitoring](infrastructure/monitoring/README.md)
- **PerformanceMonitor**: Comprehensive metrics collection
- **HealthChecker**: System health monitoring
- **MetricsDashboard**: Real-time statistics display

### Agents

#### [Enhanced Agents](agents/enhanced/README.md)
- **ProtocolMixin**: A2A and MCP protocol support
- **DelegationEngine**: Rule-based task delegation
- **CollaborativeAgent**: Multi-protocol agent implementation

### Workflows

#### [Chat Workflow](workflows/chat_workflow/README.md)
- **ChatWorkflowOrchestrator**: Complete chat workflow management
- **RequestProcessor**: Request processing pipeline
- **ResponseAggregator**: Response collection and formatting

#### [Parsl Chat Workflow](workflows/chat_workflow_parsl/README.md) ✅ **IMPLEMENTED**
- **ParslChatWorkflow**: Distributed chat processing with Parsl
- **Parsl Applications**: Distributed execution wrappers
- **Performance Monitoring**: Built-in metrics and statistics
- **Fault Tolerance**: Automatic error handling and recovery
- **Performance Monitoring**: Distributed execution metrics
- **Scalable Architecture**: From local multi-core to HPC clusters

## Examples

### Parallel Processing Example

```python
from library.infrastructure.steps import (
    ParallelConversationalAgentStep,
    ParallelConversationalAgentConfig,
    LoadBalancingStrategy
)

# Configure with advanced load balancing
config = ParallelConversationalAgentConfig(
    name="advanced_chat",
    max_parallel_requests=20,
    load_balancing_strategy=LoadBalancingStrategy.FASTEST_RESPONSE,
    enable_conversation_context=True,
    enable_response_caching=True,
    track_token_usage=True
)

step = ParallelConversationalAgentStep(config, agents)

# Process batch requests
batch_result = await step.process({
    'requests': [
        {'message': 'What is AI?', 'user_id': 'user1'},
        {'message': 'Explain quantum computing', 'user_id': 'user2'},
        {'message': 'How does machine learning work?', 'user_id': 'user3'}
    ]
})

# Get performance statistics
stats = step.get_performance_stats()
print(f"Processed {stats['total_requests']} requests")
print(f"Average response time: {stats['avg_processing_time']:.2f}s")
```

### Database Integration Example

```python
from library.infrastructure.data import ConversationHistoryUnit
from library.infrastructure.interfaces.database import SQLiteAdapter

# Configure database adapter
db_adapter = SQLiteAdapter("conversations.db")
await db_adapter.initialize()

# Create conversation history unit
history_unit = ConversationHistoryUnit(
    database_adapter=db_adapter,
    table_name="conversations"
)
await history_unit.initialize()

# Store conversation
await history_unit.save_message({
    'user_input': 'Hello',
    'agent_response': 'Hi there!',
    'conversation_id': 'conv_123',
    'timestamp': datetime.now()
})

# Retrieve conversation history
history = await history_unit.get_conversation_history('conv_123', limit=10)
```

### Enhanced Agent Example

```python
from library.agents.enhanced import CollaborativeAgent
from core.agent import AgentConfig

# Create enhanced agent with protocol support
config = AgentConfig(
    name="collaborative_agent",
    model="gpt-4",
    temperature=0.7
)

agent = CollaborativeAgent(
    config,
    a2a_config_path="config/a2a_config.yaml",
    mcp_config_path="config/mcp_config.yaml",
    delegation_rules=[
        {
            'keywords': ['calculate', 'math'],
            'agent': 'calculator_agent',
            'description': 'Delegate math calculations'
        }
    ]
)

await agent.initialize()

# Agent will automatically delegate or use tools as appropriate
response = await agent.process("Calculate the square root of 144")
```

## Migration Guide

### From Demo Scripts to Library Components

#### Before (Demo Script)
```python
# chat_workflow_demo.py - 1000+ lines
class LogManager:
    # 200+ lines of logging logic
    
class CLIInterface:
    # 300+ lines of CLI logic
    
class ChatWorkflow:
    # 500+ lines of workflow orchestration
```

#### After (Library Components)
```python
# Simplified demo script - ~50 lines
from library.workflows.chat_workflow import ChatWorkflowOrchestrator

async def main():
    orchestrator = ChatWorkflowOrchestrator.from_config("chat_config.yaml")
    await orchestrator.run()

if __name__ == "__main__":
    asyncio.run(main())
```

### Migration Steps

1. **Identify Common Patterns**: Look for repeated code across demos
2. **Extract to Library**: Move reusable components to appropriate library modules
3. **Update Imports**: Change imports to use library components
4. **Simplify Demos**: Reduce demo scripts to essential orchestration logic
5. **Add Configuration**: Use YAML configuration for component setup

## Performance Considerations

### Load Balancing Strategies

- **Round Robin**: Best for uniform workloads
- **Least Loaded**: Best for varying request complexity
- **Fastest Response**: Best for performance optimization
- **Weighted**: Best for heterogeneous agent capabilities

### Caching and Optimization

- **Response Caching**: Reduces redundant processing
- **Conversation Context**: Maintains chat continuity
- **Connection Pooling**: Optimizes database performance
- **Circuit Breakers**: Prevents cascade failures

### Monitoring and Metrics

- **Real-time Performance**: Track response times and throughput
- **Health Monitoring**: Automatic agent health checks
- **Resource Usage**: Monitor memory and CPU utilization
- **Error Tracking**: Comprehensive error logging and recovery

## Best Practices

### Configuration Management

```yaml
# chat_workflow_config.yaml
parallel_processing:
  max_parallel_requests: 10
  load_balancing_strategy: "fastest_response"
  enable_circuit_breaker: true
  
agents:
  - name: "primary_agent"
    model: "gpt-4"
    temperature: 0.7
  - name: "backup_agent"
    model: "gpt-3.5-turbo"
    temperature: 0.5

database:
  adapter: "sqlite"
  connection_string: "conversations.db"
  
logging:
  level: "INFO"
  enable_performance_tracking: true
```

### Error Handling

```python
try:
    result = await step.process(inputs)
except Exception as e:
    logger.error(f"Processing failed: {e}")
    # Implement fallback logic
    result = await fallback_processor.process(inputs)
```

### Testing

```python
import pytest
from library.infrastructure.steps import ParallelConversationalAgentStep

@pytest.fixture
async def mock_agents():
    # Create mock agents for testing
    pass

async def test_parallel_processing(mock_agents):
    step = ParallelConversationalAgentStep(config, mock_agents)
    result = await step.process({'user_input': 'test'})
    assert result['responses'][0]['success'] == True
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd nanobrain

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/library/

# Run linting
flake8 library/
black library/
```

### Adding New Components

1. **Create Module**: Add new module in appropriate library directory
2. **Write Tests**: Comprehensive unit and integration tests
3. **Add Documentation**: Update relevant README files
4. **Update Examples**: Add usage examples
5. **Submit PR**: Follow contribution guidelines

### Code Standards

- **Type Hints**: All public APIs must have type hints
- **Documentation**: Comprehensive docstrings for all classes and methods
- **Testing**: Minimum 90% test coverage
- **Logging**: Structured logging with appropriate levels
- **Error Handling**: Graceful error handling with recovery strategies

## Support

- **Documentation**: [docs/library/](.)
- **Examples**: [examples/library/](../examples/library/)
- **Issues**: [GitHub Issues](https://github.com/nanobrain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nanobrain/discussions)

---

*This documentation is part of the NanoBrain Framework. For core framework documentation, see [docs/](../README.md).* 