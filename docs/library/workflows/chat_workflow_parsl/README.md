# Parsl Chat Workflow

The Parsl Chat Workflow provides distributed chat processing capabilities using the Parsl parallel computing framework. It extends the standard chat workflow with distributed execution, enabling high-performance parallel processing across multiple workers and compute resources.

## Overview

This workflow demonstrates:
- **Distributed Processing**: Leverage Parsl for distributed execution across multiple workers
- **Parallel Agent Processing**: Multiple agents processing requests simultaneously
- **Scalable Architecture**: Scale from local multi-core to HPC clusters and cloud resources
- **Performance Monitoring**: Track distributed execution metrics and performance
- **Fault Tolerance**: Handle worker failures and network issues gracefully

## Architecture

```
nanobrain/library/workflows/chat_workflow_parsl/
├── workflow.py                           # Main ParslChatWorkflow class
├── ParslChatWorkflow.yml                # Main workflow configuration
├── config/                              # Component configurations
│   ├── EnhancedCollaborativeAgent_1.yml # Creative agent config
│   ├── EnhancedCollaborativeAgent_2.yml # Analytical agent config
│   ├── EnhancedCollaborativeAgent_3.yml # Balanced agent config
│   └── ConversationHistoryUnit.yml     # Data unit configuration
├── apps/                                # Parsl applications
│   ├── __init__.py
│   ├── logic.py                         # Core business logic (Parsl-agnostic)
│   ├── app.py                           # Parsl app wrappers
│   └── config.py                        # Parsl configuration functions
└── README.md                           # This documentation
```

## Core Components

### ParslChatWorkflow

The main workflow orchestrator that coordinates distributed chat processing.

```python
from nanobrain.library.workflows.chat_workflow_parsl.workflow import ParslChatWorkflow

# Create and initialize workflow
workflow = ParslChatWorkflow()
await workflow.initialize("config/ParslChatWorkflow.yml")

# Process messages with distributed agents
response = await workflow.process_user_input("Hello, explain quantum computing")
print(response)

# Get performance statistics
stats = await workflow.get_performance_stats()
print(f"Processed {stats['total_requests']} requests")
print(f"Average response time: {stats['avg_response_time']:.2f}s")

await workflow.shutdown()
```

**Key Features:**
- Integration with `nanobrain.core.executor.ParslExecutor`
- Multiple specialized agents for different types of queries
- Distributed conversation history management
- Performance monitoring and metrics collection
- Graceful error handling and recovery

### Parsl Applications

The workflow uses Parsl applications for distributed execution:

#### Core Logic (`apps/logic.py`)
Pure business logic without Parsl dependencies:

```python
def process_chat_message(agent_config: dict, message: str, context: dict) -> dict:
    """
    Process a chat message with an agent.
    
    Args:
        agent_config: Agent configuration dictionary
        message: User message to process
        context: Conversation context
        
    Returns:
        dict: Response with content, metadata, and performance metrics
    """
    # Implementation details...
```

#### Parsl Apps (`apps/app.py`)
Parsl application wrappers for distributed execution:

```python
from parsl import python_app
from .logic import process_chat_message

@python_app
def parsl_chat_app(agent_config, message, context):
    """Parsl app wrapper for chat message processing."""
    return process_chat_message(agent_config, message, context)

@python_app
def parsl_aggregate_app(responses):
    """Parsl app for aggregating multiple agent responses."""
    return aggregate_responses(responses)
```

#### Parsl Configuration (`apps/config.py`)
Environment-specific Parsl configurations:

```python
def get_local_config(max_workers: int = 4):
    """Get Parsl configuration for local execution."""
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    
    return Config(
        executors=[
            HighThroughputExecutor(
                label="local_htex",
                max_workers=max_workers,
                cores_per_worker=1
            )
        ]
    )

def get_hpc_config(partition: str = "compute", nodes: int = 2):
    """Get Parsl configuration for HPC execution."""
    # HPC-specific configuration...
```

## Configuration

### Main Workflow Configuration (`ParslChatWorkflow.yml`)

```yaml
name: "parsl_chat_workflow"
description: "Distributed chat workflow using Parsl for parallel processing"

config:
  # Executor configuration
  executor:
    type: "parsl"
    max_workers: 4
    timeout: 300
    
  # Agent configurations
  agents:
    - name: "creative_agent"
      config_file: "config/EnhancedCollaborativeAgent_1.yml"
    - name: "analytical_agent"
      config_file: "config/EnhancedCollaborativeAgent_2.yml"
    - name: "balanced_agent"
      config_file: "config/EnhancedCollaborativeAgent_3.yml"
  
  # Data units
  data_units:
    conversation_history:
      type: "memory"
      config_file: "config/ConversationHistoryUnit.yml"
  
  # Parsl-specific settings
  parsl:
    environment: "local"  # local, hpc, cloud
    monitoring: false
    checkpoint_mode: "task_exit"
    
  # Performance settings
  performance:
    enable_metrics: true
    batch_size: 5
    timeout_seconds: 60
```

### Agent Configuration Example (`config/EnhancedCollaborativeAgent_1.yml`)

```yaml
name: "creative_agent"
description: "Creative and imaginative conversational agent"

config:
  # Core agent settings
  model: "gpt-4"
  temperature: 0.9
  max_tokens: 1000
  
  # System prompt loaded from YAML
  system_prompt: |
    You are a creative and imaginative AI assistant. You approach problems
    with creativity and think outside the box. You provide innovative
    solutions and engaging explanations that capture the user's imagination.
    
    Key traits:
    - Creative and innovative thinking
    - Engaging storytelling abilities
    - Metaphorical explanations
    - Encouraging exploration of ideas
  
  # Collaboration settings
  collaboration:
    delegation_threshold: 0.7
    specialization_keywords:
      - "creative"
      - "story"
      - "imagine"
      - "brainstorm"
      - "innovative"
  
  # Performance settings
  performance:
    enable_caching: true
    cache_ttl: 3600
    retry_attempts: 3
```

## Usage Examples

### Basic Usage

```python
import asyncio
from nanobrain.library.workflows.chat_workflow_parsl.workflow import create_parsl_chat_workflow

async def basic_example():
    # Create workflow from configuration
    workflow = await create_parsl_chat_workflow("config/ParslChatWorkflow.yml")
    
    # Process a message
    response = await workflow.process_user_input(
        "Explain the concept of distributed computing"
    )
    
    print(f"Response: {response}")
    
    # Cleanup
    await workflow.shutdown()

asyncio.run(basic_example())
```

### Batch Processing

```python
async def batch_processing_example():
    workflow = await create_parsl_chat_workflow("config/ParslChatWorkflow.yml")
    
    # Process multiple messages in parallel
    messages = [
        "What is machine learning?",
        "Explain quantum computing",
        "How does blockchain work?",
        "What are neural networks?",
        "Describe cloud computing"
    ]
    
    # Submit all messages for parallel processing
    futures = []
    for message in messages:
        future = workflow.submit_message(message)
        futures.append(future)
    
    # Collect results as they complete
    results = []
    for future in futures:
        result = await future
        results.append(result)
    
    for message, result in zip(messages, results):
        print(f"Q: {message}")
        print(f"A: {result}")
        print("-" * 50)
    
    await workflow.shutdown()
```

### Performance Monitoring

```python
async def monitoring_example():
    workflow = await create_parsl_chat_workflow("config/ParslChatWorkflow.yml")
    
    # Process some messages
    for i in range(10):
        await workflow.process_user_input(f"Test message {i}")
    
    # Get detailed performance statistics
    stats = await workflow.get_performance_stats()
    
    print("Performance Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Successful Requests: {stats['successful_requests']}")
    print(f"  Failed Requests: {stats['failed_requests']}")
    print(f"  Average Response Time: {stats['avg_response_time']:.2f}s")
    print(f"  Min Response Time: {stats['min_response_time']:.2f}s")
    print(f"  Max Response Time: {stats['max_response_time']:.2f}s")
    print(f"  Throughput: {stats['throughput']:.2f} req/s")
    
    # Get Parsl-specific metrics
    parsl_stats = await workflow.get_parsl_stats()
    print(f"  Active Workers: {parsl_stats['active_workers']}")
    print(f"  Pending Tasks: {parsl_stats['pending_tasks']}")
    print(f"  Completed Tasks: {parsl_stats['completed_tasks']}")
    
    await workflow.shutdown()
```

## Integration with NanoBrain Core

### ParslExecutor Integration

The workflow leverages the existing `nanobrain.core.executor.ParslExecutor`:

```python
from nanobrain.core.executor import ParslExecutor, ExecutorConfig

# The workflow automatically configures ParslExecutor
executor_config = ExecutorConfig(
    executor_type="parsl",
    max_workers=4,
    timeout=300
)

executor = ParslExecutor(config=executor_config)
await executor.initialize()

# Submit tasks through the executor
future = executor.submit(process_chat_message, agent_config, message, context)
result = await future
```

### Agent Integration

Uses enhanced collaborative agents from the library:

```python
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.core.agent import AgentConfig

# Agents are configured via YAML and instantiated by the workflow
agent_config = AgentConfig.from_yaml("config/EnhancedCollaborativeAgent_1.yml")
agent = EnhancedCollaborativeAgent(agent_config)
```

### Data Management

Integrates with core data units for conversation history:

```python
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig

# Conversation history is managed through data units
history_config = DataUnitConfig(
    name="conversation_history",
    data_type="memory",
    persistent=False
)

history_unit = DataUnitMemory(history_config)
await history_unit.initialize()
```

## Configuration

### Main Workflow Configuration (`ParslChatWorkflow.yml`)

```yaml
name: "parsl_chat_workflow"
description: "Distributed chat workflow using Parsl for parallel processing"

config:
  # Executor configuration
  executor:
    type: "parsl"
    max_workers: 4
    timeout: 300
    
  # Agent configurations
  agents:
    - name: "creative_agent"
      config_file: "config/EnhancedCollaborativeAgent_1.yml"
    - name: "analytical_agent"
      config_file: "config/EnhancedCollaborativeAgent_2.yml"
    - name: "balanced_agent"
      config_file: "config/EnhancedCollaborativeAgent_3.yml"
  
  # Data units
  data_units:
    conversation_history:
      type: "memory"
      config_file: "config/ConversationHistoryUnit.yml"
  
  # Parsl-specific settings
  parsl:
    environment: "local"  # local, hpc, cloud
    monitoring: false
    checkpoint_mode: "task_exit"
    
  # Performance settings
  performance:
    enable_metrics: true
    batch_size: 5
    timeout_seconds: 60
```

### Agent Configuration Example (`config/EnhancedCollaborativeAgent_1.yml`)

```yaml
name: "creative_agent"
description: "Creative and imaginative conversational agent"

config:
  # Core agent settings
  model: "gpt-4"
  temperature: 0.9
  max_tokens: 1000
  
  # System prompt loaded from YAML
  system_prompt: |
    You are a creative and imaginative AI assistant. You approach problems
    with creativity and think outside the box. You provide innovative
    solutions and engaging explanations that capture the user's imagination.
    
    Key traits:
    - Creative and innovative thinking
    - Engaging storytelling abilities
    - Metaphorical explanations
    - Encouraging exploration of ideas
  
  # Collaboration settings
  collaboration:
    delegation_threshold: 0.7
    specialization_keywords:
      - "creative"
      - "story"
      - "imagine"
      - "brainstorm"
      - "innovative"
  
  # Performance settings
  performance:
    enable_caching: true
    cache_ttl: 3600
    retry_attempts: 3
```

## Usage Examples

### Basic Usage

```python
import asyncio
from nanobrain.library.workflows.chat_workflow_parsl.workflow import create_parsl_chat_workflow

async def basic_example():
    # Create workflow from configuration
    workflow = await create_parsl_chat_workflow("config/ParslChatWorkflow.yml")
    
    # Process a message
    response = await workflow.process_user_input(
        "Explain the concept of distributed computing"
    )
    
    print(f"Response: {response}")
    
    # Cleanup
    await workflow.shutdown()

asyncio.run(basic_example())
```

## Testing Strategy

### Unit Tests Required

1. **Component Tests**:
   - `test_parsl_chat_workflow.py` - ParslChatWorkflow functionality
   - `test_parsl_executor_integration.py` - Integration with core ParslExecutor
   - `test_parsl_conversation_history.py` - Distributed conversation history
   - `test_parsl_agent_integration.py` - Agent integration and configuration

2. **Integration Tests**:
   - `test_parsl_workflow_orchestrator.py` - Full workflow orchestration
   - `test_parsl_distributed_processing.py` - Multi-worker processing
   - `test_parsl_serialization.py` - Data serialization across workers
   - `test_parsl_performance_monitoring.py` - Performance metrics collection

3. **End-to-End Tests**:
   - `test_parsl_chat_workflow_e2e.py` - Complete workflow with real Parsl execution
   - `test_parsl_chat_workflow_scaling.py` - Scaling behavior with multiple workers
   - `test_parsl_chat_workflow_error_handling.py` - Error recovery and resilience

### Demo Validation Tests

1. **Functionality Tests**:
   - Message processing with multiple agents
   - Parallel execution verification
   - Performance metrics collection
   - Proper logging configuration respect

2. **Configuration Tests**:
   - YAML configuration loading
   - Agent configuration validation
   - Parsl configuration generation
   - Database connection testing

## Performance Considerations

### Scaling Strategies

1. **Local Multi-Core**: Use `HighThroughputExecutor` with multiple workers
2. **HPC Clusters**: Configure Slurm/PBS providers for cluster execution
3. **Cloud Resources**: Use cloud providers (AWS, GCP, Azure) for elastic scaling
4. **Hybrid Deployments**: Combine local and remote resources

### Optimization Tips

1. **Batch Processing**: Group similar requests for better throughput
2. **Caching**: Enable response caching for repeated queries
3. **Load Balancing**: Distribute requests across available agents
4. **Resource Monitoring**: Monitor CPU, memory, and network usage
5. **Fault Tolerance**: Configure retries and error handling

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `nanobrain` package is properly installed
2. **Parsl Configuration**: Check Parsl provider and executor settings
3. **Serialization Issues**: Verify all objects are properly serializable
4. **Worker Failures**: Check worker logs and resource availability
5. **Network Issues**: Verify connectivity between workers and head node

### Debugging

Enable debug logging for detailed information:

```python
import logging
from nanobrain.core.logging_system import set_debug_mode

# Enable debug mode
set_debug_mode(True)

# Configure Parsl logging
logging.getLogger('parsl').setLevel(logging.DEBUG)
```

## Implementation Requirements

### Critical Components Needed

1. **Core Infrastructure**:
   - Fix package imports and structure consistency
   - Implement basic `ParslAgent` with proper serialization
   - Create minimal `ConversationHistory` component
   - Fix logging configuration issues

2. **Workflow Components**:
   - Implement `ParslChatWorkflow` class
   - Create Parsl application wrappers (`apps/app.py`)
   - Implement core business logic (`apps/logic.py`)
   - Add Parsl configuration management (`apps/config.py`)

3. **Configuration System**:
   - Align configuration files with existing patterns
   - Create separate config files for each component class
   - Ensure YAML-based agent prompt loading
   - Implement proper configuration validation

### Success Criteria

#### Functional Requirements
- [ ] Demo runs without import errors
- [ ] Parsl executor initializes correctly
- [ ] Multiple agents process messages in parallel
- [ ] Conversation history persists correctly
- [ ] Logging respects global configuration
- [ ] Performance metrics are collected
- [ ] Proper shutdown and cleanup

#### Quality Requirements
- [ ] All unit tests pass
- [ ] Integration tests validate distributed processing
- [ ] Documentation matches implementation
- [ ] Configuration system is consistent with library patterns
- [ ] Error handling is comprehensive
- [ ] Performance is acceptable for demo scenarios

## Related Documentation

- [NanoBrain Core Executor Documentation](../../../core/executor.md)
- [Enhanced Collaborative Agent Documentation](../../agents/conversational/enhanced_collaborative_agent.md)
- [Chat Workflow Documentation](../chat_workflow/README.md)
- [Parsl Documentation](https://parsl.readthedocs.io/)
- [Performance Optimization Guide](../../performance/optimization.md) 