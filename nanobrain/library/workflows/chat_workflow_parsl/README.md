# Parsl Chat Workflow

A comprehensive distributed chat workflow implementation using Parsl for high-performance parallel execution within the NanoBrain framework.

## Overview

The Parsl Chat Workflow demonstrates how to integrate Parsl's distributed computing capabilities with NanoBrain's conversational AI infrastructure. This workflow showcases:

- **Distributed Execution**: Multiple conversational agents running in parallel using Parsl
- **Existing Infrastructure Integration**: Uses NanoBrain's `ParslExecutor` and library components
- **Integrated Architecture**: Parsl functionality integrated directly into agent and step classes
- **Performance Monitoring**: Comprehensive metrics and monitoring for distributed execution
- **Scalable Design**: Can scale from local execution to HPC and cloud environments

## Architecture

### Core Components

1. **ParslChatWorkflow**: Main workflow orchestrator
2. **ParslDistributedAgent**: Enhanced agent with integrated Parsl capabilities
3. **ParslAgentProcessingStep**: Step-based processing with distributed execution
4. **Data Units**: Memory and persistent storage for conversation management
5. **Performance Monitoring**: Metrics collection and analysis

### Directory Structure

```
library/workflows/chat_workflow_parsl/
├── workflow.py                           # Main workflow with integrated Parsl functionality
├── ParslChatWorkflow.yml                # Workflow configuration
├── config/                              # Component configurations
│   ├── EnhancedCollaborativeAgent_1.yml # Creative agent config
│   ├── EnhancedCollaborativeAgent_2.yml # Analytical agent config
│   ├── EnhancedCollaborativeAgent_3.yml # Balanced agent config
│   └── ConversationHistoryUnit.yml     # History storage config
└── README.md                           # This documentation
```

## Features

### Distributed Processing
- **Parallel Agent Execution**: Multiple agents process requests simultaneously
- **Load Balancing**: Intelligent distribution of requests across available agents
- **Fault Tolerance**: Graceful handling of agent failures and retries
- **Resource Management**: Efficient utilization of compute resources

### Parsl Integration
- **Native ParslExecutor**: Uses existing NanoBrain Parsl infrastructure
- **Integrated Apps**: Parsl apps defined directly within agent and step classes
- **Flexible Configuration**: Support for local, HPC, and cloud execution
- **Performance Monitoring**: Integration with Parsl's monitoring capabilities

### Agent Specialization
- **Agent 1 (Creative)**: High temperature (0.8) for creative and innovative responses
- **Agent 2 (Analytical)**: Low temperature (0.3) for precise and factual responses  
- **Agent 3 (Balanced)**: Medium temperature (0.7) for general-purpose responses

### Data Management
- **Conversation History**: Persistent storage with Parsl execution metadata
- **Performance Metrics**: Comprehensive tracking of execution statistics
- **Memory Management**: Efficient caching and data flow between components

## Key Classes

### ParslDistributedAgent

Extends `EnhancedCollaborativeAgent` with Parsl capabilities:

```python
class ParslDistributedAgent(EnhancedCollaborativeAgent):
    """Enhanced agent with Parsl distributed processing capabilities."""
    
    @python_app
    def _parsl_process_message(self, message: str, agent_config_dict: Dict[str, Any]) -> str:
        """Parsl app for processing messages in distributed workers."""
        # Implementation handles serialization and remote execution
    
    async def process_distributed(self, message: str) -> str:
        """Process message using Parsl distributed execution."""
        # Coordinates distributed processing with fallback to local
```

### ParslAgentProcessingStep

Manages multiple distributed agents:

```python
class ParslAgentProcessingStep:
    """Agent processing step with Parsl distributed execution capabilities."""
    
    @python_app
    def _parsl_aggregate_responses(self, responses: List[str], message: str) -> Dict[str, Any]:
        """Parsl app for aggregating responses from multiple agents."""
        # Handles response aggregation in distributed environment
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process message using multiple distributed agents."""
        # Coordinates parallel execution and aggregation
```

## Configuration

### Workflow Configuration (`ParslChatWorkflow.yml`)

The main configuration file defines:
- Parsl executor settings with HighThroughputExecutor
- Multiple agent configurations
- Data unit specifications
- Performance and monitoring settings

### Agent Configurations

Each agent has its own configuration file with:
- Model parameters (temperature, max_tokens)
- Specialized system prompts
- Parsl-specific resource requirements
- Performance optimization settings

### Parsl Configuration

Supports multiple execution environments:
- **Local**: Single-node execution for development
- **HPC**: SLURM-based cluster execution
- **Cloud**: AWS-based distributed execution
- **Multi-executor**: Hybrid execution strategies

## Usage

### Basic Usage

```python
from workflow import create_parsl_chat_workflow

# Create and initialize workflow
workflow = await create_parsl_chat_workflow("ParslChatWorkflow.yml")

# Process user input
response = await workflow.process_user_input("Hello, how does Parsl work?")
print(response)

# Get performance statistics
stats = await workflow.get_performance_stats()
print(f"Processing time: {stats['avg_response_time']:.3f}s")

# Shutdown
await workflow.shutdown()
```

### Running the Demo

```bash
# From the project root
cd demo/chat_workflow_parsl
python run_comprehensive_demo.py
```

### Demo Commands

- `/help` - Show help information
- `/stats` - Display performance statistics
- `/status` - Show workflow status
- `/batch N` - Run batch test with N messages
- `/quit` - Exit the demo

## Performance

### Benchmarks

The workflow provides comprehensive performance metrics:
- **Response Time**: Average processing time per request
- **Throughput**: Messages processed per second
- **Agent Utilization**: Distribution of work across agents
- **Resource Usage**: CPU and memory consumption
- **Success Rate**: Percentage of successful completions

### Optimization

Performance can be optimized through:
- **Agent Count**: Adjust number of parallel agents
- **Batch Processing**: Group multiple requests
- **Caching**: Enable response caching for repeated queries
- **Resource Allocation**: Tune Parsl executor configuration

## Scalability

### Local Development
- 2-4 agents on single machine
- ThreadPoolExecutor for lightweight tasks
- Minimal resource requirements

### Production Deployment
- 8-16+ agents across multiple nodes
- HighThroughputExecutor with SLURM
- Comprehensive monitoring and logging

### Cloud Scaling
- Auto-scaling based on demand
- Multi-region deployment
- Integration with cloud monitoring services

## Integration

### NanoBrain Framework
- Uses existing `ParslExecutor` from `nanobrain/core/executor.py`
- Integrates with library agents and infrastructure
- Follows framework patterns and conventions

### External Systems
- Compatible with existing chat interfaces
- API integration capabilities
- Monitoring system integration

## Development

### Adding New Agents

1. Create agent configuration in `config/`
2. Add agent to workflow configuration
3. Update agent initialization in `workflow.py`

### Custom Parsl Functionality

1. Add new `@python_app` methods to `ParslDistributedAgent`
2. Extend `ParslAgentProcessingStep` with additional processing logic
3. Update workflow orchestration as needed

### Configuration Customization

1. Modify `ParslChatWorkflow.yml` for workflow settings
2. Update component configs in `config/` directory
3. Adjust Parsl configuration in workflow initialization

## Advantages of Integrated Architecture

### Simplified Structure
- **No Separate Apps File**: Parsl functionality integrated directly into classes
- **Better Encapsulation**: Each class manages its own Parsl apps
- **Cleaner Dependencies**: Reduced file interdependencies

### Improved Maintainability
- **Single Responsibility**: Each class handles its own distributed processing
- **Easier Testing**: Parsl functionality can be tested with the class
- **Better Documentation**: Parsl apps documented alongside their usage

### Enhanced Flexibility
- **Dynamic Configuration**: Parsl behavior can be adjusted per agent/step
- **Conditional Execution**: Easy to enable/disable Parsl based on conditions
- **Fallback Handling**: Integrated fallback to local processing

## Troubleshooting

### Common Issues

1. **Parsl Import Errors**: Ensure Parsl is installed (`pip install parsl`)
2. **Serialization Issues**: Check that all dependencies are available on workers
3. **Configuration Errors**: Verify YAML configuration syntax and paths
4. **Performance Issues**: Monitor resource usage and adjust worker counts

### Debug Mode

Enable debug logging for detailed execution information:

```python
from nanobrain.core.logging_system import set_debug_mode
set_debug_mode(True)
```

### Monitoring

Check workflow status and performance:

```python
status = workflow.get_workflow_status()
stats = await workflow.get_performance_stats()
parsl_stats = await workflow.get_parsl_stats()
```

## Dependencies

### Required
- `parsl`