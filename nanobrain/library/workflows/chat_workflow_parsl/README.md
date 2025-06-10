# Parsl Chat Workflow

A comprehensive distributed chat workflow implementation using Parsl for high-performance parallel execution within the NanoBrain framework.

## Overview

The Parsl Chat Workflow demonstrates how to integrate Parsl's distributed computing capabilities with NanoBrain's conversational AI infrastructure. This workflow showcases:

- **Distributed Execution**: Multiple conversational agents running in parallel using Parsl
- **Existing Infrastructure Integration**: Uses NanoBrain's `ParslExecutor` and library components
- **Modular Architecture**: Follows Parsl best practices for separating core logic, apps, and configuration
- **Performance Monitoring**: Comprehensive metrics and monitoring for distributed execution
- **Scalable Design**: Can scale from local execution to HPC and cloud environments

## Architecture

### Core Components

1. **ParslChatWorkflow**: Main workflow orchestrator
2. **Parsl Apps**: Distributed functions for chat processing and aggregation
3. **Multiple Agents**: Specialized conversational agents for parallel processing
4. **Data Units**: Memory and persistent storage for conversation management
5. **Performance Monitoring**: Metrics collection and analysis

### Directory Structure

```
library/workflows/chat_workflow_parsl/
├── workflow.py                           # Main workflow implementation
├── ParslChatWorkflow.yml                # Workflow configuration
├── config/                              # Component configurations
│   ├── EnhancedCollaborativeAgent_1.yml # Creative agent config
│   ├── EnhancedCollaborativeAgent_2.yml # Analytical agent config
│   ├── EnhancedCollaborativeAgent_3.yml # Balanced agent config
│   └── ConversationHistoryUnit.yml     # History storage config
├── apps/                                # Parsl applications (following best practices)
│   ├── __init__.py                      # Module initialization
│   ├── logic.py                         # Core business logic (Parsl-agnostic)
│   ├── app.py                           # Parsl app wrappers
│   └── config.py                        # Parsl configuration functions
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
- **Modular Apps**: Follows Parsl best practices for app organization
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
workflow = await create_parsl_chat_workflow()

# Process user input
response = await workflow.process_user_input("Hello, how does Parsl work?")
print(response)

# Get performance statistics
stats = await workflow.get_performance_stats()
print(f"Processing time: {stats['avg_processing_time']:.3f}s")

# Shutdown
await workflow.shutdown()
```

### Running the Demo

```bash
# From the project root
cd demo/chat_workflow_parsl
python run_parsl_chat_demo.py
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
- Uses existing `ParslExecutor` from `src/core/executor.py`
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

### Custom Parsl Apps

1. Add core logic to `apps/logic.py`
2. Create app wrapper in `apps/app.py`
3. Register app in workflow

### Configuration Customization

1. Modify `ParslChatWorkflow.yml` for workflow settings
2. Update component configs in `config/` directory
3. Adjust Parsl configuration in `apps/config.py`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Parsl Initialization**: Check Parsl configuration and resources
3. **Agent Failures**: Review agent configurations and system prompts
4. **Performance Issues**: Monitor resource usage and adjust settings

### Debugging

- Enable debug logging in configuration
- Use `/status` command to check component health
- Monitor Parsl execution through built-in monitoring
- Check conversation history for execution metadata

## Dependencies

### Required
- `parsl` - Distributed computing framework
- `asyncio` - Asynchronous programming
- `yaml` - Configuration file parsing
- `sqlite3` - Conversation history storage

### Optional
- `openai` - LLM integration
- `psutil` - System monitoring
- `cloudpickle` - Enhanced serialization

## Contributing

When contributing to this workflow:

1. Follow Parsl best practices for modular applications
2. Maintain separation between core logic and Parsl apps
3. Update documentation for new features
4. Add tests for new functionality
5. Ensure compatibility with existing NanoBrain infrastructure

## License

This workflow is part of the NanoBrain framework and follows the same licensing terms. 