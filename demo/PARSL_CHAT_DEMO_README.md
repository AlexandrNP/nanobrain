# NanoBrain Parsl Chat Workflow Demo

A comprehensive demonstration of the NanoBrain framework featuring distributed parallel execution using Parsl for high-performance computing environments.

## Overview

This demo showcases the integration of Parsl (Parallel Scripting Library) with the NanoBrain framework to create a scalable, distributed chat workflow that can leverage multiple cores, nodes, and even HPC clusters for processing conversational AI requests.

## Features

### 🚀 Parallel Processing
- **Real LLM Integration**: Full OpenAI GPT integration for authentic conversational AI responses
- **Multiple Agents**: Deploy multiple conversational agents in parallel
- **Load Balancing**: Automatic distribution of chat requests across available agents
- **Concurrent Execution**: Process multiple chat requests simultaneously

### 🏗️ Distributed Architecture
- **Parsl Integration**: Leverage Parsl's distributed execution capabilities
- **HPC Support**: Scale from local multi-core to HPC clusters
- **Resource Management**: Automatic resource allocation and cleanup

### 📊 Performance Monitoring
- **Real-time Metrics**: Track processing times, throughput, and resource utilization
- **Performance Analytics**: Detailed performance reports and statistics
- **Bottleneck Detection**: Identify and analyze performance bottlenecks

### 🔧 Configuration Management
- **YAML Configuration**: Easy-to-modify configuration files
- **Environment Adaptation**: Automatic adaptation to different computing environments
- **Fallback Mechanisms**: Graceful degradation when resources are unavailable

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Input     │───▶│  Load Balancer   │───▶│ Parsl Executor  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌──────────────────┐             │
                       │ Response Aggreg. │◀────────────┘
                       └──────────────────┘
                                │
                       ┌─────────────────┐
                       │   CLI Output    │
                       └─────────────────┘

Parallel Agent Pool:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Agent 1   │  │   Agent 2   │  │   Agent 3   │
│ (Core 1-2)  │  │ (Core 3-4)  │  │ (Core 5-6)  │
└─────────────┘  └─────────────┘  └─────────────┘
```

## Prerequisites

### Required Dependencies
```bash
# Core NanoBrain dependencies
pip install asyncio pydantic

# Parsl for distributed execution
pip install parsl

# Optional: OpenAI for LLM integration
pip install openai

# Optional: Additional Parsl providers for HPC
pip install parsl[monitoring,slurm,kubernetes]
```

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 2GB RAM (4GB+ recommended for parallel execution)
- **CPU**: Multi-core processor (2+ cores recommended)
- **Network**: For distributed execution across multiple nodes

### Optional: HPC Environment
- **SLURM**: For HPC cluster execution
- **Kubernetes**: For container-based distributed execution
- **SSH**: For remote node execution

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd nanobrain
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install parsl
   ```

3. **Configure API Keys** (Optional)
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

4. **Verify Parsl Installation**
   ```bash
   python -c "import parsl; print('Parsl version:', parsl.__version__)"
   ```

## Quick Start

### Basic Local Execution
```bash
cd nanobrain/demo
python chat_workflow_parsl_demo.py
```

### With Custom Configuration
```bash
# Edit configuration file
nano config/parsl_chat_config.yml

# Run with custom config
python chat_workflow_parsl_demo.py --config config/parsl_chat_config.yml
```

## Usage

### Interactive Chat
Once started, the demo provides an interactive chat interface:

```
🚀 NanoBrain Parsl Chat Workflow Demo
==================================================
Features:
  • Parallel processing with Parsl executor
  • Multiple conversational agents
  • Distributed execution capabilities

Commands:
  /help     - Show this help
  /quit     - Exit the demo
==================================================

You: Hello! How does parallel processing work?
Bot: Hi! I'm Agent 2 in a parallel processing system using Parsl for distributed execution...

You: /help
📖 Parsl Chat Workflow Demo Help
This demo uses Parsl executor for parallel processing
Your messages are processed by multiple agents in parallel
Type /quit to exit
```

### Available Commands

| Command        | Description                           |
| -------------- | ------------------------------------- |
| `/help`        | Show help information                 |
| `/quit`        | Exit the demo                         |
| `/stats`       | Show performance statistics           |
| `/batch N`     | Send N test messages for load testing |
| `/agents`      | Show agent status and information     |
| `/performance` | Display detailed performance metrics  |

## Configuration

### Basic Configuration (`config/parsl_chat_config.yml`)

```yaml
# Executor configuration
executors:
  parsl_executor:
    executor_type: "parsl"
    max_workers: 8
    parsl_config:
      executors:
        - class: "parsl.executors.HighThroughputExecutor"
          label: "htex_local"
          max_workers: 8
          cores_per_worker: 1

# Agent configuration
agents:
  parallel_agents:
    count: 3
    base_config:
      model_name: "gpt-3.5-turbo"
      temperature: 0.7
      max_tokens: 500
```

### HPC Cluster Configuration

For SLURM-based HPC clusters:

```yaml
executors:
  parsl_executor:
    executor_type: "parsl"
    max_workers: 64
    parsl_config:
      executors:
        - class: "parsl.executors.HighThroughputExecutor"
          label: "htex_slurm"
          max_workers: 64
          provider:
            class: "parsl.providers.SlurmProvider"
            nodes_per_block: 4
            cores_per_node: 16
            walltime: "01:00:00"
            scheduler_options: "#SBATCH --partition=compute"
```

### Kubernetes Configuration

For Kubernetes-based execution:

```yaml
executors:
  parsl_executor:
    executor_type: "parsl"
    max_workers: 32
    parsl_config:
      executors:
        - class: "parsl.executors.KubernetesExecutor"
          label: "k8s_executor"
          max_workers: 32
          namespace: "nanobrain"
          image: "nanobrain:latest"
```

## Performance Monitoring

### Real-time Metrics
The demo tracks various performance metrics:

- **Request Processing Time**: Time to process individual chat requests
- **Agent Response Time**: Time for agents to generate responses
- **Executor Utilization**: How efficiently Parsl resources are used
- **Parallel Efficiency**: Effectiveness of parallel processing
- **Error Rate**: Frequency of processing errors

### Performance Reports
Detailed performance reports are automatically generated:

```
📊 Performance Statistics
==============================
Total Requests:    25
Total Responses:   25
Error Count:       0
Total Proc. Time:  12.45s
Avg Response Time: 0.498s

Detailed Metrics:
  request_processing_time:
    Count: 25
    Avg:   0.498
    Min:   0.234
    Max:   0.892
==============================
```

## Troubleshooting

### Common Issues

#### 1. Parsl Import Error
```
ImportError: No module named 'parsl'
```
**Solution**: Install Parsl
```bash
pip install parsl
```

#### 2. Executor Initialization Failed
```
⚠️  Parsl initialization failed: ...
   Falling back to local execution for demo
```
**Solution**: Check Parsl configuration and system resources

#### 3. API Key Not Found
```
⚠️  Warning: No OpenAI API key found!
```
**Solution**: Set environment variable or use mock mode
```bash
export OPENAI_API_KEY="your-key-here"
```

#### 4. Resource Allocation Issues
```
Error: Unable to allocate requested resources
```
**Solution**: Reduce `max_workers` in configuration or check system resources

### Debug Mode
Enable debug logging for detailed troubleshooting:

```bash
export NANOBRAIN_DEBUG=1
python chat_workflow_parsl_demo.py
```

### Log Files
Check log files for detailed error information:
```bash
ls logs/parsl_chat/session_*/
tail -f logs/parsl_chat/session_*/parsl/parsl_workflow.log
```

## Advanced Usage

### Custom Parsl Configurations

#### Multi-Site Execution
```python
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import SlurmProvider, LocalProvider

config = Config(
    executors=[
        HighThroughputExecutor(
            label="local_htex",
            provider=LocalProvider(),
            max_workers=4
        ),
        HighThroughputExecutor(
            label="cluster_htex", 
            provider=SlurmProvider(
                nodes_per_block=2,
                cores_per_node=16,
                walltime="02:00:00"
            ),
            max_workers=32
        )
    ]
)
```

#### GPU-Accelerated Execution
```python
from parsl.providers import SlurmProvider

gpu_config = Config(
    executors=[
        HighThroughputExecutor(
            label="gpu_htex",
            provider=SlurmProvider(
                nodes_per_block=1,
                cores_per_node=8,
                walltime="01:00:00",
                scheduler_options="#SBATCH --gres=gpu:1"
            ),
            max_workers=8
        )
    ]
)
```

### Performance Optimization

#### Tuning Worker Count
```yaml
# For CPU-bound tasks
max_workers: <number_of_cpu_cores>

# For I/O-bound tasks (like LLM API calls)
max_workers: <2-4x number_of_cpu_cores>

# For memory-constrained environments
max_workers: <available_memory_gb / memory_per_worker_gb>
```

#### Optimizing Batch Sizes
```python
# Small batches for low latency
batch_size = 1-5

# Large batches for high throughput
batch_size = 10-50

# Adaptive batching based on load
batch_size = min(queue_size, max_batch_size)
```

## Integration with NanoBrain Framework

### Using with Other Components

#### Integration with Steps
```python
from core.step import Step
from core.executor import ParslExecutor

class ParslProcessingStep(Step):
    def __init__(self, config):
        super().__init__(config)
        self.executor = ParslExecutor(config.executor_config)
    
    async def process(self, inputs):
        # Use Parsl for distributed processing
        return await self.executor.execute(self.processing_task, inputs)
```

#### Integration with Workflows
```python
from core.workflow import Workflow

class DistributedWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        self.parsl_executor = ParslExecutor()
        self.setup_distributed_steps()
```

### Testing Integration

The demo integrates with the NanoBrain testing framework:

```bash
# Run Parsl-specific tests
python -m pytest tests/test_parsl_integration.py -v

# Run performance benchmarks
python -m pytest tests/test_performance.py -k "parsl" -v
```

## Contributing

### Adding New Features

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/parsl-enhancement
   ```
3. **Implement changes**
4. **Add tests**
   ```bash
   python -m pytest tests/test_parsl_*.py
   ```
5. **Submit pull request**

### Reporting Issues

Please report issues with:
- System configuration details
- Parsl version and configuration
- Error logs and stack traces
- Steps to reproduce

## License

This demo is part of the NanoBrain framework and follows the same licensing terms.

## Support

For support and questions:
- **Documentation**: Check the main NanoBrain documentation
- **Issues**: Submit GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas

## References

- [Parsl Documentation](https://parsl.readthedocs.io/)
- [NanoBrain Framework Documentation](../README.md)
- [High-Performance Computing Best Practices](https://parsl.readthedocs.io/en/stable/userguide/performance.html) 