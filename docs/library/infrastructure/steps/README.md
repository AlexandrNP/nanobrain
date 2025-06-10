# Parallel Processing Steps

The parallel processing steps module provides advanced step implementations for concurrent processing patterns in the NanoBrain framework. It offers a hierarchy of parallel processing capabilities from generic parallel execution to specialized agent-based processing.

## Overview

This module contains:
- **Generic Parallel Processing**: Base framework for any parallel processing task
- **Agent-Specific Processing**: Specialized parallel processing for NanoBrain agents
- **Conversational Processing**: Chat-optimized parallel processing with context management
- **Load Balancing**: Multiple strategies for distributing work across processors
- **Health Monitoring**: Automatic health checks and recovery mechanisms

## Architecture

```
library/infrastructure/steps/
├── parallel_step.py                    # Base parallel processing framework
├── parallel_agent_step.py              # Agent-specific parallel processing
├── parallel_conversational_agent_step.py # Chat-optimized parallel processing
├── load_balancer.py                    # Load balancing strategies
├── circuit_breaker.py                  # Fault tolerance patterns
├── health_monitor.py                   # Health checking and monitoring
└── performance_tracker.py              # Performance metrics and optimization
```

## Core Components

### ParallelStep

Generic parallel processing framework that can be used with any type of processor.

```python
from library.infrastructure.steps import ParallelStep, ParallelProcessingConfig
from typing import Any, Dict

class ParallelStep(Step, Generic[RequestType, ResponseType]):
    """Generic parallel processing step with configurable processors."""
    
    def __init__(
        self,
        config: ParallelProcessingConfig,
        processors: List[ProcessorType],
        load_balancer: Optional[LoadBalancer] = None
    ):
        super().__init__(config)
        self.processors = processors
        self.load_balancer = load_balancer or RoundRobinLoadBalancer()
```

**Key Features:**
- Generic type support for request/response types
- Pluggable load balancing strategies
- Circuit breaker pattern for fault tolerance
- Performance monitoring and metrics
- Automatic processor health checking

### Basic Usage Example

```python
from library.infrastructure.steps import (
    ParallelStep,
    ParallelProcessingConfig,
    LoadBalancingStrategy
)

# Define a simple processor function
async def text_processor(text: str) -> str:
    # Simulate processing time
    await asyncio.sleep(0.1)
    return f"Processed: {text.upper()}"

# Configure parallel processing
config = ParallelProcessingConfig(
    name="text_processing_step",
    description="Parallel text processing",
    max_parallel_requests=5,
    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=30.0
)

# Create processors (could be functions, classes, or any callable)
processors = [text_processor for _ in range(3)]

# Create parallel step
step = ParallelStep(config, processors)
await step.initialize()

# Process requests
requests = ["hello", "world", "parallel", "processing"]
results = await step.process_batch(requests)

for request, result in zip(requests, results):
    print(f"{request} -> {result}")
```

### ParallelAgentStep

Specialized parallel processing for NanoBrain agents with agent-specific features.

```python
from library.infrastructure.steps import ParallelAgentStep, ParallelAgentConfig
from core.agent import ConversationalAgent, AgentConfig

# Create agents
agents = []
for i in range(3):
    agent_config = AgentConfig(
        name=f"agent_{i}",
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    agent = ConversationalAgent(agent_config)
    agents.append(agent)

# Configure parallel agent processing
config = ParallelAgentConfig(
    name="parallel_agent_step",
    description="Parallel agent processing",
    max_parallel_requests=10,
    load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED,
    enable_health_monitoring=True,
    health_check_interval=60.0,
    enable_performance_tracking=True
)

# Create parallel agent step
step = ParallelAgentStep(config, agents)
await step.initialize()

# Process agent requests
agent_requests = [
    AgentRequest(input_text="What is AI?", metadata={"user_id": "user1"}),
    AgentRequest(input_text="Explain quantum computing", metadata={"user_id": "user2"}),
    AgentRequest(input_text="How does ML work?", metadata={"user_id": "user3"})
]

responses = await step.process_requests(agent_requests)

for request, response in zip(agent_requests, responses):
    print(f"User: {request.input_text}")
    print(f"Agent: {response.response}")
    print(f"Processing time: {response.processing_time_ms}ms\n")
```

**Agent-Specific Features:**
- Agent health monitoring and automatic recovery
- Load balancing based on agent performance
- Request queuing with priority support
- Agent-specific error handling and retry logic
- Performance metrics per agent

### ParallelConversationalAgentStep

Chat-optimized parallel processing with conversation context management.

```python
from library.infrastructure.steps import (
    ParallelConversationalAgentStep,
    ParallelConversationalAgentConfig,
    ConversationRequest
)

# Configure conversational processing
config = ParallelConversationalAgentConfig(
    name="chat_processor",
    description="Parallel conversational processing",
    max_parallel_requests=20,
    load_balancing_strategy=LoadBalancingStrategy.FASTEST_RESPONSE,
    enable_conversation_context=True,
    context_window_size=10,
    enable_response_caching=True,
    cache_ttl=300,  # 5 minutes
    track_token_usage=True,
    enable_streaming_responses=True
)

step = ParallelConversationalAgentStep(config, agents)
await step.initialize()

# Process conversation requests
conversation_requests = [
    ConversationRequest(
        message="Hello, how are you?",
        conversation_id="conv_1",
        user_id="user_123",
        context={"previous_topic": "greetings"}
    ),
    ConversationRequest(
        message="What's the weather like?",
        conversation_id="conv_2",
        user_id="user_456",
        context={"location": "San Francisco"}
    )
]

responses = await step.process_conversations(conversation_requests)

for request, response in zip(conversation_requests, responses):
    print(f"Conversation {request.conversation_id}:")
    print(f"User: {request.message}")
    print(f"Assistant: {response.response}")
    print(f"Tokens used: {response.token_usage}")
    print(f"Response time: {response.response_time_ms}ms\n")
```

**Conversational Features:**
- Conversation context management and persistence
- Token usage tracking and optimization
- Response caching based on conversation context
- Streaming response support for real-time chat
- Conversation-aware load balancing

## Load Balancing Strategies

### Available Strategies

```python
from library.infrastructure.steps import LoadBalancingStrategy

# Round Robin - Distribute requests evenly
LoadBalancingStrategy.ROUND_ROBIN

# Least Loaded - Send to processor with fewest active requests
LoadBalancingStrategy.LEAST_LOADED

# Fastest Response - Send to processor with best average response time
LoadBalancingStrategy.FASTEST_RESPONSE

# Weighted - Distribute based on processor capabilities
LoadBalancingStrategy.WEIGHTED

# Random - Random distribution (useful for testing)
LoadBalancingStrategy.RANDOM
```

### Custom Load Balancer

```python
from library.infrastructure.steps import LoadBalancer, ProcessorInfo

class CustomLoadBalancer(LoadBalancer):
    """Custom load balancer based on processor memory usage."""
    
    async def select_processor(
        self,
        processors: List[ProcessorInfo],
        request: Any
    ) -> ProcessorInfo:
        # Select processor with lowest memory usage
        return min(processors, key=lambda p: p.memory_usage)
    
    async def update_processor_stats(
        self,
        processor: ProcessorInfo,
        response_time: float,
        success: bool
    ) -> None:
        # Update custom metrics
        processor.custom_metrics['avg_response_time'] = (
            processor.custom_metrics.get('avg_response_time', 0) * 0.9 +
            response_time * 0.1
        )

# Use custom load balancer
config = ParallelProcessingConfig(
    name="custom_balanced_step",
    load_balancer_class=CustomLoadBalancer
)
```

## Circuit Breaker Pattern

Automatic fault tolerance and recovery for failed processors.

```python
from library.infrastructure.steps import CircuitBreakerConfig

# Configure circuit breaker
circuit_config = CircuitBreakerConfig(
    failure_threshold=5,        # Open circuit after 5 failures
    recovery_timeout=30.0,      # Try to close after 30 seconds
    half_open_max_calls=3,      # Test with 3 calls in half-open state
    expected_exception_types=[  # Exceptions that trigger circuit breaker
        ConnectionError,
        TimeoutError,
        HTTPError
    ]
)

config = ParallelProcessingConfig(
    name="fault_tolerant_step",
    enable_circuit_breaker=True,
    circuit_breaker_config=circuit_config
)

step = ParallelStep(config, processors)

# Circuit breaker automatically handles failures
try:
    result = await step.process(request)
except CircuitBreakerOpenError:
    # All processors are currently failing
    logger.warning("Circuit breaker is open, using fallback")
    result = await fallback_processor.process(request)
```

**Circuit Breaker States:**
- **Closed**: Normal operation, requests pass through
- **Open**: Failures detected, requests fail fast
- **Half-Open**: Testing if service has recovered

## Health Monitoring

Automatic health checking and processor management.

```python
from library.infrastructure.steps import HealthMonitorConfig

# Configure health monitoring
health_config = HealthMonitorConfig(
    check_interval=60.0,        # Check every minute
    timeout=10.0,               # Health check timeout
    max_failures=3,             # Remove processor after 3 failed checks
    recovery_check_interval=300.0,  # Check removed processors every 5 minutes
    custom_health_checks=[      # Custom health check functions
        check_memory_usage,
        check_response_time,
        check_error_rate
    ]
)

config = ParallelProcessingConfig(
    name="monitored_step",
    enable_health_monitoring=True,
    health_monitor_config=health_config
)

# Health monitor automatically manages processor availability
step = ParallelStep(config, processors)

# Get health status
health_status = await step.get_health_status()
for processor_id, status in health_status.items():
    print(f"Processor {processor_id}: {status.status}")
    print(f"  Last check: {status.last_check}")
    print(f"  Response time: {status.avg_response_time}ms")
    print(f"  Error rate: {status.error_rate:.2%}")
```

### Custom Health Checks

```python
async def check_memory_usage(processor: Any) -> HealthCheckResult:
    """Custom health check for memory usage."""
    try:
        memory_usage = await get_processor_memory_usage(processor)
        if memory_usage > 0.9:  # 90% memory usage
            return HealthCheckResult(
                healthy=False,
                message=f"High memory usage: {memory_usage:.1%}",
                metrics={"memory_usage": memory_usage}
            )
        return HealthCheckResult(
            healthy=True,
            message="Memory usage normal",
            metrics={"memory_usage": memory_usage}
        )
    except Exception as e:
        return HealthCheckResult(
            healthy=False,
            message=f"Health check failed: {e}",
            error=str(e)
        )
```

## Performance Tracking

Comprehensive performance monitoring and optimization.

```python
from library.infrastructure.steps import PerformanceTracker

# Performance tracking is enabled by default
config = ParallelProcessingConfig(
    name="performance_tracked_step",
    enable_performance_tracking=True,
    performance_window_size=1000,  # Track last 1000 requests
    enable_detailed_metrics=True
)

step = ParallelStep(config, processors)

# Get performance statistics
stats = await step.get_performance_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Average response time: {stats.avg_response_time:.2f}ms")
print(f"95th percentile: {stats.p95_response_time:.2f}ms")
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Throughput: {stats.requests_per_second:.1f} req/s")

# Get per-processor statistics
processor_stats = await step.get_processor_stats()
for processor_id, stats in processor_stats.items():
    print(f"\nProcessor {processor_id}:")
    print(f"  Requests handled: {stats.requests_handled}")
    print(f"  Average response time: {stats.avg_response_time:.2f}ms")
    print(f"  Error rate: {stats.error_rate:.2%}")
    print(f"  Current load: {stats.current_load}")
```

### Performance Optimization

```python
# Automatic performance optimization
config = ParallelProcessingConfig(
    name="optimized_step",
    enable_auto_scaling=True,
    min_processors=2,
    max_processors=10,
    scale_up_threshold=0.8,     # Scale up when 80% loaded
    scale_down_threshold=0.3,   # Scale down when 30% loaded
    scale_check_interval=60.0   # Check every minute
)

# The step will automatically add/remove processors based on load
step = ParallelStep(config, initial_processors)

# Manual optimization
await step.optimize_performance()  # Triggers immediate optimization check
```

## Configuration

### ParallelProcessingConfig

Base configuration for all parallel processing steps.

```python
from library.infrastructure.steps import ParallelProcessingConfig, LoadBalancingStrategy

config = ParallelProcessingConfig(
    name="my_parallel_step",
    description="Custom parallel processing step",
    
    # Parallelism settings
    max_parallel_requests=10,
    request_queue_size=100,
    request_timeout=30.0,
    
    # Load balancing
    load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED,
    load_balancer_class=None,  # Use custom load balancer class
    
    # Circuit breaker
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=30.0,
    
    # Health monitoring
    enable_health_monitoring=True,
    health_check_interval=60.0,
    health_check_timeout=10.0,
    
    # Performance tracking
    enable_performance_tracking=True,
    performance_window_size=1000,
    enable_detailed_metrics=True,
    
    # Auto-scaling
    enable_auto_scaling=False,
    min_processors=1,
    max_processors=10,
    scale_up_threshold=0.8,
    scale_down_threshold=0.3
)
```

### ParallelConversationalAgentConfig

Extended configuration for conversational agent processing.

```python
from library.infrastructure.steps import ParallelConversationalAgentConfig

config = ParallelConversationalAgentConfig(
    # Base parallel processing config
    name="chat_processor",
    max_parallel_requests=20,
    load_balancing_strategy=LoadBalancingStrategy.FASTEST_RESPONSE,
    
    # Conversational features
    enable_conversation_context=True,
    context_window_size=10,
    context_storage_backend="redis",  # or "memory", "database"
    
    # Response caching
    enable_response_caching=True,
    cache_backend="redis",
    cache_ttl=300,
    cache_key_strategy="conversation_aware",
    
    # Token tracking
    track_token_usage=True,
    token_usage_backend="database",
    enable_token_optimization=True,
    
    # Streaming
    enable_streaming_responses=True,
    streaming_chunk_size=1024,
    streaming_timeout=5.0
)
```

## Advanced Usage Examples

### Multi-Stage Processing Pipeline

```python
from library.infrastructure.steps import ParallelStep, ParallelProcessingConfig

# Stage 1: Text preprocessing
preprocess_config = ParallelProcessingConfig(
    name="preprocessing_stage",
    max_parallel_requests=20,
    load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN
)
preprocess_step = ParallelStep(preprocess_config, text_preprocessors)

# Stage 2: AI processing
ai_config = ParallelProcessingConfig(
    name="ai_processing_stage",
    max_parallel_requests=5,
    load_balancing_strategy=LoadBalancingStrategy.LEAST_LOADED,
    enable_circuit_breaker=True
)
ai_step = ParallelAgentStep(ai_config, ai_agents)

# Stage 3: Post-processing
postprocess_config = ParallelProcessingConfig(
    name="postprocessing_stage",
    max_parallel_requests=15,
    load_balancing_strategy=LoadBalancingStrategy.FASTEST_RESPONSE
)
postprocess_step = ParallelStep(postprocess_config, postprocessors)

# Pipeline execution
async def process_pipeline(inputs):
    # Stage 1: Preprocess
    preprocessed = await preprocess_step.process_batch(inputs)
    
    # Stage 2: AI processing
    ai_results = await ai_step.process_batch(preprocessed)
    
    # Stage 3: Post-process
    final_results = await postprocess_step.process_batch(ai_results)
    
    return final_results
```

### Dynamic Processor Management

```python
from library.infrastructure.steps import ProcessorManager

class DynamicParallelStep(ParallelStep):
    """Parallel step with dynamic processor management."""
    
    def __init__(self, config, initial_processors):
        super().__init__(config, initial_processors)
        self.processor_manager = ProcessorManager()
    
    async def add_processor(self, processor):
        """Add a new processor at runtime."""
        await self.processor_manager.add_processor(processor)
        await self.load_balancer.register_processor(processor)
        logger.info(f"Added processor: {processor.id}")
    
    async def remove_processor(self, processor_id):
        """Remove a processor at runtime."""
        processor = await self.processor_manager.get_processor(processor_id)
        await self.load_balancer.unregister_processor(processor)
        await self.processor_manager.remove_processor(processor_id)
        logger.info(f"Removed processor: {processor_id}")
    
    async def scale_processors(self, target_count):
        """Scale processors to target count."""
        current_count = len(self.processors)
        
        if target_count > current_count:
            # Scale up
            for _ in range(target_count - current_count):
                new_processor = await self.create_processor()
                await self.add_processor(new_processor)
        elif target_count < current_count:
            # Scale down
            processors_to_remove = current_count - target_count
            for _ in range(processors_to_remove):
                # Remove least loaded processor
                processor_id = await self.find_least_loaded_processor()
                await self.remove_processor(processor_id)

# Usage
step = DynamicParallelStep(config, initial_processors)

# Scale based on load
current_load = await step.get_current_load()
if current_load > 0.8:
    await step.scale_processors(len(step.processors) + 2)
elif current_load < 0.3:
    await step.scale_processors(max(1, len(step.processors) - 1))
```

### Custom Request Prioritization

```python
from library.infrastructure.steps import PriorityQueue, RequestPriority

class PriorityParallelStep(ParallelStep):
    """Parallel step with request prioritization."""
    
    def __init__(self, config, processors):
        super().__init__(config, processors)
        self.request_queue = PriorityQueue()
    
    async def process_with_priority(self, request, priority: RequestPriority):
        """Process request with specified priority."""
        prioritized_request = PrioritizedRequest(
            request=request,
            priority=priority,
            timestamp=time.time()
        )
        
        return await self.request_queue.enqueue(prioritized_request)
    
    async def process_vip_request(self, request):
        """Process VIP request with highest priority."""
        return await self.process_with_priority(request, RequestPriority.VIP)
    
    async def process_batch_with_priorities(self, requests_with_priorities):
        """Process batch of requests with different priorities."""
        tasks = []
        for request, priority in requests_with_priorities:
            task = self.process_with_priority(request, priority)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

# Usage
step = PriorityParallelStep(config, processors)

# Process requests with different priorities
requests = [
    ("urgent request", RequestPriority.HIGH),
    ("normal request", RequestPriority.NORMAL),
    ("background task", RequestPriority.LOW),
    ("vip request", RequestPriority.VIP)
]

results = await step.process_batch_with_priorities(requests)
```

## Best Practices

### Resource Management

```python
# Always initialize and shutdown steps properly
async def use_parallel_step():
    step = ParallelStep(config, processors)
    try:
        await step.initialize()
        
        # Use the step
        results = await step.process_batch(requests)
        return results
        
    finally:
        await step.shutdown()

# Use context managers when available
async with ParallelStep(config, processors) as step:
    results = await step.process_batch(requests)
```

### Error Handling

```python
from library.infrastructure.steps import (
    ProcessorError,
    LoadBalancerError,
    CircuitBreakerOpenError
)

try:
    result = await step.process(request)
except CircuitBreakerOpenError:
    # All processors are failing, use fallback
    result = await fallback_processor.process(request)
except ProcessorError as e:
    # Specific processor failed, retry with different processor
    logger.warning(f"Processor failed: {e}")
    result = await step.retry_with_different_processor(request)
except Exception as e:
    # Unexpected error
    logger.error(f"Unexpected error in parallel processing: {e}")
    raise
```

### Performance Optimization

```python
# Configure appropriate parallelism levels
config = ParallelProcessingConfig(
    max_parallel_requests=min(
        cpu_count() * 2,  # Don't exceed CPU capacity
        100  # Reasonable upper limit
    ),
    request_queue_size=max_parallel_requests * 5,  # Buffer for bursts
    request_timeout=30.0  # Prevent hanging requests
)

# Monitor and adjust based on metrics
async def optimize_step_performance(step):
    stats = await step.get_performance_stats()
    
    if stats.avg_response_time > 5000:  # 5 seconds
        # Response time too high, scale up
        await step.scale_processors(len(step.processors) + 1)
    elif stats.cpu_usage < 0.3:  # 30% CPU usage
        # Underutilized, scale down
        await step.scale_processors(max(1, len(step.processors) - 1))
```

### Testing

```python
import pytest
from library.infrastructure.steps import ParallelStep, ParallelProcessingConfig

@pytest.fixture
async def mock_processors():
    """Create mock processors for testing."""
    async def mock_processor(request):
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"processed: {request}"
    
    return [mock_processor for _ in range(3)]

@pytest.fixture
async def parallel_step(mock_processors):
    """Create parallel step for testing."""
    config = ParallelProcessingConfig(
        name="test_step",
        max_parallel_requests=5
    )
    step = ParallelStep(config, mock_processors)
    await step.initialize()
    yield step
    await step.shutdown()

async def test_parallel_processing(parallel_step):
    """Test basic parallel processing."""
    requests = ["req1", "req2", "req3", "req4", "req5"]
    results = await parallel_step.process_batch(requests)
    
    assert len(results) == len(requests)
    for request, result in zip(requests, results):
        assert result == f"processed: {request}"

async def test_load_balancing(parallel_step):
    """Test load balancing across processors."""
    # Process many requests to test distribution
    requests = [f"req_{i}" for i in range(100)]
    results = await parallel_step.process_batch(requests)
    
    # Check that all processors were used
    processor_stats = await parallel_step.get_processor_stats()
    for processor_id, stats in processor_stats.items():
        assert stats.requests_handled > 0
```

## Migration Guide

### From Sequential Processing

```python
# Before (sequential processing)
results = []
for request in requests:
    result = await processor.process(request)
    results.append(result)

# After (parallel processing)
from library.infrastructure.steps import ParallelStep

step = ParallelStep(config, processors)
await step.initialize()
results = await step.process_batch(requests)
await step.shutdown()
```

### From Demo Implementations

```python
# Before (demo-specific parallel processing)
class ParallelConversationalAgentStep:
    def __init__(self, agents):
        self.agents = agents
        # 300+ lines of parallel processing logic

# After (library parallel processing)
from library.infrastructure.steps import ParallelConversationalAgentStep

step = ParallelConversationalAgentStep(config, agents)
```

## Performance Considerations

### Parallelism Tuning

- **CPU-bound tasks**: Set `max_parallel_requests` to `cpu_count()`
- **I/O-bound tasks**: Set `max_parallel_requests` to `cpu_count() * 2-4`
- **Memory-intensive tasks**: Consider memory usage per request

### Load Balancing Strategy Selection

- **Round Robin**: Best for uniform workloads and simple setup
- **Least Loaded**: Best for varying request complexity
- **Fastest Response**: Best for performance optimization
- **Weighted**: Best for heterogeneous processor capabilities

### Memory Management

- Monitor memory usage per processor
- Configure appropriate request queue sizes
- Use streaming for large data processing
- Implement proper cleanup in processors

---

*For more information, see the [main library documentation](../../README.md) or explore the [parallel processing examples](../../../examples/library/steps/).* 