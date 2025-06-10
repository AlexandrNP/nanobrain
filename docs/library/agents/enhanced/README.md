# Enhanced Agents

The enhanced agents module provides advanced agent implementations with protocol support, collaboration capabilities, and specialized behaviors for the NanoBrain framework. These agents extend the core agent functionality with production-ready features for complex multi-agent scenarios.

## Overview

This module contains:
- **Protocol Support**: A2A (Agent-to-Agent) and MCP (Model Context Protocol) integration
- **Collaborative Agents**: Multi-protocol agents with delegation and coordination
- **Delegation Engine**: Rule-based task delegation and routing
- **Performance Tracking**: Advanced metrics and optimization
- **Fault Tolerance**: Circuit breakers, retry logic, and fallback mechanisms

## Architecture

```
library/agents/enhanced/
├── collaborative_agent.py      # Multi-protocol collaborative agent
├── protocol_mixin.py          # Protocol support mixins
├── delegation_engine.py       # Task delegation and routing
├── performance_tracker.py     # Agent performance monitoring
├── conversation_manager.py    # Advanced conversation management
└── agent_registry.py          # Agent discovery and registration
```

## Core Components

### CollaborativeAgent

The main enhanced agent that combines multiple protocols and collaboration features.

```python
from library.agents.enhanced import CollaborativeAgent
from core.agent import AgentConfig

# Create enhanced agent configuration
config = AgentConfig(
    name="collaborative_assistant",
    model="gpt-4",
    temperature=0.7,
    system_prompt="You are a collaborative AI assistant with access to specialized agents and tools."
)

# Create collaborative agent with protocol support
agent = CollaborativeAgent(
    config,
    a2a_config_path="config/a2a_config.yaml",
    mcp_config_path="config/mcp_config.yaml",
    delegation_rules=[
        {
            'keywords': ['calculate', 'math', 'compute'],
            'agent': 'calculator_agent',
            'description': 'Delegate mathematical calculations'
        },
        {
            'keywords': ['weather', 'forecast', 'temperature'],
            'agent': 'weather_agent',
            'description': 'Delegate weather-related queries'
        },
        {
            'keywords': ['code', 'programming', 'debug'],
            'agent': 'coding_agent',
            'description': 'Delegate programming tasks'
        }
    ]
)

await agent.initialize()

# Agent automatically delegates or uses tools as appropriate
response = await agent.process("What's the weather like in San Francisco?")
# This will be automatically delegated to the weather_agent

response = await agent.process("Calculate the square root of 144")
# This will be automatically delegated to the calculator_agent

response = await agent.process("How are you today?")
# This will be handled by the main agent

await agent.shutdown()
```

**Key Features:**
- Automatic task delegation based on configurable rules
- Multi-protocol support (A2A, MCP, custom protocols)
- Performance tracking and optimization
- Conversation context management
- Fault tolerance with fallback mechanisms

### Protocol Support Mixins

Modular protocol support that can be mixed into any agent.

```python
from library.agents.enhanced import A2AProtocolMixin, MCPProtocolMixin
from core.agent import ConversationalAgent

class EnhancedAgent(A2AProtocolMixin, MCPProtocolMixin, ConversationalAgent):
    """Agent with both A2A and MCP protocol support."""
    
    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.a2a_config_path = kwargs.get('a2a_config_path')
        self.mcp_config_path = kwargs.get('mcp_config_path')
    
    async def initialize(self):
        await super().initialize()
        
        # Initialize protocol support
        if self.a2a_config_path:
            await self.initialize_a2a(self.a2a_config_path)
        
        if self.mcp_config_path:
            await self.initialize_mcp(self.mcp_config_path)
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Custom processing logic with protocol support
        if self.should_delegate_a2a(input_text):
            agent_name = self.select_a2a_agent(input_text)
            return await self.call_a2a_agent(agent_name, input_text)
        
        elif self.should_use_mcp_tool(input_text):
            tool_name = self.select_mcp_tool(input_text)
            tool_args = self.extract_tool_arguments(input_text)
            return await self.call_mcp_tool(tool_name, tool_args)
        
        else:
            return await super().process(input_text, **kwargs)

# Usage
agent = EnhancedAgent(
    config,
    a2a_config_path="config/a2a.yaml",
    mcp_config_path="config/mcp.yaml"
)
```

### A2AProtocolMixin

Agent-to-Agent protocol support for inter-agent communication.

```python
from library.agents.enhanced import A2AProtocolMixin

class A2AAgent(A2AProtocolMixin, ConversationalAgent):
    async def initialize(self):
        await super().initialize()
        await self.initialize_a2a("config/a2a_config.yaml")
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Check if we should delegate to another agent
        if "translate" in input_text.lower():
            return await self.call_a2a_agent("translator_agent", input_text)
        
        elif "summarize" in input_text.lower():
            return await self.call_a2a_agent("summarizer_agent", input_text)
        
        else:
            return await super().process(input_text, **kwargs)

# A2A Configuration (a2a_config.yaml)
"""
agents:
  translator_agent:
    endpoint: "http://translator-service:8080/translate"
    timeout: 30.0
    retry_attempts: 3
    
  summarizer_agent:
    endpoint: "http://summarizer-service:8080/summarize"
    timeout: 45.0
    retry_attempts: 2
    
connection:
  pool_size: 10
  keep_alive: true
  ssl_verify: true
"""
```

**A2A Features:**
- Service discovery and registration
- Load balancing across agent instances
- Circuit breaker pattern for fault tolerance
- Request/response caching
- Performance monitoring and metrics

### MCPProtocolMixin

Model Context Protocol support for tool integration.

```python
from library.agents.enhanced import MCPProtocolMixin

class MCPAgent(MCPProtocolMixin, ConversationalAgent):
    async def initialize(self):
        await super().initialize()
        await self.initialize_mcp("config/mcp_config.yaml")
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Check if we should use MCP tools
        if "calculate" in input_text.lower():
            # Extract calculation expression
            expression = self.extract_calculation(input_text)
            result = await self.call_mcp_tool("calculator", {"expression": expression})
            return f"The result is: {result}"
        
        elif "search" in input_text.lower():
            # Extract search query
            query = self.extract_search_query(input_text)
            results = await self.call_mcp_tool("web_search", {"query": query})
            return self.format_search_results(results)
        
        else:
            return await super().process(input_text, **kwargs)

# MCP Configuration (mcp_config.yaml)
"""
tools:
  calculator:
    server: "calculator-mcp-server"
    timeout: 10.0
    
  web_search:
    server: "search-mcp-server"
    timeout: 30.0
    
  file_operations:
    server: "file-mcp-server"
    timeout: 60.0
    
servers:
  calculator-mcp-server:
    command: ["python", "-m", "calculator_mcp_server"]
    env:
      CALCULATOR_PRECISION: "10"
      
  search-mcp-server:
    command: ["node", "search-mcp-server.js"]
    env:
      SEARCH_API_KEY: "${SEARCH_API_KEY}"
"""
```

**MCP Features:**
- Automatic tool discovery and registration
- Tool capability negotiation
- Streaming tool responses
- Tool result caching
- Error handling and retry logic

## Delegation Engine

Advanced task delegation and routing system.

```python
from library.agents.enhanced import DelegationEngine, DelegationRule

# Create delegation rules
rules = [
    DelegationRule(
        name="math_delegation",
        keywords=["calculate", "math", "compute", "equation"],
        agent="calculator_agent",
        confidence_threshold=0.8,
        description="Delegate mathematical calculations"
    ),
    DelegationRule(
        name="weather_delegation",
        keywords=["weather", "forecast", "temperature", "rain"],
        agent="weather_agent",
        confidence_threshold=0.7,
        description="Delegate weather queries"
    ),
    DelegationRule(
        name="code_delegation",
        keywords=["code", "programming", "debug", "function"],
        agent="coding_agent",
        confidence_threshold=0.9,
        description="Delegate programming tasks"
    )
]

# Create delegation engine
delegation_engine = DelegationEngine(rules)

# Use in agent
class SmartDelegatingAgent(CollaborativeAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.delegation_engine = DelegationEngine(kwargs.get('delegation_rules', []))
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Check for delegation
        delegation_decision = await self.delegation_engine.should_delegate(input_text)
        
        if delegation_decision.should_delegate:
            logger.info(f"Delegating to {delegation_decision.target_agent}: {delegation_decision.reason}")
            
            try:
                result = await self.delegate_to_agent(
                    delegation_decision.target_agent,
                    input_text,
                    context=delegation_decision.context
                )
                
                # Track delegation performance
                await self.delegation_engine.record_delegation_result(
                    delegation_decision,
                    success=True,
                    response_time=result.response_time
                )
                
                return result.response
                
            except Exception as e:
                logger.error(f"Delegation failed: {e}")
                
                # Record failure and try fallback
                await self.delegation_engine.record_delegation_result(
                    delegation_decision,
                    success=False,
                    error=str(e)
                )
                
                # Use fallback or handle locally
                return await self.handle_delegation_failure(input_text, e)
        
        else:
            # Handle locally
            return await super().process(input_text, **kwargs)
```

### Advanced Delegation Rules

```python
from library.agents.enhanced import (
    DelegationRule,
    ContextualDelegationRule,
    PerformanceDelegationRule
)

# Contextual delegation based on conversation history
contextual_rule = ContextualDelegationRule(
    name="contextual_math",
    keywords=["calculate", "math"],
    agent="calculator_agent",
    context_conditions=[
        {"previous_topic": "mathematics"},
        {"user_expertise": "beginner"}
    ],
    confidence_threshold=0.6
)

# Performance-based delegation
performance_rule = PerformanceDelegationRule(
    name="performance_based",
    keywords=["complex", "analysis"],
    agents=["expert_agent", "general_agent"],
    selection_strategy="fastest_response",  # or "least_loaded", "round_robin"
    fallback_agent="general_agent"
)

# Time-based delegation
time_rule = DelegationRule(
    name="urgent_tasks",
    keywords=["urgent", "asap", "quickly"],
    agent="fast_agent",
    conditions=[
        {"time_constraint": "< 5 seconds"}
    ]
)

# Combine rules with priorities
delegation_engine = DelegationEngine([
    (contextual_rule, priority=1),      # Highest priority
    (performance_rule, priority=2),
    (time_rule, priority=3)
])
```

## Performance Tracking

Comprehensive performance monitoring for enhanced agents.

```python
from library.agents.enhanced import PerformanceTracker, PerformanceConfig

# Configure performance tracking
perf_config = PerformanceConfig(
    track_response_times=True,
    track_token_usage=True,
    track_delegation_patterns=True,
    track_error_rates=True,
    window_size=1000,  # Track last 1000 requests
    enable_real_time_metrics=True
)

class PerformanceTrackedAgent(CollaborativeAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.performance_tracker = PerformanceTracker(perf_config)
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Start performance tracking
        with self.performance_tracker.track_request() as tracker:
            try:
                result = await super().process(input_text, **kwargs)
                
                # Record successful processing
                tracker.record_success(
                    response_length=len(result),
                    token_usage=self.get_last_token_usage(),
                    delegation_used=self.was_delegated()
                )
                
                return result
                
            except Exception as e:
                # Record error
                tracker.record_error(error_type=type(e).__name__, error_message=str(e))
                raise
    
    async def get_performance_stats(self):
        """Get comprehensive performance statistics."""
        return await self.performance_tracker.get_stats()

# Usage
agent = PerformanceTrackedAgent(config)

# Get performance statistics
stats = await agent.get_performance_stats()
print(f"Average response time: {stats.avg_response_time:.2f}ms")
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Delegation rate: {stats.delegation_rate:.2%}")
print(f"Token efficiency: {stats.tokens_per_second:.1f} tokens/s")
```

### Performance Optimization

```python
from library.agents.enhanced import PerformanceOptimizer

class OptimizedAgent(CollaborativeAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.optimizer = PerformanceOptimizer(self)
    
    async def initialize(self):
        await super().initialize()
        
        # Start performance optimization
        await self.optimizer.start_optimization()
    
    async def optimize_performance(self):
        """Trigger performance optimization."""
        recommendations = await self.optimizer.analyze_performance()
        
        for recommendation in recommendations:
            if recommendation.type == "delegation_threshold":
                # Adjust delegation thresholds
                await self.update_delegation_thresholds(recommendation.values)
            
            elif recommendation.type == "caching_strategy":
                # Update caching configuration
                await self.update_caching_strategy(recommendation.strategy)
            
            elif recommendation.type == "model_selection":
                # Switch to more appropriate model
                await self.update_model_config(recommendation.model_config)

# Automatic optimization
agent = OptimizedAgent(config)
await agent.initialize()

# Optimization runs automatically in background
# Manual optimization can be triggered
await agent.optimize_performance()
```

## Conversation Management

Advanced conversation context management and persistence.

```python
from library.agents.enhanced import ConversationManager, ConversationConfig

# Configure conversation management
conv_config = ConversationConfig(
    enable_context_persistence=True,
    context_window_size=20,
    context_storage_backend="redis",
    enable_conversation_summarization=True,
    summarization_threshold=50,  # Summarize after 50 messages
    enable_topic_tracking=True
)

class ContextAwareAgent(CollaborativeAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.conversation_manager = ConversationManager(conv_config)
    
    async def process(self, input_text: str, **kwargs) -> str:
        conversation_id = kwargs.get('conversation_id', 'default')
        user_id = kwargs.get('user_id')
        
        # Load conversation context
        context = await self.conversation_manager.get_context(conversation_id)
        
        # Add current input to context
        await self.conversation_manager.add_message(
            conversation_id,
            role="user",
            content=input_text,
            user_id=user_id
        )
        
        # Process with context
        enhanced_input = self.conversation_manager.enhance_input_with_context(
            input_text,
            context
        )
        
        result = await super().process(enhanced_input, **kwargs)
        
        # Add response to context
        await self.conversation_manager.add_message(
            conversation_id,
            role="assistant",
            content=result,
            metadata={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "delegation_used": self.was_delegated()
            }
        )
        
        return result
    
    async def get_conversation_summary(self, conversation_id: str):
        """Get conversation summary."""
        return await self.conversation_manager.get_conversation_summary(conversation_id)

# Usage with conversation context
agent = ContextAwareAgent(config)

# First message
response1 = await agent.process(
    "My name is Alice and I'm a software engineer",
    conversation_id="conv_123",
    user_id="alice"
)

# Later message - agent remembers context
response2 = await agent.process(
    "What programming languages should I learn?",
    conversation_id="conv_123",
    user_id="alice"
)
# Agent will remember Alice is a software engineer and provide relevant advice
```

## Agent Registry

Service discovery and registration for distributed agent systems.

```python
from library.agents.enhanced import AgentRegistry, AgentRegistration

# Create agent registry
registry = AgentRegistry(
    backend="redis",  # or "etcd", "consul"
    connection_string="redis://localhost:6379/0"
)

# Register agent
registration = AgentRegistration(
    agent_id="calculator_agent_001",
    agent_type="calculator",
    capabilities=["arithmetic", "algebra", "calculus"],
    endpoint="http://calculator-service:8080",
    health_check_url="http://calculator-service:8080/health",
    metadata={
        "version": "1.2.0",
        "max_concurrent_requests": 10,
        "average_response_time": 150  # milliseconds
    }
)

await registry.register_agent(registration)

# Discover agents
calculator_agents = await registry.discover_agents(
    agent_type="calculator",
    capabilities=["algebra"]
)

# Select best agent based on criteria
best_agent = await registry.select_agent(
    agent_type="calculator",
    selection_criteria={
        "lowest_latency": True,
        "highest_availability": True
    }
)

# Use in collaborative agent
class RegistryAwareAgent(CollaborativeAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.registry = AgentRegistry(kwargs.get('registry_config'))
    
    async def delegate_to_agent(self, agent_type: str, input_text: str, **kwargs):
        # Discover available agents
        agents = await self.registry.discover_agents(agent_type=agent_type)
        
        if not agents:
            raise ValueError(f"No agents available for type: {agent_type}")
        
        # Select best agent
        selected_agent = await self.registry.select_agent(
            agent_type=agent_type,
            selection_criteria={"lowest_latency": True}
        )
        
        # Delegate to selected agent
        return await self.call_agent(selected_agent.endpoint, input_text, **kwargs)
```

## Configuration

### CollaborativeAgent Configuration

```python
from library.agents.enhanced import CollaborativeAgentConfig

config = CollaborativeAgentConfig(
    # Base agent configuration
    name="collaborative_assistant",
    model="gpt-4",
    temperature=0.7,
    
    # Protocol configuration
    enable_a2a=True,
    a2a_config_path="config/a2a.yaml",
    enable_mcp=True,
    mcp_config_path="config/mcp.yaml",
    
    # Delegation configuration
    delegation_rules=[
        {
            'keywords': ['calculate', 'math'],
            'agent': 'calculator_agent',
            'confidence_threshold': 0.8
        }
    ],
    delegation_timeout=30.0,
    enable_delegation_fallback=True,
    
    # Performance configuration
    enable_performance_tracking=True,
    performance_window_size=1000,
    enable_auto_optimization=True,
    
    # Conversation configuration
    enable_conversation_context=True,
    context_window_size=20,
    context_storage_backend="redis",
    
    # Fault tolerance
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=60.0,
    max_retry_attempts=3,
    retry_backoff_factor=2.0
)
```

## Advanced Usage Examples

### Multi-Agent Collaboration

```python
from library.agents.enhanced import CollaborativeAgent, AgentOrchestrator

# Create specialized agents
research_agent = CollaborativeAgent(
    AgentConfig(name="researcher", model="gpt-4", system_prompt="You are a research specialist."),
    delegation_rules=[
        {'keywords': ['search', 'find', 'lookup'], 'agent': 'search_agent'}
    ]
)

writing_agent = CollaborativeAgent(
    AgentConfig(name="writer", model="gpt-4", system_prompt="You are a writing specialist."),
    delegation_rules=[
        {'keywords': ['grammar', 'style'], 'agent': 'grammar_agent'}
    ]
)

# Create orchestrator
orchestrator = AgentOrchestrator([research_agent, writing_agent])

# Complex task requiring multiple agents
async def write_research_article(topic: str):
    # Step 1: Research
    research_results = await research_agent.process(f"Research information about {topic}")
    
    # Step 2: Write article
    article = await writing_agent.process(
        f"Write an article about {topic} based on this research: {research_results}"
    )
    
    # Step 3: Review and refine
    final_article = await orchestrator.collaborative_process(
        f"Review and improve this article: {article}",
        agents=[research_agent, writing_agent]
    )
    
    return final_article

# Usage
article = await write_research_article("Artificial Intelligence in Healthcare")
```

### Dynamic Agent Scaling

```python
from library.agents.enhanced import AgentPool, ScalingPolicy

# Create agent pool with auto-scaling
scaling_policy = ScalingPolicy(
    min_agents=2,
    max_agents=10,
    scale_up_threshold=0.8,  # Scale up when 80% loaded
    scale_down_threshold=0.3,  # Scale down when 30% loaded
    scale_check_interval=60.0  # Check every minute
)

agent_pool = AgentPool(
    agent_factory=lambda: CollaborativeAgent(config),
    scaling_policy=scaling_policy
)

await agent_pool.initialize()

# Process requests with automatic scaling
async def process_batch_requests(requests):
    # Pool automatically scales based on load
    results = []
    for request in requests:
        agent = await agent_pool.get_agent()
        try:
            result = await agent.process(request)
            results.append(result)
        finally:
            await agent_pool.return_agent(agent)
    
    return results
```

### Custom Protocol Implementation

```python
from library.agents.enhanced import ProtocolMixin

class CustomProtocolMixin(ProtocolMixin):
    """Custom protocol implementation."""
    
    async def initialize_custom_protocol(self, config_path: str):
        """Initialize custom protocol."""
        self.custom_config = await self.load_protocol_config(config_path)
        self.custom_client = CustomProtocolClient(self.custom_config)
        await self.custom_client.connect()
    
    async def call_custom_service(self, service_name: str, data: Dict[str, Any]) -> Any:
        """Call custom protocol service."""
        try:
            response = await self.custom_client.call_service(service_name, data)
            return response
        except Exception as e:
            logger.error(f"Custom protocol call failed: {e}")
            raise

class CustomAgent(CustomProtocolMixin, CollaborativeAgent):
    async def initialize(self):
        await super().initialize()
        await self.initialize_custom_protocol("config/custom_protocol.yaml")
    
    async def process(self, input_text: str, **kwargs) -> str:
        if "custom_service" in input_text.lower():
            result = await self.call_custom_service("text_processor", {"text": input_text})
            return result
        
        return await super().process(input_text, **kwargs)
```

## Best Practices

### Error Handling and Resilience

```python
from library.agents.enhanced import CircuitBreaker, RetryPolicy

class ResilientAgent(CollaborativeAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # Configure circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        
        # Configure retry policy
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            backoff_factor=2.0,
            max_backoff=30.0
        )
    
    async def process(self, input_text: str, **kwargs) -> str:
        # Use circuit breaker pattern
        if self.circuit_breaker.is_open():
            return await self.fallback_process(input_text, **kwargs)
        
        try:
            # Retry with exponential backoff
            return await self.retry_policy.execute(
                lambda: super().process(input_text, **kwargs)
            )
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
    
    async def fallback_process(self, input_text: str, **kwargs) -> str:
        """Fallback processing when main processing fails."""
        return "I'm experiencing technical difficulties. Please try again later."
```

### Performance Monitoring

```python
import asyncio
from library.agents.enhanced import PerformanceMonitor

async def monitor_agent_performance(agent):
    """Monitor agent performance and alert on issues."""
    monitor = PerformanceMonitor(agent)
    
    while True:
        stats = await monitor.get_current_stats()
        
        # Check for performance issues
        if stats.avg_response_time > 5000:  # 5 seconds
            logger.warning(f"High response time: {stats.avg_response_time}ms")
            await monitor.trigger_optimization()
        
        if stats.error_rate > 0.1:  # 10% error rate
            logger.error(f"High error rate: {stats.error_rate:.2%}")
            await monitor.trigger_health_check()
        
        if stats.delegation_failure_rate > 0.2:  # 20% delegation failures
            logger.warning(f"High delegation failure rate: {stats.delegation_failure_rate:.2%}")
            await monitor.update_delegation_rules()
        
        await asyncio.sleep(60)  # Check every minute

# Start monitoring
asyncio.create_task(monitor_agent_performance(agent))
```

### Testing Enhanced Agents

```python
import pytest
from library.agents.enhanced import CollaborativeAgent, MockA2AClient, MockMCPClient

@pytest.fixture
async def mock_collaborative_agent():
    """Create mock collaborative agent for testing."""
    config = AgentConfig(name="test_agent", model="gpt-3.5-turbo")
    
    agent = CollaborativeAgent(config)
    
    # Mock protocol clients
    agent.a2a_client = MockA2AClient()
    agent.mcp_client = MockMCPClient()
    
    await agent.initialize()
    yield agent
    await agent.shutdown()

async def test_delegation(mock_collaborative_agent):
    """Test task delegation."""
    # Configure mock responses
    mock_collaborative_agent.a2a_client.set_response(
        "calculator_agent",
        "The result is 12"
    )
    
    # Test delegation
    response = await mock_collaborative_agent.process("Calculate 3 + 9")
    assert "12" in response

async def test_fallback_handling(mock_collaborative_agent):
    """Test fallback when delegation fails."""
    # Configure mock to fail
    mock_collaborative_agent.a2a_client.set_failure("calculator_agent")
    
    # Should fallback to main agent
    response = await mock_collaborative_agent.process("Calculate 3 + 9")
    assert response is not None  # Should not raise exception
```

## Migration Guide

### From Basic Agents

```python
# Before (basic agent)
from core.agent import ConversationalAgent

agent = ConversationalAgent(config)
response = await agent.process("Hello")

# After (enhanced agent)
from library.agents.enhanced import CollaborativeAgent

agent = CollaborativeAgent(
    config,
    delegation_rules=[...],
    a2a_config_path="config/a2a.yaml"
)
response = await agent.process("Hello")
```

### From Demo Implementations

```python
# Before (demo-specific enhanced agent)
class EnhancedChatAgent:
    def __init__(self, config):
        # 500+ lines of enhancement logic
        pass

# After (library enhanced agent)
from library.agents.enhanced import CollaborativeAgent

agent = CollaborativeAgent(config, **enhancement_options)
```

## Performance Considerations

### Memory Management

- Monitor conversation context size and implement summarization
- Use connection pooling for protocol clients
- Implement proper cleanup for long-running agents

### Network Optimization

- Configure appropriate timeouts for delegation calls
- Use connection pooling and keep-alive for HTTP clients
- Implement caching for frequently delegated requests

### Scalability

- Use agent pools for high-throughput scenarios
- Implement proper load balancing across agent instances
- Monitor and optimize delegation patterns

---

*For more information, see the [main library documentation](../../README.md) or explore the [enhanced agent examples](../../../examples/library/agents/enhanced/).* 