# Chat Workflow

The chat workflow module provides a complete, production-ready chat workflow orchestration system for the NanoBrain framework. It combines all the infrastructure components into a cohesive workflow that can handle complex conversational scenarios with multiple agents, parallel processing, and advanced features.

## Overview

This module contains:
- **Workflow Orchestration**: Complete chat workflow management and coordination
- **Request Processing**: Multi-stage request processing pipeline
- **Response Aggregation**: Intelligent response collection and formatting
- **Session Management**: User session lifecycle and context management
- **Real-time Features**: Streaming responses and live updates

## Architecture

```
library/workflows/chat_workflow/
├── orchestrator.py             # Main workflow orchestrator
├── request_processor/          # Request processing pipeline
│   ├── input_validator.py      # Input validation and sanitization
│   ├── context_enricher.py     # Context enhancement and preparation
│   ├── request_router.py       # Request routing and delegation
│   └── priority_manager.py     # Request prioritization
├── response_aggregator/        # Response processing and formatting
│   ├── response_collector.py   # Response collection and merging
│   ├── response_formatter.py   # Response formatting and presentation
│   └── streaming_handler.py    # Real-time streaming responses
├── session_manager/            # Session and context management
│   ├── session_store.py        # Session persistence and retrieval
│   ├── context_manager.py      # Conversation context management
│   └── user_preferences.py     # User preference management
└── monitoring/                 # Workflow monitoring and metrics
    ├── workflow_monitor.py     # Workflow performance monitoring
    ├── health_checker.py       # System health monitoring
    └── metrics_collector.py    # Comprehensive metrics collection
```

## Core Components

### ChatWorkflowOrchestrator

The main orchestrator that coordinates the entire chat workflow.

```python
from library.workflows.chat_workflow import ChatWorkflowOrchestrator, ChatWorkflowConfig
from library.agents.enhanced import CollaborativeAgent
from library.infrastructure.data import ConversationHistoryUnit
from library.infrastructure.interfaces.database import SQLiteAdapter

# Configure the chat workflow
config = ChatWorkflowConfig(
    name="production_chat_workflow",
    description="Production-ready chat workflow with all features",
    
    # Agent configuration
    agents=[
        {
            'name': 'primary_agent',
            'model': 'gpt-4',
            'temperature': 0.7,
            'system_prompt': 'You are a helpful AI assistant.'
        },
        {
            'name': 'specialist_agent',
            'model': 'gpt-4',
            'temperature': 0.5,
            'system_prompt': 'You are a technical specialist.'
        }
    ],
    
    # Parallel processing
    enable_parallel_processing=True,
    max_parallel_requests=10,
    load_balancing_strategy="fastest_response",
    
    # Database configuration
    database_config={
        'adapter': 'sqlite',
        'connection_string': 'chat_workflow.db'
    },
    
    # Session management
    session_timeout=3600,  # 1 hour
    enable_conversation_context=True,
    context_window_size=20,
    
    # Real-time features
    enable_streaming=True,
    streaming_chunk_size=1024,
    
    # Monitoring
    enable_monitoring=True,
    metrics_collection_interval=60.0
)

# Create and initialize orchestrator
orchestrator = ChatWorkflowOrchestrator(config)
await orchestrator.initialize()

# Process chat requests
response = await orchestrator.process_chat(
    message="Hello, how can you help me today?",
    user_id="user_123",
    session_id="session_456"
)

print(response.content)
```

**Key Features:**
- Complete workflow orchestration and coordination
- Multi-agent support with intelligent routing
- Parallel processing with load balancing
- Session and context management
- Real-time streaming responses
- Comprehensive monitoring and metrics

### Request Processing Pipeline

Multi-stage request processing with validation, enrichment, and routing.

```python
from library.workflows.chat_workflow import RequestProcessor, RequestProcessorConfig

# Configure request processor
processor_config = RequestProcessorConfig(
    enable_input_validation=True,
    enable_context_enrichment=True,
    enable_request_routing=True,
    enable_priority_management=True,
    
    # Validation settings
    max_input_length=10000,
    allowed_content_types=['text', 'markdown'],
    enable_content_filtering=True,
    
    # Context enrichment
    context_sources=['conversation_history', 'user_preferences', 'session_data'],
    max_context_tokens=4000,
    
    # Routing configuration
    routing_rules=[
        {
            'keywords': ['technical', 'programming', 'code'],
            'target_agent': 'specialist_agent',
            'confidence_threshold': 0.8
        }
    ],
    
    # Priority management
    priority_factors=['user_tier', 'request_urgency', 'session_activity']
)

processor = RequestProcessor(processor_config)

# Process request through pipeline
async def process_user_request(raw_request):
    # Stage 1: Input validation
    validated_request = await processor.validate_input(raw_request)
    
    # Stage 2: Context enrichment
    enriched_request = await processor.enrich_context(validated_request)
    
    # Stage 3: Request routing
    routed_request = await processor.route_request(enriched_request)
    
    # Stage 4: Priority assignment
    prioritized_request = await processor.assign_priority(routed_request)
    
    return prioritized_request

# Usage
processed_request = await process_user_request({
    'message': 'Can you help me debug this Python code?',
    'user_id': 'user_123',
    'session_id': 'session_456',
    'timestamp': datetime.now()
})
```

### Input Validation

Comprehensive input validation and sanitization.

```python
from library.workflows.chat_workflow import InputValidator, ValidationConfig

# Configure input validation
validation_config = ValidationConfig(
    max_length=10000,
    min_length=1,
    allowed_languages=['en', 'es', 'fr'],
    enable_profanity_filter=True,
    enable_spam_detection=True,
    enable_injection_protection=True,
    
    # Content filtering
    blocked_patterns=[
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',              # JavaScript URLs
        r'data:.*base64'            # Base64 data URLs
    ],
    
    # Rate limiting
    rate_limit_per_user=100,  # requests per hour
    rate_limit_window=3600    # 1 hour
)

validator = InputValidator(validation_config)

# Validate input
async def validate_user_input(user_input):
    try:
        validation_result = await validator.validate(user_input)
        
        if validation_result.is_valid:
            return validation_result.sanitized_input
        else:
            raise ValidationError(
                f"Input validation failed: {validation_result.error_message}"
            )
    
    except ValidationError as e:
        logger.warning(f"Input validation failed: {e}")
        return None

# Usage
sanitized_input = await validate_user_input({
    'message': 'Hello, can you help me?',
    'user_id': 'user_123'
})
```

### Context Enrichment

Intelligent context enhancement for better responses.

```python
from library.workflows.chat_workflow import ContextEnricher, ContextConfig

# Configure context enrichment
context_config = ContextConfig(
    enable_conversation_history=True,
    history_window_size=10,
    
    enable_user_preferences=True,
    preference_categories=['language', 'expertise_level', 'communication_style'],
    
    enable_session_context=True,
    session_context_fields=['current_topic', 'user_goals', 'interaction_history'],
    
    enable_external_context=True,
    external_sources=['knowledge_base', 'user_profile', 'system_state'],
    
    max_context_tokens=4000,
    context_compression_threshold=3000
)

enricher = ContextEnricher(context_config)

# Enrich request with context
async def enrich_request_context(request):
    # Get conversation history
    conversation_context = await enricher.get_conversation_context(
        request.session_id,
        window_size=10
    )
    
    # Get user preferences
    user_preferences = await enricher.get_user_preferences(request.user_id)
    
    # Get session context
    session_context = await enricher.get_session_context(request.session_id)
    
    # Combine and optimize context
    enriched_context = await enricher.combine_contexts([
        conversation_context,
        user_preferences,
        session_context
    ])
    
    # Add context to request
    request.context = enriched_context
    return request

# Usage
enriched_request = await enrich_request_context(validated_request)
```

### Request Routing

Intelligent request routing to appropriate agents.

```python
from library.workflows.chat_workflow import RequestRouter, RoutingConfig

# Configure request routing
routing_config = RoutingConfig(
    routing_strategy="intelligent",  # or "round_robin", "load_balanced"
    
    # Agent capabilities
    agent_capabilities={
        'primary_agent': ['general', 'conversation', 'help'],
        'technical_agent': ['programming', 'debugging', 'technical'],
        'creative_agent': ['writing', 'creative', 'storytelling']
    },
    
    # Routing rules
    routing_rules=[
        {
            'name': 'technical_routing',
            'keywords': ['code', 'programming', 'debug', 'technical'],
            'target_agent': 'technical_agent',
            'confidence_threshold': 0.8
        },
        {
            'name': 'creative_routing',
            'keywords': ['write', 'story', 'creative', 'poem'],
            'target_agent': 'creative_agent',
            'confidence_threshold': 0.7
        }
    ],
    
    # Fallback configuration
    default_agent='primary_agent',
    enable_multi_agent_routing=True,
    max_agents_per_request=3
)

router = RequestRouter(routing_config)

# Route request to appropriate agent(s)
async def route_request(request):
    routing_decision = await router.determine_routing(request)
    
    if routing_decision.use_multiple_agents:
        # Route to multiple agents for complex requests
        return await router.route_to_multiple_agents(
            request,
            routing_decision.target_agents
        )
    else:
        # Route to single best agent
        return await router.route_to_agent(
            request,
            routing_decision.target_agent
        )

# Usage
routed_request = await route_request(enriched_request)
```

## Response Processing

### Response Aggregation

Intelligent response collection and merging from multiple sources.

```python
from library.workflows.chat_workflow import ResponseAggregator, AggregationConfig

# Configure response aggregation
aggregation_config = AggregationConfig(
    aggregation_strategy="intelligent_merge",  # or "best_response", "consensus"
    
    # Quality scoring
    enable_response_scoring=True,
    scoring_factors=['relevance', 'completeness', 'accuracy', 'clarity'],
    
    # Merging configuration
    enable_response_merging=True,
    merge_strategy="complementary",  # or "competitive", "collaborative"
    max_merged_length=5000,
    
    # Conflict resolution
    enable_conflict_resolution=True,
    conflict_resolution_strategy="expert_preference",
    
    # Formatting
    enable_response_formatting=True,
    output_format="markdown",
    include_source_attribution=True
)

aggregator = ResponseAggregator(aggregation_config)

# Aggregate responses from multiple agents
async def aggregate_responses(responses):
    # Score individual responses
    scored_responses = await aggregator.score_responses(responses)
    
    # Detect and resolve conflicts
    resolved_responses = await aggregator.resolve_conflicts(scored_responses)
    
    # Merge complementary responses
    merged_response = await aggregator.merge_responses(resolved_responses)
    
    # Format final response
    formatted_response = await aggregator.format_response(merged_response)
    
    return formatted_response

# Usage
agent_responses = [
    {'agent': 'technical_agent', 'content': 'Technical explanation...'},
    {'agent': 'primary_agent', 'content': 'General overview...'}
]

final_response = await aggregate_responses(agent_responses)
```

### Streaming Response Handler

Real-time streaming response support.

```python
from library.workflows.chat_workflow import StreamingHandler, StreamingConfig

# Configure streaming
streaming_config = StreamingConfig(
    enable_streaming=True,
    chunk_size=1024,
    streaming_timeout=30.0,
    
    # Buffering
    enable_buffering=True,
    buffer_size=4096,
    flush_interval=0.1,  # 100ms
    
    # Quality control
    enable_chunk_validation=True,
    min_chunk_size=10,
    max_chunk_size=2048,
    
    # Error handling
    enable_error_recovery=True,
    max_retry_attempts=3,
    retry_delay=1.0
)

streaming_handler = StreamingHandler(streaming_config)

# Stream response to client
async def stream_response_to_client(response_generator, client_connection):
    async with streaming_handler.create_stream(client_connection) as stream:
        async for chunk in response_generator:
            # Validate and process chunk
            processed_chunk = await streaming_handler.process_chunk(chunk)
            
            # Send to client
            await stream.send_chunk(processed_chunk)
            
            # Handle client feedback
            if await stream.has_client_feedback():
                feedback = await stream.get_client_feedback()
                if feedback.type == 'stop_generation':
                    break

# Usage with agent streaming
async def stream_agent_response(request, client_connection):
    agent_stream = await agent.process_streaming(request.message)
    await stream_response_to_client(agent_stream, client_connection)
```

## Session Management

### Session Store

Persistent session storage and management.

```python
from library.workflows.chat_workflow import SessionStore, SessionConfig

# Configure session management
session_config = SessionConfig(
    storage_backend="redis",  # or "database", "memory"
    connection_string="redis://localhost:6379/1",
    
    # Session lifecycle
    default_timeout=3600,  # 1 hour
    max_timeout=86400,     # 24 hours
    cleanup_interval=300,  # 5 minutes
    
    # Session data
    enable_session_persistence=True,
    enable_session_encryption=True,
    encryption_key="your-encryption-key",
    
    # Concurrency
    enable_session_locking=True,
    lock_timeout=10.0
)

session_store = SessionStore(session_config)

# Session operations
async def manage_user_session(user_id, session_data=None):
    # Create or get existing session
    session = await session_store.get_or_create_session(
        user_id=user_id,
        initial_data=session_data or {}
    )
    
    # Update session data
    await session_store.update_session(session.id, {
        'last_activity': datetime.now(),
        'interaction_count': session.data.get('interaction_count', 0) + 1
    })
    
    # Get session with lock for concurrent access
    async with session_store.lock_session(session.id) as locked_session:
        # Perform atomic operations on session
        locked_session.data['processing'] = True
        await session_store.save_session(locked_session)
    
    return session

# Usage
session = await manage_user_session("user_123", {'preferences': {'theme': 'dark'}})
```

### Context Manager

Advanced conversation context management.

```python
from library.workflows.chat_workflow import ContextManager, ContextManagerConfig

# Configure context management
context_config = ContextManagerConfig(
    # Context storage
    storage_backend="database",
    context_window_size=20,
    max_context_age=86400,  # 24 hours
    
    # Context optimization
    enable_context_compression=True,
    compression_threshold=10000,  # tokens
    compression_ratio=0.5,
    
    # Context enhancement
    enable_semantic_search=True,
    enable_topic_tracking=True,
    enable_entity_extraction=True,
    
    # Performance
    enable_context_caching=True,
    cache_ttl=300,  # 5 minutes
    max_cache_size=1000
)

context_manager = ContextManager(context_config)

# Manage conversation context
async def manage_conversation_context(session_id, new_message):
    # Get current context
    current_context = await context_manager.get_context(session_id)
    
    # Add new message to context
    await context_manager.add_message(
        session_id,
        role="user",
        content=new_message,
        metadata={'timestamp': datetime.now()}
    )
    
    # Optimize context if needed
    if await context_manager.should_optimize_context(session_id):
        await context_manager.optimize_context(session_id)
    
    # Get enhanced context for processing
    enhanced_context = await context_manager.get_enhanced_context(
        session_id,
        include_semantic_search=True,
        include_topic_summary=True
    )
    
    return enhanced_context

# Usage
context = await manage_conversation_context("session_456", "Hello, how are you?")
```

## Monitoring and Metrics

### Workflow Monitor

Comprehensive workflow performance monitoring.

```python
from library.workflows.chat_workflow import WorkflowMonitor, MonitoringConfig

# Configure monitoring
monitoring_config = MonitoringConfig(
    enable_performance_monitoring=True,
    enable_health_monitoring=True,
    enable_business_metrics=True,
    
    # Performance metrics
    track_response_times=True,
    track_throughput=True,
    track_error_rates=True,
    track_resource_usage=True,
    
    # Health monitoring
    health_check_interval=60.0,
    health_check_timeout=10.0,
    enable_auto_recovery=True,
    
    # Business metrics
    track_user_satisfaction=True,
    track_conversation_quality=True,
    track_agent_performance=True,
    
    # Alerting
    enable_alerting=True,
    alert_thresholds={
        'response_time': 5000,  # 5 seconds
        'error_rate': 0.05,     # 5%
        'throughput': 10        # requests per second
    }
)

monitor = WorkflowMonitor(monitoring_config)

# Monitor workflow performance
async def monitor_workflow_health():
    while True:
        # Collect performance metrics
        performance_metrics = await monitor.collect_performance_metrics()
        
        # Check health status
        health_status = await monitor.check_health()
        
        # Collect business metrics
        business_metrics = await monitor.collect_business_metrics()
        
        # Check for alerts
        alerts = await monitor.check_alert_conditions()
        
        if alerts:
            for alert in alerts:
                await monitor.send_alert(alert)
        
        # Log metrics
        logger.info(f"Workflow metrics: {performance_metrics}")
        
        await asyncio.sleep(60)  # Check every minute

# Start monitoring
asyncio.create_task(monitor_workflow_health())
```

### Metrics Collection

Detailed metrics collection and analysis.

```python
from library.workflows.chat_workflow import MetricsCollector, MetricsConfig

# Configure metrics collection
metrics_config = MetricsConfig(
    collection_interval=60.0,
    retention_period=86400 * 7,  # 7 days
    
    # Metric categories
    performance_metrics=[
        'response_time', 'throughput', 'error_rate', 'cpu_usage', 'memory_usage'
    ],
    business_metrics=[
        'user_satisfaction', 'conversation_length', 'task_completion_rate'
    ],
    agent_metrics=[
        'agent_response_time', 'agent_accuracy', 'delegation_rate'
    ],
    
    # Storage
    storage_backend="influxdb",
    connection_string="http://localhost:8086",
    database_name="nanobrain_metrics"
)

collector = MetricsCollector(metrics_config)

# Collect and analyze metrics
async def analyze_workflow_performance():
    # Get recent metrics
    recent_metrics = await collector.get_metrics(
        time_range=timedelta(hours=1),
        categories=['performance', 'business']
    )
    
    # Calculate key performance indicators
    kpis = await collector.calculate_kpis(recent_metrics)
    
    # Generate performance report
    report = await collector.generate_performance_report(
        time_range=timedelta(days=1),
        include_trends=True,
        include_recommendations=True
    )
    
    return {
        'current_metrics': recent_metrics,
        'kpis': kpis,
        'performance_report': report
    }

# Usage
performance_analysis = await analyze_workflow_performance()
print(f"Current throughput: {performance_analysis['kpis']['throughput']} req/s")
print(f"Average response time: {performance_analysis['kpis']['avg_response_time']}ms")
```

## Configuration

### ChatWorkflowConfig

Comprehensive configuration for the chat workflow.

```python
from library.workflows.chat_workflow import ChatWorkflowConfig

config = ChatWorkflowConfig(
    # Basic configuration
    name="production_chat_workflow",
    description="Production chat workflow with all features",
    version="1.0.0",
    
    # Agent configuration
    agents=[
        {
            'name': 'primary_agent',
            'type': 'collaborative',
            'model': 'gpt-4',
            'temperature': 0.7,
            'system_prompt': 'You are a helpful AI assistant.',
            'capabilities': ['general', 'conversation'],
            'delegation_rules': [
                {'keywords': ['technical'], 'agent': 'technical_agent'}
            ]
        }
    ],
    
    # Processing configuration
    enable_parallel_processing=True,
    max_parallel_requests=20,
    load_balancing_strategy="fastest_response",
    request_timeout=30.0,
    
    # Database configuration
    database_config={
        'adapter': 'postgresql',
        'host': 'localhost',
        'database': 'nanobrain',
        'user': 'app_user',
        'password': 'secure_password',
        'pool_size': 20
    },
    
    # Session management
    session_config={
        'storage_backend': 'redis',
        'connection_string': 'redis://localhost:6379/1',
        'default_timeout': 3600,
        'enable_persistence': True
    },
    
    # Context management
    context_config={
        'window_size': 20,
        'enable_compression': True,
        'enable_semantic_search': True,
        'storage_backend': 'database'
    },
    
    # Streaming configuration
    streaming_config={
        'enable_streaming': True,
        'chunk_size': 1024,
        'buffer_size': 4096,
        'timeout': 30.0
    },
    
    # Monitoring configuration
    monitoring_config={
        'enable_performance_monitoring': True,
        'enable_health_monitoring': True,
        'collection_interval': 60.0,
        'storage_backend': 'influxdb'
    },
    
    # Security configuration
    security_config={
        'enable_input_validation': True,
        'enable_rate_limiting': True,
        'enable_content_filtering': True,
        'max_input_length': 10000
    }
)
```

## Advanced Usage Examples

### Complete Chat Application

```python
from library.workflows.chat_workflow import ChatWorkflowOrchestrator
from library.infrastructure.interfaces.cli import InteractiveCLI

async def create_chat_application():
    # Initialize workflow
    orchestrator = ChatWorkflowOrchestrator.from_config("config/chat_workflow.yaml")
    await orchestrator.initialize()
    
    # Initialize CLI
    cli = InteractiveCLI.from_config("config/cli.yaml")
    await cli.initialize()
    
    # Main chat loop
    session_id = await orchestrator.create_session("user_123")
    
    await cli.print_header("NanoBrain Chat Assistant")
    await cli.print_info("Type 'quit' to exit, 'help' for commands")
    
    while True:
        # Get user input
        user_input = await cli.get_input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            break
        elif user_input.lower() == 'help':
            await show_help(cli)
            continue
        
        # Process with workflow
        try:
            with cli.progress_context("Processing..."):
                response = await orchestrator.process_chat(
                    message=user_input,
                    user_id="user_123",
                    session_id=session_id
                )
            
            # Display response
            await cli.print_response(f"Assistant: {response.content}")
            
            # Show metrics if requested
            if response.metadata.get('show_metrics'):
                await show_metrics(cli, response.metrics)
                
        except Exception as e:
            await cli.print_error(f"Error: {e}")
    
    # Cleanup
    await orchestrator.shutdown()
    await cli.shutdown()

async def show_help(cli):
    help_text = """
Available commands:
- help: Show this help message
- metrics: Show performance metrics
- history: Show conversation history
- clear: Clear conversation context
- quit/exit: Exit the application
    """
    await cli.print_info(help_text)

# Run the application
if __name__ == "__main__":
    asyncio.run(create_chat_application())
```

### Multi-User Chat Server

```python
from library.workflows.chat_workflow import ChatWorkflowOrchestrator
from library.infrastructure.interfaces.protocols import WebSocketAdapter
import asyncio

class ChatServer:
    def __init__(self, config_path: str):
        self.orchestrator = ChatWorkflowOrchestrator.from_config(config_path)
        self.active_sessions = {}
        self.websocket_server = None
    
    async def initialize(self):
        await self.orchestrator.initialize()
        
        # Start WebSocket server
        self.websocket_server = WebSocketAdapter({
            'host': '0.0.0.0',
            'port': 8080,
            'enable_compression': True
        })
        
        await self.websocket_server.start_server(self.handle_client_connection)
    
    async def handle_client_connection(self, websocket, path):
        user_id = await self.authenticate_user(websocket)
        session_id = await self.orchestrator.create_session(user_id)
        
        self.active_sessions[websocket] = {
            'user_id': user_id,
            'session_id': session_id,
            'connected_at': datetime.now()
        }
        
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            await self.cleanup_session(websocket)
    
    async def handle_message(self, websocket, message):
        session_info = self.active_sessions[websocket]
        
        try:
            # Process message through workflow
            response = await self.orchestrator.process_chat(
                message=message,
                user_id=session_info['user_id'],
                session_id=session_info['session_id']
            )
            
            # Send response back to client
            await websocket.send(json.dumps({
                'type': 'response',
                'content': response.content,
                'metadata': response.metadata
            }))
            
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def cleanup_session(self, websocket):
        if websocket in self.active_sessions:
            session_info = self.active_sessions[websocket]
            await self.orchestrator.close_session(session_info['session_id'])
            del self.active_sessions[websocket]

# Usage
server = ChatServer("config/chat_server.yaml")
await server.initialize()
```

### Batch Processing Workflow

```python
from library.workflows.chat_workflow import ChatWorkflowOrchestrator
import asyncio

async def process_batch_conversations(input_file: str, output_file: str):
    # Initialize workflow
    orchestrator = ChatWorkflowOrchestrator.from_config("config/batch_processing.yaml")
    await orchestrator.initialize()
    
    # Load conversations from file
    with open(input_file, 'r') as f:
        conversations = json.load(f)
    
    results = []
    
    # Process conversations in batches
    batch_size = 10
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i + batch_size]
        
        # Process batch concurrently
        batch_tasks = []
        for conv in batch:
            task = orchestrator.process_chat(
                message=conv['message'],
                user_id=conv['user_id'],
                session_id=conv.get('session_id')
            )
            batch_tasks.append(task)
        
        # Wait for batch completion
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
                results.append({
                    'input': batch[j],
                    'error': str(result)
                })
            else:
                results.append({
                    'input': batch[j],
                    'output': result.content,
                    'metadata': result.metadata
                })
        
        # Progress update
        logger.info(f"Processed {min(i + batch_size, len(conversations))}/{len(conversations)} conversations")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    await orchestrator.shutdown()

# Usage
await process_batch_conversations("input_conversations.json", "processed_results.json")
```

## Best Practices

### Resource Management

```python
# Always use proper initialization and cleanup
async def use_chat_workflow():
    orchestrator = ChatWorkflowOrchestrator(config)
    try:
        await orchestrator.initialize()
        
        # Use the workflow
        response = await orchestrator.process_chat(message, user_id, session_id)
        return response
        
    finally:
        await orchestrator.shutdown()

# Use context managers when available
async with ChatWorkflowOrchestrator(config) as orchestrator:
    response = await orchestrator.process_chat(message, user_id, session_id)
```

### Error Handling

```python
from library.workflows.chat_workflow import (
    WorkflowError,
    ProcessingError,
    ValidationError
)

try:
    response = await orchestrator.process_chat(message, user_id, session_id)
except ValidationError as e:
    # Handle input validation errors
    logger.warning(f"Invalid input: {e}")
    response = "I'm sorry, but your input doesn't meet our guidelines."
except ProcessingError as e:
    # Handle processing errors
    logger.error(f"Processing failed: {e}")
    response = "I'm experiencing technical difficulties. Please try again."
except WorkflowError as e:
    # Handle workflow-level errors
    logger.error(f"Workflow error: {e}")
    response = "Something went wrong. Please contact support."
```

### Performance Optimization

```python
# Configure appropriate resource limits
config = ChatWorkflowConfig(
    max_parallel_requests=min(cpu_count() * 2, 50),
    request_timeout=30.0,
    
    # Database connection pooling
    database_config={
        'pool_size': 20,
        'max_overflow': 10,
        'pool_timeout': 30.0
    },
    
    # Context optimization
    context_config={
        'window_size': 20,
        'enable_compression': True,
        'compression_threshold': 10000
    }
)

# Monitor and optimize performance
async def optimize_workflow_performance(orchestrator):
    metrics = await orchestrator.get_performance_metrics()
    
    if metrics.avg_response_time > 5000:  # 5 seconds
        # Scale up processing capacity
        await orchestrator.scale_processing_capacity(1.5)
    
    if metrics.memory_usage > 0.8:  # 80% memory usage
        # Optimize context management
        await orchestrator.optimize_context_management()
```

## Migration Guide

### From Demo Scripts

```python
# Before (demo script with 1000+ lines)
class ChatWorkflowDemo:
    def __init__(self):
        # 200+ lines of setup
        pass
    
    async def run_chat_loop(self):
        # 800+ lines of chat logic
        pass

# After (library workflow)
from library.workflows.chat_workflow import ChatWorkflowOrchestrator

orchestrator = ChatWorkflowOrchestrator.from_config("config.yaml")
await orchestrator.run_interactive_chat()
```

### Configuration Migration

```python
# Before (hardcoded configuration)
agents = [create_agent("gpt-4"), create_agent("gpt-3.5-turbo")]
db_connection = sqlite3.connect("chat.db")
cli = CustomCLI()

# After (YAML configuration)
# config/chat_workflow.yaml
"""
agents:
  - name: primary_agent
    model: gpt-4
    temperature: 0.7
  - name: backup_agent
    model: gpt-3.5-turbo
    temperature: 0.5

database:
  adapter: sqlite
  connection_string: chat.db

cli:
  enable_colors: true
  enable_history: true
"""

orchestrator = ChatWorkflowOrchestrator.from_config("config/chat_workflow.yaml")
```

## Performance Considerations

### Scalability

- Configure appropriate parallelism based on system resources
- Use connection pooling for database and external services
- Implement proper caching strategies for frequently accessed data
- Monitor resource usage and scale components as needed

### Memory Management

- Configure context window sizes appropriately
- Enable context compression for long conversations
- Implement proper session cleanup and garbage collection
- Monitor memory usage and implement alerts

### Network Optimization

- Use connection pooling for external API calls
- Implement proper timeout and retry policies
- Enable compression for large data transfers
- Consider using CDNs for static content

---

*For more information, see the [main library documentation](../../README.md) or explore the [chat workflow examples](../../../examples/library/workflows/chat_workflow/).* 