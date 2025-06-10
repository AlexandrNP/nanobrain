# NanoBrain Library API Reference

This document provides comprehensive API documentation for all classes and functions in the NanoBrain Library. It is designed to be a complete reference for developers building applications with the library.

## Table of Contents

1. [Infrastructure](#infrastructure)
   - [Data Management](#data-management)
   - [Interfaces](#interfaces)
   - [Parallel Processing](#parallel-processing)
   - [Load Balancing](#load-balancing)
   - [Monitoring](#monitoring)
2. [Agents](#agents)
   - [Enhanced Agents](#enhanced-agents)
   - [Protocol Support](#protocol-support)
   - [Delegation](#delegation)
3. [Workflows](#workflows)
   - [Chat Workflow](#chat-workflow)
   - [Request Processing](#request-processing)
   - [Response Aggregation](#response-aggregation)
4. [Configuration](#configuration)
5. [Exceptions](#exceptions)
6. [Utilities](#utilities)

---

## Infrastructure

### Data Management

#### DataUnitBase

**Abstract base class for all data units.**

```python
class DataUnitBase(ABC):
    """Abstract base class for data units that handle data storage and retrieval."""
    
    def __init__(self, config: Optional[DataUnitConfig] = None, **kwargs):
        """
        Initialize the data unit.
        
        Args:
            config: Configuration for the data unit
            **kwargs: Additional configuration parameters
        """
```

**Methods:**

- `async def initialize() -> None`
  - Initialize the data unit and prepare for operations
  - Must be called before using the data unit

- `async def shutdown() -> None`
  - Shutdown the data unit and clean up resources
  - Should be called when done with the data unit

- `async def get() -> Any`
  - Get data from the unit
  - Returns: The stored data or None if no data

- `async def set(data: Any) -> None`
  - Set data in the unit
  - Args: data - The data to store

- `async def clear() -> None`
  - Clear all data from the unit

- `async def set_metadata(key: str, value: Any) -> None`
  - Set metadata for the data unit
  - Args: key - Metadata key, value - Metadata value

- `async def get_metadata(key: str, default: Any = None) -> Any`
  - Get metadata value
  - Args: key - Metadata key, default - Default value if key not found
  - Returns: Metadata value or default

**Properties:**

- `is_initialized: bool` - Check if data unit is initialized
- `metadata: Dict[str, Any]` - Get metadata dictionary

#### DataUnitMemory

**In-memory data unit for fast access.**

```python
class DataUnitMemory(DataUnitBase):
    """In-memory data unit for fast, temporary storage."""
    
    def __init__(self, config: Optional[DataUnitConfig] = None, **kwargs):
        """
        Initialize memory data unit.
        
        Args:
            config: Configuration for the data unit
            **kwargs: Additional configuration parameters
        """
```

**Usage Example:**
```python
config = DataUnitConfig(data_type="memory", name="cache")
data_unit = DataUnitMemory(config)
await data_unit.initialize()

await data_unit.set({"user_id": "123", "data": "example"})
data = await data_unit.get()

await data_unit.shutdown()
```

#### DataUnitFile

**File-based data unit for persistent storage.**

```python
class DataUnitFile(DataUnitBase):
    """File-based data unit for persistent storage."""
    
    def __init__(self, file_path: str, config: Optional[DataUnitConfig] = None, **kwargs):
        """
        Initialize file data unit.
        
        Args:
            file_path: Path to the file for storage
            config: Configuration for the data unit
            **kwargs: Additional configuration parameters
        """
```

**Additional Methods:**

- `async def backup(backup_path: str) -> None`
  - Create a backup of the data file
  - Args: backup_path - Path for the backup file

- `async def restore(backup_path: str) -> None`
  - Restore data from a backup file
  - Args: backup_path - Path to the backup file

#### DataUnitStream

**Stream-based data unit with subscription support.**

```python
class DataUnitStream(DataUnitBase):
    """Stream-based data unit with pub/sub capabilities."""
    
    def __init__(self, config: Optional[DataUnitConfig] = None, **kwargs):
        """
        Initialize stream data unit.
        
        Args:
            config: Configuration for the data unit
            **kwargs: Additional configuration parameters
        """
```

**Additional Methods:**

- `async def subscribe() -> asyncio.Queue`
  - Subscribe to data updates
  - Returns: Queue for receiving updates

- `async def unsubscribe(queue: asyncio.Queue) -> None`
  - Unsubscribe from data updates
  - Args: queue - The queue to unsubscribe

- `async def publish(data: Any) -> None`
  - Publish data to all subscribers
  - Args: data - Data to publish

#### ConversationHistoryUnit

**Persistent conversation storage with search capabilities.**

```python
class ConversationHistoryUnit:
    """Manages conversation history with database persistence."""
    
    def __init__(
        self,
        database_adapter: DatabaseInterface,
        table_name: str = "conversations",
        **kwargs
    ):
        """
        Initialize conversation history unit.
        
        Args:
            database_adapter: Database adapter for persistence
            table_name: Name of the database table
            **kwargs: Additional configuration parameters
        """
```

**Methods:**

- `async def save_message(message: ConversationMessage) -> int`
  - Save a conversation message
  - Args: message - The conversation message to save
  - Returns: Message ID

- `async def get_conversation_history(conversation_id: str, limit: int = 50) -> List[ConversationMessage]`
  - Get conversation history
  - Args: conversation_id - ID of the conversation, limit - Maximum number of messages
  - Returns: List of conversation messages

- `async def search_conversations(query: str, limit: int = 10) -> List[ConversationMessage]`
  - Search conversations by text
  - Args: query - Search query, limit - Maximum number of results
  - Returns: List of matching messages

- `async def delete_conversation(conversation_id: str) -> None`
  - Delete an entire conversation
  - Args: conversation_id - ID of the conversation to delete

#### ConversationMessage

**Data class representing a conversation message.**

```python
@dataclass
class ConversationMessage:
    """Represents a single conversation exchange."""
    
    conversation_id: str
    user_input: str
    agent_response: str
    timestamp: datetime
    response_time_ms: float
    message_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Interfaces

#### DatabaseInterface

**Abstract interface for database operations.**

```python
class DatabaseInterface(ABC):
    """Abstract interface for database operations."""
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize database interface.
        
        Args:
            config: Database configuration
        """
```

**Methods:**

- `async def initialize() -> None`
  - Initialize database connection

- `async def shutdown() -> None`
  - Close database connection

- `async def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> Any`
  - Execute a database query
  - Args: query - SQL query, params - Query parameters
  - Returns: Query results

- `async def begin_transaction() -> Any`
  - Begin a database transaction
  - Returns: Transaction object

- `async def commit_transaction(transaction: Any) -> None`
  - Commit a database transaction
  - Args: transaction - Transaction to commit

- `async def rollback_transaction(transaction: Any) -> None`
  - Rollback a database transaction
  - Args: transaction - Transaction to rollback

#### SQLiteAdapter

**SQLite database adapter implementation.**

```python
class SQLiteAdapter(DatabaseInterface):
    """SQLite database adapter with WAL mode support."""
    
    def __init__(self, connection_string: str, config: Optional[DatabaseConfig] = None):
        """
        Initialize SQLite adapter.
        
        Args:
            connection_string: Path to SQLite database file
            config: Database configuration
        """
```

**Additional Methods:**

- `async def vacuum() -> None`
  - Optimize database by running VACUUM

- `async def get_table_info(table_name: str) -> List[Dict[str, Any]]`
  - Get information about a table
  - Args: table_name - Name of the table
  - Returns: Table schema information

#### InteractiveCLI

**Interactive command-line interface.**

```python
class InteractiveCLI:
    """Interactive command-line interface with advanced features."""
    
    def __init__(self, config: CLIConfig):
        """
        Initialize CLI interface.
        
        Args:
            config: CLI configuration
        """
```

**Methods:**

- `async def initialize() -> None`
  - Initialize the CLI interface

- `async def shutdown() -> None`
  - Shutdown the CLI interface

- `async def get_input(prompt: str = ">>> ") -> str`
  - Get user input with prompt
  - Args: prompt - Input prompt
  - Returns: User input string

- `async def print_response(message: str) -> None`
  - Print a response message
  - Args: message - Message to print

- `async def print_error(message: str) -> None`
  - Print an error message
  - Args: message - Error message to print

- `async def print_info(message: str) -> None`
  - Print an info message
  - Args: message - Info message to print

- `async def print_header(title: str) -> None`
  - Print a header with title
  - Args: title - Header title

- `async def show_menu(title: str, options: List[MenuOption]) -> str`
  - Show a menu and get user selection
  - Args: title - Menu title, options - List of menu options
  - Returns: Selected option key

- `def progress_context(message: str) -> ProgressContext`
  - Create a progress context manager
  - Args: message - Progress message
  - Returns: Progress context manager

### Parallel Processing

#### ParallelStep

**Generic parallel processing step.**

```python
class ParallelStep(Step, Generic[RequestType, ResponseType]):
    """Generic parallel processing step with configurable processors."""
    
    def __init__(
        self,
        config: ParallelProcessingConfig,
        processors: List[ProcessorType],
        load_balancer: Optional[LoadBalancer] = None
    ):
        """
        Initialize parallel step.
        
        Args:
            config: Parallel processing configuration
            processors: List of processors for parallel execution
            load_balancer: Load balancer for request distribution
        """
```

**Methods:**

- `async def process_batch(requests: List[RequestType]) -> List[ResponseType]`
  - Process multiple requests in parallel
  - Args: requests - List of requests to process
  - Returns: List of responses

- `async def add_processor(processor: ProcessorType) -> None`
  - Add a new processor at runtime
  - Args: processor - Processor to add

- `async def remove_processor(processor_id: str) -> None`
  - Remove a processor at runtime
  - Args: processor_id - ID of processor to remove

- `async def get_performance_stats() -> Dict[str, Any]`
  - Get performance statistics
  - Returns: Dictionary of performance metrics

- `async def get_processor_stats() -> Dict[str, ProcessorStats]`
  - Get per-processor statistics
  - Returns: Dictionary of processor statistics

#### ParallelAgentStep

**Parallel processing specialized for agents.**

```python
class ParallelAgentStep(ParallelStep[AgentRequest, AgentResponse]):
    """Parallel processing step specialized for NanoBrain agents."""
    
    def __init__(
        self,
        config: ParallelAgentConfig,
        agents: List[Agent],
        load_balancer: Optional[LoadBalancer] = None
    ):
        """
        Initialize parallel agent step.
        
        Args:
            config: Parallel agent configuration
            agents: List of agents for parallel processing
            load_balancer: Load balancer for agent selection
        """
```

**Additional Methods:**

- `async def process_requests(requests: List[AgentRequest]) -> List[AgentResponse]`
  - Process multiple agent requests
  - Args: requests - List of agent requests
  - Returns: List of agent responses

- `async def get_agent_health() -> Dict[str, bool]`
  - Get health status of all agents
  - Returns: Dictionary of agent health status

#### ParallelConversationalAgentStep

**Parallel processing optimized for conversational agents.**

```python
class ParallelConversationalAgentStep(ParallelAgentStep):
    """Parallel processing optimized for conversational agents."""
    
    def __init__(
        self,
        config: ParallelConversationalAgentConfig,
        agents: List[ConversationalAgent],
        load_balancer: Optional[LoadBalancer] = None
    ):
        """
        Initialize parallel conversational agent step.
        
        Args:
            config: Parallel conversational agent configuration
            agents: List of conversational agents
            load_balancer: Load balancer for agent selection
        """
```

**Additional Methods:**

- `async def process_conversations(requests: List[ConversationRequest]) -> List[ConversationResponse]`
  - Process multiple conversation requests
  - Args: requests - List of conversation requests
  - Returns: List of conversation responses

- `async def get_token_usage_stats() -> Dict[str, Any]`
  - Get token usage statistics
  - Returns: Dictionary of token usage metrics

### Load Balancing

#### LoadBalancer

**Abstract base class for load balancers.**

```python
class LoadBalancer(ABC):
    """Abstract base class for load balancing strategies."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """
        Initialize load balancer.
        
        Args:
            config: Load balancer configuration
        """
```

**Methods:**

- `async def select_processor(processors: List[ProcessorInfo]) -> ProcessorInfo`
  - Select a processor for the next request
  - Args: processors - List of available processors
  - Returns: Selected processor

- `async def update_processor_stats(processor: ProcessorInfo, response_time: float, success: bool) -> None`
  - Update processor statistics
  - Args: processor - Processor info, response_time - Response time, success - Success status

#### RoundRobinLoadBalancer

**Round-robin load balancing strategy.**

```python
class RoundRobinLoadBalancer(LoadBalancer):
    """Round-robin load balancing strategy."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """
        Initialize round-robin load balancer.
        
        Args:
            config: Load balancer configuration
        """
```

#### LeastLoadedLoadBalancer

**Least loaded load balancing strategy.**

```python
class LeastLoadedLoadBalancer(LoadBalancer):
    """Load balancer that selects the least loaded processor."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """
        Initialize least loaded load balancer.
        
        Args:
            config: Load balancer configuration
        """
```

### Monitoring

#### PerformanceMonitor

**Performance monitoring and metrics collection.**

```python
class PerformanceMonitor:
    """Monitors performance metrics for components."""
    
    def __init__(self, config: PerformanceMonitorConfig):
        """
        Initialize performance monitor.
        
        Args:
            config: Performance monitor configuration
        """
```

**Methods:**

- `async def start_monitoring() -> None`
  - Start performance monitoring

- `async def stop_monitoring() -> None`
  - Stop performance monitoring

- `async def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None`
  - Record a performance metric
  - Args: name - Metric name, value - Metric value, tags - Optional tags

- `async def get_metrics(time_range: Optional[timedelta] = None) -> Dict[str, Any]`
  - Get performance metrics
  - Args: time_range - Time range for metrics
  - Returns: Dictionary of metrics

- `async def get_current_stats() -> PerformanceStats`
  - Get current performance statistics
  - Returns: Current performance statistics

---

## Agents

### Enhanced Agents

#### CollaborativeAgent

**Multi-protocol collaborative agent.**

```python
class CollaborativeAgent(A2AProtocolMixin, MCPProtocolMixin, ConversationalAgent):
    """Enhanced agent with multi-protocol support and collaboration features."""
    
    def __init__(
        self,
        config: AgentConfig,
        a2a_config_path: Optional[str] = None,
        mcp_config_path: Optional[str] = None,
        delegation_rules: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize collaborative agent.
        
        Args:
            config: Agent configuration
            a2a_config_path: Path to A2A configuration file
            mcp_config_path: Path to MCP configuration file
            delegation_rules: List of delegation rules
            **kwargs: Additional configuration parameters
        """
```

**Methods:**

- `async def process(input_text: str, **kwargs) -> str`
  - Process input with delegation and protocol support
  - Args: input_text - Input text to process, **kwargs - Additional parameters
  - Returns: Processed response

- `async def delegate_to_agent(agent_name: str, input_text: str, **kwargs) -> str`
  - Delegate request to another agent
  - Args: agent_name - Name of target agent, input_text - Input to delegate
  - Returns: Response from delegated agent

- `async def get_delegation_stats() -> Dict[str, Any]`
  - Get delegation statistics
  - Returns: Dictionary of delegation metrics

### Protocol Support

#### A2AProtocolMixin

**Agent-to-Agent protocol support mixin.**

```python
class A2AProtocolMixin:
    """Mixin for Agent-to-Agent protocol support."""
    
    async def initialize_a2a(self, config_path: str) -> None:
        """
        Initialize A2A protocol support.
        
        Args:
            config_path: Path to A2A configuration file
        """
    
    async def call_a2a_agent(self, agent_name: str, message: str, **kwargs) -> str:
        """
        Call another agent via A2A protocol.
        
        Args:
            agent_name: Name of the target agent
            message: Message to send
            **kwargs: Additional parameters
        
        Returns:
            Response from the target agent
        """
```

#### MCPProtocolMixin

**Model Context Protocol support mixin.**

```python
class MCPProtocolMixin:
    """Mixin for Model Context Protocol support."""
    
    async def initialize_mcp(self, config_path: str) -> None:
        """
        Initialize MCP protocol support.
        
        Args:
            config_path: Path to MCP configuration file
        """
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
        
        Returns:
            Tool execution result
        """
```

### Delegation

#### DelegationEngine

**Task delegation and routing engine.**

```python
class DelegationEngine:
    """Engine for intelligent task delegation and routing."""
    
    def __init__(self, rules: List[DelegationRule]):
        """
        Initialize delegation engine.
        
        Args:
            rules: List of delegation rules
        """
```

**Methods:**

- `async def should_delegate(input_text: str) -> DelegationDecision`
  - Determine if input should be delegated
  - Args: input_text - Input text to analyze
  - Returns: Delegation decision

- `async def add_rule(rule: DelegationRule) -> None`
  - Add a new delegation rule
  - Args: rule - Delegation rule to add

- `async def remove_rule(rule_name: str) -> None`
  - Remove a delegation rule
  - Args: rule_name - Name of rule to remove

#### DelegationRule

**Rule for task delegation.**

```python
@dataclass
class DelegationRule:
    """Rule for determining task delegation."""
    
    name: str
    keywords: List[str]
    agent: str
    confidence_threshold: float = 0.7
    description: Optional[str] = None
    conditions: Optional[List[Dict[str, Any]]] = None
```

---

## Workflows

### Chat Workflow

#### ChatWorkflowOrchestrator

**Main chat workflow orchestrator.**

```python
class ChatWorkflowOrchestrator:
    """Orchestrates complete chat workflows with all components."""
    
    def __init__(self, config: ChatWorkflowConfig):
        """
        Initialize chat workflow orchestrator.
        
        Args:
            config: Chat workflow configuration
        """
    
    @classmethod
    def from_config(cls, config_path: str) -> 'ChatWorkflowOrchestrator':
        """
        Create orchestrator from configuration file.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Configured orchestrator instance
        """
```

**Methods:**

- `async def initialize() -> None`
  - Initialize the workflow orchestrator

- `async def shutdown() -> None`
  - Shutdown the workflow orchestrator

- `async def process_chat(message: str, user_id: str, session_id: Optional[str] = None) -> ChatResponse`
  - Process a chat message
  - Args: message - Chat message, user_id - User ID, session_id - Session ID
  - Returns: Chat response

- `async def create_session(user_id: str) -> str`
  - Create a new chat session
  - Args: user_id - User ID
  - Returns: Session ID

- `async def close_session(session_id: str) -> None`
  - Close a chat session
  - Args: session_id - Session ID to close

- `async def get_conversation_history(session_id: str, limit: int = 50) -> List[ConversationMessage]`
  - Get conversation history for a session
  - Args: session_id - Session ID, limit - Maximum number of messages
  - Returns: List of conversation messages

- `async def get_performance_stats() -> Dict[str, Any]`
  - Get workflow performance statistics
  - Returns: Dictionary of performance metrics

### Request Processing

#### RequestProcessor

**Multi-stage request processing pipeline.**

```python
class RequestProcessor:
    """Processes requests through validation, enrichment, and routing stages."""
    
    def __init__(self, config: RequestProcessorConfig):
        """
        Initialize request processor.
        
        Args:
            config: Request processor configuration
        """
```

**Methods:**

- `async def process_request(request: ChatRequest) -> ProcessedRequest`
  - Process a request through the full pipeline
  - Args: request - Chat request to process
  - Returns: Processed request

- `async def validate_input(request: ChatRequest) -> ChatRequest`
  - Validate and sanitize input
  - Args: request - Request to validate
  - Returns: Validated request

- `async def enrich_context(request: ChatRequest) -> ChatRequest`
  - Enrich request with context
  - Args: request - Request to enrich
  - Returns: Enriched request

- `async def route_request(request: ChatRequest) -> ChatRequest`
  - Route request to appropriate handler
  - Args: request - Request to route
  - Returns: Routed request

### Response Aggregation

#### ResponseAggregator

**Response collection and formatting.**

```python
class ResponseAggregator:
    """Aggregates and formats responses from multiple sources."""
    
    def __init__(self, config: AggregationConfig):
        """
        Initialize response aggregator.
        
        Args:
            config: Aggregation configuration
        """
```

**Methods:**

- `async def aggregate_responses(responses: List[AgentResponse]) -> AggregatedResponse`
  - Aggregate multiple responses
  - Args: responses - List of agent responses
  - Returns: Aggregated response

- `async def score_responses(responses: List[AgentResponse]) -> List[ScoredResponse]`
  - Score responses for quality
  - Args: responses - List of responses to score
  - Returns: List of scored responses

- `async def merge_responses(responses: List[ScoredResponse]) -> MergedResponse`
  - Merge complementary responses
  - Args: responses - List of scored responses
  - Returns: Merged response

---

## Configuration

### DataUnitConfig

**Configuration for data units.**

```python
@dataclass
class DataUnitConfig:
    """Configuration for data units."""
    
    data_type: str
    name: Optional[str] = None
    persistent: bool = False
    cache_size: int = 1000
    file_path: Optional[str] = None
    encoding: str = "utf-8"
    initial_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ParallelProcessingConfig

**Configuration for parallel processing.**

```python
@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing steps."""
    
    name: str
    description: Optional[str] = None
    max_parallel_requests: int = 10
    request_queue_size: int = 100
    request_timeout: float = 30.0
    load_balancing_strategy: str = "round_robin"
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 30.0
    enable_health_monitoring: bool = True
    health_check_interval: float = 60.0
    enable_performance_tracking: bool = True
```

### ChatWorkflowConfig

**Configuration for chat workflows.**

```python
@dataclass
class ChatWorkflowConfig:
    """Configuration for chat workflows."""
    
    name: str
    description: Optional[str] = None
    agents: List[Dict[str, Any]] = field(default_factory=list)
    delegation_rules: List[Dict[str, Any]] = field(default_factory=list)
    enable_parallel_processing: bool = True
    max_parallel_requests: int = 10
    load_balancing_strategy: str = "round_robin"
    database_config: Dict[str, Any] = field(default_factory=dict)
    session_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
```

---

## Exceptions

### DataUnitError

**Base exception for data unit operations.**

```python
class DataUnitError(Exception):
    """Base exception for data unit operations."""
    pass

class DataUnitNotInitializedError(DataUnitError):
    """Raised when data unit is not initialized."""
    pass

class DataUnitStorageError(DataUnitError):
    """Raised when data storage operation fails."""
    pass
```

### AgentError

**Base exception for agent operations.**

```python
class AgentError(Exception):
    """Base exception for agent operations."""
    pass

class AgentNotInitializedError(AgentError):
    """Raised when agent is not initialized."""
    pass

class AgentProcessingError(AgentError):
    """Raised when agent processing fails."""
    pass

class DelegationError(AgentError):
    """Raised when agent delegation fails."""
    pass
```

### WorkflowError

**Base exception for workflow operations.**

```python
class WorkflowError(Exception):
    """Base exception for workflow operations."""
    pass

class WorkflowConfigurationError(WorkflowError):
    """Raised when workflow configuration is invalid."""
    pass

class WorkflowExecutionError(WorkflowError):
    """Raised when workflow execution fails."""
    pass
```

---

## Utilities

### Factory Functions

#### create_data_unit

**Factory function for creating data units.**

```python
def create_data_unit(config: DataUnitConfig) -> DataUnitBase:
    """
    Create a data unit based on configuration.
    
    Args:
        config: Data unit configuration
    
    Returns:
        Configured data unit instance
    
    Raises:
        ValueError: If data_type is not supported
    """
```

#### create_agent

**Factory function for creating agents.**

```python
def create_agent(config: AgentConfig, agent_type: str = "collaborative") -> Agent:
    """
    Create an agent based on configuration.
    
    Args:
        config: Agent configuration
        agent_type: Type of agent to create
    
    Returns:
        Configured agent instance
    
    Raises:
        ValueError: If agent_type is not supported
    """
```

### Helper Functions

#### get_logger

**Get a configured logger instance.**

```python
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger instance
    """
```

#### load_config

**Load configuration from file.**

```python
def load_config(config_path: str, config_type: Type[T]) -> T:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        config_type: Type of configuration to load
    
    Returns:
        Loaded configuration instance
    
    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If configuration is invalid
    """
```

---

## Usage Patterns

### Basic Usage Pattern

```python
# 1. Create and initialize components
data_unit = DataUnitMemory(DataUnitConfig(data_type="memory"))
await data_unit.initialize()

agent = CollaborativeAgent(AgentConfig(name="assistant", model="gpt-3.5-turbo"))
await agent.initialize()

# 2. Use components
await data_unit.set({"context": "user preferences"})
response = await agent.process("Hello, how are you?")

# 3. Clean up
await agent.shutdown()
await data_unit.shutdown()
```

### Workflow Pattern

```python
# 1. Create workflow from configuration
orchestrator = ChatWorkflowOrchestrator.from_config("config/workflow.yaml")
await orchestrator.initialize()

# 2. Process requests
response = await orchestrator.process_chat(
    message="Hello",
    user_id="user_123",
    session_id="session_456"
)

# 3. Clean up
await orchestrator.shutdown()
```

### Error Handling Pattern

```python
try:
    await component.initialize()
    result = await component.process(data)
except ComponentError as e:
    logger.error(f"Component error: {e}")
    # Handle specific component errors
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors
finally:
    await component.shutdown()
```

---

## Type Definitions

### Common Types

```python
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

# Request/Response types
RequestType = Any
ResponseType = Any
ProcessorType = Callable[[RequestType], ResponseType]

# Configuration types
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]

# Callback types
CallbackFunction = Callable[[Any], None]
AsyncCallbackFunction = Callable[[Any], Awaitable[None]]
```

### Enums

```python
class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    WEIGHTED = "weighted"
    RANDOM = "random"

class DataUnitType(Enum):
    """Data unit types."""
    MEMORY = "memory"
    FILE = "file"
    STREAM = "stream"
    STRING = "string"
```

---

This API reference provides comprehensive documentation for all public classes, methods, and functions in the NanoBrain Library. For implementation examples and usage patterns, see the [Getting Started Guide](GETTING_STARTED.md) and [Architecture Documentation](ARCHITECTURE.md).

*Note: This documentation is automatically generated from the source code. For the most up-to-date information, refer to the source code and inline documentation.* 