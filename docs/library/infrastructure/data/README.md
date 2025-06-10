# Data Infrastructure

The data infrastructure module provides core data abstractions and implementations for the NanoBrain framework. It separates data management concerns from database-specific implementations, enabling flexible and testable data handling.

## Overview

This module contains:
- **Abstract Data Interfaces**: Base classes defining data unit contracts
- **Core Implementations**: Memory, file, and stream-based data units
- **Conversation Management**: Persistent conversation storage and retrieval
- **Session Management**: Session lifecycle and metadata handling
- **Export/Import Utilities**: Data serialization and migration tools

## Architecture

```
library/infrastructure/data/
├── data_unit_base.py          # Abstract base classes and interfaces
├── memory_data_unit.py        # In-memory data storage
├── file_data_unit.py          # File-based persistent storage
├── stream_data_unit.py        # Stream-based data with subscriptions
├── string_data_unit.py        # String manipulation and append operations
├── conversation_history.py    # Conversation storage and retrieval
├── session_manager.py         # Session lifecycle management
└── export_manager.py          # Data export and import utilities
```

## Core Components

### DataUnitBase

The foundation of all data units, providing a consistent interface for data operations.

```python
from library.infrastructure.data import DataUnitBase, DataUnitConfig

class DataUnitBase(ABC):
    """Abstract base class for all data units."""
    
    async def get(self) -> Any:
        """Get data from the unit."""
        
    async def set(self, data: Any) -> None:
        """Set data in the unit."""
        
    async def clear(self) -> None:
        """Clear data from the unit."""
        
    async def initialize(self) -> None:
        """Initialize the data unit."""
        
    async def shutdown(self) -> None:
        """Shutdown the data unit."""
```

**Key Features:**
- Async/await support for all operations
- Metadata management for tracking data state
- Logging integration for debugging and monitoring
- Lifecycle management (initialize/shutdown)
- Thread-safe operations with async locks

### Memory Data Unit

Fast, in-memory data storage for temporary data and caching.

```python
from library.infrastructure.data import DataUnitMemory, DataUnitConfig

# Create memory data unit
config = DataUnitConfig(data_type="memory", name="cache")
data_unit = DataUnitMemory(config)
await data_unit.initialize()

# Store and retrieve data
await data_unit.set({"user_id": "123", "preferences": {"theme": "dark"}})
data = await data_unit.get()
print(data)  # {"user_id": "123", "preferences": {"theme": "dark"}}

# Clean up
await data_unit.shutdown()
```

**Use Cases:**
- Temporary data storage during processing
- Caching frequently accessed data
- Inter-step data transfer in workflows
- Session state management

### File Data Unit

Persistent file-based storage with automatic serialization.

```python
from library.infrastructure.data import DataUnitFile, DataUnitConfig

# Create file data unit
config = DataUnitConfig(
    data_type="file",
    file_path="user_data.json",
    encoding="utf-8"
)
data_unit = DataUnitFile("user_data.json", config)
await data_unit.initialize()

# Store data (automatically serialized to JSON)
await data_unit.set({
    "conversations": [
        {"id": "conv_1", "messages": 5},
        {"id": "conv_2", "messages": 3}
    ]
})

# Data persists across application restarts
data = await data_unit.get()
```

**Features:**
- Automatic JSON serialization/deserialization
- Configurable file encoding
- Atomic write operations
- File locking for concurrent access
- Backup and recovery support

### Stream Data Unit

Real-time data streaming with subscription support.

```python
from library.infrastructure.data import DataUnitStream, DataUnitConfig

# Create stream data unit
config = DataUnitConfig(data_type="stream", name="events")
stream = DataUnitStream(config)
await stream.initialize()

# Subscribe to data updates
subscriber_queue = await stream.subscribe()

# Publish data
await stream.set({"event": "user_login", "user_id": "123"})

# Receive data in subscriber
event_data = await subscriber_queue.get()
print(event_data)  # {"event": "user_login", "user_id": "123"}

# Unsubscribe when done
await stream.unsubscribe(subscriber_queue)
```

**Use Cases:**
- Real-time event streaming
- Pub/sub messaging patterns
- Live data feeds
- Reactive programming patterns

### String Data Unit

Specialized string handling with append operations.

```python
from library.infrastructure.data import DataUnitString, DataUnitConfig

# Create string data unit
config = DataUnitConfig(data_type="string", initial_value="Log started\n")
log_unit = DataUnitString("", config)
await log_unit.initialize()

# Append log entries
await log_unit.append("User logged in\n")
await log_unit.append("Processing request\n")
await log_unit.append("Request completed\n")

# Get complete log
log_content = await log_unit.get()
print(log_content)
# Output:
# Log started
# User logged in
# Processing request
# Request completed
```

**Features:**
- Efficient string concatenation
- Append-only operations
- Configurable initial values
- Memory-efficient for large text data

## Conversation Management

### ConversationHistoryUnit

Persistent storage and retrieval of conversation data with search capabilities.

```python
from library.infrastructure.data import ConversationHistoryUnit, ConversationMessage
from library.infrastructure.interfaces.database import SQLiteAdapter
from datetime import datetime

# Setup database adapter
db_adapter = SQLiteAdapter("conversations.db")
await db_adapter.initialize()

# Create conversation history unit
history_unit = ConversationHistoryUnit(
    database_adapter=db_adapter,
    table_name="conversations"
)
await history_unit.initialize()

# Save conversation message
message = ConversationMessage(
    conversation_id="conv_123",
    user_input="What is machine learning?",
    agent_response="Machine learning is a subset of AI...",
    timestamp=datetime.now(),
    response_time_ms=1250.0
)
await history_unit.save_message(message)

# Retrieve conversation history
history = await history_unit.get_conversation_history("conv_123", limit=10)
for msg in history:
    print(f"User: {msg.user_input}")
    print(f"Agent: {msg.agent_response}")
    print(f"Time: {msg.response_time_ms}ms\n")

# Search conversations
results = await history_unit.search_conversations(
    query="machine learning",
    limit=5
)
```

**Features:**
- SQLite backend with full-text search
- Conversation threading and context
- Performance metrics tracking
- Export/import capabilities
- Automatic schema management

### ConversationMessage

Data class representing a single conversation exchange.

```python
@dataclass
class ConversationMessage:
    conversation_id: str
    user_input: str
    agent_response: str
    timestamp: datetime
    response_time_ms: float
    message_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### ConversationQuery

Query builder for conversation searches.

```python
from library.infrastructure.data import ConversationQuery

# Build complex queries
query = ConversationQuery() \
    .conversation_id("conv_123") \
    .date_range(start_date, end_date) \
    .contains_text("machine learning") \
    .min_response_time(1000) \
    .limit(20)

results = await history_unit.query_conversations(query)
```

## Session Management

### SessionManager

Manages session lifecycle, metadata, and cleanup.

```python
from library.infrastructure.data import SessionManager, SessionConfig
from datetime import timedelta

# Configure session management
config = SessionConfig(
    session_timeout=timedelta(hours=2),
    cleanup_interval=timedelta(minutes=30),
    max_sessions=1000
)

session_manager = SessionManager(config)
await session_manager.initialize()

# Create new session
session_id = await session_manager.create_session(
    user_id="user_123",
    metadata={"client": "web", "version": "1.0"}
)

# Update session data
await session_manager.update_session(session_id, {
    "last_activity": datetime.now(),
    "page_views": 5
})

# Get session info
session_data = await session_manager.get_session(session_id)
print(f"Session created: {session_data.created_at}")
print(f"Last activity: {session_data.last_activity}")

# Cleanup expired sessions
await session_manager.cleanup_expired_sessions()
```

**Features:**
- Automatic session expiration
- Configurable cleanup policies
- Session metadata tracking
- Memory-efficient storage
- Concurrent session handling

### SessionData

Data class representing session information.

```python
@dataclass
class SessionData:
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]
    is_active: bool = True
```

## Export and Import

### ExportManager

Handles data serialization and export to various formats.

```python
from library.infrastructure.data import ExportManager, ExportFormat

export_manager = ExportManager()

# Export conversation history to JSON
await export_manager.export_conversations(
    conversation_ids=["conv_1", "conv_2"],
    format=ExportFormat.JSON,
    output_file="conversations.json",
    include_metadata=True
)

# Export to CSV for analysis
await export_manager.export_conversations(
    conversation_ids=["conv_1", "conv_2"],
    format=ExportFormat.CSV,
    output_file="conversations.csv",
    fields=["timestamp", "user_input", "agent_response", "response_time_ms"]
)

# Export session data
await export_manager.export_sessions(
    session_ids=["session_1", "session_2"],
    format=ExportFormat.JSON,
    output_file="sessions.json"
)
```

### ImportManager

Handles data import and migration from external sources.

```python
from library.infrastructure.data import ImportManager

import_manager = ImportManager()

# Import conversations from JSON
await import_manager.import_conversations(
    input_file="backup_conversations.json",
    format=ExportFormat.JSON,
    merge_strategy="append"  # or "replace", "skip_duplicates"
)

# Import from CSV
await import_manager.import_conversations(
    input_file="external_data.csv",
    format=ExportFormat.CSV,
    field_mapping={
        "user_message": "user_input",
        "bot_response": "agent_response",
        "timestamp": "timestamp"
    }
)
```

**Supported Formats:**
- JSON (structured data with full metadata)
- CSV (tabular data for analysis)
- XML (hierarchical data representation)
- Parquet (columnar format for big data)

## Configuration

### DataUnitConfig

Configuration class for data unit behavior.

```python
from library.infrastructure.data import DataUnitConfig, DataUnitType

config = DataUnitConfig(
    data_type=DataUnitType.MEMORY,
    persistent=False,
    cache_size=1000,
    file_path=None,
    encoding="utf-8",
    initial_value=None
)
```

**Configuration Options:**
- `data_type`: Type of data unit (memory, file, stream, string)
- `persistent`: Whether data survives application restarts
- `cache_size`: Maximum number of items to cache
- `file_path`: Path for file-based storage
- `encoding`: Character encoding for text data
- `initial_value`: Default value for new data units

## Best Practices

### Memory Management

```python
# Always initialize and shutdown data units
async def use_data_unit():
    data_unit = DataUnitMemory(config)
    try:
        await data_unit.initialize()
        # Use data unit
        await data_unit.set(data)
        result = await data_unit.get()
        return result
    finally:
        await data_unit.shutdown()
```

### Error Handling

```python
from library.infrastructure.data import DataUnitError

try:
    await data_unit.set(large_data)
except DataUnitError as e:
    logger.error(f"Data unit operation failed: {e}")
    # Implement fallback strategy
    await fallback_data_unit.set(large_data)
```

### Performance Optimization

```python
# Use appropriate data unit types
# - Memory: Fast access, temporary data
# - File: Persistent data, moderate access
# - Stream: Real-time data, pub/sub patterns

# Configure cache sizes appropriately
config = DataUnitConfig(
    data_type=DataUnitType.MEMORY,
    cache_size=10000  # Adjust based on memory constraints
)

# Use batch operations when possible
messages = [msg1, msg2, msg3]
await history_unit.save_messages_batch(messages)
```

### Testing

```python
import pytest
from library.infrastructure.data import DataUnitMemory

@pytest.fixture
async def memory_data_unit():
    config = DataUnitConfig(data_type="memory")
    unit = DataUnitMemory(config)
    await unit.initialize()
    yield unit
    await unit.shutdown()

async def test_data_storage(memory_data_unit):
    test_data = {"key": "value"}
    await memory_data_unit.set(test_data)
    result = await memory_data_unit.get()
    assert result == test_data
```

## Integration Examples

### With Workflow Steps

```python
from library.infrastructure.data import DataUnitMemory
from core.step import Step

class ProcessingStep(Step):
    def __init__(self, config, input_data_unit, output_data_unit):
        super().__init__(config)
        self.input_data_unit = input_data_unit
        self.output_data_unit = output_data_unit
    
    async def process(self, inputs):
        # Get data from input unit
        input_data = await self.input_data_unit.get()
        
        # Process data
        result = self.transform(input_data)
        
        # Store result in output unit
        await self.output_data_unit.set(result)
        
        return result
```

### With Database Adapters

```python
from library.infrastructure.data import ConversationHistoryUnit
from library.infrastructure.interfaces.database import PostgreSQLAdapter

# Use PostgreSQL for production
db_adapter = PostgreSQLAdapter(
    host="localhost",
    database="nanobrain",
    user="app_user",
    password="secure_password"
)

history_unit = ConversationHistoryUnit(
    database_adapter=db_adapter,
    table_name="conversations"
)
```

## Migration Guide

### From Core Data Units

```python
# Before (core data units)
from core.data_unit import DataUnitMemory

# After (library data units)
from library.infrastructure.data import DataUnitMemory
```

### From Demo Implementations

```python
# Before (demo-specific conversation history)
class ConversationHistoryManager:
    def __init__(self, db_path):
        self.db_path = db_path
        # 100+ lines of SQLite-specific code

# After (library conversation history)
from library.infrastructure.data import ConversationHistoryUnit
from library.infrastructure.interfaces.database import SQLiteAdapter

db_adapter = SQLiteAdapter(db_path)
history_unit = ConversationHistoryUnit(database_adapter=db_adapter)
```

## Performance Considerations

### Memory Usage

- **Memory Data Units**: Keep data size reasonable (< 100MB per unit)
- **File Data Units**: Use for larger datasets that don't fit in memory
- **Stream Data Units**: Configure subscriber limits to prevent memory leaks

### Concurrency

- All data units are thread-safe with async locks
- Use connection pooling for database-backed data units
- Consider read replicas for high-read workloads

### Caching

- Configure appropriate cache sizes based on available memory
- Use LRU eviction policies for memory-constrained environments
- Monitor cache hit rates and adjust sizes accordingly

---

*For more information, see the [main library documentation](../../README.md) or explore the [examples](../../../examples/library/data/).* 