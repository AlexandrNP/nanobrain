# Infrastructure Interfaces

The interfaces module provides external system interfaces and database adapters for the NanoBrain framework. It implements the adapter pattern to abstract away implementation details and enable flexible system integration.

## Overview

This module contains:
- **Database Interfaces**: SQL and NoSQL database adapters
- **CLI Components**: Interactive command-line interfaces
- **External System Integrations**: APIs, message queues, file systems
- **Protocol Adapters**: Communication protocol implementations
- **Connection Management**: Pooling, retry logic, and health monitoring

## Architecture

```
library/infrastructure/interfaces/
├── database/                   # Database adapters and interfaces
│   ├── database_interface.py   # Abstract database interface
│   ├── sql_interface.py        # SQL database abstraction
│   ├── nosql_interface.py      # NoSQL database abstraction
│   ├── sqlite_adapter.py       # SQLite implementation
│   ├── mysql_adapter.py        # MySQL implementation
│   ├── postgresql_adapter.py   # PostgreSQL implementation
│   └── mongodb_adapter.py      # MongoDB implementation
├── cli/                        # Command-line interfaces
│   ├── cli_interface.py        # Base CLI interface
│   ├── interactive_cli.py      # Interactive command-line interface
│   ├── menu_system.py          # Menu-driven interface
│   └── progress_display.py     # Progress bars and status display
├── external/                   # External system integrations
│   ├── api_client.py           # REST API client interface
│   ├── message_queue.py        # Message queue interface
│   ├── file_system.py          # File system operations
│   └── webhook_handler.py      # Webhook processing
└── protocols/                  # Protocol implementations
    ├── http_adapter.py         # HTTP protocol adapter
    ├── websocket_adapter.py    # WebSocket protocol adapter
    └── grpc_adapter.py         # gRPC protocol adapter
```

## Database Interfaces

### DatabaseInterface

Abstract base class for all database adapters.

```python
from library.infrastructure.interfaces.database import DatabaseInterface, DatabaseConfig

class DatabaseInterface(ABC):
    """Abstract interface for database operations."""
    
    async def initialize(self) -> None:
        """Initialize database connection."""
        
    async def shutdown(self) -> None:
        """Close database connection."""
        
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute a database query."""
        
    async def begin_transaction(self) -> Any:
        """Begin a database transaction."""
        
    async def commit_transaction(self, transaction: Any) -> None:
        """Commit a database transaction."""
        
    async def rollback_transaction(self, transaction: Any) -> None:
        """Rollback a database transaction."""
```

**Key Features:**
- Connection pooling and management
- Transaction support with rollback
- Query parameterization for security
- Health monitoring and reconnection
- Performance metrics collection

### SQLite Adapter

Lightweight, file-based SQL database adapter.

```python
from library.infrastructure.interfaces.database import SQLiteAdapter, DatabaseConfig

# Configure SQLite adapter
config = DatabaseConfig(
    connection_string="conversations.db",
    pool_size=5,
    timeout=30.0,
    enable_wal_mode=True  # Write-Ahead Logging for better concurrency
)

adapter = SQLiteAdapter(config)
await adapter.initialize()

# Execute queries
result = await adapter.execute_query(
    "SELECT * FROM conversations WHERE user_id = :user_id",
    {"user_id": "user_123"}
)

# Use transactions
async with adapter.transaction() as tx:
    await adapter.execute_query(
        "INSERT INTO conversations (user_id, message) VALUES (:user_id, :message)",
        {"user_id": "user_123", "message": "Hello"},
        transaction=tx
    )
    await adapter.execute_query(
        "UPDATE user_stats SET message_count = message_count + 1 WHERE user_id = :user_id",
        {"user_id": "user_123"},
        transaction=tx
    )
    # Transaction automatically committed on success

await adapter.shutdown()
```

**Features:**
- WAL mode for improved concurrency
- Full-text search support
- Automatic schema migration
- Backup and restore utilities
- Memory-mapped I/O optimization

### PostgreSQL Adapter

Production-ready PostgreSQL database adapter.

```python
from library.infrastructure.interfaces.database import PostgreSQLAdapter, DatabaseConfig

# Configure PostgreSQL adapter
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="nanobrain",
    user="app_user",
    password="secure_password",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30.0,
    enable_ssl=True
)

adapter = PostgreSQLAdapter(config)
await adapter.initialize()

# Use connection pooling
async with adapter.get_connection() as conn:
    result = await conn.execute_query(
        "SELECT * FROM conversations WHERE created_at > $1",
        [datetime.now() - timedelta(days=7)]
    )

# Batch operations
await adapter.execute_batch(
    "INSERT INTO messages (conversation_id, content, timestamp) VALUES ($1, $2, $3)",
    [
        ("conv_1", "Hello", datetime.now()),
        ("conv_1", "How are you?", datetime.now()),
        ("conv_2", "Good morning", datetime.now())
    ]
)
```

**Features:**
- Connection pooling with overflow handling
- SSL/TLS encryption support
- Prepared statement caching
- Streaming result sets for large queries
- Advanced indexing and partitioning support

### MySQL Adapter

MySQL database adapter with replication support.

```python
from library.infrastructure.interfaces.database import MySQLAdapter, DatabaseConfig

# Configure MySQL with read/write splitting
config = DatabaseConfig(
    write_host="mysql-master.example.com",
    read_hosts=["mysql-replica1.example.com", "mysql-replica2.example.com"],
    database="nanobrain",
    user="app_user",
    password="secure_password",
    charset="utf8mb4",
    enable_read_write_splitting=True
)

adapter = MySQLAdapter(config)
await adapter.initialize()

# Write operations go to master
await adapter.execute_write_query(
    "INSERT INTO conversations (user_id, content) VALUES (%s, %s)",
    ("user_123", "Hello world")
)

# Read operations use replicas
result = await adapter.execute_read_query(
    "SELECT * FROM conversations WHERE user_id = %s",
    ("user_123",)
)
```

**Features:**
- Master-slave replication support
- Read/write query splitting
- Connection failover and recovery
- Character set and collation handling
- Performance schema integration

### MongoDB Adapter

NoSQL document database adapter.

```python
from library.infrastructure.interfaces.database import MongoDBAdapter, DatabaseConfig

# Configure MongoDB adapter
config = DatabaseConfig(
    connection_string="mongodb://localhost:27017/nanobrain",
    replica_set="rs0",
    read_preference="secondaryPreferred",
    write_concern={"w": "majority", "j": True}
)

adapter = MongoDBAdapter(config)
await adapter.initialize()

# Document operations
conversation_id = await adapter.insert_document(
    "conversations",
    {
        "user_id": "user_123",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ],
        "created_at": datetime.now()
    }
)

# Query with aggregation
pipeline = [
    {"$match": {"user_id": "user_123"}},
    {"$unwind": "$messages"},
    {"$group": {"_id": "$user_id", "message_count": {"$sum": 1}}}
]
result = await adapter.aggregate("conversations", pipeline)
```

**Features:**
- Replica set support with read preferences
- Aggregation pipeline support
- GridFS for large file storage
- Change streams for real-time updates
- Automatic index management

## CLI Interfaces

### InteractiveCLI

Interactive command-line interface with menu systems and user input handling.

```python
from library.infrastructure.interfaces.cli import InteractiveCLI, CLIConfig, MenuOption

# Configure CLI
config = CLIConfig(
    app_name="NanoBrain Chat",
    prompt_style=">>> ",
    enable_history=True,
    history_file=".nanobrain_history",
    enable_autocomplete=True
)

cli = InteractiveCLI(config)

# Define menu options
menu_options = [
    MenuOption("1", "Start Chat", "Begin a new conversation"),
    MenuOption("2", "View History", "Show conversation history"),
    MenuOption("3", "Settings", "Configure application settings"),
    MenuOption("q", "Quit", "Exit the application")
]

# Run interactive session
async def main():
    await cli.initialize()
    
    while True:
        choice = await cli.show_menu("Main Menu", menu_options)
        
        if choice == "1":
            await start_chat_session(cli)
        elif choice == "2":
            await show_conversation_history(cli)
        elif choice == "3":
            await configure_settings(cli)
        elif choice == "q":
            break
    
    await cli.shutdown()

async def start_chat_session(cli):
    await cli.print_header("Chat Session")
    
    while True:
        user_input = await cli.get_input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # Process with agent
        with cli.progress_context("Processing..."):
            response = await agent.process(user_input)
        
        await cli.print_response("Assistant: " + response)
```

**Features:**
- Command history and autocomplete
- Progress bars and status indicators
- Colored output and formatting
- Input validation and error handling
- Menu-driven navigation

### ProgressDisplay

Advanced progress tracking and status display.

```python
from library.infrastructure.interfaces.cli import ProgressDisplay, ProgressConfig

# Configure progress display
config = ProgressConfig(
    show_percentage=True,
    show_eta=True,
    show_rate=True,
    bar_width=50,
    update_interval=0.1
)

progress = ProgressDisplay(config)

# Single progress bar
async def process_batch(items):
    with progress.progress_bar("Processing items", total=len(items)) as bar:
        for i, item in enumerate(items):
            await process_item(item)
            bar.update(i + 1)

# Multiple progress bars
async def complex_processing():
    with progress.multi_progress() as multi:
        # Download progress
        download_bar = multi.add_task("Downloading", total=100)
        
        # Processing progress
        process_bar = multi.add_task("Processing", total=50)
        
        # Upload progress
        upload_bar = multi.add_task("Uploading", total=25)
        
        # Update bars as work progresses
        for i in range(100):
            await asyncio.sleep(0.1)
            multi.update(download_bar, advance=1)
            
            if i % 2 == 0 and i < 100:
                multi.update(process_bar, advance=1)
                
            if i % 4 == 0 and i < 100:
                multi.update(upload_bar, advance=1)
```

**Features:**
- Multiple concurrent progress bars
- ETA calculation and rate display
- Customizable bar styles and colors
- Spinner animations for indeterminate progress
- Console-friendly output formatting

## External System Integrations

### APIClient

REST API client with retry logic and rate limiting.

```python
from library.infrastructure.interfaces.external import APIClient, APIConfig

# Configure API client
config = APIConfig(
    base_url="https://api.example.com/v1",
    timeout=30.0,
    max_retries=3,
    retry_backoff=2.0,
    rate_limit=100,  # requests per minute
    enable_caching=True,
    cache_ttl=300  # 5 minutes
)

client = APIClient(config)
await client.initialize()

# Make API requests
response = await client.get("/users/123", headers={"Authorization": "Bearer token"})
user_data = response.json()

# POST with automatic retry
result = await client.post(
    "/conversations",
    json={
        "user_id": "user_123",
        "message": "Hello",
        "timestamp": datetime.now().isoformat()
    }
)

# Batch requests
requests = [
    ("GET", "/users/1"),
    ("GET", "/users/2"),
    ("GET", "/users/3")
]
responses = await client.batch_request(requests)
```

**Features:**
- Automatic retry with exponential backoff
- Rate limiting and request throttling
- Response caching with TTL
- Request/response logging and metrics
- Connection pooling and keep-alive

### MessageQueue

Message queue interface for asynchronous communication.

```python
from library.infrastructure.interfaces.external import MessageQueue, QueueConfig

# Configure message queue
config = QueueConfig(
    broker_url="redis://localhost:6379/0",
    queue_name="nanobrain_tasks",
    max_retries=3,
    retry_delay=5.0,
    enable_dlq=True,  # Dead letter queue
    message_ttl=3600  # 1 hour
)

queue = MessageQueue(config)
await queue.initialize()

# Publish messages
await queue.publish({
    "task_type": "process_conversation",
    "conversation_id": "conv_123",
    "user_input": "Hello world",
    "priority": 1
})

# Subscribe to messages
async def message_handler(message):
    try:
        # Process message
        result = await process_task(message.data)
        await message.ack()  # Acknowledge successful processing
    except Exception as e:
        logger.error(f"Message processing failed: {e}")
        await message.nack()  # Negative acknowledgment for retry

await queue.subscribe(message_handler)
```

**Features:**
- Multiple broker support (Redis, RabbitMQ, AWS SQS)
- Message persistence and durability
- Dead letter queue for failed messages
- Priority queues and message routing
- Consumer groups and load balancing

### FileSystem

File system operations with cloud storage support.

```python
from library.infrastructure.interfaces.external import FileSystem, FileSystemConfig

# Configure file system
config = FileSystemConfig(
    storage_type="s3",  # or "local", "gcs", "azure"
    bucket_name="nanobrain-data",
    region="us-west-2",
    enable_encryption=True,
    enable_versioning=True
)

fs = FileSystem(config)
await fs.initialize()

# File operations
await fs.upload_file("conversations.json", "/data/conversations/2024/")
file_content = await fs.download_file("/data/conversations/2024/conversations.json")

# Directory operations
files = await fs.list_files("/data/conversations/", pattern="*.json")
await fs.create_directory("/data/backups/2024/")

# Streaming for large files
async with fs.stream_upload("/data/large_file.dat") as stream:
    async for chunk in large_data_generator():
        await stream.write(chunk)
```

**Features:**
- Multi-cloud storage support (AWS S3, Google Cloud, Azure)
- Streaming uploads/downloads for large files
- File versioning and metadata management
- Encryption at rest and in transit
- Automatic retry and error recovery

## Protocol Adapters

### WebSocketAdapter

WebSocket protocol adapter for real-time communication.

```python
from library.infrastructure.interfaces.protocols import WebSocketAdapter, WebSocketConfig

# Configure WebSocket adapter
config = WebSocketConfig(
    url="wss://api.example.com/ws",
    heartbeat_interval=30.0,
    max_reconnect_attempts=5,
    reconnect_delay=2.0,
    enable_compression=True
)

ws = WebSocketAdapter(config)
await ws.initialize()

# Send messages
await ws.send_message({
    "type": "chat_message",
    "content": "Hello world",
    "timestamp": datetime.now().isoformat()
})

# Receive messages
async def message_handler(message):
    if message["type"] == "chat_response":
        print(f"Received: {message['content']}")

await ws.subscribe(message_handler)

# Handle connection events
@ws.on_connect
async def on_connect():
    print("WebSocket connected")

@ws.on_disconnect
async def on_disconnect():
    print("WebSocket disconnected")
```

**Features:**
- Automatic reconnection with exponential backoff
- Heartbeat/ping-pong for connection health
- Message compression and binary support
- Event-driven message handling
- Connection state management

### HTTPAdapter

HTTP protocol adapter with advanced features.

```python
from library.infrastructure.interfaces.protocols import HTTPAdapter, HTTPConfig

# Configure HTTP adapter
config = HTTPConfig(
    base_url="https://api.example.com",
    timeout=30.0,
    max_connections=100,
    enable_http2=True,
    enable_compression=True,
    user_agent="NanoBrain/1.0"
)

http = HTTPAdapter(config)
await http.initialize()

# HTTP/2 multiplexing
async with http.session() as session:
    # Multiple concurrent requests over single connection
    tasks = [
        session.get("/endpoint1"),
        session.get("/endpoint2"),
        session.get("/endpoint3")
    ]
    responses = await asyncio.gather(*tasks)
```

**Features:**
- HTTP/2 support with multiplexing
- Connection pooling and reuse
- Automatic compression (gzip, brotli)
- Cookie and session management
- SSL/TLS verification and client certificates

## Configuration Management

### Database Configuration

```python
from library.infrastructure.interfaces.database import DatabaseConfig

# SQLite configuration
sqlite_config = DatabaseConfig(
    adapter_type="sqlite",
    connection_string="app.db",
    pool_size=5,
    enable_wal_mode=True,
    pragma_settings={
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "cache_size": -64000  # 64MB cache
    }
)

# PostgreSQL configuration
postgres_config = DatabaseConfig(
    adapter_type="postgresql",
    host="localhost",
    port=5432,
    database="nanobrain",
    user="app_user",
    password="secure_password",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30.0,
    enable_ssl=True,
    ssl_cert_path="/path/to/client.crt",
    ssl_key_path="/path/to/client.key"
)
```

### CLI Configuration

```python
from library.infrastructure.interfaces.cli import CLIConfig

config = CLIConfig(
    app_name="NanoBrain Assistant",
    version="1.0.0",
    prompt_style="🧠 >>> ",
    enable_colors=True,
    color_scheme="dark",
    enable_history=True,
    history_file="~/.nanobrain_history",
    history_size=1000,
    enable_autocomplete=True,
    completion_timeout=1.0
)
```

## Best Practices

### Connection Management

```python
# Use context managers for automatic cleanup
async with adapter.get_connection() as conn:
    result = await conn.execute_query("SELECT * FROM table")
    # Connection automatically returned to pool

# Handle connection failures gracefully
try:
    await adapter.execute_query(query)
except ConnectionError as e:
    logger.error(f"Database connection failed: {e}")
    # Implement fallback or retry logic
```

### Error Handling

```python
from library.infrastructure.interfaces.database import DatabaseError, ConnectionError

try:
    result = await adapter.execute_query(query, params)
except ConnectionError:
    # Handle connection issues
    await adapter.reconnect()
    result = await adapter.execute_query(query, params)
except DatabaseError as e:
    # Handle database-specific errors
    logger.error(f"Database error: {e}")
    raise
```

### Performance Optimization

```python
# Use connection pooling
config = DatabaseConfig(
    pool_size=20,  # Adjust based on concurrent load
    max_overflow=10,
    pool_timeout=30.0,
    pool_recycle=3600  # Recycle connections hourly
)

# Use prepared statements for repeated queries
prepared_query = await adapter.prepare(
    "SELECT * FROM conversations WHERE user_id = $1"
)

# Execute multiple times efficiently
for user_id in user_ids:
    result = await prepared_query.execute(user_id)
```

### Testing

```python
import pytest
from library.infrastructure.interfaces.database import SQLiteAdapter

@pytest.fixture
async def db_adapter():
    config = DatabaseConfig(connection_string=":memory:")
    adapter = SQLiteAdapter(config)
    await adapter.initialize()
    
    # Setup test schema
    await adapter.execute_query("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
    """)
    
    yield adapter
    await adapter.shutdown()

async def test_database_operations(db_adapter):
    # Test insert
    await db_adapter.execute_query(
        "INSERT INTO test_table (name) VALUES (?)",
        ("test_name",)
    )
    
    # Test select
    result = await db_adapter.execute_query(
        "SELECT * FROM test_table WHERE name = ?",
        ("test_name",)
    )
    
    assert len(result) == 1
    assert result[0]["name"] == "test_name"
```

## Migration Guide

### From Direct Database Usage

```python
# Before (direct SQLite usage)
import sqlite3

conn = sqlite3.connect("app.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM table")
results = cursor.fetchall()
conn.close()

# After (using adapter)
from library.infrastructure.interfaces.database import SQLiteAdapter

adapter = SQLiteAdapter("app.db")
await adapter.initialize()
results = await adapter.execute_query("SELECT * FROM table")
await adapter.shutdown()
```

### From Demo CLI Implementations

```python
# Before (demo-specific CLI)
class SimpleCLI:
    def __init__(self):
        # 200+ lines of CLI logic
        pass

# After (library CLI)
from library.infrastructure.interfaces.cli import InteractiveCLI

cli = InteractiveCLI(config)
await cli.initialize()
```

## Performance Considerations

### Database Performance

- **Connection Pooling**: Configure appropriate pool sizes based on concurrent load
- **Query Optimization**: Use prepared statements and proper indexing
- **Transaction Management**: Batch operations in transactions for better performance
- **Read Replicas**: Use read/write splitting for high-read workloads

### Network Performance

- **Connection Reuse**: Enable HTTP keep-alive and connection pooling
- **Compression**: Enable gzip/brotli compression for large payloads
- **Caching**: Implement response caching with appropriate TTL values
- **Rate Limiting**: Respect API rate limits to avoid throttling

### Memory Management

- **Streaming**: Use streaming for large file transfers and result sets
- **Connection Limits**: Configure maximum connection limits to prevent resource exhaustion
- **Cleanup**: Always properly close connections and clean up resources

---

*For more information, see the [main library documentation](../../README.md) or explore the [database examples](../../../examples/library/interfaces/database/).* 