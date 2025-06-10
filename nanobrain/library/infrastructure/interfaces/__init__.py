"""
NanoBrain Library - Infrastructure Interfaces

External system interfaces and database adapters for the NanoBrain framework.

This module provides:
- Database interfaces and adapters (SQL, NoSQL)
- CLI and user interface components
- External system integrations (APIs, message queues)
- Protocol adapters and connectors

Separation of Concerns:
- Database-specific implementations are separated from core data abstractions
- Each interface type has its own submodule for organization
- Adapters follow common interface patterns for interchangeability
"""

# Database interfaces
from .database import (
    DatabaseInterface,
    SQLInterface,
    NoSQLInterface,
    MySQLAdapter,
    PostgreSQLAdapter,
    SQLiteAdapter,
    MongoDBAdapter
)

# CLI interfaces
from .cli import (
    BaseCLI,
    CLIConfig,
    InteractiveCLI,
    InteractiveCLIConfig,
    CLIStep,
    CLIStepConfig,
    CommandProcessor,
    CLICommand,
    CLIContext,
    ResponseFormatter,
    FormatterConfig,
    ProgressIndicator,
    ProgressConfig
)

# External system interfaces
from .external import (
    APIInterface,
    MessageQueueInterface,
    FileSystemInterface,
    WebSocketInterface
)

__all__ = [
    # Database interfaces
    'DatabaseInterface',
    'SQLInterface', 
    'NoSQLInterface',
    'MySQLAdapter',
    'PostgreSQLAdapter',
    'SQLiteAdapter',
    'MongoDBAdapter',
    
    # CLI interfaces
    'BaseCLI',
    'CLIConfig',
    'InteractiveCLI',
    'InteractiveCLIConfig',
    'CLIStep',
    'CLIStepConfig',
    'CommandProcessor',
    'CLICommand',
    'CLIContext',
    'ResponseFormatter',
    'FormatterConfig',
    'ProgressIndicator',
    'ProgressConfig',
    
    # External interfaces
    'APIInterface',
    'MessageQueueInterface', 
    'FileSystemInterface',
    'WebSocketInterface'
] 