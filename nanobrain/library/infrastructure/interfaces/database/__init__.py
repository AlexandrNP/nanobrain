"""
NanoBrain Library - Database Interfaces

Database adapters and interfaces for the NanoBrain framework.

This module provides:
- Abstract database interfaces
- SQL database adapters (MySQL, PostgreSQL, SQLite)
- NoSQL database adapters (MongoDB, etc.)
- Connection management and pooling
- Query builders and ORM integration

Separation of Concerns:
- Abstract interfaces define contracts for database operations
- Concrete adapters implement database-specific functionality
- Data units in library.infrastructure.data use these interfaces
- No direct database dependencies in core data abstractions
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

# Base interfaces - minimal stubs for now
@dataclass
class DatabaseConfig:
    """Base database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "nanobrain"
    username: str = "user"
    password: str = "password"

class ConnectionPool:
    """Database connection pool stub."""
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    async def get_connection(self):
        """Get a connection from the pool."""
        pass
    
    async def release_connection(self, connection):
        """Release a connection back to the pool."""
        pass

class DatabaseInterface(ABC):
    """Abstract database interface."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the database."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the database."""
        pass
    
    @abstractmethod
    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a query."""
        pass

# SQL interfaces - stubs
class SQLQuery:
    """SQL query representation."""
    def __init__(self, query: str, params: Optional[Dict] = None):
        self.query = query
        self.params = params or {}

class SQLResult:
    """SQL query result."""
    def __init__(self, rows: List[Dict], affected_rows: int = 0):
        self.rows = rows
        self.affected_rows = affected_rows

class SQLInterface(DatabaseInterface):
    """SQL database interface."""
    pass

# NoSQL interfaces - stubs
class NoSQLQuery:
    """NoSQL query representation."""
    def __init__(self, collection: str, query: Dict, options: Optional[Dict] = None):
        self.collection = collection
        self.query = query
        self.options = options or {}

class NoSQLResult:
    """NoSQL query result."""
    def __init__(self, documents: List[Dict], count: int = 0):
        self.documents = documents
        self.count = count

class NoSQLInterface(DatabaseInterface):
    """NoSQL database interface."""
    pass

# Adapter stubs
class MySQLAdapter(SQLInterface):
    """MySQL database adapter stub."""
    pass

class MySQLConfig(DatabaseConfig):
    """MySQL configuration stub."""
    port: int = 3306

class PostgreSQLAdapter(SQLInterface):
    """PostgreSQL database adapter stub."""
    pass

class PostgreSQLConfig(DatabaseConfig):
    """PostgreSQL configuration stub."""
    port: int = 5432

class SQLiteAdapter(SQLInterface):
    """SQLite database adapter stub."""
    pass

class SQLiteConfig(DatabaseConfig):
    """SQLite configuration stub."""
    database: str = "nanobrain.db"

class MongoDBAdapter(NoSQLInterface):
    """MongoDB database adapter stub."""
    pass

class MongoDBConfig(DatabaseConfig):
    """MongoDB configuration stub."""
    port: int = 27017

# Utility stubs
class QueryBuilder:
    """Base query builder stub."""
    pass

class SQLQueryBuilder(QueryBuilder):
    """SQL query builder stub."""
    pass

class NoSQLQueryBuilder(QueryBuilder):
    """NoSQL query builder stub."""
    pass

class ConnectionManager:
    """Connection manager stub."""
    pass

class ConnectionConfig:
    """Connection configuration stub."""
    pass

__all__ = [
    # Base interfaces
    'DatabaseInterface',
    'DatabaseConfig',
    'ConnectionPool',
    
    # SQL interfaces
    'SQLInterface',
    'SQLQuery',
    'SQLResult',
    
    # NoSQL interfaces
    'NoSQLInterface',
    'NoSQLQuery', 
    'NoSQLResult',
    
    # SQL adapters
    'MySQLAdapter',
    'MySQLConfig',
    'PostgreSQLAdapter',
    'PostgreSQLConfig',
    'SQLiteAdapter',
    'SQLiteConfig',
    
    # NoSQL adapters
    'MongoDBAdapter',
    'MongoDBConfig',
    
    # Utilities
    'QueryBuilder',
    'SQLQueryBuilder',
    'NoSQLQueryBuilder',
    'ConnectionManager',
    'ConnectionConfig'
] 