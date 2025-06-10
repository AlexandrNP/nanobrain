"""
NanoBrain Library - External System Interfaces

External system interfaces for the NanoBrain framework.

This module provides:
- API interfaces and adapters
- Message queue interfaces
- File system interfaces
- WebSocket interfaces

These are minimal stubs for now to allow the library to import successfully.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

class APIInterface(ABC):
    """Abstract API interface."""
    
    @abstractmethod
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Make a GET request."""
        pass
    
    @abstractmethod
    async def post(self, endpoint: str, data: Optional[Dict] = None) -> Any:
        """Make a POST request."""
        pass

class MessageQueueInterface(ABC):
    """Abstract message queue interface."""
    
    @abstractmethod
    async def publish(self, topic: str, message: Any) -> bool:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, callback) -> bool:
        """Subscribe to a topic."""
        pass

class FileSystemInterface(ABC):
    """Abstract file system interface."""
    
    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read a file."""
        pass
    
    @abstractmethod
    async def write_file(self, path: str, content: str) -> bool:
        """Write a file."""
        pass

class WebSocketInterface(ABC):
    """Abstract WebSocket interface."""
    
    @abstractmethod
    async def connect(self, url: str) -> bool:
        """Connect to WebSocket."""
        pass
    
    @abstractmethod
    async def send(self, message: Any) -> bool:
        """Send a message."""
        pass
    
    @abstractmethod
    async def receive(self) -> Any:
        """Receive a message."""
        pass

__all__ = [
    'APIInterface',
    'MessageQueueInterface',
    'FileSystemInterface',
    'WebSocketInterface'
] 