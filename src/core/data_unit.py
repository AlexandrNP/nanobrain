"""
Data Unit System for NanoBrain Framework

Provides data interfaces and ingestion capabilities for Steps.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class DataUnitType(Enum):
    """Types of data units."""
    MEMORY = "memory"
    FILE = "file"
    STRING = "string"
    STREAM = "stream"
    DATABASE = "database"


class DataUnitConfig(BaseModel):
    """Configuration for data units."""
    model_config = ConfigDict(use_enum_values=True)
    
    data_type: DataUnitType = DataUnitType.MEMORY
    persistent: bool = False
    cache_size: int = Field(default=1000, ge=1)
    file_path: Optional[str] = None
    encoding: str = "utf-8"
    initial_value: Optional[str] = None  # For string data units


class DataUnitBase(ABC):
    """
    Base class for data units that handle data storage and retrieval.
    
    Biological analogy: Synaptic vesicles storing neurotransmitters.
    Justification: Like how synaptic vesicles store and release neurotransmitters
    for neural communication, data units store and provide data for step communication.
    """
    
    def __init__(self, config: Optional[DataUnitConfig] = None, **kwargs):
        self.config = config or DataUnitConfig()
        self.name = kwargs.get('name', self.__class__.__name__)
        self._data: Any = None
        self._metadata: Dict[str, Any] = {}
        self._is_initialized = False
        self._lock = asyncio.Lock()
        
    @abstractmethod
    async def get(self) -> Any:
        """Get data from the unit."""
        pass
    
    @abstractmethod
    async def set(self, data: Any) -> None:
        """Set data in the unit."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear data from the unit."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the data unit."""
        if not self._is_initialized:
            self._is_initialized = True
            logger.debug(f"DataUnit {self.name} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the data unit."""
        await self.clear()
        self._is_initialized = False
        logger.debug(f"DataUnit {self.name} shutdown")
    
    @property
    def is_initialized(self) -> bool:
        """Check if data unit is initialized."""
        return self._is_initialized
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata."""
        return self._metadata.copy()
    
    async def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata."""
        async with self._lock:
            self._metadata[key] = value
    
    async def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    async def read(self) -> Any:
        """Read data from the unit (alias for get)."""
        return await self.get()
    
    async def write(self, data: Any) -> None:
        """Write data to the unit (alias for set)."""
        await self.set(data)


class DataUnitMemory(DataUnitBase):
    """
    In-memory data unit for fast access.
    """
    
    def __init__(self, config: Optional[DataUnitConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self._data = None
        
    async def get(self) -> Any:
        """Get data from memory."""
        if not self.is_initialized:
            await self.initialize()
        return self._data
    
    async def set(self, data: Any) -> None:
        """Set data in memory."""
        if not self.is_initialized:
            await self.initialize()
        async with self._lock:
            self._data = data
            self._metadata['last_updated'] = time.time()
    
    async def clear(self) -> None:
        """Clear data from memory."""
        async with self._lock:
            self._data = None
            self._metadata.clear()


class DataUnitFile(DataUnitBase):
    """
    File-based data unit for persistent storage.
    """
    
    def __init__(self, file_path: str, config: Optional[DataUnitConfig] = None, **kwargs):
        config = config or DataUnitConfig(data_type=DataUnitType.FILE, persistent=True)
        config.file_path = file_path
        super().__init__(config, **kwargs)
        self.file_path = Path(file_path)
        
    async def get(self) -> Any:
        """Get data from file."""
        if not self.is_initialized:
            await self.initialize()
            
        if not self.file_path.exists():
            return None
            
        try:
            async with self._lock:
                # Read file content
                content = self.file_path.read_text(encoding=self.config.encoding)
                
                # Try to parse as JSON if possible
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
                    
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            raise
    
    async def set(self, data: Any) -> None:
        """Set data to file."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            async with self._lock:
                # Ensure parent directory exists
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert data to string
                if isinstance(data, (dict, list)):
                    content = json.dumps(data, indent=2)
                else:
                    content = str(data)
                
                # Write to file
                self.file_path.write_text(content, encoding=self.config.encoding)
                self._metadata['last_updated'] = time.time()
                
        except Exception as e:
            logger.error(f"Error writing file {self.file_path}: {e}")
            raise
    
    async def clear(self) -> None:
        """Clear file data."""
        async with self._lock:
            if self.file_path.exists():
                self.file_path.unlink()
            self._metadata.clear()


class DataUnitString(DataUnitBase):
    """
    String-based data unit for text data.
    """
    
    def __init__(self, initial_value: str = "", config: Optional[DataUnitConfig] = None, **kwargs):
        config = config or DataUnitConfig(data_type=DataUnitType.STRING)
        super().__init__(config, **kwargs)
        self._data = initial_value
        
    async def get(self) -> str:
        """Get string data."""
        if not self.is_initialized:
            await self.initialize()
        return self._data or ""
    
    async def set(self, data: Any) -> None:
        """Set string data."""
        if not self.is_initialized:
            await self.initialize()
        async with self._lock:
            self._data = str(data) if data is not None else ""
            self._metadata['last_updated'] = time.time()
    
    async def append(self, data: str) -> None:
        """Append to string data."""
        async with self._lock:
            current = await self.get()
            await self.set(current + str(data))
    
    async def clear(self) -> None:
        """Clear string data."""
        async with self._lock:
            self._data = ""
            self._metadata.clear()


class DataUnitStream(DataUnitBase):
    """
    Stream-based data unit for continuous data flow.
    """
    
    def __init__(self, config: Optional[DataUnitConfig] = None, **kwargs):
        config = config or DataUnitConfig(data_type=DataUnitType.STREAM)
        super().__init__(config, **kwargs)
        self._queue: Optional[asyncio.Queue] = None
        self._subscribers: List[asyncio.Queue] = []
        
    async def initialize(self) -> None:
        """Initialize the stream."""
        if not self._is_initialized:
            self._queue = asyncio.Queue(maxsize=self.config.cache_size)
            await super().initialize()
    
    async def get(self) -> Any:
        """Get next item from stream."""
        if not self.is_initialized:
            await self.initialize()
        return await self._queue.get()
    
    async def set(self, data: Any) -> None:
        """Add data to stream."""
        if not self.is_initialized:
            await self.initialize()
            
        # Add to main queue
        try:
            await self._queue.put(data)
        except asyncio.QueueFull:
            # Remove oldest item and add new one
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            await self._queue.put(data)
        
        # Notify subscribers
        for subscriber_queue in self._subscribers:
            try:
                await subscriber_queue.put(data)
            except asyncio.QueueFull:
                # Skip if subscriber queue is full
                pass
    
    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to stream updates."""
        if not self.is_initialized:
            await self.initialize()
        subscriber_queue = asyncio.Queue(maxsize=self.config.cache_size)
        self._subscribers.append(subscriber_queue)
        return subscriber_queue
    
    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from stream updates."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
    
    async def clear(self) -> None:
        """Clear stream data."""
        if self._queue:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        # Clear subscriber queues
        for subscriber_queue in self._subscribers:
            while not subscriber_queue.empty():
                try:
                    subscriber_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        self._metadata.clear()


def create_data_unit(config: Union[Dict[str, Any], DataUnitConfig]) -> DataUnitBase:
    """
    Factory function to create data units.
    
    Args:
        config: Data unit configuration (dict or DataUnitConfig)
        
    Returns:
        Configured data unit instance
    """
    if isinstance(config, dict):
        config = DataUnitConfig(**config)
    
    # Handle both enum and string values (due to use_enum_values=True)
    data_type = config.data_type
    if isinstance(data_type, str):
        data_type = DataUnitType(data_type)
    
    if data_type == DataUnitType.MEMORY:
        return DataUnitMemory(config)
    elif data_type == DataUnitType.FILE:
        # DataUnitFile requires file_path parameter
        file_path = config.file_path or "/tmp/default_file.txt"
        return DataUnitFile(file_path, config)
    elif data_type == DataUnitType.STRING:
        # DataUnitString can take initial_value parameter
        initial_value = getattr(config, 'initial_value', "")
        return DataUnitString(initial_value, config)
    elif data_type == DataUnitType.STREAM:
        return DataUnitStream(config)
    else:
        raise ValueError(f"Unknown data unit type: {data_type}") 