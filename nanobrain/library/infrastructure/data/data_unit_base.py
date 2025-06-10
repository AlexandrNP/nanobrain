"""
Abstract base classes and interfaces for data units.

This module provides the foundation for all data units in the NanoBrain framework,
defining consistent interfaces for data operations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
from nanobrain.core.logging_system import get_logger


class DataUnitBase(ABC):
    """Abstract base class for all data units."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.config.get('name', 'unnamed_data_unit')
        self.data_type = self.config.get('data_type', 'generic')
        self.metadata = {}
        self.initialized = False
        self._lock = asyncio.Lock()
        self.logger = get_logger(f"data_unit.{self.name}")
        
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
        async with self._lock:
            if not self.initialized:
                await self._initialize_impl()
                self.initialized = True
                self.logger.info(f"Data unit {self.name} initialized")
                
    async def shutdown(self) -> None:
        """Shutdown the data unit."""
        async with self._lock:
            if self.initialized:
                await self._shutdown_impl()
                self.initialized = False
                self.logger.info(f"Data unit {self.name} shutdown")
                
    async def _initialize_impl(self) -> None:
        """Implementation-specific initialization."""
        pass
        
    async def _shutdown_impl(self) -> None:
        """Implementation-specific shutdown."""
        pass
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the data unit."""
        return {
            'name': self.name,
            'data_type': self.data_type,
            'initialized': self.initialized,
            'last_updated': self.metadata.get('last_updated'),
            **self.metadata
        }
        
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the data unit."""
        self.metadata[key] = value
        self.metadata['last_updated'] = datetime.now().isoformat()
        
    async def exists(self) -> bool:
        """Check if data exists in the unit."""
        try:
            data = await self.get()
            return data is not None
        except Exception:
            return False 