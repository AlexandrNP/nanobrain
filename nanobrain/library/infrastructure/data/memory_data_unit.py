"""
In-memory data storage implementation.

Fast, in-memory data storage for temporary data and caching.
"""

import asyncio
from typing import Any, Dict, Optional
from .data_unit_base import DataUnitBase


class DataUnitMemory(DataUnitBase):
    """Fast, in-memory data storage for temporary data and caching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._data: Any = None
        self._access_count = 0
        
    async def get(self) -> Any:
        """Get data from memory."""
        async with self._lock:
            self._access_count += 1
            self.set_metadata('access_count', self._access_count)
            return self._data
            
    async def set(self, data: Any) -> None:
        """Set data in memory."""
        async with self._lock:
            self._data = data
            self.set_metadata('data_size_bytes', self._estimate_size(data))
            self.logger.debug(f"Data set in memory unit {self.name}")
            
    async def clear(self) -> None:
        """Clear data from memory."""
        async with self._lock:
            self._data = None
            self._access_count = 0
            self.metadata.clear()
            self.logger.debug(f"Memory unit {self.name} cleared")
            
    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            import sys
            return sys.getsizeof(data)
        except Exception:
            return 0
            
    async def append(self, data: Any) -> None:
        """Append data to existing data (if it's a list)."""
        async with self._lock:
            if self._data is None:
                self._data = []
            if isinstance(self._data, list):
                self._data.append(data)
                self.set_metadata('data_size_bytes', self._estimate_size(self._data))
            else:
                raise TypeError("Cannot append to non-list data")
                
    async def extend(self, data_list: list) -> None:
        """Extend existing list data."""
        async with self._lock:
            if self._data is None:
                self._data = []
            if isinstance(self._data, list):
                self._data.extend(data_list)
                self.set_metadata('data_size_bytes', self._estimate_size(self._data))
            else:
                raise TypeError("Cannot extend non-list data")
                
    async def get_size(self) -> int:
        """Get the current size of stored data."""
        async with self._lock:
            if isinstance(self._data, (list, dict, str)):
                return len(self._data)
            return 1 if self._data is not None else 0 