"""
In-memory data storage implementation.

Fast, in-memory data storage for temporary data and caching.
"""

import asyncio
import sys
import json
from typing import Any, Dict, Optional
from .data_unit_base import DataUnitBase


class DataUnitMemory(DataUnitBase):
    """Fast, in-memory data storage for temporary data and caching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._data: Any = None
        self._access_count = 0
        
        try:
            # Get global configuration for data unit tracking
            from nanobrain.config import get_config_manager
            config_manager = get_config_manager()
            logging_config = config_manager.get_config_dict().get('logging', {})
            file_config = logging_config.get('file', {})
            
            # Configure tracking settings
            self._track_values = file_config.get('track_data_unit_values', True)
            self._max_value_size = file_config.get('max_data_value_size', 10000)
        except:
            # Default to tracking with reasonable limits if config not available
            self._track_values = True
            self._max_value_size = 10000
            
    async def get(self) -> Any:
        """Get data from memory."""
        async with self._lock:
            self._access_count += 1
            self.set_metadata('access_count', self._access_count)
            
            # Log data access with value (if tracking enabled)
            if self._track_values:
                serialized_data = self._serialize_data(self._data)
                self.logger.log_data_unit_operation(
                    operation='get',
                    data_unit_name=self.name,
                    data=serialized_data,
                    metadata={
                        'access_count': self._access_count,
                        'data_type': type(self._data).__name__ if self._data is not None else 'None',
                        'data_size': self._estimate_size(self._data)
                    }
                )
            else:
                # Log without detailed data values
                self.logger.debug(f"Data retrieved from {self.name}", 
                                 operation='get',
                                 access_count=self._access_count)
                
            return self._data
            
    async def set(self, data: Any) -> None:
        """Set data in memory."""
        async with self._lock:
            self._data = data
            data_size = self._estimate_size(data)
            self.set_metadata('data_size_bytes', data_size)
            
            # Log data operations with value (if tracking enabled)
            if self._track_values:
                serialized_data = self._serialize_data(data)
                self.logger.log_data_unit_operation(
                    operation='set',
                    data_unit_name=self.name,
                    data=serialized_data,
                    metadata={
                        'data_type': type(data).__name__ if data is not None else 'None',
                        'data_size': data_size
                    }
                )
            else:
                # Log without detailed data values
                self.logger.debug(f"Data set in memory unit {self.name}",
                                 operation='set',
                                 data_type=type(data).__name__ if data is not None else 'None',
                                 data_size=data_size)
            
    async def clear(self) -> None:
        """Clear data from memory."""
        async with self._lock:
            had_data = self._data is not None
            
            # Log before clearing
            if had_data and self._track_values:
                self.logger.log_data_unit_operation(
                    operation='clear',
                    data_unit_name=self.name,
                    metadata={
                        'had_data': had_data,
                        'previous_data_type': type(self._data).__name__ if self._data is not None else 'None'
                    }
                )
            
            # Clear the data
            self._data = None
            self._access_count = 0
            self.metadata.clear()
            self.logger.debug(f"Memory unit {self.name} cleared")
            
    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            return sys.getsizeof(data)
        except Exception:
            return 0
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for logging, respecting size constraints."""
        if data is None:
            return None
            
        try:
            # Handle common types
            if isinstance(data, (str, int, float, bool, type(None))):
                if isinstance(data, str) and len(data) > self._max_value_size:
                    return data[:self._max_value_size] + "... [truncated]"
                return data
                
            # Handle lists
            elif isinstance(data, list):
                if len(str(data)) > self._max_value_size:
                    if len(data) > 10:
                        return f"[{len(data)} items, first 3: {data[:3]}]"
                    return f"{data[:5]}... [truncated]"
                return data
                
            # Handle dictionaries
            elif isinstance(data, dict):
                if len(str(data)) > self._max_value_size:
                    keys = list(data.keys())
                    if len(keys) > 10:
                        sample_keys = keys[:3]
                        sample_dict = {k: data[k] for k in sample_keys}
                        return f"{sample_dict}... [truncated, {len(keys)} keys total]"
                    return f"{data}... [truncated]"
                return data
                
            # Try JSON serialization for custom objects
            else:
                try:
                    json_str = json.dumps(data)
                    if len(json_str) > self._max_value_size:
                        return f"[Object of type {type(data).__name__}, size: {len(json_str)} chars]"
                    return data
                except:
                    return f"[Non-serializable object of type {type(data).__name__}]"
                    
        except Exception as e:
            return f"[Error serializing data: {str(e)}]"
            
    async def append(self, data: Any) -> None:
        """Append data to existing data (if it's a list)."""
        async with self._lock:
            if self._data is None:
                self._data = []
            if isinstance(self._data, list):
                self._data.append(data)
                data_size = self._estimate_size(self._data)
                self.set_metadata('data_size_bytes', data_size)
                
                # Log the append operation with detailed tracking
                if self._track_values:
                    serialized_data = self._serialize_data(data)
                    self.logger.log_data_unit_operation(
                        operation='append',
                        data_unit_name=self.name,
                        data=serialized_data,
                        metadata={
                            'list_size_after': len(self._data),
                            'appended_type': type(data).__name__ if data is not None else 'None',
                            'data_size_after': data_size
                        }
                    )
                
            else:
                error_msg = "Cannot append to non-list data"
                self.logger.error(error_msg, operation='append', data_unit_name=self.name)
                raise TypeError(error_msg)
                
    async def extend(self, data_list: list) -> None:
        """Extend existing list data."""
        async with self._lock:
            if self._data is None:
                self._data = []
            if isinstance(self._data, list):
                self._data.extend(data_list)
                data_size = self._estimate_size(self._data)
                self.set_metadata('data_size_bytes', data_size)
                
                # Log the extend operation with detailed tracking
                if self._track_values:
                    serialized_data = self._serialize_data(data_list)
                    self.logger.log_data_unit_operation(
                        operation='extend',
                        data_unit_name=self.name,
                        data=serialized_data,
                        metadata={
                            'items_added': len(data_list),
                            'list_size_after': len(self._data),
                            'data_size_after': data_size
                        }
                    )
                
            else:
                error_msg = "Cannot extend non-list data"
                self.logger.error(error_msg, operation='extend', data_unit_name=self.name)
                raise TypeError(error_msg)
                
    async def get_size(self) -> int:
        """Get the current size of stored data."""
        async with self._lock:
            if isinstance(self._data, (list, dict, str)):
                return len(self._data)
            return 1 if self._data is not None else 0 