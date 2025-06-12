"""
Abstract base classes and interfaces for data units.

This module provides the foundation for all data units in the NanoBrain framework,
defining consistent interfaces for data operations.
"""

import asyncio
import json
import inspect
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
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
        self.logger = get_logger(f"{self.name}", category="data_units")
        
        # Configure value tracking
        try:
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
            
        # Track history of operations
        self._operation_history: List[Dict[str, Any]] = []
        self._max_history_items = self.config.get('max_history_items', 100)
        
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
                self._log_operation('initialize', None, {
                    'initialized': True,
                    'config': self.config
                })
                self.logger.info(f"Data unit {self.name} initialized")
                
    async def shutdown(self) -> None:
        """Shutdown the data unit."""
        async with self._lock:
            if self.initialized:
                await self._shutdown_impl()
                self.initialized = False
                self._log_operation('shutdown', None, {
                    'initialized': False
                })
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
            'operations_count': len(self._operation_history),
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
            
    def _log_operation(self, operation: str, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a data operation with value tracking."""
        # Create operation record
        timestamp = datetime.now().isoformat()
        op_record = {
            'operation': operation,
            'timestamp': timestamp,
            'caller': self._get_caller_info(),
            'metadata': metadata or {}
        }
        
        # Add to history (capped)
        self._operation_history.append(op_record)
        if len(self._operation_history) > self._max_history_items:
            self._operation_history = self._operation_history[-self._max_history_items:]
        
        # Log the operation
        if self._track_values:
            serialized_data = self._serialize_for_logging(data)
            self.logger.log_data_unit_operation(
                operation=operation,
                data_unit_name=self.name,
                data=serialized_data,
                metadata=metadata
            )
        else:
            # Log without data value
            self.logger.debug(f"Data unit {self.name}: {operation}",
                             operation=operation,
                             **metadata if metadata else {})
                             
    def _get_caller_info(self) -> Dict[str, str]:
        """Get information about the caller for better traceability."""
        try:
            stack = inspect.stack()
            # Look for the first caller outside this class
            for frame in stack[1:]:
                if frame.function != '_log_operation' and 'data_unit_base.py' not in frame.filename:
                    return {
                        'function': frame.function,
                        'file': frame.filename,
                        'line': frame.lineno
                    }
            return {'function': 'unknown', 'file': 'unknown', 'line': 0}
        except:
            return {'function': 'unknown', 'file': 'unknown', 'line': 0}
            
    def _serialize_for_logging(self, data: Any) -> Any:
        """Serialize data for logging with size constraints."""
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
            
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get the history of operations performed on this data unit."""
        return self._operation_history.copy() 