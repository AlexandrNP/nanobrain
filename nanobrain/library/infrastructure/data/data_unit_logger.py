"""
Enhanced logging for data units.

This module provides specialized logging capabilities for data units,
with support for detailed value tracking and operation history.
"""

import os
import json
import inspect
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path

from nanobrain.core.logging_system import get_logger


class DataUnitLogger:
    """Enhanced logger for data units with value tracking and session organization."""
    
    def __init__(self, data_unit_name: str, data_unit_type: str = "memory"):
        """
        Initialize a DataUnitLogger.
        
        Args:
            data_unit_name: Name of the data unit
            data_unit_type: Type of data unit (memory, file, etc.)
        """
        self.data_unit_name = data_unit_name
        self.data_unit_type = data_unit_type
        
        # Create nanobrain logger - special category for data units
        sub_category = data_unit_type if data_unit_type else "generic"
        self.logger = get_logger(f"{data_unit_name}", f"data_units.{sub_category}")
        
        # Find log directory for additional files
        self.log_dir = self._find_log_directory()
        
        # Configure value tracking
        try:
            from nanobrain.core.config import get_config_manager
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
        
        # Additional specialized log files
        self.values_log_file = None
        self.history_log_file = None
        
        if self.log_dir and self._track_values:
            # Create subdirectory for this data unit
            unit_dir = self.log_dir / f"{data_unit_name}"
            unit_dir.mkdir(parents=True, exist_ok=True)
            
            # Create specialized log files
            self.values_log_file = unit_dir / "values.jsonl"
            self.history_log_file = unit_dir / "history.jsonl"
    
    def _find_log_directory(self) -> Optional[Path]:
        """Find the log directory based on the logger's configuration."""
        try:
            # Get main log directory from configuration
            from nanobrain.core.config import get_config_manager
            config_manager = get_config_manager()
            logging_config = config_manager.get_config_dict().get('logging', {})
            file_config = logging_config.get('file', {})
            
            base_dir = file_config.get('base_directory', 'logs')
            use_session_dirs = file_config.get('use_session_directories', True)
            
            if use_session_dirs:
                # Find most recent session directory
                base_path = Path(base_dir)
                if base_path.exists():
                    session_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("session_")]
                    if session_dirs:
                        # Sort by creation time, newest first
                        latest_session = sorted(session_dirs, key=lambda d: d.stat().st_ctime, reverse=True)[0]
                        data_units_dir = latest_session / "data"
                        data_units_dir.mkdir(parents=True, exist_ok=True)
                        return data_units_dir
            
            # If session directories not used or not found, use direct path
            data_units_dir = Path(base_dir) / "data"
            data_units_dir.mkdir(parents=True, exist_ok=True)
            return data_units_dir
            
        except Exception as e:
            # If any error occurs, log it but continue without value tracking files
            self.logger.warning(f"Could not setup value tracking files: {e}")
            return None
    
    def log_operation(self, operation: str, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a data unit operation with comprehensive tracking.
        
        Args:
            operation: Operation name ('get', 'set', 'clear', etc.)
            data: Data involved in the operation (might be truncated)
            metadata: Additional metadata for the operation
        """
        # Log via standard logger
        meta = metadata or {}
        self.logger.log_data_unit_operation(operation, self.data_unit_name, data, meta)
        
        # Only proceed with value logging if tracking enabled and we have a log directory
        if not self._track_values or not self.log_dir:
            return
            
        # Create record for specialized log files
        timestamp = datetime.now().isoformat()
        caller = self._get_caller_info()
        
        # Basic record for history log
        history_record = {
            'timestamp': timestamp,
            'operation': operation,
            'caller': caller,
            'metadata': meta,
            'data_type': type(data).__name__ if data is not None else 'None'
        }
        
        # Extended record for values log
        values_record = history_record.copy()
        if data is not None:
            # Add serialized data to values record
            try:
                serialized_data = self._serialize_for_logging(data)
                values_record['data'] = serialized_data
            except Exception as e:
                values_record['data_error'] = str(e)
        
        # Write to specialized log files
        try:
            # Append to history log
            if self.history_log_file:
                with open(self.history_log_file, 'a') as f:
                    f.write(json.dumps(history_record) + '\n')
                    
            # Append to values log
            if self.values_log_file:
                with open(self.values_log_file, 'a') as f:
                    f.write(json.dumps(values_record) + '\n')
                    
        except Exception as e:
            # Log error but continue without crashing
            self.logger.error(f"Failed to write to data unit log files: {e}")
    
    def _get_caller_info(self) -> Dict[str, str]:
        """Get information about the caller for better traceability."""
        try:
            stack = inspect.stack()
            # Look for the first caller outside this logger
            for frame in stack[1:]:
                if frame.function != 'log_operation' and 'data_unit_logger.py' not in frame.filename:
                    return {
                        'function': frame.function,
                        'file': os.path.basename(frame.filename),
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
                        sample = data[:3]
                        return f"[{len(data)} items, first 3: {sample}]"
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
    
    def create_session_summary(self, data_unit_info: Dict[str, Any]) -> None:
        """Create a summary of the data unit's activity for the current session."""
        if not self._track_values or not self.log_dir:
            return
            
        unit_dir = self.log_dir / f"{self.data_unit_name}"
        if not unit_dir.exists():
            unit_dir.mkdir(parents=True, exist_ok=True)
            
        # Create summary file
        try:
            summary_file = unit_dir / "summary.json"
            summary = {
                "name": self.data_unit_name,
                "type": self.data_unit_type,
                "timestamp": datetime.now().isoformat(),
                "info": data_unit_info
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write data unit summary: {e}") 