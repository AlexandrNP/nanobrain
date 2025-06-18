"""
Data Infrastructure Components

This module provides core data abstractions and implementations for the NanoBrain framework.
It separates data management concerns from database-specific implementations.
"""

from .data_unit_base import DataUnitBase
from .memory_data_unit import DataUnitMemory
from .stream_data_unit import DataUnitStream
from .string_data_unit import DataUnitString
from .conversation_history import ConversationHistoryUnit
from .session_manager import SessionManager
from .export_manager import ExportManager

# Optional file-based storage (requires aiofiles)
try:
    from .file_data_unit import DataUnitFile
    _file_data_available = True
except ImportError:
    DataUnitFile = None
    _file_data_available = False

__all__ = [
    'DataUnitBase',
    'DataUnitMemory', 
    'DataUnitStream',
    'DataUnitString',
    'ConversationHistoryUnit',
    'SessionManager',
    'ExportManager'
]

# Add DataUnitFile to exports only if available
if _file_data_available:
    __all__.append('DataUnitFile') 