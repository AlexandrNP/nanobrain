"""
String-based data storage implementation.

Specialized string handling with append operations.
"""

import asyncio
from typing import Any, Dict, Optional
from .data_unit_base import DataUnitBase


class DataUnitString(DataUnitBase):
    """Specialized string handling with append operations."""
    
    def __init__(self, initial_value: str = "", config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._data: str = initial_value
        self._append_count = 0
        
    async def get(self) -> str:
        """Get string data."""
        async with self._lock:
            return self._data
            
    async def set(self, data: Any) -> None:
        """Set string data."""
        async with self._lock:
            self._data = str(data) if data is not None else ""
            self.set_metadata('length', len(self._data))
            self.set_metadata('append_count', self._append_count)
            self.logger.debug(f"String data set in unit {self.name}")
            
    async def clear(self) -> None:
        """Clear string data."""
        async with self._lock:
            self._data = ""
            self._append_count = 0
            self.metadata.clear()
            self.logger.debug(f"String unit {self.name} cleared")
            
    async def append(self, text: str) -> None:
        """Append text to the string."""
        async with self._lock:
            self._data += str(text)
            self._append_count += 1
            self.set_metadata('length', len(self._data))
            self.set_metadata('append_count', self._append_count)
            self.logger.debug(f"Text appended to string unit {self.name}")
            
    async def prepend(self, text: str) -> None:
        """Prepend text to the string."""
        async with self._lock:
            self._data = str(text) + self._data
            self._append_count += 1
            self.set_metadata('length', len(self._data))
            self.set_metadata('append_count', self._append_count)
            self.logger.debug(f"Text prepended to string unit {self.name}")
            
    async def append_line(self, text: str) -> None:
        """Append text with a newline."""
        await self.append(str(text) + "\n")
        
    async def get_length(self) -> int:
        """Get the length of the string."""
        async with self._lock:
            return len(self._data)
            
    async def get_lines(self) -> list:
        """Get string as list of lines."""
        async with self._lock:
            return self._data.splitlines()
            
    async def get_line_count(self) -> int:
        """Get number of lines in the string."""
        lines = await self.get_lines()
        return len(lines)
        
    async def truncate(self, max_length: int) -> None:
        """Truncate string to maximum length."""
        async with self._lock:
            if len(self._data) > max_length:
                self._data = self._data[:max_length]
                self.set_metadata('length', len(self._data))
                self.set_metadata('truncated', True)
                self.logger.debug(f"String unit {self.name} truncated to {max_length} characters")
                
    async def replace(self, old: str, new: str, count: int = -1) -> int:
        """Replace occurrences of old with new."""
        async with self._lock:
            original_data = self._data
            self._data = self._data.replace(old, new, count)
            replacements = original_data.count(old) if count == -1 else min(count, original_data.count(old))
            self.set_metadata('length', len(self._data))
            self.logger.debug(f"Replaced {replacements} occurrences in string unit {self.name}")
            return replacements
            
    async def contains(self, substring: str) -> bool:
        """Check if string contains substring."""
        async with self._lock:
            return substring in self._data
            
    async def startswith(self, prefix: str) -> bool:
        """Check if string starts with prefix."""
        async with self._lock:
            return self._data.startswith(prefix)
            
    async def endswith(self, suffix: str) -> bool:
        """Check if string ends with suffix."""
        async with self._lock:
            return self._data.endswith(suffix) 