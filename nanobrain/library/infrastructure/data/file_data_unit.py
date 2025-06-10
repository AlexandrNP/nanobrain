"""
File-based persistent storage implementation.

Persistent file-based storage with automatic serialization.
"""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import Any, Dict, Optional
from .data_unit_base import DataUnitBase


class DataUnitFile(DataUnitBase):
    """Persistent file-based storage with automatic serialization."""
    
    def __init__(self, file_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.file_path = Path(file_path)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.backup_enabled = self.config.get('backup_enabled', True)
        self._file_lock = asyncio.Lock()
        
    async def _initialize_impl(self) -> None:
        """Initialize file storage."""
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty file if it doesn't exist
        if not self.file_path.exists():
            await self._write_file({})
            
    async def get(self) -> Any:
        """Get data from file."""
        async with self._file_lock:
            try:
                async with aiofiles.open(self.file_path, 'r', encoding=self.encoding) as f:
                    content = await f.read()
                    if not content.strip():
                        return None
                    data = json.loads(content)
                    self.set_metadata('last_read', True)
                    return data
            except FileNotFoundError:
                return None
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error in {self.file_path}: {e}")
                return None
                
    async def set(self, data: Any) -> None:
        """Set data in file."""
        async with self._file_lock:
            # Create backup if enabled
            if self.backup_enabled and self.file_path.exists():
                await self._create_backup()
                
            await self._write_file(data)
            self.set_metadata('file_size_bytes', self.file_path.stat().st_size)
            self.logger.debug(f"Data written to file {self.file_path}")
            
    async def clear(self) -> None:
        """Clear data from file."""
        async with self._file_lock:
            if self.backup_enabled and self.file_path.exists():
                await self._create_backup()
                
            await self._write_file({})
            self.metadata.clear()
            self.logger.debug(f"File {self.file_path} cleared")
            
    async def _write_file(self, data: Any) -> None:
        """Write data to file with atomic operation."""
        temp_path = self.file_path.with_suffix(self.file_path.suffix + '.tmp')
        
        try:
            async with aiofiles.open(temp_path, 'w', encoding=self.encoding) as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
                await f.flush()
                
            # Atomic move
            temp_path.replace(self.file_path)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e
            
    async def _create_backup(self) -> None:
        """Create backup of current file."""
        if not self.file_path.exists():
            return
            
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.file_path.with_suffix(f'.backup_{timestamp}{self.file_path.suffix}')
        
        try:
            async with aiofiles.open(self.file_path, 'rb') as src:
                async with aiofiles.open(backup_path, 'wb') as dst:
                    content = await src.read()
                    await dst.write(content)
                    
            self.logger.debug(f"Backup created: {backup_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            
    async def append_to_list(self, item: Any) -> None:
        """Append item to a list stored in the file."""
        data = await self.get()
        if data is None:
            data = []
        elif not isinstance(data, list):
            raise TypeError("Cannot append to non-list data")
            
        data.append(item)
        await self.set(data)
        
    async def get_file_info(self) -> Dict[str, Any]:
        """Get file information."""
        if not self.file_path.exists():
            return {'exists': False}
            
        stat = self.file_path.stat()
        return {
            'exists': True,
            'size_bytes': stat.st_size,
            'modified_time': stat.st_mtime,
            'path': str(self.file_path)
        } 