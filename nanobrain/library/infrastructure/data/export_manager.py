"""
Data export and import utilities.

Data serialization and migration tools.
"""

import json
import csv
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from nanobrain.core.data_unit import DataUnitBase


class ExportManager(DataUnitBase):
    """Data export and import utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.export_directory = Path(self.config.get('export_directory', 'exports'))
        self.supported_formats = ['json', 'csv', 'txt']
        
    async def _initialize_impl(self) -> None:
        """Initialize export manager."""
        # Create export directory
        self.export_directory.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Export manager initialized with directory: {self.export_directory}")
        
    async def get(self) -> Any:
        """Get export statistics."""
        return await self.get_export_statistics()
        
    async def set(self, data: Any) -> None:
        """Export data."""
        if isinstance(data, dict) and 'data' in data and 'format' in data:
            await self.export_data(
                data['data'],
                data.get('filename', 'export'),
                data['format']
            )
        else:
            raise TypeError("Data must be dict with 'data' and 'format' keys")
            
    async def clear(self) -> None:
        """Clear export directory."""
        for file_path in self.export_directory.iterdir():
            if file_path.is_file():
                file_path.unlink()
        self.logger.info("Export directory cleared")
        
    async def export_data(self, data: Any, filename: str, format: str = 'json', **kwargs) -> str:
        """Export data to file."""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.supported_formats}")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_timestamp = f"{filename}_{timestamp}.{format}"
        file_path = self.export_directory / filename_with_timestamp
        
        if format == 'json':
            await self._export_json(data, file_path, **kwargs)
        elif format == 'csv':
            await self._export_csv(data, file_path, **kwargs)
        elif format == 'txt':
            await self._export_txt(data, file_path, **kwargs)
            
        self.logger.info(f"Exported data to {file_path}")
        return str(file_path)
        
    async def _export_json(self, data: Any, file_path: Path, indent: int = 2, **kwargs) -> None:
        """Export data as JSON."""
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=json_serializer, ensure_ascii=False)
            
    async def _export_csv(self, data: Any, file_path: Path, **kwargs) -> None:
        """Export data as CSV."""
        if not isinstance(data, list):
            raise TypeError("CSV export requires list of dictionaries")
            
        if not data:
            # Create empty CSV file
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                pass
            return
            
        # Get fieldnames from first item
        if isinstance(data[0], dict):
            fieldnames = list(data[0].keys())
        else:
            raise TypeError("CSV export requires list of dictionaries")
            
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                # Convert datetime objects to strings
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        processed_row[key] = value.isoformat()
                    else:
                        processed_row[key] = value
                writer.writerow(processed_row)
                
    async def _export_txt(self, data: Any, file_path: Path, separator: str = '\n', **kwargs) -> None:
        """Export data as text."""
        with open(file_path, 'w', encoding='utf-8') as f:
            if isinstance(data, (list, tuple)):
                f.write(separator.join(str(item) for item in data))
            else:
                f.write(str(data))
                
    async def import_data(self, file_path: Union[str, Path], format: Optional[str] = None) -> Any:
        """Import data from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Auto-detect format from extension if not provided
        if format is None:
            format = file_path.suffix.lstrip('.')
            
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
            
        if format == 'json':
            return await self._import_json(file_path)
        elif format == 'csv':
            return await self._import_csv(file_path)
        elif format == 'txt':
            return await self._import_txt(file_path)
            
    async def _import_json(self, file_path: Path) -> Any:
        """Import data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.logger.info(f"Imported JSON data from {file_path}")
        return data
        
    async def _import_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Import data from CSV file."""
        data = []
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        self.logger.info(f"Imported {len(data)} rows from CSV file {file_path}")
        return data
        
    async def _import_txt(self, file_path: Path) -> str:
        """Import data from text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        self.logger.info(f"Imported text data from {file_path}")
        return data
        
    async def export_multiple_data_units(self, data_units: Dict[str, DataUnitBase], format: str = 'json') -> str:
        """Export multiple data units to a single file."""
        export_data = {}
        
        for name, data_unit in data_units.items():
            try:
                export_data[name] = await data_unit.get()
            except Exception as e:
                self.logger.warning(f"Failed to export data unit {name}: {e}")
                export_data[name] = None
                
        filename = f"multi_export_{len(data_units)}_units"
        return await self.export_data(export_data, filename, format)
        
    async def backup_data_unit(self, data_unit: DataUnitBase, backup_name: Optional[str] = None) -> str:
        """Create backup of a data unit."""
        if backup_name is None:
            backup_name = f"backup_{data_unit.name}"
            
        data = await data_unit.get()
        metadata = data_unit.get_metadata()
        
        backup_data = {
            'data': data,
            'metadata': metadata,
            'backup_timestamp': datetime.now().isoformat(),
            'data_unit_name': data_unit.name,
            'data_unit_type': data_unit.data_type
        }
        
        return await self.export_data(backup_data, backup_name, 'json')
        
    async def restore_data_unit(self, data_unit: DataUnitBase, backup_file: Union[str, Path]) -> bool:
        """Restore data unit from backup."""
        try:
            backup_data = await self.import_data(backup_file, 'json')
            
            if 'data' in backup_data:
                await data_unit.set(backup_data['data'])
                
                # Restore metadata if available
                if 'metadata' in backup_data:
                    for key, value in backup_data['metadata'].items():
                        data_unit.set_metadata(key, value)
                        
                self.logger.info(f"Restored data unit {data_unit.name} from {backup_file}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to restore data unit from {backup_file}: {e}")
            
        return False
        
    async def get_export_statistics(self) -> Dict[str, Any]:
        """Get export directory statistics."""
        if not self.export_directory.exists():
            return {'export_directory_exists': False}
            
        files = list(self.export_directory.iterdir())
        file_stats = {}
        total_size = 0
        
        for file_path in files:
            if file_path.is_file():
                stat = file_path.stat()
                file_stats[file_path.name] = {
                    'size_bytes': stat.st_size,
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'format': file_path.suffix.lstrip('.')
                }
                total_size += stat.st_size
                
        format_counts = {}
        for stats in file_stats.values():
            format_type = stats['format']
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
            
        return {
            'export_directory': str(self.export_directory),
            'export_directory_exists': True,
            'total_files': len(file_stats),
            'total_size_bytes': total_size,
            'files': file_stats,
            'formats': format_counts,
            'supported_formats': self.supported_formats
        }
        
    async def cleanup_old_exports(self, days_old: int = 30) -> int:
        """Clean up old export files."""
        if not self.export_directory.exists():
            return 0
            
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        removed_count = 0
        
        for file_path in self.export_directory.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                removed_count += 1
                
        self.logger.info(f"Cleaned up {removed_count} old export files")
        return removed_count
        
    async def compress_exports(self, output_filename: str = 'exports_archive') -> str:
        """Compress all exports into a single archive."""
        import zipfile
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.export_directory / f"{output_filename}_{timestamp}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.export_directory.iterdir():
                if file_path.is_file() and file_path.suffix != '.zip':
                    zipf.write(file_path, file_path.name)
                    
        self.logger.info(f"Compressed exports to {archive_path}")
        return str(archive_path) 