#!/usr/bin/env python3
"""
Format Converter for NanoBrain Framework
Configurable format conversion for diverse response types and data transformations.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import logging
import json
import csv
import io
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from abc import ABC, abstractmethod

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase

# Format converter logger
logger = logging.getLogger(__name__)


class FormatConverterConfig(ConfigBase):
    """Configuration for format converter"""
    
    def __init__(self, config_data: Dict[str, Any]):
        super().__init__(config_data)
        
        # Supported format conversions
        self.supported_conversions: Dict[str, List[str]] = config_data.get('supported_conversions', {
            'json': ['dict', 'string', 'csv', 'table', 'xml'],
            'csv': ['json', 'table', 'dict', 'string'],
            'table': ['json', 'csv', 'html', 'markdown'],
            'text': ['html', 'markdown', 'json'],
            'html': ['text', 'markdown'],
            'markdown': ['html', 'text'],
            'xml': ['json', 'dict', 'string']
        })
        
        # Conversion configuration
        self.conversion_config: Dict[str, Any] = config_data.get('conversion_config', {
            'preserve_metadata': True,
            'validate_output': True,
            'handle_encoding_issues': True,
            'max_conversion_size': 5242880  # 5MB
        })
        
        # Format-specific settings
        self.format_settings: Dict[str, Dict[str, Any]] = config_data.get('format_settings', {
            'csv': {
                'delimiter': ',',
                'quote_char': '"',
                'escape_char': '\\',
                'include_headers': True
            },
            'json': {
                'indent': 2,
                'ensure_ascii': False,
                'sort_keys': False
            },
            'html': {
                'include_css': False,
                'table_class': 'nanobrain-table',
                'escape_html': True
            },
            'markdown': {
                'table_format': 'pipe',
                'header_level': 2
            }
        })


class BaseFormatConverter(ABC):
    """Abstract base class for format converters"""
    
    def __init__(self, config: FormatConverterConfig):
        self.config = config
        self.source_format: str = ""
        self.target_format: str = ""
    
    @abstractmethod
    async def convert(self, data: Any, source_format: str, target_format: str) -> Dict[str, Any]:
        """
        Convert data from source format to target format.
        
        Args:
            data: Data to convert
            source_format: Source format identifier
            target_format: Target format identifier
            
        Returns:
            Dictionary with converted data and metadata
        """
        pass
    
    def validate_conversion(self, source_format: str, target_format: str) -> bool:
        """Validate if conversion is supported"""
        supported = self.config.supported_conversions.get(source_format, [])
        return target_format in supported
    
    def get_format_settings(self, format_name: str) -> Dict[str, Any]:
        """Get format-specific settings"""
        return self.config.format_settings.get(format_name, {})


class JSONConverter(BaseFormatConverter):
    """Converter for JSON format transformations"""
    
    async def convert(self, data: Any, source_format: str, target_format: str) -> Dict[str, Any]:
        """Convert JSON data to various formats"""
        try:
            if target_format == 'dict':
                return await self.to_dict(data)
            elif target_format == 'string':
                return await self.to_string(data)
            elif target_format == 'csv':
                return await self.to_csv(data)
            elif target_format == 'table':
                return await self.to_table(data)
            elif target_format == 'xml':
                return await self.to_xml(data)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
                
        except Exception as e:
            logger.error(f"❌ JSON conversion failed: {e}")
            raise
    
    async def to_dict(self, data: Any) -> Dict[str, Any]:
        """Convert JSON to Python dictionary"""
        if isinstance(data, dict):
            return {'data': data, 'format': 'dict', 'conversion': 'direct'}
        elif isinstance(data, str):
            try:
                parsed = json.loads(data)
                return {'data': parsed, 'format': 'dict', 'conversion': 'parsed'}
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        else:
            return {'data': data, 'format': 'dict', 'conversion': 'as_is'}
    
    async def to_string(self, data: Any) -> Dict[str, Any]:
        """Convert JSON to formatted string"""
        settings = self.get_format_settings('json')
        
        if isinstance(data, (dict, list)):
            json_string = json.dumps(
                data,
                indent=settings.get('indent', 2),
                ensure_ascii=settings.get('ensure_ascii', False),
                sort_keys=settings.get('sort_keys', False),
                default=str
            )
            return {'data': json_string, 'format': 'string', 'conversion': 'serialized'}
        else:
            return {'data': str(data), 'format': 'string', 'conversion': 'stringified'}
    
    async def to_csv(self, data: Any) -> Dict[str, Any]:
        """Convert JSON to CSV format"""
        settings = self.get_format_settings('csv')
        
        try:
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # List of dictionaries - perfect for CSV
                output = io.StringIO()
                if data:
                    fieldnames = list(data[0].keys())
                    writer = csv.DictWriter(
                        output,
                        fieldnames=fieldnames,
                        delimiter=settings.get('delimiter', ','),
                        quotechar=settings.get('quote_char', '"')
                    )
                    
                    if settings.get('include_headers', True):
                        writer.writeheader()
                    
                    for row in data:
                        writer.writerow(row)
                
                csv_data = output.getvalue()
                output.close()
                
                return {
                    'data': csv_data,
                    'format': 'csv',
                    'conversion': 'tabular',
                    'row_count': len(data),
                    'columns': fieldnames if data else []
                }
            
            elif isinstance(data, dict):
                # Single dictionary - convert to key-value CSV
                output = io.StringIO()
                writer = csv.writer(output, delimiter=settings.get('delimiter', ','))
                
                if settings.get('include_headers', True):
                    writer.writerow(['Key', 'Value'])
                
                for key, value in data.items():
                    writer.writerow([key, str(value)])
                
                csv_data = output.getvalue()
                output.close()
                
                return {
                    'data': csv_data,
                    'format': 'csv',
                    'conversion': 'key_value',
                    'row_count': len(data),
                    'columns': ['Key', 'Value']
                }
            
            else:
                raise ValueError(f"Cannot convert {type(data)} to CSV")
                
        except Exception as e:
            logger.error(f"❌ JSON to CSV conversion failed: {e}")
            raise
    
    async def to_table(self, data: Any) -> Dict[str, Any]:
        """Convert JSON to table format"""
        try:
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # List of dictionaries
                headers = list(data[0].keys())
                rows = []
                
                for item in data:
                    row = [str(item.get(header, '')) for header in headers]
                    rows.append(row)
                
                return {
                    'data': {
                        'headers': headers,
                        'rows': rows,
                        'row_count': len(rows),
                        'column_count': len(headers)
                    },
                    'format': 'table',
                    'conversion': 'structured'
                }
            
            elif isinstance(data, dict):
                # Single dictionary
                headers = ['Key', 'Value']
                rows = [[key, str(value)] for key, value in data.items()]
                
                return {
                    'data': {
                        'headers': headers,
                        'rows': rows,
                        'row_count': len(rows),
                        'column_count': 2
                    },
                    'format': 'table',
                    'conversion': 'key_value'
                }
            
            else:
                raise ValueError(f"Cannot convert {type(data)} to table")
                
        except Exception as e:
            logger.error(f"❌ JSON to table conversion failed: {e}")
            raise
    
    async def to_xml(self, data: Any) -> Dict[str, Any]:
        """Convert JSON to XML format"""
        try:
            def dict_to_xml(d: dict, root_name: str = 'root') -> str:
                """Convert dictionary to XML string"""
                xml_lines = [f'<{root_name}>']
                
                for key, value in d.items():
                    if isinstance(value, dict):
                        xml_lines.append(dict_to_xml(value, key))
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                xml_lines.append(dict_to_xml(item, key))
                            else:
                                xml_lines.append(f'  <{key}>{str(item)}</{key}>')
                    else:
                        xml_lines.append(f'  <{key}>{str(value)}</{key}>')
                
                xml_lines.append(f'</{root_name}>')
                return '\n'.join(xml_lines)
            
            if isinstance(data, dict):
                xml_string = dict_to_xml(data, 'data')
                return {'data': xml_string, 'format': 'xml', 'conversion': 'structured'}
            elif isinstance(data, list):
                xml_string = dict_to_xml({'items': data}, 'data')
                return {'data': xml_string, 'format': 'xml', 'conversion': 'list'}
            else:
                xml_string = f'<data>{str(data)}</data>'
                return {'data': xml_string, 'format': 'xml', 'conversion': 'simple'}
                
        except Exception as e:
            logger.error(f"❌ JSON to XML conversion failed: {e}")
            raise


class CSVConverter(BaseFormatConverter):
    """Converter for CSV format transformations"""
    
    async def convert(self, data: Any, source_format: str, target_format: str) -> Dict[str, Any]:
        """Convert CSV data to various formats"""
        try:
            if target_format == 'json':
                return await self.to_json(data)
            elif target_format == 'table':
                return await self.to_table(data)
            elif target_format == 'dict':
                return await self.to_dict(data)
            elif target_format == 'string':
                return await self.to_string(data)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
                
        except Exception as e:
            logger.error(f"❌ CSV conversion failed: {e}")
            raise
    
    async def to_json(self, data: str) -> Dict[str, Any]:
        """Convert CSV to JSON format"""
        settings = self.get_format_settings('csv')
        
        try:
            input_stream = io.StringIO(data)
            reader = csv.DictReader(
                input_stream,
                delimiter=settings.get('delimiter', ','),
                quotechar=settings.get('quote_char', '"')
            )
            
            rows = list(reader)
            input_stream.close()
            
            return {
                'data': rows,
                'format': 'json',
                'conversion': 'tabular',
                'row_count': len(rows),
                'columns': list(rows[0].keys()) if rows else []
            }
            
        except Exception as e:
            logger.error(f"❌ CSV to JSON conversion failed: {e}")
            raise
    
    async def to_table(self, data: str) -> Dict[str, Any]:
        """Convert CSV to table format"""
        settings = self.get_format_settings('csv')
        
        try:
            input_stream = io.StringIO(data)
            reader = csv.reader(
                input_stream,
                delimiter=settings.get('delimiter', ','),
                quotechar=settings.get('quote_char', '"')
            )
            
            rows = list(reader)
            input_stream.close()
            
            if rows:
                headers = rows[0] if settings.get('include_headers', True) else [f'Column_{i}' for i in range(len(rows[0]))]
                data_rows = rows[1:] if settings.get('include_headers', True) else rows
            else:
                headers = []
                data_rows = []
            
            return {
                'data': {
                    'headers': headers,
                    'rows': data_rows,
                    'row_count': len(data_rows),
                    'column_count': len(headers)
                },
                'format': 'table',
                'conversion': 'parsed'
            }
            
        except Exception as e:
            logger.error(f"❌ CSV to table conversion failed: {e}")
            raise
    
    async def to_dict(self, data: str) -> Dict[str, Any]:
        """Convert CSV to dictionary format"""
        json_result = await self.to_json(data)
        return {
            'data': json_result['data'],
            'format': 'dict',
            'conversion': json_result['conversion']
        }
    
    async def to_string(self, data: str) -> Dict[str, Any]:
        """Convert CSV to formatted string"""
        return {'data': data, 'format': 'string', 'conversion': 'direct'}


class TextConverter(BaseFormatConverter):
    """Converter for text format transformations"""
    
    async def convert(self, data: Any, source_format: str, target_format: str) -> Dict[str, Any]:
        """Convert text data to various formats"""
        try:
            if target_format == 'html':
                return await self.to_html(data)
            elif target_format == 'markdown':
                return await self.to_markdown(data)
            elif target_format == 'json':
                return await self.to_json(data)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
                
        except Exception as e:
            logger.error(f"❌ Text conversion failed: {e}")
            raise
    
    async def to_html(self, data: str) -> Dict[str, Any]:
        """Convert text to HTML format"""
        settings = self.get_format_settings('html')
        
        try:
            # Basic text to HTML conversion
            if settings.get('escape_html', True):
                # Escape HTML characters
                html_content = data.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            else:
                html_content = data
            
            # Convert line breaks to <br> tags
            html_content = html_content.replace('\n', '<br>\n')
            
            # Wrap in paragraph tags
            html_content = f'<p>{html_content}</p>'
            
            return {
                'data': html_content,
                'format': 'html',
                'conversion': 'text_to_html',
                'escaped': settings.get('escape_html', True)
            }
            
        except Exception as e:
            logger.error(f"❌ Text to HTML conversion failed: {e}")
            raise
    
    async def to_markdown(self, data: str) -> Dict[str, Any]:
        """Convert text to Markdown format"""
        try:
            # Basic text to Markdown conversion
            # Add proper line spacing for Markdown
            markdown_content = data.replace('\n\n', '\n\n')  # Preserve paragraph breaks
            
            return {
                'data': markdown_content,
                'format': 'markdown',
                'conversion': 'text_to_markdown'
            }
            
        except Exception as e:
            logger.error(f"❌ Text to Markdown conversion failed: {e}")
            raise
    
    async def to_json(self, data: str) -> Dict[str, Any]:
        """Convert text to JSON format"""
        try:
            # Convert text to structured JSON
            lines = data.split('\n')
            structured_data = {
                'text': data,
                'lines': lines,
                'line_count': len(lines),
                'character_count': len(data)
            }
            
            return {
                'data': structured_data,
                'format': 'json',
                'conversion': 'text_analysis'
            }
            
        except Exception as e:
            logger.error(f"❌ Text to JSON conversion failed: {e}")
            raise


class FormatConverter(FromConfigBase):
    """Configurable format conversion for diverse response types"""
    
    def __init__(self):
        """Initialize format converter - use from_config for creation"""
        super().__init__()
        self.config: Optional[FormatConverterConfig] = None
        self.converters: Dict[str, BaseFormatConverter] = {}
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return FormatConverterConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize converter from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        logger.info("🔄 Initializing Format Converter")
        self.config = config
        
        # Setup format converters
        self.setup_format_converters()
        
        logger.info("✅ Format Converter initialized successfully")
    
    def setup_format_converters(self) -> None:
        """Setup format-specific converters"""
        self.converters = {
            'json': JSONConverter(self.config),
            'csv': CSVConverter(self.config),
            'text': TextConverter(self.config)
        }
        
        logger.debug(f"✅ Initialized {len(self.converters)} format converters")
    
    async def convert_to_universal_format(self, data: Any, source_format: str) -> Dict[str, Any]:
        """
        Convert data to universal format based on source type.
        
        Args:
            data: Data to convert
            source_format: Source format identifier
            
        Returns:
            Dictionary with converted data in universal format
        """
        try:
            # Validate input
            if not self.validate_input(data, source_format):
                raise ValueError(f"Invalid input for format: {source_format}")
            
            # Check conversion size limit
            if not self.check_size_limit(data):
                raise ValueError("Data exceeds maximum conversion size limit")
            
            # Get appropriate converter
            converter = self.get_converter(source_format)
            if not converter:
                raise ValueError(f"No converter available for format: {source_format}")
            
            # Convert to universal format (JSON as default universal format)
            result = await converter.convert(data, source_format, 'json')
            
            # Add metadata if configured
            if self.config.conversion_config.get('preserve_metadata', True):
                result['conversion_metadata'] = {
                    'source_format': source_format,
                    'target_format': 'json',
                    'conversion_timestamp': datetime.now().isoformat(),
                    'converter_type': converter.__class__.__name__
                }
            
            # Validate output if configured
            if self.config.conversion_config.get('validate_output', True):
                await self.validate_conversion_output(result)
            
            logger.debug(f"✅ Converted {source_format} to universal format")
            return result
            
        except Exception as e:
            logger.error(f"❌ Universal format conversion failed: {e}")
            raise
    
    async def convert_format(self, data: Any, source_format: str, target_format: str) -> Dict[str, Any]:
        """
        Convert data from source format to target format.
        
        Args:
            data: Data to convert
            source_format: Source format identifier
            target_format: Target format identifier
            
        Returns:
            Dictionary with converted data
        """
        try:
            # Validate conversion is supported
            if not self.is_conversion_supported(source_format, target_format):
                raise ValueError(f"Conversion from {source_format} to {target_format} not supported")
            
            # Get appropriate converter
            converter = self.get_converter(source_format)
            if not converter:
                raise ValueError(f"No converter available for format: {source_format}")
            
            # Perform conversion
            result = await converter.convert(data, source_format, target_format)
            
            # Add metadata
            if self.config.conversion_config.get('preserve_metadata', True):
                result['conversion_metadata'] = {
                    'source_format': source_format,
                    'target_format': target_format,
                    'conversion_timestamp': datetime.now().isoformat(),
                    'converter_type': converter.__class__.__name__
                }
            
            logger.debug(f"✅ Converted {source_format} to {target_format}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Format conversion failed: {e}")
            raise
    
    def get_converter(self, source_format: str) -> Optional[BaseFormatConverter]:
        """Get appropriate converter for source format"""
        return self.converters.get(source_format)
    
    def is_conversion_supported(self, source_format: str, target_format: str) -> bool:
        """Check if conversion is supported"""
        supported = self.config.supported_conversions.get(source_format, [])
        return target_format in supported
    
    def validate_input(self, data: Any, source_format: str) -> bool:
        """Validate input data for conversion"""
        try:
            if data is None:
                return False
            
            # Format-specific validation
            if source_format == 'json' and not isinstance(data, (dict, list, str)):
                return False
            elif source_format == 'csv' and not isinstance(data, str):
                return False
            elif source_format == 'text' and not isinstance(data, str):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return False
    
    def check_size_limit(self, data: Any) -> bool:
        """Check if data size is within limits"""
        try:
            max_size = self.config.conversion_config.get('max_conversion_size', 5242880)
            
            if isinstance(data, str):
                size = len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                size = len(json.dumps(data, default=str).encode('utf-8'))
            else:
                size = len(str(data).encode('utf-8'))
            
            return size <= max_size
            
        except Exception as e:
            logger.error(f"❌ Size check failed: {e}")
            return False
    
    async def validate_conversion_output(self, result: Dict[str, Any]) -> None:
        """Validate conversion output"""
        try:
            # Basic validation
            if 'data' not in result:
                raise ValueError("Conversion result missing 'data' field")
            
            if 'format' not in result:
                raise ValueError("Conversion result missing 'format' field")
            
            # Format-specific validation could be added here
            
        except Exception as e:
            logger.error(f"❌ Output validation failed: {e}")
            raise
    
    def get_supported_conversions(self) -> Dict[str, List[str]]:
        """Get supported format conversions"""
        return self.config.supported_conversions
    
    async def get_health_status(self) -> str:
        """Get converter health status"""
        try:
            if self.config and len(self.converters) > 0:
                return "healthy"
            else:
                return "unhealthy"
        except Exception as e:
            logger.error(f"❌ Converter health check failed: {e}")
            return "unhealthy" 