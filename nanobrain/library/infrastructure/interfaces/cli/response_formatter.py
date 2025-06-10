"""
Response Formatter

Output formatting and display management for CLI interfaces.

This module provides:
- Response formatting and styling
- Output templating and customization
- Error and status message formatting
- Rich text and color support
"""

import sys
import os
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from nanobrain.core.logging_system import get_logger


class OutputFormat(Enum):
    """Output format types."""
    PLAIN = "plain"
    JSON = "json"
    TABLE = "table"
    LIST = "list"
    TREE = "tree"
    RICH = "rich"


class MessageType(Enum):
    """Message types for formatting."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"


@dataclass
class FormatterConfig:
    """Configuration for response formatter."""
    
    # Output settings
    default_format: OutputFormat = OutputFormat.PLAIN
    enable_colors: bool = True
    enable_rich_text: bool = True
    
    # Formatting options
    indent_size: int = 2
    max_line_length: int = 80
    enable_word_wrap: bool = True
    
    # Color scheme
    colors: Dict[str, str] = field(default_factory=lambda: {
        'info': '\033[94m',      # Blue
        'success': '\033[92m',   # Green
        'warning': '\033[93m',   # Yellow
        'error': '\033[91m',     # Red
        'debug': '\033[90m',     # Gray
        'reset': '\033[0m'       # Reset
    })
    
    # Templates
    templates: Dict[str, str] = field(default_factory=dict)
    
    # Table formatting
    table_border: bool = True
    table_header: bool = True
    
    # JSON formatting
    json_indent: int = 2
    json_sort_keys: bool = True


class ResponseFormatter:
    """
    Response formatter for CLI interfaces.
    
    This class provides:
    - Multiple output format support
    - Color and styling management
    - Template-based formatting
    - Error and status message formatting
    - Rich text rendering
    """
    
    def __init__(self, config: Optional[FormatterConfig] = None):
        """
        Initialize the response formatter.
        
        Args:
            config: Formatter configuration
        """
        self.config = config or FormatterConfig()
        self.logger = get_logger("cli.response_formatter")
        
        # State
        self.current_format = self.config.default_format
        
        self.logger.info("Response formatter initialized")
    
    async def initialize(self) -> None:
        """Initialize the response formatter."""
        # Setup templates and configurations
        self._setup_default_templates()
        self.logger.info("Response formatter initialization complete")
    
    async def cleanup(self) -> None:
        """Cleanup formatter resources."""
        self.logger.info("Response formatter cleanup complete")
    
    def _setup_default_templates(self) -> None:
        """Setup default formatting templates."""
        self.config.templates.update({
            'command_result': "{result}",
            'error': "{color_start}Error: {message}{color_end}",
            'warning': "{color_start}Warning: {message}{color_end}",
            'success': "{color_start}Success: {message}{color_end}",
            'info': "{color_start}Info: {message}{color_end}",
            'status': "Status: {status}",
            'table_row': "| {columns} |",
            'table_separator': "+{separators}+",
            'list_item': "  • {item}",
            'tree_node': "{indent}{marker} {item}"
        })
    
    async def format_output(self, data: Any, format_type: Optional[OutputFormat] = None) -> str:
        """
        Format output data.
        
        Args:
            data: Data to format
            format_type: Output format type
            
        Returns:
            Formatted string
        """
        format_type = format_type or self.current_format
        
        try:
            if format_type == OutputFormat.JSON:
                return await self._format_json(data)
            elif format_type == OutputFormat.TABLE:
                return await self._format_table(data)
            elif format_type == OutputFormat.LIST:
                return await self._format_list(data)
            elif format_type == OutputFormat.TREE:
                return await self._format_tree(data)
            elif format_type == OutputFormat.RICH:
                return await self._format_rich(data)
            else:
                return await self._format_plain(data)
                
        except Exception as e:
            self.logger.error(f"Error formatting output: {e}")
            return str(data)
    
    async def format_command_result(self, result: Any) -> str:
        """
        Format command execution result.
        
        Args:
            result: Command result
            
        Returns:
            Formatted result string
        """
        if result is None:
            return ""
        
        template = self.config.templates.get('command_result', "{result}")
        
        if isinstance(result, dict):
            if 'error' in result:
                return await self.format_error(result['error'])
            elif 'message' in result:
                return str(result['message'])
            else:
                return await self.format_output(result)
        
        return template.format(result=str(result))
    
    async def format_error(self, message: str) -> str:
        """
        Format error message.
        
        Args:
            message: Error message
            
        Returns:
            Formatted error string
        """
        return await self._format_message(message, MessageType.ERROR)
    
    async def format_warning(self, message: str) -> str:
        """
        Format warning message.
        
        Args:
            message: Warning message
            
        Returns:
            Formatted warning string
        """
        return await self._format_message(message, MessageType.WARNING)
    
    async def format_success(self, message: str) -> str:
        """
        Format success message.
        
        Args:
            message: Success message
            
        Returns:
            Formatted success string
        """
        return await self._format_message(message, MessageType.SUCCESS)
    
    async def format_info(self, message: str) -> str:
        """
        Format info message.
        
        Args:
            message: Info message
            
        Returns:
            Formatted info string
        """
        return await self._format_message(message, MessageType.INFO)
    
    async def _format_message(self, message: str, message_type: MessageType) -> str:
        """Format a message with type-specific styling."""
        template_key = message_type.value
        template = self.config.templates.get(template_key, "{message}")
        
        # Get colors
        color_start = ""
        color_end = ""
        
        if self.config.enable_colors:
            color_start = self.config.colors.get(message_type.value, "")
            color_end = self.config.colors.get('reset', "")
        
        return template.format(
            message=message,
            color_start=color_start,
            color_end=color_end
        )
    
    async def _format_plain(self, data: Any) -> str:
        """Format data as plain text."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            lines = []
            for key, value in data.items():
                lines.append(f"{key}: {value}")
            return "\n".join(lines)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)
    
    async def _format_json(self, data: Any) -> str:
        """Format data as JSON."""
        try:
            return json.dumps(
                data,
                indent=self.config.json_indent,
                sort_keys=self.config.json_sort_keys,
                ensure_ascii=False
            )
        except (TypeError, ValueError) as e:
            self.logger.warning(f"JSON formatting failed: {e}")
            return str(data)
    
    async def _format_table(self, data: Any) -> str:
        """Format data as a table."""
        if not isinstance(data, (list, dict)):
            return str(data)
        
        if isinstance(data, dict):
            # Convert dict to list of key-value pairs
            rows = [[key, str(value)] for key, value in data.items()]
            headers = ["Key", "Value"]
        else:
            # Assume list of dicts or list of lists
            if not data:
                return "Empty table"
            
            first_item = data[0]
            if isinstance(first_item, dict):
                headers = list(first_item.keys())
                rows = [[str(item.get(header, "")) for header in headers] for item in data]
            elif isinstance(first_item, (list, tuple)):
                headers = [f"Column {i+1}" for i in range(len(first_item))]
                rows = [[str(cell) for cell in row] for row in data]
            else:
                headers = ["Value"]
                rows = [[str(item)] for item in data]
        
        return self._render_table(headers, rows)
    
    def _render_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Render a table with headers and rows."""
        if not rows:
            return "Empty table"
        
        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))
        
        lines = []
        
        # Table border
        if self.config.table_border:
            separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
            lines.append(separator)
        
        # Headers
        if self.config.table_header:
            header_row = "|" + "|".join(f" {header:<{col_widths[i]}} " for i, header in enumerate(headers)) + "|"
            lines.append(header_row)
            
            if self.config.table_border:
                lines.append(separator)
        
        # Data rows
        for row in rows:
            data_row = "|" + "|".join(f" {str(cell):<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
            lines.append(data_row)
        
        # Bottom border
        if self.config.table_border:
            lines.append(separator)
        
        return "\n".join(lines)
    
    async def _format_list(self, data: Any) -> str:
        """Format data as a list."""
        if isinstance(data, dict):
            items = [f"{key}: {value}" for key, value in data.items()]
        elif isinstance(data, list):
            items = [str(item) for item in data]
        else:
            items = [str(data)]
        
        template = self.config.templates.get('list_item', "  • {item}")
        return "\n".join(template.format(item=item) for item in items)
    
    async def _format_tree(self, data: Any, level: int = 0) -> str:
        """Format data as a tree structure."""
        indent = "  " * level
        template = self.config.templates.get('tree_node', "{indent}{marker} {item}")
        
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                marker = "├─" if level > 0 else "└─"
                lines.append(template.format(indent=indent, marker=marker, item=key))
                
                if isinstance(value, (dict, list)) and value:
                    sub_tree = await self._format_tree(value, level + 1)
                    lines.append(sub_tree)
                else:
                    sub_indent = "  " * (level + 1)
                    lines.append(template.format(indent=sub_indent, marker="└─", item=str(value)))
            
            return "\n".join(lines)
        
        elif isinstance(data, list):
            lines = []
            for i, item in enumerate(data):
                marker = "├─" if i < len(data) - 1 else "└─"
                
                if isinstance(item, (dict, list)):
                    lines.append(template.format(indent=indent, marker=marker, item=f"[{i}]"))
                    sub_tree = await self._format_tree(item, level + 1)
                    lines.append(sub_tree)
                else:
                    lines.append(template.format(indent=indent, marker=marker, item=str(item)))
            
            return "\n".join(lines)
        
        else:
            marker = "└─"
            return template.format(indent=indent, marker=marker, item=str(data))
    
    async def _format_rich(self, data: Any) -> str:
        """Format data with rich text features."""
        # This would integrate with rich text libraries like 'rich'
        # For now, fall back to enhanced plain formatting
        
        if isinstance(data, dict):
            lines = []
            for key, value in data.items():
                if self.config.enable_colors:
                    key_colored = f"{self.config.colors.get('info', '')}{key}{self.config.colors.get('reset', '')}"
                    lines.append(f"{key_colored}: {value}")
                else:
                    lines.append(f"{key}: {value}")
            return "\n".join(lines)
        
        return await self._format_plain(data)
    
    def set_format(self, format_type: OutputFormat) -> None:
        """
        Set the current output format.
        
        Args:
            format_type: Output format to use
        """
        self.current_format = format_type
        self.logger.debug(f"Output format set to: {format_type.value}")
    
    def enable_colors(self, enabled: bool = True) -> None:
        """
        Enable or disable color output.
        
        Args:
            enabled: Whether to enable colors
        """
        self.config.enable_colors = enabled
        self.logger.debug(f"Color output {'enabled' if enabled else 'disabled'}")
    
    def set_color_scheme(self, colors: Dict[str, str]) -> None:
        """
        Set custom color scheme.
        
        Args:
            colors: Color mapping
        """
        self.config.colors.update(colors)
        self.logger.debug("Color scheme updated")
    
    def add_template(self, name: str, template: str) -> None:
        """
        Add a custom formatting template.
        
        Args:
            name: Template name
            template: Template string
        """
        self.config.templates[name] = template
        self.logger.debug(f"Added template: {name}")


# Convenience functions

def create_formatter(config: Optional[FormatterConfig] = None) -> ResponseFormatter:
    """
    Create a response formatter with optional configuration.
    
    Args:
        config: Formatter configuration
        
    Returns:
        Response formatter instance
    """
    return ResponseFormatter(config)


def format_simple_output(data: Any, format_type: OutputFormat = OutputFormat.PLAIN) -> str:
    """
    Simple output formatting function.
    
    Args:
        data: Data to format
        format_type: Output format
        
    Returns:
        Formatted string
    """
    formatter = ResponseFormatter()
    # Since this is a sync function, we can't use async methods
    # This is a simplified version for basic use cases
    
    if format_type == OutputFormat.JSON:
        try:
            return json.dumps(data, indent=2, sort_keys=True)
        except (TypeError, ValueError):
            return str(data)
    elif isinstance(data, dict):
        return "\n".join(f"{key}: {value}" for key, value in data.items())
    elif isinstance(data, list):
        return "\n".join(str(item) for item in data)
    else:
        return str(data) 