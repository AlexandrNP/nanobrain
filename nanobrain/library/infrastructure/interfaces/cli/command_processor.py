"""
Command Processor

Advanced command parsing and processing for CLI interfaces.

This module provides:
- Command parsing and validation
- Argument processing and type conversion
- Command context management
- Command execution pipeline
"""

import sys
import os
import shlex
import re
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from nanobrain.core.logging_system import get_logger


class CommandType(Enum):
    """Types of commands."""
    SYSTEM = "system"
    USER = "user"
    PLUGIN = "plugin"
    SCRIPT = "script"


@dataclass
class CLICommand:
    """Represents a CLI command."""
    
    name: str
    handler: Callable
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    command_type: CommandType = CommandType.USER
    
    # Argument specification
    required_args: List[str] = field(default_factory=list)
    optional_args: List[str] = field(default_factory=list)
    arg_types: Dict[str, type] = field(default_factory=dict)
    
    # Validation
    min_args: int = 0
    max_args: Optional[int] = None
    
    # Metadata
    category: str = "general"
    hidden: bool = False
    deprecated: bool = False
    
    # Help information
    usage: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class CLIContext:
    """Context for command execution."""
    
    command: str
    args: List[str]
    raw_input: str
    
    # Parsed arguments
    parsed_args: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    options: Dict[str, str] = field(default_factory=dict)
    
    # Execution context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    # State
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)


class CommandProcessor:
    """
    Advanced command processor for CLI interfaces.
    
    This class provides:
    - Command parsing and validation
    - Argument processing and type conversion
    - Command registration and management
    - Context-aware command execution
    - Plugin command support
    """
    
    def __init__(self):
        """Initialize the command processor."""
        self.logger = get_logger("cli.command_processor")
        
        # Command registry
        self.commands: Dict[str, CLICommand] = {}
        self.aliases: Dict[str, str] = {}
        
        # Processing settings
        self.case_sensitive = False
        self.allow_partial_matches = True
        self.command_prefix = "/"
        
        # Parsing patterns
        self.flag_pattern = re.compile(r'^-{1,2}([a-zA-Z][a-zA-Z0-9_-]*)')
        self.option_pattern = re.compile(r'^-{1,2}([a-zA-Z][a-zA-Z0-9_-]*)=(.+)')
        
        self.logger.info("Command processor initialized")
    
    async def initialize(self) -> None:
        """Initialize the command processor."""
        # Setup default patterns and configurations
        self.logger.info("Command processor initialization complete")
    
    async def cleanup(self) -> None:
        """Cleanup command processor resources."""
        self.commands.clear()
        self.aliases.clear()
        self.logger.info("Command processor cleanup complete")
    
    def register_command(self, command: CLICommand) -> None:
        """
        Register a command.
        
        Args:
            command: Command to register
        """
        # Register main command
        self.commands[command.name] = command
        
        # Register aliases
        for alias in command.aliases:
            self.aliases[alias] = command.name
        
        self.logger.debug(f"Registered command: {command.name}")
    
    def unregister_command(self, name: str) -> None:
        """
        Unregister a command.
        
        Args:
            name: Command name to unregister
        """
        if name in self.commands:
            command = self.commands[name]
            
            # Remove aliases
            for alias in command.aliases:
                if alias in self.aliases:
                    del self.aliases[alias]
            
            # Remove command
            del self.commands[name]
            
            self.logger.debug(f"Unregistered command: {name}")
    
    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and determine if it's a command.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Dict containing processed input information
        """
        user_input = user_input.strip()
        
        # Check if it's a command
        is_command = user_input.startswith(self.command_prefix)
        
        if is_command:
            return await self._process_command_input(user_input)
        else:
            return await self._process_regular_input(user_input)
    
    async def _process_command_input(self, command_input: str) -> Dict[str, Any]:
        """Process command input."""
        # Remove command prefix
        command_line = command_input[len(self.command_prefix):]
        
        # Parse command line
        try:
            parts = shlex.split(command_line)
        except ValueError as e:
            return {
                'is_command': True,
                'is_valid': False,
                'error': f"Command parsing error: {e}",
                'raw_input': command_input
            }
        
        if not parts:
            return {
                'is_command': True,
                'is_valid': False,
                'error': "Empty command",
                'raw_input': command_input
            }
        
        command_name = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        # Resolve command name (handle aliases and case sensitivity)
        resolved_command = self._resolve_command_name(command_name)
        
        if not resolved_command:
            return {
                'is_command': True,
                'is_valid': False,
                'error': f"Unknown command: {command_name}",
                'command': command_name,
                'args': args,
                'raw_input': command_input
            }
        
        # Create context
        context = CLIContext(
            command=resolved_command,
            args=args,
            raw_input=command_input
        )
        
        # Parse arguments
        await self._parse_arguments(context)
        
        return {
            'is_command': True,
            'is_valid': context.is_valid,
            'command': resolved_command,
            'args': args,
            'context': context,
            'raw_input': command_input,
            'validation_errors': context.validation_errors
        }
    
    async def _process_regular_input(self, user_input: str) -> Dict[str, Any]:
        """Process regular (non-command) input."""
        return {
            'is_command': False,
            'is_valid': True,
            'text': user_input,
            'raw_input': user_input
        }
    
    def _resolve_command_name(self, command_name: str) -> Optional[str]:
        """
        Resolve command name handling aliases and case sensitivity.
        
        Args:
            command_name: Command name to resolve
            
        Returns:
            Resolved command name or None if not found
        """
        # Handle case sensitivity
        if not self.case_sensitive:
            command_name = command_name.lower()
            
            # Check direct match
            for cmd in self.commands:
                if cmd.lower() == command_name:
                    return cmd
            
            # Check aliases
            for alias, target in self.aliases.items():
                if alias.lower() == command_name:
                    return target
        else:
            # Case sensitive matching
            if command_name in self.commands:
                return command_name
            
            if command_name in self.aliases:
                return self.aliases[command_name]
        
        # Try partial matching if enabled
        if self.allow_partial_matches:
            matches = []
            
            for cmd in self.commands:
                if (self.case_sensitive and cmd.startswith(command_name)) or \
                   (not self.case_sensitive and cmd.lower().startswith(command_name.lower())):
                    matches.append(cmd)
            
            if len(matches) == 1:
                return matches[0]
        
        return None
    
    async def _parse_arguments(self, context: CLIContext) -> None:
        """
        Parse command arguments.
        
        Args:
            context: Command context to populate
        """
        command = self.commands.get(context.command)
        if not command:
            context.is_valid = False
            context.validation_errors.append(f"Command not found: {context.command}")
            return
        
        # Parse flags and options
        remaining_args = []
        i = 0
        
        while i < len(context.args):
            arg = context.args[i]
            
            # Check for option with value (--option=value)
            option_match = self.option_pattern.match(arg)
            if option_match:
                option_name = option_match.group(1)
                option_value = option_match.group(2)
                context.options[option_name] = option_value
                i += 1
                continue
            
            # Check for flag (--flag or -f)
            flag_match = self.flag_pattern.match(arg)
            if flag_match:
                flag_name = flag_match.group(1)
                context.flags[flag_name] = True
                
                # Check if next argument is a value for this flag
                if i + 1 < len(context.args) and not context.args[i + 1].startswith('-'):
                    context.options[flag_name] = context.args[i + 1]
                    i += 2
                else:
                    i += 1
                continue
            
            # Regular argument
            remaining_args.append(arg)
            i += 1
        
        # Validate argument count
        if len(remaining_args) < command.min_args:
            context.is_valid = False
            context.validation_errors.append(
                f"Command '{context.command}' requires at least {command.min_args} arguments, got {len(remaining_args)}"
            )
        
        if command.max_args is not None and len(remaining_args) > command.max_args:
            context.is_valid = False
            context.validation_errors.append(
                f"Command '{context.command}' accepts at most {command.max_args} arguments, got {len(remaining_args)}"
            )
        
        # Type conversion for arguments
        for i, arg in enumerate(remaining_args):
            arg_name = None
            
            # Try to get argument name from command definition
            if i < len(command.required_args):
                arg_name = command.required_args[i]
            elif i - len(command.required_args) < len(command.optional_args):
                arg_name = command.optional_args[i - len(command.required_args)]
            
            if arg_name and arg_name in command.arg_types:
                try:
                    converted_value = command.arg_types[arg_name](arg)
                    context.parsed_args[arg_name] = converted_value
                except (ValueError, TypeError) as e:
                    context.is_valid = False
                    context.validation_errors.append(
                        f"Invalid type for argument '{arg_name}': {e}"
                    )
            else:
                # Store as string if no type specified
                if arg_name:
                    context.parsed_args[arg_name] = arg
        
        # Update args with remaining positional arguments
        context.args = remaining_args
    
    def get_command_help(self, command_name: str) -> Optional[str]:
        """
        Get help text for a command.
        
        Args:
            command_name: Command name
            
        Returns:
            Help text or None if command not found
        """
        resolved_name = self._resolve_command_name(command_name)
        if not resolved_name or resolved_name not in self.commands:
            return None
        
        command = self.commands[resolved_name]
        
        help_lines = [
            f"Command: {command.name}",
            f"Description: {command.description or 'No description available'}"
        ]
        
        if command.aliases:
            help_lines.append(f"Aliases: {', '.join(command.aliases)}")
        
        if command.usage:
            help_lines.append(f"Usage: {command.usage}")
        else:
            # Generate usage from command definition
            usage_parts = [command.name]
            
            for arg in command.required_args:
                usage_parts.append(f"<{arg}>")
            
            for arg in command.optional_args:
                usage_parts.append(f"[{arg}]")
            
            help_lines.append(f"Usage: {' '.join(usage_parts)}")
        
        if command.required_args:
            help_lines.append(f"Required arguments: {', '.join(command.required_args)}")
        
        if command.optional_args:
            help_lines.append(f"Optional arguments: {', '.join(command.optional_args)}")
        
        if command.examples:
            help_lines.append("Examples:")
            for example in command.examples:
                help_lines.append(f"  {example}")
        
        return "\n".join(help_lines)
    
    def list_commands(self, category: Optional[str] = None, 
                     include_hidden: bool = False) -> List[CLICommand]:
        """
        List available commands.
        
        Args:
            category: Filter by category
            include_hidden: Include hidden commands
            
        Returns:
            List of commands
        """
        commands = []
        
        for command in self.commands.values():
            # Skip hidden commands unless requested
            if command.hidden and not include_hidden:
                continue
            
            # Filter by category
            if category and command.category != category:
                continue
            
            commands.append(command)
        
        # Sort by name
        commands.sort(key=lambda c: c.name)
        
        return commands
    
    def get_command_suggestions(self, partial_command: str) -> List[str]:
        """
        Get command suggestions for partial input.
        
        Args:
            partial_command: Partial command name
            
        Returns:
            List of suggested command names
        """
        suggestions = []
        
        if not self.case_sensitive:
            partial_command = partial_command.lower()
        
        for command_name in self.commands:
            if (self.case_sensitive and command_name.startswith(partial_command)) or \
               (not self.case_sensitive and command_name.lower().startswith(partial_command)):
                suggestions.append(command_name)
        
        # Also check aliases
        for alias in self.aliases:
            if (self.case_sensitive and alias.startswith(partial_command)) or \
               (not self.case_sensitive and alias.lower().startswith(partial_command)):
                suggestions.append(alias)
        
        return sorted(suggestions)


# Convenience functions

def create_command(name: str, handler: Callable, description: str = "", **kwargs) -> CLICommand:
    """
    Create a CLI command with simplified interface.
    
    Args:
        name: Command name
        handler: Command handler function
        description: Command description
        **kwargs: Additional command properties
        
    Returns:
        CLI command instance
    """
    return CLICommand(
        name=name,
        handler=handler,
        description=description,
        **kwargs
    )


def create_system_command(name: str, handler: Callable, description: str = "", **kwargs) -> CLICommand:
    """
    Create a system CLI command.
    
    Args:
        name: Command name
        handler: Command handler function
        description: Command description
        **kwargs: Additional command properties
        
    Returns:
        System CLI command instance
    """
    return CLICommand(
        name=name,
        handler=handler,
        description=description,
        command_type=CommandType.SYSTEM,
        **kwargs
    ) 