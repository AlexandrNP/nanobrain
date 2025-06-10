"""
Base CLI Interface

Abstract base class for command-line interfaces in the NanoBrain framework.

This module provides:
- Abstract base CLI class with common functionality
- Configuration management for CLI behavior
- Event handling and lifecycle management
- Extensible command system
- Input/output abstraction
"""

import sys
import os
import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from nanobrain.core.logging_system import get_logger


class CLIMode(Enum):
    """CLI operation modes."""
    INTERACTIVE = "interactive"
    BATCH = "batch"
    SINGLE_COMMAND = "single_command"


class CLITheme(Enum):
    """CLI visual themes."""
    DEFAULT = "default"
    MINIMAL = "minimal"
    RICH = "rich"
    DARK = "dark"
    LIGHT = "light"


@dataclass
class CLIConfig:
    """Configuration for CLI interfaces."""
    
    # Basic settings
    name: str = "nanobrain_cli"
    description: str = "NanoBrain CLI Interface"
    version: str = "1.0.0"
    
    # Operation mode
    mode: CLIMode = CLIMode.INTERACTIVE
    
    # Display settings
    theme: CLITheme = CLITheme.DEFAULT
    show_timestamps: bool = True
    show_prompts: bool = True
    prompt_prefix: str = "> "
    prompt_suffix: str = " "
    
    # Input/Output settings
    input_encoding: str = "utf-8"
    output_encoding: str = "utf-8"
    buffer_size: int = 8192
    
    # Behavior settings
    enable_history: bool = True
    history_size: int = 1000
    enable_autocomplete: bool = True
    enable_help: bool = True
    enable_quit_commands: bool = True
    
    # Command settings
    command_prefix: str = "/"
    case_sensitive_commands: bool = False
    enable_aliases: bool = True
    
    # Display formatting
    max_line_length: int = 80
    enable_word_wrap: bool = True
    enable_colors: bool = True
    enable_progress_bars: bool = True
    
    # Error handling
    show_stack_traces: bool = False
    continue_on_error: bool = True
    error_prefix: str = "Error: "
    warning_prefix: str = "Warning: "
    
    # Performance settings
    async_input: bool = True
    input_timeout: Optional[float] = None
    output_flush_interval: float = 0.1
    
    # Logging
    enable_logging: bool = True
    log_level: str = "INFO"
    log_user_input: bool = True
    log_commands: bool = True
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class BaseCLI(ABC):
    """
    Abstract base class for CLI interfaces.
    
    This class provides:
    - Common CLI functionality and lifecycle management
    - Configuration-driven behavior
    - Event handling system
    - Input/output abstraction
    - Command processing framework
    - Error handling and recovery
    
    Subclasses should implement:
    - _setup_commands(): Register available commands
    - _handle_user_input(): Process user input
    - _format_output(): Format output for display
    - _cleanup(): Cleanup resources
    """
    
    def __init__(self, config: Optional[CLIConfig] = None, **kwargs):
        """
        Initialize the base CLI.
        
        Args:
            config: CLI configuration
            **kwargs: Additional configuration options
        """
        self.config = config or CLIConfig()
        
        # Apply kwargs to config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.custom_settings[key] = value
        
        # Initialize logging
        self.logger = get_logger(f"cli.{self.config.name}")
        
        # State management
        self.is_initialized = False
        self.is_running = False
        self.should_stop = False
        
        # Input/output handling
        self.input_stream = sys.stdin
        self.output_stream = sys.stdout
        self.error_stream = sys.stderr
        
        # Threading
        self.input_thread: Optional[threading.Thread] = None
        self.output_thread: Optional[threading.Thread] = None
        
        # Command system
        self.commands: Dict[str, Callable] = {}
        self.aliases: Dict[str, str] = {}
        self.command_history: List[str] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'startup': [],
            'shutdown': [],
            'command': [],
            'input': [],
            'output': [],
            'error': []
        }
        
        # Statistics
        self.start_time: Optional[datetime] = None
        self.command_count = 0
        self.error_count = 0
        self.input_count = 0
        self.output_count = 0
        
        self.logger.info(f"CLI {self.config.name} initialized", config=self.config.__dict__)
    
    async def initialize(self) -> None:
        """Initialize the CLI interface."""
        if self.is_initialized:
            return
        
        try:
            # Setup commands
            await self._setup_commands()
            
            # Setup default commands
            await self._setup_default_commands()
            
            # Initialize components
            await self._initialize_components()
            
            # Fire startup events
            await self._fire_event('startup')
            
            self.is_initialized = True
            self.start_time = datetime.now()
            
            self.logger.info(f"CLI {self.config.name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CLI: {e}")
            raise
    
    async def start(self) -> None:
        """Start the CLI interface."""
        if not self.is_initialized:
            await self.initialize()
        
        if self.is_running:
            return
        
        self.is_running = True
        self.should_stop = False
        
        try:
            if self.config.mode == CLIMode.INTERACTIVE:
                await self._start_interactive_mode()
            elif self.config.mode == CLIMode.BATCH:
                await self._start_batch_mode()
            else:
                await self._start_single_command_mode()
                
        except Exception as e:
            self.logger.error(f"Error in CLI execution: {e}")
            await self._handle_error(e)
        finally:
            self.is_running = False
    
    async def stop(self) -> None:
        """Stop the CLI interface."""
        self.should_stop = True
        
        # Wait for threads to finish
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=1.0)
        
        # Fire shutdown events
        await self._fire_event('shutdown')
        
        self.is_running = False
        self.logger.info(f"CLI {self.config.name} stopped")
    
    async def shutdown(self) -> None:
        """Shutdown the CLI interface and cleanup resources."""
        if self.is_running:
            await self.stop()
        
        # Cleanup resources
        await self._cleanup()
        
        # Log statistics
        if self.start_time:
            uptime = datetime.now() - self.start_time
            self.logger.info(
                f"CLI {self.config.name} shutdown complete",
                uptime_seconds=uptime.total_seconds(),
                command_count=self.command_count,
                error_count=self.error_count,
                input_count=self.input_count,
                output_count=self.output_count
            )
        
        self.is_initialized = False
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    async def _setup_commands(self) -> None:
        """Setup CLI-specific commands. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _handle_user_input(self, user_input: str) -> Any:
        """Handle user input processing. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _format_output(self, data: Any) -> str:
        """Format output for display. Override in subclasses."""
        pass
    
    @abstractmethod
    async def _cleanup(self) -> None:
        """Cleanup resources. Override in subclasses."""
        pass
    
    # Command system methods
    
    def register_command(self, name: str, handler: Callable, 
                        description: str = "", aliases: Optional[List[str]] = None) -> None:
        """
        Register a command handler.
        
        Args:
            name: Command name
            handler: Command handler function
            description: Command description
            aliases: Command aliases
        """
        self.commands[name] = handler
        
        if aliases and self.config.enable_aliases:
            for alias in aliases:
                self.aliases[alias] = name
        
        self.logger.debug(f"Registered command: {name}")
    
    async def execute_command(self, command: str, args: List[str] = None) -> Any:
        """
        Execute a command.
        
        Args:
            command: Command name
            args: Command arguments
            
        Returns:
            Command result
        """
        args = args or []
        
        # Resolve aliases
        if command in self.aliases:
            command = self.aliases[command]
        
        # Check if command exists
        if command not in self.commands:
            raise ValueError(f"Unknown command: {command}")
        
        # Execute command
        try:
            self.command_count += 1
            await self._fire_event('command', {'command': command, 'args': args})
            
            handler = self.commands[command]
            if asyncio.iscoroutinefunction(handler):
                result = await handler(*args)
            else:
                result = handler(*args)
            
            # Add to history
            if self.config.enable_history:
                self.command_history.append(f"{command} {' '.join(args)}")
                if len(self.command_history) > self.config.history_size:
                    self.command_history.pop(0)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            await self._handle_error(e)
            raise
    
    # Event system methods
    
    def add_event_handler(self, event: str, handler: Callable) -> None:
        """Add an event handler."""
        if event in self.event_handlers:
            self.event_handlers[event].append(handler)
    
    async def _fire_event(self, event: str, data: Any = None) -> None:
        """Fire an event to all registered handlers."""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler for {event}: {e}")
    
    # Input/output methods
    
    async def get_input(self, prompt: str = None) -> str:
        """Get user input."""
        if prompt is None:
            prompt = self.config.prompt_prefix
        
        try:
            if self.config.async_input:
                # Async input handling
                loop = asyncio.get_event_loop()
                user_input = await loop.run_in_executor(None, input, prompt)
            else:
                user_input = input(prompt)
            
            self.input_count += 1
            await self._fire_event('input', {'input': user_input})
            
            if self.config.log_user_input:
                self.logger.debug(f"User input: {user_input}")
            
            return user_input.strip()
            
        except (EOFError, KeyboardInterrupt):
            self.should_stop = True
            return ""
        except Exception as e:
            await self._handle_error(e)
            return ""
    
    async def print_output(self, message: str, end: str = "\n") -> None:
        """Print output message."""
        try:
            formatted_message = await self._format_output(message)
            
            if self.config.show_timestamps:
                timestamp = datetime.now().strftime('%H:%M:%S')
                formatted_message = f"[{timestamp}] {formatted_message}"
            
            print(formatted_message, end=end, file=self.output_stream)
            
            if self.config.output_flush_interval == 0:
                self.output_stream.flush()
            
            self.output_count += 1
            await self._fire_event('output', {'message': formatted_message})
            
        except Exception as e:
            await self._handle_error(e)
    
    async def print_error(self, message: str) -> None:
        """Print error message."""
        error_message = f"{self.config.error_prefix}{message}"
        print(error_message, file=self.error_stream)
        self.error_stream.flush()
        
        await self._fire_event('error', {'message': error_message})
    
    # Default command implementations
    
    async def _setup_default_commands(self) -> None:
        """Setup default commands available in all CLIs."""
        if self.config.enable_help:
            self.register_command("help", self._cmd_help, "Show available commands", ["?"])
        
        if self.config.enable_quit_commands:
            self.register_command("quit", self._cmd_quit, "Exit the CLI", ["exit", "bye"])
        
        self.register_command("status", self._cmd_status, "Show CLI status")
        self.register_command("history", self._cmd_history, "Show command history")
        self.register_command("clear", self._cmd_clear, "Clear the screen")
    
    async def _cmd_help(self) -> str:
        """Help command implementation."""
        help_lines = ["Available commands:"]
        
        for cmd_name, handler in self.commands.items():
            # Get description from handler docstring or registration
            description = getattr(handler, '__doc__', 'No description')
            if description:
                description = description.split('\n')[0].strip()
            
            # Get aliases
            aliases = [alias for alias, target in self.aliases.items() if target == cmd_name]
            alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
            
            help_lines.append(f"  {cmd_name:<15} - {description}{alias_str}")
        
        return "\n".join(help_lines)
    
    async def _cmd_quit(self) -> str:
        """Quit command implementation."""
        self.should_stop = True
        return "Goodbye!"
    
    async def _cmd_status(self) -> str:
        """Status command implementation."""
        uptime = datetime.now() - self.start_time if self.start_time else None
        
        status_lines = [
            f"CLI Status for {self.config.name}:",
            f"  Version: {self.config.version}",
            f"  Mode: {self.config.mode.value}",
            f"  Running: {self.is_running}",
            f"  Uptime: {uptime.total_seconds():.1f}s" if uptime else "  Uptime: N/A",
            f"  Commands executed: {self.command_count}",
            f"  Errors: {self.error_count}",
            f"  Input count: {self.input_count}",
            f"  Output count: {self.output_count}"
        ]
        
        return "\n".join(status_lines)
    
    async def _cmd_history(self) -> str:
        """History command implementation."""
        if not self.config.enable_history or not self.command_history:
            return "No command history available"
        
        history_lines = ["Command history:"]
        for i, cmd in enumerate(self.command_history[-10:], 1):  # Show last 10
            history_lines.append(f"  {i:2d}. {cmd}")
        
        return "\n".join(history_lines)
    
    async def _cmd_clear(self) -> str:
        """Clear command implementation."""
        os.system('cls' if os.name == 'nt' else 'clear')
        return ""
    
    # Mode-specific implementations
    
    async def _start_interactive_mode(self) -> None:
        """Start interactive mode."""
        await self.print_output(f"🚀 {self.config.description}")
        await self.print_output("=" * 50)
        await self.print_output("Type 'help' for commands or start interacting!")
        await self.print_output("=" * 50)
        
        while not self.should_stop:
            try:
                user_input = await self.get_input(self.config.prompt_prefix)
                
                if not user_input:
                    continue
                
                # Check if it's a command
                if user_input.startswith(self.config.command_prefix):
                    await self._handle_command_input(user_input[1:])
                else:
                    # Handle as regular input
                    result = await self._handle_user_input(user_input)
                    if result is not None:
                        await self.print_output(str(result))
                
            except Exception as e:
                await self._handle_error(e)
                if not self.config.continue_on_error:
                    break
    
    async def _start_batch_mode(self) -> None:
        """Start batch mode."""
        # Read from stdin or file
        for line in self.input_stream:
            line = line.strip()
            if line:
                result = await self._handle_user_input(line)
                if result is not None:
                    await self.print_output(str(result))
    
    async def _start_single_command_mode(self) -> None:
        """Start single command mode."""
        # Execute a single command and exit
        if len(sys.argv) > 1:
            command_line = " ".join(sys.argv[1:])
            result = await self._handle_user_input(command_line)
            if result is not None:
                await self.print_output(str(result))
    
    async def _handle_command_input(self, command_input: str) -> None:
        """Handle command input."""
        parts = command_input.split()
        if not parts:
            return
        
        command = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            result = await self.execute_command(command, args)
            if result is not None:
                await self.print_output(str(result))
        except Exception as e:
            await self.print_error(str(e))
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors."""
        error_message = str(error)
        
        if self.config.show_stack_traces:
            import traceback
            error_message = traceback.format_exc()
        
        await self.print_error(error_message)
        self.logger.error(f"CLI error: {error}")
    
    async def _initialize_components(self) -> None:
        """Initialize CLI components. Override in subclasses."""
        pass 