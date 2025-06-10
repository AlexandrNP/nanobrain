"""
Interactive CLI

Full-featured interactive command-line interface for NanoBrain applications.

This module provides:
- Rich interactive CLI experience
- Advanced command processing
- User experience enhancements
- Session management
- Extensible plugin system
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from .base_cli import BaseCLI, CLIConfig, CLIMode
from .command_processor import CommandProcessor
from .response_formatter import ResponseFormatter
from .progress_indicator import ProgressIndicator


@dataclass
class InteractiveCLIConfig(CLIConfig):
    """Configuration for interactive CLI."""
    
    # Interactive-specific settings
    app_name: str = "NanoBrain Interactive CLI"
    welcome_message: str = ""
    goodbye_message: str = "Thank you for using NanoBrain!"
    
    # Session management
    enable_sessions: bool = True
    session_timeout: Optional[float] = None
    auto_save_session: bool = True
    
    # User experience
    enable_autocomplete: bool = True
    enable_syntax_highlighting: bool = True
    enable_rich_output: bool = True
    enable_paging: bool = True
    
    # Advanced features
    enable_plugins: bool = True
    plugin_directories: List[str] = field(default_factory=list)
    enable_scripting: bool = True
    script_extensions: List[str] = field(default_factory=lambda: ['.nb', '.nanobrain'])
    
    # Customization
    custom_prompt_function: Optional[Callable] = None
    custom_formatter: Optional[Callable] = None
    custom_commands: Dict[str, Callable] = field(default_factory=dict)


class InteractiveCLI(BaseCLI):
    """
    Interactive CLI for NanoBrain applications.
    
    This class provides:
    - Rich interactive command-line experience
    - Advanced command processing and completion
    - Session management and persistence
    - Plugin system for extensibility
    - User experience enhancements
    
    Features:
    - Auto-completion and syntax highlighting
    - Command history and session management
    - Progress indicators and rich output
    - Plugin system for custom functionality
    - Scripting support for automation
    """
    
    def __init__(self, config: Optional[InteractiveCLIConfig] = None, **kwargs):
        """
        Initialize the Interactive CLI.
        
        Args:
            config: Interactive CLI configuration
            **kwargs: Additional configuration options
        """
        self.interactive_config = config or InteractiveCLIConfig()
        
        # Force interactive mode
        self.interactive_config.mode = CLIMode.INTERACTIVE
        
        # Initialize base CLI
        super().__init__(self.interactive_config, **kwargs)
        
        # Components
        self.command_processor = CommandProcessor()
        self.response_formatter = ResponseFormatter()
        self.progress_indicator = ProgressIndicator()
        
        # Session management
        self.current_session = None
        self.session_data = {}
        
        # Plugin system
        self.plugins = {}
        self.plugin_commands = {}
        
        # State
        self.completion_enabled = False
        
        self.logger.info(f"Interactive CLI {self.interactive_config.app_name} initialized")
    
    async def initialize(self) -> None:
        """Initialize the interactive CLI."""
        await super().initialize()
        
        # Initialize components
        await self.command_processor.initialize()
        await self.response_formatter.initialize()
        await self.progress_indicator.initialize()
        
        # Setup completion if enabled
        if self.interactive_config.enable_autocomplete:
            await self._setup_completion()
        
        # Load plugins if enabled
        if self.interactive_config.enable_plugins:
            await self._load_plugins()
        
        # Start session if enabled
        if self.interactive_config.enable_sessions:
            await self._start_session()
        
        self.logger.info("Interactive CLI initialized successfully")
    
    async def _setup_commands(self) -> None:
        """Setup interactive CLI specific commands."""
        # Session commands
        if self.interactive_config.enable_sessions:
            self.register_command("session", self._cmd_session, "Session management", ["sess"])
            self.register_command("save", self._cmd_save, "Save current session")
            self.register_command("load", self._cmd_load, "Load a session")
        
        # Plugin commands
        if self.interactive_config.enable_plugins:
            self.register_command("plugins", self._cmd_plugins, "Plugin management")
            self.register_command("plugin", self._cmd_plugin, "Plugin operations")
        
        # Scripting commands
        if self.interactive_config.enable_scripting:
            self.register_command("script", self._cmd_script, "Execute script file")
            self.register_command("record", self._cmd_record, "Record commands to script")
        
        # Utility commands
        self.register_command("config", self._cmd_config, "Show configuration")
        self.register_command("theme", self._cmd_theme, "Change CLI theme")
        
        # Add custom commands
        for name, handler in self.interactive_config.custom_commands.items():
            self.register_command(name, handler, f"Custom command: {name}")
    
    async def _handle_user_input(self, user_input: str) -> Any:
        """
        Handle user input with enhanced processing.
        
        Args:
            user_input: User's input
            
        Returns:
            Processed result
        """
        # Process through command processor
        processed_input = await self.command_processor.process_input(user_input)
        
        # Handle different input types
        if processed_input.get('is_command'):
            return await self._handle_command_input(processed_input)
        else:
            return await self._handle_regular_input(processed_input)
    
    async def _handle_command_input(self, processed_input: Dict[str, Any]) -> Any:
        """Handle command input."""
        command = processed_input.get('command')
        args = processed_input.get('args', [])
        
        try:
            result = await self.execute_command(command, args)
            return await self.response_formatter.format_command_result(result)
        except Exception as e:
            return await self.response_formatter.format_error(str(e))
    
    async def _handle_regular_input(self, processed_input: Dict[str, Any]) -> Any:
        """Handle regular input (non-command)."""
        # Default implementation - can be overridden
        return {
            'type': 'regular_input',
            'input': processed_input.get('text', ''),
            'processed': True,
            'timestamp': self._get_timestamp()
        }
    
    async def _format_output(self, data: Any) -> str:
        """Format output using the response formatter."""
        return await self.response_formatter.format_output(data)
    
    async def _cleanup(self) -> None:
        """Cleanup interactive CLI resources."""
        # Save session if enabled
        if self.interactive_config.enable_sessions and self.interactive_config.auto_save_session:
            await self._save_current_session()
        
        # Cleanup components
        await self.command_processor.cleanup()
        await self.response_formatter.cleanup()
        await self.progress_indicator.cleanup()
        
        # Unload plugins
        await self._unload_plugins()
    
    # Session management
    
    async def _start_session(self) -> None:
        """Start a new session."""
        from datetime import datetime
        
        self.current_session = {
            'id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now(),
            'commands': [],
            'data': {}
        }
        
        self.logger.info(f"Started session: {self.current_session['id']}")
    
    async def _save_current_session(self) -> None:
        """Save the current session."""
        if not self.current_session:
            return
        
        # Implementation would save to file or database
        self.logger.info(f"Session {self.current_session['id']} saved")
    
    # Plugin system
    
    async def _load_plugins(self) -> None:
        """Load plugins from configured directories."""
        for plugin_dir in self.interactive_config.plugin_directories:
            if os.path.exists(plugin_dir):
                await self._load_plugins_from_directory(plugin_dir)
    
    async def _load_plugins_from_directory(self, plugin_dir: str) -> None:
        """Load plugins from a specific directory."""
        # Plugin loading implementation
        self.logger.info(f"Loading plugins from: {plugin_dir}")
    
    async def _unload_plugins(self) -> None:
        """Unload all plugins."""
        for plugin_name in list(self.plugins.keys()):
            await self._unload_plugin(plugin_name)
    
    async def _unload_plugin(self, plugin_name: str) -> None:
        """Unload a specific plugin."""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            self.logger.info(f"Unloaded plugin: {plugin_name}")
    
    # Completion system
    
    async def _setup_completion(self) -> None:
        """Setup command completion."""
        try:
            # Try to import readline for completion
            import readline
            
            # Set completion function
            readline.set_completer(self._complete_command)
            readline.parse_and_bind("tab: complete")
            
            self.completion_enabled = True
            self.logger.info("Command completion enabled")
            
        except ImportError:
            self.logger.warning("Readline not available, completion disabled")
    
    def _complete_command(self, text: str, state: int) -> Optional[str]:
        """Command completion function."""
        if state == 0:
            # Generate completion options
            if text.startswith(self.config.command_prefix):
                # Complete commands
                command_text = text[1:]  # Remove prefix
                self._completion_options = [
                    f"{self.config.command_prefix}{cmd}" 
                    for cmd in self.commands.keys() 
                    if cmd.startswith(command_text)
                ]
            else:
                # Complete other text (could be file names, etc.)
                self._completion_options = []
        
        try:
            return self._completion_options[state]
        except IndexError:
            return None
    
    # Interactive CLI commands
    
    async def _cmd_session(self, *args) -> str:
        """Session management command."""
        if not args:
            if self.current_session:
                return f"Current session: {self.current_session['id']}"
            else:
                return "No active session"
        
        action = args[0].lower()
        
        if action == "new":
            await self._start_session()
            return f"Started new session: {self.current_session['id']}"
        elif action == "save":
            await self._save_current_session()
            return "Session saved"
        elif action == "info":
            if self.current_session:
                info = [
                    f"Session ID: {self.current_session['id']}",
                    f"Start time: {self.current_session['start_time']}",
                    f"Commands executed: {len(self.current_session['commands'])}"
                ]
                return "\n".join(info)
            else:
                return "No active session"
        else:
            return f"Unknown session action: {action}"
    
    async def _cmd_save(self, filename: str = None) -> str:
        """Save session command."""
        await self._save_current_session()
        return f"Session saved{f' as {filename}' if filename else ''}"
    
    async def _cmd_load(self, filename: str) -> str:
        """Load session command."""
        # Implementation would load from file
        return f"Session loaded from {filename}"
    
    async def _cmd_plugins(self) -> str:
        """Plugin management command."""
        if not self.plugins:
            return "No plugins loaded"
        
        plugin_info = ["Loaded plugins:"]
        for name, plugin in self.plugins.items():
            plugin_info.append(f"  {name}: {getattr(plugin, 'description', 'No description')}")
        
        return "\n".join(plugin_info)
    
    async def _cmd_plugin(self, action: str, plugin_name: str = None) -> str:
        """Plugin operations command."""
        if action == "load" and plugin_name:
            # Load specific plugin
            return f"Loading plugin: {plugin_name}"
        elif action == "unload" and plugin_name:
            await self._unload_plugin(plugin_name)
            return f"Unloaded plugin: {plugin_name}"
        elif action == "reload" and plugin_name:
            await self._unload_plugin(plugin_name)
            # Reload plugin
            return f"Reloaded plugin: {plugin_name}"
        else:
            return "Usage: plugin <load|unload|reload> <plugin_name>"
    
    async def _cmd_script(self, filename: str) -> str:
        """Execute script command."""
        if not os.path.exists(filename):
            return f"Script file not found: {filename}"
        
        try:
            with open(filename, 'r') as f:
                commands = f.readlines()
            
            results = []
            for line in commands:
                line = line.strip()
                if line and not line.startswith('#'):
                    result = await self._handle_user_input(line)
                    results.append(str(result))
            
            return f"Executed {len(results)} commands from {filename}"
            
        except Exception as e:
            return f"Error executing script: {e}"
    
    async def _cmd_record(self, action: str = "start", filename: str = None) -> str:
        """Record commands command."""
        # Implementation for command recording
        if action == "start":
            return "Started recording commands"
        elif action == "stop":
            return f"Stopped recording{f', saved to {filename}' if filename else ''}"
        else:
            return "Usage: record <start|stop> [filename]"
    
    async def _cmd_config(self) -> str:
        """Show configuration command."""
        config_info = [
            f"Configuration for {self.interactive_config.app_name}:",
            f"  Mode: {self.interactive_config.mode.value}",
            f"  Theme: {self.interactive_config.theme.value}",
            f"  Sessions enabled: {self.interactive_config.enable_sessions}",
            f"  Plugins enabled: {self.interactive_config.enable_plugins}",
            f"  Autocomplete enabled: {self.interactive_config.enable_autocomplete}",
            f"  Rich output enabled: {self.interactive_config.enable_rich_output}"
        ]
        
        return "\n".join(config_info)
    
    async def _cmd_theme(self, theme_name: str = None) -> str:
        """Change theme command."""
        if not theme_name:
            return f"Current theme: {self.interactive_config.theme.value}"
        
        # Implementation would change theme
        return f"Changed theme to: {theme_name}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    # Override start method to show welcome message
    
    async def start(self) -> None:
        """Start the interactive CLI with welcome message."""
        if self.interactive_config.welcome_message:
            await self.print_output(self.interactive_config.welcome_message)
        
        await super().start()
        
        if self.interactive_config.goodbye_message:
            await self.print_output(self.interactive_config.goodbye_message)


# Convenience function for creating interactive CLIs
def create_interactive_cli(config: Optional[InteractiveCLIConfig] = None, **kwargs) -> InteractiveCLI:
    """
    Create an interactive CLI with optional configuration.
    
    Args:
        config: Interactive CLI configuration
        **kwargs: Additional configuration options
        
    Returns:
        Configured interactive CLI
    """
    return InteractiveCLI(config=config, **kwargs) 