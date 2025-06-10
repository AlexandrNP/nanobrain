"""
CLI Interface Step

A step that handles command-line interface interactions for the chat workflow.

This step provides:
- User input collection from command line
- Output display to command line
- Command processing and help system
- Interactive chat interface
"""

import sys
import os
import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime

# Add src to path for core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'src'))

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.data_unit import DataUnitBase
from nanobrain.core.executor import ExecutorBase
from nanobrain.core.logging_system import get_logger


class CLIInterfaceStep(Step):
    """
    CLI Interface Step for handling user interactions.
    
    This step manages:
    - Command-line input collection
    - Output display and formatting
    - Interactive chat session management
    - Command processing (help, quit, etc.)
    - User experience enhancements
    """
    
    def __init__(self, input_data_unit: DataUnitBase, output_data_unit: DataUnitBase, 
                 executor: Optional[ExecutorBase] = None, **kwargs):
        """
        Initialize the CLI Interface Step.
        
        Args:
            input_data_unit: Data unit for user input
            output_data_unit: Data unit for output display
            executor: Optional executor for async operations
            **kwargs: Additional configuration
        """
        # Create step configuration
        config = StepConfig(
            name="cli_interface_step",
            description="Command-line interface for user interaction"
        )
        
        super().__init__(config, executor, **kwargs)
        
        # Data units
        self.input_data_unit = input_data_unit
        self.output_data_unit = output_data_unit
        
        # Configuration
        self.prompt_prefix = kwargs.get('prompt_prefix', 'Chat> ')
        self.show_timestamps = kwargs.get('show_timestamps', True)
        self.enable_commands = kwargs.get('enable_commands', True)
        
        # State
        self.is_interactive = False
        self.input_thread = None
        self.should_stop = False
        
        # Logger
        self.logger = get_logger("cli_interface_step", "steps")
        
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process CLI interface operations.
        
        Args:
            inputs: Input data containing user messages or commands
            
        Returns:
            Dict[str, Any]: Processed output for display
        """
        try:
            input_data = inputs.get('input_0', inputs)
            
            if isinstance(input_data, dict):
                message = input_data.get('message', str(input_data))
                command = input_data.get('command')
                
                if command and self.enable_commands:
                    return await self._handle_command(command)
                else:
                    return await self._handle_user_input(message)
            else:
                return await self._handle_user_input(str(input_data))
                
        except Exception as e:
            self.logger.error(f"Error in CLI interface processing: {e}")
            return {'error': str(e), 'message': 'CLI processing failed'}
    
    async def _handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Handle user input processing.
        
        Args:
            user_input: User's input message
            
        Returns:
            Dict[str, Any]: Processed input data
        """
        # Store user input
        await self.input_data_unit.set({
            'message': user_input,
            'timestamp': datetime.now(),
            'source': 'cli'
        })
        
        # Format for display
        if self.show_timestamps:
            timestamp = datetime.now().strftime('%H:%M:%S')
            display_message = f"[{timestamp}] User: {user_input}"
        else:
            display_message = f"User: {user_input}"
        
        return {
            'message': user_input,
            'display': display_message,
            'timestamp': datetime.now(),
            'processed': True
        }
    
    async def _handle_command(self, command: str) -> Dict[str, Any]:
        """
        Handle special commands.
        
        Args:
            command: Command to process
            
        Returns:
            Dict[str, Any]: Command result
        """
        command = command.lower().strip()
        
        if command in ['help', '?']:
            return await self._show_help()
        elif command in ['quit', 'exit', 'bye']:
            return await self._handle_quit()
        elif command == 'status':
            return await self._show_status()
        elif command == 'clear':
            return await self._clear_screen()
        else:
            return {
                'error': f"Unknown command: {command}",
                'message': "Type 'help' for available commands"
            }
    
    async def _show_help(self) -> Dict[str, Any]:
        """Show help information."""
        help_text = """
Available commands:
  help, ?     - Show this help message
  quit, exit  - Exit the chat
  status      - Show workflow status
  clear       - Clear the screen
  
Just type your message to chat with the assistant.
"""
        return {
            'message': help_text,
            'display': help_text,
            'command_result': True
        }
    
    async def _handle_quit(self) -> Dict[str, Any]:
        """Handle quit command."""
        self.should_stop = True
        return {
            'message': 'Goodbye!',
            'display': 'Goodbye!',
            'quit': True
        }
    
    async def _show_status(self) -> Dict[str, Any]:
        """Show workflow status."""
        status_info = f"""
CLI Interface Status:
  Interactive mode: {self.is_interactive}
  Execution count: {self.execution_count}
  Error count: {self.error_count}
  Prompt prefix: {self.prompt_prefix}
  Show timestamps: {self.show_timestamps}
"""
        return {
            'message': status_info,
            'display': status_info,
            'command_result': True
        }
    
    async def _clear_screen(self) -> Dict[str, Any]:
        """Clear the screen."""
        # Clear screen command for different platforms
        os.system('cls' if os.name == 'nt' else 'clear')
        return {
            'message': 'Screen cleared',
            'display': '',
            'command_result': True
        }
    
    async def handle_output(self, output_data: Any) -> None:
        """
        Handle output display.
        
        Args:
            output_data: Data to display
        """
        try:
            if isinstance(output_data, dict):
                message = output_data.get('response', str(output_data))
            else:
                message = str(output_data)
            
            # Format output for display
            if self.show_timestamps:
                timestamp = datetime.now().strftime('%H:%M:%S')
                display_message = f"[{timestamp}] Assistant: {message}"
            else:
                display_message = f"Assistant: {message}"
            
            # Display the message
            print(display_message)
            
            # Store in output data unit
            await self.output_data_unit.set({
                'message': message,
                'display': display_message,
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            self.logger.error(f"Error handling output: {e}")
            print(f"Error displaying output: {e}")
    
    async def start_interactive(self) -> None:
        """Start interactive CLI mode."""
        if self.is_interactive:
            return
        
        self.is_interactive = True
        self.should_stop = False
        
        print("🚀 Starting Enhanced Chat Workflow")
        print("=" * 50)
        print("Type 'help' for commands or just start chatting!")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        self.logger.info("CLI interface started in interactive mode")
    
    def _input_loop(self) -> None:
        """Input loop running in separate thread."""
        while not self.should_stop:
            try:
                user_input = input(self.prompt_prefix).strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:]
                    asyncio.run_coroutine_threadsafe(
                        self._handle_command_sync(command),
                        asyncio.get_event_loop()
                    )
                else:
                    # Handle regular input
                    asyncio.run_coroutine_threadsafe(
                        self._handle_user_input_sync(user_input),
                        asyncio.get_event_loop()
                    )
                    
            except (EOFError, KeyboardInterrupt):
                self.should_stop = True
                break
            except Exception as e:
                self.logger.error(f"Error in input loop: {e}")
    
    async def _handle_command_sync(self, command: str) -> None:
        """Handle command from input thread."""
        result = await self._handle_command(command)
        
        if result.get('quit'):
            self.should_stop = True
        
        if 'display' in result:
            print(result['display'])
    
    async def _handle_user_input_sync(self, user_input: str) -> None:
        """Handle user input from input thread."""
        await self._handle_user_input(user_input)
        # Note: The actual response will be handled by the workflow
        # through the agent processing step
    
    async def stop_interactive(self) -> None:
        """Stop interactive CLI mode."""
        self.should_stop = True
        self.is_interactive = False
        
        if self.input_thread and self.input_thread.is_alive():
            # Wait for input thread to finish
            self.input_thread.join(timeout=1.0)
        
        self.logger.info("CLI interface stopped")
    
    async def shutdown(self) -> None:
        """Shutdown the CLI interface step."""
        await self.stop_interactive()
        await super().shutdown()
        self.logger.info("CLI interface step shutdown complete") 