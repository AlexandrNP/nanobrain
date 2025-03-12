"""
DataStorageCommandLine Module

This module provides a command-line interface for data storage and interaction.
"""

from typing import Any, List, Optional, Dict, Union, Callable
import asyncio
import sys
import os
from datetime import datetime

from src.DataStorageBase import DataStorageBase
from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase
from src.TriggerBase import TriggerBase
from src.enums import ComponentState


class DataStorageCommandLine(DataStorageBase):
    """
    Data storage for command line input and output.
    
    Biological analogy: Sensory processing and motor output areas.
    Justification: Like how sensory areas process input and motor areas
    produce output, this class processes command line input and produces
    command line output.
    """
    
    def __init__(self, 
                 executor: ExecutorBase,
                 prompt: str = "nb> ",
                 exit_command: str = "exit",
                 welcome_message: Optional[str] = None,
                 goodbye_message: Optional[str] = None,
                 supported_commands: Optional[Dict[str, str]] = None,
                 command_handlers: Optional[Dict[str, Callable]] = None,
                 **kwargs):
        """
        Initialize the CommandLineStorage.
        
        Args:
            executor: The executor responsible for running this step
            prompt: The prompt to display for user input
            exit_command: The command to exit the input loop
            welcome_message: Optional welcome message to display when starting
            goodbye_message: Optional goodbye message to display when exiting
            supported_commands: Dictionary of supported commands and their descriptions
            command_handlers: Dictionary of command handlers
            **kwargs: Additional keyword arguments
        """
        # Initialize with base Step class instead of DataStorageBase
        # to avoid the trigger.runnable = self assignment
        from src.Step import Step
        Step.__init__(self, executor, **kwargs)
        
        # Initialize attributes that would normally be set by DataStorageBase
        self.input = None
        self.output = None
        self.trigger = None
        self.last_query = None
        self.last_response = None
        self.processing_history = []
        self.max_history_size = kwargs.get('max_history_size', 10)
        
        # Command line specific attributes
        self.prompt = prompt
        self.exit_command = exit_command
        self.welcome_message = welcome_message or "Welcome to NanoBrain Command Line Interface. Type 'exit' to quit."
        self.goodbye_message = goodbye_message or "Thank you for using NanoBrain. Goodbye!"
        self.running = False
        self.last_input = None
        self.last_output = None
        
        # Command handling
        self.supported_commands = supported_commands or {}
        self.command_handlers = command_handlers or {}
        
        # Add help command by default
        self.supported_commands["help"] = "Display this help message"
        self.command_handlers["help"] = self._display_help
        
        # Add exit command
        self.supported_commands[self.exit_command] = "Exit the current session"
    
    async def _process_query(self, query: Any) -> Any:
        """
        Process the command line input.
        
        Args:
            query: The user input from command line
            
        Returns:
            The processed response
        """
        # Store the input
        self.last_input = query
        
        # Process the input
        response = await self._handle_command(query)
        
        # Store the output
        self.last_output = response
        
        # Display the response
        self.display_response(response)
        
        return response
    
    async def _handle_command(self, command: str) -> Any:
        """
        Handle a command from the user.
        
        Args:
            command: The command to handle
            
        Returns:
            The response to the command
        """
        if not command:
            return None
            
        # Parse the command
        parts = command.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        # Check if it's a supported command
        if cmd in self.command_handlers:
            # Call the handler with arguments
            handler = self.command_handlers[cmd]
            if asyncio.iscoroutinefunction(handler):
                return await handler(*args)
            return handler(*args)
        
        # If no specific handler, treat as general input
        return command
    
    def _display_help(self, *args) -> str:
        """Display help information about supported commands."""
        help_text = "Available commands:\n"
        for cmd, desc in self.supported_commands.items():
            help_text += f"  {cmd} - {desc}\n"
        return help_text
    
    def display_response(self, response: Any):
        """
        Display the response to the command line.
        
        Args:
            response: The response to display
        """
        if response:
            print(f"{response}")
    
    async def _get_user_input(self) -> str:
        """
        Get input from the command line asynchronously.
        
        Returns:
            The user input as a string
        """
        # Create a future to hold the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Run the blocking input call in a separate thread
        def _get_input():
            try:
                result = input(self.prompt)
                loop.call_soon_threadsafe(future.set_result, result)
            except (EOFError, KeyboardInterrupt) as e:
                loop.call_soon_threadsafe(future.set_exception, e)
        
        # Run the input function in a thread
        await loop.run_in_executor(None, _get_input)
        
        # Wait for the result
        return await future
    
    async def start_monitoring(self):
        """
        Start monitoring for user input.
        
        This method runs an infinite loop that waits for user input
        and processes it until the exit command is received.
        """
        # Set the state to active
        self._state = ComponentState.ACTIVE
        self.running = True
        
        # Display welcome message
        if self.welcome_message:
            print(self.welcome_message)
        
        # Display help by default
        print(self._display_help())
        
        # Main input loop
        while self.running:
            try:
                # Wait for user input
                user_input = await self._get_user_input()
                
                # Check if it's the exit command
                if user_input.lower() == self.exit_command.lower():
                    self.running = False
                    if self.goodbye_message:
                        print(self.goodbye_message)
                    break
                
                # Process the input
                await self.process([user_input])
                
            except asyncio.CancelledError:
                self.running = False
                break
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled by user.")
                self.running = False
                if self.goodbye_message:
                    print(self.goodbye_message)
                break
            except Exception as e:
                print(f"Error processing input: {e}")
                # Continue the loop despite errors
    
    async def stop_monitoring(self):
        """
        Stop monitoring for user input.
        """
        # Set the state to inactive
        self._state = ComponentState.INACTIVE
        self.running = False
    
    def _update_history(self, query: Any, response: Any) -> None:
        """
        Update the processing history.
        
        Args:
            query: The processed query
            response: The generated response
        """
        # Add to history
        self.processing_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history if it exceeds max size
        if len(self.processing_history) > self.max_history_size:
            self.processing_history = self.processing_history[-self.max_history_size:]
            
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process the input data and produce output.
        
        Biological analogy: Memory retrieval and encoding.
        Justification: Like how memory systems retrieve stored information
        based on cues and encode new information, this method processes
        input queries and produces appropriate responses.
        
        Args:
            inputs: List of input data (typically a single query)
            
        Returns:
            The processed output data
        """
        # Extract the query from inputs
        query = inputs[0] if inputs else None
        self.last_query = query
        
        # If no query provided, return None
        if query is None:
            return None
        
        # Process the query
        response = await self._process_query(query)
        self.last_response = response
        
        # Record this processing in history
        self._update_history(query, response)
        
        return response 