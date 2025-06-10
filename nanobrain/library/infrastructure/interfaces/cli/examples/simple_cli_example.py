"""
Simple CLI Example

Demonstrates how to use the NanoBrain CLI infrastructure.

This example shows:
- Creating a custom CLI interface
- Extending the base CLI class
- Adding custom commands
- Using the CLI in different modes
"""

import sys
import os
import asyncio
from typing import Dict, Any

# Add library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from library.infrastructure.interfaces.cli import (
    BaseCLI, CLIConfig, CLIMode,
    InteractiveCLI, InteractiveCLIConfig,
    CLIStep, CLIStepConfig,
    create_cli_step
)


class MyCLI(BaseCLI):
    """
    Example custom CLI implementation.
    
    This demonstrates how to extend the base CLI class
    with custom functionality and commands.
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize the custom CLI."""
        super().__init__(config, **kwargs)
        
        # Custom state
        self.user_data = {}
        self.session_count = 0
    
    async def _setup_commands(self) -> None:
        """Setup custom commands."""
        # Register custom commands
        self.register_command("greet", self._cmd_greet, "Greet the user", ["hello", "hi"])
        self.register_command("echo", self._cmd_echo, "Echo back the input")
        self.register_command("calc", self._cmd_calc, "Simple calculator")
        self.register_command("data", self._cmd_data, "Manage user data")
    
    async def _handle_user_input(self, user_input: str) -> Any:
        """Handle regular user input."""
        # Custom input processing
        return {
            'input': user_input,
            'response': f"You said: {user_input}",
            'timestamp': self._get_timestamp()
        }
    
    async def _format_output(self, data: Any) -> str:
        """Format output for display."""
        if isinstance(data, dict) and 'response' in data:
            return data['response']
        return str(data)
    
    async def _cleanup(self) -> None:
        """Cleanup custom resources."""
        self.user_data.clear()
    
    # Custom commands
    
    async def _cmd_greet(self, name: str = "User") -> str:
        """Greet command implementation."""
        self.session_count += 1
        return f"Hello, {name}! This is session #{self.session_count}."
    
    async def _cmd_echo(self, *args) -> str:
        """Echo command implementation."""
        if not args:
            return "Echo: (nothing to echo)"
        return f"Echo: {' '.join(args)}"
    
    async def _cmd_calc(self, expression: str) -> str:
        """Simple calculator command."""
        try:
            # Simple evaluation (in real apps, use a proper parser)
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    async def _cmd_data(self, action: str = "list", key: str = None, value: str = None) -> str:
        """Data management command."""
        if action == "set" and key and value:
            self.user_data[key] = value
            return f"Set {key} = {value}"
        elif action == "get" and key:
            value = self.user_data.get(key, "Not found")
            return f"{key} = {value}"
        elif action == "list":
            if not self.user_data:
                return "No data stored"
            items = [f"{k} = {v}" for k, v in self.user_data.items()]
            return "Stored data:\n" + "\n".join(items)
        elif action == "clear":
            self.user_data.clear()
            return "Data cleared"
        else:
            return "Usage: data <set|get|list|clear> [key] [value]"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


async def example_basic_cli():
    """Example of basic CLI usage."""
    print("=== Basic CLI Example ===")
    
    # Create a basic CLI configuration
    config = CLIConfig(
        name="example_cli",
        description="Example CLI Application",
        prompt_prefix="example> ",
        enable_colors=True,
        show_timestamps=True
    )
    
    # Create and initialize CLI
    cli = MyCLI(config)
    await cli.initialize()
    
    # Simulate some interactions
    print("Simulating CLI interactions...")
    
    # Test regular input
    result = await cli._handle_user_input("Hello, world!")
    print(f"Input result: {await cli._format_output(result)}")
    
    # Test commands
    greet_result = await cli.execute_command("greet", ["Alice"])
    print(f"Greet result: {greet_result}")
    
    echo_result = await cli.execute_command("echo", ["This", "is", "a", "test"])
    print(f"Echo result: {echo_result}")
    
    calc_result = await cli.execute_command("calc", ["2 + 3 * 4"])
    print(f"Calc result: {calc_result}")
    
    # Test data commands
    await cli.execute_command("data", ["set", "name", "Alice"])
    await cli.execute_command("data", ["set", "age", "30"])
    data_result = await cli.execute_command("data", ["list"])
    print(f"Data result: {data_result}")
    
    # Cleanup
    await cli.shutdown()
    print("Basic CLI example completed.\n")


async def example_interactive_cli():
    """Example of interactive CLI usage."""
    print("=== Interactive CLI Example ===")
    
    # Create interactive CLI configuration
    config = InteractiveCLIConfig(
        app_name="Example Interactive CLI",
        welcome_message="Welcome to the Example CLI! Type 'help' for commands.",
        goodbye_message="Thanks for using the Example CLI!",
        enable_sessions=True,
        enable_autocomplete=True
    )
    
    # Add custom commands
    config.custom_commands = {
        "time": lambda: f"Current time: {__import__('datetime').datetime.now()}",
        "random": lambda: f"Random number: {__import__('random').randint(1, 100)}"
    }
    
    # Create interactive CLI
    cli = InteractiveCLI(config)
    await cli.initialize()
    
    print("Interactive CLI created (would start interactive mode in real usage)")
    print("Available commands:", list(cli.commands.keys()))
    
    # Cleanup
    await cli.shutdown()
    print("Interactive CLI example completed.\n")


async def example_cli_step():
    """Example of CLI step usage in workflows."""
    print("=== CLI Step Example ===")
    
    # Create CLI step configuration
    config = CLIStepConfig(
        step_name="example_cli_step",
        step_description="Example CLI step for workflows",
        prompt_prefix="workflow> ",
        auto_start=False,  # Don't auto-start for this example
        interactive_in_workflow=False
    )
    
    # Create CLI step
    cli_step = create_cli_step(config=config)
    await cli_step.initialize()
    
    # Simulate workflow processing
    test_inputs = [
        {"user_input": "Hello from workflow"},
        {"user_input": "/help"},
        {"user_input": "/status"}
    ]
    
    for i, inputs in enumerate(test_inputs):
        print(f"Processing input {i+1}: {inputs}")
        result = await cli_step.process(inputs)
        print(f"Result: {result.get('cli_output', {}).get('message', 'No output')}")
    
    # Cleanup
    await cli_step.shutdown()
    print("CLI Step example completed.\n")


async def main():
    """Run all CLI examples."""
    print("NanoBrain CLI Infrastructure Examples")
    print("=" * 50)
    
    try:
        await example_basic_cli()
        await example_interactive_cli()
        await example_cli_step()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 