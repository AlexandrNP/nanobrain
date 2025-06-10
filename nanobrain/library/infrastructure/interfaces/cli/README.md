# NanoBrain CLI Infrastructure

A comprehensive, configurable command-line interface system for the NanoBrain framework.

## Overview

The CLI infrastructure provides a flexible, extensible foundation for building command-line interfaces in NanoBrain applications. It supports multiple operation modes, rich formatting, progress indication, and seamless integration with the NanoBrain workflow system.

## Features

- **🎯 Multiple CLI Types**: Base CLI, Interactive CLI, and Workflow-integrated CLI Step
- **⚙️ Highly Configurable**: Extensive configuration options for behavior, appearance, and functionality
- **🎨 Rich Output**: Multiple output formats (plain, JSON, table, list, tree) with color support
- **📊 Progress Tracking**: Built-in progress indicators with multiple styles
- **🔧 Extensible**: Easy to extend with custom commands and functionality
- **🔄 Async/Await**: Full async support for non-blocking operations
- **📝 Command Processing**: Advanced command parsing with argument validation
- **🎪 Interactive Features**: Auto-completion, command history, session management

## Architecture

```
CLI Infrastructure
├── BaseCLI              # Abstract base class for all CLI implementations
├── InteractiveCLI       # Full-featured interactive CLI
├── CLIStep             # Workflow-integrated CLI step
├── CommandProcessor    # Command parsing and validation
├── ResponseFormatter   # Output formatting and styling
└── ProgressIndicator   # Progress tracking and display
```

## Quick Start

### Basic CLI Usage

```python
from library.infrastructure.interfaces.cli import BaseCLI, CLIConfig

class MyCLI(BaseCLI):
    async def _setup_commands(self):
        self.register_command("greet", self._cmd_greet, "Greet the user")
    
    async def _cmd_greet(self, name="User"):
        return f"Hello, {name}!"
    
    async def _handle_user_input(self, user_input):
        return f"You said: {user_input}"
    
    async def _format_output(self, data):
        return str(data)
    
    async def _cleanup(self):
        pass

# Usage
config = CLIConfig(name="my_cli", prompt_prefix="my> ")
cli = MyCLI(config)
await cli.initialize()
await cli.start()  # Starts interactive mode
```

### Interactive CLI

```python
from library.infrastructure.interfaces.cli import InteractiveCLI, InteractiveCLIConfig

config = InteractiveCLIConfig(
    app_name="My App",
    welcome_message="Welcome to My App!",
    enable_sessions=True,
    enable_autocomplete=True
)

cli = InteractiveCLI(config)
await cli.initialize()
await cli.start()
```

### CLI Step for Workflows

```python
from library.infrastructure.interfaces.cli import CLIStep, CLIStepConfig

config = CLIStepConfig(
    step_name="user_interface",
    prompt_prefix="workflow> "
)

cli_step = CLIStep(config=config)
await cli_step.initialize()

# Process workflow data
result = await cli_step.process({"user_input": "Hello"})
```

## Configuration

### CLIConfig

Base configuration for all CLI types:

```python
@dataclass
class CLIConfig:
    # Basic settings
    name: str = "nanobrain_cli"
    description: str = "NanoBrain CLI Interface"
    version: str = "1.0.0"
    
    # Operation mode
    mode: CLIMode = CLIMode.INTERACTIVE
    
    # Display settings
    theme: CLITheme = CLITheme.DEFAULT
    show_timestamps: bool = True
    prompt_prefix: str = "> "
    
    # Behavior settings
    enable_history: bool = True
    enable_autocomplete: bool = True
    enable_help: bool = True
    
    # Command settings
    command_prefix: str = "/"
    case_sensitive_commands: bool = False
    
    # Error handling
    continue_on_error: bool = True
    show_stack_traces: bool = False
```

### InteractiveCLIConfig

Extended configuration for interactive CLIs:

```python
@dataclass
class InteractiveCLIConfig(CLIConfig):
    # Interactive-specific settings
    app_name: str = "NanoBrain Interactive CLI"
    welcome_message: str = ""
    goodbye_message: str = "Thank you for using NanoBrain!"
    
    # Session management
    enable_sessions: bool = True
    auto_save_session: bool = True
    
    # Advanced features
    enable_plugins: bool = True
    enable_scripting: bool = True
    
    # Customization
    custom_commands: Dict[str, Callable] = field(default_factory=dict)
```

### CLIStepConfig

Configuration for workflow-integrated CLI steps:

```python
@dataclass
class CLIStepConfig(CLIConfig):
    # Step-specific settings
    step_name: str = "cli_step"
    input_key: str = "user_input"
    output_key: str = "cli_output"
    
    # Workflow integration
    auto_start: bool = True
    pass_through_data: bool = True
    interactive_in_workflow: bool = True
```

## Components

### BaseCLI

Abstract base class providing core CLI functionality:

- **Command System**: Register and execute commands with validation
- **Event Handling**: Startup, shutdown, command, input, output, error events
- **Input/Output**: Async input handling and formatted output
- **Lifecycle Management**: Initialize, start, stop, shutdown
- **Configuration**: Extensive configuration options

Key methods to override:
- `_setup_commands()`: Register CLI-specific commands
- `_handle_user_input()`: Process user input
- `_format_output()`: Format output for display
- `_cleanup()`: Cleanup resources

### InteractiveCLI

Full-featured interactive CLI with:

- **Session Management**: Create, save, load sessions
- **Plugin System**: Load and manage plugins
- **Auto-completion**: Command and argument completion
- **Scripting**: Execute script files
- **Rich Features**: Syntax highlighting, progress bars

### CLIStep

Workflow-integrated CLI that combines CLI functionality with Step interface:

- **Workflow Integration**: Process workflow data through CLI
- **Data Unit Support**: Input/output data unit management
- **Event Coordination**: Workflow event handling
- **Step Lifecycle**: Full Step interface implementation

### CommandProcessor

Advanced command parsing and processing:

- **Command Registration**: Register commands with metadata
- **Argument Parsing**: Parse flags, options, and positional arguments
- **Validation**: Argument count and type validation
- **Context Management**: Command execution context
- **Help Generation**: Automatic help text generation

### ResponseFormatter

Output formatting and styling:

- **Multiple Formats**: Plain, JSON, table, list, tree, rich
- **Color Support**: Configurable color schemes
- **Templates**: Customizable output templates
- **Message Types**: Info, success, warning, error formatting

### ProgressIndicator

Progress tracking and display:

- **Multiple Styles**: Bar, spinner, dots, percentage, counter
- **Real-time Updates**: Thread-safe progress updates
- **ETA Calculation**: Estimated time to completion
- **Context Manager**: Easy progress tracking with `with` statement

## Examples

### Custom Commands

```python
class MyCLI(BaseCLI):
    async def _setup_commands(self):
        # Simple command
        self.register_command("hello", self._cmd_hello, "Say hello")
        
        # Command with aliases
        self.register_command("quit", self._cmd_quit, "Exit CLI", ["exit", "bye"])
        
        # Command with arguments
        self.register_command("calc", self._cmd_calc, "Calculate expression")
    
    async def _cmd_hello(self):
        return "Hello, World!"
    
    async def _cmd_quit(self):
        self.should_stop = True
        return "Goodbye!"
    
    async def _cmd_calc(self, expression):
        try:
            result = eval(expression)  # Use proper parser in production
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"
```

### Progress Tracking

```python
from library.infrastructure.interfaces.cli import ProgressIndicator, ProgressStyle

# Create progress indicator
progress = ProgressIndicator()
await progress.initialize()

# Create progress bar
progress_id = progress.create_progress(
    name="processing",
    total=100,
    description="Processing items",
    style=ProgressStyle.BAR
)

# Update progress
for i in range(100):
    progress.update_progress(progress_id, current=i+1)
    await asyncio.sleep(0.1)

# Complete
progress.complete_progress(progress_id, "Processing complete!")
```

### Output Formatting

```python
from library.infrastructure.interfaces.cli import ResponseFormatter, OutputFormat

formatter = ResponseFormatter()
await formatter.initialize()

# Format as table
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]
table_output = await formatter.format_output(data, OutputFormat.TABLE)
print(table_output)

# Format as JSON
json_output = await formatter.format_output(data, OutputFormat.JSON)
print(json_output)
```

## Integration with Existing CLI

The new CLI infrastructure is designed to work alongside the existing CLI interface step. You can:

1. **Migrate gradually**: Replace existing CLI components one by one
2. **Use as base**: Extend the existing CLI with new functionality
3. **Workflow integration**: Use CLIStep in workflows while keeping existing interfaces

### Migration Example

```python
# Old approach (existing CLI interface step)
from library.workflows.chat_workflow.steps.cli_interface_step import CLIInterfaceStep

# New approach (CLI infrastructure)
from library.infrastructure.interfaces.cli import CLIStep, CLIStepConfig

# Create equivalent functionality
config = CLIStepConfig(
    step_name="cli_interface_step",
    prompt_prefix="Chat> ",
    show_timestamps=True,
    enable_commands=True
)

cli_step = CLIStep(
    input_data_unit=input_data_unit,
    output_data_unit=output_data_unit,
    config=config
)
```

## Best Practices

1. **Configuration**: Use configuration objects for customization instead of hardcoding
2. **Error Handling**: Implement proper error handling in command methods
3. **Async/Await**: Use async methods for non-blocking operations
4. **Resource Cleanup**: Always call `shutdown()` to cleanup resources
5. **Command Design**: Keep commands focused and provide good help text
6. **Testing**: Test CLI components with different configurations
7. **Documentation**: Document custom commands and their usage

## Testing

```python
import pytest
from library.infrastructure.interfaces.cli import BaseCLI, CLIConfig

@pytest.mark.asyncio
async def test_cli_basic_functionality():
    config = CLIConfig(name="test_cli")
    cli = TestCLI(config)
    
    await cli.initialize()
    
    # Test command execution
    result = await cli.execute_command("test_command", ["arg1", "arg2"])
    assert result == "expected_result"
    
    # Test input handling
    input_result = await cli._handle_user_input("test input")
    assert input_result is not None
    
    await cli.shutdown()
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the library path is correctly added to `sys.path`
2. **Async Issues**: Make sure to use `await` with async methods
3. **Command Not Found**: Check command registration and case sensitivity settings
4. **Threading Issues**: Progress indicators use threads; ensure proper cleanup

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
config = CLIConfig(
    log_level="DEBUG",
    show_stack_traces=True
)
```

## Contributing

When extending the CLI infrastructure:

1. Follow the existing patterns and interfaces
2. Add comprehensive configuration options
3. Include proper error handling and logging
4. Write tests for new functionality
5. Update documentation

## License

This CLI infrastructure is part of the NanoBrain framework and follows the same licensing terms. 