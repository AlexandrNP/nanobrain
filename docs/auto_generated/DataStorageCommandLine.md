# DataStorageCommandLine

Data storage for command line input and output.

Biological analogy: Sensory processing and motor output areas.
Justification: Like how sensory areas process input and motor areas
produce output, this class processes command line input and produces
command line output.

## Methods

### Constructor

```python
def __init__(self, executor: ExecutorBase, prompt: str, exit_command: str, welcome_message: Optional[str], goodbye_message: Optional[str], supported_commands: Optional[Dict[str, str]], command_handlers: Optional[Dict[str, Callable]], **kwargs)
```

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

### display_response

```python
def display_response(self, response: Any)
```

Display the response to the command line.

Args:
    response: The response to display

