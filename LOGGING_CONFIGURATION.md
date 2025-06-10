# NanoBrain Logging Configuration System

## Overview

The NanoBrain framework now includes a comprehensive logging configuration system that allows you to control where and how logging information is output. This system provides three distinct logging modes to suit different use cases and environments.

## Logging Modes

### 🖥️ Console Mode (`"console"`)
- **Description**: Logs only to console/terminal output
- **Use Case**: Interactive development, debugging, real-time monitoring
- **Behavior**: 
  - All log messages appear in the terminal
  - No log files are created
  - Ideal for development and testing

### 📁 File Mode (`"file"`)
- **Description**: Logs only to files (silent console operation)
- **Use Case**: Production environments, automated systems, background processes
- **Behavior**:
  - All log messages are written to organized log files
  - Console output is completely silent (except for user input prompts)
  - Perfect for production deployments and automated workflows

### 🔄 Both Mode (`"both"`) - Default
- **Description**: Logs to both console and files
- **Use Case**: Development with persistent logging, comprehensive monitoring
- **Behavior**:
  - Log messages appear in both console and files
  - Full visibility and persistent storage
  - Best for development and debugging

## Configuration

### Global Configuration File

The logging mode is configured in `nanobrain/config/global_config.yml`:

```yaml
# Logging Configuration
logging:
  # Logging mode: "console", "file", or "both"
  mode: "both"  # Change this to "console", "file", or "both"
  
  # Log level for configuration loading
  level: "INFO"
  
  # File logging specific settings (when mode is "file" or "both")
  file:
    base_directory: "logs"
    use_session_directories: true
    max_file_size_mb: 10
    backup_count: 5
    compress_old_logs: true
  
  # Console logging specific settings (when mode is "console" or "both")
  console:
    use_colors: true
    show_timestamps: true
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Programmatic Access

You can also check and use the logging configuration programmatically:

```python
from src.config import get_logging_mode, should_log_to_console, should_log_to_file

# Check current logging mode
mode = get_logging_mode()  # Returns "console", "file", or "both"

# Check specific capabilities
console_enabled = should_log_to_console()  # Returns True/False
file_enabled = should_log_to_file()        # Returns True/False
```

## Automatic Logger Configuration

The `NanoBrainLogger` class automatically respects the global logging configuration:

```python
from core.logging_system import NanoBrainLogger
from pathlib import Path

# Logger automatically uses global configuration
logger = NanoBrainLogger(
    name="my_component",
    log_file=Path("logs/my_component.log"),
    debug_mode=True
    # enable_console and enable_file are determined automatically
)

# Check what was configured
print(f"Console enabled: {logger.enable_console}")
print(f"File enabled: {logger.enable_file}")
```

## Chat Workflow Demo Integration

The enhanced chat workflow demo (`nanobrain/demo/chat_workflow_demo.py`) fully respects the logging configuration:

### Console Mode Example
```bash
# Set mode to "console" in global_config.yml
python demo/chat_workflow_demo.py
# Output: Full interactive experience with console messages
```

### File Mode Example
```bash
# Set mode to "file" in global_config.yml
python demo/chat_workflow_demo.py
# Output: Silent operation, all logs go to files in logs/ directory
```

### Both Mode Example
```bash
# Set mode to "both" in global_config.yml (default)
python demo/chat_workflow_demo.py
# Output: Interactive console + comprehensive file logging
```

## Log File Organization

When file logging is enabled, logs are organized in a structured directory hierarchy:

```
logs/
└── session_20250606_103105/
    ├── agents/
    │   └── chat_assistant.log
    ├── components/
    │   ├── chat_workflow.log
    │   └── cli_interface.log
    ├── data_units/
    ├── links/
    │   ├── user_to_agent_link.log
    │   ├── agent_input_to_step_link.log
    │   └── step_to_output_link.log
    ├── steps/
    │   └── conversational_agent_step.log
    ├── triggers/
    │   ├── user_input_trigger.log
    │   ├── agent_input_trigger.log
    │   └── agent_output_trigger.log
    └── session_summary.json
```

## Use Cases

### Development and Debugging
```yaml
logging:
  mode: "both"  # See everything in console + keep files for analysis
```

### Production Deployment
```yaml
logging:
  mode: "file"  # Silent operation, comprehensive file logging
```

### Interactive Testing
```yaml
logging:
  mode: "console"  # Real-time feedback, no file clutter
```

### CI/CD Pipelines
```yaml
logging:
  mode: "file"  # Capture all logs for analysis, silent execution
```

## Advanced Features

### Session Management
- Each run creates a timestamped session directory
- Session summaries include metadata and performance metrics
- Automatic cleanup of old log sessions (configurable retention)

### Structured Logging
- JSON-formatted log entries for easy parsing
- Rich metadata including timestamps, request IDs, and context
- Performance metrics and execution tracing

### Log Rotation
- Configurable file size limits
- Automatic backup and compression
- Prevents disk space issues in long-running processes

## Migration Guide

### From Previous Versions
If you're upgrading from a previous version of NanoBrain:

1. **No changes required** - The system defaults to `"both"` mode (previous behavior)
2. **Optional**: Set your preferred mode in `global_config.yml`
3. **Optional**: Customize file and console logging settings

### Environment Variables
The logging configuration respects environment variables for dynamic configuration:

```bash
# Override logging mode via environment
export NANOBRAIN_LOGGING_MODE="file"
python demo/chat_workflow_demo.py
```

## Troubleshooting

### Issue: No console output in file mode
**Expected behavior** - File mode is designed to be silent for production use.

### Issue: No log files in console mode
**Expected behavior** - Console mode doesn't create files to avoid clutter.

### Issue: Logs appear twice in both mode
**Expected behavior** - Both mode intentionally outputs to both destinations.

### Issue: Configuration not taking effect
**Solution**: Ensure `global_config.yml` is in the correct location and properly formatted.

## Performance Considerations

- **Console Mode**: Fastest, no file I/O overhead
- **File Mode**: Minimal console overhead, optimized file writing
- **Both Mode**: Slight overhead from dual output, but comprehensive coverage

## Security Notes

- Log files may contain sensitive information
- File mode is recommended for production to avoid console information leakage
- API keys and sensitive data are automatically masked in logs
- Log file permissions should be restricted in production environments

## Examples

### Quick Mode Switch
```python
# Temporarily override logging mode
import os
os.environ['NANOBRAIN_LOGGING_MODE'] = 'file'

# Your NanoBrain code here
```

### Custom Logger with Mode Respect
```python
from core.logging_system import NanoBrainLogger
from src.config import should_log_to_file, should_log_to_console

# Create logger that respects global configuration
logger = NanoBrainLogger(
    name="custom_component",
    log_file=Path("logs/custom.log") if should_log_to_file() else None,
    enable_console=should_log_to_console(),
    enable_file=should_log_to_file()
)
```

This logging configuration system provides the flexibility needed for different deployment scenarios while maintaining the rich debugging capabilities that make NanoBrain powerful for development and production use. 