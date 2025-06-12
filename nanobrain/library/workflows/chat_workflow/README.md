# Chat Workflow

A comprehensive chat workflow implementation using the NanoBrain framework with proper step interconnections via data units, links, and triggers.

## Overview

The Chat Workflow demonstrates the core NanoBrain framework capabilities through a modular, step-based architecture. This workflow has been updated to work with the latest nanobrain package structure while maintaining its original design principles.

## Features

- **Modular Step Architecture**: Individual step directories with clear separation of concerns
- **Data Flow Management**: Proper data flow through data units and links
- **Event-Driven Processing**: Triggers for responsive workflow execution
- **Conversation History**: Persistent storage with SQLite backend
- **Performance Monitoring**: Comprehensive metrics and monitoring
- **Enhanced Agent Integration**: Uses EnhancedCollaborativeAgent with A2A and MCP support

## Architecture

```
User Input → CLI Interface Step → Conversation Manager Step → Agent Processing Step → Output
                                        ↓
                                   History Persistence (substep)
                                   Performance Tracking (substep)
```

## Updated Structure

The workflow has been updated to use the current nanobrain package structure:

### Import Updates
- `from nanobrain.core.*` - Core framework components
- `from nanobrain.library.agents.conversational` - Enhanced collaborative agent
- `from nanobrain.library.infrastructure.data` - Data infrastructure components

### Configuration Updates
- Updated `AgentConfig` parameters to match current API
- String-based data types for data unit configuration
- Proper ConversationHistoryUnit initialization

## Components

### Core Components

1. **ChatWorkflow**: Main workflow orchestrator
2. **EnhancedCollaborativeAgent**: Advanced conversational agent with protocol support
3. **ConversationHistoryUnit**: Persistent conversation storage
4. **Data Units**: Memory-based data storage for workflow state
5. **LocalExecutor**: Workflow execution management

### Data Flow

1. **User Input Data Unit**: Stores incoming user messages
2. **Agent Output Data Unit**: Stores agent responses
3. **Conversation History**: Persistent storage with search capabilities

## Usage

### Basic Usage

```python
from nanobrain.library.workflows.chat_workflow.chat_workflow import create_chat_workflow

# Create and initialize workflow
workflow = await create_chat_workflow()

# Process user input
response = await workflow.process_user_input("Hello, how are you?")
print(response)

# Get workflow status
status = workflow.get_workflow_status()
print(f"Workflow initialized: {status['is_initialized']}")

# Get conversation statistics
stats = await workflow.get_conversation_stats()
print(f"Total messages: {stats['total_messages']}")

# Shutdown
await workflow.shutdown()
```

### Direct Workflow Creation

```python
from nanobrain.library.workflows.chat_workflow.chat_workflow import ChatWorkflow

# Create workflow directly
workflow = ChatWorkflow()
await workflow.initialize()

# Use workflow
response = await workflow.process_user_input("Test message")

# Shutdown
await workflow.shutdown()
```

## Running the Demo

### Interactive Demo

```bash
cd demo/chat_workflow_parsl
python run_chat_workflow_demo.py
```

Choose option 1 for interactive chat or option 2 for automated demo.

### Test Scripts

```bash
# Test imports
python test_chat_workflow_import.py

# Test functionality
python test_updated_chat_workflow.py
```

## Configuration

The workflow uses the following configuration structure:

### Agent Configuration
- **Model**: gpt-3.5-turbo (configurable)
- **Temperature**: 0.7
- **Max Tokens**: 2000
- **System Prompt**: Customizable assistant behavior
- **Enhanced Features**: A2A and MCP protocol support

### Data Units
- **Memory-based**: Fast in-memory storage for workflow state
- **Persistent History**: SQLite database for conversation storage
- **Automatic Cleanup**: Configurable retention policies

### Performance Features
- **Metrics Collection**: Response times, success rates, error tracking
- **Status Monitoring**: Real-time workflow health checks
- **Resource Management**: Proper initialization and cleanup

## API Reference

### ChatWorkflow Class

#### Methods

- `__init__()`: Initialize workflow instance
- `initialize()`: Setup all workflow components
- `process_user_input(message: str) -> str`: Process user message and return response
- `get_workflow_status() -> Dict[str, Any]`: Get current workflow status
- `get_conversation_stats() -> Dict[str, Any]`: Get conversation statistics (async)
- `shutdown()`: Clean shutdown of all components

#### Properties

- `is_initialized`: Boolean indicating if workflow is ready
- `agent`: EnhancedCollaborativeAgent instance
- `data_units`: Dictionary of active data units
- `conversation_history`: ConversationHistoryUnit instance

### Factory Functions

- `create_chat_workflow() -> ChatWorkflow`: Create and initialize workflow

## Error Handling

The workflow includes comprehensive error handling:

- **Graceful Degradation**: Continues operation when non-critical components fail
- **Error Logging**: Detailed error information for debugging
- **Fallback Responses**: Meaningful error messages for users
- **Resource Cleanup**: Proper cleanup even when errors occur

## Performance Considerations

### Memory Usage
- Data units use efficient memory management
- Conversation history includes automatic cleanup
- Resource pooling for database connections

### Response Times
- Optimized agent initialization
- Efficient data unit operations
- Minimal overhead for workflow orchestration

### Scalability
- Modular architecture supports horizontal scaling
- Database backend can be upgraded for production use
- Agent configuration supports different models and parameters

## Migration from Previous Versions

If upgrading from an older version:

1. **Update Imports**: Change to use nanobrain package structure
2. **Agent Configuration**: Update AgentConfig parameters
3. **Data Types**: Use string-based data type specifications
4. **Async Methods**: Use async methods for statistics and cleanup

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure nanobrain package is properly installed
2. **Database Issues**: Check file permissions for SQLite database
3. **Agent Initialization**: Verify API keys and model availability
4. **Memory Issues**: Monitor data unit cache sizes

### Debug Mode

Enable debug logging for detailed execution information:

```python
from nanobrain.core.logging_system import set_debug_mode
set_debug_mode(True)
```

## Future Enhancements

- **Step-based Architecture**: Full implementation of individual step directories
- **Advanced Triggers**: More sophisticated event-driven processing
- **Link Management**: Enhanced data flow between components
- **Performance Optimization**: Further optimization for high-throughput scenarios
- **Protocol Integration**: Enhanced A2A and MCP protocol support

## Contributing

When contributing to this workflow:

1. Maintain the modular architecture
2. Follow nanobrain package structure conventions
3. Update tests for any changes
4. Document new features and configuration options
5. Ensure backward compatibility where possible

---

*For more information, see the [main library documentation](../../README.md) or explore the [workflow examples](../../../demo/chat_workflow_parsl/).* 