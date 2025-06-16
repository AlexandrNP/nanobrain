# Enhanced Logging System for NanoBrain

## Overview

NanoBrain's enhanced logging system provides comprehensive, process-safe, and concurrency-aware logging capabilities that work seamlessly across distributed execution environments including Parsl workers, async operations, and multi-process scenarios.

## Key Features

### ✅ **Concurrency-Safe Architecture** 
- **Process-safe logging**: Works across main process and Parsl workers
- **Queue-based async logging**: Non-blocking background processing  
- **No file locking conflicts**: Concurrent agents can log simultaneously
- **Context-aware execution**: Automatically detects execution environment

### ✅ **Fixed Enum Serialization**
- **Before**: `"name": "<max_depth_exceeded: str>", "value": "<max_depth_exceeded: str>"`
- **After**: `"name": "MEMORY", "value": "memory"` 
- **Critical attributes preserved**: Names and values always visible for debugging

### ✅ **Guaranteed Agent I/O Logging**
- **Always captured**: Agent inputs/outputs logged regardless of configuration
- **Process-boundary safe**: Works in distributed Parsl execution
- **Interaction context**: Structured logging with metadata and performance metrics
- **No silent failures**: Robust error handling with fallback mechanisms

### ✅ **Clean Inheritance Hierarchy**
- **Parent class isolation**: Base `Agent` class doesn't create redundant loggers
- **Concrete instance detection**: Only `SimpleAgent`, `ConversationalAgent`, etc. get full logging
- **No initialization pollution**: Clean separation between abstract and concrete classes

### ✅ **Comprehensive Component Tracking**
- **Structured categories**: `agents/`, `data/`, `workflows/`, `parsl/`, `nanobrain/`
- **Performance metrics**: Execution times, token usage, error rates
- **Lifecycle events**: Initialization, shutdown, state changes
- **Metadata preservation**: Full context and debugging information

## Architecture

### Process-Safe Logging Infrastructure

```python
# nanobrain/core/async_logging.py
class ProcessSafeLogger:
    """Thread-safe, queue-based logger that works across process boundaries."""
    
    def __init__(self, name: str, category: str = "components"):
        self.execution_context = self._detect_execution_context()
        
        if self.execution_context == "main_process":
            self._setup_main_process_logging()
        elif self.execution_context == "parsl_worker":  
            self._setup_worker_process_logging()
        else:
            self._setup_fallback_logging()
```

### Agent Logger with Inheritance Detection

```python
# nanobrain/core/agent_logging.py  
class AgentLogger:
    """Specialized logger for agent interactions with concrete instance detection."""
    
    def __init__(self, agent_name: str, agent_type: str):
        # Only concrete agents (SimpleAgent, ConversationalAgent) get full logging
        self._is_concrete_instance = self._is_concrete_agent_instance(agent_type)
        
        if self._is_concrete_instance:
            self.logger = get_process_safe_logger(agent_name, category="agents")
        else:
            self.logger = None  # Parent classes get minimal logging
```

### Guaranteed I/O Capture

```python
# Always logged regardless of settings
async with self.agent_logger.interaction_context(input_text) as context:
    result = await self.process(input_text, **kwargs)
    
    if context:
        context['response_text'] = result
        context['llm_calls'] = self._total_llm_calls - initial_llm_calls
        context['total_tokens'] = self._total_tokens_used - initial_total_tokens
    
    return result
```

## Log Structure

### Agent Interaction Logs
```json
{
  "timestamp": "2025-06-13T19:41:17.079989+00:00",
  "level": "INFO", 
  "logger_name": "test_agent",
  "message": "Agent interaction #1",
  "metadata": {
    "agent_name": "test_agent",
    "agent_type": "ConversationalAgent", 
    "input_text": "Hello, how are you?",
    "response_text": "Hello! How can I help?",
    "input_length": 19,
    "response_length": 21,
    "duration_ms": 425.56,
    "llm_calls": 1,
    "total_tokens": 45,
    "success": true,
    "execution_context": "main_79391"
  },
  "category": "agents",
  "process_id": 79391,
  "thread_id": 8214597504
}
```

### LLM Call Logs
```json
{
  "timestamp": "2025-06-13T19:41:18.043363+00:00",
  "level": "INFO",
  "logger_name": "test_agent", 
  "message": "LLM call completed",
  "metadata": {
    "model": "gpt-3.5-turbo-0125",
    "messages_count": 6,
    "response_preview": "Hello! How can I help?",
    "response_length": 21,
    "tokens_used": 72,
    "finish_reason": "stop", 
    "duration_ms": 367.19
  },
  "category": "agents"
}
```

### Fixed Enum Serialization
```json
{
  "timestamp": "2025-06-13T19:33:32.032300+00:00",
  "level": "INFO",
  "message": "DataUnit initialize: test_unit",
  "metadata": {
    "data_type": {
      "name": "MEMORY",     // ✅ Now visible!
      "value": "memory"     // ✅ Now visible!
    },
    "persistent": false
  }
}
```

## Usage Examples

### Basic Agent Logging
```python
from nanobrain.core.agent import ConversationalAgent, AgentConfig

config = AgentConfig(
    name="my_agent",
    description="Test agent",
    enable_logging=True,      # Always respected
    log_conversations=True    # Always captured
)

agent = ConversationalAgent(config)
await agent.initialize()

# This interaction will ALWAYS be logged
result = await agent.execute("Hello, how are you?")
```

### Process-Safe Manual Logging
```python
from nanobrain.core.async_logging import get_process_safe_logger

# Works in main process or Parsl workers
logger = get_process_safe_logger("my_component", "workflows")

logger.info("Processing started", user_id=123, task_type="analysis")
logger.debug("Intermediate result", step=5, progress=0.75)
logger.error("Processing failed", error_type="timeout", retry_count=3)
```

### Concurrent Safe Operations
```python
# Multiple agents can log simultaneously without conflicts
agents = [ConversationalAgent(AgentConfig(name=f"agent_{i}")) for i in range(10)]

async def process_concurrently():
    tasks = [agent.execute(f"Task {i}") for i, agent in enumerate(agents)]
    results = await asyncio.gather(*tasks)  # All I/O properly logged
    
    return results
```

## File Organization

```
logs/
├── session_20250613_144308/
│   ├── agents/
│   │   ├── my_agent.log           # Agent interactions, LLM calls
│   │   └── worker_agent.log       # Distributed agent logs
│   ├── data/
│   │   ├── memory_unit.log        # Data unit operations  
│   │   └── file_storage.log       # File operations
│   ├── workflows/
│   │   └── chat_workflow.log      # Workflow execution
│   ├── parsl/
│   │   └── distributed_tasks.log  # Parsl worker logs
│   └── nanobrain/
│       └── system.log             # System-level events
```

## Migration from Old System

### Before (Issues):
```python
# ❌ Multiple logger initialization from parent classes
# ❌ Enum serialization broken: "<max_depth_exceeded: str>"
# ❌ Agent I/O missing due to concurrency issues  
# ❌ File locking conflicts with concurrent agents
# ❌ Silent failures in distributed execution

class Agent:
    def __init__(self):
        self.nb_logger = get_logger(...)  # ❌ Created for ALL classes
```

### After (Fixed):
```python  
# ✅ Clean inheritance - only concrete instances get loggers
# ✅ Enum serialization: "name": "MEMORY", "value": "memory"
# ✅ Guaranteed I/O capture with interaction contexts
# ✅ Process-safe concurrent logging
# ✅ Robust distributed execution support

class Agent:
    def __init__(self):
        self.agent_logger = AgentLogger(...)  # ✅ Smart detection
        
class ConversationalAgent(Agent):  # ✅ Gets full logging
class SimpleAgent(Agent):          # ✅ Gets full logging  
```

## Performance Impact

- **Minimal overhead**: Background queue processing
- **Non-blocking**: Async I/O doesn't slow down agents
- **Batched writes**: Efficient file operations
- **Memory efficient**: Bounded queue sizes
- **Concurrent safe**: No performance degradation with multiple agents

## Error Handling

- **Graceful degradation**: Falls back to console logging if files unavailable
- **No exceptions**: Logging errors don't break agent execution  
- **Retry mechanisms**: Automatic recovery from temporary failures
- **Process isolation**: Worker failures don't affect main process logging

## Validation

All improvements verified with comprehensive test suite:

- ✅ **Enum serialization fix**: Names/values properly visible
- ✅ **Process-safe logging**: No conflicts across processes/threads
- ✅ **Agent I/O capture**: 100% interaction logging regardless of config
- ✅ **Parent class pollution fix**: Clean inheritance hierarchy  
- ✅ **Concurrency safety**: Verified with concurrent operations
- ✅ **File organization**: Proper categorization and structure

The logging system now provides enterprise-grade reliability, debugging capability, and performance monitoring for the NanoBrain framework. 