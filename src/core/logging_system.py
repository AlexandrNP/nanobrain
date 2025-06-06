"""
Advanced Logging System for NanoBrain Framework

Provides comprehensive logging, tracing, and monitoring capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
import functools
import inspect
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LogLevel(Enum):
    """Log levels for NanoBrain operations."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class OperationType(Enum):
    """Types of operations that can be logged."""
    AGENT_PROCESS = "agent_process"
    TOOL_CALL = "tool_call"
    STEP_EXECUTE = "step_execute"
    DATA_TRANSFER = "data_transfer"
    TRIGGER_ACTIVATE = "trigger_activate"
    EXECUTOR_RUN = "executor_run"
    LLM_CALL = "llm_call"
    WORKFLOW_RUN = "workflow_run"

@dataclass
class ExecutionContext:
    """Context information for execution tracking."""
    request_id: str
    operation_type: OperationType
    component_name: str
    parent_request_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.start_time is None:
            self.start_time = time.time()

@dataclass
class ToolCallLog:
    """Detailed logging for tool calls."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class AgentConversationLog:
    """Logging for agent conversations."""
    agent_name: str
    input_text: str
    response_text: str = ""
    tool_calls: List[ToolCallLog] = None
    llm_calls: int = 0
    total_tokens: Optional[int] = None
    duration_ms: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

class NanoBrainLogger:
    """
    Advanced logger for NanoBrain framework with structured logging,
    execution tracing, and performance monitoring.
    """
    
    def __init__(self, name: str, log_file: Optional[Path] = None, 
                 enable_console: Optional[bool] = None, enable_file: Optional[bool] = None,
                 debug_mode: bool = False):
        self.name = name
        self.debug_mode = debug_mode
        self.logger = logging.getLogger(name)
        
        # Get global logging configuration if available
        try:
            # Try to import config manager
            import sys
            import os
            config_path = Path(__file__).parent.parent.parent / "config"
            sys.path.insert(0, str(config_path))
            
            from config_manager import should_log_to_console, should_log_to_file, get_logging_config
            
            # Use global configuration if local parameters not specified
            if enable_console is None:
                enable_console = should_log_to_console()
            if enable_file is None:
                enable_file = should_log_to_file()
                
            # Get logging configuration
            logging_config = get_logging_config()
            console_config = logging_config.get('console', {})
            
        except ImportError:
            # Fallback to defaults if config manager not available
            if enable_console is None:
                enable_console = True
            if enable_file is None:
                enable_file = True
            console_config = {}
        
        # Set log level based on debug mode
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            
            # Use configured format or default
            console_format = console_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_formatter = logging.Formatter(console_format)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file and log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Store configuration for reference
        self.enable_console = enable_console
        self.enable_file = enable_file
        
        # Execution context stack
        self._context_stack: List[ExecutionContext] = []
        self._performance_metrics: Dict[str, List[float]] = {}
        self._conversation_logs: List[AgentConversationLog] = []
        self._tool_call_logs: List[ToolCallLog] = []
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())[:8]
    
    def _get_current_context(self) -> Optional[ExecutionContext]:
        """Get the current execution context."""
        return self._context_stack[-1] if self._context_stack else None
    
    def _log_structured(self, level: LogLevel, message: str, 
                       context: Optional[ExecutionContext] = None, **kwargs):
        """Log a structured message with context."""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "message": message,
            "logger": self.name
        }
        
        if context:
            log_data["context"] = asdict(context)
        
        log_data.update(kwargs)
        
        # Log to appropriate level
        if level == LogLevel.TRACE or level == LogLevel.DEBUG:
            self.logger.debug(json.dumps(log_data, default=str))
        elif level == LogLevel.INFO:
            self.logger.info(json.dumps(log_data, default=str))
        elif level == LogLevel.WARNING:
            self.logger.warning(json.dumps(log_data, default=str))
        elif level == LogLevel.ERROR:
            self.logger.error(json.dumps(log_data, default=str))
        elif level == LogLevel.CRITICAL:
            self.logger.critical(json.dumps(log_data, default=str))
    
    @contextmanager
    def execution_context(self, operation_type: OperationType, 
                         component_name: str, **metadata):
        """Context manager for tracking execution."""
        request_id = self._generate_request_id()
        parent_id = self._get_current_context().request_id if self._get_current_context() else None
        
        context = ExecutionContext(
            request_id=request_id,
            operation_type=operation_type,
            component_name=component_name,
            parent_request_id=parent_id,
            metadata=metadata
        )
        
        self._context_stack.append(context)
        
        self._log_structured(
            LogLevel.DEBUG,
            f"Starting {operation_type.value} in {component_name}",
            context,
            operation="start"
        )
        
        try:
            yield context
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error_message = str(e)
            self._log_structured(
                LogLevel.ERROR,
                f"Error in {operation_type.value}: {e}",
                context,
                operation="error",
                error_type=type(e).__name__
            )
            raise
            
        finally:
            context.end_time = time.time()
            context.duration_ms = (context.end_time - context.start_time) * 1000
            
            # Record performance metrics
            op_key = f"{operation_type.value}:{component_name}"
            if op_key not in self._performance_metrics:
                self._performance_metrics[op_key] = []
            self._performance_metrics[op_key].append(context.duration_ms)
            
            self._log_structured(
                LogLevel.DEBUG,
                f"Completed {operation_type.value} in {component_name}",
                context,
                operation="complete",
                duration_ms=context.duration_ms,
                success=context.success
            )
            
            self._context_stack.pop()
    
    @asynccontextmanager
    async def async_execution_context(self, operation_type: OperationType, 
                                    component_name: str, **metadata):
        """Async context manager for tracking execution."""
        request_id = self._generate_request_id()
        parent_id = self._get_current_context().request_id if self._get_current_context() else None
        
        context = ExecutionContext(
            request_id=request_id,
            operation_type=operation_type,
            component_name=component_name,
            parent_request_id=parent_id,
            metadata=metadata
        )
        
        self._context_stack.append(context)
        
        self._log_structured(
            LogLevel.DEBUG,
            f"Starting async {operation_type.value} in {component_name}",
            context,
            operation="start"
        )
        
        try:
            yield context
            context.success = True
            
        except Exception as e:
            context.success = False
            context.error_message = str(e)
            self._log_structured(
                LogLevel.ERROR,
                f"Error in async {operation_type.value}: {e}",
                context,
                operation="error",
                error_type=type(e).__name__
            )
            raise
            
        finally:
            context.end_time = time.time()
            context.duration_ms = (context.end_time - context.start_time) * 1000
            
            # Record performance metrics
            op_key = f"{operation_type.value}:{component_name}"
            if op_key not in self._performance_metrics:
                self._performance_metrics[op_key] = []
            self._performance_metrics[op_key].append(context.duration_ms)
            
            self._log_structured(
                LogLevel.DEBUG,
                f"Completed async {operation_type.value} in {component_name}",
                context,
                operation="complete",
                duration_ms=context.duration_ms,
                success=context.success
            )
            
            self._context_stack.pop()
    
    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any], 
                     result: Any = None, error: Optional[str] = None,
                     duration_ms: Optional[float] = None):
        """Log a tool call with parameters and results."""
        tool_log = ToolCallLog(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            error=error,
            duration_ms=duration_ms
        )
        
        self._tool_call_logs.append(tool_log)
        
        self._log_structured(
            LogLevel.INFO,
            f"Tool call: {tool_name}",
            tool_call=asdict(tool_log)
        )
    
    def log_agent_conversation(self, agent_name: str, input_text: str,
                             response_text: str = "", tool_calls: List[ToolCallLog] = None,
                             llm_calls: int = 0, total_tokens: Optional[int] = None,
                             duration_ms: Optional[float] = None):
        """Log an agent conversation with full context."""
        conv_log = AgentConversationLog(
            agent_name=agent_name,
            input_text=input_text,
            response_text=response_text,
            tool_calls=tool_calls or [],
            llm_calls=llm_calls,
            total_tokens=total_tokens,
            duration_ms=duration_ms
        )
        
        self._conversation_logs.append(conv_log)
        
        self._log_structured(
            LogLevel.INFO,
            f"Agent conversation: {agent_name}",
            conversation=asdict(conv_log)
        )
    
    def log_step_execution(self, step_name: str, inputs: Dict[str, Any],
                          outputs: Any = None, duration_ms: Optional[float] = None,
                          success: bool = True, error: Optional[str] = None):
        """Log step execution details."""
        self._log_structured(
            LogLevel.INFO,
            f"Step execution: {step_name}",
            step_name=step_name,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms,
            success=success,
            error=error
        )
    
    def log_data_transfer(self, source: str, destination: str, 
                         data_type: str, size_bytes: Optional[int] = None):
        """Log data transfer between components."""
        self._log_structured(
            LogLevel.DEBUG,
            f"Data transfer: {source} -> {destination}",
            source=source,
            destination=destination,
            data_type=data_type,
            size_bytes=size_bytes
        )
    
    def log_trigger_activation(self, trigger_name: str, trigger_type: str,
                             conditions: Dict[str, Any], activated: bool = True):
        """Log trigger activation."""
        self._log_structured(
            LogLevel.DEBUG,
            f"Trigger {'activated' if activated else 'checked'}: {trigger_name}",
            trigger_name=trigger_name,
            trigger_type=trigger_type,
            conditions=conditions,
            activated=activated
        )
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics summary."""
        summary = {}
        for operation, times in self._performance_metrics.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "total_ms": sum(times)
                }
        return summary
    
    def get_conversation_history(self, agent_name: Optional[str] = None) -> List[AgentConversationLog]:
        """Get conversation history, optionally filtered by agent."""
        if agent_name:
            return [log for log in self._conversation_logs if log.agent_name == agent_name]
        return self._conversation_logs.copy()
    
    def get_tool_call_history(self, tool_name: Optional[str] = None) -> List[ToolCallLog]:
        """Get tool call history, optionally filtered by tool name."""
        if tool_name:
            return [log for log in self._tool_call_logs if log.tool_name == tool_name]
        return self._tool_call_logs.copy()
    
    def clear_logs(self):
        """Clear all logged data."""
        self._conversation_logs.clear()
        self._tool_call_logs.clear()
        self._performance_metrics.clear()
    
    # Convenience methods for different log levels
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log_structured(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_structured(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_structured(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_structured(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_structured(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_structured(LogLevel.CRITICAL, message, **kwargs)


def trace_function_calls(logger: NanoBrainLogger):
    """Decorator to trace function calls with parameters and results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Get function signature for parameter logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Filter out sensitive parameters
            safe_params = {}
            for name, value in bound_args.arguments.items():
                if name in ['password', 'token', 'key', 'secret']:
                    safe_params[name] = "***REDACTED***"
                else:
                    safe_params[name] = str(value)[:100]  # Truncate long values
            
            with logger.execution_context(
                OperationType.EXECUTOR_RUN,
                func_name,
                parameters=safe_params
            ) as context:
                result = func(*args, **kwargs)
                context.metadata['result_type'] = type(result).__name__
                return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Get function signature for parameter logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Filter out sensitive parameters
            safe_params = {}
            for name, value in bound_args.arguments.items():
                if name in ['password', 'token', 'key', 'secret']:
                    safe_params[name] = "***REDACTED***"
                else:
                    safe_params[name] = str(value)[:100]  # Truncate long values
            
            async with logger.async_execution_context(
                OperationType.EXECUTOR_RUN,
                func_name,
                parameters=safe_params
            ) as context:
                result = await func(*args, **kwargs)
                context.metadata['result_type'] = type(result).__name__
                return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global logger instances
_global_loggers: Dict[str, NanoBrainLogger] = {}

def get_logger(name: str = "nanobrain", **kwargs) -> NanoBrainLogger:
    """Get or create a NanoBrain logger instance."""
    global _global_loggers
    if name not in _global_loggers:
        _global_loggers[name] = NanoBrainLogger(name, **kwargs)
    return _global_loggers[name]

def set_debug_mode(enabled: bool = True):
    """Enable or disable debug mode globally."""
    global _global_loggers
    for logger in _global_loggers.values():
        logger.debug_mode = enabled
        if enabled:
            logger.logger.setLevel(logging.DEBUG)
        else:
            logger.logger.setLevel(logging.INFO) 