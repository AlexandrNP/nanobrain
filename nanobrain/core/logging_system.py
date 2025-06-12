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
import sys

# Configure structured logging conditionally based on global configuration
def _configure_global_logging():
    """Configure global logging based on NanoBrain configuration."""
    try:
        # Check if config module is already loaded to avoid circular imports
        if 'config' in sys.modules:
            config_module = sys.modules['config']
            should_log_to_console = getattr(config_module, 'should_log_to_console', None)
            should_log_to_file = getattr(config_module, 'should_log_to_file', None)
            get_logging_config = getattr(config_module, 'get_logging_config', None)
            
            if should_log_to_console and should_log_to_file and get_logging_config:
                # Get logging configuration
                logging_config = get_logging_config()
                console_config = logging_config.get('console', {})
                
                console_enabled = should_log_to_console()
                file_enabled = should_log_to_file()
                
                # Debug output (remove in production)
                # print(f"DEBUG: _configure_global_logging called - console: {console_enabled}, file: {file_enabled}")
                
                # Configure root logger based on mode
                root_logger = logging.getLogger()
                
                # Clear any existing handlers to ensure clean configuration
                root_logger.handlers.clear()
                
                if console_enabled:
                    # Console logging enabled - set up console handler
                    console_format = console_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(logging.Formatter(console_format))
                    root_logger.addHandler(console_handler)
                    root_logger.setLevel(logging.INFO)
                else:
                    # File-only mode - use null handler to suppress all console output
                    null_handler = logging.NullHandler()
                    root_logger.addHandler(null_handler)
                    root_logger.setLevel(logging.CRITICAL)  # Set very high level to suppress everything
                    
                    # Suppress third-party library console output
                    _suppress_third_party_console_logging()
                return
        
        # If config not available, try to import it
        from ..config import should_log_to_console, should_log_to_file, get_logging_config
        
        # Get logging configuration
        logging_config = get_logging_config()
        console_config = logging_config.get('console', {})
        
        console_enabled = should_log_to_console()
        file_enabled = should_log_to_file()
        
        # Configure root logger based on mode
        root_logger = logging.getLogger()
        
        # Clear any existing handlers to ensure clean configuration
        root_logger.handlers.clear()
        
        if console_enabled:
            # Console logging enabled - set up console handler
            console_format = console_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(console_format))
            root_logger.addHandler(console_handler)
            root_logger.setLevel(logging.INFO)
        else:
            # File-only mode - use null handler to suppress all console output
            null_handler = logging.NullHandler()
            root_logger.addHandler(null_handler)
            root_logger.setLevel(logging.CRITICAL)  # Set very high level to suppress everything
            
            # Suppress third-party library console output
            _suppress_third_party_console_logging()
            
    except ImportError:
        # Fallback to default configuration if config manager not available
        # Debug output (remove in production)
        # print("DEBUG: _configure_global_logging fallback - using console logging")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def _suppress_third_party_console_logging():
    """Suppress console logging for common third-party libraries when in file-only mode."""
    # List of common third-party libraries that might log to console
    third_party_loggers = [
        # Standard HTTP libraries
        'httpx',
        'httpcore', 
        'urllib3',
        'requests',
        
        # API client libraries
        'openai',
        'anthropic',
        'google.cloud',
        'azure',
        'cohere',
        
        # Python standard libraries that may log
        'asyncio',
        'concurrent.futures',
        'multiprocessing',
        'threading',
        
        # NanoBrain specific
        'config.config_manager',
        'workflows',
        'nanobrain',
        'agent',
        'enhanced',
        'workflow',
        'library',
        'src',
        
        # Parsl and distributed computing
        'parsl',
        'parsl.dataflow.dflow',
        'parsl.executors',
        'parsl.executors.high_throughput.executor',
        'parsl.executors.high_throughput.zmq_pipes',
        'parsl.executors.status_handling',
        'parsl.jobs.strategy',
        'parsl.jobs.job_status_poller',
        'parsl.providers',
        'parsl.providers.local.local',
        'parsl.monitoring',
        'parsl.utils',
        'parsl.process_loggers',
        'parsl.dataflow.memoization',
        'parsl.usage_tracking.usage',
        'parsl.log_utils',
        'parsl.config',
        'parsl.app.app',
        'parsl.dataflow.futures'
    ]
    
    # Aggressively suppress all console output for these loggers
    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        
        # Critical: clear all handlers to prevent any console output
        logger.handlers.clear()
        
        # Add null handler to prevent "No handlers found" warnings
        logger.addHandler(logging.NullHandler())
        
        # Set level to CRITICAL to suppress almost everything
        logger.setLevel(logging.CRITICAL)
        
        # Most important: prevent propagation to root logger which might have console handlers
        logger.propagate = False

def _configure_comprehensive_parsl_logging():
    """Configure Parsl logging to redirect all output to semantic directory structure."""
    try:
        # Import here to avoid circular imports
        global _system_log_manager
        
        # Only configure if we have a system log manager
        if _system_log_manager is None:
            return  # Don't interfere with normal logging initialization
        
        # Get the parsl directory from the existing session
        if hasattr(_system_log_manager, 'semantic_structure'):
            parsl_dir_name = _system_log_manager.semantic_structure.get('distributed_processing', 'parsl')
            parsl_log_dir = _system_log_manager.session_dir / parsl_dir_name
        else:
            parsl_log_dir = _system_log_manager.session_dir / "parsl"
        
        # Create parsl log directory
        parsl_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual log files for different Parsl components as per documentation
        parsl_loggers = {
            'parsl': 'parsl.log',
            'parsl.dataflow.dflow': 'parsl_dataflow_dflow.log',
            'parsl.executors': 'parsl_executors.log',
            'parsl.monitoring': 'parsl_monitoring.log',
            'parsl.dataflow.memoization': 'parsl_dataflow_memoization.log',
        }
        
        # Check if we should log to console
        try:
            from ..config import should_log_to_console
            console_enabled = should_log_to_console()
        except ImportError:
            console_enabled = True
        
        # Configure each Parsl logger with its own file
        for logger_name, log_filename in parsl_loggers.items():
            logger = logging.getLogger(logger_name)
            
            # Create file handler for this specific logger
            try:
                log_file = parsl_log_dir / log_filename
                file_handler = logging.FileHandler(str(log_file))
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                
                # Set appropriate log level
                logger.setLevel(logging.INFO)
                
                # Remove existing handlers to avoid duplicates
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                
                # Add the file handler
                logger.addHandler(file_handler)
                
                # Add console handler only if console logging is enabled
                if console_enabled:
                    console_handler = logging.StreamHandler()
                    console_formatter = logging.Formatter(
                        'PARSL - %(name)s - %(levelname)s - %(message)s'
                    )
                    console_handler.setFormatter(console_formatter)
                    logger.addHandler(console_handler)
                
                # Prevent propagation to avoid duplicate messages
                logger.propagate = False
                
            except Exception as e:
                # If file handler creation fails, use null handler
                logger.addHandler(logging.NullHandler())
                logger.propagate = False
        
        # Configure Parsl's runinfo directory to use our semantic structure
        import os
        os.environ['PARSL_RUNINFO_DIR'] = str(parsl_log_dir / "runinfo")
            
    except Exception as e:
        # Fallback: if configuration fails, use basic suppression
        for logger_name in ['parsl', 'parsl.dataflow', 'parsl.executors']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())

# Initialize global logging configuration
_configure_global_logging()

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
                 debug_mode: bool = False, create_on_first_message: bool = True):
        self.name = name
        self.debug_mode = debug_mode
        self.log_file = log_file
        self.create_on_first_message = create_on_first_message
        self.file_created = False  # Track if file has been created
        self.file_handler = None  # Initialize file_handler attribute
        
        # Get global logging configuration
        try:
            # Check if sys.modules contains the config module to avoid circular imports
            if 'nanobrain.config' in sys.modules:
                from ..config import should_log_to_console, should_log_to_file
                self.enable_console = should_log_to_console() if enable_console is None else enable_console
                self.enable_file = should_log_to_file() if enable_file is None else enable_file
            else:
                # Default behavior if config module not available
                self.enable_console = True if enable_console is None else enable_console
                self.enable_file = True if enable_file is None else enable_file
        except ImportError:
            # Fallback configuration
            self.enable_console = True if enable_console is None else enable_console
            self.enable_file = True if enable_file is None else enable_file
        
        # Setup base logger
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
            
            # Console handler
            if self.enable_console:
                console_handler = logging.StreamHandler()
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            
            # File handler (created lazily if create_on_first_message is True)
            if self.enable_file and self.log_file and not self.create_on_first_message:
                self._create_file_handler()
        
        # Performance tracking
        self._execution_contexts: Dict[str, ExecutionContext] = {}
        self._performance_metrics: Dict[str, List[float]] = {}
        self._conversation_logs: List[AgentConversationLog] = []
        self._tool_call_logs: List[ToolCallLog] = []
        
        # Configure Parsl logging if needed
        self._configure_parsl_logging()
        
        # Additional parameters
        self.create_on_first_message = create_on_first_message
    
    def _create_file_handler(self):
        """Create the file handler for logging."""
        if self.log_file and not self.file_created:
            # Ensure parent directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            self.file_handler = logging.FileHandler(str(self.log_file))
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.file_handler.setFormatter(file_formatter)
            self.logger.addHandler(self.file_handler)
            self.file_created = True
    
    def _ensure_file_handler(self):
        """Ensure file handler exists and is properly configured."""
        if not self.file_handler and self.enable_file and self.log_file:
            # Create parent directories if they don't exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create file handler with JSON formatter
            self.file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter('%(message)s')  # Raw message for JSON
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
            
            # Log initialization message
            init_message = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'INFO',
                'logger': self.name,
                'message': 'Logger initialized',
                'log_file': str(self.log_file),
                'debug_mode': self.debug_mode
            }
            self.logger.info(json.dumps(init_message))

    def _configure_parsl_logging(self):
        """Configure Parsl logging if needed."""
        try:
            if not self.enable_console and self.enable_file:
                # In file-only mode, suppress all console output from Parsl
                
                # First, specifically suppress httpx logger which is noisy
                httpx_logger = logging.getLogger('httpx')
                httpx_logger.handlers.clear()
                httpx_logger.addHandler(logging.NullHandler())
                httpx_logger.propagate = False
                httpx_logger.setLevel(logging.CRITICAL)
                
                # Also suppress other HTTP client loggers
                for logger_name in ['urllib3', 'requests', 'httpcore']:
                    http_logger = logging.getLogger(logger_name)
                    http_logger.handlers.clear()
                    http_logger.addHandler(logging.NullHandler())
                    http_logger.propagate = False
                    http_logger.setLevel(logging.CRITICAL)
                
                # Then suppress the Parsl loggers
                if 'parsl' in sys.modules:
                    parsl_logger = logging.getLogger('parsl')
                    parsl_logger.handlers.clear()
                    parsl_logger.addHandler(logging.NullHandler())
                    parsl_logger.propagate = False
                    parsl_logger.setLevel(logging.CRITICAL)
                    
                    # Also suppress specific Parsl submodules
                    for parsl_submodule in [
                        'parsl.dataflow', 
                        'parsl.executors',
                        'parsl.providers'
                    ]:
                        sub_logger = logging.getLogger(parsl_submodule)
                        sub_logger.handlers.clear()
                        sub_logger.addHandler(logging.NullHandler())
                        sub_logger.propagate = False
                        sub_logger.setLevel(logging.CRITICAL)
        except Exception as e:
            # Don't let logging configuration issues break functionality
            pass

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return str(uuid.uuid4())[:8]
    
    def _get_current_context(self) -> Optional[ExecutionContext]:
        """Get the current execution context."""
        return self._execution_contexts.get(self._generate_request_id())
    
    def _serialize_data_for_logging(self, data: Any, max_length: int = 1000, max_depth: int = 3) -> Any:
        """
        Serialize data for logging in a readable format.
        
        Args:
            data: The data to serialize
            max_length: Maximum length for string representations
            max_depth: Maximum depth for nested structures
            
        Returns:
            Serialized data suitable for logging
        """
        def _serialize_recursive(obj: Any, current_depth: int = 0) -> Any:
            if current_depth > max_depth:
                return f"<max_depth_exceeded: {type(obj).__name__}>"
            
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                if isinstance(obj, str) and len(obj) > max_length:
                    return f"{obj[:max_length]}... <truncated, total_length: {len(obj)}>"
                return obj
            elif isinstance(obj, dict):
                if len(obj) == 0:
                    return {}
                # Limit number of keys shown
                items = list(obj.items())[:10]  # Show first 10 items
                result = {}
                for k, v in items:
                    key_str = str(k)[:50]  # Limit key length
                    result[key_str] = _serialize_recursive(v, current_depth + 1)
                if len(obj) > 10:
                    result["<additional_keys>"] = f"{len(obj) - 10} more items"
                return result
            elif isinstance(obj, (list, tuple)):
                if len(obj) == 0:
                    return []
                # Limit number of items shown
                items = obj[:5]  # Show first 5 items
                result = [_serialize_recursive(item, current_depth + 1) for item in items]
                if len(obj) > 5:
                    result.append(f"<additional_items: {len(obj) - 5} more>")
                return result
            elif hasattr(obj, '__dict__'):
                # Handle custom objects
                try:
                    obj_dict = {
                        "__type__": type(obj).__name__,
                        "__module__": getattr(type(obj), '__module__', 'unknown')
                    }
                    # Add a few key attributes
                    for attr in ['name', 'id', 'value', 'data', 'content'][:3]:
                        if hasattr(obj, attr):
                            obj_dict[attr] = _serialize_recursive(getattr(obj, attr), current_depth + 1)
                    return obj_dict
                except Exception:
                    return f"<{type(obj).__name__} object>"
            else:
                # Fallback to string representation
                str_repr = str(obj)
                if len(str_repr) > max_length:
                    return f"{str_repr[:max_length]}... <{type(obj).__name__}>"
                return str_repr
        
        return _serialize_recursive(data)

    def _log_structured(self, level: LogLevel, message: str, 
                       context: Optional[ExecutionContext] = None, **kwargs):
        """Log a structured message with context and metadata."""
        try:
            # Ensure file handler exists if needed
            if self.enable_file and not self.file_handler:
                self._ensure_file_handler()
            
            # Get current execution context if not provided
            if context is None:
                context = self._get_current_context()
            
            # Build structured log entry
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': level.value,
                'logger': self.name,
                'message': message
            }
            
            # Add execution context if available
            if context:
                log_entry.update({
                    'request_id': context.request_id,
                    'operation_type': context.operation_type.value,
                    'component_name': context.component_name,
                    'parent_request_id': context.parent_request_id,
                    'duration_ms': context.duration_ms,
                    'success': context.success,
                    'error_message': context.error_message
                })
                
                # Add context metadata if available
                if context.metadata:
                    log_entry['context'] = self._serialize_data_for_logging(context.metadata)
            
            # Add any additional kwargs
            if kwargs:
                log_entry['metadata'] = self._serialize_data_for_logging(kwargs)
            
            # Convert to string for logging
            log_message = json.dumps(log_entry)
            
            # Log using appropriate level
            if level == LogLevel.TRACE:
                self.logger.debug(log_message)  # Map TRACE to DEBUG
            elif level == LogLevel.DEBUG:
                self.logger.debug(log_message)
            elif level == LogLevel.INFO:
                self.logger.info(log_message)
            elif level == LogLevel.WARNING:
                self.logger.warning(log_message)
            elif level == LogLevel.ERROR:
                self.logger.error(log_message)
            elif level == LogLevel.CRITICAL:
                self.logger.critical(log_message)
                
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            basic_message = f"{message} (Logging error: {str(e)})"
            self.logger.error(basic_message)

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
        
        self._execution_contexts[request_id] = context
        
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
            
            del self._execution_contexts[request_id]
    
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
        
        self._execution_contexts[request_id] = context
        
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
            
            del self._execution_contexts[request_id]
    
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
    
    def log_data_unit_operation(self, operation: str, data_unit_name: str, 
                               data: Any = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Log data unit operations with detailed content logging.
        
        Args:
            operation: The operation performed ('read', 'write', 'clear', etc.)
            data_unit_name: Name of the data unit
            data: The data involved in the operation
            metadata: Additional metadata about the operation
        """
        log_entry = {
            "operation": operation,
            "data_unit": data_unit_name,
            "data_type": type(data).__name__ if data is not None else "None",
            "data_size": len(str(data)) if data is not None else 0
        }
        
        if metadata:
            log_entry["metadata"] = metadata
        
        # Include actual data content for better visibility
        if data is not None:
            log_entry["data"] = data
        
        self._log_structured(
            LogLevel.INFO,
            f"DataUnit {operation}: {data_unit_name}",
            **log_entry
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


# Global registry for loggers and system components
_global_loggers: Dict[str, NanoBrainLogger] = {}
_system_log_manager: Optional['SystemLogManager'] = None


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


class SystemLogManager:
    """
    Comprehensive system-level logging manager for NanoBrain framework.
    
    Provides organized logging with:
    - Component-based directory structure (agents/, steps/, triggers/, etc.)
    - Session summaries and metadata tracking
    - Lifecycle logging for all framework components
    - Performance and execution tracking
    """
    
    def __init__(self, base_log_dir: Optional[str] = None):
        # Get global logging configuration
        try:
            from ..config import get_logging_config, should_log_to_file, should_log_to_console
            
            self.logging_config = get_logging_config()
            self.should_log_to_file = should_log_to_file()
            self.should_log_to_console = should_log_to_console()
            
            file_config = self.logging_config.get('file', {})
            if base_log_dir is None:
                base_log_dir = file_config.get('base_directory', 'logs')
            
            # Check if we should use semantic directories instead of timestamped sessions
            self.use_session_directories = file_config.get('use_session_directories', True)
            self.use_semantic_directories = file_config.get('use_semantic_directories', True)
            self.semantic_structure = file_config.get('semantic_structure', {
                'nanobrain_components': 'nanobrain',
                'distributed_processing': 'parsl', 
                'workflows': 'workflows',
                'agents': 'agents',
                'data_units': 'data'
            })
            
            # File creation settings
            self.create_on_first_message = file_config.get('create_on_first_message', True)
            self.avoid_empty_files = file_config.get('avoid_empty_files', True)
            
        except ImportError:
            self.logging_config = {}
            self.should_log_to_file = True
            self.should_log_to_console = True
            if base_log_dir is None:
                base_log_dir = "logs"
            self.use_session_directories = True
            self.use_semantic_directories = True
            self.semantic_structure = {
                'nanobrain_components': 'nanobrain',
                'distributed_processing': 'parsl',
                'workflows': 'workflows', 
                'agents': 'agents',
                'data_units': 'data'
            }
            self.create_on_first_message = True
            self.avoid_empty_files = True
        
        self.base_log_dir = Path(base_log_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start_time = datetime.now().isoformat()
        
        # Set up base directory structure
        if self.use_session_directories and self.use_semantic_directories:
            # Use both: logs/session_YYYYMMDD_HHMMSS/nanobrain/, etc.
            self.session_dir = self.base_log_dir / f"session_{self.session_id}"
        elif self.use_semantic_directories:
            # Use semantic organization only: logs/nanobrain/, logs/parsl/, etc.
            self.session_dir = self.base_log_dir
        elif self.use_session_directories:
            # Use timestamped sessions only: logs/session_YYYYMMDD_HHMMSS/
            self.session_dir = self.base_log_dir / f"session_{self.session_id}"
        else:
            # Use direct logging: logs/
            self.session_dir = self.base_log_dir
            
        self.loggers_created = []
        self.log_files = []
        self.component_registry = {}
        
        if self.should_log_to_file:
            self._setup_directories()
        
    def _setup_directories(self):
        """Set up the logging directory structure."""
        # Get configuration - self.logging_config already contains the logging section
        file_config = self.logging_config.get('file', {})
        
        # Create base log directory if it doesn't exist
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory if enabled
        if file_config.get('use_session_directories', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_dir = self.base_log_dir / f"session_{timestamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create session info file
            session_info = {
                'start_time': datetime.now().isoformat(),
                'framework_version': '1.0.0',
                'python_version': sys.version,
                'platform': sys.platform,
                'semantic_directories': file_config.get('use_semantic_directories', True)
            }
            with open(self.session_dir / 'session_info.yml', 'w') as f:
                import yaml
                yaml.dump(session_info, f)
        else:
            self.session_dir = self.base_log_dir
        
        # Create semantic directories if enabled
        if file_config.get('use_semantic_directories', True):
            semantic_structure = file_config.get('semantic_structure', {})
            self.semantic_structure = semantic_structure
            
            # Create semantic directories inside session directory
            for component_type, dir_name in semantic_structure.items():
                component_dir = self.session_dir / dir_name
                component_dir.mkdir(parents=True, exist_ok=True)
                
                # Create subdirectories for data units
                if component_type == 'data_units':
                    for subdir in ['memory', 'file', 'stream', 'history']:
                        (component_dir / subdir).mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str, category: str = "components", debug_mode: bool = True) -> NanoBrainLogger:
        """Get a logger for a specific component."""
        # Get configuration - self.logging_config already contains the logging section
        file_config = self.logging_config.get('file', {})
        
        # Determine log file path based on category and semantic structure
        if file_config.get('use_semantic_directories', True):
            semantic_structure = file_config.get('semantic_structure', {})
            
            # Map category to semantic directory
            category_mapping = {
                'components': 'nanobrain_components',
                'parsl': 'distributed_processing',
                'workflows': 'workflows',
                'agents': 'agents',
                'data': 'data_units',  # Map 'data' category to 'data_units'
                'data_units': 'data_units',
                'steps': 'steps',
                'triggers': 'triggers',
                'links': 'links',
                'executors': 'executors'
            }
            
            mapped_category = category_mapping.get(category, 'nanobrain_components')
            dir_name = semantic_structure.get(mapped_category)
            
            if not dir_name:
                # If no mapping found in semantic structure, use the mapped category
                dir_name = mapped_category
            
            # Create log file path inside appropriate semantic directory
            log_dir = self.session_dir / dir_name
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for data units
            if mapped_category == 'data_units':
                for subdir in ['memory', 'file', 'stream', 'history']:
                    (log_dir / subdir).mkdir(parents=True, exist_ok=True)
            
            # Create log file in appropriate directory
            if mapped_category == 'data_units':
                # For data units, put log files in appropriate subdirectory
                data_type = name.split('_')[0] if '_' in name else 'memory'
                log_file = log_dir / data_type / f"{name}.log"
            else:
                log_file = log_dir / f"{name}.log"
        else:
            # Default to simple log file in session directory
            log_file = self.session_dir / f"{name}.log"
        
        # Create logger with appropriate handlers
        logger = NanoBrainLogger(
            name=name,
            log_file=log_file,
            enable_console=self.logging_config.get('mode') in ['console', 'both'],
            enable_file=self.logging_config.get('mode') in ['file', 'both'],
            debug_mode=debug_mode,
            create_on_first_message=file_config.get('create_on_first_message', True)
        )
        
        return logger
    
    def register_component(self, component_type: str, component_name: str, 
                          component_instance: Any, metadata: Dict[str, Any] = None):
        """Register a framework component for lifecycle tracking."""
        component_id = f"{component_type}_{component_name}"
        
        self.component_registry[component_id] = {
            "type": component_type,
            "name": component_name,
            "instance": component_instance,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Get or create logger for this component
        logger = self.get_logger(component_name, component_type)
        logger.info(f"Component registered: {component_name}",
                   component_type=component_type,
                   component_id=component_id,
                   metadata=metadata)
    
    def log_component_lifecycle(self, component_type: str, component_name: str, 
                               event: str, details: Dict[str, Any] = None):
        """Log component lifecycle events (initialize, start, stop, shutdown)."""
        logger = self.get_logger(component_name, component_type)
        
        logger.info(f"Component {event}: {component_name}",
                   component_type=component_type,
                   lifecycle_event=event,
                   details=details or {})
        
        # Update component status in registry
        component_id = f"{component_type}_{component_name}"
        if component_id in self.component_registry:
            self.component_registry[component_id]["status"] = event
            self.component_registry[component_id]["last_event"] = datetime.now().isoformat()
    
    def log_data_unit_operation(self, data_unit_name: str, operation: str, 
                               data: Any = None, metadata: Dict[str, Any] = None):
        """Log data unit operations (create, read, write, update, delete)."""
        logger = self.get_logger(data_unit_name, "data_units")
        
        logger.log_data_unit_operation(operation, data_unit_name, data, metadata)
    
    def log_trigger_event(self, trigger_name: str, trigger_type: str,
                         event: str, conditions: Dict[str, Any] = None, 
                         activated: bool = True):
        """Log trigger activation and events."""
        logger = self.get_logger(trigger_name, "triggers")
        
        logger.log_trigger_activation(trigger_name, trigger_type, conditions or {}, activated)
        logger.info(f"Trigger {event}: {trigger_name}",
                   trigger_type=trigger_type,
                   trigger_event=event,
                   activated=activated,
                   conditions=conditions)
    
    def log_link_operation(self, link_name: str, operation: str, 
                          source: str, destination: str, data: Any = None):
        """Log link operations and data transfers."""
        logger = self.get_logger(link_name, "links")
        
        logger.log_data_transfer(source, destination, type(data).__name__ if data else "unknown")
        logger.info(f"Link {operation}: {link_name}",
                   link_operation=operation,
                   source=source,
                   destination=destination,
                   data_type=type(data).__name__ if data else "unknown")
    
    def log_workflow_event(self, workflow_name: str, event: str, 
                          details: Dict[str, Any] = None):
        """Log workflow-level events and orchestration."""
        logger = self.get_logger(workflow_name, "workflows")
        
        logger.info(f"Workflow {event}: {workflow_name}",
                   workflow_event=event,
                   details=details or {})
    
    def create_session_summary(self):
        """Create a comprehensive session summary."""
        if not self.should_log_to_file:
            return
        
        # Update log file sizes
        for log_file_info in self.log_files:
            log_path = self.session_dir / log_file_info["path"]
            if log_path.exists():
                log_file_info["size_bytes"] = log_path.stat().st_size
        
        summary = {
            "session_id": self.session_id,
            "start_time": self.session_start_time,
            "end_time": datetime.now().isoformat(),
            "log_directory": str(self.session_dir),
            "loggers_created": self.loggers_created,
            "log_files": self.log_files,
            "components_registered": len(self.component_registry),
            "component_registry": {
                comp_id: {
                    "type": comp_info["type"],
                    "name": comp_info["name"],
                    "status": comp_info["status"],
                    "created_at": comp_info["created_at"],
                    "metadata": comp_info["metadata"]
                }
                for comp_id, comp_info in self.component_registry.items()
            }
        }
        
        summary_path = self.session_dir / "session_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.should_log_to_console:
            print(f"📊 Session summary created: {summary_path}")
        
        return summary

def get_system_log_manager() -> SystemLogManager:
    """Get or create the global system log manager."""
    global _system_log_manager
    
    # Always check if we need to recreate based on current configuration
    try:
        from ..config import get_logging_config
        current_config = get_logging_config()
        file_config = current_config.get('file', {})
        
        # Check if we should use semantic directories
        use_semantic = file_config.get('use_semantic_directories', True)
        use_session = file_config.get('use_session_directories', True)
        
        # If we have an existing manager but the configuration has changed, recreate it
        if _system_log_manager is not None:
            existing_semantic = getattr(_system_log_manager, 'use_semantic_directories', False)
            existing_session = getattr(_system_log_manager, 'use_session_directories', True)
            
            # Recreate if configuration changed
            if existing_semantic != use_semantic or existing_session != use_session:
                _system_log_manager = None
                
    except ImportError:
        # If config not available, proceed with existing manager
        pass
    
    if _system_log_manager is None:
        _system_log_manager = SystemLogManager()
    return _system_log_manager

def get_logger(name: str = "nanobrain", category: str = None, **kwargs) -> NanoBrainLogger:
    """
    Get or create a NanoBrain logger instance.
    
    Args:
        name: Logger name
        category: Optional category for system-level logging (agents, steps, triggers, etc.)
        **kwargs: Additional arguments passed to NanoBrainLogger
    
    Returns:
        NanoBrainLogger instance
    """
    global _global_loggers
    
    # If category is specified, use system log manager for organized logging
    if category:
        system_manager = get_system_log_manager()
        return system_manager.get_logger(name, category, **kwargs)
    
    # Otherwise use simple global logger registry
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

def reconfigure_global_logging():
    """Reconfigure global logging based on current configuration.
    This is called when configuration changes to update logging settings.
    """
    # Run the global logging configuration
    _configure_global_logging()
    
    try:
        # Check if config is available and if we're in file-only mode
        from ..config import should_log_to_console
        
        if not should_log_to_console():
            # We're in file-only mode, so aggressively suppress console output
            _suppress_third_party_console_logging()
            # Note: Parsl logging will be configured when workflows are initialized
            
            # Also apply specific fixes for known noisy loggers
            httpx_logger = logging.getLogger('httpx')
            httpx_logger.handlers.clear()
            httpx_logger.addHandler(logging.NullHandler())
            httpx_logger.propagate = False
            httpx_logger.setLevel(logging.CRITICAL)
            
            # Disable propagation for root logger children
            for name in logging.root.manager.loggerDict:
                logger = logging.getLogger(name)
                if '.' not in name:  # Top-level loggers only
                    logger.propagate = False
                    
            # Configure root logger to prevent any output
            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            root_logger.addHandler(logging.NullHandler())
            root_logger.setLevel(logging.CRITICAL)
    except Exception as e:
        # Config not available or error occurred
        pass

def configure_third_party_loggers(console_enabled: bool = None):
    """Configure third-party library loggers based on logging mode.
    
    Args:
        console_enabled: If None, will check global configuration. 
                        If False, will suppress console output for third-party libraries.
    """
    if console_enabled is None:
        try:
            from ..config import should_log_to_console
            console_enabled = should_log_to_console()
        except ImportError:
            console_enabled = True
    
    if not console_enabled:
        _suppress_third_party_console_logging()
        # Note: Parsl logging will be configured when workflows are initialized
    else:
        # Re-enable console logging for third-party libraries
        third_party_loggers = [
            'openai',
            'httpx',
            'httpcore', 
            'parsl',
            'parsl.executors',
            'parsl.providers',
            'parsl.monitoring',
            'urllib3',
            'requests',
            'asyncio',
            'concurrent.futures'
        ]
        
        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)
            # Reset to default behavior
            logger.setLevel(logging.NOTSET)
            logger.propagate = True

# Convenience functions for system-level logging
def log_component_lifecycle(component_type: str, component_name: str, 
                           event: str, details: Dict[str, Any] = None):
    """Convenience function for logging component lifecycle events."""
    system_manager = get_system_log_manager()
    system_manager.log_component_lifecycle(component_type, component_name, event, details)

def register_component(component_type: str, component_name: str, 
                      component_instance: Any, metadata: Dict[str, Any] = None):
    """Convenience function for registering framework components."""
    system_manager = get_system_log_manager()
    system_manager.register_component(component_type, component_name, component_instance, metadata)

def log_workflow_event(workflow_name: str, event: str, details: Dict[str, Any] = None):
    """Convenience function for logging workflow events."""
    system_manager = get_system_log_manager()
    system_manager.log_workflow_event(workflow_name, event, details)

def log_data_unit_operation(data_unit_name: str, operation: str, 
                           data: Any = None, metadata: Dict[str, Any] = None):
    """Convenience function for logging data unit operations."""
    system_manager = get_system_log_manager()
    system_manager.log_data_unit_operation(data_unit_name, operation, data, metadata)

def log_trigger_event(trigger_name: str, trigger_type: str, event: str, 
                     conditions: Dict[str, Any] = None, activated: bool = True):
    """Convenience function for logging trigger events."""
    system_manager = get_system_log_manager()
    system_manager.log_trigger_event(trigger_name, trigger_type, event, conditions, activated)

def log_link_operation(link_name: str, operation: str, source: str, 
                      destination: str, data: Any = None):
    """Convenience function for logging link operations."""
    system_manager = get_system_log_manager()
    system_manager.log_link_operation(link_name, operation, source, destination, data)

def create_session_summary():
    """Convenience function for creating session summary."""
    system_manager = get_system_log_manager()
    return system_manager.create_session_summary()

def get_logging_status():
    """Get comprehensive logging status information."""
    try:
        from ..config import get_logging_config, should_log_to_file, should_log_to_console
        
        config = get_logging_config()
        file_config = config.get('file', {})
        
        system_manager = get_system_log_manager()
        
        status = {
            "logging_mode": config.get('mode', 'unknown'),
            "console_enabled": should_log_to_console(),
            "file_enabled": should_log_to_file(),
            "use_semantic_directories": system_manager.use_semantic_directories,
            "use_session_directories": system_manager.use_session_directories,
            "base_log_dir": str(system_manager.base_log_dir),
            "semantic_structure": getattr(system_manager, 'semantic_structure', {}),
            "loggers_created": len(system_manager.loggers_created),
            "components_registered": len(system_manager.component_registry)
        }
        
        return status
        
    except Exception as e:
        return {"error": str(e)}

def cleanup_empty_logs(base_dir: str = "logs") -> Dict[str, int]:
    """
    Clean up empty log files and optionally old session directories.
    
    Args:
        base_dir: Base logging directory
        
    Returns:
        Dictionary with cleanup statistics
    """
    from pathlib import Path
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return {"empty_files_removed": 0, "empty_dirs_removed": 0, "session_dirs_found": 0}
    
    stats = {
        "empty_files_removed": 0,
        "empty_dirs_removed": 0, 
        "session_dirs_found": 0
    }
    
    # Remove empty log files
    for log_file in base_path.rglob("*.log"):
        if log_file.stat().st_size == 0:
            try:
                log_file.unlink()
                stats["empty_files_removed"] += 1
            except Exception:
                pass
    
    # Count session directories (for information)
    session_dirs = list(base_path.glob("session_*"))
    stats["session_dirs_found"] = len(session_dirs)
    
    # Remove empty directories
    for dir_path in base_path.rglob("*"):
        if dir_path.is_dir():
            try:
                # Try to remove if empty
                dir_path.rmdir()
                stats["empty_dirs_removed"] += 1
            except OSError:
                # Directory not empty, that's fine
                pass
    
    return stats 