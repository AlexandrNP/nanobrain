"""
Process-safe async logging system for NanoBrain framework.

This module provides concurrency-safe logging that works across process boundaries,
particularly important for Parsl distributed execution.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict
from queue import Queue, Empty
from threading import Thread
import uuid


@dataclass
class LogMessage:
    """Structured log message for queue-based logging."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    metadata: Dict[str, Any]
    category: str
    component_name: str
    process_id: int
    thread_id: int
    execution_context: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ProcessSafeLogger:
    """
    Process-safe logger that works across main process and worker processes.
    
    Uses queues for message passing and background threads for actual I/O.
    """
    
    def __init__(self, name: str, category: str = "components", 
                 log_directory: Optional[Path] = None):
        self.name = name
        self.category = category
        self.process_id = os.getpid()
        self.thread_id = threading.get_ident()
        
        # Execution context detection
        self.execution_context = self._detect_execution_context()
        
        # Set up logging based on context
        if self.execution_context == "main_process":
            self._setup_main_process_logging(log_directory)
        elif self.execution_context == "worker_process":
            self._setup_worker_process_logging()
        else:
            self._setup_fallback_logging()
    
    def _detect_execution_context(self) -> str:
        """Detect the current execution context."""
        try:
            # Check if we're in a Parsl worker process
            if 'PARSL_WORKER' in os.environ:
                return "worker_process"
            
            # Check if we're in the main process with multiprocessing
            if hasattr(multiprocessing, 'current_process'):
                current_proc = multiprocessing.current_process()
                if current_proc.name == 'MainProcess':
                    return "main_process"
                else:
                    return "worker_process"
            
            # Default to main process
            return "main_process"
            
        except Exception:
            return "unknown"
    
    def _setup_main_process_logging(self, log_directory: Optional[Path]):
        """Set up logging for the main process with file handlers."""
        self.log_queue = Queue()
        self.background_thread = None
        self.should_stop = threading.Event()
        
        # Set up log directory
        if log_directory is None:
            try:
                from nanobrain.core.logging_system import get_system_log_manager
                system_manager = get_system_log_manager()
                self.log_directory = system_manager.session_dir
            except ImportError:
                # Fallback to default directory
                self.log_directory = Path("logs") / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.log_directory.mkdir(parents=True, exist_ok=True)
        else:
            self.log_directory = log_directory
        
        # Create category-specific directory
        category_dir = self.log_directory / self.category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up log file
        self.log_file = category_dir / f"{self.name}.log"
        
        # Start background logging thread
        self._start_background_logger()
    
    def _setup_worker_process_logging(self):
        """Set up logging for worker processes with queue-based message passing."""
        # Try to connect to shared queue or create fallback
        try:
            # In worker processes, we'll use a simple queue approach
            # Real implementation would use inter-process communication
            self.log_queue = Queue()
            self.background_thread = None
            self.should_stop = threading.Event()
            
            # For now, use local file with process ID
            temp_dir = Path(f"/tmp/nanobrain_worker_{self.process_id}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = temp_dir / f"{self.name}.log"
            
            self._start_background_logger()
            
        except Exception:
            self._setup_fallback_logging()
    
    def _setup_fallback_logging(self):
        """Set up fallback logging when other methods fail."""
        # Use Python's standard logging as fallback
        self.stdlib_logger = logging.getLogger(f"nanobrain.{self.name}")
        if not self.stdlib_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.stdlib_logger.addHandler(handler)
            self.stdlib_logger.setLevel(logging.INFO)
        
        self.log_queue = None
        self.background_thread = None
    
    def _start_background_logger(self):
        """Start the background thread for processing log messages."""
        if self.log_queue is not None:
            self.background_thread = Thread(
                target=self._process_log_queue,
                daemon=True
            )
            self.background_thread.start()
    
    def _process_log_queue(self):
        """Background thread function to process log messages."""
        batch_size = 10
        batch_timeout = 1.0  # seconds
        
        message_batch = []
        last_flush = time.time()
        
        while not self.should_stop.is_set():
            try:
                # Try to get a message with timeout
                try:
                    message = self.log_queue.get(timeout=0.1)
                    message_batch.append(message)
                except Empty:
                    pass
                
                # Flush batch if we have enough messages or timeout exceeded
                current_time = time.time()
                should_flush = (
                    len(message_batch) >= batch_size or
                    (message_batch and current_time - last_flush >= batch_timeout)
                )
                
                if should_flush:
                    self._flush_message_batch(message_batch)
                    message_batch.clear()
                    last_flush = current_time
                    
            except Exception as e:
                # Don't let logging errors break the thread
                print(f"Background logger error: {e}")
                continue
        
        # Flush remaining messages on shutdown
        if message_batch:
            self._flush_message_batch(message_batch)
    
    def _flush_message_batch(self, messages: List[LogMessage]):
        """Write a batch of messages to the log file."""
        if not messages:
            return
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for message in messages:
                    json_line = json.dumps(message.to_dict())
                    f.write(json_line + '\n')
                f.flush()
        except Exception as e:
            # Fall back to stderr if file writing fails
            print(f"Failed to write to log file {self.log_file}: {e}")
            for message in messages:
                print(f"[{message.level}] {message.logger_name}: {message.message}")
    
    def _log(self, level: str, message: str, **metadata):
        """Internal logging method."""
        log_message = LogMessage(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            logger_name=self.name,
            message=message,
            metadata=metadata,
            category=self.category,
            component_name=self.name,
            process_id=self.process_id,
            thread_id=self.thread_id,
            execution_context=self.execution_context
        )
        
        if self.log_queue is not None:
            try:
                self.log_queue.put_nowait(log_message)
            except Exception:
                # Queue full or other error - fall back to direct logging
                self._direct_log(log_message)
        else:
            # Use fallback logger
            self._fallback_log(level, message, metadata)
    
    def _direct_log(self, log_message: LogMessage):
        """Directly write log message (fallback when queue fails)."""
        try:
            if hasattr(self, 'log_file'):
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    json_line = json.dumps(log_message.to_dict())
                    f.write(json_line + '\n')
                    f.flush()
        except Exception:
            print(f"[{log_message.level}] {log_message.logger_name}: {log_message.message}")
    
    def _fallback_log(self, level: str, message: str, metadata: Dict[str, Any]):
        """Use stdlib logger as fallback."""
        if hasattr(self, 'stdlib_logger'):
            log_level = getattr(logging, level.upper(), logging.INFO)
            self.stdlib_logger.log(log_level, f"{message} {metadata}")
    
    # Public logging methods
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)
    
    def shutdown(self):
        """Shutdown the logger and cleanup resources."""
        if hasattr(self, 'should_stop'):
            self.should_stop.set()
        
        if hasattr(self, 'background_thread') and self.background_thread:
            self.background_thread.join(timeout=2.0)


class AsyncLoggerFactory:
    """Factory for creating process-safe loggers."""
    
    _loggers: Dict[str, ProcessSafeLogger] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_logger(cls, name: str, category: str = "components",
                   log_directory: Optional[Path] = None) -> ProcessSafeLogger:
        """Get or create a process-safe logger."""
        logger_key = f"{category}.{name}"
        
        with cls._lock:
            if logger_key not in cls._loggers:
                cls._loggers[logger_key] = ProcessSafeLogger(
                    name=name,
                    category=category,
                    log_directory=log_directory
                )
            return cls._loggers[logger_key]
    
    @classmethod
    def shutdown_all(cls):
        """Shutdown all loggers."""
        with cls._lock:
            for logger in cls._loggers.values():
                try:
                    logger.shutdown()
                except Exception as e:
                    print(f"Error shutting down logger: {e}")
            cls._loggers.clear()


# Convenience functions
def get_process_safe_logger(name: str, category: str = "components",
                           log_directory: Optional[Path] = None) -> ProcessSafeLogger:
    """Get a process-safe logger instance."""
    return AsyncLoggerFactory.get_logger(name, category, log_directory)


def shutdown_all_loggers():
    """Shutdown all process-safe loggers."""
    AsyncLoggerFactory.shutdown_all() 