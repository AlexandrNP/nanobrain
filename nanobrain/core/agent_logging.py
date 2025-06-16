"""
Specialized agent logging module that provides context-aware, process-safe logging
for NanoBrain agents without parent class initialization pollution.
"""

import os
import time
import asyncio
import threading
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .async_logging import get_process_safe_logger, ProcessSafeLogger


@dataclass
class AgentInteractionLog:
    """Structured log for agent input/output interactions."""
    timestamp: str
    agent_name: str
    agent_type: str
    input_text: str
    response_text: str
    input_length: int
    response_length: int
    duration_ms: float
    llm_calls: int
    total_tokens: int
    success: bool
    error_message: Optional[str] = None
    execution_context: Optional[str] = None
    metadata: Dict[str, Any] = None


class AgentLogger:
    """
    Specialized logger for NanoBrain agents that:
    1. Only initializes for concrete agent instances (not parent classes)
    2. Works across process boundaries (main/worker processes)
    3. Always captures agent inputs/outputs regardless of configuration
    4. Provides minimal performance overhead
    """
    
    def __init__(self, agent_name: str, agent_type: str):
        """Initialize agent logger with context-aware setup."""
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.process_id = os.getpid()
        self.thread_id = threading.get_ident()
        
        # Determine if this is a concrete instance that should have full logging
        self._is_concrete_instance = self._is_concrete_agent_instance(agent_type)
        
        # Initialize performance tracking
        self._interaction_count = 0
        self._error_count = 0
        self._total_llm_calls = 0
        self._total_tokens = 0
        self._start_time = time.time()
        
        # Set up logger only for concrete instances
        if self._is_concrete_instance:
            try:
                self.logger = get_process_safe_logger(agent_name, category="agents")
                self.logger.info(f"Agent logger initialized for {agent_type}",
                               agent_name=agent_name,
                               agent_type=agent_type,
                               is_concrete=True)
            except Exception as e:
                print(f"Failed to initialize agent logger for {agent_name}: {e}")
                self.logger = None
        else:
            # Parent classes get minimal logging
            self.logger = None
    
    def _is_concrete_agent_instance(self, agent_type: str) -> bool:
        """
        Determine if this is a concrete agent instance that should have full logging.
        
        Concrete instances are actual agent implementations, not parent classes.
        """
        # List of concrete agent classes that should have full logging
        concrete_agent_classes = {
            'ConversationalAgent',
            'SimpleAgent',
            'CustomAgent',
            'ToolAgent',
            'WorkflowAgent'
        }
        
        # Also check if it ends with 'Agent' but is not just 'Agent'
        is_specific_agent = (
            agent_type in concrete_agent_classes or
            (agent_type.endswith('Agent') and agent_type != 'Agent' and len(agent_type) > 5)
        )
        
        return is_specific_agent
    
    def log_lifecycle_event(self, event: str, details: Dict[str, Any] = None):
        """Log agent lifecycle events (initialize, shutdown, etc.)."""
        if not self._is_concrete_instance or not self.logger:
            return
        
        self.logger.info(f"Agent lifecycle: {event}",
                        agent_name=self.agent_name,
                        agent_type=self.agent_type,
                        lifecycle_event=event,
                        details=details or {},
                        timestamp=datetime.now(timezone.utc).isoformat())
    
    def log_interaction(self, input_text: str, response_text: str,
                       duration_ms: float, llm_calls: int = 0,
                       total_tokens: int = 0, success: bool = True,
                       error_message: Optional[str] = None,
                       **additional_metadata):
        """
        Log agent input/output interaction - ALWAYS logged regardless of settings.
        This is the core method that ensures agent outputs are captured.
        """
        if not self._is_concrete_instance or not self.logger:
            return
        
        self._interaction_count += 1
        self._total_llm_calls += llm_calls
        self._total_tokens += total_tokens
        
        # Create structured interaction log
        interaction = AgentInteractionLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            input_text=self._truncate_for_logging(input_text, 1000),
            response_text=self._truncate_for_logging(response_text, 1000),
            input_length=len(input_text),
            response_length=len(response_text),
            duration_ms=duration_ms,
            llm_calls=llm_calls,
            total_tokens=total_tokens,
            success=success,
            error_message=error_message,
            execution_context=self._get_execution_context(),
            metadata=additional_metadata
        )
        
        # Log the interaction - this is CRITICAL for debugging
        self.logger.info(f"Agent interaction #{self._interaction_count}",
                        **interaction.__dict__)
    
    def log_llm_call(self, model: str, messages_count: int, 
                     response_content: str, tokens_used: int,
                     finish_reason: str, duration_ms: float):
        """Log LLM interaction details."""
        if not self._is_concrete_instance or not self.logger:
            return
        
        self.logger.info(f"LLM call completed",
                        agent_name=self.agent_name,
                        model=model,
                        messages_count=messages_count,
                        response_preview=self._truncate_for_logging(response_content, 200),
                        response_length=len(response_content),
                        tokens_used=tokens_used,
                        finish_reason=finish_reason,
                        duration_ms=duration_ms)
    
    def log_error(self, error_message: str, error_type: str = None,
                  context: Dict[str, Any] = None):
        """Log agent errors with context."""
        if not self._is_concrete_instance or not self.logger:
            return
        
        self.logger.error(f"Agent error: {error_message}",
                         agent_name=self.agent_name,
                         agent_type=self.agent_type,
                         error_type=error_type,
                         context=context or {})
    
    def log_debug(self, message: str, **metadata):
        """Log debug information."""
        if not self._is_concrete_instance or not self.logger:
            return
        
        if self._is_concrete_instance:
            self.logger.debug(message,
                            agent_name=self.agent_name,
                            agent_type=self.agent_type,
                            **metadata)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary."""
        if not self._is_concrete_instance:
            return {}
        
        uptime = time.time() - self._start_time
        avg_processing_time = (
            self._total_llm_calls / self._interaction_count
            if self._interaction_count > 0 else 0
        )
        
        return {
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "uptime_seconds": uptime,
            "interaction_count": self._interaction_count,
            "total_llm_calls": self._total_llm_calls,
            "total_tokens": self._total_tokens,
            "avg_processing_time_ms": avg_processing_time,
            "interactions_per_minute": (
                (self._interaction_count / uptime) * 60
                if uptime > 0 else 0
            )
        }
    
    def _truncate_for_logging(self, text: str, max_length: int) -> str:
        """Truncate text for logging while preserving readability."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + f"... [truncated, total: {len(text)} chars]"
    
    def _get_execution_context(self) -> str:
        """Get current execution context."""
        try:
            # Check if we're in a Parsl worker
            if 'PARSL_WORKER' in os.environ:
                return f"parsl_worker_{os.getpid()}"
            
            # Check for other distributed contexts
            process_name = os.environ.get('NANOBRAIN_PROCESS_TYPE', 'main')
            return f"{process_name}_{os.getpid()}"
            
        except Exception:
            return f"unknown_{os.getpid()}"
    
    @asynccontextmanager
    async def interaction_context(self, input_text: str):
        """
        Context manager for tracking agent interactions with automatic logging.
        Ensures that ALL agent interactions are logged, even if errors occur.
        """
        if not self._is_concrete_instance:
            # For parent classes, just yield without logging
            yield None
            return
        
        start_time = time.time()
        error_occurred = None
        response_text = ""
        llm_calls = 0
        total_tokens = 0
        
        try:
            # Create interaction context object
            context = {
                'start_time': start_time,
                'input_text': input_text,
                'llm_calls': 0,
                'total_tokens': 0
            }
            
            yield context
            
            # Extract results from context
            response_text = context.get('response_text', '')
            llm_calls = context.get('llm_calls', 0)
            total_tokens = context.get('total_tokens', 0)
            
        except Exception as e:
            error_occurred = str(e)
            response_text = context.get('response_text', '') if 'context' in locals() else ''
            raise
            
        finally:
            # ALWAYS log the interaction, regardless of success/failure
            duration_ms = (time.time() - start_time) * 1000
            
            self.log_interaction(
                input_text=input_text,
                response_text=response_text,
                duration_ms=duration_ms,
                llm_calls=llm_calls,
                total_tokens=total_tokens,
                success=error_occurred is None,
                error_message=error_occurred
            )
    
    def shutdown(self):
        """Shutdown the agent logger."""
        if not self._is_concrete_instance or not self.logger:
            return
        
        # Log final performance summary
        summary = self.get_performance_summary()
        self.log_lifecycle_event("shutdown", summary)
        
        # Shutdown the underlying logger
        if hasattr(self.logger, 'shutdown'):
            self.logger.shutdown()


def create_agent_logger(agent_instance: Any) -> AgentLogger:
    """
    Factory function to create an agent logger only for concrete agent instances.
    
    Args:
        agent_instance: The agent instance (self)
        
    Returns:
        AgentLogger instance that will only log if it's a concrete agent
    """
    agent_name = getattr(agent_instance, 'name', 'unknown_agent')
    agent_type = agent_instance.__class__.__name__
    
    # Check configuration for detailed logging
    enable_detailed = getattr(agent_instance, 'enable_logging', True)
    if hasattr(agent_instance, 'config'):
        enable_detailed = getattr(agent_instance.config, 'enable_logging', True)
    
    return AgentLogger(
        agent_name=agent_name,
        agent_type=agent_type
    ) 