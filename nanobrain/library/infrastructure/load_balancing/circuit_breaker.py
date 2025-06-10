"""
Circuit breaker pattern implementation.

Fault tolerance patterns for handling failures gracefully.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Callable, Optional, Dict
from nanobrain.core.logging_system import get_logger


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED
        self._lock = asyncio.Lock()
        self.logger = get_logger("circuit_breaker")
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN")
                    
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._on_success()
            return result
            
        except self.expected_exception as e:
            await self._on_failure()
            raise e
            
    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info("Circuit breaker transitioning to CLOSED")
                
    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker transitioning to OPEN after {self.failure_count} failures")
                
    async def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        async with self._lock:
            return self.state
            
    async def reset(self):
        """Reset circuit breaker to closed state."""
        async with self._lock:
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.state = CircuitBreakerState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
            
    async def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        async with self._lock:
            return {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'failure_threshold': self.failure_threshold,
                'timeout': self.timeout,
                'last_failure_time': self.last_failure_time,
                'time_since_last_failure': time.time() - self.last_failure_time if self.last_failure_time > 0 else 0
            } 