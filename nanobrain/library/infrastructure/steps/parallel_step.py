"""
Parallel Step Implementation

Provides generic parallel processing capabilities for NanoBrain steps.

This module implements a hierarchy of parallel processing steps:
- ParallelStep: Base class for any parallel processing
- ParallelAgentStep: Specialized for agent-based parallel processing
- ParallelConversationalAgentStep: Specialized for conversational agents

Key Features:
- Generic request/response handling with configurable types
- Pluggable load balancing strategies
- Configurable parallelism and resource management
- Comprehensive error handling and recovery
- Performance monitoring and metrics collection
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.agent import Agent
from nanobrain.core.logging_system import get_logger


# Generic type variables for request/response handling
RequestType = TypeVar('RequestType')
ResponseType = TypeVar('ResponseType')
ProcessorType = TypeVar('ProcessorType')


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for parallel processing."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    WEIGHTED = "weighted"
    FASTEST_RESPONSE = "fastest_response"


class ParallelProcessingConfig(StepConfig):
    """Configuration for parallel processing steps."""
    max_parallel_requests: int = 10
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    performance_tracking: bool = True
    batch_processing: bool = True
    max_batch_size: int = 100


@dataclass
class ProcessingRequest:
    """Base class for processing requests."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority,
            'context': self.context,
            'timeout': self.timeout
        }


@dataclass
class ProcessingResponse:
    """Base class for processing responses."""
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    processor_id: str = ""
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'processor_id': self.processor_id,
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata
        }


class ProcessorPool(Generic[ProcessorType]):
    """Generic processor pool for managing parallel processing resources."""
    
    def __init__(self, processors: List[ProcessorType], strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.processors = processors
        self.strategy = strategy
        self.current_index = 0
        self.processor_stats = {i: {'requests': 0, 'errors': 0, 'avg_time': 0.0, 'total_time': 0.0} 
                               for i in range(len(processors))}
        self.circuit_breakers = {i: {'failures': 0, 'last_failure': None, 'is_open': False} 
                                for i in range(len(processors))}
    
    def get_next_processor(self) -> tuple[ProcessorType, int]:
        """Get the next processor based on load balancing strategy."""
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin()
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded()
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random()
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted()
        elif self.strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            return self._fastest_response()
        else:
            return self._round_robin()
    
    def _round_robin(self) -> tuple[ProcessorType, int]:
        """Round-robin processor selection."""
        available_processors = [i for i in range(len(self.processors)) 
                               if not self.circuit_breakers[i]['is_open']]
        
        if not available_processors:
            # All processors are circuit-broken, reset and try again
            self._reset_circuit_breakers()
            available_processors = list(range(len(self.processors)))
        
        index = available_processors[self.current_index % len(available_processors)]
        self.current_index = (self.current_index + 1) % len(available_processors)
        return self.processors[index], index
    
    def _least_loaded(self) -> tuple[ProcessorType, int]:
        """Select processor with least current load."""
        available_processors = [(i, self.processor_stats[i]['requests']) 
                               for i in range(len(self.processors))
                               if not self.circuit_breakers[i]['is_open']]
        
        if not available_processors:
            return self._round_robin()
        
        index = min(available_processors, key=lambda x: x[1])[0]
        return self.processors[index], index
    
    def _random(self) -> tuple[ProcessorType, int]:
        """Random processor selection."""
        import random
        available_processors = [i for i in range(len(self.processors)) 
                               if not self.circuit_breakers[i]['is_open']]
        
        if not available_processors:
            return self._round_robin()
        
        index = random.choice(available_processors)
        return self.processors[index], index
    
    def _weighted(self) -> tuple[ProcessorType, int]:
        """Weighted processor selection based on performance."""
        # Weight based on inverse of average response time
        weights = []
        available_processors = []
        
        for i in range(len(self.processors)):
            if not self.circuit_breakers[i]['is_open']:
                avg_time = self.processor_stats[i]['avg_time']
                weight = 1.0 / (avg_time + 0.001)  # Add small value to avoid division by zero
                weights.append(weight)
                available_processors.append(i)
        
        if not available_processors:
            return self._round_robin()
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                index = available_processors[i]
                return self.processors[index], index
        
        # Fallback
        return self._round_robin()
    
    def _fastest_response(self) -> tuple[ProcessorType, int]:
        """Select processor with fastest average response time."""
        available_processors = [(i, self.processor_stats[i]['avg_time']) 
                               for i in range(len(self.processors))
                               if not self.circuit_breakers[i]['is_open']]
        
        if not available_processors:
            return self._round_robin()
        
        index = min(available_processors, key=lambda x: x[1] if x[1] > 0 else float('inf'))[0]
        return self.processors[index], index
    
    def update_processor_stats(self, processor_index: int, processing_time: float, success: bool):
        """Update processor statistics."""
        stats = self.processor_stats[processor_index]
        stats['requests'] += 1
        
        if success:
            stats['total_time'] += processing_time
            stats['avg_time'] = stats['total_time'] / stats['requests']
            # Reset circuit breaker on success
            self.circuit_breakers[processor_index]['failures'] = 0
            self.circuit_breakers[processor_index]['is_open'] = False
        else:
            stats['errors'] += 1
            # Update circuit breaker
            cb = self.circuit_breakers[processor_index]
            cb['failures'] += 1
            cb['last_failure'] = datetime.now()
            
            # Open circuit breaker if threshold exceeded
            if cb['failures'] >= 5:  # Configurable threshold
                cb['is_open'] = True
    
    def _reset_circuit_breakers(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            cb['failures'] = 0
            cb['is_open'] = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor pool statistics."""
        return {
            'processor_count': len(self.processors),
            'strategy': self.strategy.value,
            'processor_stats': self.processor_stats,
            'circuit_breakers': self.circuit_breakers
        }


class ParallelStep(Step, Generic[RequestType, ResponseType, ProcessorType], ABC):
    """
    Base class for parallel processing steps.
    
    Provides generic parallel processing capabilities that can be specialized
    for different types of processors (agents, functions, services, etc.).
    """
    
    def __init__(self, config: ParallelProcessingConfig, processors: List[ProcessorType], **kwargs):
        super().__init__(config, **kwargs)
        self.parallel_config = config
        self.processor_pool = ProcessorPool(processors, config.load_balancing_strategy)
        
        # Performance tracking
        self.total_requests = 0
        self.total_responses = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        # Request queue for batch processing
        self.request_queue = asyncio.Queue(maxsize=config.max_batch_size * 2)
        self.processing_semaphore = asyncio.Semaphore(config.max_parallel_requests)
        
        # Logger
        self.logger = get_logger(f"parallel_step.{self.name}", "steps")
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs using parallel processors.
        
        Supports both single requests and batch processing.
        """
        start_time = time.time()
        
        try:
            # Extract requests from inputs
            requests = await self._extract_requests(inputs)
            
            if not requests:
                self.logger.warning("No valid requests found in inputs")
                return {'responses': [], 'metadata': {'error': 'no_requests'}}
            
            self.logger.info(f"Processing {len(requests)} requests in parallel",
                           request_count=len(requests),
                           processor_count=len(self.processor_pool.processors))
            
            # Process requests in parallel
            responses = await self._process_requests_parallel(requests)
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_requests += len(requests)
            self.total_responses += len(responses)
            
            # Update performance metrics
            if self.parallel_config.performance_tracking:
                self._update_performance_metrics(len(requests), len(responses), processing_time)
            
            return {
                'responses': [self._response_to_dict(r) for r in responses],
                'metadata': {
                    'processing_time': processing_time,
                    'request_count': len(requests),
                    'response_count': len(responses),
                    'parallel_execution': True,
                    'processor_stats': self.processor_pool.get_stats()
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1
            self.logger.error(f"Error in parallel processing: {e}",
                            error_type=type(e).__name__,
                            processing_time=processing_time)
            
            return {
                'responses': [],
                'error': str(e),
                'metadata': {
                    'processing_time': processing_time,
                    'error': True
                }
            }
    
    async def _process_requests_parallel(self, requests: List[RequestType]) -> List[ResponseType]:
        """Process multiple requests in parallel."""
        # Create tasks for parallel processing
        tasks = []
        
        for request in requests:
            task = self._process_single_request_with_semaphore(request)
            tasks.append(task)
        
        # Execute all tasks in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and create valid responses
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Request {getattr(requests[i], 'id', i)} failed: {response}")
                error_response = await self._create_error_response(requests[i], str(response))
                valid_responses.append(error_response)
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    async def _process_single_request_with_semaphore(self, request: RequestType) -> ResponseType:
        """Process a single request with semaphore control."""
        async with self.processing_semaphore:
            return await self._process_single_request(request)
    
    async def _process_single_request(self, request: RequestType) -> ResponseType:
        """Process a single request using an available processor."""
        start_time = time.time()
        processor, processor_index = self.processor_pool.get_next_processor()
        
        try:
            # Process the request
            response = await self._execute_processor(processor, request, processor_index)
            
            processing_time = time.time() - start_time
            
            # Update processor statistics
            self.processor_pool.update_processor_stats(processor_index, processing_time, True)
            
            # Set response metadata
            if hasattr(response, 'processing_time'):
                response.processing_time = processing_time
            if hasattr(response, 'processor_id'):
                response.processor_id = f"processor_{processor_index}"
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update processor statistics for failure
            self.processor_pool.update_processor_stats(processor_index, processing_time, False)
            
            # Create error response
            return await self._create_error_response(request, str(e), processing_time, f"processor_{processor_index}")
    
    @abstractmethod
    async def _extract_requests(self, inputs: Dict[str, Any]) -> List[RequestType]:
        """Extract requests from input data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _execute_processor(self, processor: ProcessorType, request: RequestType, processor_index: int) -> ResponseType:
        """Execute a processor with a request. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _create_error_response(self, request: RequestType, error_message: str, 
                                   processing_time: float = 0.0, processor_id: str = "") -> ResponseType:
        """Create an error response. Must be implemented by subclasses."""
        pass
    
    def _response_to_dict(self, response: ResponseType) -> Dict[str, Any]:
        """Convert response to dictionary. Can be overridden by subclasses."""
        if hasattr(response, 'to_dict'):
            return response.to_dict()
        elif hasattr(response, '__dict__'):
            return response.__dict__
        else:
            return {'response': str(response)}
    
    def _update_performance_metrics(self, request_count: int, response_count: int, processing_time: float):
        """Update performance metrics."""
        self.logger.debug(f"Performance update: {request_count} requests, {response_count} responses, {processing_time:.3f}s",
                         total_requests=self.total_requests,
                         total_responses=self.total_responses,
                         avg_processing_time=self.total_processing_time / max(self.total_requests, 1))
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        base_stats = super().get_performance_stats()
        
        parallel_stats = {
            'total_requests': self.total_requests,
            'total_responses': self.total_responses,
            'total_processing_time': self.total_processing_time,
            'error_count': self.error_count,
            'avg_processing_time': self.total_processing_time / max(self.total_requests, 1),
            'success_rate': (self.total_responses - self.error_count) / max(self.total_responses, 1),
            'processor_pool_stats': self.processor_pool.get_stats()
        }
        
        return {**base_stats, **parallel_stats} 