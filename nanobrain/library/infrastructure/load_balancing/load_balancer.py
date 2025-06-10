"""
Load balancing strategies implementation.

Multiple load balancing strategies for distributing work across processors.
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic
from collections import defaultdict
from nanobrain.core.logging_system import get_logger

ProcessorType = TypeVar('ProcessorType')


class LoadBalancingStrategy(Enum):
    """Load balancing strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    RANDOM = "random"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"


class LoadBalancer(Generic[ProcessorType]):
    """Multiple load balancing strategies for request distribution."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.processors: List[ProcessorType] = []
        self.processor_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'request_count': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'error_count': 0,
            'last_used': 0.0,
            'weight': 1.0,
            'active_requests': 0
        })
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        self.logger = get_logger("load_balancer")
        
    async def add_processor(self, processor: ProcessorType, weight: float = 1.0) -> None:
        """Add a processor to the load balancer."""
        async with self._lock:
            self.processors.append(processor)
            processor_id = self._get_processor_id(processor)
            self.processor_stats[processor_id]['weight'] = weight
            self.logger.debug(f"Added processor {processor_id} with weight {weight}")
            
    async def remove_processor(self, processor: ProcessorType) -> bool:
        """Remove a processor from the load balancer."""
        async with self._lock:
            if processor in self.processors:
                self.processors.remove(processor)
                processor_id = self._get_processor_id(processor)
                if processor_id in self.processor_stats:
                    del self.processor_stats[processor_id]
                self.logger.debug(f"Removed processor {processor_id}")
                return True
            return False
            
    async def select_processor(self) -> Optional[ProcessorType]:
        """Select a processor based on the configured strategy."""
        async with self._lock:
            if not self.processors:
                return None
                
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return await self._round_robin_select()
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return await self._least_loaded_select()
            elif self.strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
                return await self._fastest_response_select()
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return await self._random_select()
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return await self._weighted_round_robin_select()
            else:
                return await self._round_robin_select()
                
    async def _round_robin_select(self) -> ProcessorType:
        """Round-robin processor selection."""
        processor = self.processors[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(self.processors)
        return processor
        
    async def _least_loaded_select(self) -> ProcessorType:
        """Select processor with least active requests."""
        min_load = float('inf')
        selected_processor = self.processors[0]
        
        for processor in self.processors:
            processor_id = self._get_processor_id(processor)
            active_requests = self.processor_stats[processor_id]['active_requests']
            if active_requests < min_load:
                min_load = active_requests
                selected_processor = processor
                
        return selected_processor
        
    async def _fastest_response_select(self) -> ProcessorType:
        """Select processor with fastest average response time."""
        min_response_time = float('inf')
        selected_processor = self.processors[0]
        
        for processor in self.processors:
            processor_id = self._get_processor_id(processor)
            avg_response_time = self.processor_stats[processor_id]['avg_response_time']
            
            # If no previous response time, give it a chance
            if avg_response_time == 0.0:
                return processor
                
            if avg_response_time < min_response_time:
                min_response_time = avg_response_time
                selected_processor = processor
                
        return selected_processor
        
    async def _random_select(self) -> ProcessorType:
        """Random processor selection."""
        return random.choice(self.processors)
        
    async def _weighted_round_robin_select(self) -> ProcessorType:
        """Weighted round-robin selection based on processor weights."""
        # Create weighted list
        weighted_processors = []
        for processor in self.processors:
            processor_id = self._get_processor_id(processor)
            weight = int(self.processor_stats[processor_id]['weight'])
            weighted_processors.extend([processor] * weight)
            
        if not weighted_processors:
            return self.processors[0]
            
        processor = weighted_processors[self._round_robin_index % len(weighted_processors)]
        self._round_robin_index += 1
        return processor
        
    async def record_request_start(self, processor: ProcessorType) -> None:
        """Record the start of a request for a processor."""
        async with self._lock:
            processor_id = self._get_processor_id(processor)
            self.processor_stats[processor_id]['active_requests'] += 1
            self.processor_stats[processor_id]['last_used'] = time.time()
            
    async def record_request_end(self, processor: ProcessorType, response_time: float, success: bool = True) -> None:
        """Record the end of a request for a processor."""
        async with self._lock:
            processor_id = self._get_processor_id(processor)
            stats = self.processor_stats[processor_id]
            
            stats['active_requests'] = max(0, stats['active_requests'] - 1)
            stats['request_count'] += 1
            
            if success:
                stats['total_response_time'] += response_time
                stats['avg_response_time'] = stats['total_response_time'] / stats['request_count']
            else:
                stats['error_count'] += 1
                
    async def get_processor_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all processors."""
        async with self._lock:
            return dict(self.processor_stats)
            
    async def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across processors."""
        async with self._lock:
            total_requests = sum(stats['request_count'] for stats in self.processor_stats.values())
            
            if total_requests == 0:
                return {self._get_processor_id(p): 0.0 for p in self.processors}
                
            distribution = {}
            for processor in self.processors:
                processor_id = self._get_processor_id(processor)
                request_count = self.processor_stats[processor_id]['request_count']
                distribution[processor_id] = (request_count / total_requests) * 100
                
            return distribution
            
    async def reset_stats(self) -> None:
        """Reset all processor statistics."""
        async with self._lock:
            for stats in self.processor_stats.values():
                stats.update({
                    'request_count': 0,
                    'total_response_time': 0.0,
                    'avg_response_time': 0.0,
                    'error_count': 0,
                    'active_requests': 0
                })
            self.logger.info("Reset all processor statistics")
            
    async def set_processor_weight(self, processor: ProcessorType, weight: float) -> None:
        """Set weight for a processor (used in weighted strategies)."""
        async with self._lock:
            processor_id = self._get_processor_id(processor)
            if processor_id in self.processor_stats:
                self.processor_stats[processor_id]['weight'] = weight
                self.logger.debug(f"Set weight {weight} for processor {processor_id}")
                
    async def get_healthy_processors(self, error_threshold: float = 0.1) -> List[ProcessorType]:
        """Get processors with error rate below threshold."""
        healthy_processors = []
        
        async with self._lock:
            for processor in self.processors:
                processor_id = self._get_processor_id(processor)
                stats = self.processor_stats[processor_id]
                
                if stats['request_count'] == 0:
                    # No requests yet, consider healthy
                    healthy_processors.append(processor)
                else:
                    error_rate = stats['error_count'] / stats['request_count']
                    if error_rate <= error_threshold:
                        healthy_processors.append(processor)
                        
        return healthy_processors
        
    def _get_processor_id(self, processor: ProcessorType) -> str:
        """Get unique identifier for a processor."""
        if hasattr(processor, 'name'):
            return processor.name
        elif hasattr(processor, 'id'):
            return processor.id
        else:
            return str(id(processor))
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        async with self._lock:
            total_requests = sum(stats['request_count'] for stats in self.processor_stats.values())
            total_errors = sum(stats['error_count'] for stats in self.processor_stats.values())
            total_active = sum(stats['active_requests'] for stats in self.processor_stats.values())
            
            avg_response_times = [
                stats['avg_response_time'] 
                for stats in self.processor_stats.values() 
                if stats['avg_response_time'] > 0
            ]
            
            overall_avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0.0
            
            return {
                'strategy': self.strategy.value,
                'total_processors': len(self.processors),
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': (total_errors / total_requests) if total_requests > 0 else 0.0,
                'active_requests': total_active,
                'overall_avg_response_time': overall_avg_response_time,
                'load_distribution': await self.get_load_distribution()
            } 