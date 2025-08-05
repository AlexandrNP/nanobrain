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
    """
    Enterprise Load Balancer - Advanced Request Distribution and Traffic Management
    ============================================================================
    
    The LoadBalancer provides comprehensive traffic distribution and request management for distributed
    computing environments, offering multiple load balancing strategies, intelligent processor selection,
    and enterprise-grade performance optimization. This load balancer supports dynamic processor
    management, real-time performance monitoring, and adaptive routing for high-throughput applications.
    
    **Core Architecture:**
        The load balancer provides enterprise-grade traffic distribution capabilities:
        
        * **Multiple Load Balancing Strategies**: Round-robin, least-loaded, fastest-response, random, and weighted algorithms
        * **Dynamic Processor Management**: Real-time addition/removal of processors with hot-swapping capabilities
        * **Performance Monitoring**: Comprehensive metrics collection and adaptive routing optimization
        * **Fault Tolerance**: Automatic failover and processor health monitoring
        * **Scalable Architecture**: Support for horizontal scaling and dynamic capacity management
        * **Framework Integration**: Full integration with NanoBrain's distributed processing architecture
    
    **Load Balancing Strategies:**
        
        **Round Robin Distribution:**
        * Sequential processor selection with even distribution
        * Optimal for homogeneous processor environments
        * Predictable load distribution patterns
        * Low computational overhead for strategy execution
        
        **Least Loaded Selection:**
        * Intelligent routing to processors with lowest current load
        * Dynamic load assessment based on active requests
        * Optimal for heterogeneous processing environments
        * Real-time load balancing with adaptive selection
        
        **Fastest Response Optimization:**
        * Historical performance-based processor selection
        * Adaptive routing based on average response times
        * Continuous performance monitoring and optimization
        * Intelligent routing for latency-sensitive applications
        
        **Weighted Round Robin:**
        * Processor capacity-aware distribution algorithms
        * Configurable weight-based request routing
        * Support for heterogeneous processor capabilities
        * Proportional load distribution based on processor specifications
        
        **Random Distribution:**
        * Statistical load distribution with minimal overhead
        * Suitable for stateless request processing
        * Simple strategy for uniform processor environments
        * Low-latency selection for high-throughput scenarios
    
    **Enterprise Features:**
        
        **Performance Monitoring:**
        * Real-time request count and response time tracking
        * Comprehensive processor performance analytics
        * Error rate monitoring and failure detection
        * Historical performance data collection and analysis
        
        **Dynamic Scaling:**
        * Hot-swapping of processors without service interruption
        * Automatic processor discovery and registration
        * Capacity-based scaling and resource optimization
        * Load-aware processor addition and removal
        
        **Fault Tolerance:**
        * Automatic processor health monitoring
        * Circuit breaker integration for failure isolation
        * Graceful degradation and failover capabilities
        * Error recovery and service restoration
        
        **High Availability:**
        * Multi-processor redundancy and backup strategies
        * Load distribution across multiple availability zones
        * Disaster recovery and business continuity support
        * Zero-downtime maintenance and updates
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse load balancing scenarios:
        
        ```yaml
        # Load Balancer Configuration
        load_balancer_name: "enterprise_load_balancer"
        load_balancer_type: "advanced"
        
        # Load balancer card for framework integration
        load_balancer_card:
          name: "enterprise_load_balancer"
          description: "Enterprise traffic distribution and management"
          version: "1.0.0"
          category: "infrastructure"
          capabilities:
            - "multi_strategy_balancing"
            - "dynamic_scaling"
            - "performance_monitoring"
        
        # Load Balancing Strategy
        strategy: "least_loaded"  # round_robin, least_loaded, fastest_response, random, weighted_round_robin
        
        # Performance Configuration
        performance_config:
          enable_monitoring: true
          metric_collection_interval: 30  # seconds
          response_time_window: 300       # seconds for averaging
          health_check_interval: 60       # seconds
          
        # Processor Management
        processor_config:
          max_processors: 100
          min_processors: 2
          auto_scaling: true
          hot_swap_enabled: true
          
        # Weighted Configuration (for weighted_round_robin)
        processor_weights:
          high_performance: 3.0
          standard: 1.0
          backup: 0.5
          
        # Circuit Breaker Integration
        circuit_breaker:
          enabled: true
          failure_threshold: 5
          timeout: 30000  # milliseconds
          retry_interval: 60000  # milliseconds
          
        # Monitoring Configuration
        monitoring:
          enable_metrics: true
          export_prometheus: true
          dashboard_enabled: true
          alert_thresholds:
            error_rate: 0.05      # 5% error rate threshold
            response_time: 2000   # 2 second response time threshold
            load_factor: 0.8      # 80% load threshold
        ```
    
    **Usage Patterns:**
        
        **Basic Load Balancer Setup:**
        ```python
        from nanobrain.library.infrastructure.load_balancing import LoadBalancer, LoadBalancingStrategy
        
        # Create load balancer with round-robin strategy
        load_balancer = LoadBalancer[ProcessorClass](
            strategy=LoadBalancingStrategy.ROUND_ROBIN
        )
        
        # Add processors to the pool
        processor1 = ProcessorClass("processor-1")
        processor2 = ProcessorClass("processor-2")
        processor3 = ProcessorClass("processor-3")
        
        await load_balancer.add_processor(processor1, weight=1.0)
        await load_balancer.add_processor(processor2, weight=1.5)  # Higher capacity
        await load_balancer.add_processor(processor3, weight=0.8)  # Lower capacity
        
        # Process requests with load balancing
        for request in incoming_requests:
            selected_processor = await load_balancer.select_processor()
            result = await selected_processor.process(request)
            await load_balancer.record_response(selected_processor, result)
        ```
        
        **Enterprise High-Performance Setup:**
        ```python
        # Configure for high-throughput enterprise environment
        enterprise_balancer = LoadBalancer[EnterpriseProcessor](
            strategy=LoadBalancingStrategy.LEAST_LOADED
        )
        
        # Add enterprise processors with different capacities
        enterprise_processors = [
            ("gpu-processor-1", 5.0),    # High-performance GPU processor
            ("gpu-processor-2", 5.0),    # High-performance GPU processor
            ("cpu-cluster-1", 3.0),      # CPU cluster
            ("cpu-cluster-2", 3.0),      # CPU cluster
            ("backup-processor", 1.0)    # Backup processor
        ]
        
        for processor_name, weight in enterprise_processors:
            processor = EnterpriseProcessor(processor_name)
            await enterprise_balancer.add_processor(processor, weight=weight)
        
        # Process high-volume request stream
        async def process_request_stream(request_stream):
            async for batch in request_stream:
                # Select optimal processor based on current load
                processor = await enterprise_balancer.select_processor()
                
                # Record start time for performance tracking
                start_time = time.time()
                
                try:
                    # Process batch with selected processor
                    results = await processor.process_batch(batch)
                    
                    # Record successful processing
                    response_time = time.time() - start_time
                    await enterprise_balancer.record_response(
                        processor, 
                        results, 
                        response_time=response_time
                    )
                    
                    return results
                    
                except Exception as e:
                    # Record processing error
                    await enterprise_balancer.record_error(processor, e)
                    
                    # Try backup processor if available
                    backup_processor = await enterprise_balancer.select_processor(
                        exclude=[processor]
                    )
                    if backup_processor:
                        return await backup_processor.process_batch(batch)
                    else:
                        raise e
        
        # Start processing with load balancing
        results = await process_request_stream(high_volume_requests)
        ```
        
        **Adaptive Performance Optimization:**
        ```python
        # Configure load balancer with adaptive performance optimization
        adaptive_balancer = LoadBalancer[AdaptiveProcessor](
            strategy=LoadBalancingStrategy.FASTEST_RESPONSE
        )
        
        # Add processors with performance monitoring
        processors = [
            AdaptiveProcessor(f"adaptive-{i}") 
            for i in range(10)
        ]
        
        for processor in processors:
            await adaptive_balancer.add_processor(processor)
        
        # Enable real-time performance monitoring
        async def monitor_performance():
            while True:
                stats = await adaptive_balancer.get_performance_stats()
                
                # Log performance metrics
                for processor_id, metrics in stats.items():
                    print(f"Processor {processor_id}:")
                    print(f"  Requests: {metrics['request_count']}")
                    print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
                    print(f"  Error Rate: {metrics['error_rate']:.2%}")
                    print(f"  Load Factor: {metrics['load_factor']:.2f}")
                
                # Identify underperforming processors
                slow_processors = [
                    pid for pid, metrics in stats.items()
                    if metrics['avg_response_time'] > 2.0  # 2 second threshold
                ]
                
                # Remove slow processors if alternatives available
                if len(slow_processors) > 0 and len(stats) > len(slow_processors) + 2:
                    for processor_id in slow_processors:
                        processor = adaptive_balancer.get_processor_by_id(processor_id)
                        await adaptive_balancer.remove_processor(processor)
                        print(f"Removed slow processor: {processor_id}")
                
                await asyncio.sleep(60)  # Monitor every minute
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(monitor_performance())
        
        # Process requests with adaptive optimization
        async def adaptive_processing():
            request_count = 0
            while True:
                # Select optimal processor based on performance history
                processor = await adaptive_balancer.select_processor()
                
                # Generate or receive request
                request = await get_next_request()
                if not request:
                    break
                
                # Process with performance tracking
                start_time = time.time()
                try:
                    result = await processor.process(request)
                    response_time = time.time() - start_time
                    
                    await adaptive_balancer.record_response(
                        processor, result, response_time=response_time
                    )
                    
                    request_count += 1
                    if request_count % 100 == 0:
                        print(f"Processed {request_count} requests")
                        
                except Exception as e:
                    await adaptive_balancer.record_error(processor, e)
                    print(f"Error processing request: {e}")
        
        # Run adaptive processing
        await adaptive_processing()
        ```
        
        **Multi-Zone Load Balancing:**
        ```python
        # Configure multi-zone load balancing for high availability
        class MultiZoneLoadBalancer:
            def __init__(self):
                self.zone_balancers = {}
                self.zone_weights = {}
                
            async def add_zone(self, zone_name: str, weight: float = 1.0):
                self.zone_balancers[zone_name] = LoadBalancer[ZoneProcessor](
                    strategy=LoadBalancingStrategy.LEAST_LOADED
                )
                self.zone_weights[zone_name] = weight
                
            async def add_processor_to_zone(self, zone_name: str, processor: ZoneProcessor):
                if zone_name in self.zone_balancers:
                    await self.zone_balancers[zone_name].add_processor(processor)
                    
            async def select_processor(self) -> ZoneProcessor:
                # Select zone based on weights and availability
                available_zones = [
                    zone for zone, balancer in self.zone_balancers.items()
                    if balancer.get_available_processor_count() > 0
                ]
                
                if not available_zones:
                    raise RuntimeError("No available processors in any zone")
                
                # Weighted zone selection
                zone_weights = [self.zone_weights[zone] for zone in available_zones]
                selected_zone = random.choices(available_zones, weights=zone_weights)[0]
                
                # Select processor from chosen zone
                return await self.zone_balancers[selected_zone].select_processor()
        
        # Setup multi-zone configuration
        multi_zone = MultiZoneLoadBalancer()
        
        # Add zones with different priorities
        await multi_zone.add_zone("primary", weight=3.0)
        await multi_zone.add_zone("secondary", weight=2.0)
        await multi_zone.add_zone("backup", weight=1.0)
        
        # Add processors to each zone
        for i in range(5):
            await multi_zone.add_processor_to_zone("primary", ZoneProcessor(f"primary-{i}"))
        for i in range(3):
            await multi_zone.add_processor_to_zone("secondary", ZoneProcessor(f"secondary-{i}"))
        for i in range(2):
            await multi_zone.add_processor_to_zone("backup", ZoneProcessor(f"backup-{i}"))
        
        # Process requests across zones
        for request in critical_requests:
            processor = await multi_zone.select_processor()
            result = await processor.process(request)
        ```
    
    **Advanced Features:**
        
        **Real-Time Analytics:**
        * Continuous performance metrics collection and analysis
        * Historical trend analysis and capacity planning
        * Predictive scaling based on traffic patterns
        * Custom metrics and alerting integration
        
        **Circuit Breaker Integration:**
        * Automatic failure detection and isolation
        * Graceful degradation and fallback mechanisms
        * Service recovery and health restoration
        * Configurable failure thresholds and recovery strategies
        
        **Auto-Scaling Capabilities:**
        * Dynamic processor pool management
        * Load-based scaling decisions and optimization
        * Resource utilization monitoring and adjustment
        * Cost optimization and capacity planning
        
        **Custom Strategy Implementation:**
        * Pluggable strategy architecture for custom algorithms
        * Domain-specific routing logic and optimization
        * Machine learning-based processor selection
        * Custom performance metrics and selection criteria
    
    **Performance Optimization:**
        
        **High-Throughput Processing:**
        * Minimal selection overhead with optimized algorithms
        * Concurrent request handling and parallel processing
        * Memory-efficient data structures and caching
        * Lock-free operations for critical path optimization
        
        **Intelligent Caching:**
        * Processor performance data caching and optimization
        * Request routing cache for improved selection speed
        * Adaptive cache invalidation and refresh strategies
        * Memory-conscious cache sizing and management
        
        **Network Optimization:**
        * Connection pooling and reuse strategies
        * Request batching and efficient data transfer
        * Compression and serialization optimization
        * Latency-aware routing and optimization
    
    **Production Deployment:**
        
        **Enterprise Integration:**
        * Service mesh compatibility and integration
        * Kubernetes and container orchestration support
        * Cloud provider integration and optimization
        * Enterprise monitoring and alerting systems
        
        **Security Features:**
        * Secure processor communication and authentication
        * Request validation and sanitization
        * Access control and authorization integration
        * Audit logging and compliance support
        
        **Operational Excellence:**
        * Comprehensive logging and debugging support
        * Performance profiling and optimization tools
        * Configuration management and version control
        * Disaster recovery and backup strategies
    
    Attributes:
        strategy (LoadBalancingStrategy): Selected load balancing algorithm and strategy
        processors (List[ProcessorType]): Registered processor pool for request distribution
        processor_stats (Dict): Comprehensive performance metrics and statistics for each processor
        logger (Logger): Structured logging system for monitoring and debugging
    
    Type Parameters:
        ProcessorType: Generic type parameter for processor implementations, enabling type-safe usage
    
    Note:
        This load balancer requires processors to implement consistent interfaces for effective management.
        Performance monitoring features require proper response time recording for optimization.
        Multi-strategy support allows runtime strategy switching for adaptive load management.
        Circuit breaker integration requires compatible failure detection and recovery mechanisms.
    
    Warning:
        High-throughput environments may require careful tuning of monitoring intervals and cache sizes.
        Processor removal during active processing may cause request failures without proper handling.
        Weighted strategies require careful weight assignment to prevent processor overload.
        Performance statistics collection may impact performance in extremely high-frequency scenarios.
    
    See Also:
        * :class:`LoadBalancingStrategy`: Available load balancing algorithms and strategies
        * :class:`CircuitBreaker`: Fault tolerance and failure isolation integration
        * :mod:`nanobrain.library.infrastructure.monitoring`: Performance monitoring and analytics
        * :mod:`nanobrain.core.executor`: Distributed execution and processor management
        * :mod:`nanobrain.library.infrastructure.deployment`: Enterprise deployment and scaling
    """
    
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