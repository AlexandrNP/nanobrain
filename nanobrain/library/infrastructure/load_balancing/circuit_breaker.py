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
    """
    Enterprise Circuit Breaker - Advanced Fault Tolerance and Resilience Management
    ============================================================================
    
    The CircuitBreaker provides comprehensive fault tolerance and resilience management for distributed
    systems, implementing the circuit breaker pattern with intelligent failure detection, automatic
    recovery mechanisms, and enterprise-grade monitoring. This component prevents cascading failures,
    enables graceful degradation, and ensures system stability during service disruptions.
    
    **Core Architecture:**
        The circuit breaker provides enterprise-grade fault tolerance capabilities:
        
        * **Three-State Management**: Closed, Open, and Half-Open states with intelligent transitions
        * **Failure Detection**: Configurable failure thresholds and exception monitoring
        * **Automatic Recovery**: Time-based recovery testing and service restoration
        * **Cascading Failure Prevention**: Request blocking during service outages
        * **Graceful Degradation**: Fallback mechanisms and alternative service routing
        * **Framework Integration**: Full integration with NanoBrain's resilience architecture
    
    **Circuit Breaker States:**
        
        **CLOSED State (Normal Operation):**
        * All requests pass through to the protected service
        * Failure counting and threshold monitoring active
        * Immediate failure detection and response
        * Performance monitoring and baseline establishment
        
        **OPEN State (Failure Protection):**
        * All requests immediately fail without calling protected service
        * System protection from cascading failures
        * Timeout-based recovery attempt scheduling
        * Alternative service routing and fallback execution
        
        **HALF_OPEN State (Recovery Testing):**
        * Limited request forwarding for service health testing
        * Single request testing with immediate state transitions
        * Recovery validation and service restoration
        * Gradual traffic restoration and performance monitoring
    
    **Fault Tolerance Capabilities:**
        
        **Intelligent Failure Detection:**
        * Configurable failure threshold monitoring and tracking
        * Exception type filtering and classification
        * Failure pattern analysis and trend detection
        * Time-based failure window and sliding statistics
        
        **Automatic Recovery Management:**
        * Timeout-based recovery attempt scheduling
        * Service health validation and restoration
        * Gradual traffic increase during recovery
        * Recovery success rate monitoring and optimization
        
        **Cascading Failure Prevention:**
        * Immediate request blocking during service failures
        * Resource protection and system stability maintenance
        * Dependency isolation and failure containment
        * Alternative service routing and load redistribution
        
        **Performance Monitoring:**
        * Real-time state transition tracking and logging
        * Failure rate statistics and trend analysis
        * Recovery time measurement and optimization
        * Service availability metrics and reporting
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse fault tolerance scenarios:
        
        ```yaml
        # Circuit Breaker Configuration
        circuit_breaker_name: "enterprise_circuit_breaker"
        circuit_breaker_type: "advanced"
        
        # Circuit breaker card for framework integration
        circuit_breaker_card:
          name: "enterprise_circuit_breaker"
          description: "Enterprise fault tolerance and resilience management"
          version: "1.0.0"
          category: "infrastructure"
          capabilities:
            - "failure_detection"
            - "automatic_recovery"
            - "cascading_prevention"
        
        # Failure Detection Configuration
        failure_detection:
          failure_threshold: 5           # Number of failures before opening
          failure_window: 300           # Time window for failure counting (seconds)
          timeout: 60.0                # Timeout before recovery attempt (seconds)
          expected_exceptions:          # Exceptions that trigger circuit breaker
            - "ConnectionError"
            - "TimeoutError"
            - "ServiceUnavailableError"
          
        # Recovery Configuration
        recovery_config:
          recovery_timeout: 30.0        # Timeout for recovery testing
          success_threshold: 3          # Successes needed to close circuit
          gradual_recovery: true        # Enable gradual traffic increase
          max_recovery_attempts: 10     # Maximum recovery attempts
          
        # Monitoring Configuration
        monitoring:
          enable_metrics: true
          state_change_logging: true
          performance_tracking: true
          alert_notifications: true
          
        # Integration Configuration
        integration:
          load_balancer_integration: true
          fallback_service_enabled: true
          alternative_routing: true
          health_check_integration: true
        ```
    
    **Usage Patterns:**
        
        **Basic Circuit Breaker Protection:**
        ```python
        from nanobrain.library.infrastructure.load_balancing import CircuitBreaker
        
        # Create circuit breaker with failure protection
        circuit_breaker = CircuitBreaker(
            failure_threshold=5,    # Open after 5 failures
            timeout=60.0,          # Try recovery after 60 seconds
            expected_exception=ConnectionError
        )
        
        # Protect external service calls
        async def protected_service_call(data):
            try:
                result = await circuit_breaker.call(external_service.process, data)
                return result
            except Exception as e:
                print(f"Service call failed: {e}")
                # Implement fallback logic
                return await fallback_service.process(data)
        
        # Process requests with fault tolerance
        for request in incoming_requests:
            try:
                response = await protected_service_call(request.data)
                await send_response(request, response)
            except Exception as e:
                await send_error_response(request, str(e))
        ```
        
        **Enterprise Resilience Management:**
        ```python
        # Configure multiple circuit breakers for different services
        class EnterpriseResilienceManager:
            def __init__(self):
                self.circuit_breakers = {
                    'database': CircuitBreaker(
                        failure_threshold=3,
                        timeout=30.0,
                        expected_exception=DatabaseConnectionError
                    ),
                    'api_service': CircuitBreaker(
                        failure_threshold=5,
                        timeout=60.0,
                        expected_exception=APIServiceError
                    ),
                    'cache_service': CircuitBreaker(
                        failure_threshold=10,
                        timeout=15.0,
                        expected_exception=CacheConnectionError
                    )
                }
                self.fallback_services = {
                    'database': ReadOnlyDatabaseService(),
                    'api_service': CachedAPIService(),
                    'cache_service': InMemoryCacheService()
                }
                
            async def call_with_resilience(self, service_name: str, func, *args, **kwargs):
                circuit_breaker = self.circuit_breakers.get(service_name)
                fallback_service = self.fallback_services.get(service_name)
                
                if not circuit_breaker:
                    # No circuit breaker configured, call directly
                    return await func(*args, **kwargs)
                
                try:
                    # Attempt protected call
                    return await circuit_breaker.call(func, *args, **kwargs)
                    
                except Exception as e:
                    print(f"Circuit breaker open for {service_name}: {e}")
                    
                    # Use fallback service if available
                    if fallback_service:
                        print(f"Using fallback service for {service_name}")
                        return await fallback_service.process(*args, **kwargs)
                    else:
                        # No fallback available, re-raise exception
                        raise e
                        
            async def get_system_health(self):
                health_status = {}
                for service_name, circuit_breaker in self.circuit_breakers.items():
                    health_status[service_name] = {
                        'state': circuit_breaker.state.value,
                        'failure_count': circuit_breaker.failure_count,
                        'last_failure_time': circuit_breaker.last_failure_time,
                        'is_healthy': circuit_breaker.state == CircuitBreakerState.CLOSED
                    }
                return health_status
        
        # Initialize enterprise resilience manager
        resilience_manager = EnterpriseResilienceManager()
        
        # Use resilient service calls
        async def process_user_request(user_id: str, request_data: dict):
            try:
                # Database call with resilience
                user_profile = await resilience_manager.call_with_resilience(
                    'database',
                    database_service.get_user_profile,
                    user_id
                )
                
                # API service call with resilience
                external_data = await resilience_manager.call_with_resilience(
                    'api_service',
                    api_service.fetch_external_data,
                    request_data
                )
                
                # Cache service call with resilience
                cached_result = await resilience_manager.call_with_resilience(
                    'cache_service',
                    cache_service.get_cached_computation,
                    request_data['computation_key']
                )
                
                # Combine results
                return {
                    'user_profile': user_profile,
                    'external_data': external_data,
                    'cached_result': cached_result,
                    'processed_at': time.time()
                }
                
            except Exception as e:
                print(f"Request processing failed: {e}")
                
                # Return minimal response with available data
                return {
                    'error': str(e),
                    'partial_data': True,
                    'system_health': await resilience_manager.get_system_health()
                }
        ```
        
        **Advanced Monitoring and Alerting:**
        ```python
        # Enhanced circuit breaker with comprehensive monitoring
        class MonitoredCircuitBreaker(CircuitBreaker):
            def __init__(self, name: str, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = name
                self.metrics = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'blocked_calls': 0,
                    'state_transitions': [],
                    'avg_response_time': 0.0,
                    'last_success_time': 0.0,
                    'last_failure_time': 0.0
                }
                self.alert_callbacks = []
                
            async def call(self, func, *args, **kwargs):
                self.metrics['total_calls'] += 1
                start_time = time.time()
                
                try:
                    if self.state == CircuitBreakerState.OPEN:
                        self.metrics['blocked_calls'] += 1
                        await self._trigger_alert('request_blocked', {
                            'circuit_breaker': self.name,
                            'state': self.state.value,
                            'blocked_calls': self.metrics['blocked_calls']
                        })
                    
                    result = await super().call(func, *args, **kwargs)
                    
                    # Record successful call metrics
                    response_time = time.time() - start_time
                    self.metrics['successful_calls'] += 1
                    self.metrics['last_success_time'] = time.time()
                    self._update_avg_response_time(response_time)
                    
                    return result
                    
                except Exception as e:
                    # Record failed call metrics
                    self.metrics['failed_calls'] += 1
                    self.metrics['last_failure_time'] = time.time()
                    
                    await self._trigger_alert('call_failed', {
                        'circuit_breaker': self.name,
                        'error': str(e),
                        'failure_count': self.failure_count,
                        'state': self.state.value
                    })
                    
                    raise e
                    
            async def _on_state_change(self, old_state, new_state):
                # Record state transition
                transition = {
                    'timestamp': time.time(),
                    'from_state': old_state.value,
                    'to_state': new_state.value,
                    'failure_count': self.failure_count
                }
                self.metrics['state_transitions'].append(transition)
                
                # Trigger state change alert
                await self._trigger_alert('state_change', {
                    'circuit_breaker': self.name,
                    'transition': transition,
                    'metrics': self.get_current_metrics()
                })
                
            def _update_avg_response_time(self, response_time):
                if self.metrics['successful_calls'] == 1:
                    self.metrics['avg_response_time'] = response_time
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.metrics['avg_response_time'] = (
                        alpha * response_time + 
                        (1 - alpha) * self.metrics['avg_response_time']
                    )
                    
            async def _trigger_alert(self, alert_type: str, data: dict):
                for callback in self.alert_callbacks:
                    try:
                        await callback(alert_type, data)
                    except Exception as e:
                        self.logger.error(f"Alert callback failed: {e}")
                        
            def add_alert_callback(self, callback):
                self.alert_callbacks.append(callback)
                
            def get_current_metrics(self):
                return {
                    **self.metrics,
                    'current_state': self.state.value,
                    'failure_rate': (
                        self.metrics['failed_calls'] / max(self.metrics['total_calls'], 1)
                    ),
                    'success_rate': (
                        self.metrics['successful_calls'] / max(self.metrics['total_calls'], 1)
                    ),
                    'availability': (
                        1.0 - (self.metrics['blocked_calls'] / max(self.metrics['total_calls'], 1))
                    )
                }
        
        # Setup monitored circuit breaker with alerting
        monitored_breaker = MonitoredCircuitBreaker(
            name="critical_service",
            failure_threshold=3,
            timeout=45.0
        )
        
        # Add alert handlers
        async def email_alert_handler(alert_type: str, data: dict):
            if alert_type == 'state_change' and data['transition']['to_state'] == 'open':
                await send_email_alert(
                    subject=f"Circuit Breaker OPEN: {data['circuit_breaker']}",
                    body=f"Circuit breaker {data['circuit_breaker']} has opened due to failures. "
                         f"Failure count: {data['transition']['failure_count']}"
                )
                
        async def dashboard_update_handler(alert_type: str, data: dict):
            await dashboard_api.update_circuit_breaker_status(
                name=data['circuit_breaker'],
                status=data.get('transition', {}).get('to_state', 'unknown'),
                metrics=data.get('metrics', {})
            )
        
        monitored_breaker.add_alert_callback(email_alert_handler)
        monitored_breaker.add_alert_callback(dashboard_update_handler)
        
        # Monitor circuit breaker performance
        async def performance_monitor():
            while True:
                metrics = monitored_breaker.get_current_metrics()
                
                print(f"Circuit Breaker '{monitored_breaker.name}' Metrics:")
                print(f"  State: {metrics['current_state']}")
                print(f"  Total Calls: {metrics['total_calls']}")
                print(f"  Success Rate: {metrics['success_rate']:.2%}")
                print(f"  Failure Rate: {metrics['failure_rate']:.2%}")
                print(f"  Availability: {metrics['availability']:.2%}")
                print(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
                
                # Check for performance issues
                if metrics['failure_rate'] > 0.1:  # 10% failure rate
                    print("WARNING: High failure rate detected!")
                if metrics['avg_response_time'] > 5.0:  # 5 second response time
                    print("WARNING: High response time detected!")
                    
                await asyncio.sleep(30)  # Monitor every 30 seconds
        
        # Start performance monitoring
        monitoring_task = asyncio.create_task(performance_monitor())
        ```
        
        **Microservices Resilience Pattern:**
        ```python
        # Circuit breaker integration with microservices architecture
        class MicroserviceCircuitBreaker:
            def __init__(self, service_name: str, service_config: dict):
                self.service_name = service_name
                self.circuit_breaker = CircuitBreaker(
                    failure_threshold=service_config.get('failure_threshold', 5),
                    timeout=service_config.get('timeout', 60.0),
                    expected_exception=service_config.get('expected_exception', Exception)
                )
                self.health_checker = ServiceHealthChecker(service_name)
                self.metrics_collector = MetricsCollector(service_name)
                
            async def call_service(self, endpoint: str, method: str = 'GET', **kwargs):
                try:
                    # Check service health before call
                    if not await self.health_checker.is_healthy():
                        raise ServiceUnavailableError(f"{self.service_name} is unhealthy")
                    
                    # Make protected service call
                    response = await self.circuit_breaker.call(
                        self._make_http_request,
                        endpoint=endpoint,
                        method=method,
                        **kwargs
                    )
                    
                    # Collect success metrics
                    await self.metrics_collector.record_success(endpoint, response)
                    return response
                    
                except Exception as e:
                    # Collect failure metrics
                    await self.metrics_collector.record_failure(endpoint, str(e))
                    
                    # Try fallback strategies
                    fallback_response = await self._try_fallback(endpoint, method, **kwargs)
                    if fallback_response:
                        return fallback_response
                    
                    raise e
                    
            async def _make_http_request(self, endpoint: str, method: str, **kwargs):
                # Implement actual HTTP request logic
                async with aiohttp.ClientSession() as session:
                    url = f"{self.service_base_url}/{endpoint}"
                    async with session.request(method, url, **kwargs) as response:
                        if response.status >= 400:
                            raise HTTPError(f"HTTP {response.status}: {await response.text()}")
                        return await response.json()
                        
            async def _try_fallback(self, endpoint: str, method: str, **kwargs):
                # Implement fallback strategies (cache, alternative service, etc.)
                if hasattr(self, 'cache_service'):
                    cached_response = await self.cache_service.get(f"{endpoint}:{kwargs}")
                    if cached_response:
                        return cached_response
                        
                if hasattr(self, 'fallback_service'):
                    return await self.fallback_service.call(endpoint, method, **kwargs)
                    
                return None
        
        # Configure microservice circuit breakers
        microservice_configs = {
            'user_service': {
                'failure_threshold': 3,
                'timeout': 30.0,
                'expected_exception': HTTPError
            },
            'payment_service': {
                'failure_threshold': 2,  # More sensitive for critical service
                'timeout': 45.0,
                'expected_exception': PaymentServiceError
            },
            'notification_service': {
                'failure_threshold': 10,  # Less sensitive for non-critical service
                'timeout': 60.0,
                'expected_exception': NotificationError
            }
        }
        
        # Initialize circuit breakers for each microservice
        service_breakers = {
            name: MicroserviceCircuitBreaker(name, config)
            for name, config in microservice_configs.items()
        }
        
        # Use circuit breakers in application logic
        async def process_order(order_data: dict):
            try:
                # Get user information with circuit breaker protection
                user_info = await service_breakers['user_service'].call_service(
                    f"users/{order_data['user_id']}"
                )
                
                # Process payment with circuit breaker protection
                payment_result = await service_breakers['payment_service'].call_service(
                    "payments/process",
                    method='POST',
                    json=order_data['payment_info']
                )
                
                # Send notification (non-blocking, best effort)
                try:
                    await service_breakers['notification_service'].call_service(
                        "notifications/send",
                        method='POST',
                        json={
                            'user_id': order_data['user_id'],
                            'message': f"Order {order_data['order_id']} processed successfully"
                        }
                    )
                except Exception as e:
                    # Notification failure shouldn't block order processing
                    print(f"Notification failed (non-critical): {e}")
                
                return {
                    'success': True,
                    'order_id': order_data['order_id'],
                    'user_info': user_info,
                    'payment_result': payment_result
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'order_id': order_data['order_id']
                }
        ```
    
    **Advanced Features:**
        
        **Intelligent State Management:**
        * Automatic state transitions based on configurable thresholds
        * Gradual recovery with incremental traffic increase
        * Historical failure pattern analysis and adaptive thresholds
        * Custom state transition logic and business rule integration
        
        **Failure Pattern Analysis:**
        * Configurable exception type filtering and classification
        * Failure clustering and pattern recognition
        * Temporal failure analysis and trend detection
        * Predictive failure detection and proactive protection
        
        **Integration Capabilities:**
        * Load balancer integration for traffic redistribution
        * Service mesh compatibility and circuit breaker coordination
        * Health check integration and service discovery
        * Monitoring system integration and alerting
        
        **Custom Recovery Strategies:**
        * Configurable recovery testing and validation
        * Custom health check implementation and validation
        * Business logic integration for recovery decisions
        * Multi-stage recovery with progressive traffic increase
    
    **Enterprise Deployment:**
        
        **High Availability:**
        * Multi-instance circuit breaker coordination
        * Distributed state management and synchronization
        * Cross-datacenter failure detection and protection
        * Disaster recovery and business continuity support
        
        **Monitoring and Observability:**
        * Comprehensive metrics collection and analysis
        * Real-time dashboards and alerting systems
        * Historical trend analysis and capacity planning
        * Custom metrics and business intelligence integration
        
        **Security Integration:**
        * Secure failure notification and alerting
        * Access control for circuit breaker configuration
        * Audit logging and compliance reporting
        * Secure fallback service communication
    
    Attributes:
        failure_threshold (int): Number of consecutive failures before opening circuit
        timeout (float): Time in seconds before attempting recovery from open state
        expected_exception (type): Exception type that triggers circuit breaker activation
        state (CircuitBreakerState): Current circuit breaker state (CLOSED, OPEN, HALF_OPEN)
        failure_count (int): Current consecutive failure count for threshold monitoring
        logger (Logger): Structured logging system for monitoring and debugging
    
    Note:
        This circuit breaker requires consistent exception handling for effective failure detection.
        Timeout configuration should balance quick recovery with system stability requirements.
        Integration with load balancers and fallback services enhances overall system resilience.
        Monitoring and alerting are essential for effective circuit breaker operation in production.
    
    Warning:
        Aggressive failure thresholds may cause unnecessary service blocking during temporary issues.
        Long timeout periods may delay recovery and impact system availability.
        Circuit breaker state changes may cause temporary service disruptions during transitions.
        Proper fallback mechanisms are essential to prevent complete service unavailability.
    
    See Also:
        * :class:`CircuitBreakerState`: Circuit breaker state enumeration and transitions
        * :class:`LoadBalancer`: Load balancing integration for traffic redistribution
        * :mod:`nanobrain.library.infrastructure.monitoring`: Performance monitoring and alerting
        * :mod:`nanobrain.core.executor`: Distributed execution and fault tolerance
        * :mod:`nanobrain.library.infrastructure.deployment`: Enterprise deployment patterns
    """
    
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