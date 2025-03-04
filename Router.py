from enums import ExecutorBase, ComponentState
from LinkBase import LinkBase
from regulations import ConnectionStrength, CircuitBreaker
from typing import List, Dict, Any
from collections import deque
from interfaces import IRunnable
from Runner import Runner
import random
import time
import asyncio


class Router(IRunnable):
    """
    Routes one input to multiple outputs with adaptive routing and resource management.
    
    Biological analogy: Combined axonal terminal and routing functionality.
    Justification: Like how axon terminals manage neurotransmitter resources and
    adaptively route signals to multiple targets, this router manages transmission
    resources and adaptively routes data to multiple outputs.
    """
    def __init__(self, executor: ExecutorBase, input_source: LinkBase, 
                 output_sinks: List[LinkBase], **kwargs):
        self.runner = Runner(executor)
        self.input_source = input_source
        self.output_sinks = output_sinks
        self.fanout_reliability = 0.9  # Reliability decreases with more outputs
        self.routing_strategy = "adaptive"  # broadcast, random, weighted, adaptive
        self.sink_weights = [ConnectionStrength() for _ in output_sinks]
        
        # Resource management for each sink
        self.resource_levels = [1.0] * len(output_sinks)  # Full capacity initially
        self.recovery_rates = [0.1] * len(output_sinks)  # Rate of resource recovery
        
        # Circuit breakers for each sink
        self.circuit_breakers = [CircuitBreaker() for _ in output_sinks]
        
        # Performance metrics
        self.metrics = {
            'latency': deque(maxlen=100),  # Last 100 latency measurements
            'success_rate': deque(maxlen=100),  # Last 100 success/failure records
            'load': [0] * len(output_sinks),  # Current load per sink
            'resource_usage': [deque(maxlen=100) for _ in output_sinks]  # Resource usage history
        }
        
        # Adaptive routing parameters
        self.learning_rate = 0.05  # How quickly weights adapt
        self.exploration_rate = 0.1  # Probability of trying new routes
        self.load_threshold = 0.8  # Load level that triggers load balancing
        self.error_threshold = 0.3  # Error rate that triggers circuit breaker
        self.recovery_time = 5.0  # Seconds to wait before retrying failed sink
        self.resource_threshold = 0.1  # Minimum resource level for transmission
        
        # Sink state tracking
        self.sink_states = [ComponentState.INACTIVE for _ in output_sinks]
        self.last_error_time = [0.0 for _ in output_sinks]
        self.consecutive_failures = [0 for _ in output_sinks]
    
    async def execute(self):
        """
        Gets data from input and routes to outputs using adaptive strategies.
        
        Biological analogy: Adaptive neurotransmitter release and routing.
        Justification: Like how synaptic terminals adaptively manage and release
        neurotransmitters to multiple targets, this method manages resources
        and routes data to multiple outputs.
        """
        start_time = time.time()
        results = []
        success = False
        
        try:
            # Get input data
            await self.input_source.transfer()
            data = self.input_source.output.get()
            
            if self.routing_strategy == "adaptive":
                results = await self._adaptive_route(data)
            elif self.routing_strategy == "broadcast":
                results = await self._broadcast_route(data)
            elif self.routing_strategy == "random":
                results = await self._random_route(data)
            elif self.routing_strategy == "weighted":
                results = await self._weighted_route(data)
            
            success = True
            
        except Exception as e:
            self._handle_error(e)
            raise
            
        finally:
            # Update metrics
            latency = time.time() - start_time
            self.metrics['latency'].append(latency)
            self.metrics['success_rate'].append(1.0 if success else 0.0)
            
            # Update resource usage metrics
            for i in range(len(self.output_sinks)):
                self.metrics['resource_usage'][i].append(self.resource_levels[i])
            
            # Trigger recovery if needed
            if not success:
                asyncio.create_task(self._recover())
            
            # Always trigger resource recovery
            asyncio.create_task(self._recover_resources())
        
        return results
    
    async def _adaptive_route(self, data: Any) -> List[Any]:
        """
        Implements adaptive routing based on performance metrics and load.
        """
        results = []
        available_sinks = self._get_available_sinks()
        
        if not available_sinks:
            raise RuntimeError("No available output sinks")
            
        # Calculate routing probabilities based on weights and load
        total_weight = sum(self.sink_weights[i].strength * (1.0 - self.metrics['load'][i])
                          for i in available_sinks)
                          
        if total_weight <= 0:
            # If all weights are zero, use uniform distribution
            probabilities = [1.0 / len(available_sinks)] * len(available_sinks)
        else:
            # Calculate normalized probabilities
            probabilities = [(self.sink_weights[i].strength * (1.0 - self.metrics['load'][i])) / total_weight
                           for i in available_sinks]
        
        # Route data based on probabilities
        for i, prob in zip(available_sinks, probabilities):
            if random.random() < prob or random.random() < self.exploration_rate:
                try:
                    result = await self._send_to_sink(data, i)
                    results.append(result)
                    
                    # Update metrics and weights based on success
                    self._update_sink_metrics(i, True)
                    self.sink_weights[i].increase(self.learning_rate)
                    
                except Exception as e:
                    self._update_sink_metrics(i, False)
                    self.sink_weights[i].decrease(self.learning_rate)
                    
        return results
    
    async def _broadcast_route(self, data: Any) -> List[Any]:
        """
        Implements broadcast routing with improved error handling.
        """
        results = []
        for i, sink in enumerate(self.output_sinks):
            if self._can_use_sink(i):
                try:
                    result = await self._send_to_sink(data, i)
                    results.append(result)
                    self._update_sink_metrics(i, True)
                except Exception as e:
                    self._update_sink_metrics(i, False)
        return results
    
    async def _random_route(self, data: Any) -> List[Any]:
        """
        Implements random routing with reliability checks.
        """
        results = []
        for i, sink in enumerate(self.output_sinks):
            if self._can_use_sink(i) and random.random() < self.fanout_reliability:
                try:
                    result = await self._send_to_sink(data, i)
                    results.append(result)
                    self._update_sink_metrics(i, True)
                except Exception as e:
                    self._update_sink_metrics(i, False)
        return results
    
    async def _weighted_route(self, data: Any) -> List[Any]:
        """
        Implements weighted routing with dynamic weight adjustment.
        """
        results = []
        for i, sink in enumerate(self.output_sinks):
            if self._can_use_sink(i) and random.random() < self.sink_weights[i].strength:
                try:
                    result = await self._send_to_sink(data, i)
                    results.append(result)
                    self._update_sink_metrics(i, True)
                    self.sink_weights[i].increase(0.01)
                except Exception as e:
                    self._update_sink_metrics(i, False)
                    self.sink_weights[i].decrease(0.005)
        return results
    
    async def _send_to_sink(self, data: Any, sink_index: int) -> Any:
        """
        Sends data to a specific sink with resource management.
        
        Biological analogy: Neurotransmitter release at a synapse.
        Justification: Like how synapses manage neurotransmitter resources
        during signal transmission, this method manages transmission resources
        for each sink.
        """
        sink = self.output_sinks[sink_index]
        
        if not self.circuit_breakers[sink_index].can_execute():
            raise RuntimeError(f"Circuit breaker open for sink {sink_index}")
            
        if self.resource_levels[sink_index] < self.resource_threshold:
            raise RuntimeError(f"Insufficient resources for sink {sink_index}")
            
        try:
            # Consume resources for transmission
            self.resource_levels[sink_index] = max(0.0, 
                self.resource_levels[sink_index] - self.resource_threshold)
            
            sink.input.set(data)
            await sink.transfer()
            result = sink.output.get()
            
            # Update load metrics
            self.metrics['load'][sink_index] = min(1.0, 
                self.metrics['load'][sink_index] + 0.1)
            
            return result
            
        except Exception as e:
            self.circuit_breakers[sink_index].record_failure()
            raise
    
    async def _recover_resources(self):
        """
        Recovers transmission resources for all sinks.
        
        Biological analogy: Neurotransmitter reuptake and synthesis.
        Justification: Like how synapses recover their neurotransmitter
        resources over time, this method restores transmission capacity
        for each sink.
        """
        for i in range(len(self.output_sinks)):
            self.resource_levels[i] = min(1.0, 
                self.resource_levels[i] + self.recovery_rates[i])
    
    def _can_use_sink(self, sink_index: int) -> bool:
        """
        Checks if a sink is available for use.
        """
        # Check circuit breaker
        if not self.circuit_breakers[sink_index].can_execute():
            return False
            
        # Check sink state
        if self.sink_states[sink_index] != ComponentState.INACTIVE:
            return False
            
        # Check error recovery time
        if time.time() - self.last_error_time[sink_index] < self.recovery_time:
            return False
            
        # Check load
        if self.metrics['load'][sink_index] > self.load_threshold:
            return False
            
        # Check resources
        if self.resource_levels[sink_index] < self.resource_threshold:
            return False
            
        return True
    
    def _get_available_sinks(self) -> List[int]:
        """
        Returns indices of available sinks.
        """
        return [i for i in range(len(self.output_sinks)) if self._can_use_sink(i)]
    
    def _update_sink_metrics(self, sink_index: int, success: bool):
        """
        Updates metrics for a sink based on execution result.
        """
        if success:
            self.consecutive_failures[sink_index] = 0
        else:
            self.consecutive_failures[sink_index] += 1
            self.last_error_time[sink_index] = time.time()
            
            if self.consecutive_failures[sink_index] >= 3:
                self.sink_states[sink_index] = ComponentState.RECOVERING
                
        # Decay load over time
        self.metrics['load'][sink_index] = max(0.0, self.metrics['load'][sink_index] - 0.05)
    
    def _handle_error(self, error: Exception):
        """
        Handles routing errors and updates system state.
        """
        # Update system modulators
        self.runner.executor.system_modulators.update_from_event("failure", 0.1)
        
        # Log error for monitoring
        print(f"Router error: {str(error)}")
    
    async def _recover(self):
        """
        Implements recovery logic for failed sinks.
        """
        for i in range(len(self.output_sinks)):
            if self.sink_states[i] == ComponentState.RECOVERING:
                # Wait for recovery time
                await asyncio.sleep(self.recovery_time)
                
                # Reset sink state
                self.sink_states[i] = ComponentState.INACTIVE
                self.consecutive_failures[i] = 0
                self.circuit_breakers[i].reset()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns current routing metrics.
        """
        return {
            'avg_latency': sum(self.metrics['latency']) / len(self.metrics['latency']) if self.metrics['latency'] else 0,
            'success_rate': sum(self.metrics['success_rate']) / len(self.metrics['success_rate']) if self.metrics['success_rate'] else 0,
            'load': self.metrics['load'],
            'sink_weights': [w.strength for w in self.sink_weights],
            'circuit_breaker_states': [cb.can_execute() for cb in self.circuit_breakers],
            'resource_usage': [list(deque) for deque in self.metrics['resource_usage']]
        }
        
    # IRunnable interface implementation
    async def invoke(self) -> Any:
        """Execute the router and return results."""
        return await self.execute()
        
    def check_runnable_config(self) -> bool:
        """Check if router is properly configured."""
        return self.runner.check_runnable_config()
        
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration from runner."""
        return self.runner.get_config(class_dir)
        
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Update configuration through runner."""
        return self.runner.update_config(updates, adaptability_threshold)
        
    @property
    def adaptability(self) -> float:
        """Get adaptability from runner."""
        return self.runner.adaptability
        
    @adaptability.setter
    def adaptability(self, value: float):
        """Set adaptability through runner."""
        self.runner.adaptability = value
        
    @property
    def state(self):
        """Get state from runner."""
        return self.runner.state
        
    @state.setter
    def state(self, value):
        """Set state through runner."""
        self.runner.state = value
        
    @property
    def running(self) -> bool:
        """Get running state from runner."""
        return self.runner.running
        
    @running.setter
    def running(self, value: bool):
        """Set running state through runner."""
        self.runner.running = value