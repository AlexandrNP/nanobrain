import time
from typing import List
from enums import ComponentState


class InhibitorySignal:
    """
    Represents a blocking signal that prevents component activation.
    
    Biological analogy: Inhibitory neurotransmitters (GABA, glycine).
    Justification: Like how inhibitory neurotransmitters prevent neurons from
    firing despite excitatory inputs, inhibitory signals block components
    from activating despite receiving activation signals.
    """
    def __init__(self, strength: float = 0.8, duration: float = 5.0):
        self.strength = strength
        self.duration = duration
        self.start_time = None
    
    def activate(self):
        """
        Activate the inhibitory signal.
        
        Biological analogy: Release of inhibitory neurotransmitters.
        Justification: When activated, inhibitory neurons release transmitters
        that temporarily prevent target neurons from firing.
        """
        self.start_time = time.time()
    
    def is_active(self) -> bool:
        """
        Check if the inhibitory signal is still active.
        
        Biological analogy: Duration of inhibitory postsynaptic potentials.
        Justification: Inhibitory signals have a specific duration before dissipating.
        """
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) < self.duration
    
    def get_strength(self) -> float:
        """
        Get the current inhibitory strength.
        
        Biological analogy: Decreasing concentration of inhibitory neurotransmitters.
        Justification: The effect of inhibition gradually decreases as neurotransmitters
        are cleared from the synaptic cleft.
        """
        if not self.is_active():
            return 0.0
        
        # Strength decays over time
        elapsed = time.time() - self.start_time
        remaining_ratio = max(0.0, (self.duration - elapsed) / self.duration)
        return self.strength * remaining_ratio


class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern to prevent cascade failures.
    
    Biological analogy: Neural inhibitory circuits.
    Justification: Like how inhibitory neurons prevent runaway excitation in neural
    circuits, the circuit breaker prevents cascading failures in the system by 
    blocking activation when too many errors occur.
    """
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.state = ComponentState.INACTIVE
        self.last_failure_time = 0
        self.inhibitory_signal = InhibitorySignal()
    
    def record_success(self):
        """
        Record a successful operation and reset failure count.
        
        Biological analogy: Successful processing reinforces normal function.
        Justification: Successful neural operations reinforce normal activation
        patterns and reduce inhibitory tone.
        """
        self.failure_count = max(0, self.failure_count - 1)  # Gradual decrease
        if self.failure_count == 0:
            self.state = ComponentState.INACTIVE
    
    def record_failure(self):
        """
        Record a failure and potentially open the circuit.
        
        Biological analogy: Recruitment of inhibitory neurons after abnormal activity.
        Justification: Like how seizure activity recruits inhibitory mechanisms,
        repeated failures trigger protective inhibition.
        """
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = ComponentState.BLOCKED
            self.inhibitory_signal.activate()
    
    def can_execute(self) -> bool:
        """
        Determines if execution should be allowed.
        
        Biological analogy: Gating of neural activity based on circuit state.
        Justification: Like how inhibitory gating controls information flow in
        neural circuits, the circuit breaker gates execution based on system state.
        """
        current_time = time.time()
        
        # Regular circuit breaker logic with component states
        if self.state == ComponentState.INACTIVE:
            return True
        
        if self.state == ComponentState.BLOCKED:
            if (current_time - self.last_failure_time) > self.reset_timeout:
                # Try a test request
                self.state = ComponentState.ENHANCED  # More sensitive state
                return True
        
        if self.state == ComponentState.ENHANCED:
            # Recovering state - allows execution but sensitive to failures
            return True
            
        return False
    
    def get_inhibition_level(self) -> float:
        """
        Returns the current level of inhibition (0.0-1.0).
        
        Biological analogy: Strength of inhibitory tone in a neural circuit.
        Justification: Inhibitory strength in neural circuits varies continuously,
        affecting the likelihood of circuit activation.
        """
        if self.state == ComponentState.BLOCKED:
            return self.inhibitory_signal.get_strength()
        return 0.0


class DeadlockDetector:
    """
    Detects potential deadlocks in the workflow by monitoring dependency cycles.
    
    Biological analogy: Neural circuits that detect and break pathological synchronization.
    Justification: Like how the brain prevents seizures by detecting and disrupting over-synchronized
    neural activity, this class prevents workflow deadlocks by detecting and breaking resource contention cycles.
    """
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.resource_locks = {}  # resource_id -> owner_id
        self.waiting_for = {}     # owner_id -> resource_id
        self.timestamp = {}       # owner_id -> lock_acquisition_time
        self.inhibitory_signals = {}  # owner_id -> InhibitorySignal
    
    def request_resource(self, owner_id: str, resource_id: str) -> bool:
        """
        Attempts to acquire a resource lock.
        
        Biological analogy: Competition for limited metabolic resources.
        Justification: Like how neural circuits compete for limited metabolic
        resources, components compete for computational resources.
        """
        current_time = time.time()
        
        # Check if this owner is currently inhibited
        if owner_id in self.inhibitory_signals and self.inhibitory_signals[owner_id].is_active():
            # This component is being inhibited to break a deadlock
            return False
        
        # Check if resource is available
        if resource_id in self.resource_locks:
            # Resource is locked
            self.waiting_for[owner_id] = resource_id
            
            # Check for deadlock
            if self._detect_cycle(owner_id, set()):
                # Apply inhibitory signal to break the deadlock
                self._resolve_deadlock(owner_id)
                return False
                
            return False
        
        # Resource is available
        self.resource_locks[resource_id] = owner_id
        self.timestamp[owner_id] = current_time
        
        # Check for timeouts
        self._check_timeouts(current_time)
        
        return True
    
    def release_resource(self, owner_id: str, resource_id: str):
        """
        Releases a resource lock.
        
        Biological analogy: Release of metabolic resources after neural activity.
        Justification: Like how neurons release metabolic resources after completing
        activity, components should release computational resources after use.
        """
        if resource_id in self.resource_locks and self.resource_locks[resource_id] == owner_id:
            del self.resource_locks[resource_id]
            
        if owner_id in self.waiting_for:
            del self.waiting_for[owner_id]
            
        if owner_id in self.timestamp:
            del self.timestamp[owner_id]
    
    def _detect_cycle(self, start_id: str, visited: set) -> bool:
        """
        Detects cycles in the wait-for graph.
        
        Biological analogy: Detection of pathological neural synchronization.
        Justification: Like how certain brain circuits detect over-synchronized
        neural activity that could lead to seizures, this method detects 
        circular wait patterns that could lead to deadlocks.
        """
        if start_id in visited:
            return True
            
        if start_id not in self.waiting_for:
            return False
            
        visited.add(start_id)
        
        # Get the resource this owner is waiting for
        resource_id = self.waiting_for[start_id]
        
        # Get the owner of that resource
        if resource_id in self.resource_locks:
            next_owner = self.resource_locks[resource_id]
            return self._detect_cycle(next_owner, visited)
            
        return False
    
    def _resolve_deadlock(self, owner_id: str):
        """
        Resolves deadlock by applying inhibitory signals.
        
        Biological analogy: Breaking pathological neural synchronization.
        Justification: Like how inhibitory neurons break up over-synchronized
        neural activity to prevent seizures, this method applies inhibitory
        signals to break up deadlocks.
        """
        # Find the cycle
        cycle = self._find_cycle(owner_id, [])
        
        if not cycle:
            return
            
        # Apply inhibitory signals to components in the cycle
        # Use different strengths and durations to stagger recovery
        for i, component_id in enumerate(cycle):
            # Create an inhibitory signal with staggered duration
            strength = 0.7 + (0.3 * (i / len(cycle)))  # 0.7-1.0
            duration = self.timeout * (0.5 + (0.5 * (i / len(cycle))))  # 0.5-1.0 times timeout
            
            inhibitory_signal = InhibitorySignal(strength=strength, duration=duration)
            inhibitory_signal.activate()
            
            self.inhibitory_signals[component_id] = inhibitory_signal
            
            # Release resources held by this component
            for resource, owner in list(self.resource_locks.items()):
                if owner == component_id:
                    self.release_resource(component_id, resource)
    
    def _find_cycle(self, start_id: str, path: List[str]) -> List[str]:
        """
        Finds and returns the cycle in the wait-for graph.
        
        Biological analogy: Tracing the loop in a reverberating neural circuit.
        Justification: Like identifying the specific neurons involved in a
        self-sustaining loop of activity.
        """
        path.append(start_id)
        
        if start_id not in self.waiting_for:
            return []
            
        resource_id = self.waiting_for[start_id]
        
        if resource_id not in self.resource_locks:
            return []
            
        next_owner = self.resource_locks[resource_id]
        
        if next_owner in path:
            # Found cycle
            cycle_start_index = path.index(next_owner)
            return path[cycle_start_index:]
        
        return self._find_cycle(next_owner, path)
    
    def _check_timeouts(self, current_time: float):
        """
        Checks for timed-out locks and releases them.
        
        Biological analogy: Timeout mechanisms in neural processing.
        Justification: The brain has mechanisms to prevent resources from
        being locked indefinitely, ensuring processing can continue even
        after unexpected interruptions.
        """
        for owner_id, timestamp in list(self.timestamp.items()):
            if current_time - timestamp > self.timeout:
                # Release all resources held by this owner
                for resource_id, resource_owner in list(self.resource_locks.items()):
                    if resource_owner == owner_id:
                        self.release_resource(owner_id, resource_id)