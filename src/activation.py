import time
from typing import Any

class ActivationGate:
    """
    Controls activation based on threshold and recovery dynamics.
    
    Biological analogy: Neural membrane potential and action potential threshold.
    Justification: Like how neurons have activation thresholds and refractory
    periods, this gate controls when components can activate based on signal
    strength and recovery state.
    """
    def __init__(self, threshold: float = 0.7, resting_level: float = 0.0, 
                 recovery_period: float = 1.0):
        self.threshold = threshold
        self.resting_level = resting_level
        self.recovery_period = recovery_period
        self.current_level = resting_level
        self.last_activation_time = 0
        self.is_refractory = False
    
    def receive_signal(self, signal_strength: float) -> bool:
        """
        Receives a signal and determines if it crosses the activation threshold.
        
        Biological analogy: Neural integration and firing.
        Justification: Like how neurons integrate inputs and fire if threshold
        is reached, this method integrates signal strength with current level
        and determines if activation occurs.
        """
        current_time = time.time()
        
        # Check if in refractory period
        if self.is_refractory:
            if current_time - self.last_activation_time >= self.recovery_period:
                self.is_refractory = False
                self.current_level = self.resting_level
            else:
                return False
        
        # Integrate signal
        self.current_level += signal_strength
        
        # Check if threshold is crossed
        if self.current_level >= self.threshold:
            # Activation occurs
            self.last_activation_time = current_time
            self.is_refractory = True
            self.current_level = self.resting_level
            return True
            
        return False
    
    def decay(self, amount: float = 0.1):
        """
        Decays the current activation level toward resting level.
        
        Biological analogy: Leaky integration in neurons.
        Justification: Like how neural membrane potential decays over time,
        this method reduces the current level toward the resting level.
        """
        if self.current_level > self.resting_level:
            self.current_level = max(self.resting_level, 
                                    self.current_level - amount)
        elif self.current_level < self.resting_level:
            self.current_level = min(self.resting_level, 
                                    self.current_level + amount) 