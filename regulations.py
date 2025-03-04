import time
from typing import Any
from DataUnitBase import DataUnitBase
from enums import ComponentState

class ConnectionStrength:
    """
    Models the strength of connections between components.
    
    Biological analogy: Synaptic weights in neural networks.
    Justification: Like neural synapses that strengthen or weaken based on usage,
    connections between components should adaptively change in strength based on
    successful interactions, allowing the system to optimize data flow paths
    through experience.
    """
    def __init__(self, initial_strength: float = 0.5, min_strength: float = 0.0, max_strength: float = 1.0):
        self.strength = initial_strength
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.history = [(time.time(), initial_strength)]
    
    def increase(self, amount: float = 0.1) -> float:
        """
        Increase connection strength.
        
        Biological analogy: Long-Term Potentiation (LTP) in neural systems.
        Justification: In the brain, frequently used connections strengthen over time,
        improving efficiency for common operations.
        """
        self.strength = min(self.strength + amount, self.max_strength)
        self.history.append((time.time(), self.strength))
        return self.strength
    
    def decrease(self, amount: float = 0.1) -> float:
        """
        Decrease connection strength.
        
        Biological analogy: Long-Term Depression (LTD) in neural systems.
        Justification: In the brain, rarely used connections weaken over time,
        allowing resources to be reallocated to more important pathways.
        """
        self.strength = max(self.strength - amount, self.min_strength)
        self.history.append((time.time(), self.strength))
        return self.strength
    
    def adapt(self, source_activity: float, target_activity: float, adaptation_rate: float = 0.01) -> float:
        """
        Adapt connection strength based on correlated activity.
        
        Biological analogy: Hebbian learning - "Neurons that fire together, wire together"
        Justification: This implements an adaptive learning mechanism similar to how
        neural connections strengthen when input and output neurons activate together.
        """
        delta = adaptation_rate * source_activity * target_activity
        if delta > 0:
            self.increase(delta)
        else:
            self.decrease(abs(delta))
        return self.strength


class SystemRegulator:
    """
    Maintains system stability by regulating component behavior.
    
    Biological analogy: Homeostatic regulation in biological systems.
    Justification: Like how the body maintains stable internal conditions despite
    changing external conditions, software systems need mechanisms to maintain
    operational stability despite varying loads and conditions.
    """
    def __init__(self, target_value: float, acceptable_range: float = 0.2, correction_strength: float = 0.1):
        self.target_value = target_value
        self.acceptable_range = acceptable_range
        self.correction_strength = correction_strength
        self.current_value = target_value
    
    def regulate(self, current_value: float) -> float:
        """
        Adjusts the current value toward the target if it's outside acceptable range.
        Returns the correction factor to apply.
        
        Biological analogy: Negative feedback loops in biological homeostasis.
        Justification: Similar to how homeostatic mechanisms detect deviations from
        optimal conditions and generate corrective responses proportional to the deviation.
        """
        self.current_value = current_value
        
        # Check if within acceptable range
        lower_bound = self.target_value - self.acceptable_range
        upper_bound = self.target_value + self.acceptable_range
        
        if lower_bound <= current_value <= upper_bound:
            return 0.0  # No correction needed
            
        # Calculate correction
        if current_value < lower_bound:
            # Need to increase
            return self.correction_strength
        else:
            # Need to decrease
            return -self.correction_strength
    
    def is_stable(self) -> bool:
        """Checks if the current value is within the acceptable range."""
        lower_bound = self.target_value - self.acceptable_range
        upper_bound = self.target_value + self.acceptable_range
        return lower_bound <= self.current_value <= upper_bound


class SystemModulator:
    """
    Manages global system state parameters that influence component behavior.
    
    Biological analogy: Neuromodulator systems in the brain (dopamine, serotonin, etc.)
    Justification: Like how neuromodulators can broadly influence neural processing 
    across the brain, this class provides global parameters that can affect many 
    components simultaneously to tune system behavior.
    """
    def __init__(self):
        # Initialize various system modulators that affect behavior
        self.performance_modulator = 0.5  # Affects processing speed/throughput
        self.reliability_modulator = 0.5  # Affects error handling and robustness
        self.adaptability_modulator = 0.5  # Affects learning and configuration changes
        self.energy_efficiency_modulator = 0.5  # Affects resource usage
        
        self.modulators = {
            "performance": self.performance_modulator,
            "reliability": self.reliability_modulator, 
            "adaptability": self.adaptability_modulator,
            "energy_efficiency": self.energy_efficiency_modulator
        }
        
        # Regulators for each modulator
        self.regulators = {name: SystemRegulator(value) for name, value in self.modulators.items()}
    
    def get_modulator(self, name: str) -> float:
        """Get current level of a specific system modulator."""
        return self.modulators.get(name, 0.5)
    
    def set_modulator(self, name: str, value: float):
        """Set level of a specific system modulator."""
        if name in self.modulators:
            self.modulators[name] = max(0.0, min(1.0, value))
    
    def update_from_event(self, event_type: str, magnitude: float = 0.1):
        """
        Update modulator levels based on system events.
        
        Biological analogy: How environmental events trigger neuromodulator release.
        Justification: Different types of events should trigger appropriate system-wide
        responses, similar to how stress increases norepinephrine in biological systems.
        """
        if event_type == "success":
            # Success increases performance modulator
            self.modulators["performance"] = min(1.0, self.modulators["performance"] + magnitude)
            
        elif event_type == "failure":
            # Failure increases reliability modulator (to prevent future failures)
            self.modulators["reliability"] = min(1.0, self.modulators["reliability"] + magnitude)
            
        elif event_type == "config_change":
            # Configuration changes increase adaptability modulator
            self.modulators["adaptability"] = min(1.0, self.modulators["adaptability"] + magnitude)
            
        elif event_type == "resource_pressure":
            # Resource pressure increases energy efficiency modulator
            self.modulators["energy_efficiency"] = min(1.0, self.modulators["energy_efficiency"] + magnitude)
    
    def apply_regulation(self):
        """
        Apply homeostatic regulation to all modulators.
        
        Biological analogy: Homeostatic return to baseline conditions.
        Justification: Systems need to return to balanced states after perturbations
        to maintain long-term stability.
        """
        for name, regulator in self.regulators.items():
            correction = regulator.regulate(self.modulators[name])
            self.modulators[name] = max(0.0, min(1.0, self.modulators[name] + correction))


class ActivationGate:
    """
    Controls activation threshold and signal propagation.
    
    Biological analogy: Neural membrane with threshold potential.
    Justification: Like how a neuron's membrane determines when it fires based on
    incoming signals, this class determines when components activate based on inputs.
    """
    def __init__(self, threshold: float = 0.7, resting_level: float = 0.0, 
                 recovery_period: float = 1.0):
        self.level = resting_level
        self.threshold = threshold
        self.resting_level = resting_level
        self.recovery_period = recovery_period
        self.last_activation_time = 0
        self.state = ComponentState.INACTIVE
    
    def receive_signal(self, signal_strength: float) -> bool:
        """
        Receives an input signal and updates activation level.
        Returns True if threshold is crossed, False otherwise.
        
        Biological analogy: Integration of synaptic potentials at the neural membrane.
        Justification: Like how a neuron integrates incoming signals and fires when a
        threshold is reached, components should activate only when input signals are strong enough.
        """
        current_time = time.time()
        
        # Check if in recovery period
        if current_time - self.last_activation_time < self.recovery_period:
            self.state = ComponentState.RECOVERING
            return False
            
        # Update level
        self.level += signal_strength
        
        # Check for activation
        if self.level >= self.threshold:
            self.last_activation_time = current_time
            self.level = self.resting_level
            self.state = ComponentState.ACTIVE
            return True
        
        self.state = ComponentState.INACTIVE
        return False
    
    def decay(self, amount: float = 0.1):
        """
        Gradually return to resting level over time.
        
        Biological analogy: Leaky integrate-and-fire neuron model.
        Justification: Neural membranes naturally leak current, causing the membrane
        potential to return to its resting state in the absence of stimulation.
        """
        if self.level > self.resting_level:
            self.level = max(self.resting_level, self.level - amount)
        elif self.level < self.resting_level:
            self.level = min(self.resting_level, self.level + amount)

