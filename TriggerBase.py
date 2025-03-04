from typing import Any
from enums import ActivationGate
from ConfigManager import ConfigManager
import random

class TriggerBase:
    """
    Base trigger class that checks conditions for execution.
    
    Biological analogy: Sensory neuron or receptor.
    Justification: Like how sensory neurons detect environmental changes and
    initiate processing pathways, triggers detect conditions and initiate
    workflow execution.
    """
    def __init__(self, runnable: Any, **kwargs):
        self.runnable = runnable
        self.config_manager = ConfigManager(**kwargs)
        self.activation_gate = ActivationGate(threshold=0.3)  # More sensitive than regular components
        self.sensitivity = 0.8  # Sensitivity to stimuli (0.0-1.0)
        self.adaptation_rate = 0.05  # How quickly it adapts to repeated stimuli
    
    def check_condition(self, **kwargs) -> bool:
        """
        Checks if condition is met. To be implemented by subclasses.
        
        Biological analogy: Stimulus detection in sensory receptors.
        Justification: Like how different sensory receptors detect specific
        types of stimuli (light, sound, pressure), different trigger
        subclasses detect specific conditions.
        """
        raise NotImplementedError("Subclasses must implement check_condition()")
    
    async def monitor(self, **kwargs):
        """
        Monitors for condition and triggers runnable when met.
        
        Biological analogy: Sensory transduction and adaptation.
        Justification: Like how sensory neurons convert environmental stimuli
        into neural signals but adapt to persistent stimuli, triggers convert
        conditions into workflow activations but adapt to repeated occurrences.
        """
        if self.check_condition(**kwargs):
            # Adapt to repeated stimuli (like sensory adaptation)
            self.sensitivity = max(0.2, self.sensitivity - self.adaptation_rate)
            
            # Check if we're sensitive enough to respond
            if random.random() < self.sensitivity:
                return await self.runnable.invoke()
        else:
            # Recover sensitivity during periods of no stimulation
            self.sensitivity = min(1.0, self.sensitivity + (self.adaptation_rate / 2))
        
        return None
        
    def get_config(self, class_dir: str = None) -> dict:
        """Delegate to config manager."""
        return self.config_manager.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)