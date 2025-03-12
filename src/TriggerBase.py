import random
from typing import Any, Optional
from src.ActivationGate import ActivationGate
from src.ConfigManager import ConfigManager
from src.DirectoryTracer import DirectoryTracer
from src.enums import ComponentState

class TriggerBase:
    """
    Base class for components that detect conditions for activation.
    
    Biological analogy: Sensory neuron.
    Justification: Like how sensory neurons detect specific environmental
    conditions and convert them to neural signals, triggers detect specific
    computational conditions and convert them to workflow activations.
    """
    def __init__(self, runnable: Any, **kwargs):
        self.runnable = runnable
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        self.activation_gate = ActivationGate(threshold=0.3)  # More sensitive than regular components
        self.sensitivity = 0.8  # Initial sensitivity to conditions
        self.adaptation_rate = 0.05  # How quickly sensitivity changes
    
    def check_condition(self, **kwargs) -> bool:
        """
        Checks if the condition for triggering is met.
        
        Biological analogy: Sensory transduction.
        Justification: Like how sensory neurons convert specific environmental
        stimuli into neural signals, this method converts specific computational
        conditions into boolean signals.
        """
        # Base implementation always returns False
        # Subclasses should override this
        return False
    
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
        """Get configuration for this class."""
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)