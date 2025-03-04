from typing import Any
from TriggerBase import TriggerBase
import random

class TriggerStart:
    """
    Trigger that always returns True, starting execution immediately.
    
    Biological analogy: Pacemaker neuron.
    Justification: Like how pacemaker neurons spontaneously initiate signals
    without external input (e.g., in the cardiac system), this trigger
    spontaneously initiates workflow execution.
    """
    def __init__(self, runnable: Any, **kwargs):
        self.base_trigger = TriggerBase(runnable, **kwargs)
        self.spontaneous_rate = 1.0  # High probability of spontaneous firing
        
    def check_condition(self, **kwargs) -> bool:
        """
        Almost always returns True, with some randomness to model 
        biological variability.
        
        Biological analogy: Spontaneous firing in pacemaker neurons.
        Justification: Pacemaker neurons have intrinsic activity with
        some biological variability in their firing patterns.
        """
        return random.random() < self.spontaneous_rate
        
    async def monitor(self, **kwargs):
        """Delegate to base trigger."""
        return await self.base_trigger.monitor(**kwargs)
        
    # Delegate methods to base trigger
    def get_config(self, class_dir: str = None) -> dict:
        return self.base_trigger.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        return self.base_trigger.update_config(updates, adaptability_threshold)
        
    @property
    def sensitivity(self):
        return self.base_trigger.sensitivity
        
    @sensitivity.setter
    def sensitivity(self, value):
        self.base_trigger.sensitivity = value
        
    @property
    def adaptation_rate(self):
        return self.base_trigger.adaptation_rate
        
    @adaptation_rate.setter
    def adaptation_rate(self, value):
        self.base_trigger.adaptation_rate = value
        
    @property
    def activation_gate(self):
        return self.base_trigger.activation_gate
    