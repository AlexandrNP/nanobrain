from typing import Any, List
from TriggerBase import TriggerBase
from LinkBase import LinkBase

class TriggerAllDataReceived:
    """
    Trigger that activates when all input data sources have received data.
    
    Biological analogy: Integration neuron.
    Justification: Like how some neurons only fire when receiving input from
    multiple sources (e.g., coincidence detectors), this trigger only activates
    when all data sources have provided input.
    """
    def __init__(self, runnable: Any, input_sources: List[LinkBase], **kwargs):
        self.base_trigger = TriggerBase(runnable, **kwargs)
        self.input_sources = input_sources
        self.required_inputs = len(input_sources)  # Number of inputs needed
        self.integration_threshold = 1.0  # All inputs must be present
        
    def check_condition(self, **kwargs) -> bool:
        """
        Checks if all input sources have received data.
        
        Biological analogy: Synaptic integration.
        Justification: Like how neurons integrate inputs from multiple synapses
        to determine if firing threshold is reached, this trigger checks if
        enough inputs are present to initiate execution.
        """
        received_count = sum(1 for source in self.input_sources if source.output.get() is not None)
        return (received_count / self.required_inputs) >= self.integration_threshold
        
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