from typing import Any, Optional
from src.TriggerBase import TriggerBase
from src.LinkBase import LinkBase

class TriggerDataChanged:
    """
    Trigger that activates when input data of a link changes.
    
    Biological analogy: Change detector neuron.
    Justification: Like how some neurons in sensory systems respond
    specifically to changes in input rather than constant stimuli,
    this trigger activates when data changes rather than at all times.
    """
    def __init__(self, runnable: Optional[Any] = None, **kwargs):
        self.base_trigger = TriggerBase(runnable, **kwargs)
        self.last_data = None
        self.ignore_none = True  # Whether to ignore None values
        
    def check_condition(self, **kwargs) -> bool:
        """
        Checks if the input data has changed since last check.
        
        Biological analogy: Neural adaptation and change detection.
        Justification: Like how certain neurons adapt to constant stimuli
        but respond vigorously to changes, this trigger ignores constant 
        data but responds to changes.
        """
        # Get the current data from the runnable's input
        if not self.base_trigger.runnable or not hasattr(self.base_trigger.runnable, 'input'):
            return False
            
        current_data = self.base_trigger.runnable.input.get()
        
        # If we're ignoring None values and the current data is None, return False
        if self.ignore_none and current_data is None:
            return False
        
        # Check if the data has changed
        has_changed = current_data != self.last_data
        
        # Update the last data
        self.last_data = current_data
        
        return has_changed
        
    async def monitor(self, **kwargs):
        """Delegate to base trigger."""
        return await self.base_trigger.monitor(**kwargs)
        
    # Delegate methods to base trigger
    def get_config(self, class_dir: str = None) -> dict:
        return self.base_trigger.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        return self.base_trigger.update_config(updates, adaptability_threshold)
        
    @property
    def runnable(self):
        return self.base_trigger.runnable
        
    @runnable.setter
    def runnable(self, value):
        self.base_trigger.runnable = value
        # Reset the last data when the runnable changes
        self.last_data = None
        
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