from typing import Any, Optional
from src.TriggerBase import TriggerBase

class TriggerDataChanged:
    """
    Trigger that activates when the source step's output data changes.
    
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
        Checks if the source step's output data has changed since last check.
        
        Biological analogy: Neural adaptation and change detection.
        Justification: Like how certain neurons adapt to constant stimuli
        but respond vigorously to changes, this trigger ignores constant 
        data but responds to changes.
        """
        # Get the current data from the source step's output
        if not self.base_trigger.runnable or not hasattr(self.base_trigger.runnable, 'source_step'):
            return False
            
        source_step = self.base_trigger.runnable.source_step
        if not hasattr(source_step, 'output') or source_step.output is None:
            return False
            
        current_data = source_step.output.get()
        
        # If we're ignoring None values and the current data is None, return False
        if self.ignore_none and current_data is None:
            return False
        
        # Check if the data has changed
        has_changed = current_data != self.last_data
        
        # Update the last data
        self.last_data = current_data
        
        return has_changed
        
    async def monitor(self, **kwargs):
        """
        Monitor for data changes.
        
        Biological analogy: Continuous sensory monitoring.
        Justification: Like how sensory neurons continuously monitor for
        changes in stimuli, this method continuously checks for changes
        in source step output data.
        """
        return await self.base_trigger.monitor(**kwargs)
        
    # Delegate methods to base trigger
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration for this class."""
        return self.base_trigger.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Update configuration if adaptability threshold is met."""
        return self.base_trigger.update_config(updates, adaptability_threshold)
        
    @property
    def runnable(self):
        """Get the runnable object."""
        return self.base_trigger.runnable
        
    @runnable.setter
    def runnable(self, value):
        """
        Set the runnable object.
        
        This also resets the last_data to ensure proper change detection.
        """
        self.base_trigger.runnable = value
        # Reset the last data when the runnable changes
        self.last_data = None
        
    @property
    def sensitivity(self):
        """Get the sensitivity level."""
        return self.base_trigger.sensitivity
        
    @sensitivity.setter
    def sensitivity(self, value):
        """Set the sensitivity level."""
        self.base_trigger.sensitivity = value
        
    @property
    def adaptation_rate(self):
        """Get the adaptation rate."""
        return self.base_trigger.adaptation_rate
        
    @adaptation_rate.setter
    def adaptation_rate(self, value):
        """Set the adaptation rate."""
        self.base_trigger.adaptation_rate = value
        
    @property
    def activation_gate(self):
        """Get the activation gate."""
        return self.base_trigger.activation_gate
        
    @activation_gate.setter
    def activation_gate(self, value):
        """Set the activation gate."""
        self.base_trigger.activation_gate = value 