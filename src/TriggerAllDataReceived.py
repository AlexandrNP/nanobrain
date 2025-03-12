from typing import Any, List, Dict
from src.TriggerBase import TriggerBase
from src.DataUnitBase import DataUnitBase
import asyncio

class TriggerAllDataReceived(TriggerBase):
    """
    Trigger that activates when all input data sources have received data.
    
    Biological analogy: Integration neuron.
    Justification: Like how some neurons only fire when receiving input from
    multiple sources (e.g., coincidence detectors), this trigger only activates
    when all data sources have provided input.
    """
    def __init__(self, step: Any, **kwargs):
        # Initialize attributes that are set by properties in parent class
        self._sensitivity = 0.8
        self._adaptation_rate = 0.05
        self._activation_gate = None
        
        # Now call parent init
        super().__init__(step, **kwargs)
        
        # Verify step has input_sources dictionary
        if not hasattr(step, 'input_sources'):
            raise ValueError("Step must have input_sources dictionary attribute")
            
        self.step = step
        self.integration_threshold = 1.0  # All inputs must be present
        
    def check_condition(self, **kwargs) -> bool:
        """
        Checks if all input sources in the step have received data.
        
        Biological analogy: Synaptic integration.
        Justification: Like how neurons integrate inputs from multiple synapses
        to determine if firing threshold is reached, this trigger checks if
        enough inputs are present to initiate execution.
        """
        if not self.step or not hasattr(self.step, 'input_sources') or not self.step.input_sources:
            return True
            
        # Check if all input sources have data
        all_data_received = True
        for link_id, data_unit in self.step.input_sources.items():
            if data_unit.get() is None:
                all_data_received = False
                break
                
        return all_data_received
        
    async def monitor(self, **kwargs):
        """
        Monitor input sources for data availability.
        
        Biological analogy: Continuous synaptic integration.
        Justification: Like how neurons continuously monitor synaptic inputs
        to determine when to fire, this trigger continuously monitors input
        sources to determine when to activate.
        """
        # Check if condition is met
        if self.check_condition(**kwargs):
            return True
            
        # Wait a bit and check again
        await asyncio.sleep(0.1)
        return self.check_condition(**kwargs)
        
    def update_input_sources(self):
        """
        Update knowledge of the step's input sources.
        
        Biological analogy: Synaptic remodeling.
        Justification: Like how neurons can form new synaptic connections
        and prune existing ones, this trigger can update its understanding
        of the step's input sources.
        """
        # Nothing to do here as we always get the latest input_sources directly from the step
        pass
        
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration for this class."""
        return super().get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Update configuration if adaptability threshold is met."""
        return super().update_config(updates, adaptability_threshold)
        
    @property
    def sensitivity(self):
        """Get trigger sensitivity."""
        return self._sensitivity
        
    @sensitivity.setter
    def sensitivity(self, value):
        """Set trigger sensitivity."""
        self._sensitivity = value
        
    @property
    def adaptation_rate(self):
        """Get adaptation rate."""
        return self._adaptation_rate
        
    @adaptation_rate.setter
    def adaptation_rate(self, value):
        """Set adaptation rate."""
        self._adaptation_rate = value
        
    @property
    def activation_gate(self):
        """Get the activation gate."""
        return self._activation_gate
        
    @activation_gate.setter
    def activation_gate(self, value):
        """Set the activation gate.
        
        Biological analogy: Adjusting activation threshold.
        Justification: Like how neurons can adjust their firing thresholds,
        triggers can adjust their activation thresholds.
        """
        self._activation_gate = value