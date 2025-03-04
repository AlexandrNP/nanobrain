from TriggerBase import TriggerBase
from mixins import RunnableMixin
from enums import DataUnitBase
from regulations import ConnectionStrength
from typing import List


class TriggerAllDataReceived(TriggerBase):
    """
    Trigger that waits for all input data units to be updated.
    
    Biological analogy: Integrator neuron requiring multiple inputs.
    Justification: Like how some neurons require multiple strong inputs
    to reach firing threshold (e.g., in coincidence detection), this trigger
    requires multiple data sources to be updated before activating.
    """
    def __init__(self, runnable: RunnableMixin, dendrites: List[DataUnitBase], **kwargs):
        super().__init__(runnable, **kwargs)
        self.dendrites = dendrites
        self.integration_threshold = len(dendrites) * 0.7  # Need 70% of inputs
        self.weights = [ConnectionStrength() for _ in dendrites]  # Different weights for each input
    
    def check_condition(self, **kwargs) -> bool:
        """
        Returns True if enough dendrites are updated to cross threshold.
        
        Biological analogy: Neural integration of multiple inputs.
        Justification: Like how neurons integrate multiple incoming signals
        and fire only when their combined strength crosses a threshold,
        this trigger activates only when enough data sources are updated.
        """
        # Count updated dendrites, weighted by their importance
        weighted_sum = sum(self.weights[i].strength for i, dendrite in enumerate(self.dendrites) 
                           if dendrite.updated)
        
        # Apply activation model
        signal_strength = weighted_sum / len(self.dendrites)
        return self.activation_gate.receive_signal(signal_strength)
    
    async def reset_dendrites(self):
        """
        Resets the updated status of all dendrites.
        
        Biological analogy: Resetting of neural integrator.
        Justification: After firing, integrator neurons reset their accumulated
        potential to be ready for the next integration cycle.
        """
        for dendrite in self.dendrites:
            dendrite.updated = False
            
        # Apply Hebbian-like learning to adjust weights
        for i, dendrite in enumerate(self.dendrites):
            # If this dendrite contributed to activation, strengthen its connection
            if dendrite.updated:
                self.weights[i].increase(0.01)
            else:
                # Otherwise weaken it slightly
                self.weights[i].decrease(0.005)