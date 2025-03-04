from mixins import ConfigurableMixin
from enums import DataUnitBase
from regulations import ConnectionStrength
import random


class LinkBase(ConfigurableMixin):
    """
    Base link class that connects two steps in the workflow.
    
    Biological analogy: Axonal projection between brain regions.
    Justification: Like how axonal projections connect different brain regions
    and enable information flow between them, links connect different workflow
    steps and enable data flow between them.
    """
    def __init__(self, input_data: DataUnitBase, output_data: DataUnitBase, **kwargs):
        super().__init__(**kwargs)
        self.input = input_data
        self.output = output_data
        self.connection_strength = ConnectionStrength()  # Connection strength
        self.adaptability = 0.5  # How easily the connection changes (0.0-1.0)
        self.reliability = 0.95  # Probability of successful transmission
    
    async def transfer(self):
        """
        Transfers data from input to output.
        
        Biological analogy: Axonal transmission and synaptic transmission.
        Justification: Like how neural signals propagate down axons and across
        synapses with a certain reliability, data transfers along links with
        a certain reliability.
        """
        # Check reliability - sometimes transmissions fail
        if random.random() > self.reliability:
            # Transmission failed
            return False
            
        data = self.input.get()
        
        # Apply connection weight to modify data
        if isinstance(data, (int, float)):
            # For numerical data, multiply by weight
            weighted_data = data * self.connection_strength.strength
        else:
            # For non-numerical data, just pass through
            weighted_data = data
            
        # Send to output
        self.output.set(weighted_data)
        
        # Apply Hebbian-like learning - strengthen connection if used
        if data:  # If meaningful data was transmitted
            self.connection_strength.increase(0.01 * self.adaptability)
        
        return True