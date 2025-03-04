from enums import DataUnitBase
from LinkBase import LinkBase
import asyncio

class LinkDirect:
    """
    Direct link implementation that transfers data directly.
    
    Biological analogy: Fast, direct neural pathway.
    Justification: Like how some neural pathways offer rapid, direct transmission
    (e.g., reflexes), direct links provide immediate data transfer between components.
    """
    def __init__(self, input_data: DataUnitBase, output_data: DataUnitBase, **kwargs):
        self.base_link = LinkBase(input_data, output_data, **kwargs)
        self.base_link.reliability = 0.98  # Higher reliability than average
        self.base_link.transmission_delay = 0.01  # Fast transmission
    
    async def transfer(self):
        """
        Transfers data directly from input to output.
        
        Biological analogy: Fast, direct synaptic transmission.
        Justification: Like how some synapses use fast ionotropic receptors
        for immediate signal transmission, direct links provide immediate
        data transfer with minimal processing.
        """
        # Simulate transmission delay
        await asyncio.sleep(self.base_link.transmission_delay)
        
        return await self.base_link.transfer()
        
    # Delegate methods to base link
    def get_config(self, class_dir: str = None) -> dict:
        return self.base_link.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        return self.base_link.update_config(updates, adaptability_threshold)
        
    @property
    def input(self):
        return self.base_link.input
        
    @property
    def output(self):
        return self.base_link.output
        
    @property
    def connection_strength(self):
        return self.base_link.connection_strength
        
    @property
    def adaptability(self):
        return self.base_link.adaptability
        
    @adaptability.setter
    def adaptability(self, value):
        self.base_link.adaptability = value
        
    @property
    def reliability(self):
        return self.base_link.reliability
        
    @reliability.setter
    def reliability(self, value):
        self.base_link.reliability = value