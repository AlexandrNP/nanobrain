from LinkBase import LinkBase
from enums import DataUnitBase
import asyncio


class LinkDirect(LinkBase):
    """
    Direct link implementation that transfers data directly.
    
    Biological analogy: Fast, direct neural pathway.
    Justification: Like how some neural pathways offer rapid, direct transmission
    (e.g., reflexes), direct links provide immediate data transfer between components.
    """
    def __init__(self, input_data: DataUnitBase, output_data: DataUnitBase, **kwargs):
        super().__init__(input_data, output_data, **kwargs)
        self.reliability = 0.98  # Higher reliability than average
        self.transmission_delay = 0.01  # Fast transmission
    
    async def transfer(self):
        """
        Transfers data directly from input to output.
        
        Biological analogy: Fast, direct synaptic transmission.
        Justification: Like how some synapses use fast ionotropic receptors
        for immediate signal transmission, direct links provide immediate
        data transfer with minimal processing.
        """
        # Simulate transmission delay
        await asyncio.sleep(self.transmission_delay)
        
        return await super().transfer()