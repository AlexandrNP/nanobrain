from typing import Optional, Any
from src.DataUnitBase import DataUnitBase
from src.LinkBase import LinkBase
from src.TriggerBase import TriggerBase
import asyncio

class LinkDirect(LinkBase):
    """
    Direct link implementation that transfers data directly.
    
    Biological analogy: Fast, direct neural pathway.
    Justification: Like how some neural pathways offer rapid, direct transmission
    (e.g., reflexes), direct links provide immediate data transfer between components.
    """
    def __init__(self, 
                 source_step: Any,
                 sink_step: Any,
                 link_id: str = None,
                 **kwargs):
        # Initialize the base class with source and sink steps
        super().__init__(source_step, sink_step, link_id, **kwargs)
        self.reliability = 0.98  # Higher reliability than average
        self.transmission_delay = 0.01  # Fast transmission
    
    async def transfer(self):
        """
        Transfers data directly from source to sink.
        
        Biological analogy: Fast, direct synaptic transmission.
        Justification: Like how some synapses use fast ionotropic receptors
        for immediate signal transmission, direct links provide immediate
        data transfer with minimal processing.
        """
        # Simulate transmission delay
        await asyncio.sleep(self.transmission_delay)
        
        # Use the base class transfer method
        return await super().transfer()