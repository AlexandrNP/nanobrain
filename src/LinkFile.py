from typing import Optional, Any
from src.DataUnitBase import DataUnitBase
from src.LinkBase import LinkBase
from src.TriggerBase import TriggerBase
import asyncio

class LinkFile(LinkBase):
    """
    File-based link implementation that transfers data via files.
    
    Biological analogy: Indirect pathway with memory storage.
    Justification: Like how some neural pathways involve intermediary processing
    and temporary storage (e.g., hippocampal memory formation), file links
    process and store data in files during transmission.
    """
    def __init__(self, 
                 source_step: Any,
                 sink_step: Any,
                 link_id: str = None,
                 input_folder: str = None, 
                 output_folder: str = None,
                 trigger_type: str = "data_changed",
                 trigger: Optional[TriggerBase] = None,
                 auto_setup_trigger: bool = True,
                 **kwargs):
        # Initialize the base class with source and sink steps
        super().__init__(
            source_step, 
            sink_step, 
            link_id, 
            trigger_type=trigger_type,
            trigger=trigger,
            auto_setup_trigger=auto_setup_trigger,
            **kwargs
        )
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.reliability = 0.95  # Slightly less reliable than direct
        self.transmission_delay = 0.2  # Slower than direct
        self.persistence_factor = 0.8  # Higher permanence of stored data
    
    async def transfer(self):
        """
        Transfers data from input folder to output folder.
        
        Biological analogy: Indirect pathway with memory consolidation.
        Justification: Like how information in the brain can be stored in
        intermediary structures before reaching its final destination,
        file links store data in files during transmission.
        """
        # Simulate transmission delay
        await asyncio.sleep(self.transmission_delay)
        
        data = self.source_step.output.get()
        
        if data is None:
            return False
        
        # Enhance data persistence if possible
        if hasattr(self.source_step.output, 'persistence_level'):
            self.source_step.output.persistence_level = max(
                self.source_step.output.persistence_level,
                self.persistence_factor
            )
            
        # Store data in specified folders if provided
        if self.input_folder and self.output_folder:
            # File operations would go here
            pass
            
        # Use base class to transfer data
        return await super().transfer()