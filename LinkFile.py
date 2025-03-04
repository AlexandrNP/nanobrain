from LinkBase import LinkBase
from enums import DataUnitBase
import asyncio


class LinkFile(LinkBase):
    """
    File-based link implementation that transfers data via files.
    
    Biological analogy: Indirect pathway with memory storage.
    Justification: Like how some neural pathways involve intermediary processing
    and temporary storage (e.g., hippocampal memory formation), file links
    process and store data in an intermediary format.
    """
    def __init__(self, input_data: DataUnitBase, output_data: DataUnitBase, 
                 input_folder: str = None, output_folder: str = None, **kwargs):
        super().__init__(input_data, output_data, **kwargs)
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
        
        data = self.input.get()
        
        # Enhance data persistence (make it more permanent)
        if isinstance(self.output, DataUnitBase):
            self.output.persistence_level = max(
                self.output.persistence_level,
                self.persistence_factor
            )
            
        # Store data in specified folders if provided
        if self.input_folder and self.output_folder:
            # File operations would go here
            pass
            
        # Transfer to output
        self.output.set(data)
        
        return True