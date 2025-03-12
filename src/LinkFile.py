from typing import Optional
from enums import DataUnitBase
from LinkBase import LinkBase
from TriggerBase import TriggerBase
import asyncio

class LinkFile:
    """
    File-based link implementation that transfers data via files.
    
    Biological analogy: Indirect pathway with memory storage.
    Justification: Like how some neural pathways involve intermediary processing
    and temporary storage (e.g., hippocampal memory formation), file links
    process and store data in files during transmission.
    """
    def __init__(self, 
                 input_data: DataUnitBase, 
                 output_data: DataUnitBase, 
                 trigger: Optional[TriggerBase] = None,
                 input_folder: str = None, 
                 output_folder: str = None, 
                 **kwargs):
        self.base_link = LinkBase(input_data, output_data, trigger, **kwargs)
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.base_link.reliability = 0.95  # Slightly less reliable than direct
        self.base_link.transmission_delay = 0.2  # Slower than direct
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
        await asyncio.sleep(self.base_link.transmission_delay)
        
        data = self.base_link.input.get()
        
        if data is None:
            return False
        
        # Enhance data persistence (make it more permanent)
        if isinstance(self.base_link.output, DataUnitBase):
            self.base_link.output.persistence_level = max(
                self.base_link.output.persistence_level,
                self.persistence_factor
            )
            
        # Store data in specified folders if provided
        if self.input_folder and self.output_folder:
            # File operations would go here
            pass
            
        # Transfer to output
        self.base_link.output.set(data)
        
        return True
    
    async def start_monitoring(self):
        """
        Start monitoring the trigger condition.
        
        Biological analogy: Delayed synaptic vigilance.
        Justification: Like how some synapses with neuromodulators monitor for
        specific conditions over time, file links monitor for conditions to 
        trigger file-based transfer.
        """
        return await self.base_link.start_monitoring()
        
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
        
    @property
    def trigger(self):
        return self.base_link.trigger
        
    @trigger.setter
    def trigger(self, value):
        self.base_link.trigger = value
        if self.base_link.trigger:
            self.base_link.trigger.runnable = self.base_link