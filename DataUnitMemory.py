from typing import Any
from DataUnitBase import DataUnitBase

class DataUnitMemory:
    """
    Implementation of data storage that uses RAM.
    
    Biological analogy: Short-term memory in neural circuits.
    Justification: Like how neural circuits maintain information through
    persistent activity patterns, this class maintains data in volatile
    memory with rapid access but limited duration.
    """
    def __init__(self, **kwargs):
        self.base_unit = DataUnitBase(**kwargs)
        self.base_unit.decay_rate = 0.05  # Faster decay in memory
        self.base_unit.persistence_level = 0.3  # Lower persistence by default
        
    def get(self) -> Any:
        """
        Returns data from memory.
        
        Biological analogy: Rapid memory retrieval.
        Justification: Like how information in short-term memory is
        quickly accessible but subject to rapid decay, data in RAM
        is quickly accessible but volatile.
        """
        return self.base_unit.get()
    
    def set(self, data: Any) -> None:
        """
        Stores data in memory.
        
        Biological analogy: Short-term memory encoding.
        Justification: Like how short-term memory quickly encodes
        new information but requires active maintenance or consolidation
        to persist, data in RAM requires active maintenance.
        """
        self.base_unit.set(data)
        
    # Delegate methods to base unit
    def decay(self):
        """Delegate to base unit."""
        self.base_unit.decay()
    
    def consolidate(self):
        """Delegate to base unit."""
        self.base_unit.consolidate()
        
    def get_config(self, class_dir: str = None) -> dict:
        """Delegate to base unit."""
        return self.base_unit.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to base unit."""
        return self.base_unit.update_config(updates, adaptability_threshold)
        
    @property
    def updated(self):
        return self.base_unit.updated
        
    @updated.setter
    def updated(self, value):
        self.base_unit.updated = value
        
    @property
    def persistence_level(self):
        return self.base_unit.persistence_level
        
    @persistence_level.setter
    def persistence_level(self, value):
        self.base_unit.persistence_level = value
        
    @property
    def decay_rate(self):
        return self.base_unit.decay_rate
        
    @decay_rate.setter
    def decay_rate(self, value):
        self.base_unit.decay_rate = value
        
    @property
    def activation_gate(self):
        return self.base_unit.activation_gate 