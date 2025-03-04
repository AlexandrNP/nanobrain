import time
from typing import Any
from ConfigManager import ConfigManager

class DataUnitBase:
    """
    Abstract base class for data storage and retrieval.
    
    Biological analogy: Memory engram in the brain.
    Justification: Like how memory engrams store information in the brain with
    varying degrees of persistence and accessibility, DataUnit provides storage
    with configurable characteristics.
    """
    def __init__(self, **kwargs):
        self.config_manager = ConfigManager(**kwargs)
        self.updated = False
        self.data = None
        self.last_access_time = 0
        self.decay_rate = 0.01  # Memory decay rate
        self.persistence_level = 0.0  # How permanent the data is (0.0-1.0)
    
    def get(self) -> Any:
        """
        Returns stored data.
        
        Biological analogy: Memory retrieval with rehearsal effect.
        Justification: Accessing stored memories strengthens them through
        the process of reconsolidation, making them more resistant to decay.
        """
        current_time = time.time()
        self.last_access_time = current_time
        
        # Accessing strengthens the memory (persistence)
        self.persistence_level = min(1.0, self.persistence_level + 0.05)
        
        return self.data
    
    def set(self, data: Any) -> None:
        """
        Sets data and updates status.
        
        Biological analogy: Memory encoding.
        Justification: New memories start less consolidated and require 
        repeated access or active consolidation to become permanent.
        """
        self.data = data
        self.updated = True
        self.last_access_time = time.time()
        
        # New data starts with low persistence
        if self.persistence_level < 0.2:
            self.persistence_level = 0.2
    
    def decay(self):
        """
        Models data decay over time.
        
        Biological analogy: Memory decay through lack of access.
        Justification: Memories that aren't accessed regularly fade over time,
        with stronger (more consolidated) memories decaying more slowly.
        """
        current_time = time.time()
        time_since_access = current_time - self.last_access_time
        
        # Calculate decay amount
        # Higher persistence = slower decay
        decay_modifier = 1.0 - self.persistence_level
        decay_amount = self.decay_rate * time_since_access * decay_modifier
        
        # Apply decay to numerical data
        if isinstance(self.data, (int, float)):
            decay_factor = max(0.0, 1.0 - decay_amount)
            self.data *= decay_factor
        
        # For complete loss of data
        if decay_amount > 0.9 and self.persistence_level < 0.1:
            self.data = None
    
    def consolidate(self):
        """
        Makes data more permanent.
        
        Biological analogy: Memory consolidation during sleep/rest.
        Justification: Like how important memories are consolidated during rest,
        important data should be actively protected from decay.
        """
        self.persistence_level = min(1.0, self.persistence_level + 0.1)
        
    def get_config(self, class_dir: str = None) -> dict:
        """Delegate to config manager."""
        return self.config_manager.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)