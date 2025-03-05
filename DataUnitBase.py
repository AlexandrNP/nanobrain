import time
from typing import Any
from ConfigManager import ConfigManager
from DirectoryTracer import DirectoryTracer
from activation import ActivationGate

class DataUnitBase:
    """
    Base class for data storage units.
    
    Biological analogy: Memory storage in the brain.
    Justification: Like how different brain regions store different types of
    information with varying persistence, data units store different types of
    data with configurable decay and persistence.
    """
    def __init__(self, **kwargs):
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        self.data = None
        self.updated = False
        self.persistence_level = 0.5  # How long data persists (0.0-1.0)
        self.decay_rate = 0.1  # How quickly data decays
        self.activation_gate = ActivationGate()  # Controls access to data
    
    def get(self) -> Any:
        """
        Retrieves the stored data.
        
        Biological analogy: Neuron activation and memory retrieval.
        Justification: Like how neurons must reach activation threshold
        to transmit signals, data units must pass activation checks to
        retrieve data. This models the energy cost of memory access.
        """
        # Check if we can access the data (like neural activation threshold)
        if not self.activation_gate.receive_signal(1.0):
            return None
            
        # Accessing data strengthens its persistence (like memory reconsolidation)
        self.persistence_level = min(1.0, self.persistence_level + 0.05)
        
        return self.data
    
    def set(self, data: Any) -> None:
        """
        Stores new data.
        
        Biological analogy: Synaptic plasticity during memory formation.
        Justification: Like how synapses strengthen during memory formation,
        data units update their persistence level when storing new data.
        """
        # Check if we can modify the data (like neural plasticity requirements)
        if not self.activation_gate.receive_signal(0.8):
            return
            
        self.data = data
        self.updated = True
        
        # New data has high persistence (like strong initial memory formation)
        self.persistence_level = 0.8
    
    def decay(self):
        """
        Reduces persistence of stored data over time.
        
        Biological analogy: Memory decay.
        Justification: Like how memories fade over time without
        reinforcement, stored data gradually loses persistence.
        """
        # Only decay if we have data
        if self.data is not None:
            # Reduce persistence based on decay rate
            self.persistence_level = max(0.0, self.persistence_level - self.decay_rate)
            
            # If persistence drops to zero, data is forgotten
            if self.persistence_level <= 0.0:
                self.data = None
                self.updated = False
    
    def consolidate(self):
        """
        Strengthens persistence of important data.
        
        Biological analogy: Memory consolidation.
        Justification: Like how important memories are consolidated during
        sleep and rest periods, important data is preserved through
        increased persistence.
        """
        if self.updated:
            self.persistence_level = min(1.0, self.persistence_level + 0.2)
            self.updated = False
    
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration for this class."""
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)