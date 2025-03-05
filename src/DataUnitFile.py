from typing import Any
from DataUnitBase import DataUnitBase
from WorkingMemory import WorkingMemory

class DataUnitFile:
    """
    Implementation of data storage that uses files.
    
    Biological analogy: External memory storage (like writing).
    Justification: Like how humans externalize memories through writing,
    this class provides persistent storage outside the main memory system.
    """
    def __init__(self, file_path: str, **kwargs):
        self.base_unit = DataUnitBase(**kwargs)
        self.file_path = file_path
        self.buffer = WorkingMemory(capacity=10)  # Cache for frequently accessed data
    
    def get(self) -> Any:
        """
        Reads and returns file content.
        
        Biological analogy: Reading externalized information.
        Justification: Like how humans need to read externalized information
        and bring it back into working memory to use it.
        """
        # Check if data is in buffer
        buffered_data = self.buffer.retrieve(self.file_path)
        if buffered_data is not None:
            return buffered_data
            
        # Read from file
        try:
            with open(self.file_path, 'r') as file:
                self.base_unit.data = file.read()
            
            # Store in buffer
            self.buffer.store(self.file_path, self.base_unit.data)
            
            return self.base_unit.get()
        except FileNotFoundError:
            return None
    
    def set(self, data: Any) -> None:
        """
        Writes data to file and updates status.
        
        Biological analogy: Writing to external storage.
        Justification: Externalizing information for more permanent record.
        """
        try:
            with open(self.file_path, 'w') as file:
                file.write(str(data))
            
            # Update buffer
            self.buffer.store(self.file_path, data)
            
            self.base_unit.set(data)
        except Exception as e:
            # Handle error
            print(f"Error writing to file: {e}")
            
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