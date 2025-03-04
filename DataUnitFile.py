from typing import Any
from DataUnitBase import DataUnitBase
from enums import WorkingMemory


class DataUnitFile(DataUnitBase):
    """
    Implementation of DataUnitBase that stores data in a file.
    
    Biological analogy: External memory storage (like writing).
    Justification: Like how humans externalize memories through writing,
    this class provides persistent storage outside the main memory system.
    """
    def __init__(self, file_path: str, **kwargs):
        super().__init__(**kwargs)
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
                self.data = file.read()
            
            # Store in buffer
            self.buffer.store(self.file_path, self.data)
            
            return super().get()
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
            
            super().set(data)
        except Exception as e:
            # Handle error
            print(f"Error writing to file: {e}")