from typing import Optional, Any
from src.DataUnitBase import DataUnitBase
import copy
import hashlib

class DataUnitString(DataUnitBase):
    """
    Specialized data unit for storing and processing string data.
    
    Biological analogy: Language processing area.
    Justification: Like how specialized brain regions process linguistic information,
    this class specializes in storing and processing string data.
    """
    
    def __init__(self, name="StringData", initial_value: Optional[str] = None, **kwargs):
        """
        Initialize the string data unit.
        
        Args:
            name: Name of this data unit
            initial_value: Optional initial string value
            **kwargs: Additional parameters passed to DataUnitBase
        """
        super().__init__(name=name, **kwargs)
        self._value = initial_value
        self._hash = self._compute_hash(initial_value) if initial_value is not None else None
        self._debug_mode = kwargs.get('debug', False)
        
    def _compute_hash(self, value: str) -> str:
        """
        Compute a hash of the string value for change detection.
        
        Args:
            value: The string value to hash
            
        Returns:
            Hash string representing the value
        """
        if value is None:
            return None
        return hashlib.md5(str(value).encode()).hexdigest()
        
    def get(self) -> Optional[str]:
        """
        Get the current string value.
        
        Biological analogy: Memory retrieval.
        Justification: Like how the brain retrieves stored memories,
        this method retrieves the stored string data.
        
        Returns:
            The current string value, or None if not set
        """
        return self._value
        
    def set(self, value: Any) -> bool:
        """
        Set the string value.
        
        Biological analogy: Memory encoding.
        Justification: Like how the brain encodes new memories,
        this method stores a new string value.
        
        Args:
            value: The value to set (will be converted to string)
            
        Returns:
            True if the value was changed, False otherwise
        """
        # Convert to string if not None
        string_value = str(value) if value is not None else None
        
        # Compute hash of the new value
        new_hash = self._compute_hash(string_value)
        
        # Check if the value has changed
        if new_hash != self._hash:
            # Store both the value and its hash
            self._value = string_value
            self._hash = new_hash
            
            if self._debug_mode:
                print(f"DataUnitString: Value changed to '{string_value}'")
                
            return True
        return False
        
    def clear(self) -> bool:
        """
        Clear the string value.
        
        Biological analogy: Memory clearance.
        Justification: Like how the brain can clear working memory,
        this method clears the stored string data.
        
        Returns:
            True if the value was cleared, False if already None
        """
        if self._value is not None:
            old_value = self._value
            self._value = None
            self._hash = None
            
            return True
        return False 