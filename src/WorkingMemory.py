import time
from typing import Dict, Any, Optional
from src.ConfigManager import ConfigManager
from src.DirectoryTracer import DirectoryTracer
from src.PackageBase import PackageBase
from src.ExecutorBase import ExecutorBase


class WorkingMemory(PackageBase):
    """
    Short-term memory storage with limited capacity.
    
    Biological analogy: Working memory in the prefrontal cortex.
    Justification: Like how working memory has limited capacity and
    follows recency-based retention, this class implements a capacity-limited
    storage with LRU (Least Recently Used) eviction policy.
    """
    def __init__(self, executor: ExecutorBase = None, **kwargs):
        super().__init__(executor, **kwargs)
        
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        
        # Allow direct override via kwargs, otherwise use config or default
        self.capacity = kwargs.get('capacity', 
                                  self.config_manager.get_config(self.__class__.__name__).get('capacity', 7))
        
        # Memory storage and access order tracking
        self._memory: Dict[str, Any] = {}
        self._access_order: list = []  # Track access order for LRU implementation
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store a value in working memory.
        
        Biological analogy: Encoding information into working memory.
        Justification: Like how the brain encodes new information into
        working memory, potentially displacing older items when capacity
        is reached.
        """
        # If key already exists, update its position in access order
        if key in self._memory:
            self._access_order.remove(key)
        
        # If at capacity and adding new item, remove least recently used
        elif len(self._memory) >= self.capacity and key not in self._memory:
            # Remove the least recently used item (first in access_order)
            lru_key = self._access_order[0]
            del self._memory[lru_key]
            self._access_order.remove(lru_key)
        
        # Add/update the item and mark it as most recently used
        self._memory[key] = value
        self._access_order.append(key)
        return True
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from working memory.
        
        Biological analogy: Retrieving information from working memory.
        Justification: Like how retrieving information from working memory
        makes it more likely to be retained (recency effect), this method
        updates the item's recency status.
        """
        if key in self._memory:
            # Update access order (mark as most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._memory[key]
        return None