import time
from typing import Any


class WorkingMemory:
    """
    Provides temporary storage for processed information.
    
    Biological analogy: Working memory in the prefrontal cortex.
    Justification: Like how working memory holds information temporarily for
    ongoing cognitive operations, this class provides short-term storage for
    data being processed across workflow steps.
    """
    def __init__(self, capacity: int = 7):  # Miller's Law: 7±2 items
        self.items = {}
        self.capacity = capacity
        self.access_times = {}  # For LRU replacement
    
    def store(self, key: str, value: Any) -> bool:
        """
        Store an item in working memory.
        
        Biological analogy: Encoding information in working memory.
        Justification: Similar to how the brain must selectively encode information
        with limited capacity, requiring older items to be cleared out.
        """
        # If at capacity, remove least recently used item
        if len(self.items) >= self.capacity and key not in self.items:
            self._remove_lru()
            
        self.items[key] = value
        self.access_times[key] = time.time()
        return True
    
    def retrieve(self, key: str) -> Any:
        """
        Retrieve an item from working memory.
        
        Biological analogy: Retrieval from working memory with rehearsal effect.
        Justification: Like how retrieving items from working memory strengthens their
        retention by resetting their decay timers.
        """
        if key in self.items:
            self.access_times[key] = time.time()
            return self.items[key]
        return None
    
    def _remove_lru(self):
        """
        Remove least recently used item.
        
        Biological analogy: Displacement in limited-capacity memory.
        Justification: When working memory reaches capacity, the least recently
        accessed items are most likely to be forgotten.
        """
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.items[oldest_key]
        del self.access_times[oldest_key]