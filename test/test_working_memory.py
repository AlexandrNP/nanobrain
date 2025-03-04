import unittest
import time
from WorkingMemory import WorkingMemory

class TestWorkingMemory(unittest.TestCase):
    def setUp(self):
        self.memory = WorkingMemory(capacity=3)  # Small capacity for testing
        
    def test_store_and_retrieve(self):
        """Test basic storage and retrieval functionality."""
        # Store and retrieve a simple value
        self.assertTrue(self.memory.store("key1", "value1"))
        self.assertEqual(self.memory.retrieve("key1"), "value1")
        
        # Store and retrieve different types of values
        self.assertTrue(self.memory.store("key2", 42))
        self.assertEqual(self.memory.retrieve("key2"), 42)
        
        self.assertTrue(self.memory.store("key3", {"test": "dict"}))
        self.assertEqual(self.memory.retrieve("key3"), {"test": "dict"})
        
    def test_capacity_limit(self):
        """Test that memory respects capacity limits."""
        # Fill memory to capacity
        self.memory.store("key1", "value1")
        self.memory.store("key2", "value2")
        self.memory.store("key3", "value3")
        
        # Try to store beyond capacity
        self.memory.store("key4", "value4")
        
        # Verify oldest item was removed (key1)
        self.assertIsNone(self.memory.retrieve("key1"))
        self.assertIsNotNone(self.memory.retrieve("key4"))
        
    def test_lru_behavior(self):
        """Test Least Recently Used (LRU) replacement policy."""
        # Fill memory
        self.memory.store("key1", "value1")
        self.memory.store("key2", "value2")
        self.memory.store("key3", "value3")
        
        # Access key1 to make it most recently used
        self.memory.retrieve("key1")
        
        # Add new item - should remove key2 (least recently used)
        self.memory.store("key4", "value4")
        
        # Verify key2 was removed but key1 remains
        self.assertIsNone(self.memory.retrieve("key2"))
        self.assertIsNotNone(self.memory.retrieve("key1"))
        
    def test_update_existing(self):
        """Test updating existing keys."""
        self.memory.store("key1", "value1")
        self.memory.store("key1", "updated_value")
        
        self.assertEqual(self.memory.retrieve("key1"), "updated_value")
        
if __name__ == '__main__':
    unittest.main() 