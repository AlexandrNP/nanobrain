import unittest
import time
from regulations import ConnectionStrength

class TestConnectionStrength(unittest.TestCase):
    def setUp(self):
        self.connection = ConnectionStrength(initial_strength=0.5)
        
    def test_initial_values(self):
        """Test initial connection strength values."""
        self.assertEqual(self.connection.strength, 0.5)
        self.assertEqual(self.connection.min_strength, 0.0)
        self.assertEqual(self.connection.max_strength, 1.0)
        self.assertEqual(len(self.connection.history), 1)
        
    def test_increase_strength(self):
        """Test increasing connection strength."""
        initial = self.connection.strength
        self.connection.increase(0.1)
        self.assertGreater(self.connection.strength, initial)
        self.assertLessEqual(self.connection.strength, 1.0)
        
    def test_decrease_strength(self):
        """Test decreasing connection strength."""
        initial = self.connection.strength
        self.connection.decrease(0.1)
        self.assertLess(self.connection.strength, initial)
        self.assertGreaterEqual(self.connection.strength, 0.0)
        
    def test_bounds(self):
        """Test that strength stays within bounds."""
        # Test upper bound
        self.connection.increase(1.0)
        self.assertEqual(self.connection.strength, 1.0)
        self.connection.increase(0.1)  # Should not increase further
        self.assertEqual(self.connection.strength, 1.0)
        
        # Test lower bound
        self.connection.decrease(2.0)
        self.assertEqual(self.connection.strength, 0.0)
        self.connection.decrease(0.1)  # Should not decrease further
        self.assertEqual(self.connection.strength, 0.0)
        
    def test_history_recording(self):
        """Test that strength changes are recorded in history."""
        initial_history_len = len(self.connection.history)
        self.connection.increase(0.1)
        self.assertEqual(len(self.connection.history), initial_history_len + 1)
        
        self.connection.decrease(0.1)
        self.assertEqual(len(self.connection.history), initial_history_len + 2)
        
    def test_hebbian_adaptation(self):
        """Test Hebbian learning-like adaptation."""
        initial = self.connection.strength
        
        # Correlated activity should strengthen connection
        self.connection.adapt(source_activity=1.0, target_activity=1.0)
        self.assertGreater(self.connection.strength, initial)
        
        # Anti-correlated activity should weaken connection
        current = self.connection.strength
        self.connection.adapt(source_activity=1.0, target_activity=-1.0)
        self.assertLess(self.connection.strength, current)
        
if __name__ == '__main__':
    unittest.main() 