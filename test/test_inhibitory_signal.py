import unittest
import time
from concurrency import InhibitorySignal

class TestInhibitorySignal(unittest.TestCase):
    def setUp(self):
        self.signal = InhibitorySignal(strength=0.8, duration=0.1)  # Short duration for testing
        
    def test_initial_state(self):
        """Test initial signal state."""
        self.assertEqual(self.signal.strength, 0.8)
        self.assertEqual(self.signal.duration, 0.1)
        self.assertIsNone(self.signal.start_time)
        self.assertFalse(self.signal.is_active())
        
    def test_activation(self):
        """Test signal activation."""
        self.signal.activate()
        self.assertIsNotNone(self.signal.start_time)
        self.assertTrue(self.signal.is_active())
        
    def test_strength_decay(self):
        """Test signal strength decay over time."""
        self.signal.activate()
        initial_strength = self.signal.get_strength()
        
        # Wait a short time
        time.sleep(0.05)  # Wait half the duration
        
        # Check that strength has decreased but signal is still active
        current_strength = self.signal.get_strength()
        self.assertLess(current_strength, initial_strength)
        self.assertGreater(current_strength, 0)
        self.assertTrue(self.signal.is_active())
        
    def test_signal_expiration(self):
        """Test that signal expires after duration."""
        self.signal.activate()
        
        # Wait for signal to expire
        time.sleep(0.15)  # Wait longer than duration
        
        self.assertFalse(self.signal.is_active())
        self.assertEqual(self.signal.get_strength(), 0.0)
        
    def test_inactive_strength(self):
        """Test strength of inactive signal."""
        # Signal should have zero strength when not activated
        self.assertEqual(self.signal.get_strength(), 0.0)
        
    def test_reactivation(self):
        """Test signal can be reactivated."""
        self.signal.activate()
        time.sleep(0.15)  # Wait for expiration
        
        # Reactivate
        self.signal.activate()
        self.assertTrue(self.signal.is_active())
        self.assertGreater(self.signal.get_strength(), 0.0)
        
if __name__ == '__main__':
    unittest.main() 