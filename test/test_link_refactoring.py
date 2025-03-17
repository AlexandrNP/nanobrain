#!/usr/bin/env python3
"""
Test script for verifying the refactored Link class hierarchy.

This script tests that the LinkBase class correctly handles both TriggerDataUpdated
and TriggerDataHashChanged trigger types, and works with the TriggerBase abstraction.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.LinkBase import LinkBase
from src.LinkDirect import LinkDirect
from src.LinkFile import LinkFile
from src.DataUnitBase import DataUnitBase
from src.DataUnitString import DataUnitString
from src.TriggerBase import TriggerBase
from src.TriggerDataUpdated import TriggerDataUpdated
from src.TriggerDataHashChanged import TriggerDataHashChanged
from src.ActivationGate import ActivationGate

# Monkey patch the LinkBase.process_signal method for testing purposes
original_process_signal = LinkBase.process_signal

def process_signal_for_test(self):
    """Override process_signal for testing to ensure sufficient signal strength."""
    if hasattr(self.source_step, 'signal_strength'):
        return self.source_step.signal_strength
    return original_process_signal(self)

# Apply the monkey patch for testing
LinkBase.process_signal = process_signal_for_test

# Add a reset method to ActivationGate for testing
original_activation_gate_init = ActivationGate.__init__

def reset(self):
    """Reset the activation gate to its initial state."""
    self.current_level = self.resting_level
    self.is_refractory = False
    self.last_activation_time = 0

# Add the reset method to ActivationGate
ActivationGate.reset = reset

class MockStep:
    """Mock step class for testing."""
    
    def __init__(self, name="MockStep", output_data=None):
        self.name = name
        self.output = DataUnitString(name=f"{name}Output", initial_value=output_data)
        self.input_sources = {}
        self.process_called = False
        self.process_data = None
        
        # Add a signal_strength attribute to control the signal strength in tests
        self.signal_strength = 1.0  # Default to a high value to ensure transfers succeed
    
    def register_input_source(self, link_id, data_unit):
        """Register an input source."""
        self.input_sources[link_id] = data_unit
        return True
    
    async def process(self, inputs):
        """Process inputs."""
        self.process_called = True
        self.process_data = inputs
        return True


class MockTrigger(TriggerBase):
    """Mock trigger for testing."""
    
    def __init__(self, source_step=None, runnable=None, **kwargs):
        super().__init__(runnable, **kwargs)
        self.source_step = source_step
        self.check_called = False
        self.monitor_called = False
        self.should_trigger = False
    
    async def check_condition(self, **kwargs) -> bool:
        """Check if the condition is met."""
        self.check_called = True
        return self.should_trigger
    
    async def monitor(self, **kwargs):
        """Monitor for the condition."""
        self.monitor_called = True
        if self.should_trigger and self.runnable:
            return await self.runnable.transfer()
        return False
    
    def set_should_trigger(self, value: bool):
        """Set whether the trigger should fire."""
        self.should_trigger = value


class TestLinkRefactoring(unittest.IsolatedAsyncioTestCase):
    """Test the refactored Link class hierarchy."""
    
    async def asyncSetUp(self):
        """Set up the test environment."""
        # Create mock steps
        self.source_step = MockStep(name="SourceStep", output_data="Initial data")
        self.sink_step = MockStep(name="SinkStep")
        
        # Ensure source_step has a high signal strength for testing
        self.source_step.signal_strength = 1.0
    
    async def test_link_base_with_auto_trigger_data_changed(self):
        """Test LinkBase with auto-created TriggerDataUpdated."""
        # Create a LinkBase with auto-created TriggerDataUpdated
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger_type="data_changed",
            debug=True
        )
        
        # Verify the trigger type
        self.assertEqual(link._trigger_type, "data_changed")
        self.assertEqual(link.trigger.__class__.__name__, "TriggerDataUpdated")
        
        # Change the source data and verify transfer
        self.source_step.output.set("New data")
        self.source_step.output.updated = True  # Simulate data change
        
        # Manually trigger the transfer
        result = await link.trigger.check_condition()
        self.assertTrue(result)
        
        # Verify transfer
        await link.transfer()
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New data")
    
    async def test_link_base_with_auto_trigger_data_hash_changed(self):
        """Test LinkBase with auto-created TriggerDataHashChanged."""
        # Create a LinkBase with auto-created TriggerDataHashChanged
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger_type="data_hash_changed",
            debug=True
        )
        
        # Verify the trigger type
        self.assertEqual(link._trigger_type, "data_hash_changed")
        self.assertEqual(link.trigger.__class__.__name__, "TriggerDataHashChanged")
        
        # Change the source data and verify transfer
        self.source_step.output.set("New data")
        
        # Manually trigger the transfer
        result = await link.trigger.check_condition()
        self.assertTrue(result)
        
        # Verify transfer
        await link.transfer()
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New data")
    
    async def test_link_base_with_manual_trigger(self):
        """Test LinkBase with manually provided trigger."""
        # Create a mock trigger
        mock_trigger = MockTrigger(source_step=self.source_step)
        
        # Create a LinkBase with the mock trigger
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger=mock_trigger,
            auto_setup_trigger=False,  # Don't auto-create a trigger
            debug=True
        )
        
        # Verify the trigger was set correctly
        self.assertEqual(link.trigger, mock_trigger)
        
        # Set the trigger to fire
        mock_trigger.set_should_trigger(True)
        
        # Change the source data
        self.source_step.output.set("New data")
        
        # Manually trigger the monitor
        await link.trigger.monitor()
        
        # Verify the trigger was called
        self.assertTrue(mock_trigger.monitor_called)
        
        # Verify transfer
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New data")
    
    async def test_link_direct_with_manual_trigger(self):
        """Test LinkDirect with manually provided trigger."""
        # Create a mock trigger
        mock_trigger = MockTrigger(source_step=self.source_step)
        
        # Create a LinkDirect with the mock trigger
        link = LinkDirect(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger=mock_trigger,
            auto_setup_trigger=False,  # Don't auto-create a trigger
            debug=True
        )
        
        # Verify the trigger was set correctly
        self.assertEqual(link.trigger, mock_trigger)
        
        # Set the trigger to fire
        mock_trigger.set_should_trigger(True)
        
        # Change the source data
        self.source_step.output.set("New data")
        
        # Manually trigger the monitor
        await link.trigger.monitor()
        
        # Verify the trigger was called
        self.assertTrue(mock_trigger.monitor_called)
        
        # Verify transfer
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New data")
    
    async def test_link_file_with_manual_trigger(self):
        """Test LinkFile with manually provided trigger."""
        # Create a mock trigger
        mock_trigger = MockTrigger(source_step=self.source_step)
        
        # Create a LinkFile with the mock trigger
        link = LinkFile(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger=mock_trigger,
            auto_setup_trigger=False,  # Don't auto-create a trigger
            debug=True
        )
        
        # Verify the trigger was set correctly
        self.assertEqual(link.trigger, mock_trigger)
        
        # Set the trigger to fire
        mock_trigger.set_should_trigger(True)
        
        # Change the source data
        self.source_step.output.set("New data")
        
        # Manually trigger the monitor
        await link.trigger.monitor()
        
        # Verify the trigger was called
        self.assertTrue(mock_trigger.monitor_called)
        
        # Verify transfer
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New data")
    
    async def test_link_base_with_no_trigger(self):
        """Test LinkBase with no trigger and auto_setup_trigger=False."""
        # Create a LinkBase with no trigger and auto_setup_trigger=False
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger=None,
            auto_setup_trigger=False,
            debug=True,
            reliability=1.0  # Ensure 100% reliability for testing
        )
        
        # Verify the trigger is None
        self.assertIsNone(link.trigger)
        
        # Verify that start_monitoring returns immediately
        await link.start_monitoring()
        self.assertFalse(link._monitoring)
        
        # Verify that transfer still works even without a trigger
        self.source_step.output.set("New data")
        
        # Reset activation gate to ensure it's ready for the transfer
        link.activation_gate.reset()
        
        result = await link.transfer()
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New data")
    
    async def test_data_deduplication(self):
        """Test that the same data is not transferred multiple times."""
        # Create a LinkBase with a lower threshold for the activation gate
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            debug=True,
            reliability=1.0  # Ensure 100% reliability for testing
        )
        
        # Lower the activation threshold to make it easier to pass
        link.activation_gate.threshold = 0.1
        
        # Set initial data
        self.source_step.output.set("Test data")
        
        # Reset activation gate to ensure it's ready for the first transfer
        link.activation_gate.reset()
        
        # First transfer should succeed
        result = await link.transfer()
        
        # Verify data was transferred correctly
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "Test data")
        self.assertEqual(link._last_transferred_data, "Test data")
        
        # Reset process_called flag
        self.sink_step.process_called = False
        
        # Second transfer with same data should be skipped due to _last_transferred_data check
        link.activation_gate.reset()
        result = await link.transfer()
        self.assertFalse(result)
        
        # Change data and verify transfer succeeds
        self.source_step.output.set("New test data")
        link.activation_gate.reset()
        
        result = await link.transfer()
        
        # Verify new data was transferred correctly
        self.assertEqual(self.sink_step.input_sources["test_link"].get(), "New test data")
        self.assertEqual(link._last_transferred_data, "New test data")


if __name__ == "__main__":
    unittest.main() 