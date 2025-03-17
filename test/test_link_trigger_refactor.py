#!/usr/bin/env python3
"""
Test script for verifying the refactored Link hierarchy with different trigger types.
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock

# Set testing mode to bypass activation gate checks and other restrictions
os.environ['NANOBRAIN_TESTING'] = '1'

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.LinkBase import LinkBase
from src.LinkDirect import LinkDirect
from src.DataUnitBase import DataUnitBase
from src.DataUnitString import DataUnitString
from src.ExecutorFunc import ExecutorFunc
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
def reset(self):
    """Reset the activation gate to its initial state."""
    self.current_level = self.resting_level
    self.is_refractory = False
    self.last_activation_time = 0

# Add the reset method to ActivationGate
ActivationGate.reset = reset


class MockStep:
    """Mock step class for testing links."""
    
    def __init__(self, name="MockStep"):
        self.name = name
        self.output = DataUnitString(name=f"{name}Output")
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
        self.process_data = inputs[0] if inputs else None
        return self.process_data


class TestLinkRefactor(unittest.IsolatedAsyncioTestCase):
    """Test the refactored Link hierarchy."""
    
    async def asyncSetUp(self):
        """Set up the test environment."""
        # Ensure NANOBRAIN_TESTING is set
        self.original_testing = os.environ.get('NANOBRAIN_TESTING')
        os.environ['NANOBRAIN_TESTING'] = '1'
        
        self.source_step = MockStep(name="SourceStep")
        self.sink_step = MockStep(name="SinkStep")
        self.executor = ExecutorFunc()
        
        # Ensure source_step has a high signal strength for testing
        self.source_step.signal_strength = 1.0
    
    async def asyncTearDown(self):
        """Tear down the test environment."""
        # Restore original NANOBRAIN_TESTING value
        if self.original_testing:
            os.environ['NANOBRAIN_TESTING'] = self.original_testing
        else:
            del os.environ['NANOBRAIN_TESTING']
    
    async def test_linkbase_data_changed_trigger(self):
        """Test LinkBase with TriggerDataUpdated."""
        # Create a LinkBase with data_changed trigger
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link",
            trigger_type="data_changed",
            debug=True,
            reliability=1.0,  # Ensure 100% reliability for testing
            auto_setup_trigger=False  # Don't start monitoring automatically
        )
        
        # Lower the activation threshold to make it easier to pass
        link.activation_gate.threshold = 0.1
        
        # Set data in the source step
        test_data = "Test data for LinkBase"
        self.source_step.output.set(test_data)
        
        # Reset activation gate to ensure it's ready for the transfer
        link.activation_gate.reset()
        
        # Monkey patch the process method to directly update process_called instead of using a task
        original_process = self.sink_step.process
        
        async def process_sync(inputs):
            return await original_process(inputs)
            
        # Replace the process method with our synchronous version
        self.sink_step.process = process_sync
        
        # Monkey patch the transfer method to await the process call
        original_transfer = link.transfer
        
        async def transfer_and_wait_for_process():
            result = await original_transfer()
            # Small delay to ensure any tasks have time to complete
            await asyncio.sleep(0.1)
            return result
            
        link.transfer = transfer_and_wait_for_process
        
        # Manually trigger the transfer
        result = await link.transfer()
        
        # Verify the transfer succeeded
        self.assertTrue(result)
        self.assertTrue(link.has_data_transferred())
        self.assertEqual(link._last_transferred_data, test_data)
        
        # Verify the sink step was triggered to process
        self.assertTrue(self.sink_step.process_called)
        self.assertEqual(self.sink_step.process_data, test_data)
    
    async def test_linkbase_data_hash_changed_trigger(self):
        """Test LinkBase with TriggerDataHashChanged."""
        # Create a LinkBase with hash_changed trigger
        link = LinkBase(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_link_hash",
            trigger_type="hash_changed",
            debug=True,
            reliability=1.0,  # Ensure 100% reliability for testing
            auto_setup_trigger=False  # Don't start monitoring automatically
        )
        
        # Lower the activation threshold to make it easier to pass
        link.activation_gate.threshold = 0.1
        
        # Set data in the source step
        test_data = "Test data for LinkBase with hash_changed trigger"
        self.source_step.output.set(test_data)
        
        # Reset activation gate to ensure it's ready for the transfer
        link.activation_gate.reset()
        
        # Monkey patch the process method to directly update process_called instead of using a task
        original_process = self.sink_step.process
        
        async def process_sync(inputs):
            return await original_process(inputs)
            
        # Replace the process method with our synchronous version
        self.sink_step.process = process_sync
        
        # Monkey patch the transfer method to await the process call
        original_transfer = link.transfer
        
        async def transfer_and_wait_for_process():
            result = await original_transfer()
            # Small delay to ensure any tasks have time to complete
            await asyncio.sleep(0.1)
            return result
            
        link.transfer = transfer_and_wait_for_process
        
        # Manually trigger the transfer
        result = await link.transfer()
        
        # Verify the transfer succeeded
        self.assertTrue(result)
        self.assertTrue(link.has_data_transferred())
        self.assertEqual(link._last_transferred_data, test_data)
        
        # Verify the sink step was triggered to process
        self.assertTrue(self.sink_step.process_called)
        self.assertEqual(self.sink_step.process_data, test_data)
    
    async def test_linkdirect_with_data_changed_trigger(self):
        """Test LinkDirect with TriggerDataUpdated."""
        # Create a LinkDirect with data_changed trigger
        link = LinkDirect(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_direct_link",
            trigger_type="data_changed",
            debug=True,
            reliability=1.0,  # Ensure 100% reliability for testing
            auto_setup_trigger=False  # Don't start monitoring automatically
        )
        
        # Lower the activation threshold to make it easier to pass
        link.activation_gate.threshold = 0.1
        
        # Set data in the source step
        test_data = "Test data for LinkDirect"
        self.source_step.output.set(test_data)
        
        # Reset activation gate to ensure it's ready for the transfer
        link.activation_gate.reset()
        
        # Monkey patch the process method to directly update process_called instead of using a task
        original_process = self.sink_step.process
        
        async def process_sync(inputs):
            return await original_process(inputs)
            
        # Replace the process method with our synchronous version
        self.sink_step.process = process_sync
        
        # Monkey patch the transfer method to await the process call
        original_transfer = link.transfer
        
        async def transfer_and_wait_for_process():
            result = await original_transfer()
            # Small delay to ensure any tasks have time to complete
            await asyncio.sleep(0.1)
            return result
            
        link.transfer = transfer_and_wait_for_process
        
        # Manually trigger the transfer
        result = await link.transfer()
        
        # Verify the transfer succeeded
        self.assertTrue(result)
        self.assertTrue(link.has_data_transferred())
        self.assertEqual(link._last_transferred_data, test_data)
        
        # Verify the sink step was triggered to process
        self.assertTrue(self.sink_step.process_called)
        self.assertEqual(self.sink_step.process_data, test_data)
    
    async def test_linkdirect_with_hash_changed_trigger(self):
        """Test LinkDirect with TriggerDataHashChanged."""
        # Create a LinkDirect with hash_changed trigger
        link = LinkDirect(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="test_direct_link_hash",
            trigger_type="hash_changed",
            debug=True,
            reliability=1.0,  # Ensure 100% reliability for testing
            auto_setup_trigger=False  # Don't start monitoring automatically
        )
        
        # Lower the activation threshold to make it easier to pass
        link.activation_gate.threshold = 0.1
        
        # Set data in the source step
        test_data = "Test data for LinkDirect with hash_changed trigger"
        self.source_step.output.set(test_data)
        
        # Reset activation gate to ensure it's ready for the transfer
        link.activation_gate.reset()
        
        # Monkey patch the process method to directly update process_called instead of using a task
        original_process = self.sink_step.process
        
        async def process_sync(inputs):
            return await original_process(inputs)
            
        # Replace the process method with our synchronous version
        self.sink_step.process = process_sync
        
        # Monkey patch the transfer method to await the process call
        original_transfer = link.transfer
        
        async def transfer_and_wait_for_process():
            result = await original_transfer()
            # Small delay to ensure any tasks have time to complete
            await asyncio.sleep(0.1)
            return result
            
        link.transfer = transfer_and_wait_for_process
        
        # Manually trigger the transfer
        result = await link.transfer()
        
        # Verify the transfer succeeded
        self.assertTrue(result)
        self.assertTrue(link.has_data_transferred())
        self.assertEqual(link._last_transferred_data, test_data)
        
        # Verify the sink step was triggered to process
        self.assertTrue(self.sink_step.process_called)
        self.assertEqual(self.sink_step.process_data, test_data)
    
    async def test_duplicate_transfer_prevention(self):
        """Test that duplicate transfers are prevented."""
        link = LinkDirect(
            source_step=self.source_step,
            sink_step=self.sink_step,
            link_id="duplicate_test",
            debug=True,
            reliability=1.0,  # Ensure 100% reliability for testing
            auto_setup_trigger=False
        )
        
        # Lower the activation threshold to make it easier to pass
        link.activation_gate.threshold = 0.1
        
        # Set initial data
        test_data = "Test data for duplicate prevention"
        self.source_step.output.set(test_data)
        
        # Reset activation gate to ensure it's ready for the first transfer
        link.activation_gate.reset()
        
        # Monkey patch the process method to directly update process_called instead of using a task
        original_process = self.sink_step.process
        
        async def process_sync(inputs):
            return await original_process(inputs)
            
        # Replace the process method with our synchronous version
        self.sink_step.process = process_sync
        
        # Monkey patch the transfer method to await the process call
        original_transfer = link.transfer
        
        async def transfer_and_wait_for_process():
            result = await original_transfer()
            # Small delay to ensure any tasks have time to complete
            await asyncio.sleep(0.1)
            return result
            
        link.transfer = transfer_and_wait_for_process
        
        # First transfer should succeed
        result1 = await link.transfer()
        self.assertTrue(result1)
        
        # Reset the process_called flag
        self.sink_step.process_called = False
        
        # Reset activation gate for the second transfer
        link.activation_gate.reset()
        
        # Second transfer with same data should not trigger another transfer
        result2 = await link.transfer()
        self.assertFalse(result2)  # Should return False for no transfer
        self.assertFalse(self.sink_step.process_called)  # Process should not be called
        
        # Set new data
        new_test_data = "New test data"
        self.source_step.output.set(new_test_data)
        
        # Reset activation gate for the third transfer
        link.activation_gate.reset()
        
        # Transfer with new data should succeed
        result3 = await link.transfer()
        self.assertTrue(result3)
        self.assertTrue(self.sink_step.process_called)
        self.assertEqual(self.sink_step.process_data, new_test_data)
    
    async def test_monitoring_activation(self):
        """Test that monitoring can be activated and deactivated."""
        # Import the trigger classes
        from src.TriggerDataUpdated import TriggerDataUpdated
        from src.TriggerDataHashChanged import TriggerDataHashChanged
        
        # Create a source step with output
        source_step = MockStep(name="SourceStep")
        source_step.output = DataUnitString(name="SourceOutput")
        source_step.output.set("Initial data")
        
        # Create a target step
        target_step = MockStep(name="TargetStep")
        
        # Create links
        link1 = LinkDirect(
            source_step=source_step, 
            sink_step=target_step, 
            link_id="test_link1",
            debug=True,
            auto_setup_trigger=False  # Don't start monitoring automatically
        )
        
        link2 = LinkDirect(
            source_step=source_step, 
            sink_step=target_step, 
            link_id="test_link2",
            debug=True,
            auto_setup_trigger=False  # Don't start monitoring automatically
        )
        
        # Create and set triggers manually
        trigger1 = TriggerDataUpdated(source_step=source_step, runnable=link1, debug=True)
        link1.trigger = trigger1
        
        trigger2 = TriggerDataHashChanged(source_step=source_step, runnable=link2, debug=True)
        link2.trigger = trigger2
        
        # Verify the trigger types
        self.assertIsInstance(link1.trigger, TriggerDataUpdated)
        self.assertIsInstance(link2.trigger, TriggerDataHashChanged)
        
        # Test that monitoring can be started and stopped
        link1.trigger.start_monitoring()
        self.assertTrue(link1.trigger._monitoring)
        
        link1.trigger.stop_monitoring()
        self.assertFalse(link1.trigger._monitoring)
        
        link2.trigger.start_monitoring()
        self.assertTrue(link2.trigger._monitoring)
        
        link2.trigger.stop_monitoring()
        self.assertFalse(link2.trigger._monitoring)


if __name__ == "__main__":
    unittest.main() 