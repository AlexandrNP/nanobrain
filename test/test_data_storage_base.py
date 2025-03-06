import unittest
import asyncio
import os
from unittest.mock import MagicMock, patch

# Set testing mode
os.environ['NANOBRAIN_TESTING'] = '1'

from src.DataStorageBase import DataStorageBase
from src.DataUnitBase import DataUnitBase
from src.TriggerBase import TriggerBase
from src.ExecutorBase import ExecutorBase
from src.enums import ComponentState

class TestDataStorageBase(unittest.TestCase):
    def setUp(self):
        # Create mock objects
        self.executor = MagicMock(spec=ExecutorBase)
        self.input_unit = MagicMock(spec=DataUnitBase)
        self.output_unit = MagicMock(spec=DataUnitBase)
        self.trigger = MagicMock(spec=TriggerBase)
        
        # Create DataStorageBase instance
        self.storage = DataStorageBase(
            executor=self.executor,
            input_unit=self.input_unit,
            output_unit=self.output_unit,
            trigger=self.trigger
        )
    
    def test_initialization(self):
        """Test that the DataStorageBase initializes correctly."""
        self.assertEqual(self.storage.input, self.input_unit)
        self.assertEqual(self.storage.output, self.output_unit)
        self.assertEqual(self.storage.trigger, self.trigger)
        self.assertEqual(self.storage.max_history_size, 10)
        self.assertEqual(self.storage.processing_history, [])
        self.assertIsNone(self.storage.last_query)
        self.assertIsNone(self.storage.last_response)
        
        # Check that the trigger is configured to activate this storage
        self.assertEqual(self.trigger.runnable, self.storage)
    
    def test_process_query(self):
        """Test that the _process_query method works correctly."""
        # The default implementation should just return the query
        result = asyncio.run(self.storage._process_query("test_query"))
        self.assertEqual(result, "test_query")
    
    def test_process(self):
        """Test that the process method works correctly."""
        # Process a query
        result = asyncio.run(self.storage.process(["test_query"]))
        
        # Check that the result is correct
        self.assertEqual(result, "test_query")
        
        # Check that the last query and response are set
        self.assertEqual(self.storage.last_query, "test_query")
        self.assertEqual(self.storage.last_response, "test_query")
        
        # Check that the output unit is updated
        self.output_unit.set.assert_called_once_with("test_query")
        
        # Check that the history is updated
        self.assertEqual(len(self.storage.processing_history), 1)
        self.assertEqual(self.storage.processing_history[0]['query'], "test_query")
        self.assertEqual(self.storage.processing_history[0]['response'], "test_query")
    
    def test_update_history(self):
        """Test that the _update_history method works correctly."""
        # Update history multiple times
        self.storage._update_history("query1", "response1")
        self.storage._update_history("query2", "response2")
        
        # Check that the history is updated
        self.assertEqual(len(self.storage.processing_history), 2)
        self.assertEqual(self.storage.processing_history[0]['query'], "query1")
        self.assertEqual(self.storage.processing_history[0]['response'], "response1")
        self.assertEqual(self.storage.processing_history[1]['query'], "query2")
        self.assertEqual(self.storage.processing_history[1]['response'], "response2")
        
        # Test history size limit
        self.storage.max_history_size = 2
        self.storage._update_history("query3", "response3")
        
        # Check that the oldest entry is removed
        self.assertEqual(len(self.storage.processing_history), 2)
        self.assertEqual(self.storage.processing_history[0]['query'], "query2")
        self.assertEqual(self.storage.processing_history[1]['query'], "query3")
    
    def test_start_monitoring(self):
        """Test that the start_monitoring method works correctly."""
        # Configure the trigger's monitor method to return a coroutine
        self.trigger.monitor = MagicMock(return_value=asyncio.sleep(0))
        
        # Start monitoring
        asyncio.run(self.storage.start_monitoring())
        
        # Check that the state is set to ACTIVE (for monitoring)
        self.assertEqual(self.storage._state, ComponentState.ACTIVE)
        
        # Check that the trigger's monitor method is called
        self.trigger.monitor.assert_called_once()
    
    def test_stop_monitoring(self):
        """Test that the stop_monitoring method works correctly."""
        # Configure the trigger's stop_monitoring method to return a coroutine
        self.trigger.stop_monitoring = MagicMock(return_value=asyncio.sleep(0))
        
        # Stop monitoring
        asyncio.run(self.storage.stop_monitoring())
        
        # Check that the state is set to INACTIVE
        self.assertEqual(self.storage._state, ComponentState.INACTIVE)
        
        # Check that the trigger's stop_monitoring method is called
        self.trigger.stop_monitoring.assert_called_once()
    
    def test_get_history(self):
        """Test that the get_history method works correctly."""
        # Add some history
        self.storage._update_history("query1", "response1")
        self.storage._update_history("query2", "response2")
        
        # Get the history
        history = self.storage.get_history()
        
        # Check that the history is correct
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['query'], "query1")
        self.assertEqual(history[1]['query'], "query2")
    
    def test_clear_history(self):
        """Test that the clear_history method works correctly."""
        # Add some history
        self.storage._update_history("query1", "response1")
        self.storage._update_history("query2", "response2")
        
        # Clear the history
        self.storage.clear_history()
        
        # Check that the history is cleared
        self.assertEqual(len(self.storage.processing_history), 0)
    
    def test_get_last_interaction(self):
        """Test that the get_last_interaction method works correctly."""
        # When there's no interaction
        interaction = self.storage.get_last_interaction()
        self.assertEqual(interaction, {})
        
        # After an interaction
        self.storage.last_query = "test_query"
        self.storage.last_response = "test_response"
        
        interaction = self.storage.get_last_interaction()
        self.assertEqual(interaction, {
            'query': "test_query",
            'response': "test_response"
        })


if __name__ == '__main__':
    unittest.main() 