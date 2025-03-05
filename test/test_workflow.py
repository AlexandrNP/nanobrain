import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Workflow import Workflow
from Step import Step
from ExecutorBase import ExecutorBase
from LinkBase import LinkBase
from DataUnitBase import DataUnitBase
from enums import ComponentState
from regulations import SystemModulator


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create mock steps
        self.step1 = MagicMock(spec=Step)
        self.step1.execute = AsyncMock()
        self.step1.execute.return_value = "step1_result"
        self.step1.state = ComponentState.INACTIVE
        self.step1.output_sink = None
        self.step1.input_sources = []
        self.step1.adaptability = 0.5
        
        self.step2 = MagicMock(spec=Step)
        self.step2.execute = AsyncMock()
        self.step2.execute.return_value = "step2_result"
        self.step2.state = ComponentState.INACTIVE
        self.step2.output_sink = None
        self.step2.input_sources = []
        self.step2.adaptability = 0.5
        
        # Create workflow instance with patched SystemModulator
        with patch('Workflow.SystemModulator') as mock_system_modulator:
            self.mock_modulator_instance = MagicMock(spec=SystemModulator)
            self.mock_modulator_instance.get_modulator.return_value = 0.5
            mock_system_modulator.return_value = self.mock_modulator_instance
            
            self.workflow = Workflow(
                executor=self.executor,
                steps=[self.step1, self.step2]
            )
    
    def test_initialization(self):
        """Test that Workflow initializes correctly with proper inheritance."""
        # Verify attributes are set correctly
        self.assertEqual(self.workflow.steps, [self.step1, self.step2])
        self.assertIsNotNone(self.workflow.deadlock_detector)
        self.assertEqual(self.workflow.state, ComponentState.INACTIVE)
        self.assertFalse(self.workflow.running)
        
        # Verify inheritance from Step and PackageBase
        self.assertTrue(hasattr(self.workflow, 'directory_tracer'))
        self.assertTrue(hasattr(self.workflow, 'config_manager'))
        self.assertTrue(hasattr(self.workflow, 'circuit_breaker'))
        
    def test_organize_hierarchy(self):
        """Test the organize_hierarchy method."""
        # Set up dependencies between steps
        self.step1.input_sources = []
        self.step2.input_sources = [MagicMock()]
        self.step2.input_sources[0].output.get.return_value = "step1_result"
        
        # Organize hierarchy
        self.workflow.organize_hierarchy()
        
        # Verify step_order is created
        self.assertTrue(hasattr(self.workflow, 'step_order'))
        self.assertIsInstance(self.workflow.step_order, dict)
    
    async def async_test_execute(self):
        """Test the execute method."""
        # Configure mocks
        self.mock_modulator_instance.get_modulator.return_value = 0.5
        
        # Execute the workflow
        result = await self.workflow.execute()
        
        # Verify each step's execute method was called
        self.step1.execute.assert_called_once()
        self.step2.execute.assert_called_once()
        
        # Verify the result is a dictionary with step results
        self.assertIsInstance(result, dict)
        self.assertEqual(result['step_0'], "step1_result")
        self.assertEqual(result['step_1'], "step2_result")
        self.assertEqual(self.workflow.result, result)
    
    def test_execute(self):
        """Run the async test."""
        asyncio.run(self.async_test_execute())
    
    async def async_test_execute_with_exception(self):
        """Test execute method when an exception occurs."""
        # Configure step1 to raise an exception
        self.step1.execute.side_effect = Exception("Test exception")
        
        # Configure deadlock detector to allow execution
        self.workflow.deadlock_detector.request_resource = MagicMock(return_value=True)
        
        # Execute the workflow - we expect the exception to be caught and handled
        result = await self.workflow.execute()
        
        # Verify step1 was executed and failed
        self.step1.execute.assert_called_once()
        self.assertIsNone(result['step_0'])
        
        # Verify step2 was still executed
        self.step2.execute.assert_called_once()
        self.assertEqual(result['step_1'], "step2_result")
    
    def test_execute_with_exception(self):
        """Run the async test for exception handling."""
        asyncio.run(self.async_test_execute_with_exception())
    
    async def async_test_apply_modulator_effects(self):
        """Test the apply_modulator_effects method."""
        # Configure system modulator
        self.mock_modulator_instance.get_modulator.return_value = 0.7  # High value
        
        # Apply modulator effects
        self.workflow.apply_modulator_effects()
        
        # Verify modulator effects were applied
        # With performance = 0.7, network_efficiency should be 0.3 + (0.7 * 0.7) = 0.79
        self.assertAlmostEqual(self.workflow.network_efficiency, 0.79, places=2)
    
    def test_apply_modulator_effects(self):
        """Run the test for apply_modulator_effects."""
        asyncio.run(self.async_test_apply_modulator_effects())
    
    async def async_test_process(self):
        """Test the process method."""
        # Configure workflow
        self.workflow.result = None
        self.workflow.execute = AsyncMock()
        self.workflow.execute.return_value = {"step_0": "step1_result", "step_1": "step2_result"}
        
        # Call process with test input
        result = await self.workflow.process(["test_input"])
        
        # Verify execute was called
        self.workflow.execute.assert_called_once()
        self.assertEqual(result, {"step_0": "step1_result", "step_1": "step2_result"})
    
    def test_process(self):
        """Run the async test for process."""
        asyncio.run(self.async_test_process())
    
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def async_test_decay_inhibition(self, mock_sleep):
        """Test the decay_inhibition method."""
        # Configure mocks
        mock_sleep.return_value = None
        
        # Set up active inhibition
        self.workflow.active_inhibition = {"step1": 0.8}
        
        # Mock the sleep function to avoid waiting
        async def fast_sleep(seconds):
            return None
        mock_sleep.side_effect = fast_sleep
        
        # Call decay_inhibition but don't wait for it to complete
        # Just verify it starts the decay process
        decay_task = asyncio.create_task(self.workflow.decay_inhibition("step1"))
        
        # Allow the task to run for a short time
        await asyncio.sleep(0.1)
        
        # Cancel the task (we don't need to wait for it to complete)
        decay_task.cancel()
        
        # Verify sleep was called at least once
        mock_sleep.assert_called()
    
    def test_decay_inhibition(self):
        """Run the async test for decay_inhibition."""
        asyncio.run(self.async_test_decay_inhibition())


if __name__ == '__main__':
    unittest.main() 