import unittest
import asyncio
import os
import shutil
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from builder.WorkflowSteps import CreateStep
from src.DataStorageCommandLine import DataStorageCommandLine
from src.ConfigManager import ConfigManager
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.ExecutorBase import ExecutorBase
from src.DataUnitString import DataUnitString
from src.LinkDirect import LinkDirect
from src.TriggerDataUpdated import TriggerDataUpdated


class TestCreateStep(unittest.TestCase):
    """Test the CreateStep class."""
    
    def setUp(self):
        """Set up the test case."""
        self.executor = MagicMock(spec=ExecutorBase)
        self.builder = MagicMock()
        self.builder.executor = self.executor
        self.builder.get_current_workflow.return_value = "/tmp/test_workflow"
        
        # Set up the ConfigManager mock
        self.config_manager_mock = MagicMock(spec=ConfigManager)
        self.config_manager_patcher = patch('src.ConfigManager.ConfigManager', return_value=self.config_manager_mock)
        self.config_manager_creator = self.config_manager_patcher.start()
        
        # Mock the AgentWorkflowBuilder and DataStorageCommandLine
        self.agent_builder_mock = MagicMock(spec=AgentWorkflowBuilder)
        self.command_line_mock = MagicMock(spec=DataStorageCommandLine)
        
        # Configure the ConfigManager to return our mocks
        self.config_manager_mock.create_instance.side_effect = lambda class_name, **kwargs: \
            self.agent_builder_mock if class_name == "AgentWorkflowBuilder" else \
            self.command_line_mock if class_name == "DataStorageCommandLine" else None
        
        # Clean up test directory if it exists
        self.test_dir = "/tmp/test_workflow/src/StepTest"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after the test."""
        self.config_manager_patcher.stop()
        
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_name_attribute_exist(self):
        """Test that the name attribute exists in the Step class."""
        from src.Step import Step
        
        # Create a Step instance with a name
        step = Step(self.executor, name="TestStep")
        self.assertEqual(step.name, "TestStep", "Step should have the name attribute set to the provided value")
        
        # Create a Step instance without a name (should default to class name)
        step_without_name = Step(self.executor)
        self.assertEqual(step_without_name.name, "Step", "Step should have the name attribute default to the class name")
    
    def test_data_storage_command_line_name(self):
        """Test that DataStorageCommandLine has a name attribute."""
        # Create a DataStorageCommandLine instance with a name (passed via kwargs)
        # Note: The constructor takes 'name' as a positional argument, so we need to use kwargs
        cmd_line = DataStorageCommandLine(executor=self.executor, name="TestCommandLine")
        self.assertEqual(cmd_line.name, "TestCommandLine", "DataStorageCommandLine should have the name attribute set")
        
        # Create a DataStorageCommandLine instance with default name
        cmd_line_default = DataStorageCommandLine(executor=self.executor)
        self.assertEqual(cmd_line_default.name, "CommandLine", "DataStorageCommandLine should have the default name")
    
    def test_link_direct_uses_name(self):
        """Test that LinkDirect can access the name attribute of its endpoints."""
        # Create Step instances with names
        source_step = MagicMock()
        source_step.name = "SourceStep"
        
        sink_step = MagicMock()
        sink_step.name = "SinkStep"
        
        # Mock asyncio.create_task to prevent any async tasks from being created
        with patch('asyncio.create_task'), patch('src.TriggerDataUpdated.TriggerDataUpdated.start_monitoring'):
            # Create a LinkDirect instance with debug mode on, but don't auto setup trigger
            link = LinkDirect(source_step=source_step, sink_step=sink_step, debug=True, auto_setup_trigger=False)
            
            # The test passes if the LinkDirect initialization completes without raising an AttributeError
            # This verifies that it can access the name attributes of both steps
            self.assertTrue(hasattr(link, 'source_step'), "LinkDirect should have a source_step attribute")
            self.assertTrue(hasattr(link, 'sink_step'), "LinkDirect should have a sink_step attribute")


if __name__ == '__main__':
    unittest.main() 