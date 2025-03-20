"""
Unit tests for workflow steps.

This module contains tests for the workflow step implementations.
"""

import unittest
import asyncio
import os
import shutil
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from src.ExecutorBase import ExecutorBase
from src.DataStorageCommandLine import DataStorageCommandLine
from src.LinkDirect import LinkDirect
from src.TriggerDataUpdated import TriggerDataUpdated
from builder.WorkflowSteps import CreateStep, CreateWorkflow
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from builder.test_create_step_integration import setup_llm_mocks

# Create a non-interactive version of start_monitoring
async def non_interactive_start_monitoring(self):
    """A non-interactive version of start_monitoring that automatically issues a 'finish' command."""
    if self._monitoring:
        return None  # Already monitoring
    
    self._monitoring = True
    
    # Show welcome message
    if hasattr(self, "welcome_message"):
        print(self.welcome_message)
    
    # Automatically send 'finish' command
    self._monitoring = False
    response = "Step creation completed."
    if hasattr(self, "_add_to_history"):
        self._add_to_history("finish", response)
    if hasattr(self, "_force_output_change"):
        self._force_output_change(response)
    
    print("\n✅ Step creation session ended (non-interactive mode). Files will be saved.")
    return None


class TestCreateStep(unittest.IsolatedAsyncioTestCase):
    """Test cases for the CreateStep class."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        # Set up LLM mocks
        self.llm_patcher, self.prompt_patcher, self.mock_llm = setup_llm_mocks()
        
        self.test_dir = Path("test_workflow")
        self.executor = MagicMock(spec=ExecutorBase)
        self.builder = MagicMock()
        self.builder.executor = self.executor
        self.builder.get_current_workflow.return_value = str(self.test_dir)
        
        # Create test workflow directory
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.test_dir / "src", exist_ok=True)
        os.makedirs(self.test_dir / "config", exist_ok=True)
        os.makedirs(self.test_dir / "test", exist_ok=True)
        
        # Patch DataStorageCommandLine.start_monitoring with our non-interactive version
        self.start_monitoring_patcher = patch('src.DataStorageCommandLine.DataStorageCommandLine.start_monitoring', 
                                             side_effect=non_interactive_start_monitoring)
        self.mock_start_monitoring = self.start_monitoring_patcher.start()
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        # Stop the patchers
        self.llm_patcher.stop()
        self.prompt_patcher.stop()
        self.start_monitoring_patcher.stop()
        
        # Clean up test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('builder.WorkflowSteps.AgentWorkflowBuilder', new_callable=AsyncMock)
    @patch('builder.WorkflowSteps.DataStorageCommandLine')
    @patch('builder.WorkflowSteps.LinkDirect')
    @patch('builder.WorkflowSteps.TriggerDataUpdated')
    @patch('builder.WorkflowSteps.ConfigManager')
    @patch('builder.WorkflowSteps.os.path.exists')
    @patch('builder.WorkflowSteps.os.makedirs')
    async def test_create_step_basic(self, mock_makedirs, mock_exists, mock_config_manager, mock_trigger, mock_link, mock_cli, mock_builder):
        """Test basic step creation functionality."""
        # Set up mocks
        # Mock os.path.exists to return False for step directory check and True for workflow directory
        mock_exists.side_effect = lambda path: 'StepTestStep' not in path
        mock_makedirs.return_value = None  # Mock os.makedirs

        # Create a template for the agent builder generate_step_template method
        template_code = """#!/usr/bin/env python3
\"\"\"
Test Step - Test Step

This step implements test functionality for NanoBrain workflows.
\"\"\"

from src.Step import Step


class StepTestStep(Step):
    \"\"\"
    Test Step
    
    This is a test step.
    \"\"\"
    
    def __init__(self, **kwargs):
        \"\"\"Initialize the step.\"\"\"
        super().__init__(**kwargs)
        
    def process(self, data_dict):
        \"\"\"Process input data.\"\"\"
        # Process the input data
        result = {}
        
        # Add your custom processing logic here
        # ...
        
        return result
"""
        # Create a coroutine function for generate_step_template
        async def mock_generate_template(*args, **kwargs):
            return template_code
            
        # Create a properly configured mock for the agent builder
        mock_builder_instance = MagicMock()
        mock_builder_instance.generate_step_template = mock_generate_template
        mock_builder_instance.get_generated_code = MagicMock(return_value="test code")
        mock_builder_instance.get_generated_config = MagicMock(return_value="test config")
        mock_builder_instance.get_generated_tests = MagicMock(return_value="test tests")
        mock_builder_instance.input_storage = None
        mock_builder_instance.input_sources = {}
        mock_builder_instance._monitoring = False
        mock_builder_instance._debug_mode = False
        mock_builder_instance.name = "MockAgentBuilder"
        mock_builder.return_value = mock_builder_instance
        
        mock_cli_instance = MagicMock()
        mock_cli_instance.start_monitoring = AsyncMock(side_effect=non_interactive_start_monitoring)
        mock_cli_instance._monitoring = False
        mock_cli_instance._debug_mode = False
        mock_cli_instance.name = "MockCommandLine"
        mock_cli_instance.prompt = "Test prompt"
        mock_cli_instance.welcome_message = "Test welcome message"
        mock_cli_instance.output = MagicMock()
        mock_cli_instance.history = []
        mock_cli_instance.chat_history = []
        mock_cli.return_value = mock_cli_instance
        
        mock_link_instance = MagicMock()
        mock_link_instance.start_monitoring = AsyncMock()
        mock_link_instance._debug_mode = False  # Add _debug_mode attribute
        mock_link_instance.debug_mode = False   # Add debug_mode attribute
        mock_link_instance._monitoring = False  # Add _monitoring attribute
        mock_link.return_value = mock_link_instance
        
        # Mock TriggerDataUpdated to avoid init error
        mock_trigger_instance = MagicMock()
        mock_trigger_instance.start_monitoring = MagicMock()  # Not async since we updated it in TriggerDataUpdated
        mock_trigger.return_value = mock_trigger_instance
        
        # Mock ConfigManager
        mock_config_manager_instance = MagicMock()
        mock_config_manager_instance.create_instance = MagicMock()
        mock_config_manager_instance.create_instance.side_effect = [
            mock_builder_instance,  # AgentWorkflowBuilder
            mock_cli_instance       # DataStorageCommandLine
        ]
        mock_config_manager.return_value = mock_config_manager_instance
        
        # Add file_writer_tool mock to builder
        file_writer_tool = MagicMock()
        file_writer_tool.create_file = AsyncMock(return_value=True)
        file_writer_tool.__class__ = MagicMock()
        file_writer_tool.__class__.__name__ = 'StepFileWriter'
        self.builder.agent = MagicMock()
        self.builder.agent.tools = [file_writer_tool]
        
        # Use the static execute method directly
        try:
            result = await CreateStep.execute(
                self.builder, 
                "test_step",
                command_line=mock_cli_instance,
                agent_builder=mock_builder_instance
            )
        except Exception as e:
            print(f"Error in test_create_step_basic: {e}")
            result = {"success": False, "error": str(e)}
        
        # Print error message if present
        if not result.get("success", False) and "error" in result:
            print(f"Error in CreateStep.execute: {result['error']}")
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["step_class_name"], "StepTestStep")
        
        # Verify config manager was called
        # mock_config_manager.assert_called_once()  # Removed since we're bypassing ConfigManager
        
        # Print the actual calls for debugging
        print("Actual create_instance calls:")
        for call in mock_config_manager_instance.create_instance.call_args_list:
            print(f"  {call}")
        
        # Verify file creation calls
        from unittest.mock import ANY
        file_writer_tool.create_file.assert_any_call(
            os.path.join(str(self.test_dir), "src", "StepTestStep", "StepTestStep.py"), 
            ANY
        )
    
    @patch('builder.WorkflowSteps.ConfigManager')
    async def test_create_step_no_workflow(self, mock_config_manager):
        """Test step creation with no active workflow."""
        self.builder.get_current_workflow.return_value = None
        
        # Use the static execute method directly
        result = await CreateStep.execute(self.builder, "test_step")
        
        # Verify failure due to no active workflow
        self.assertFalse(result["success"])
        self.assertIn("No active workflow", result["error"])
        
        # ConfigManager should not be called if workflow path validation fails
        mock_config_manager.assert_not_called()
    
    @patch('builder.WorkflowSteps.os.path.exists')
    @patch('builder.WorkflowSteps.ConfigManager')
    async def test_create_step_existing_directory(self, mock_config_manager, mock_exists):
        """Test step creation when directory already exists."""
        # Mock that the directory exists
        mock_exists.return_value = True
        
        # Use the static execute method directly
        result = await CreateStep.execute(self.builder, "test_step")
        
        # Verify failure due to directory already existing
        self.assertFalse(result["success"])
        self.assertIn("Step directory already exists", result["error"])
        
        # ConfigManager should not be called if directory already exists
        mock_config_manager.assert_not_called()
    
    @patch('builder.WorkflowSteps.AgentWorkflowBuilder', new_callable=AsyncMock)
    @patch('builder.WorkflowSteps.DataStorageCommandLine')
    @patch('builder.WorkflowSteps.LinkDirect')
    @patch('builder.WorkflowSteps.TriggerDataUpdated')
    @patch('builder.WorkflowSteps.ConfigManager')
    @patch('builder.WorkflowSteps.os.path.exists')
    @patch('builder.WorkflowSteps.os.makedirs')
    async def test_create_step_with_description(self, mock_makedirs, mock_exists, mock_config_manager, mock_trigger, mock_link, mock_cli, mock_builder):
        """Test step creation with description."""
        # Set up mocks
        # Mock os.path.exists to return False for step directory check and True for workflow directory
        mock_exists.side_effect = lambda path: 'src/StepTestStep' not in path
        mock_makedirs.return_value = None  # Mock os.makedirs

        # Create a template for the agent builder generate_step_template method with description
        template_code = """#!/usr/bin/env python3
\"\"\"
Test Step - A test step for testing

This step implements testing functionality for NanoBrain workflows.
\"\"\"

from src.Step import Step


class StepTestStep(Step):
    \"\"\"
    A test step for testing
    
    This is a test step with a description.
    \"\"\"
    
    def __init__(self, **kwargs):
        \"\"\"Initialize the step.\"\"\"
        super().__init__(**kwargs)
        
    def process(self, data_dict):
        \"\"\"Process input data.\"\"\"
        # Process the input data
        result = {}
        
        # Add your custom processing logic here
        # ...
        
        return result
"""
        # Create a coroutine function for generate_step_template
        async def mock_generate_template(*args, **kwargs):
            return template_code
            
        # Create a properly configured mock for the agent builder
        mock_builder_instance = MagicMock()
        mock_builder_instance.generate_step_template = mock_generate_template
        mock_builder_instance.get_generated_code = MagicMock(return_value="test code with description")
        mock_builder_instance.get_generated_config = MagicMock(return_value="test config with description")
        mock_builder_instance.get_generated_tests = MagicMock(return_value="test tests with description")
        mock_builder_instance.input_storage = None
        mock_builder_instance.input_sources = {}
        mock_builder_instance._monitoring = False
        mock_builder_instance._debug_mode = False
        mock_builder_instance.name = "MockAgentBuilder"
        mock_builder.return_value = mock_builder_instance
        
        mock_cli_instance = MagicMock()
        mock_cli_instance.start_monitoring = AsyncMock(side_effect=non_interactive_start_monitoring)
        mock_cli_instance._monitoring = False
        mock_cli_instance._debug_mode = False
        mock_cli_instance.name = "MockCommandLine"
        mock_cli_instance.prompt = "Test prompt"
        mock_cli_instance.welcome_message = "Test welcome message"
        mock_cli_instance.output = MagicMock()
        mock_cli_instance.history = []
        mock_cli_instance.chat_history = []
        mock_cli.return_value = mock_cli_instance
        
        mock_link_instance = MagicMock()
        mock_link_instance.start_monitoring = AsyncMock()
        mock_link_instance._debug_mode = False  # Add _debug_mode attribute
        mock_link_instance.debug_mode = False   # Add debug_mode attribute
        mock_link_instance._monitoring = False  # Add _monitoring attribute
        mock_link.return_value = mock_link_instance
        
        # Mock TriggerDataUpdated to avoid init error
        mock_trigger_instance = MagicMock()
        mock_trigger_instance.start_monitoring = MagicMock()  # Not async since we updated it in TriggerDataUpdated
        mock_trigger.return_value = mock_trigger_instance
        
        # Mock ConfigManager
        mock_config_manager_instance = MagicMock()
        mock_config_manager_instance.create_instance = MagicMock()
        mock_config_manager_instance.create_instance.side_effect = [
            mock_builder_instance,  # AgentWorkflowBuilder
            mock_cli_instance       # DataStorageCommandLine
        ]
        mock_config_manager.return_value = mock_config_manager_instance
        
        # Add file_writer_tool mock to builder
        file_writer_tool = MagicMock()
        file_writer_tool.create_file = AsyncMock(return_value=True)
        file_writer_tool.__class__ = MagicMock()
        file_writer_tool.__class__.__name__ = 'StepFileWriter'
        self.builder.agent = MagicMock()
        self.builder.agent.tools = [file_writer_tool]
        
        # Execute create step with description
        try:
            result = await CreateStep.execute(
                self.builder,
                "test_step",
                description="A test step for testing",
                command_line=mock_cli_instance,
                agent_builder=mock_builder_instance
            )
        except Exception as e:
            print(f"Error in test_create_step_with_description: {e}")
            result = {"success": False, "error": str(e)}
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["step_class_name"], "StepTestStep")
        
        # Verify config manager was called
        # mock_config_manager.assert_called_once()  # Removed since we're bypassing ConfigManager
        
        # Print the actual calls for debugging
        print("Actual create_instance calls:")
        for call in mock_config_manager_instance.create_instance.call_args_list:
            print(f"  {call}")
        
        # Verify file creation calls
        from unittest.mock import ANY
        file_writer_tool.create_file.assert_any_call(
            os.path.join(str(self.test_dir), "src", "StepTestStep", "StepTestStep.py"), 
            ANY
        )
    
    @patch('builder.WorkflowSteps.os.path.exists')
    @patch('builder.WorkflowSteps.os.makedirs')
    @patch('builder.WorkflowSteps.ConfigManager')
    async def test_create_step_simplified(self, mock_config_manager, mock_makedirs, mock_exists):
        """Simplified test that directly patches the necessary functions."""
        # Mock os.path.exists to return False for step directory check and True for workflow directory
        mock_exists.side_effect = lambda path: 'src/StepTestStep' not in path
        
        # Create AsyncMock for generate_step_template with a defined return value
        template_mock = AsyncMock(return_value="// Generated template code")
        
        # Create AsyncMock for create_file
        create_file_mock = AsyncMock(return_value=True)
        
        # Create AsyncMock for CLI start_monitoring
        cli_monitoring_mock = AsyncMock(return_value=None)
        
        # Mock ConfigManager to return configured instances
        mock_config_instance = MagicMock()
        
        # Create builder instance with all required methods mocked
        builder_instance = MagicMock()
        builder_instance.generate_step_template = template_mock
        builder_instance.get_generated_code = MagicMock(return_value="test code")
        builder_instance.get_generated_config = MagicMock(return_value="test config")
        builder_instance.get_generated_tests = MagicMock(return_value="test tests")
        builder_instance._debug_mode = False
        builder_instance.input_storage = None
        builder_instance.input_sources = {}
        builder_instance._monitoring = False
        builder_instance.name = "MockAgentBuilder"
        
        # Create command line instance with all required methods mocked
        cli_instance = MagicMock()
        cli_instance.start_monitoring = cli_monitoring_mock
        cli_instance._debug_mode = False
        cli_instance._monitoring = False
        cli_instance.name = "MockCommandLine"
        cli_instance.prompt = "Test prompt"
        cli_instance.welcome_message = "Test welcome message"
        cli_instance.output = MagicMock()
        cli_instance.history = []
        cli_instance.chat_history = []
        
        # Create code writer instance with all required methods mocked
        code_writer_instance = MagicMock()
        code_writer_instance._debug_mode = False
        code_writer_instance._monitoring = False
        code_writer_instance.input_sources = {}  # Required for LinkDirect
        code_writer_instance.name = "MockCodeWriter"
        code_writer_instance.process = AsyncMock(return_value="Code writer result")
        code_writer_instance.recent_response = "Generated solution code"
        code_writer_instance._extract_code = MagicMock(return_value="// Extracted solution code")
        code_writer_instance.generated_code = "// Generated solution code"
        
        # Configure ConfigManager.create_instance to return our mocks
        def create_instance_side_effect(*args, **kwargs):
            if args[0] == "AgentWorkflowBuilder":
                return builder_instance
            elif args[0] == "DataStorageCommandLine":
                return cli_instance
            elif args[0] == "AgentCodeWriter":
                return code_writer_instance
            else:
                return MagicMock()
        
        mock_config_instance.create_instance = MagicMock(side_effect=create_instance_side_effect)
        mock_config_manager.return_value = mock_config_instance
        
        # Create a mock for the file writer tool
        file_writer_mock = MagicMock()
        file_writer_mock.create_file = create_file_mock
        file_writer_mock.__class__ = MagicMock()
        file_writer_mock.__class__.__name__ = 'StepFileWriter'
        
        # Add tool to agent
        self.builder.agent = MagicMock()
        self.builder.agent.tools = [file_writer_mock]
        
        # Patch necessary methods
        with patch('builder.WorkflowSteps.LinkDirect') as mock_link, \
             patch('builder.WorkflowSteps.TriggerDataUpdated') as mock_trigger:
            
            # Mock the link and trigger instances
            mock_link.return_value = MagicMock(start_monitoring=AsyncMock())
            mock_trigger.return_value = MagicMock(start_monitoring=MagicMock())
            
            # Execute the test, passing command_line directly to avoid ConfigManager issues
            result = await CreateStep.execute(
                self.builder, 
                "test_step", 
                command_line=cli_instance,
                agent_builder=builder_instance
            )
            
            # Print error if test failed
            if not result.get("success", False):
                print(f"Creation failed: {result.get('error', 'Unknown error')}")
            
            # Verify success
            self.assertTrue(result["success"], f"Creation failed: {result.get('error', 'Unknown error')}")
            self.assertEqual(result["step_class_name"], "StepTestStep")
            
            # Verify create_file was called
            self.assertTrue(create_file_mock.called, "The create_file mock was not called")


if __name__ == '__main__':
    unittest.main() 