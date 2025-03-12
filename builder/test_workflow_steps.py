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
from src.TriggerDataChanged import TriggerDataChanged
from builder.WorkflowSteps import CreateStep, CreateWorkflow
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder


class TestCreateStep(unittest.IsolatedAsyncioTestCase):
    """Test cases for the CreateStep class."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
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
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('builder.WorkflowSteps.AgentWorkflowBuilder')
    @patch('builder.WorkflowSteps.DataStorageCommandLine')
    @patch('builder.WorkflowSteps.LinkDirect')
    async def test_create_step_basic(self, mock_link, mock_cli, mock_builder):
        """Test basic step creation functionality."""
        # Set up mocks
        mock_builder_instance = MagicMock()
        mock_builder_instance.get_generated_code.return_value = "test code"
        mock_builder_instance.get_generated_config.return_value = "test config"
        mock_builder_instance.get_generated_tests.return_value = "test tests"
        mock_builder.return_value = mock_builder_instance
        
        mock_cli_instance = MagicMock()
        mock_cli_instance.start_monitoring = AsyncMock()
        mock_cli.return_value = mock_cli_instance
        
        mock_link_instance = MagicMock()
        mock_link_instance.start_monitoring = AsyncMock()
        mock_link.return_value = mock_link_instance
        
        # Execute create step
        result = await CreateStep.execute(self.builder, "test_step")
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["step_class_name"], "StepTestStep")
        
        # Verify directory creation
        step_dir = self.test_dir / "StepTestStep"
        self.assertTrue(step_dir.exists())
        self.assertTrue((step_dir / "config").exists())
        
        # Verify interactions
        mock_builder.assert_called_once()
        mock_cli.assert_called_once()
        mock_link.assert_called_once()
        mock_cli_instance.start_monitoring.assert_called_once()
        mock_link_instance.start_monitoring.assert_called_once()
    
    @patch('builder.WorkflowSteps.AgentWorkflowBuilder')
    @patch('builder.WorkflowSteps.DataStorageCommandLine')
    async def test_create_step_no_workflow(self, mock_cli, mock_builder):
        """Test step creation with no active workflow."""
        self.builder.get_current_workflow.return_value = None
        
        result = await CreateStep.execute(self.builder, "test_step")
        
        self.assertFalse(result["success"])
        self.assertIn("No active workflow", result["error"])
        
        mock_builder.assert_not_called()
        mock_cli.assert_not_called()
    
    @patch('builder.WorkflowSteps.AgentWorkflowBuilder')
    @patch('builder.WorkflowSteps.DataStorageCommandLine')
    async def test_create_step_existing_directory(self, mock_cli, mock_builder):
        """Test step creation when directory already exists."""
        # Create the step directory beforehand
        os.makedirs(self.test_dir / "StepTestStep", exist_ok=True)
        
        result = await CreateStep.execute(self.builder, "test_step")
        
        self.assertFalse(result["success"])
        self.assertIn("already exists", result["error"])
        
        mock_builder.assert_not_called()
        mock_cli.assert_not_called()
    
    @patch('builder.WorkflowSteps.AgentWorkflowBuilder')
    @patch('builder.WorkflowSteps.DataStorageCommandLine')
    @patch('builder.WorkflowSteps.LinkDirect')
    async def test_create_step_with_description(self, mock_link, mock_cli, mock_builder):
        """Test step creation with description."""
        # Set up mocks
        mock_builder_instance = MagicMock()
        mock_builder_instance.get_generated_code.return_value = "test code"
        mock_builder_instance.get_generated_config.return_value = "test config"
        mock_builder_instance.get_generated_tests.return_value = "test tests"
        mock_builder.return_value = mock_builder_instance
        
        mock_cli_instance = MagicMock()
        mock_cli_instance.start_monitoring = AsyncMock()
        mock_cli.return_value = mock_cli_instance
        
        mock_link_instance = MagicMock()
        mock_link_instance.start_monitoring = AsyncMock()
        mock_link.return_value = mock_link_instance
        
        # Execute create step with description
        result = await CreateStep.execute(
            self.builder,
            "test_step",
            description="A test step for testing"
        )
        
        # Verify result
        self.assertTrue(result["success"])
        self.assertEqual(result["step_class_name"], "StepTestStep")
        
        # Verify directory creation
        step_dir = self.test_dir / "StepTestStep"
        self.assertTrue(step_dir.exists())
        self.assertTrue((step_dir / "config").exists())
        
        # Verify interactions with description
        mock_builder.assert_called_once()
        mock_cli.assert_called_once()
        mock_link.assert_called_once()
        mock_cli_instance.start_monitoring.assert_called_once()
        mock_link_instance.start_monitoring.assert_called_once()


if __name__ == '__main__':
    unittest.main() 