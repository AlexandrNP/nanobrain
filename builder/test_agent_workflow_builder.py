"""
Unit tests for AgentWorkflowBuilder.

This module contains tests for the AgentWorkflowBuilder class.
"""

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from src.ExecutorBase import ExecutorBase
from src.DataUnitBase import DataUnitBase
from src.Agent import Agent
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from langchain_core.prompts import PromptTemplate


class TestAgentWorkflowBuilder(unittest.IsolatedAsyncioTestCase):
    """Test cases for the AgentWorkflowBuilder class."""
    
    @patch('src.Agent.Agent._initialize_llm')
    @patch('src.Agent.Agent._load_prompt_template')
    def setUp(self, mock_load_prompt, mock_init_llm):
        """Set up test environment."""
        self.executor = MagicMock(spec=ExecutorBase)
        self.mock_llm = MagicMock()
        mock_init_llm.return_value = self.mock_llm
        
        # Mock prompt template
        self.mock_prompt = MagicMock(spec=PromptTemplate)
        mock_load_prompt.return_value = self.mock_prompt
        
        self.builder = AgentWorkflowBuilder(executor=self.executor)
    
    async def test_initialization(self):
        """Test that the builder initializes correctly."""
        self.assertIsInstance(self.builder, AgentWorkflowBuilder)
        self.assertIsInstance(self.builder, Agent)
        self.assertIsInstance(self.builder, DataUnitBase)
        self.assertEqual(self.builder.persistence_level, 0.8)
        self.assertEqual(self.builder.data, None)
        self.assertEqual(self.builder.generated_code, "")
        self.assertEqual(self.builder.generated_config, "")
        self.assertEqual(self.builder.generated_tests, "")
        self.assertEqual(self.builder.current_context, {})
    
    async def test_data_unit_interface(self):
        """Test DataUnitBase interface implementation."""
        test_data = "test input"
        
        # Test get/set
        self.assertEqual(self.builder.get(), None)
        self.assertTrue(self.builder.set(test_data))
        self.assertEqual(self.builder.get(), test_data)
    
    @patch.object(AgentWorkflowBuilder, '_process_input')
    async def test_process_input_called(self, mock_process):
        """Test that setting data triggers input processing."""
        test_data = "test input"
        mock_process.return_value = None
        
        self.builder.set(test_data)
        await asyncio.sleep(0)  # Allow async task to run
        
        mock_process.assert_called_once_with(test_data)
    
    async def test_handle_link_command(self):
        """Test handling of link command."""
        source = "source_step"
        target = "target_step"
        
        # Mock the _generate_link_code method
        self.builder._generate_link_code = AsyncMock(return_value="test link code")
        
        # Execute link command
        await self.builder._handle_link_command(source, target)
        
        # Verify context update
        self.assertIn("links", self.builder.current_context)
        self.assertEqual(len(self.builder.current_context["links"]), 1)
        self.assertEqual(
            self.builder.current_context["links"][0],
            {"source": source, "target": target}
        )
        
        # Verify code generation
        self.assertEqual(self.builder.generated_code, "\ntest link code")
    
    async def test_handle_code_input(self):
        """Test handling of code input."""
        test_input = "def test_function():\n    pass"
        
        # Mock the process method
        self.builder.process = AsyncMock(return_value="generated code")
        
        # Execute code input
        await self.builder._handle_code_input(test_input)
        
        # Verify context update
        self.assertIn("code_inputs", self.builder.current_context)
        self.assertEqual(len(self.builder.current_context["code_inputs"]), 1)
        self.assertEqual(self.builder.current_context["code_inputs"][0], test_input)
        
        # Verify code update
        self.assertEqual(self.builder.generated_code, "generated code")
    
    async def test_process_input_link_command(self):
        """Test processing of link command input."""
        # Mock the _handle_link_command method
        self.builder._handle_link_command = AsyncMock()
        
        # Test link command
        await self.builder._process_input("link source_step target_step")
        
        # Verify link command handling
        self.builder._handle_link_command.assert_called_once_with("source_step", "target_step")
    
    async def test_process_input_help_command(self):
        """Test processing of help command."""
        # Mock the _display_help method
        self.builder._display_help = MagicMock()
        
        # Test help command
        await self.builder._process_input("help")
        
        # Verify help command handling
        self.builder._display_help.assert_called_once()
    
    async def test_process_input_code(self):
        """Test processing of code input."""
        # Mock the _handle_code_input method
        self.builder._handle_code_input = AsyncMock()
        
        # Test code input
        test_code = "def test():\n    pass"
        await self.builder._process_input(test_code)
        
        # Verify code input handling
        self.builder._handle_code_input.assert_called_once_with(test_code)
    
    async def test_generate_link_code(self):
        """Test generation of link code."""
        source = "source_step"
        target = "target_step"
        
        # Mock the process method
        self.builder.process = AsyncMock(return_value="generated link code")
        
        # Generate link code
        result = await self.builder._generate_link_code(source, target)
        
        # Verify code generation
        self.assertEqual(result, "generated link code")
        self.builder.process.assert_called_once_with([
            "Generate code to link step source_step to target_step"
        ])
    
    def test_get_generated_outputs(self):
        """Test getting generated outputs."""
        # Set test values
        self.builder.generated_code = "test code"
        self.builder.generated_config = "test config"
        self.builder.generated_tests = "test tests"
        
        # Verify getters
        self.assertEqual(self.builder.get_generated_code(), "test code")
        self.assertEqual(self.builder.get_generated_config(), "test config")
        self.assertEqual(self.builder.get_generated_tests(), "test tests")


if __name__ == '__main__':
    unittest.main() 