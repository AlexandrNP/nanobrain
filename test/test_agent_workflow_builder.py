#!/usr/bin/env python3
"""
Test script for AgentWorkflowBuilder instantiation.

This script tests the instantiation of the AgentWorkflowBuilder class
using the ConfigManager factory method based on the YAML configuration.
"""

import os
import sys
import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from src.ConfigManager import ConfigManager
from src.ExecutorFunc import ExecutorFunc
from src.DataStorageCommandLine import DataStorageCommandLine
from src.Agent import Agent
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder

class TestAgentWorkflowBuilder(unittest.TestCase):
    @patch('src.Agent._initialize_llm')
    def test_agent_workflow_builder(self, mock_initialize_llm):
        """Test the AgentWorkflowBuilder instantiation."""
        # Mock the LLM to avoid needing an API key
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value="Mocked response")
        mock_initialize_llm.return_value = mock_llm
        
        try:
            # Get the base path
            base_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Base path: {base_path}")
            
            print("Creating executor...")
            executor = ExecutorFunc()
            
            # Create a mock input storage
            mock_input_storage = MagicMock()
            mock_input_storage.process = AsyncMock(return_value="Mock input response")

            print("\nTesting direct instantiation")
            # Create an instance directly - using debug_mode instead of _debug_mode
            builder = AgentWorkflowBuilder(
                executor=executor,
                input_storage=mock_input_storage,
                model_name="gpt-3.5-turbo",
                debug_mode=True
            )
            
            print(f"Successfully created AgentWorkflowBuilder instance.")
            print(f"Debug mode: {builder.debug_mode}")
            print(f"Model name: {builder.model_name}")
            
            # Check key attributes
            print("\nChecking key attributes:")
            self.assertIsInstance(builder, Agent)
            self.assertTrue(builder.debug_mode)
            self.assertEqual(builder.model_name, "gpt-3.5-turbo")
            self.assertEqual(builder.executor, executor)
            self.assertEqual(builder.input_storage, mock_input_storage)
            self.assertTrue(builder.use_code_writer)
            self.assertTrue(builder.prioritize_existing_classes)
            
            print("\nTest completed successfully.")
            self.assertIsNotNone(builder)

        except Exception as e:
            import traceback
            print(f"Error testing AgentWorkflowBuilder: {e}")
            traceback.print_exc()
            self.fail(f"Exception occurred: {e}")

if __name__ == "__main__":
    unittest.main() 