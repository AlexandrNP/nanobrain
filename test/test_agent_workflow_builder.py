#!/usr/bin/env python3
"""
Test script for AgentWorkflowBuilder instantiation.

This script tests the instantiation of the AgentWorkflowBuilder class
using the ConfigManager factory method based on the YAML configuration.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from src.ConfigManager import ConfigManager
from src.ExecutorFunc import ExecutorFunc
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder

@pytest.mark.asyncio
async def test_agent_workflow_builder():
    """Test the AgentWorkflowBuilder instantiation."""
    try:
        # Get the base path
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"Base path: {base_path}")
        
        print("Setting up ConfigManager...")
        config_manager = ConfigManager(base_path=base_path)

        print("Creating executor...")
        executor = config_manager.create_instance("ExecutorFunc")
        if executor is None:
            print("Failed to create executor from ConfigManager.")
            executor = ExecutorFunc()
            print("Created fallback executor directly.")

        print("\nTesting approach 1: Direct instantiation for comparison")
        # Create an instance directly
        direct_builder = AgentWorkflowBuilder(
            executor=executor,
            model_name="gpt-3.5-turbo",
            _debug_mode=True
        )
        print(f"Successfully created AgentWorkflowBuilder instance directly.")
        print(f"Debug mode: {direct_builder._debug_mode}")
        print(f"Model name: {direct_builder.model_name}")

        print("\nTesting approach 2: ConfigManager factory instantiation")
        # Create an instance using the ConfigManager
        builder = config_manager.create_instance(
            "AgentWorkflowBuilder",
            executor=executor,
            _debug_mode=True
        )
        
        if builder:
            print(f"Successfully created AgentWorkflowBuilder instance via ConfigManager.")
            print(f"Debug mode: {builder._debug_mode}")
            print(f"Model name: {builder.model_name}")
            
            # Check for _process_config method
            if hasattr(builder, '_process_config'):
                print("_process_config method exists.")
            else:
                print("ERROR: _process_config method does not exist.")
                
            # Check other key attributes
            print("\nChecking key attributes:")
            attributes = [
                'use_code_writer', 'executor', 'config_manager', 
                'generated_code', 'generated_config', 'generated_tests'
            ]
            
            for attr in attributes:
                if hasattr(builder, attr):
                    print(f"✅ {attr}: {getattr(builder, attr)}")
                else:
                    print(f"❌ {attr} not found")
        else:
            print("Failed to create AgentWorkflowBuilder instance via ConfigManager.")

        print("\nTest completed.")
        assert builder is not None

    except Exception as e:
        import traceback
        print(f"Error testing AgentWorkflowBuilder: {e}")
        traceback.print_exc()
        pytest.fail(f"Exception occurred: {e}")

if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=strict", __file__]) 