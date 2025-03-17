#!/usr/bin/env python3
"""
Test script for verifying the updated AgentWorkflowBuilder class.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set testing mode
os.environ['NANOBRAIN_TESTING'] = '1'
# Mock OpenAI API key for tests that need it
os.environ['OPENAI_API_KEY'] = 'test_key'

from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.ExecutorFunc import ExecutorFunc
from src.Agent import Agent


@pytest.fixture
def builder():
    """Fixture that creates a mocked AgentWorkflowBuilder instance."""
    executor = MagicMock()
    executor.execute = AsyncMock(return_value="Mocked response")
    
    builder = AgentWorkflowBuilder(
        executor=executor,
        _debug_mode=True,
        use_code_writer=True
    )
    
    # Mock the _provide_guidance method to avoid real API calls
    builder._provide_guidance = AsyncMock(return_value="Mocked guidance")
    return builder


def test_initialization(builder):
    """Test that the builder initializes correctly."""
    # Check that the builder inherits from Agent
    assert isinstance(builder, Agent)
    
    # Check that the builder has the expected attributes
    assert builder.use_code_writer
    assert hasattr(builder, 'prioritize_existing_classes')
    assert hasattr(builder, 'code_writer')


def test_code_writer_initialization(builder):
    """Test that the code writer is initialized correctly."""
    assert builder.code_writer is not None


@pytest.mark.asyncio
async def test_process_method(builder):
    """Test that the process method processes input."""
    # Process test input
    test_input = ["How does NanoBrain work?"]
    response = await builder.process(test_input)
    
    # Check that _provide_guidance was called with the input
    builder._provide_guidance.assert_called_once_with(test_input[0])
    
    # Check that the response is correct
    assert response == "Mocked guidance"


if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=strict", __file__]) 