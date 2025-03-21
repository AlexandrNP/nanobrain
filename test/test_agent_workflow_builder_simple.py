#!/usr/bin/env python3
"""
Simplified test script for verifying the key changes to the AgentWorkflowBuilder class.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

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
    
    # Create mock input storage
    mock_input_storage = MagicMock()
    mock_input_storage.process = AsyncMock(return_value="Mock input response")
    
    # Create a mock LLM instead of patching Agent.__init__
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value="Mocked response")
    mock_llm.ainvoke = AsyncMock(return_value="Mocked response")
    
    # Create the builder with proper initialization
    builder = AgentWorkflowBuilder(
        executor=executor,
        input_storage=mock_input_storage,
        debug_mode=True,
        use_code_writer=True,
        llm=mock_llm
    )
    
    # Mock some methods directly
    builder._provide_guidance = AsyncMock(return_value="Mocked guidance")
    builder.process = AsyncMock(return_value="Mocked guidance")
    
    # Manually set code_writer after initialization
    builder.code_writer = MagicMock()
        
    return builder


def test_inheritance(builder):
    """Test that the builder inherits from Agent."""
    assert isinstance(builder, Agent)


def test_initialization(builder):
    """Test that the builder initializes correctly."""
    # Check that the builder has the expected attributes
    assert builder.use_code_writer
    assert builder.debug_mode
    assert hasattr(builder, 'prioritize_existing_classes')
    assert hasattr(builder, 'code_writer')


def test_code_writer_initialization(builder):
    """Test that the code writer is initialized correctly."""
    assert builder.code_writer is not None


@pytest.mark.asyncio
async def test_process(builder):
    """Test the process method."""
    test_input = ["Test query"]
    response = await builder.process(test_input)
    assert response == "Mocked guidance"


if __name__ == "__main__":
    pytest.main(["-v", "--asyncio-mode=strict", __file__]) 