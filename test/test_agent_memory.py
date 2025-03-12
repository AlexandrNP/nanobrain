#!/usr/bin/env python3
"""
Test file for the Agent class memory management.
Tests both traditional memory management and Langchain memory integration.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set testing mode
os.environ['NANOBRAIN_TESTING'] = '1'

from src.Agent import Agent
from src.ExecutorBase import ExecutorBase

# Import the appropriate classes based on testing mode
TESTING_MODE = os.environ.get('NANOBRAIN_TESTING', '0') == '1'
if TESTING_MODE:
    from test.mock_langchain import (
        MockConversationBufferMemory as ExpectedBufferMemory,
        MockConversationBufferWindowMemory as ExpectedWindowMemory
    )
else:
    from langchain.memory import (
        ConversationBufferMemory as ExpectedBufferMemory,
        ConversationBufferWindowMemory as ExpectedWindowMemory
    )

class MockExecutor(ExecutorBase):
    """Mock executor for testing."""
    def __init__(self):
        self.runnable_types = {"all"}
        self.energy_level = 1.0
        
    def can_execute(self, type_name):
        return True
        
    def execute(self, runnable):
        return runnable
        
    def recover_energy(self):
        self.energy_level = 1.0

class TestAgentMemory(unittest.TestCase):
    """Test case for Agent class memory management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = MockExecutor()
        
    def test_buffer_memory_initialization(self):
        """Test that the Agent correctly initializes with ConversationBufferMemory."""
        agent = Agent(
            executor=self.executor,
            use_buffer_window_memory=False,
            memory_key="test_history"
        )
        
        self.assertIsInstance(agent.langchain_memory, ExpectedBufferMemory)
        self.assertEqual(agent.langchain_memory.memory_key, "test_history")
        self.assertTrue(agent.langchain_memory.return_messages)
        
    def test_buffer_window_memory_initialization(self):
        """Test that the Agent correctly initializes with ConversationBufferWindowMemory."""
        agent = Agent(
            executor=self.executor,
            use_buffer_window_memory=True,
            memory_window_size=3,
            memory_key="test_history"
        )
        
        self.assertIsInstance(agent.langchain_memory, ExpectedWindowMemory)
        self.assertEqual(agent.langchain_memory.memory_key, "test_history")
        self.assertEqual(agent.langchain_memory.k, 3)
        self.assertTrue(agent.langchain_memory.return_messages)
        
    def test_memory_update(self):
        """Test that memories are updated in both internal memory and Langchain memory."""
        agent = Agent(
            executor=self.executor,
            memory_window_size=5
        )
        
        # Process inputs to update memory
        agent._update_memories("Test user input", "Test assistant response")
        agent.langchain_memory.save_context({"input": "Test user input"}, {"output": "Test assistant response"})
        
        # Check internal memory
        self.assertEqual(len(agent.memory), 2)
        self.assertEqual(agent.memory[0]["role"], "user")
        self.assertEqual(agent.memory[0]["content"], "Test user input")
        self.assertEqual(agent.memory[1]["role"], "assistant")
        self.assertEqual(agent.memory[1]["content"], "Test assistant response")
        
        # Check Langchain memory
        memory_vars = agent.langchain_memory.load_memory_variables({})
        self.assertIn(agent.memory_key, memory_vars)
        
    def test_clear_memory(self):
        """Test that clear_memories clears both internal and Langchain memory."""
        agent = Agent(
            executor=self.executor,
            memory_window_size=5
        )
        
        # Add memories
        agent._update_memories("Test user input", "Test assistant response")
        agent.langchain_memory.save_context({"input": "Test user input"}, {"output": "Test assistant response"})
        
        # Verify memories exist
        self.assertEqual(len(agent.memory), 2)
        memory_vars = agent.langchain_memory.load_memory_variables({})
        self.assertIn(agent.memory_key, memory_vars)
        
        # Clear memories
        agent.clear_memories()
        
        # Verify memories are cleared
        self.assertEqual(len(agent.memory), 0)
        memory_vars = agent.langchain_memory.load_memory_variables({})
        self.assertEqual(len(memory_vars[agent.memory_key]), 0)
        
    def test_shared_context(self):
        """Test that shared context works with Langchain memory."""
        # Create first agent
        agent1 = Agent(
            executor=self.executor,
            memory_window_size=5,
            use_shared_context=True,
            shared_context_key="test_shared"
        )
        
        # Add memories to first agent
        agent1._update_memories("Test user input", "Test assistant response")
        agent1.langchain_memory.save_context({"input": "Test user input"}, {"output": "Test assistant response"})
        
        # Save to shared context
        agent1.save_to_shared_context("test_shared")
        
        # Create second agent with same shared context
        agent2 = Agent(
            executor=self.executor,
            memory_window_size=5,
            use_shared_context=True,
            shared_context_key="test_shared"
        )
        
        # Load from shared context
        agent2.load_from_shared_context("test_shared")
        
        # Verify memories in second agent
        self.assertEqual(len(agent2.memory), 2)
        self.assertEqual(agent2.memory[0]["role"], "user")
        self.assertEqual(agent2.memory[0]["content"], "Test user input")
        
        # Verify Langchain memory in second agent
        memory_vars = agent2.langchain_memory.load_memory_variables({})
        self.assertIn(agent2.memory_key, memory_vars)
        
if __name__ == "__main__":
    unittest.main() 