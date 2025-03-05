import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set testing environment variable
os.environ['NANOBRAIN_TESTING'] = '1'

from Agent import Agent
from Step import Step
from ExecutorBase import ExecutorBase
from LinkBase import LinkBase
from DataUnitBase import DataUnitBase
from enums import ComponentState
from mock_langchain import MockOpenAI, MockChatOpenAI, MockPromptTemplate

class TestAgent(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create agent instance (no need to patch since we're using mock_langchain)
        self.agent = Agent(
            executor=self.executor,
            model_name="gpt-3.5-turbo",
            memory_window_size=5
        )
    
    def test_initialization(self):
        """Test that Agent initializes correctly with proper inheritance."""
        # Verify attributes are set correctly
        self.assertEqual(self.agent.model_name, "gpt-3.5-turbo")
        self.assertEqual(self.agent.memory_window_size, 5)
        self.assertEqual(self.agent.state, ComponentState.INACTIVE)
        self.assertFalse(self.agent.running)
        
        # Verify inheritance from Step and PackageBase
        self.assertTrue(hasattr(self.agent, 'directory_tracer'))
        self.assertTrue(hasattr(self.agent, 'config_manager'))
        self.assertTrue(hasattr(self.agent, 'circuit_breaker'))
        
        # Verify Agent-specific attributes
        self.assertTrue(hasattr(self.agent, 'llm'))
        self.assertTrue(hasattr(self.agent, 'memory'))
        self.assertTrue(hasattr(self.agent, 'prompt_template'))
    
    def test_initialize_llm(self):
        """Test the _initialize_llm method."""
        # Call _initialize_llm
        llm = self.agent._initialize_llm("gpt-4")
        
        # Verify the correct type was returned
        self.assertIsInstance(llm, MockOpenAI)
        self.assertEqual(llm.model_name, "gpt-4")
    
    def test_load_prompt_template(self):
        """Test the _load_prompt_template method."""
        # Call _load_prompt_template with a template string
        template = self.agent._load_prompt_template(None, "You are an AI assistant. {input}")
        
        # Verify the correct type was returned
        self.assertIsInstance(template, MockPromptTemplate)
    
    @patch('os.path.exists')
    @patch('builtins.open')
    def test_load_prompt_template_from_file(self, mock_open, mock_exists):
        """Test the _load_prompt_template method with a file."""
        # Configure mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "You are an AI assistant. {input}"
        mock_open.return_value = mock_file
        
        # Call _load_prompt_template with a file path
        template = self.agent._load_prompt_template("prompt.txt", None)
        
        # Verify file was opened
        mock_open.assert_called_once()
        self.assertIsInstance(template, MockPromptTemplate)
    
    async def async_test_process(self):
        """Test the process method."""
        # Configure agent
        self.agent.llm.predict = MagicMock(return_value="AI response")
        
        # Call process with test input
        result = await self.agent.process(["Hello, AI!"])
        
        # Verify result
        self.assertEqual(result, "AI response")
    
    def test_process(self):
        """Run the async test for process."""
        asyncio.run(self.async_test_process())
    
    def test_update_memories(self):
        """Test the _update_memories method."""
        # Call _update_memories
        self.agent._update_memories("Hello, AI!", "AI response")
        
        # Verify memory was updated
        self.assertEqual(len(self.agent.memory), 2)
        self.assertEqual(self.agent.memory[0]["role"], "user")
        self.assertEqual(self.agent.memory[0]["content"], "Hello, AI!")
        self.assertEqual(self.agent.memory[1]["role"], "assistant")
        self.assertEqual(self.agent.memory[1]["content"], "AI response")
    
    def test_get_full_history(self):
        """Test the get_full_history method."""
        # Set up memory
        self.agent.memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        # Call get_full_history
        history = self.agent.get_full_history()
        
        # Verify result
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello")
    
    def test_get_context_history(self):
        """Test the get_context_history method."""
        # Set up memory with more messages than window size
        self.agent.memory_window_size = 2
        self.agent.memory = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Response 3"}
        ]
        
        # Call get_context_history
        history = self.agent.get_context_history()
        
        # Verify result contains only the last 2 exchanges (4 messages)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "Message 2")
        self.assertEqual(history[3]["content"], "Response 3")
    
    def test_clear_memories(self):
        """Test the clear_memories method."""
        # Set up memory
        self.agent.memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        # Call clear_memories
        self.agent.clear_memories()
        
        # Verify memory is cleared
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_shared_context_operations(self):
        """Test shared context operations."""
        # Clear any existing shared context
        Agent.clear_shared_context()
        
        # Set up memory
        self.agent.memory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        
        # Test dump_to_shared_context
        self.agent.dump_to_shared_context("test_context")
        
        # Verify shared context was updated
        self.assertEqual(len(Agent.get_shared_context("test_context")), 2)
        
        # Clear memory
        self.agent.clear_memories()
        self.assertEqual(len(self.agent.memory), 0)
        
        # Test load_from_shared_context
        self.agent.load_from_shared_context("test_context")
        
        # Verify memory was loaded from shared context
        self.assertEqual(len(self.agent.memory), 2)
        self.assertEqual(self.agent.memory[0]["content"], "Hello")
        
        # Test clear_shared_context
        Agent.clear_shared_context("test_context")
        
        # Verify shared context was cleared
        self.assertEqual(len(Agent.get_shared_context("test_context")), 0)


if __name__ == '__main__':
    unittest.main() 