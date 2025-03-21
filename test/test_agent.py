import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, mock_open

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add src directory to Python path
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Set testing environment variable
os.environ['NANOBRAIN_TESTING'] = '1'

from src.Agent import Agent
from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.LinkBase import LinkBase
from src.DataUnitBase import DataUnitBase
from src.enums import ComponentState
from src.ConfigManager import ConfigManager
from test.mock_langchain import MockOpenAI, MockChatOpenAI, MockPromptTemplate, MockAIMessage

class TestAgent(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create a ConfigManager instance with the test directory as the base path
        test_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_manager = ConfigManager(base_path=test_dir)
        
        # Load configuration from YAML file
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data='''
defaults:
  model_name: "gpt-3.5-turbo"
  model_class: null
  memory_window_size: 5
  prompt_file: "prompts/templates"
  prompt_template: "BASE_ASSISTANT"
  prompt_variables:
    role_description: "assist users with general tasks"
    specific_instructions: "Focus on clear communication"
  use_shared_context: false
  shared_context_key: null
''')):
                self.config = self.config_manager.get_config('Agent')
        
        # Create agent instance using configuration
        self.agent = Agent(
            executor=self.executor,
            model_name=self.config.get('defaults', {}).get('model_name', "gpt-3.5-turbo"),
            memory_window_size=self.config.get('defaults', {}).get('memory_window_size', 5)
        )
    
    def test_initialization(self):
        """Test that Agent initializes correctly with proper inheritance."""
        # Verify attributes are set correctly
        self.assertEqual(self.agent.model_name, "gpt-3.5-turbo")
        self.assertEqual(self.agent.memory_window_size, 5)
        self.assertEqual(self.agent.state, ComponentState.INACTIVE)
        self.assertFalse(self.agent.running)
        
        # Verify inheritance
        self.assertIsInstance(self.agent, Step)
        
        # Verify memory initialization
        self.assertEqual(len(self.agent.memory), 0)
    
    def test_initialize_llm(self):
        """Test the _initialize_llm method."""
        # Call _initialize_llm
        llm = self.agent._initialize_llm("gpt-4")
        
        # Verify the correct type was returned (either MockOpenAI or MockChatOpenAI is acceptable)
        self.assertTrue(isinstance(llm, (MockOpenAI, MockChatOpenAI)), 
                       f"Expected MockOpenAI or MockChatOpenAI, got {type(llm)}")
        
        # Since we're in testing mode, let's just verify we have some model name
        self.assertTrue(hasattr(llm, 'model_name'))
    
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
        from test.mock_langchain import MockAIMessage
        
        # Configure agent
        mock_response = MockAIMessage(content="AI response")
        self.agent.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Call process with test input
        result = await self.agent.process(["Hello, AI!"])
        
        # The result should be the same MockAIMessage object we mocked
        self.assertEqual(result, mock_response)
    
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
        self.assertIn("Message 2", history)
        self.assertIn("Response 2", history)
        self.assertIn("Message 3", history)
        self.assertIn("Response 3", history)
        self.assertNotIn("Message 1", history)
        self.assertNotIn("Response 1", history)
    
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
    
        # Test save_to_shared_context
        self.agent.save_to_shared_context("test_context")
        
        # Verify context was saved
        shared_context = Agent.get_shared_context("test_context")
        self.assertEqual(len(shared_context), 2)
        self.assertEqual(shared_context[0]["content"], "Hello")
        
        # Clear agent memory
        self.agent.clear_memories()
        self.assertEqual(len(self.agent.memory), 0)
        
        # Load from shared context
        self.agent.load_from_shared_context("test_context")
        self.assertEqual(len(self.agent.memory), 2)
        self.assertEqual(self.agent.memory[0]["content"], "Hello")
        
        # Clear shared context
        Agent.clear_shared_context("test_context")
        self.assertEqual(Agent.get_shared_context("test_context"), [])


if __name__ == '__main__':
    unittest.main() 