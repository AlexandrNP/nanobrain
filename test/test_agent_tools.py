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

class CalculatorStep(Step):
    """
    A step that performs basic arithmetic operations.
    """
    def __init__(self, executor, **kwargs):
        super().__init__(executor, **kwargs)
    
    async def process(self, inputs):
        """
        Process arithmetic operations.
        
        Input format: ["operation", num1, num2]
        Supported operations: add, subtract, multiply, divide
        """
        if len(inputs) < 3:
            return "Error: Not enough inputs. Format should be [operation, num1, num2]"
        
        operation = inputs[0]
        try:
            num1 = float(inputs[1])
            num2 = float(inputs[2])
        except ValueError:
            return "Error: Inputs must be numbers"
        
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "multiply":
            return num1 * num2
        elif operation == "divide":
            if num2 == 0:
                return "Error: Cannot divide by zero"
            return num1 / num2
        else:
            return f"Error: Unsupported operation '{operation}'"

class TextProcessingStep(Step):
    """
    A step that performs basic text processing operations.
    """
    def __init__(self, executor, **kwargs):
        super().__init__(executor, **kwargs)
    
    async def process(self, inputs):
        """
        Process text operations.
        
        Input format: ["operation", text]
        Supported operations: uppercase, lowercase, reverse, count_words
        """
        if len(inputs) < 2:
            return "Error: Not enough inputs. Format should be [operation, text]"
        
        operation = inputs[0]
        text = inputs[1]
        
        if operation == "uppercase":
            return text.upper()
        elif operation == "lowercase":
            return text.lower()
        elif operation == "reverse":
            return text[::-1]
        elif operation == "count_words":
            return len(text.split())
        else:
            return f"Error: Unsupported operation '{operation}'"

class TestAgentTools(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create tool steps
        self.calculator_step = CalculatorStep(executor=self.executor)
        self.text_processing_step = TextProcessingStep(executor=self.executor)
        
        # Create agent with tools
        self.agent = Agent(
            executor=self.executor,
            model_name="gpt-3.5-turbo",
            memory_window_size=5,
            tools=[self.calculator_step, self.text_processing_step]
        )
    
    def test_tool_registration(self):
        """Test that tools are properly registered with the agent."""
        # Verify tools are stored
        self.assertEqual(len(self.agent.tools), 2)
        self.assertIn(self.calculator_step, self.agent.tools)
        self.assertIn(self.text_processing_step, self.agent.tools)
        
        # Verify langchain tools are created
        self.assertEqual(len(self.agent.langchain_tools), 2)
        
        # Verify tool names
        tool_names = [tool.name for tool in self.agent.langchain_tools]
        self.assertIn("CalculatorStep", tool_names)
        self.assertIn("TextProcessingStep", tool_names)
    
    def test_add_tool(self):
        """Test adding a tool to the agent."""
        # Create a new agent without tools
        agent = Agent(
            executor=self.executor,
            model_name="gpt-3.5-turbo"
        )
        
        # Verify no tools initially
        self.assertEqual(len(agent.tools), 0)
        self.assertEqual(len(agent.langchain_tools), 0)
        
        # Add a tool
        agent.add_tool(self.calculator_step)
        
        # Verify tool was added
        self.assertEqual(len(agent.tools), 1)
        self.assertEqual(len(agent.langchain_tools), 1)
        self.assertEqual(agent.langchain_tools[0].name, "CalculatorStep")
    
    def test_remove_tool(self):
        """Test removing a tool from the agent."""
        # Verify initial tools
        self.assertEqual(len(self.agent.tools), 2)
        
        # Remove a tool
        self.agent.remove_tool(self.calculator_step)
        
        # Verify tool was removed
        self.assertEqual(len(self.agent.tools), 1)
        self.assertEqual(len(self.agent.langchain_tools), 1)
        self.assertEqual(self.agent.langchain_tools[0].name, "TextProcessingStep")
    
    def test_tool_creation(self):
        """Test the creation of a tool from a Step object."""
        # Create a tool from a step
        tool = self.agent._create_tool_from_step(self.calculator_step)
        
        # Verify tool properties
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "CalculatorStep")
        self.assertIn("A step that performs basic arithmetic operations", tool.description)
    
    async def async_test_process_with_tools(self):
        """Test processing with tools."""
        # Mock the LLM's predict method to simulate tool calling
        self.agent.llm.predict = MagicMock(return_value="I'll use the calculator to add 5 and 3. The result is 8.")
        
        # Process input with tools
        result = await self.agent.process_with_tools(["Calculate 5 + 3"])
        
        # Verify result
        self.assertEqual(result, "I'll use the calculator to add 5 and 3. The result is 8.")
        
        # Verify memory was updated
        self.assertEqual(len(self.agent.memory), 2)
        self.assertEqual(self.agent.memory[0]["role"], "user")
        self.assertEqual(self.agent.memory[0]["content"], "Calculate 5 + 3")
        self.assertEqual(self.agent.memory[1]["role"], "assistant")
        self.assertEqual(self.agent.memory[1]["content"], "I'll use the calculator to add 5 and 3. The result is 8.")
    
    def test_process_with_tools(self):
        """Run the async test for process_with_tools."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.async_test_process_with_tools())
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main() 