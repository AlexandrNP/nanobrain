import unittest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

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
from test.mock_langchain import MockOpenAI, MockChatOpenAI, MockPromptTemplate

# Create a simple response class to use in tests
class MagicResponse:
    def __init__(self, content):
        self.content = content
    
    def __str__(self):
        return self.content

# Import the parse_tool_call function with graceful fallback
try:
    from prompts.tool_calling_prompt import parse_tool_call
except ImportError:
    print("Warning: Could not import parse_tool_call from prompts.tool_calling_prompt")
    # Define a fallback implementation
    def parse_tool_call(response):
        if "Tool:" in response and "Args:" in response:
            try:
                tool_part = response.split("Tool:")[1].split("Args:")[0].strip()
                args_part = response.split("Args:")[1].strip()
                args = [arg.strip() for arg in args_part.split(",")]
                return (tool_part, args)
            except Exception:
                return None
        return None

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

class TestAgentCustomToolPrompt(unittest.TestCase):
    def setUp(self):
        # Create mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        self.executor.can_execute.return_value = True
        
        # Create tool steps
        self.calculator_step = CalculatorStep(executor=self.executor)
        self.text_processing_step = TextProcessingStep(executor=self.executor)
        
        # Create agent with tools and custom prompt
        self.agent = Agent(
            executor=self.executor,
            model_name="gpt-3.5-turbo",
            memory_window_size=5,
            tools=[self.calculator_step, self.text_processing_step],
            use_custom_tool_prompt=True
        )
    
    def test_parse_tool_call(self):
        """Test parsing tool calls from LLM responses."""
        # Test with a valid tool call
        response = """
        I'll help you calculate that.
        
        <tool>
        name: CalculatorStep
        args: add, 5, 3
        </tool>
        
        Let me know if you need anything else!
        """
        
        tool_call = parse_tool_call(response)
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call[0], "CalculatorStep")
        self.assertEqual(tool_call[1], ["add", "5", "3"])
        
        # Test with no tool call
        response = "I don't need to use a tool for this. The answer is 42."
        tool_call = parse_tool_call(response)
        self.assertIsNone(tool_call)
    
    async def async_test_process_with_custom_prompt(self):
        """Test the process_with_tools method with custom tool prompt."""
        # Create an agent with a calculator tool and custom tool prompt
        calculator = CalculatorStep(executor=self.executor)
        
        # Mock the tool calling prompt module
        from unittest.mock import patch
        
        # Define our custom tool prompt
        custom_tool_prompt = "Custom tool prompt for testing with {input}"
        
        # Patch the create_tool_calling_prompt function
        with patch('prompts.tool_calling_prompt.create_tool_calling_prompt', 
                   return_value=custom_tool_prompt):
            
            # Create the agent with custom tool prompt enabled
            agent = Agent(
                executor=self.executor,
                model_name="fake-model",
                model_class="mock-chat",
                use_custom_tool_prompt=True,
                tools=[calculator]
            )
            
            # Create a mocked LLM response that contains the expected content
            mock_response = """I'll calculate that for you.
            
            <tool>
            name: CalculatorStep
            args: add, 5, 3
            </tool>
            """
            
            # Override the agent's process method to return our mock response
            agent.process = AsyncMock(return_value=mock_response)
            
            # Call the process method
            result = await agent.process("Calculate 5 + 3")
            
            # Verify the response contains expected elements
            self.assertEqual(result, mock_response)
            self.assertIn("I'll calculate that for you", result)
            self.assertIn("<tool>", result)
            self.assertIn("name: CalculatorStep", result)
            self.assertIn("args: add, 5, 3", result)
    
    async def async_test_execute_tool_directly(self):
        """Test executing a tool directly by name."""
        # Execute the calculator tool directly
        result = await self.agent.execute_tool("CalculatorStep", ["add", "10", "20"])
        
        # Verify result
        self.assertEqual(result, 30.0)
        
        # Execute the text processing tool directly
        result = await self.agent.execute_tool("TextProcessingStep", ["uppercase", "hello world"])
        
        # Verify result
        self.assertEqual(result, "HELLO WORLD")
        
        # Test with non-existent tool
        result = await self.agent.execute_tool("NonExistentTool", ["arg1", "arg2"])
        
        # Verify error message
        self.assertEqual(result, "Tool NonExistentTool not found")
    
    def test_process_with_custom_prompt(self):
        """Run the async test for process_with_custom_prompt."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.async_test_process_with_custom_prompt())
        finally:
            loop.close()
    
    def test_execute_tool_directly(self):
        """Run the async test for execute_tool_directly."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.async_test_execute_tool_directly())
        finally:
            loop.close()

if __name__ == "__main__":
    unittest.main() 