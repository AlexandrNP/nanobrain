#!/usr/bin/env python3
"""
AgentWorkflowBuilder - Agent that provides guidance on using the NanoBrain framework.

This module provides an agent that guides users through the process of creating
workflows and steps in the NanoBrain framework, offering explanations, suggestions,
and best practices.

Biological analogy: Prefrontal cortex's executive function combined with hippocampus's memory.
Justification: Like how the prefrontal cortex provides executive function guidance
based on hippocampus's stored memories, this agent guides users through workflow creation
based on its knowledge of the NanoBrain framework.
"""

from typing import List, Dict, Any, Optional, Union
import os
import asyncio
from pathlib import Path
import yaml
import re
import importlib
import glob

from src.Agent import Agent
from src.ExecutorBase import ExecutorBase
from src.Workflow import Workflow
from src.Step import Step
from src.enums import ComponentState
from src.ConfigManager import ConfigManager

# Import AgentCodeWriter for optional code writing capability
from builder.AgentCodeWriter import AgentCodeWriter

# Import prompt templates
from builder.prompts import WORKFLOW_BUILDER_PROMPT, WORKFLOW_BUILDER_FRAMEWORK_CONTEXT, CODE_WRITER_PROMPT

class AgentWorkflowBuilder(Agent):
    """
    AgentWorkflowBuilder - Agent that provides guidance on using the NanoBrain framework.
    
    This agent helps users understand how to:
    1. Create and structure workflows
    2. Design effective steps
    3. Connect components appropriately
    4. Follow best practices for the NanoBrain framework
    
    It can optionally include code writing capabilities through AgentCodeWriter.
    
    Biological analogy: Prefrontal cortex guidance with hippocampus knowledge.
    Justification: Like how the prefrontal cortex provides guidance based on
    stored knowledge, this agent guides users based on its knowledge of the
    NanoBrain framework and workflow patterns.
    """
    
    def __init__(self, executor: ExecutorBase, system_prompt: str = None, 
                 model_name: str = None, max_tokens: int = 2000, temperature: float = 0.5, 
                 _debug_mode: bool = False, **kwargs):
        """
        Initialize the AgentWorkflowBuilder.
        
        Args:
            executor: Executor for running the agent
            system_prompt: System prompt for the agent (optional)
            model_name: Name of the model to use for the agent (optional)
            max_tokens: Maximum number of tokens for the model (default: 2000)
            temperature: Temperature for the model (default: 0.5)
            _debug_mode: Whether to run in debug mode (default: False)
            **kwargs: Additional arguments to pass to the parent class
        """
        # Store debug mode explicitly
        self._debug_mode = _debug_mode
        
        # Save the executor and other key parameters
        self.executor = executor
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Configuration defaults
        self.prompt_variables = {}
        self.use_code_writer = kwargs.get('use_code_writer', True)
        self.current_step_dir = None
        self.workflow_context = {}
        
        # Process configuration if provided
        config = kwargs.get('config', {})
        if config:
            self._process_config(config)
        
        # Debug info
        if self._debug_mode:
            print(f"Debug mode enabled for AgentWorkflowBuilder")
            print(f"Model name: {model_name}")
            print(f"use_code_writer: {self.use_code_writer}")
        
        # Initialize the base class
        super().__init__(
            executor=executor, 
            system_prompt=system_prompt, 
            model_name=model_name, 
            max_tokens=max_tokens, 
            temperature=temperature, 
            _debug_mode=_debug_mode, 
            **kwargs
        )
        
        # Extract variables from kwargs or use defaults
        self.defaults_directory = kwargs.get('defaults_directory', 'defaults')
        self.code_directory = kwargs.get('code_directory', 'src')
        self.default_config_path = kwargs.get('default_config_path', os.path.join(os.getcwd(), 'config'))
        
        # Storage for generated code, config, and tests
        self.generated_code = ""
        self.generated_config = ""
        self.generated_tests = ""
        
        # Initialize the config manager for finding existing classes
        self.config_manager = kwargs.get('config_manager', ConfigManager(base_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Track if we should prioritize reusing existing classes
        self.prioritize_existing_classes = kwargs.get('prioritize_existing_classes', True)
        
        # Initialize code writer if enabled
        self.code_writer = None
        if self.use_code_writer:
            self._init_code_writer(executor, _debug_mode)
    
    def _init_code_writer(self, executor: ExecutorBase, _debug_mode: bool = False):
        """
        Initialize the code writer component if needed.
        
        Args:
            executor: Executor for the code writer
            _debug_mode: Whether to run in debug mode
        """
        try:
            # Create the code writer agent - ensuring we pass enough information for it to function
            self.code_writer = AgentCodeWriter(
                executor=executor,
                system_prompt=CODE_WRITER_PROMPT,  # Import this from prompts.py
                model_name=self.model_name,        # Pass our model name (consistent with the parameter name in AgentCodeWriter)
                max_tokens=self.max_tokens,        # Pass our max tokens
                temperature=self.temperature,      # Pass our temperature
                _debug_mode=_debug_mode           # Pass debug mode
            )
            
            if _debug_mode:
                print("Successfully initialized AgentCodeWriter component")
        except Exception as e:
            print(f"Error initializing code writer: {e}")
            self.use_code_writer = False
    
    async def _safe_execute(self, executor, messages, max_tokens=None, temperature=None):
        """
        Safely execute a prompt using the provided executor with fallback options.
        
        Args:
            executor: The executor to use
            messages: The messages to process
            max_tokens: Maximum tokens for response (optional)
            temperature: Temperature for response (optional)
            
        Returns:
            The executor's response or an error message
        """
        if executor is None:
            return "Error: Executor is not available."
            
        # Try different approaches to execute with the executor
        try:
            # First check if the executor has an execute_async method
            if hasattr(executor, 'execute_async'):
                return await executor.execute_async(messages)
            
            # Fall back to the regular execute method
            return await executor.execute(messages)
        except TypeError as e:
            print(f"First execution attempt failed: {e}, trying with parameters...")
            
            try:
                # Try with parameters dictionary
                params = {"messages": messages}
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if temperature is not None:
                    params["temperature"] = temperature
                
                # Check if the executor has an execute_async method
                if hasattr(executor, 'execute_async'):
                    return await executor.execute_async(**params)
                
                # Fall back to the regular execute method
                return await executor.execute(**params)
            except Exception as e2:
                print(f"Second execution attempt failed: {e2}, trying with stringified input...")
                
                try:
                    # Last resort - convert to string
                    if isinstance(messages, list) and len(messages) > 0:
                        user_message = ""
                        for msg in messages:
                            if msg.get("role") == "user":
                                user_message = msg.get("content", "")
                                break
                        if user_message:
                            # Check if the executor has an execute_async method
                            if hasattr(executor, 'execute_async'):
                                return await executor.execute_async(user_message)
                            
                            # Fall back to the regular execute method
                            return await executor.execute(user_message)
                    
                    # Check if the executor has an execute_async method
                    if hasattr(executor, 'execute_async'):
                        return await executor.execute_async(str(messages))
                    
                    # Fall back to the regular execute method
                    return await executor.execute(str(messages))
                except Exception as e3:
                    print(f"All execution attempts failed: {e3}")
                    return f"Error processing request: {str(e3)}"
                    
    async def process(self, inputs=None):
        """
        Process input data and generate a response.
        
        Args:
            inputs: Input data to process
            
        Returns:
            Generated response
        """
        # Extract user input from inputs parameter
        user_input = self._extract_user_input(inputs)
        
        # Skip if there's no input
        if not user_input:
            return "No input provided."
        
        # Process the user input to generate response
        try:
            # Use the _provide_guidance method to handle all user inputs
            return await self._provide_guidance(user_input)
            
        except Exception as e:
            if self._debug_mode:
                import traceback
                traceback.print_exc()
            return f"Error processing input: {str(e)}"
    
    def _should_generate_code(self, user_input: str) -> bool:
        """
        Determine if the input is requesting code generation.
        
        Args:
            user_input: User's input message
        
        Returns:
            True if code generation is requested, False otherwise
        """
        # Check for explicit code generation requests
        code_patterns = [
            r'(?i)generate\s+code',
            r'(?i)write\s+code',
            r'(?i)create\s+(a|the)\s+step',
            r'(?i)implement\s+(a|the)',
            r'(?i)provide\s+code',
            r'(?i)code\s+example',
            r'(?i)sample\s+code'
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, user_input):
                return True
        
        return False
    
    def _is_requesting_new_class(self, user_input: str) -> bool:
        """
        Determine if the input is explicitly requesting a new class.
        
        Args:
            user_input: User's input message
        
        Returns:
            True if a new class is explicitly requested, False otherwise
        """
        # Check for explicit requests for new classes
        new_class_patterns = [
            r'(?i)create\s+new\s+class',
            r'(?i)implement\s+new\s+class',
            r'(?i)write\s+new\s+class',
            r'(?i)new\s+implementation',
            r'(?i)custom\s+class',
            r'(?i)don\'t\s+use\s+existing',
            r'(?i)from\s+scratch'
        ]
        
        for pattern in new_class_patterns:
            if re.search(pattern, user_input):
                return True
        
        return False
    
    def _ensure_code_writer(self):
        """
        Ensure that the code writer is initialized if use_code_writer is True.
        """
        if self.use_code_writer and self.code_writer is None:
            self._init_code_writer(self.executor, self._debug_mode)
        return self.code_writer is not None
        
    async def _provide_guidance(self, user_input):
        """
        Provide guidance on using the NanoBrain framework.
        
        Args:
            user_input: The user's input requesting guidance
            
        Returns:
            Guidance response
        """
        # Check if user is asking for code generation
        if self._should_generate_code(user_input):
            if not self._ensure_code_writer():
                return "I'm sorry, the code writer component is not available."
            return await self._generate_code_from_user_input(user_input)
        
        # Check if we should enhance the input with context
        enhanced_input = user_input
        
        # If the user is asking about components and we prioritize existing classes,
        # enhance the input with a reminder about reusing components
        if hasattr(self, 'prioritize_existing_classes') and self._is_asking_about_components(user_input) and self.prioritize_existing_classes:
            enhanced_input = f"{user_input}\n\nRemember to prioritize reusing existing components with custom configurations over creating new classes."
        
        # Process the input using the LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": enhanced_input}
        ]
        
        # Prepare explanatory context about NanoBrain if needed
        if self._needs_context(user_input):
            messages.insert(1, {"role": "system", "content": self.framework_context})
        
        # Use our safe execute method
        return await self._safe_execute(
            self.executor,
            messages,
            self.max_tokens,
            self.temperature
        )
    
    def _needs_context(self, user_input):
        """
        Check if the user needs context about the NanoBrain framework.
        
        Args:
            user_input: User input to check
            
        Returns:
            True if the user needs context, False otherwise
        """
        # Convert DataUnitString to string if needed
        if hasattr(user_input, 'get') and callable(getattr(user_input, 'get')):
            user_input = user_input.get()
            
        # If user_input is not a string, return False
        if not isinstance(user_input, str):
            return False
            
        # Check if the user needs context
        pattern = r"(?i)(what is|how does|explain|tell me about|describe|help me understand).*?(nanobrain|framework|workflow|step|link|data unit)"
        return bool(re.search(pattern, user_input))
    
    def _is_asking_about_components(self, user_input):
        """
        Check if the user is asking about components.
        
        Args:
            user_input: User input to check
            
        Returns:
            True if the user is asking about components, False otherwise
        """
        # Convert DataUnitString to string if needed
        if hasattr(user_input, 'get') and callable(getattr(user_input, 'get')):
            user_input = user_input.get()
            
        # If user_input is not a string, return False
        if not isinstance(user_input, str):
            return False
            
        # Check if the user is asking about components
        pattern = r"(?i)(how|what|which|is there|are there|do you have|can i use|should i use).*?(component|class|step|workflow|link|data unit)"
        return bool(re.search(pattern, user_input))
    
    def _extract_user_input(self, inputs):
        """
        Extract user input from the inputs parameter.
        
        Args:
            inputs: Input data to process
            
        Returns:
            Extracted user input as a string
        """
        if inputs is None:
            return None
            
        # If inputs is a list, extract the first item
        if isinstance(inputs, list):
            if not inputs:
                return None
            # If the first item is a string, return it directly
            if isinstance(inputs[0], str):
                return inputs[0]
            # If the first item is a dictionary, try to extract the input
            elif isinstance(inputs[0], dict):
                for key in ['input', 'query', 'message', 'text', 'content']:
                    if key in inputs[0] and inputs[0][key] is not None:
                        return inputs[0][key]
            # If we can't extract a string, return the first item as is
            return inputs[0]
            
        # If inputs is a dictionary, look for common keys
        if isinstance(inputs, dict):
            for key in ['input', 'query', 'message', 'text', 'content']:
                if key in inputs and inputs[key] is not None:
                    return inputs[key]
            
            # If no common keys found, return the first value
            if inputs:
                return next(iter(inputs.values()))
                
        # If inputs is a string, return it directly
        if isinstance(inputs, str):
            return inputs
            
        # If inputs is a DataUnitBase, get its value
        if hasattr(inputs, 'get') and callable(getattr(inputs, 'get')):
            return inputs.get()
            
        # If we can't extract a string, return the inputs as is
        return str(inputs)
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from a markdown-formatted response.
        
        Args:
            response: Response from the model
        
        Returns:
            Cleaned code
        """
        # Check if the response contains a Python code block
        if "```python" in response:
            # Extract the code block
            code_start = response.find("```python") + 10
            code_end = response.find("```", code_start)
            
            if code_end > code_start:
                return response[code_start:code_end].strip()
        
        # If no code block is found, check if the entire response looks like code
        if response.strip().startswith(("import ", "from ", "class ", "def ")):
            return response.strip()
        
        # Otherwise, return the original response
        return response

    async def _generate_config_from_code(self, code: str) -> str:
        """
        Generate a YAML configuration file based on the provided code.
        
        Args:
            code: The code to generate a configuration for
            
        Returns:
            YAML configuration as a string
        """
        if not self._ensure_code_writer():
            return "Error: Code writer is not available. Cannot generate configuration."
        
        # Build a prompt for generating the configuration
        prompt = f"""Generate a YAML configuration file for this NanoBrain code:
        
        ```python
        {code}
        ```
        
        The configuration should include:
        - defaults section with class and parameters
        - metadata section with description and biological analogy
        - validation section with required and optional parameters
        - examples section with at least one example
        
        Return ONLY the YAML content without any explanations or markdown formatting.
        """
        
        config_yaml = await self.code_writer.process([prompt])
        
        # Store the generated configuration
        self.generated_config = config_yaml
        
        return config_yaml
        
    async def _generate_tests_from_code(self, code: str) -> str:
        """
        Generate test code for the provided implementation.
        
        Args:
            code: The code to generate tests for
            
        Returns:
            Test code as a string
        """
        if not self._ensure_code_writer():
            return "Error: Code writer is not available. Cannot generate tests."
        
        # Extract the class name from the code
        class_name_match = re.search(r'class\s+(\w+)', code)
        if not class_name_match:
            return "Error: Could not extract class name from the code."
            
        class_name = class_name_match.group(1)
        
        # Build a prompt for generating tests
        prompt = f"""Generate test code for this NanoBrain class:
        
        ```python
        {code}
        ```
        
        The tests should:
        - Use unittest framework
        - Include setup and teardown methods
        - Test all public methods
        - Include at least one async test
        - Provide proper mocks for dependencies
        
        Return ONLY the Python test code without any explanations or markdown formatting.
        """
        
        test_code = await self.code_writer.process([prompt])
        
        # Store the generated tests
        self.generated_tests = test_code
        
        return test_code
    
    # Additional helper methods can be added here
    
    async def generate_step_template(self, step_name: str, base_class: str = "Step", description: str = None) -> str:
        """
        Generate a step template with the specified name and description.
        
        Args:
            step_name: The name of the step class (without 'Step' prefix)
            base_class: The base class for the step (default: "Step")
            description: Description of the step's purpose
            
        Returns:
            Generated step template code
        """
        if not self._ensure_code_writer():
            return "Error: Code writer is not available. Cannot create step template."
            
        # Format the class name with Step prefix if not already included
        if not step_name.startswith('Step'):
            class_name = f"Step{step_name}"
        else:
            class_name = step_name
            
        # Create the prompt for the code writer
        prompt = f"""Generate a template for a {class_name} class that inherits from {base_class}.
        
        The step should:
        - Include a proper docstring with biological analogy
        - Have a well-structured __init__ method that calls the parent constructor
        - Implement the process method to handle the data
        - Include proper type hints and documentation
        
        {f"Description: {description}" if description else ""}
        """
        
        # Get code from the code writer
        code = await self.code_writer.process([prompt])
        
        # Store the generated code
        self.generated_code = code
        
        # Also generate the configuration
        self.generated_config = await self._generate_config_from_code(code)
        
        # And generate tests
        self.generated_tests = await self._generate_tests_from_code(code)
        
        return code

    async def suggest_implementation(self, step_name: str, description: str) -> str:
        """
        Suggest an implementation for a step with the given requirements.
        
        Args:
            step_name: The name of the step
            description: Description of what the step should do
            
        Returns:
            Suggested implementation
        """
        if not self._ensure_code_writer():
            return "Error: Code writer is not available. Cannot suggest implementation."
            
        # Create a prompt for the code writer
        prompt = f"""Suggest an implementation for a {step_name} in the NanoBrain framework.
        
        The step should:
        {description}
        
        Provide the complete Python class definition including:
        - Appropriate imports
        - Class documentation with biological analogy
        - Constructor that initializes necessary fields
        - Asynchronous process method that handles the logic
        - Any additional helper methods
        
        Focus on making the code maintainable, testable, and following NanoBrain best practices.
        """
        
        # Get implementation from the code writer
        implementation = await self.code_writer.process([prompt])
        
        # Store the generated code
        self.generated_code = implementation
        
        # Also generate the configuration
        self.generated_config = await self._generate_config_from_code(implementation)
        
        # And generate tests
        self.generated_tests = await self._generate_tests_from_code(implementation)
        
        return implementation
        
    async def analyze_workflow(self, workflow_config: Dict[str, Any]) -> str:
        """
        Analyze a workflow configuration and provide insights.
        
        Args:
            workflow_config: Configuration dictionary for the workflow
            
        Returns:
            Analysis text
        """
        if not self._ensure_code_writer():
            return "Error: Code writer is not available. Cannot analyze workflow."
            
        # Convert the config to YAML format
        try:
            config_yaml = yaml.dump(workflow_config, default_flow_style=False)
        except Exception as e:
            return f"Error converting workflow config to YAML: {e}"
        
        # Create a prompt for the workflow analysis
        prompt = f"""Analyze this NanoBrain workflow configuration:
        
        ```yaml
        {config_yaml}
        ```
        
        Provide insights on:
        - Overall workflow structure
        - Potential issues or improvements
        - Best practices being followed or violated
        - Suggestions for enhanced performance or readability
        """
        
        # Get analysis from the code writer
        analysis = await self.code_writer.process([prompt])
        
        return analysis

    def _process_config(self, config: Dict[str, Any]) -> None:
        """
        Process the configuration passed to the constructor.
        
        Args:
            config: Dictionary with configuration parameters
        """
        # Process configuration parameters
        if 'prompt_variables' in config:
            self.prompt_variables.update(config['prompt_variables'])
        
        # Process use_code_writer parameter
        if 'use_code_writer' in config:
            self.use_code_writer = config['use_code_writer']
            
        # Process current_step_dir parameter
        if 'current_step_dir' in config:
            self.current_step_dir = config['current_step_dir']
            
        # Process workflow_context parameter
        if 'workflow_context' in config:
            self.workflow_context = config['workflow_context']

    async def _generate_code_from_user_input(self, user_input: str) -> str:
        """
        Generate code using the code writer based on user input.
        
        Args:
            user_input: The user's input requesting code generation
            
        Returns:
            Generated code
        """
        if not self._ensure_code_writer():
            return f"Error: Code writer is not available. Cannot generate code from: {user_input}"
        
        result = await self.code_writer.process([user_input])
        
        # Store the generated code for later reference
        self.generated_code = result
        
        return result

    def get_generated_code(self) -> str:
        """
        Get the generated code for the current step.
        
        This method is used by the CreateStep to get the code for the step class file.
        
        Returns:
            The generated code for the step class
        """
        # Return the stored generated code if available
        if hasattr(self, 'generated_code') and self.generated_code:
            return self.generated_code
            
        # Default implementation - return a basic step template
        step_name = os.path.basename(self.current_step_dir) if self.current_step_dir else "CustomStep"
        
        return f'''#!/usr/bin/env python3
"""
{step_name} - A custom step for the NanoBrain framework.

This step was generated by AgentWorkflowBuilder.
"""

from src.Step import Step

class {step_name}(Step):
    """
    {step_name} - A custom step for the NanoBrain framework.
    
    This step was generated by AgentWorkflowBuilder.
    """
    
    def __init__(self, executor, name="{step_name}", **kwargs):
        """
        Initialize the {step_name}.
        
        Args:
            executor: Executor for running the step
            name: Name of the step (default: "{step_name}")
            **kwargs: Additional arguments to pass to the parent class
        """
        super().__init__(executor=executor, name=name, **kwargs)
    
    async def process(self, data_dict):
        """
        Process the input data.
        
        Args:
            data_dict: Dictionary containing the input data
            
        Returns:
            The processed data
        """
        # Create a copy of the input data
        result = data_dict.copy() if isinstance(data_dict, dict) else {{"input": data_dict}}
        
        # Add a processed flag
        result['processed_by'] = "{step_name}"
        
        return result
'''
    
    def get_generated_config(self) -> str:
        """
        Get the generated configuration for the current step.
        
        This method is used by the CreateStep to get the configuration for the step.
        
        Returns:
            The generated configuration for the step
        """
        # Return the stored generated config if available
        if hasattr(self, 'generated_config') and self.generated_config:
            return self.generated_config
            
        # Default implementation - return a basic config template
        step_name = os.path.basename(self.current_step_dir) if self.current_step_dir else "CustomStep"
        
        return f'''# Configuration for {step_name}
defaults:
  class: src.{step_name}.{step_name}
  name: {step_name}
  executor: ExecutorFunc

metadata:
  description: "A custom step for the NanoBrain framework"
  author: "AgentWorkflowBuilder"
  version: "0.1.0"
  biological_analogy: "Neural circuit processing specific information"
  justification: "Like how dedicated neural circuits process specific types of information, this step performs a specific processing task in the workflow."

validation:
  required:
    - name
    - executor
  optional:
    - debug_mode

examples:
  - name: "Example{step_name}"
    executor: ExecutorFunc
    debug_mode: true
'''
    
    def get_generated_tests(self) -> str:
        """
        Get the generated tests for the current step.
        
        This method is used by the CreateStep to get the tests for the step.
        
        Returns:
            The generated tests for the step
        """
        # Return the stored generated tests if available
        if hasattr(self, 'generated_tests') and self.generated_tests:
            return self.generated_tests
            
        # Default implementation - return a basic test template
        step_name = os.path.basename(self.current_step_dir) if self.current_step_dir else "CustomStep"
        
        return f'''#!/usr/bin/env python3
"""
Tests for {step_name}.

This test file was generated by AgentWorkflowBuilder.
"""

import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Import the step class
from src.{step_name} import {step_name}

class Test{step_name}(unittest.TestCase):
    """Test suite for {step_name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock executor
        self.mock_executor = MagicMock()
        self.mock_executor.can_execute = MagicMock(return_value=True)
        self.mock_executor.execute = AsyncMock(return_value={{"success": True, "result": "test_result"}})
        
        # Create the step
        self.step = {step_name}(executor=self.mock_executor, name="Test{step_name}")
    
    def tearDown(self):
        """Clean up after tests."""
        self.step = None
    
    async def test_process(self):
        """Test the process method."""
        # Create test data
        test_data = {{"input": "test"}}
        
        # Process the data
        result = await self.step.process(test_data)
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertIn("processed_by", result)
        self.assertEqual(result["processed_by"], "{step_name}")
    
    def test_process_sync(self):
        """Synchronous wrapper for test_process."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.test_process())
        finally:
            loop.close()

if __name__ == '__main__':
    unittest.main()
'''

    def list_existing_components(self, component_type: str = None) -> str:
        """
        List existing components that can be reused with custom configurations.
        
        Args:
            component_type: Optional type filter ('step', 'workflow', 'link', 'data_unit', 'trigger')
            
        Returns:
            Formatted list of available components with descriptions
        """
        # Map of component types to file patterns
        type_patterns = {
            'step': ['Step*.yml', '*Step.yml'],
            'workflow': ['Workflow*.yml', '*Workflow.yml'],
            'link': ['Link*.yml', '*Link.yml'],
            'data_unit': ['DataUnit*.yml', '*Unit*.yml'],
            'trigger': ['Trigger*.yml', '*Trigger.yml'],
        }
        
        # Get patterns to search for
        patterns = []
        if component_type and component_type in type_patterns:
            patterns = type_patterns[component_type]
        else:
            # If no type specified, search for all components
            patterns = ['*.yml']
        
        # Find matching configuration files
        config_files = []
        for pattern in patterns:
            config_files.extend(glob.glob(os.path.join(self.default_config_path, pattern)))
        
        # Remove duplicates and sort
        config_files = sorted(list(set(config_files)))
        
        if not config_files:
            return f"No matching components found{' for type ' + component_type if component_type else ''}."
        
        # Extract component information
        components = []
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    continue
                
                file_name = os.path.basename(config_file)
                component_name = os.path.splitext(file_name)[0]
                
                # Extract class name
                class_name = None
                if 'defaults' in config and 'class' in config['defaults']:
                    class_name = config['defaults']['class']
                
                # Extract description
                description = "No description available"
                if 'metadata' in config and 'description' in config['metadata']:
                    description = config['metadata']['description']
                    if isinstance(description, list):
                        description = ' '.join(description)
                    description = description.replace('\n', ' ').strip()
                    # Truncate if too long
                    if len(description) > 150:
                        description = description[:147] + "..."
                
                # Extract key parameters
                key_params = []
                if 'defaults' in config:
                    for key, value in config['defaults'].items():
                        if key != 'class' and key not in ['executor', 'input_unit', 'output_unit', 'trigger']:
                            key_params.append(f"{key}: {value}")
                
                components.append({
                    'name': component_name,
                    'class': class_name,
                    'description': description,
                    'key_params': key_params[:3]  # Show at most 3 key parameters
                })
            except Exception as e:
                if self._debug_mode:
                    print(f"Error processing {config_file}: {e}")
        
        # Format the response
        if not components:
            return f"No components found with usable configurations{' for type ' + component_type if component_type else ''}."
        
        # Build response
        result = f"Available {''+component_type if component_type else ''} components that can be reused with custom configurations:\n\n"
        
        for component in components:
            result += f"### {component['name']}\n"
            result += f"**Class:** {component['class'] or 'Unknown'}\n"
            result += f"**Description:** {component['description']}\n"
            
            if component['key_params']:
                result += "**Key parameters:**\n"
                for param in component['key_params']:
                    result += f"- {param}\n"
            
            result += "\n"
        
        result += "\nTo use any of these components with a custom configuration:\n"
        result += "1. Create a YAML configuration file (e.g., `your_component_name.yml`) in your workflow's `config` directory\n"
        result += "2. Set the `class` field in the `defaults` section to the class name shown above\n"
        result += "3. Customize parameters as needed\n"
        result += "4. Use `ConfigManager.create_instance(configuration_name=\"your_component_name\")` to instantiate\n"
        
        return result