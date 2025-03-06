from typing import List, Any, Optional, Dict, Union
import os
import asyncio
from pathlib import Path

from src.Step import Step
from src.ExecutorBase import ExecutorBase
from src.Agent import Agent


class StepCoder(Step):
    """
    Tool for generating software code based on requirements.
    
    Biological analogy: Language generation areas of the brain.
    Justification: Like how language areas generate coherent sentences
    based on concepts, this tool generates coherent code based on requirements.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.model_name = kwargs.get('model_name', "gpt-3.5-turbo")
        self.model_class = kwargs.get('model_class', None)
        self.temperature = kwargs.get('temperature', 0.2)  # Lower temperature for more focused code generation
        self.max_tokens = kwargs.get('max_tokens', 4000)
        
        # Create the coding agent
        self.coding_agent = Agent(
            executor=executor,
            model_name=self.model_name,
            model_class=self.model_class,
            prompt_template="You are an expert software developer focusing on generating clean, well-documented code. Your task is to generate code based on the requirements provided. Follow best practices and document your code thoroughly. Always include proper error handling and edge cases. The code should be modular, maintainable, and follow the principles of clean code.\n\nRequirements: {input}",
            **kwargs
        )
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Generate code based on requirements.
        
        Args:
            inputs: List containing:
                - requirements: Description of the code to generate
                - language: Programming language (optional, defaults to Python)
                - context: Additional context or code snippets (optional)
        
        Returns:
            Dictionary with the generated code
        """
        # Extract inputs
        if not inputs or len(inputs) < 1:
            return {
                "success": False,
                "error": "Missing required input: requirements"
            }
        
        requirements = inputs[0]
        language = inputs[1] if len(inputs) > 1 else "Python"
        context = inputs[2] if len(inputs) > 2 else None
        
        # Format the requirements with language and context
        formatted_requirements = f"Generate {language} code that meets the following requirements:\n\n{requirements}"
        
        if context:
            formatted_requirements += f"\n\nAdditional context or related code:\n{context}"
        
        try:
            # Generate code using the coding agent
            generated_code = await self.coding_agent.process([formatted_requirements])
            
            # Return the generated code
            return {
                "success": True,
                "code": generated_code,
                "language": language
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to generate code: {e}"
            }
    
    async def generate_class(self, class_name: str, base_class: str, description: str, methods: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate a Python class with the specified methods.
        
        Args:
            class_name: Name of the class to generate
            base_class: Name of the base class to inherit from
            description: Description of the class
            methods: List of method descriptions
        
        Returns:
            Dictionary with the generated class code
        """
        # Format the requirements for a class
        requirements = f"""
Create a Python class named {class_name} that inherits from {base_class}.

Class description:
{description}

The class should include the following methods:
"""

        # Add method descriptions
        for method in methods:
            method_name = method.get('name', '')
            method_desc = method.get('description', '')
            method_params = method.get('parameters', '')
            method_returns = method.get('returns', '')
            
            requirements += f"\n- Method: {method_name}\n"
            requirements += f"  Description: {method_desc}\n"
            
            if method_params:
                requirements += f"  Parameters: {method_params}\n"
            
            if method_returns:
                requirements += f"  Returns: {method_returns}\n"
        
        # Generate the class code
        return await self.process([requirements])
    
    async def generate_function(self, function_name: str, description: str, parameters: str, returns: str) -> Dict[str, Any]:
        """
        Generate a Python function with the specified parameters.
        
        Args:
            function_name: Name of the function to generate
            description: Description of the function
            parameters: Description of the function parameters
            returns: Description of the function return value
        
        Returns:
            Dictionary with the generated function code
        """
        # Format the requirements for a function
        requirements = f"""
Create a Python function named {function_name}.

Function description:
{description}

Parameters:
{parameters}

Returns:
{returns}

The function should include proper error handling and documentation.
"""
        
        # Generate the function code
        return await self.process([requirements])
    
    async def refactor_code(self, code: str, requirements: str) -> Dict[str, Any]:
        """
        Refactor existing code based on requirements.
        
        Args:
            code: Existing code to refactor
            requirements: Requirements for the refactoring
        
        Returns:
            Dictionary with the refactored code
        """
        # Format the requirements for refactoring
        formatted_requirements = f"""
Refactor the following code according to these requirements:
{requirements}

Original code:
```
{code}
```

Provide the refactored code with explanations of the changes made.
"""
        
        # Generate the refactored code
        return await self.process([formatted_requirements]) 