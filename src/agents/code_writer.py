"""
Code Writer Agent for NanoBrain Framework

Specialized agent for generating and writing code files.
"""

import logging
from typing import Any, Dict, Optional
from core.agent import SimpleAgent, AgentConfig

logger = logging.getLogger(__name__)


class CodeWriterAgent(SimpleAgent):
    """
    Specialized agent for generating code and writing files.
    
    This agent can generate code based on natural language descriptions
    and use other agents as tools for file operations.
    
    All configuration including tools is loaded from YAML configuration.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="code_writer",
                description="Specialized agent for generating and writing code",
                model="gpt-4",
                system_prompt="You are a specialized code generation agent.",
                temperature=0.3
            )
        
        super().__init__(config, **kwargs)
        
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process code generation request.
        
        Args:
            input_text: Natural language description of code to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated code or response
        """
        # Process using parent class with tool calling capability
        response = await super().process(input_text, **kwargs)
        
        logger.info(f"CodeWriterAgent processed request: {input_text[:100]}...")
        return response
    
    async def generate_python_function(self, function_name: str, description: str, 
                                     parameters: Optional[Dict[str, str]] = None,
                                     return_type: str = "Any") -> str:
        """
        Generate a Python function based on specifications.
        
        Args:
            function_name: Name of the function
            description: Description of what the function should do
            parameters: Dictionary of parameter names and types
            return_type: Return type annotation
            
        Returns:
            Generated function code
        """
        params_str = ""
        if parameters:
            param_list = [f"{name}: {type_hint}" for name, type_hint in parameters.items()]
            params_str = ", ".join(param_list)
        
        request = f"""Generate a Python function with the following specifications:
- Function name: {function_name}
- Description: {description}
- Parameters: {params_str}
- Return type: {return_type}

Include comprehensive docstring with Args and Returns sections, type hints, and error handling where appropriate."""
        
        return await self.process(request)
    
    async def generate_python_class(self, class_name: str, description: str,
                                  base_classes: Optional[list[str]] = None,
                                  methods: Optional[list[Dict[str, str]]] = None) -> str:
        """
        Generate a Python class based on specifications.
        
        Args:
            class_name: Name of the class
            description: Description of the class purpose
            base_classes: List of base class names
            methods: List of method specifications
            
        Returns:
            Generated class code
        """
        base_str = ""
        if base_classes:
            base_str = f"({', '.join(base_classes)})"
        
        methods_str = ""
        if methods:
            methods_list = []
            for method in methods:
                methods_list.append(f"- {method.get('name', 'method')}: {method.get('description', 'No description')}")
            methods_str = "\n".join(methods_list)
        
        request = f"""Generate a Python class with the following specifications:
- Class name: {class_name}{base_str}
- Description: {description}

Methods to implement:
{methods_str}

Include comprehensive class docstring, type hints for all methods, proper __init__ method, and error handling where appropriate."""
        
        return await self.process(request)
    
    async def generate_nanobrain_step(self, step_name: str, description: str,
                                    input_types: Optional[list[str]] = None,
                                    output_types: Optional[list[str]] = None) -> str:
        """
        Generate a NanoBrain Step class.
        
        Args:
            step_name: Name of the step class
            description: Description of the step's purpose
            input_types: List of input data types
            output_types: List of output data types
            
        Returns:
            Generated step class code
        """
        input_str = ", ".join(input_types) if input_types else "Any"
        output_str = ", ".join(output_types) if output_types else "Any"
        
        request = f"""Generate a NanoBrain Step class with the following specifications:
- Class name: {step_name}
- Description: {description}
- Input types: {input_str}
- Output types: {output_str}

The class should inherit from the Step base class, implement the process method, include proper configuration handling, follow NanoBrain framework patterns, include comprehensive docstrings, and handle errors gracefully."""
        
        return await self.process(request)
    
    async def write_code_to_file(self, code: str, file_path: str, 
                               description: Optional[str] = None) -> str:
        """
        Write generated code to a file using available tools.
        
        Args:
            code: Code content to write
            file_path: Path where to save the file
            description: Optional description of the file
            
        Returns:
            Result of the file writing operation
        """
        request = f"""Please save the following code to {file_path}:

{description or 'Generated code'}

Code:
```
{code}
```

File path: {file_path}"""
        
        return await self.process(request)
    
 