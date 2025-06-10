"""
Code Writer Agent for NanoBrain Framework

Specialized agent for generating and writing code files.
"""

import logging
from typing import Any, Dict, Optional, List

# Updated imports for nanobrain package structure  
from nanobrain.core.agent import AgentConfig
from nanobrain.core.logging_system import get_logger
from .base import SpecializedAgentBase, SimpleSpecializedAgent, ConversationalSpecializedAgent

logger = logging.getLogger(__name__)


class CodeWriterAgentMixin(SpecializedAgentBase):
    """
    Mixin class providing code generation capabilities for specialized agents.
    
    This agent can generate code based on natural language descriptions
    and use other agents as tools for file operations.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Code generation keywords
        self.code_keywords = [
            'code', 'function', 'class', 'method', 'script', 'program',
            'generate', 'create', 'write', 'implement', 'develop'
        ]
        
        # Programming language patterns
        self.language_patterns = {
            'python': ['python', 'py', 'def ', 'class ', 'import '],
            'javascript': ['javascript', 'js', 'function ', 'const ', 'let '],
            'java': ['java', 'public class', 'private ', 'public '],
            'cpp': ['c++', 'cpp', '#include', 'int main', 'std::'],
            'html': ['html', '<html>', '<div>', '<body>'],
            'css': ['css', 'style', '{', '}', 'color:', 'font-']
        }
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize code writer specific features."""
        await super()._initialize_specialized_features()
        self.specialized_logger.info("CodeWriter specialized features initialized")
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by code generation logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by code generation logic
        """
        # Check for explicit code generation parameters
        has_code_params = any(param in kwargs for param in ['function_name', 'class_name', 'language'])
        
        # Check for code generation keywords in input
        has_keywords = any(keyword in input_text.lower() for keyword in self.code_keywords)
        
        # Check for programming language indicators
        has_language_indicators = any(
            any(pattern in input_text.lower() for pattern in patterns)
            for patterns in self.language_patterns.values()
        )
        
        return has_code_params or (has_keywords and has_language_indicators)
    
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process code generation requests directly when possible.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Generated code if handled, None if should fall back to LLM
        """
        try:
            # Check for specific code generation requests
            if 'function_name' in kwargs:
                return await self._generate_function_from_params(**kwargs)
            
            if 'class_name' in kwargs:
                return await self._generate_class_from_params(**kwargs)
            
            # Try to parse simple code generation requests
            result = await self._try_parse_code_request(input_text)
            if result:
                return result
            
            return None
            
        except Exception as e:
            self.specialized_logger.error(f"Error in specialized code processing: {e}")
            return None
    
    async def _generate_function_from_params(self, **kwargs) -> str:
        """Generate a function from explicit parameters."""
        function_name = kwargs.get('function_name', 'generated_function')
        description = kwargs.get('description', 'Generated function')
        parameters = kwargs.get('parameters', {})
        return_type = kwargs.get('return_type', 'Any')
        
        # Generate parameter string
        params_str = ""
        if parameters:
            param_list = [f"{name}: {type_hint}" for name, type_hint in parameters.items()]
            params_str = ", ".join(param_list)
        
        # Generate function code
        code = f'''def {function_name}({params_str}) -> {return_type}:
    """
    {description}
    
    Args:
        {chr(10).join(f"{name}: {desc}" for name, desc in parameters.items()) if parameters else "None"}
    
    Returns:
        {return_type}: Generated return value
    """
    # TODO: Implement function logic
    pass'''
        
        self._track_specialized_operation("generate_function", success=True)
        return code
    
    async def _generate_class_from_params(self, **kwargs) -> str:
        """Generate a class from explicit parameters."""
        class_name = kwargs.get('class_name', 'GeneratedClass')
        description = kwargs.get('description', 'Generated class')
        base_classes = kwargs.get('base_classes', [])
        methods = kwargs.get('methods', [])
        
        # Generate inheritance string
        base_str = ""
        if base_classes:
            base_str = f"({', '.join(base_classes)})"
        
        # Generate class code
        code = f'''class {class_name}{base_str}:
    """
    {description}
    """
    
    def __init__(self):
        """Initialize the {class_name}."""
        super().__init__()
'''
        
        # Add methods if specified
        for method in methods:
            method_name = method.get('name', 'method')
            method_desc = method.get('description', 'Generated method')
            code += f'''
    def {method_name}(self):
        """
        {method_desc}
        """
        # TODO: Implement method logic
        pass
'''
        
        self._track_specialized_operation("generate_class", success=True)
        return code
    
    async def _try_parse_code_request(self, input_text: str) -> Optional[str]:
        """Try to parse simple code generation requests."""
        input_lower = input_text.lower()
        
        # Simple function generation
        if 'function' in input_lower and ('called' in input_lower or 'named' in input_lower):
            import re
            func_match = re.search(r'function.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)', input_text, re.IGNORECASE)
            if func_match:
                func_name = func_match.group(1)
                code = f'''def {func_name}():
    """Generated function: {func_name}"""
    # TODO: Implement function logic
    pass'''
                self._track_specialized_operation("parse_function", success=True)
                return code
        
        # Simple class generation
        if 'class' in input_lower and ('called' in input_lower or 'named' in input_lower):
            import re
            class_match = re.search(r'class.*?(?:called|named)\s+([a-zA-Z_][a-zA-Z0-9_]*)', input_text, re.IGNORECASE)
            if class_match:
                class_name = class_match.group(1)
                code = f'''class {class_name}:
    """Generated class: {class_name}"""
    
    def __init__(self):
        """Initialize the {class_name}."""
        pass'''
                self._track_specialized_operation("parse_class", success=True)
                return code
        
        return None
    
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
        try:
            result = await self._generate_function_from_params(
                function_name=function_name,
                description=description,
                parameters=parameters or {},
                return_type=return_type
            )
            self._track_specialized_operation("generate_python_function", success=True)
            return result
        except Exception as e:
            self._track_specialized_operation("generate_python_function", success=False)
            self.specialized_logger.error(f"Error generating Python function: {e}")
            raise
    
    async def generate_python_class(self, class_name: str, description: str,
                                  base_classes: Optional[List[str]] = None,
                                  methods: Optional[List[Dict[str, str]]] = None) -> str:
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
        try:
            result = await self._generate_class_from_params(
                class_name=class_name,
                description=description,
                base_classes=base_classes or [],
                methods=methods or []
            )
            self._track_specialized_operation("generate_python_class", success=True)
            return result
        except Exception as e:
            self._track_specialized_operation("generate_python_class", success=False)
            self.specialized_logger.error(f"Error generating Python class: {e}")
            raise
    
    async def generate_nanobrain_step(self, step_name: str, description: str,
                                    input_types: Optional[List[str]] = None,
                                    output_types: Optional[List[str]] = None) -> str:
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
        try:
            input_str = ", ".join(input_types) if input_types else "Any"
            output_str = ", ".join(output_types) if output_types else "Any"
            
            code = f'''from typing import Any, Dict
from nanobrain.core.step import Step, StepConfig


class {step_name}(Step):
    """
    {description}
    
    Input types: {input_str}
    Output types: {output_str}
    """
    
    def __init__(self, config: StepConfig):
        """Initialize the {step_name}."""
        super().__init__(config)
    
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the step inputs.
        
        Args:
            inputs: Input data dictionary
            
        Returns:
            Output data dictionary
        """
        # TODO: Implement step processing logic
        return {{"output": inputs.get("input", "")}}
'''
            
            self._track_specialized_operation("generate_nanobrain_step", success=True)
            return code
        except Exception as e:
            self._track_specialized_operation("generate_nanobrain_step", success=False)
            self.specialized_logger.error(f"Error generating NanoBrain step: {e}")
            raise
    
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
        try:
            # This will use the LLM to handle file writing via tools
            request = f"""Please save the following code to {file_path}:

{description or 'Generated code'}

Code:
```
{code}
```

File path: {file_path}"""
            
            # Fall back to LLM processing for file operations
            if hasattr(self, 'process') and hasattr(super(), 'process'):
                result = await super().process(request)
                self._track_specialized_operation("write_code_to_file", success=True)
                return result
            else:
                self._track_specialized_operation("write_code_to_file", success=False)
                return f"Error: Cannot write code to file - no processing method available"
                
        except Exception as e:
            self._track_specialized_operation("write_code_to_file", success=False)
            self.specialized_logger.error(f"Error writing code to file: {e}")
            return f"Error writing code to file: {str(e)}"


class CodeWriterAgent(CodeWriterAgentMixin, SimpleSpecializedAgent):
    """
    Simple code writer agent that generates code without conversation history.
    
    This agent can generate code based on natural language descriptions
    and use other agents as tools for file operations.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="code_writer",
                description="Specialized agent for generating and writing code",
                model="gpt-4",
                system_prompt="""You are a specialized code generation agent for the NanoBrain framework.

Your capabilities:
1. Generate high-quality code based on natural language descriptions
2. Create functions, classes, and complete programs
3. Follow best practices and coding standards
4. Include comprehensive documentation and type hints
5. Handle multiple programming languages

When generating code:
1. Always include proper documentation and docstrings
2. Use appropriate type hints where applicable
3. Follow language-specific best practices
4. Include error handling where appropriate
5. Make code readable and maintainable

You can generate code for various purposes including:
- Python functions and classes
- NanoBrain framework components (Steps, Agents, Tools)
- Web applications and APIs
- Data processing scripts
- Utility functions and helpers""",
                temperature=0.3
            )
        
        super().__init__(config=config, **kwargs)


class ConversationalCodeWriterAgent(CodeWriterAgentMixin, ConversationalSpecializedAgent):
    """
    Conversational code writer agent that maintains conversation history.
    
    This agent can generate code based on natural language descriptions
    while maintaining conversation context for iterative development.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Use provided config or create minimal default
        if config is None:
            config = AgentConfig(
                name="conversational_code_writer",
                description="Conversational specialized agent for code generation",
                model="gpt-4",
                system_prompt="""You are a conversational code generation agent for the NanoBrain framework.

You maintain conversation context while generating code, allowing for iterative development and refinement.

Your capabilities:
1. Generate high-quality code based on natural language descriptions
2. Refine and improve code based on feedback
3. Maintain context across multiple code generation requests
4. Explain code and provide implementation guidance
5. Handle complex, multi-step development tasks

When working conversationally:
1. Remember previous code generations and modifications
2. Build upon earlier work in the conversation
3. Provide explanations and rationale for code decisions
4. Ask clarifying questions when requirements are unclear
5. Suggest improvements and alternatives

You can help with:
- Iterative code development and refinement
- Code reviews and improvements
- Complex multi-file projects
- Learning and educational coding assistance
- Debugging and troubleshooting""",
                temperature=0.4  # Slightly higher for conversational creativity
            )
        
        super().__init__(config=config, **kwargs)
 