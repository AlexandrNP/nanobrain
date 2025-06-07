"""
Code Writer Agent for NanoBrain Framework

Specialized agent for generating and writing code files.
"""

import logging
from typing import Any, Dict, Optional
from ..core.agent import SimpleAgent, AgentConfig

logger = logging.getLogger(__name__)


class CodeWriterAgent(SimpleAgent):
    """
    Specialized agent for generating code and writing files.
    
    This agent can generate code based on natural language descriptions
    and use other agents as tools for file operations.
    
    All prompts and templates are loaded from YAML configuration.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, **kwargs):
        # Set default configuration if not provided
        if config is None:
            config = AgentConfig(
                name="code_writer",
                description="Specialized agent for generating and writing code",
                model="gpt-4",
                system_prompt=self._get_default_system_prompt(),
                temperature=0.3,
                tools=[]
            )
        else:
            # If config is provided but system_prompt is empty, use default
            if not config.system_prompt or config.system_prompt.strip() == "":
                config.system_prompt = self._get_default_system_prompt()
        
        super().__init__(config, **kwargs)
        
        # Load prompt templates from configuration
        self.prompt_templates = self._load_prompt_templates()
        
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for CodeWriterAgent."""
        # All prompts should be loaded from YAML configuration
        # No hardcoded defaults
        return ""

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from configuration."""
        # All prompt templates must be loaded from YAML configuration
        if hasattr(self.config, 'prompt_templates') and self.config.prompt_templates:
            return self.config.prompt_templates
        
        # No fallback defaults - all prompts must come from YAML
        logger.warning("No prompt templates found in configuration. All prompts should be defined in YAML.")
        return {}

    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process code generation request.
        
        Args:
            input_text: Natural language description of code to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated code or response
        """
        # Use enhanced input template from configuration
        enhanced_input = self.prompt_templates["enhanced_input"].format(
            input_text=input_text,
            available_tools=', '.join(self.available_tools)
        )
        
        # Process using parent class
        response = await super().process(enhanced_input, **kwargs)
        
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
        
        # Use python_function template from configuration
        request = self.prompt_templates["python_function"].format(
            function_name=function_name,
            description=description,
            parameters=params_str,
            return_type=return_type
        )
        
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
        
        # Use python_class template from configuration
        request = self.prompt_templates["python_class"].format(
            class_name=class_name,
            base_classes=base_str,
            description=description,
            methods=methods_str
        )
        
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
        
        # Use nanobrain_step template from configuration
        request = self.prompt_templates["nanobrain_step"].format(
            step_name=step_name,
            description=description,
            input_types=input_str,
            output_types=output_str
        )
        
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
        if not self.available_tools:
            return "No file writing tools available. Code generated but not saved."
        
        # Use the first available tool (assuming it's a file writer)
        tool_name = self.available_tools[0]
        
        # Use write_code_to_file template from configuration
        request = self.prompt_templates["write_code_to_file"].format(
            tool_name=tool_name,
            file_path=file_path,
            description=description or 'Generated code',
            code=code
        )
        
        return await self.process(request)
    
 