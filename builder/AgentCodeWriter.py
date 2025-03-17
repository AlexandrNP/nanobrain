#!/usr/bin/env python3
"""
AgentCodeWriter - Specialized agent for generating NanoBrain code.

This module provides an agent that specializes in generating code for the
NanoBrain framework, including steps, workflows, links, and other components,
following best practices and architectural patterns.

Biological analogy: Motor cortex producing precise movement patterns.
Justification: Like how the motor cortex generates precisely structured
movement commands based on stored patterns, this agent generates precisely
structured code based on learned code patterns and best practices.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import os
import asyncio
import re
import importlib
from pathlib import Path
import yaml
import glob

from src.Agent import Agent
from src.ExecutorBase import ExecutorBase
from src.enums import ComponentState
from src.ConfigManager import ConfigManager

# Import prompt templates
from builder.prompts import (
    CODE_WRITER_PROMPT,
    CODE_WRITER_STEP_CONTEXT,
    CODE_WRITER_WORKFLOW_CONTEXT,
    CODE_WRITER_LINK_CONTEXT,
    CODE_WRITER_DATA_UNIT_CONTEXT,
    CODE_WRITER_TRIGGER_CONTEXT
)

class AgentCodeWriter(Agent):
    """
    Specialized agent for generating NanoBrain framework code.
    
    This agent is designed to:
    1. Generate code for NanoBrain components (steps, workflows, links, etc.)
    2. Follow NanoBrain architectural patterns and best practices
    3. Include proper biological analogies and documentation
    4. Create testable and maintainable implementations
    
    Biological analogy: Motor cortex producing precise movement patterns.
    Justification: Like how the motor cortex generates precise sequences of
    movement commands based on stored patterns, this agent generates precisely
    structured code based on learned code patterns and best practices.
    """
    
    def __init__(self, executor: ExecutorBase, system_prompt: str = None, 
                 model: str = None, max_tokens: int = 2000, temperature: float = 0.5, 
                 _debug_mode: bool = False, **kwargs):
        """
        Initialize the AgentCodeWriter.
        
        Args:
            executor: Executor for running the agent
            system_prompt: System prompt for the agent (optional)
            model: Model to use for the agent (optional)
            max_tokens: Maximum number of tokens for the model (default: 2000)
            temperature: Temperature for the model (default: 0.5)
            _debug_mode: Whether to run in debug mode (default: False)
            **kwargs: Additional arguments to pass to the parent class
        """
        # Store executor as instance attribute
        self.executor = executor
        
        # Store model name
        self.model = model or "gpt-4"
        
        # Store other parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._debug_mode = _debug_mode
        
        # Load prompts from configuration or use defaults
        self.use_prompt_file = kwargs.get('use_prompt_file', False)
        self.prompt_file = kwargs.get('prompt_file', 'builder.prompts')
        self.prompt_template = kwargs.get('prompt_template', 'CODE_WRITER_PROMPT')
        
        # Context templates for different code types
        self.code_context_templates = kwargs.get('code_context_templates', {})
        if not self.code_context_templates:
            self.code_context_templates = {
                'step': 'CODE_WRITER_STEP_CONTEXT',
                'workflow': 'CODE_WRITER_WORKFLOW_CONTEXT',
                'link': 'CODE_WRITER_LINK_CONTEXT',
                'data_unit': 'CODE_WRITER_DATA_UNIT_CONTEXT',
                'trigger': 'CODE_WRITER_TRIGGER_CONTEXT'
            }
        
        # Get prompts either from external file or default constants
        if self.use_prompt_file and self.prompt_file and self.prompt_template:
            try:
                # Try to import the specified prompt file
                prompt_module = importlib.import_module(self.prompt_file)
                system_prompt = getattr(prompt_module, self.prompt_template)
                
                # Load context templates
                self.context_templates = {}
                for code_type, template_name in self.code_context_templates.items():
                    try:
                        self.context_templates[code_type] = getattr(prompt_module, template_name)
                    except AttributeError:
                        if _debug_mode:
                            print(f"Warning: Context template {template_name} not found in {self.prompt_file}")
                
            except (ImportError, AttributeError) as e:
                if _debug_mode:
                    print(f"Failed to load prompts from file: {e}. Using defaults.")
                system_prompt = CODE_WRITER_PROMPT
                # Use default context templates
                self.context_templates = self._get_default_context_templates()
        else:
            # Use the built-in prompts
            if system_prompt is None:
                system_prompt = CODE_WRITER_PROMPT
            # Use default context templates
            self.context_templates = self._get_default_context_templates()
        
        # Set system_prompt as instance attribute
        self.system_prompt = system_prompt or CODE_WRITER_PROMPT
        
        # Initialize the config manager for finding existing classes
        self.config_manager = kwargs.get('config_manager', ConfigManager(base_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Default paths for searching configuration files
        self.default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'default_configs')
        
        # Initialize the parent class after setting up our own attributes
        super().__init__(
            executor=executor,
            system_prompt=self.system_prompt,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            _debug_mode=self._debug_mode,
            **kwargs
        )
        
        # Track recently generated code for context
        self.recent_code = []
        self.max_recent_code = 3
    
    def _get_default_context_templates(self) -> Dict[str, str]:
        """
        Get the default context templates for each code type.
        
        Returns:
            Dictionary mapping code types to their context templates
        """
        return {
            'step': CODE_WRITER_STEP_CONTEXT,
            'workflow': CODE_WRITER_WORKFLOW_CONTEXT,
            'link': CODE_WRITER_LINK_CONTEXT,
            'data_unit': CODE_WRITER_DATA_UNIT_CONTEXT,
            'trigger': CODE_WRITER_TRIGGER_CONTEXT
        }
    
    async def _safe_execute(self, executor, messages, max_tokens=None, temperature=None, debug_mode=False):
        """
        Safely execute a prompt using the provided executor with fallback options.
        
        Args:
            executor: The executor to use
            messages: The messages to process
            max_tokens: Maximum tokens for response (optional)
            temperature: Temperature for response (optional)
            debug_mode: Whether to enable debug mode (optional)
            
        Returns:
            The executor's response or an error message
        """
        if executor is None:
            return "Error: Executor is not available."
            
        # Try different approaches to execute with the executor
        try:
            # Try the simplest approach first - just pass messages
            return await executor.execute(messages)
        except TypeError as e:
            if debug_mode:
                print(f"First execution attempt failed: {e}, trying with parameters...")
            
            try:
                # Try with parameters dictionary
                params = {"messages": messages}
                if max_tokens is not None:
                    params["max_tokens"] = max_tokens
                if temperature is not None:
                    params["temperature"] = temperature
                
                # Convert to keyword arguments
                return await executor.execute(**params)
            except Exception as e2:
                if debug_mode:
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
                            return await executor.execute(user_message)
                    return await executor.execute(str(messages))
                except Exception as e3:
                    if debug_mode:
                        print(f"All execution attempts failed: {e3}")
                    return f"Error processing request: {str(e3)}"

    async def process(self, input_data: List[Any]) -> Any:
        """
        Process input and generate NanoBrain framework code.
        
        This method:
        1. Analyzes the user's request to understand what code to generate
        2. Applies best practices for the NanoBrain framework
        3. Generates well-structured, documented code with biological analogies
        
        Args:
            input_data: List of input data items (typically user's code requests)
        
        Returns:
            Generated code for the NanoBrain framework
        """
        # Extract the input from the list
        if not input_data or not isinstance(input_data, list):
            return "Error: Input must be a non-empty list."
        
        user_input = input_data[0]
        
        # Determine code type to generate
        code_type = self._determine_code_type(user_input)
        
        # Add appropriate context based on code type
        context = self._get_context_for_code_type(code_type)
        
        # Include recent code for context continuity
        recent_code_context = self._get_recent_code_context()
        
        # Process the input using the LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": context}
        ]
        
        # Add recent code context if available
        if recent_code_context:
            messages.append({"role": "system", "content": recent_code_context})
        
        # Add the user's input
        messages.append({"role": "user", "content": user_input})
        
        # Get the response using our safe execute method
        response = await self._safe_execute(
            self.executor, 
            messages, 
            self.max_tokens, 
            self.temperature, 
            self._debug_mode
        )
        
        # Add the generated code to recent code for future context
        self._update_recent_code(code_type, response)
        
        # Extract and clean code if it's in a markdown code block
        cleaned_code = self._extract_code(response)
        
        return cleaned_code
    
    def _determine_code_type(self, user_input: str) -> str:
        """
        Determine what type of NanoBrain code to generate.
        
        Args:
            user_input: User's input request
        
        Returns:
            Code type (step, workflow, link, etc.)
        """
        # Check for step-related patterns
        if any(pattern in user_input.lower() for pattern in ['step', 'processor', 'component']):
            return 'step'
        
        # Check for workflow-related patterns
        elif any(pattern in user_input.lower() for pattern in ['workflow', 'pipeline', 'process']):
            return 'workflow'
        
        # Check for link-related patterns
        elif any(pattern in user_input.lower() for pattern in ['link', 'connection', 'connect']):
            return 'link'
        
        # Check for data unit patterns
        elif any(pattern in user_input.lower() for pattern in ['data unit', 'dataunit', 'storage']):
            return 'data_unit'
        
        # Check for trigger patterns
        elif any(pattern in user_input.lower() for pattern in ['trigger', 'event', 'callback']):
            return 'trigger'
        
        # Default to step as the most common type
        return 'step'
    
    def _get_context_for_code_type(self, code_type: str) -> str:
        """
        Get specialized context for the specific code type.
        
        Args:
            code_type: Type of code to generate
        
        Returns:
            Specialized context for the code type
        """
        # Use the context templates loaded from config or defaults
        return self.context_templates.get(code_type, self.context_templates['step'])
    
    def _get_recent_code_context(self) -> str:
        """
        Get context from recently generated code.
        
        Returns:
            Context from recent code snippets
        """
        if not self.recent_code:
            return ""
        
        context = "## RECENTLY GENERATED CODE (FOR REFERENCE)\n\n"
        for code_type, code in self.recent_code:
            # Only include a preview of the code to avoid token issues
            preview = code.split("\n")[:10]
            preview = "\n".join(preview)
            if len(preview) < len(code):
                preview += "\n# ... [additional code omitted for brevity]"
            
            context += f"### {code_type.upper()} CODE\n```python\n{preview}\n```\n\n"
        
        return context
    
    def _update_recent_code(self, code_type: str, code: str) -> None:
        """
        Update the recent code history.
        
        Args:
            code_type: Type of code generated
            code: Generated code
        """
        # Extract code from markdown if needed
        cleaned_code = self._extract_code(code)
        
        # Add to recent code list
        self.recent_code.append((code_type, cleaned_code))
        
        # Keep only the most recent code snippets
        if len(self.recent_code) > self.max_recent_code:
            self.recent_code = self.recent_code[-self.max_recent_code:]
    
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
    
    def _find_existing_class(self, component_type: str, requirements: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Find an existing class that meets the specified requirements.
        
        Args:
            component_type: Type of component (step, workflow, link, data_unit, trigger)
            requirements: Description of the requirements for the component
            
        Returns:
            Tuple containing class name and configuration dict if found, or (None, None) if not
        """
        if self._debug_mode:
            print(f"Searching for existing {component_type} class that meets requirements: {requirements}")
        
        # Map component types to common base classes and directories to search
        component_map = {
            'step': {'base_classes': ['Step'], 'search_dirs': ['src']},
            'workflow': {'base_classes': ['Workflow'], 'search_dirs': ['src']},
            'link': {'base_classes': ['LinkBase', 'LinkDirect'], 'search_dirs': ['src']},
            'data_unit': {'base_classes': ['DataUnitBase', 'DataUnitMemory', 'DataUnitFile'], 'search_dirs': ['src']},
            'trigger': {'base_classes': ['TriggerBase', 'TriggerDataUpdated'], 'search_dirs': ['src']}
        }
        
        # Get relevant base classes and search directories
        base_classes = component_map.get(component_type, {}).get('base_classes', [])
        search_dirs = component_map.get(component_type, {}).get('search_dirs', ['src'])
        
        # Find configuration files for this component type
        config_files = []
        for pattern in [f'*{base_class}*.yml' for base_class in base_classes]:
            config_files.extend(glob.glob(os.path.join(self.default_config_path, pattern)))
        
        # Add all yml files in default_configs directory
        config_files.extend(glob.glob(os.path.join(self.default_config_path, '*.yml')))
        config_files = list(set(config_files))  # Remove duplicates
        
        if not config_files and self._debug_mode:
            print(f"No configuration files found for {component_type}")
            return None, None
        
        # Analyze each configuration file to see if it meets requirements
        config_matches = []
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    continue
                
                # Extract class name and metadata
                class_name = None
                description = ""
                if 'defaults' in config and 'class' in config['defaults']:
                    class_name = config['defaults']['class'].split('.')[-1]
                
                if 'metadata' in config and 'description' in config['metadata']:
                    description = config['metadata']['description']
                elif 'metadata' in config and 'objective' in config['metadata']:
                    description = config['metadata']['objective']
                
                # Skip if no class name or description
                if not class_name or not description:
                    continue
                
                # Calculate a simple relevance score based on keywords
                relevance_score = self._calculate_relevance(requirements, description)
                config_matches.append({
                    'class_name': class_name,
                    'config_file': os.path.basename(config_file),
                    'description': description,
                    'score': relevance_score,
                    'full_config': config
                })
            except Exception as e:
                if self._debug_mode:
                    print(f"Error processing config file {config_file}: {e}")
        
        # Sort by relevance score
        config_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Return the most relevant match if it has a good enough score
        if config_matches and config_matches[0]['score'] > 0.3:  # Threshold for relevance
            return config_matches[0]['class_name'], config_matches[0]['full_config']
        
        return None, None

    def _calculate_relevance(self, requirements: str, description: str) -> float:
        """
        Calculate relevance score of a description to the requirements.
        
        Args:
            requirements: The requirements description
            description: The component description
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Simple keyword-based relevance scoring
        # This could be improved with more sophisticated NLP techniques
        req_words = set(re.findall(r'\w+', requirements.lower()))
        desc_words = set(re.findall(r'\w+', description.lower()))
        
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'for', 'with', 'on', 'by', 'is', 'that', 'be'}
        req_words = req_words - stop_words
        desc_words = desc_words - stop_words
        
        if not req_words:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(req_words.intersection(desc_words))
        union = len(req_words.union(desc_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_config_suggestion(self, class_name: str, base_config: Dict, requirements: str) -> Dict:
        """
        Create a suggested configuration for an existing class based on requirements.
        
        Args:
            class_name: Name of the class to configure
            base_config: Base configuration for the class
            requirements: Requirements that the configuration should meet
            
        Returns:
            Dictionary with suggested configuration
        """
        # Start with defaults section
        suggested_config = {'defaults': {}}
        
        if 'defaults' in base_config:
            # Copy class field and other essential fields
            for key, value in base_config['defaults'].items():
                if key == 'class' or key in ['executor', 'input_unit', 'output_unit', 'trigger']:
                    suggested_config['defaults'][key] = value
        
        # Extract parameter suggestions from requirements using basic pattern matching
        # This is a simplified implementation - in a real system, consider using LLM reasoning
        
        # Common parameter patterns to look for in requirements
        param_patterns = {
            'persistence_level': r'persist(ence|ent).*(\d+(\.\d+)?)',
            'reliability': r'reliab(le|ility).*(\d+(\.\d+)?)',
            'capacity': r'capacity.*(\d+)',
            'sensitivity': r'sensitiv(e|ity).*(\d+(\.\d+)?)',
            'threshold': r'threshold.*(\d+(\.\d+)?)'
        }
        
        for param, pattern in param_patterns.items():
            match = re.search(pattern, requirements, re.IGNORECASE)
            if match and match.group(2):
                try:
                    value = float(match.group(2))
                    # Only add if the param exists in the base config
                    if 'defaults' in base_config and param in base_config['defaults']:
                        suggested_config['defaults'][param] = value
                except ValueError:
                    pass
        
        return suggested_config

    async def generate_step(self, step_name: str, base_class: str = "Step", description: str = None) -> str:
        """
        Generate a complete Step implementation or suggest an existing one.
        
        Args:
            step_name: Name of the step
            base_class: Base class for the step (default: "Step")
            description: Description of what the step should do (optional)
        
        Returns:
            Complete Step implementation or configuration suggestion
        """
        # First, check if an existing class can be used
        existing_class, config = self._find_existing_class('step', description or "")
        
        if existing_class and config:
            # Suggest using the existing class with configuration
            suggested_config = self._create_config_suggestion(existing_class, config, description or "")
            
            return f"""Instead of creating a new Step class, you can use the existing `{existing_class}` class with a custom configuration:

```yaml
# Configuration for {step_name} using {existing_class}
{yaml.dump(suggested_config, default_flow_style=False)}
```

To use this configuration:
1. Save it to a file, e.g., `{step_name}.yml` in your workflow's `config` directory
2. Create an instance using the ConfigManager:

```python
from src.ConfigManager import ConfigManager

config_manager = ConfigManager(base_path="your_workflow_path")
{step_name.lower()} = config_manager.create_instance(configuration_name="{step_name}")
```

This approach follows the NanoBrain recommendation to reuse existing components when possible.
"""
        
        # If no suitable existing class, generate a new one
        # Build a prompt for generating the step
        prompt = f"""Generate a complete implementation for a NanoBrain step class named '{step_name}' that inherits from '{base_class}'.
        
        {f"Description: {description}" if description else ""}
        
        The step should include:
        - Appropriate imports
        - Class docstring with biological analogy
        - Constructor with typed parameters
        - Implementation of required methods (process, get_state, etc.)
        - Any necessary helper methods
        - Comprehensive error handling
        
        Follow NanoBrain framework conventions and patterns.
        """
        
        # Generate the step implementation
        return await self.process([prompt])

    async def generate_workflow(self, workflow_name: str, steps: List[str] = None, description: str = None) -> str:
        """
        Generate a complete Workflow implementation or suggest an existing one.
        
        Args:
            workflow_name: Name of the workflow
            steps: List of step names to include (optional)
            description: Description of what the workflow should do (optional)
        
        Returns:
            Complete Workflow implementation or configuration suggestion
        """
        # First, check if an existing class can be used
        existing_class, config = self._find_existing_class('workflow', description or "")
        
        if existing_class and config:
            # Suggest using the existing class with configuration
            suggested_config = self._create_config_suggestion(existing_class, config, description or "")
            
            # Add steps to the configuration if provided
            if steps and 'defaults' in suggested_config:
                suggested_config['defaults']['steps'] = steps
            
            return f"""Instead of creating a new Workflow class, you can use the existing `{existing_class}` class with a custom configuration:

```yaml
# Configuration for {workflow_name} using {existing_class}
{yaml.dump(suggested_config, default_flow_style=False)}
```

To use this configuration:
1. Save it to a file, e.g., `{workflow_name}.yml` in your workflow's `config` directory
2. Create an instance using the ConfigManager:

```python
from src.ConfigManager import ConfigManager

config_manager = ConfigManager(base_path="your_workflow_path")
{workflow_name.lower()} = config_manager.create_instance(configuration_name="{workflow_name}")
```

This approach follows the NanoBrain recommendation to reuse existing components when possible.
"""
        
        # Prepare the steps information
        steps_info = ""
        if steps:
            steps_info = "Steps to include:\n" + "\n".join([f"- {step}" for step in steps])
        
        # Build a prompt for generating the workflow
        prompt = f"""Generate a complete implementation for a NanoBrain workflow class named '{workflow_name}'.
        
        {f"Description: {description}" if description else ""}
        {steps_info}
        
        The workflow should include:
        - Appropriate imports
        - Class docstring with biological analogy
        - Constructor with typed parameters
        - Methods to initialize and connect steps
        - Entry point definition
        - Any necessary helper methods
        
        Follow NanoBrain framework conventions and patterns.
        """
        
        # Generate the workflow implementation
        return await self.process([prompt])

    async def generate_link(self, link_name: str, source_step: str, target_step: str, description: str = None) -> str:
        """
        Generate a complete Link implementation or suggest an existing one.
        
        Args:
            link_name: Name of the link
            source_step: Name of the source step
            target_step: Name of the target step
            description: Description of what the link should do (optional)
        
        Returns:
            Complete Link implementation or configuration suggestion
        """
        # First, check if an existing class can be used
        requirements = f"{description or ''} connect {source_step} to {target_step}"
        existing_class, config = self._find_existing_class('link', requirements)
        
        if existing_class and config:
            # Suggest using the existing class with configuration
            suggested_config = self._create_config_suggestion(existing_class, config, requirements)
            
            # Add input/output to the configuration
            if 'defaults' in suggested_config:
                suggested_config['defaults']['input_data'] = f"{source_step.lower()}_output"
                suggested_config['defaults']['output_data'] = f"{target_step.lower()}_input"
            
            return f"""Instead of creating a new Link class, you can use the existing `{existing_class}` class with a custom configuration:

```yaml
# Configuration for {link_name} using {existing_class}
{yaml.dump(suggested_config, default_flow_style=False)}
```

To use this configuration:
1. Save it to a file, e.g., `{link_name}.yml` in your workflow's `config` directory
2. Create an instance using the ConfigManager:

```python
from src.ConfigManager import ConfigManager

config_manager = ConfigManager(base_path="your_workflow_path")
{link_name.lower()} = config_manager.create_instance(configuration_name="{link_name}")
```

This approach follows the NanoBrain recommendation to reuse existing components when possible.
"""
        
        # Build a prompt for generating the link
        prompt = f"""Generate a complete implementation for a NanoBrain link class named '{link_name}'.
        
        This link should connect step '{source_step}' to step '{target_step}'.
        {f"Description: {description}" if description else ""}
        
        The link should include:
        - Appropriate imports
        - Class docstring with biological analogy
        - Constructor with typed parameters
        - Implementation of required methods (transmit, etc.)
        - Any necessary helper methods
        
        Follow NanoBrain framework conventions and patterns.
        """
        
        # Generate the link implementation
        return await self.process([prompt])

    async def generate_test(self, class_name: str, code: str) -> str:
        """
        Generate tests for a NanoBrain component.
        
        Args:
            class_name: Name of the class to test
            code: Code of the class to test
        
        Returns:
            Test implementation for the class
        """
        # Build a prompt for generating tests
        prompt = f"""Generate comprehensive unit tests for the following NanoBrain class:
        
        ```python
        {code}
        ```
        
        The tests should:
        - Use pytest
        - Test all public methods
        - Include setup and teardown as needed
        - Use appropriate mocks and fixtures
        - Verify error handling
        
        Follow NanoBrain testing conventions and patterns.
        """
        
        # Generate the test implementation
        return await self.process([prompt]) 