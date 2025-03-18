from typing import List, Dict, Any, Optional, Union
import os
import asyncio
from pathlib import Path
import sys

from src.Agent import Agent
from src.ExecutorBase import ExecutorBase
from src.Step import Step
from src.DataStorageBase import DataStorageBase


class AgentWorkflowBuilder(Agent):
    """
    AI assistant for creating and managing NanoBrain workflow structures.
    
    Biological analogy: Prefrontal cortex with planning capabilities.
    Justification: Like how the prefrontal cortex plans and organizes complex
    behaviors, this agent plans and organizes the creation of workflows.
    """
    def __init__(
        self,
        executor: ExecutorBase,
        input_storage: DataStorageBase,
        model_name: str = "gpt-3.5-turbo",
        model_class: Optional[str] = None,
        tools_config_path: Optional[str] = None,
        **kwargs
    ):
        # Store the executor as an instance attribute
        self.executor = executor
        
        # Initialize the Agent base class
        super().__init__(
            executor=executor,
            model_name=model_name,
            model_class=model_class,
            **kwargs
        )
        
        # AgentWorkflowBuilder-specific attributes
        self.input_storage = input_storage
        self.tools_config_path = tools_config_path or os.path.join(
            os.path.dirname(__file__), "config", "tools.yml"
        )
        
        # Context management
        self.documentation_context = {}  # NanoBrain documentation
        self.workflow_context = {}  # Current workflow visibility
        
        # Load tools from configuration
        self._load_tools()
    
    def _load_tools(self):
        """Load tools from configuration file."""
        # Check if the tools configuration file exists and is readable
        valid_config_file = False
        
        if self.tools_config_path and os.path.exists(self.tools_config_path):
            try:
                # Try to read the file
                with open(self.tools_config_path, 'r') as f:
                    file_content = f.read()
                    if file_content.strip():  # File exists and has content
                        valid_config_file = True
            except Exception as e:
                print(f"Error reading tools configuration file: {e}")
        
        if not valid_config_file:
            print(f"Tools configuration file not found or empty: {self.tools_config_path}")
            print("Creating a default configuration file...")
            self._create_default_tools_config()
        
        try:
            import yaml
            with open(self.tools_config_path, 'r') as f:
                tools_config = yaml.safe_load(f)
            
            # Check if the file has the expected structure
            if not isinstance(tools_config, dict) or 'tools' not in tools_config:
                print(f"Invalid tools configuration format. Expected a dict with 'tools' key.")
                print("Creating a default configuration file...")
                self._create_default_tools_config()
                with open(self.tools_config_path, 'r') as f:
                    tools_config = yaml.safe_load(f)
            
            # Initialize tools from configuration
            for tool_config in tools_config.get('tools', []):
                self._initialize_tool(tool_config)
        
        except Exception as e:
            print(f"Error loading tools configuration: {e}")
    
    def _create_default_tools_config(self):
        """Create a default tools configuration file."""
        import yaml
        
        # Define default tools
        self.default_tools = [
            {
                'name': 'PlannerTool',
                'class': 'tools_common.StepPlanner.StepPlanner',
                'description': 'Plans the implementation of a new step or workflow.'
            },
            {
                'name': 'FileWriterTool',
                'class': 'tools_common.StepFileWriter.StepFileWriter',
                'description': 'Creates or modifies a file with the provided content.'
            },
            {
                'name': 'CoderTool',
                'class': 'tools_common.StepCoder.StepCoder',
                'description': 'Generates software code based on requirements.'
            },
            {
                'name': 'ContextSearchTool',
                'class': 'tools_common.StepContextSearch.StepContextSearch',
                'description': 'Searches the surrounding context for relevant information.'
            },
            {
                'name': 'WebSearchTool',
                'class': 'tools_common.StepWebSearch.StepWebSearch',
                'description': 'Searches the web for relevant information.'
            },
            {
                'name': 'ContextArchiverTool',
                'class': 'tools_common.StepContextArchiver.StepContextArchiver',
                'description': 'Archives and summarizes old messages.'
            },
            {
                'name': 'GitInitTool',
                'class': 'tools_common.StepGitInit.StepGitInit',
                'description': 'Initializes a git repository with a given name in the specified folder.'
            },
            {
                'name': 'GitExcludeTool',
                'class': 'tools_common.StepGitExclude.StepGitExclude',
                'description': 'Excludes directory and all files in it from the current git repository.'
            },
            {
                'name': 'DependencySearchTool',
                'class': 'tools_common.StepDependencySearch.StepDependencySearch',
                'description': 'Searches for dependencies in Python files and saves them to requirements.txt.'
            }
        ]
        
        # If tools_config_path is not set, use a default path
        if not self.tools_config_path:
            # Set a default path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(project_root, "builder", "config")
            self.tools_config_path = os.path.join(config_dir, "tools.yml")
        
        # Create the directory if it doesn't exist
        config_dir = os.path.dirname(self.tools_config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        
        # Create a default tools.yml with proper structure
        default_config = {'tools': self.default_tools}
        
        # Write the default configuration
        with open(self.tools_config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _initialize_tool(self, tool_config: Dict[str, Any]):
        """Initialize a tool from configuration."""
        try:
            # Extract tool information
            tool_name = tool_config.get('name')
            tool_class_path = tool_config.get('class')
            tool_description = tool_config.get('description', '')
            
            if not tool_name or not tool_class_path:
                print(f"Invalid tool configuration: {tool_config}")
                return
            
            # Split the class path into module and class
            module_parts = tool_class_path.split('.')
            
            # Add tools_common to sys.path if not already there
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            tools_common_dir = os.path.join(project_root, 'tools_common')
            if tools_common_dir not in sys.path:
                sys.path.insert(0, tools_common_dir)
            
            # Try to import the module and get the class
            try:
                # Case 1: If it's a fully qualified path (module.class)
                if len(module_parts) >= 2:
                    class_name = module_parts[-1]
                    module_path = '.'.join(module_parts[:-1])
                    
                    import importlib
                    module = importlib.import_module(module_path)
                    tool_class = getattr(module, class_name)
                # Case 2: If it's just a class name, assume it's in tools_common
                else:
                    class_name = module_parts[0]
                    module_path = f"tools_common.{class_name}"
                    
                    try:
                        import importlib
                        module = importlib.import_module(module_path)
                        tool_class = getattr(module, class_name)
                    except ImportError:
                        # Try alternative approach - just the class name
                        # Search in common module paths
                        found = False
                        for base_module in ['tools_common', 'src', 'builder']:
                            try:
                                module_path = f"{base_module}"
                                module = importlib.import_module(module_path)
                                if hasattr(module, class_name):
                                    tool_class = getattr(module, class_name)
                                    found = True
                                    break
                            except (ImportError, AttributeError):
                                pass
                        
                        if not found:
                            print(f"Could not find class {class_name} in any module")
                            return
                
                # Create an instance of the tool
                tool_instance = tool_class(executor=self.executor)
                
                # Add the tool to the agent
                self.add_tool(tool_instance)
                
                print(f"Initialized tool: {tool_name}")
            except ImportError as e:
                print(f"Error importing module for {tool_class_path}: {e}")
            except AttributeError as e:
                print(f"Error getting class from module for {tool_class_path}: {e}")
        
        except Exception as e:
            print(f"Error initializing tool {tool_config.get('name')}: {e}")
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs using the language model with tools.
        
        Biological analogy: Higher-order cognitive processing with tool use.
        Justification: Like how humans use tools to extend their cognitive
        capabilities, this agent uses tools to extend its processing capabilities.
        """
        # Process the input with tools
        response = await self.process_with_tools(inputs)
        
        # Display the response if input_storage is available and has display_response method
        if self.input_storage is not None and hasattr(self.input_storage, 'display_response'):
            self.input_storage.display_response(response)
        
        return response
    
    def load_documentation_context(self):
        """Load NanoBrain documentation context."""
        # TODO: Implement loading documentation context
        pass
    
    def update_workflow_context(self, workflow_path: str):
        """Update the current workflow context."""
        # TODO: Implement updating workflow context
        pass
    
    def archive_old_messages(self):
        """Archive old messages and summarize them."""
        # TODO: Implement message archiving
        pass 
        
    def _is_requesting_new_class(self, input_text: str) -> bool:
        """
        Detect if a request is asking to create a new class from scratch.
        
        Args:
            input_text: The input text to analyze
            
        Returns:
            bool: True if the request is for a new class, False otherwise
        """
        # Look for phrases that indicate creating a new class
        new_class_patterns = [
            "new class",
            "create a class",
            "from scratch",
            "build a class",
            "implement a class"
        ]
        
        # Check if any pattern is present in the input
        input_lower = input_text.lower()
        return any(pattern in input_lower for pattern in new_class_patterns)
        
    async def suggest_implementation(self, step_name: str, description: str) -> str:
        """
        Suggest an implementation for a step based on description.
        
        Args:
            step_name: The name of the step to implement
            description: Description of what the step should do
            
        Returns:
            str: A suggestion for implementation, either using an existing component
                 or creating a new one
        """
        # Check if this is a forceful request for a new class
        if self._is_requesting_new_class(description):
            # If explicitly requesting a new class, provide guidance for creating one
            return await self._provide_guidance(step_name, description)
            
        # Try to find an existing component that matches the description
        component_type = "step"  # Default to step component type
        
        # If the code writer is available, use it to find existing classes
        if hasattr(self, 'code_writer') and self.code_writer is not None:
            class_name, config = self.code_writer._find_existing_class(component_type, description)
            
            if class_name and config:
                # Found an existing class that matches, suggest using it
                return f"""
```yaml
{config}
```

I suggest using the existing **{class_name}** class, which matches your requirements.
This component is already available in NanoBrain and can be configured as shown above
to meet your needs. No need to create a new class from scratch!

Would you like me to explain how to use this component in your workflow?
"""
            
        # No existing class found that matches, create a new one
        return await self._provide_guidance(step_name, description)
        
    async def _provide_guidance(self, component_name: str, description: str) -> str:
        """
        Provide guidance on how to implement a component.
        
        Args:
            component_name: The name of the component to implement
            description: Description of what the component should do
            
        Returns:
            str: Guidance on implementing the component
        """
        # Use the LLM to generate implementation guidance
        prompt = f"""
        I need to implement a Step component called '{component_name}' in NanoBrain.
        
        The component should: {description}
        
        Please provide:
        1. A brief explanation of how I should approach implementing this
        2. An outline of the code structure
        3. Any potential pitfalls or considerations
        """
        
        # Use the process_with_tools method to leverage all available tools
        if hasattr(self, 'process_with_tools'):
            return await self.process_with_tools([prompt])
        else:
            # Fallback if process_with_tools not available
            return "To implement this component, start by creating a new class that inherits from the appropriate base class."
            
    async def list_existing_components(self, component_type: str = None) -> str:
        """
        List existing components of the specified type.
        
        Args:
            component_type: The type of component to list (e.g., 'step', 'link', 'trigger')
                          If None, list all components
                          
        Returns:
            str: A formatted list of existing components with descriptions
        """
        # Define paths to search for components
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths_to_search = [
            os.path.join(base_path, 'src'),
            os.path.join(base_path, 'builder')
        ]
        
        components = {}
        
        # Iterate through the paths
        for path in paths_to_search:
            if not os.path.exists(path):
                continue
                
            # Use glob to find Python files
            for py_file in Path(path).glob('**/*.py'):
                # Skip __init__ files
                if '__init__' in py_file.name:
                    continue
                    
                # Read the file
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                        
                    # Extract class definitions and their docstrings
                    import re
                    class_matches = re.finditer(r'class\s+(\w+)(?:\(.*?\))?:\s*(?:"""(.*?)""")?', content, re.DOTALL)
                    
                    for match in class_matches:
                        class_name = match.group(1)
                        docstring = match.group(2) if match.group(2) else "No description available"
                        
                        # Clean up docstring (remove newlines, extra spaces)
                        docstring = ' '.join(docstring.split())
                        
                        # Determine component type from class name or inheritance
                        detected_type = None
                        if 'Step' in class_name:
                            detected_type = 'step'
                        elif 'Link' in class_name:
                            detected_type = 'link'
                        elif 'Trigger' in class_name:
                            detected_type = 'trigger'
                        elif 'Agent' in class_name:
                            detected_type = 'agent'
                        
                        # Skip if component_type is specified and doesn't match
                        if component_type and detected_type != component_type:
                            continue
                            
                        # Add to components dictionary
                        if detected_type:
                            if detected_type not in components:
                                components[detected_type] = []
                            components[detected_type].append((class_name, docstring))
                            
                except Exception as e:
                    print(f"Error processing file {py_file}: {e}")
        
        # Format the results
        result = "# Existing Components\n\n"
        
        if not components:
            return result + "No components found."
            
        for ctype, comps in components.items():
            result += f"## {ctype.capitalize()} Components\n\n"
            for name, desc in comps:
                result += f"### {name}\n{desc}\n\n"
                
        return result 

    async def generate_step_template(self, step_name: str, base_class: str = "Step", description: str = None) -> str:
        """
        Generate a template for a new step class.
        
        Biological analogy: Neural template formation.
        Justification: Like how the brain has templates for generating new neurons with
        specific functions, this method creates templates for new step classes with 
        specific functionality.
        
        Args:
            step_name: Name of the step to create
            base_class: Base class for the step (default: "Step")
            description: Description of the step's functionality
            
        Returns:
            str: Template code for the step class
        """
        # Use ConfigManager to get a consistent template
        from src.ConfigManager import ConfigManager
        
        # Create a config manager with the current directory as base path
        config_manager = ConfigManager(base_path=os.getcwd())
        
        # Prepare a prompt for the template generation
        prompt = f"Generate Python code for a class named {step_name} that inherits from {base_class}."
        if description:
            prompt += f" The class should {description}"
            
        # Use the code writer if available
        if hasattr(self, 'code_writer') and self.code_writer is not None:
            # Use the existing code writer to generate the code
            code = await self.code_writer.process([prompt])
            return code
        
        # If no code writer, generate a basic template
        template = f"""#!/usr/bin/env python3
\"\"\"
{step_name} - {description or 'A custom step for NanoBrain workflows'}

This step implements {description or 'custom functionality'} for NanoBrain workflows.
\"\"\"

from src.{base_class} import {base_class}


class {step_name}({base_class}):
    \"\"\"
    {description or 'A custom step for NanoBrain workflows'}
    
    Biological analogy: Specialized neuron.
    Justification: Like how specialized neurons perform specific functions
    in the brain, this step performs a specific function in the workflow.
    \"\"\"
    
    def __init__(self, **kwargs):
        \"\"\"Initialize the step.\"\"\"
        super().__init__(**kwargs)
        
    def process(self, data_dict):
        \"\"\"
        Process input data.
        
        Args:
            data_dict: Dictionary containing input data
            
        Returns:
            Dictionary containing output data
        \"\"\"
        # Process the input data
        result = {{}}
        
        # Add your custom processing logic here
        # ...
        
        return result
"""
        return template
        
    def get_generated_config(self) -> str:
        """
        Get the generated configuration YAML.
        
        Biological analogy: Gene expression regulation.
        Justification: Like how gene expression is regulated by configuration
        in DNA and epigenetic factors, this method provides configuration
        for the step's behavior.
        
        Returns:
            str: YAML configuration for the step
        """
        # Use ConfigManager to generate a consistent configuration
        from src.ConfigManager import ConfigManager
        
        # Create a config manager with the current directory as base path
        config_manager = ConfigManager(base_path=os.getcwd())
        
        # Generate a basic configuration
        config = """# Default configuration
defaults:
  # Add your default configuration parameters here
  debug_mode: false
  log_level: "INFO"
  
# Step-specific configuration
step:
  # Add step-specific configuration parameters here
  # These will override the defaults
  
# Custom parameters
parameters:
  # Add custom parameters here
  # These will be available to the step during execution
"""
        return config
        
    def get_generated_tests(self) -> str:
        """
        Get the generated test code.
        
        Biological analogy: Immune system testing.
        Justification: Like how the immune system tests new cells to
        ensure they function correctly, this method provides tests to
        ensure the step functions correctly.
        
        Returns:
            str: Test code for the step
        """
        # Get the current step name from the context
        step_class_name = ""
        if hasattr(self, 'current_step_dir'):
            step_class_name = os.path.basename(self.current_step_dir)
        
        if not step_class_name:
            step_class_name = "CustomStep"
            
        # Generate a basic test template
        test_template = f"""#!/usr/bin/env python3
\"\"\"
Test file for {step_class_name}
\"\"\"

import unittest
import os
import sys

# Add the parent directory to sys.path for imports to work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.{step_class_name} import {step_class_name}


class Test{step_class_name}(unittest.TestCase):
    \"\"\"Test cases for {step_class_name} class.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.step = {step_class_name}()
    
    def test_process(self):
        \"\"\"Test the process method.\"\"\"
        # Test input data
        input_data = {{}}
        
        # Process the data
        result = self.step.process(input_data)
        
        # Verify the result
        self.assertIsNotNone(result)
        
    # Add more test methods as needed


if __name__ == '__main__':
    unittest.main()
"""
        return test_template
        
    def get_generated_code(self) -> str:
        """
        Get the generated code.
        
        This is a convenience method that delegates to the code writer if available.
        
        Returns:
            str: Generated code or empty string if no code writer is available
        """
        # Use the code writer if available
        if hasattr(self, 'code_writer') and self.code_writer is not None and hasattr(self.code_writer, 'generated_code'):
            return self.code_writer.generated_code
        
        return "" 

    def add_tool(self, tool):
        """
        Add a tool to the agent's repertoire.
        
        In biological analogy, this is akin to an organism acquiring a new capability,
        like a new enzyme or sensory ability that helps it interact with its environment.
        
        Parameters:
            tool: The tool to add to the agent's repertoire.
        """
        super().add_tool(tool) 