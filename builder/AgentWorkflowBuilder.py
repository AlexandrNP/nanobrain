from typing import List, Dict, Any, Optional, Union
import os
import asyncio
import glob
from pathlib import Path
import yaml
import importlib
import inspect

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
        self.archive_context = {}  # Archived messages
        
        # Load tools from configuration
        self._load_tools()
    
    def _load_tools(self):
        """Load tools from configuration file."""
        # Check if the tools configuration file exists
        if not os.path.exists(self.tools_config_path):
            print(f"Tools configuration file not found: {self.tools_config_path}")
            print("Creating a default configuration file...")
            self._create_default_tools_config()
        
        try:
            import yaml
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
        default_tools = {
            'tools': [
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
                },
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
                    'description': 'Searches the web for information about APIs and best practices.'
                },
                {
                    'name': 'ContextArchiverTool',
                    'class': 'tools_common.StepContextArchiver.StepContextArchiver',
                    'description': 'Archives and summarizes old messages.'
                }
            ]
        }
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.tools_config_path), exist_ok=True)
        
        # Write the default configuration
        with open(self.tools_config_path, 'w') as f:
            yaml.dump(default_tools, f, default_flow_style=False)
    
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
            
            # Import the tool class
            module_path, class_name = tool_class_path.rsplit('.', 1)
            module_path = module_path.replace('-', '_').replace('.', '_')
            
            # Check if the module exists
            module_file = module_path.replace('.', '/') + '.py'
            if not os.path.exists(module_file):
                print(f"Tool module not found: {module_file}")
                return
            
            # Import the module
            import importlib
            module = importlib.import_module(module_path)
            
            # Get the class
            tool_class = getattr(module, class_name)
            
            # Create an instance of the tool
            tool_instance = tool_class(executor=self.executor)
            
            # Add the tool to the agent
            self.add_tool(tool_instance)
            
            print(f"Initialized tool: {tool_name}")
        
        except Exception as e:
            print(f"Error initializing tool {tool_config.get('name')}: {e}")
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs using the language model with tools.
        
        Biological analogy: Higher-order cognitive processing with tool use.
        Justification: Like how humans use tools to extend their cognitive
        capabilities, this agent uses tools to extend its processing capabilities.
        """
        # Archive old messages if needed
        if len(self.memory) > self.memory_window_size * 2:
            self.archive_old_messages()
            
        # Update the prompt with the current context
        prompt_variables = self.prompt_variables or {}
        prompt_variables['context'] = self.get_context_history()
        
        # Process the input with tools
        response = await self.process_with_tools(inputs)
        
        # Display the response
        if hasattr(self.input_storage, 'display_response'):
            self.input_storage.display_response(response)
        
        return response
    
    def load_documentation_context(self):
        """Load NanoBrain documentation context."""
        # Find the NanoBrain package directory
        try:
            # Try to find the package in the current directory
            src_dir = os.path.join(os.getcwd(), 'src')
            if os.path.isdir(src_dir):
                nanobrain_dir = src_dir
            else:
                # Try to find the installed package
                import src
                nanobrain_dir = os.path.dirname(os.path.abspath(src.__file__))
            
            # Collect documentation from source files
            for root, _, files in os.walk(nanobrain_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            module_name = os.path.splitext(file)[0]
                            spec = importlib.util.spec_from_file_location(module_name, file_path)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # Get module docstring
                            module_doc = module.__doc__
                            if module_doc:
                                self.documentation_context[module_name] = module_doc
                            
                            # Get class docstrings
                            for name, obj in inspect.getmembers(module):
                                if inspect.isclass(obj) and obj.__module__ == module_name:
                                    class_doc = obj.__doc__
                                    if class_doc:
                                        self.documentation_context[f"{module_name}.{name}"] = class_doc
                        except Exception as e:
                            print(f"Error loading documentation from {file_path}: {e}")
            
            # Collect documentation from YAML configuration files
            config_dir = os.path.join(os.path.dirname(nanobrain_dir), 'default_configs')
            if os.path.isdir(config_dir):
                for file in glob.glob(os.path.join(config_dir, '*.yml')):
                    try:
                        with open(file, 'r') as f:
                            config = yaml.safe_load(f)
                            
                            if 'metadata' in config:
                                class_name = os.path.splitext(os.path.basename(file))[0]
                                self.documentation_context[f"config.{class_name}"] = config['metadata']
                    except Exception as e:
                        print(f"Error loading documentation from {file}: {e}")
            
            print(f"Loaded documentation context with {len(self.documentation_context)} entries")
        
        except Exception as e:
            print(f"Error loading documentation context: {e}")
    
    def update_workflow_context(self, workflow_path: str):
        """Update the current workflow context."""
        if not os.path.isdir(workflow_path):
            print(f"Workflow directory not found: {workflow_path}")
            return
        
        # Clear the current workflow context
        self.workflow_context = {}
        
        # Collect workflow context from source files
        for root, _, files in os.walk(workflow_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, workflow_path)
                
                # Read file content
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Store file content in context
                    self.workflow_context[rel_path] = content
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        
        print(f"Updated workflow context with {len(self.workflow_context)} files")
    
    def archive_old_messages(self):
        """Archive old messages and summarize them."""
        # Get messages to archive (keep the most recent memory_window_size messages)
        messages_to_archive = self.memory[:-self.memory_window_size]
        
        if not messages_to_archive:
            return
        
        # Generate a summary of the archived messages
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages_to_archive])
        summary_prompt = f"Summarize the following conversation, focusing on key decisions and information that will be relevant for future interactions:\n\n{input_text}"
        
        try:
            # Use the LLM to generate a summary
            summary = self.llm.predict(summary_prompt)
            
            # Add the summary to the archive context
            timestamp = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
            self.archive_context[f"summary_{timestamp}"] = summary
            
            # Update the memory to remove archived messages
            self.memory = self.memory[-self.memory_window_size:]
            
            # Add the summary as a system message
            self._update_memories(
                user_input="",
                assistant_response="",
                system_message=f"[Previous conversation summary: {summary}]"
            )
            
            print(f"Archived {len(messages_to_archive)} messages and generated a summary")
        
        except Exception as e:
            print(f"Error archiving messages: {e}")
    
    def _update_memories(self, user_input: str, assistant_response: str, system_message: Optional[str] = None):
        """Update the memory with a new interaction."""
        if system_message:
            self.memory.append({
                "role": "system",
                "content": system_message
            })
        
        if user_input:
            self.memory.append({
                "role": "user",
                "content": user_input
            })
        
        if assistant_response:
            self.memory.append({
                "role": "assistant",
                "content": assistant_response
            }) 