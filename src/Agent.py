from typing import Any, List, Optional, Dict, ClassVar, Callable, Tuple, Union
from langchain.llms.base import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool, Tool, BaseTool, StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import MessagesPlaceholder
from src.Step import Step
from src.GlobalConfig import GlobalConfig
from src.ExecutorBase import ExecutorBase
import importlib
import sys
import os
import yaml
import json
import inspect
import asyncio
from pathlib import Path
from enum import Enum
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from src.PackageBase import PackageBase

# Try to import prompts, with fallbacks for missing modules
try:
    from prompts.templates import create_chat_template, BASE_ASSISTANT, TECHNICAL_EXPERT, CREATIVE_ASSISTANT
except ImportError:
    # Define fallback simple templates if imports fail
    print("Warning: Could not import templates from prompts package.")
    from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    
    # Define minimal templates as fallbacks
    BASE_ASSISTANT = PromptTemplate(
        input_variables=["role_description", "specific_instructions"],
        template="You are an AI assistant. Your role is to: {role_description}\n\n{specific_instructions}"
    )
    
    TECHNICAL_EXPERT = PromptTemplate(
        input_variables=["expertise_areas"],
        template="You are a technical expert in {expertise_areas}."
    )
    
    CREATIVE_ASSISTANT = PromptTemplate(
        input_variables=["creative_domains"],
        template="You are a creative assistant specializing in {creative_domains}."
    )
    
    def create_chat_template(system_template):
        """Fallback implementation of create_chat_template."""
        system_message = SystemMessagePromptTemplate(prompt=system_template)
        human_message = HumanMessagePromptTemplate.from_template("{input}")
        return ChatPromptTemplate.from_messages([system_message, human_message])

from datetime import datetime
from src.DirectoryTracer import DirectoryTracer
from src.ConfigManager import ConfigManager

# Check if we're in testing mode
TESTING_MODE = os.environ.get('NANOBRAIN_TESTING', '0') == '1'

# Import mock classes if in testing mode
if TESTING_MODE:
    from test.mock_langchain import (
        MockChatOpenAI,
        MockOpenAI,
        MockSystemMessage as SystemMessage,
        MockHumanMessage as HumanMessage,
        MockAIMessage as AIMessage,
        MockPromptTemplate as PromptTemplate,
        MockConversationBufferMemory as ConversationBufferMemory,
        MockConversationBufferWindowMemory as ConversationBufferWindowMemory
    )

class Agent(Step):
    """
    LLM-powered agent that processes inputs using language models.
    
    Biological analogy: Higher-order cognitive processing area.
    Justification: Like how prefrontal cortex integrates information from multiple
    sources and uses past experiences to generate adaptive responses, this agent
    integrates inputs with context memory to generate intelligent responses.
    """
    # Class-level shared context (like collective memory)
    shared_context: ClassVar[Dict[str, List[Dict]]] = {}
    
    # Fields for Pydantic
    model_name: str = ""
    model_class: Optional[str] = None
    memory_window_size: int = 5
    use_shared_context: bool = False
    shared_context_key: Optional[str] = None
    use_custom_tool_prompt: bool = False
    tools_config_path: Optional[str] = None
    memory_key: str = "chat_history"
    debug_mode: bool = False
    llm: Any = None
    memory: List[Dict] = []
    langchain_memory: Any = None
    prompt_template: Any = None
    prompt_variables: Dict = {}
    tools: List = []
    agent_executor: Any = None
    
    def __init__(self, 
                 executor: ExecutorBase,
                 model_name: str = "gpt-3.5-turbo",
                 prompt_variables: Dict,
                 model_class: Optional[str] = None,
                 memory_window_size: int = 5,
                 prompt_file: str = None,
                 prompt_template: str = None,
                 use_shared_context: bool = False,
                 shared_context_key: Optional[str] = None,
                 tools: Optional[List[Step]] = None,
                 use_custom_tool_prompt: bool = False,
                 tools_config_path: Optional[str] = None,
                 use_buffer_window_memory: bool = True,
                 memory_key: str = "chat_history",
                 debug_mode: bool = False,
                 **kwargs):
        """
        Initialize the agent with LLM configuration.
        
        Biological analogy: Neural circuit formation.
        Justification: Like how neural circuits form with specific connectivity
        patterns based on genetic and environmental factors, the agent initializes
        with specific configuration parameters.
        
        Args:
            executor: ExecutorBase instance for running steps
            model_name: Name of the LLM model to use
            model_class: Optional class name for the LLM model
            memory_window_size: Number of recent conversations to keep in context
            prompt_file: Path to file containing prompt template
            prompt_template: String containing prompt template
            prompt_variables: Variables to fill in the prompt template
            use_shared_context: Whether to use shared context between agents
            shared_context_key: Key for shared context group
            tools: Optional list of Step objects to use as tools
            use_custom_tool_prompt: Whether to use a custom prompt for tool calling
            tools_config_path: Path to YAML file with tool configurations
            use_buffer_window_memory: Whether to use ConversationBufferWindowMemory (True) or ConversationBufferMemory (False)
            memory_key: The key to use for the memory in the prompt template
            debug_mode: Whether to enable debug mode
            **kwargs: Additional keyword arguments
        """
        super().__init__(executor, **kwargs)
        
        # Store configuration
        self.model_name = model_name
        self.model_class = model_class
        self.memory_window_size = memory_window_size
        self.use_shared_context = use_shared_context
        self.shared_context_key = shared_context_key or self.__class__.__name__
        self.use_custom_tool_prompt = use_custom_tool_prompt
        self.tools_config_path = tools_config_path
        self.memory_key = memory_key
        self.debug_mode = debug_mode
        
        # Initialize LLM
        self.llm = self._initialize_llm(self.model_name, self.model_class)
        
        # Initialize memory (both internal storage and Langchain memory)
        self.memory = []
        
        # Set up Langchain memory
        if use_buffer_window_memory:
            self.langchain_memory = ConversationBufferWindowMemory(
                k=self.memory_window_size,
                memory_key=self.memory_key,
                return_messages=True
            )
        else:
            self.langchain_memory = ConversationBufferMemory(
                memory_key=self.memory_key,
                return_messages=True
            )
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_file, prompt_template)
        self.prompt_variables = prompt_variables or {}
        
        # Load from shared context if needed
        if self.use_shared_context:
            self.load_from_shared_context(self.shared_context_key)
        
        # Store tools
        self.tools = tools or []
        
        # Register tools from config if provided
        if self.tools_config_path:
            self._load_tools_from_config()
        
        # Register tools if provided
        if self.tools:
            self._register_tools(self.tools)
        
        # Set up agent executor for tool calling
        self.agent_executor = None
        breakpoint()
        self._setup_agent_executor()
        
    
    def _initialize_llm(self, model_name: str, model_class: Optional[str] = None) -> Union[BaseLLM, 'BaseChatModel']:
        """
        Initialize the language model based on the specified name and class.
        
        Biological analogy: Cognitive development.
        Justification: Like how the brain develops specific cognitive 
        capabilities during development, this method initializes the 
        specific language model capabilities for the agent.
        
        Args:
            model_name: The name of the model to use, e.g., "gpt-3.5-turbo"
            model_class: Optional class name for the model, e.g., "ChatOpenAI"
            
        Returns:
            The initialized language model
        """
        try:
            # Special case for testing - handle mock-chat model class
            if os.environ.get('NANOBRAIN_TESTING', '0') == '1' and model_class == "mock-chat":
                from test.mock_langchain import MockChatOpenAI
                return MockChatOpenAI(model_name=model_name)
                
            # Determine the appropriate class to use
            if model_class:
                # Use the specified class
                if getattr(self, "debug_mode") and self.debug_mode:
                    print(f"Using specified model class: {model_class}")
                    
                # Try to import and create the specified class
                try:
                    # Split by last dot to get module path and class name
                    if '.' in model_class:
                        module_path, class_name = model_class.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        llm_class = getattr(module, class_name)
                    else:
                        # Try to import from langchain.llms or langchain.chat_models
                        # Try multiple import paths
                        llm_class = None
                        for import_path in ["langchain.llms", "langchain.chat_models"]:
                            try:
                                module = importlib.import_module(import_path)
                                if hasattr(module, model_class):
                                    llm_class = getattr(module, model_class)
                                    break
                            except (ImportError, AttributeError):
                                continue
                        
                        if not llm_class:
                            raise ImportError(f"Could not import {model_class} from standard paths")
                    
                    # Create an instance of the LLM class
                    if getattr(self, "debug_mode") and self.debug_mode:
                        print(f"Creating LLM instance with class {llm_class.__name__} and model {model_name}")
                        
                    return llm_class(model_name=model_name)
                except Exception as e:
                    print(f"Error creating LLM instance: {e}")
                    raise
            else:
                # If model_class is None, use the default model_name
                model_name = model_name or "gpt-3.5-turbo"
                if getattr(self, "debug_mode") and self.debug_mode:
                    print(f"Model name is None, defaulting to {model_name}")
            
            # Handle special testing cases
            if os.environ.get('NANOBRAIN_TESTING', '0') == '1':
                # Import mock classes for testing
                from test.mock_langchain import MockChatOpenAI, MockOpenAI
                # Default to Mock classes in testing mode
                return MockChatOpenAI(model_name=model_name) if model_name.startswith("gpt") else MockOpenAI(model_name=model_name)
            
            # Try to get the global configuration
            global_config = GlobalConfig()  # This will return the singleton instance due to the __new__ method
            # Load config if not already loaded
            if not global_config.config:
                global_config.load_config()
                global_config.load_from_env()
            
            # Determine the model provider based on the model name
            if model_name.startswith("gpt"):
                provider = "openai"
                if model_class is None:
                    model_class = "ChatOpenAI"
            elif model_name.startswith("claude"):
                provider = "anthropic"
                if model_class is None:
                    model_class = "ChatAnthropic"
            elif model_name.startswith("gemini"):
                provider = "google"
                if model_class is None:
                    model_class = "ChatGoogleGenerativeAI"
            elif model_name.startswith("mistral"):
                provider = "mistral"
                if model_class is None:
                    model_class = "ChatMistralAI"
            else:
                provider = "openai"  # Default to OpenAI
                if model_class is None:
                    model_class = "ChatOpenAI"
            
            # Get API key from global configuration if available
            api_key = None
            if global_config:
                api_key = global_config.get_api_key(provider)
            
            # Handle different model classes
            if model_class == "OpenAI":
                from langchain.llms import OpenAI
                return OpenAI(model_name=model_name, openai_api_key=api_key)
            elif model_class == "ChatOpenAI":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
            elif model_class == "ChatAnthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(model_name=model_name, anthropic_api_key=api_key)
            elif model_class == "ChatGoogleGenerativeAI":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
            elif model_class == "ChatMistralAI":
                from langchain_mistralai import ChatMistralAI
                return ChatMistralAI(model=model_name, mistral_api_key=api_key)
            else:
                # Default to ChatOpenAI
                from langchain.chat_models import ChatOpenAI
                return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    
    def _load_prompt_template(self, prompt_file=None, prompt_template=None):
        """
        Load the prompt template from a file or string.
        
        Biological analogy: Reading neuronal activation patterns.
        Justification: Like how neurons follow specific activation patterns,
        this method loads specific prompt patterns to guide agent behavior.
        """
        # Base assistant template as fallback
        BASE_ASSISTANT = """You are a helpful AI assistant. 
        
        Current conversation:
        {chat_history}
        
        User: {input}
        AI: """
        
        # If prompt_template is already a PromptTemplate, return it directly
        if isinstance(prompt_template, PromptTemplate):
            return prompt_template
            
        # Load from file if specified
        if prompt_file:
            try:
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        template_content = f.read()
                else:
                    # Default to a basic template if file not found
                    template_content = BASE_ASSISTANT
            except Exception as e:
                print(f"Error loading prompt file: {e}")
                template_content = BASE_ASSISTANT
        elif prompt_template:
            template_content = prompt_template
        else:
            template_content = BASE_ASSISTANT
            
        # Use the appropriate PromptTemplate class based on testing mode
        return PromptTemplate.from_template(template_content)
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs using the language model.
        
        Biological analogy: Higher-order cognitive processing.
        Justification: Like how the prefrontal cortex integrates information
        from multiple sources and generates adaptive responses, this method
        integrates inputs with context to generate responses.
        """
        # If we have tools available, use process_with_tools instead
        breakpoint()
        if self.tools and self.agent_executor:
            return await self.process_with_tools(inputs)
            
        # Extract the input text from the inputs list
        input_text = inputs[0] if inputs else ""
        
        # Get context from memory
        context = self.get_context_history()
        
        # Format the prompt with input and context
        prompt_vars = {
            "input": input_text,
            "context": context,
            **self.prompt_variables
        }
        
        # Add chat history from langchain memory
        memory_dict = self.langchain_memory.load_memory_variables({})
        if self.memory_key in memory_dict:
            prompt_vars[self.memory_key] = memory_dict[self.memory_key]
        
        formatted_prompt = self.prompt_template.format(**prompt_vars)
        
        # Process with LLM based on its type
        from langchain_core.language_models.chat_models import BaseChatModel
        if issubclass(type(self.llm), BaseChatModel):
            # For chat models, create a chat template
            from langchain.schema import SystemMessage, HumanMessage
            system_template = "You are an AI assistant designed to {role_description}. {specific_instructions}"
            system_message = SystemMessage(content=system_template.format(**self.prompt_variables))
            human_message = HumanMessage(content=f"Context: {context}\n\nUser input: {input_text}")
            messages = [system_message, human_message]
            response_obj = await self.llm.ainvoke(messages)
            response = response_obj.content
        else:
            # For completion models, use the formatted prompt directly
            response = await self.llm.ainvoke(formatted_prompt)
        
        # Update memory with this interaction
        self._update_memories(input_text, response)
        
        # Update langchain memory
        self.langchain_memory.save_context({"input": input_text}, {"output": response})
        
        return response
    
    def _update_memories(self, user_input: str, assistant_response: str):
        """
        Update the agent's memory with new interactions.
        
        Biological analogy: Memory formation.
        Justification: Like how the brain forms memories from experiences,
        the agent updates its memory with new interactions.
        """
        # Add the new interaction to memory
        self.memory.append({"role": "user", "content": user_input})
        self.memory.append({"role": "assistant", "content": assistant_response})
        
        # If using shared context, update it
        if self.use_shared_context:
            self.save_to_shared_context(self.shared_context_key)
    
    def get_full_history(self) -> List[Dict]:
        """
        Get the full conversation history.
        
        Biological analogy: Long-term memory retrieval.
        Justification: Like how the brain can retrieve complete episodic
        memories, this method retrieves the full conversation history.
        """
        return self.memory
    
    def get_context_history(self) -> str:
        """
        Get recent conversation history formatted as context.
        
        Biological analogy: Working memory access.
        Justification: Like how the brain maintains recent information in
        working memory for immediate use, this method retrieves recent
        conversation history for context.
        """
        # Try to get context from langchain memory first
        try:
            memory_dict = self.langchain_memory.load_memory_variables({})
            if self.memory_key in memory_dict and memory_dict[self.memory_key]:
                # Format langchain messages to string
                context_str = ""
                for message in memory_dict[self.memory_key]:
                    role = "User" if isinstance(message, HumanMessage) else "Assistant"
                    context_str += f"{role}: {message.content}\n"
                return context_str
        except Exception as e:
            # Fall back to internal memory if there's an issue with langchain memory
            pass
        
        # Get the most recent interactions based on memory_window_size
        recent_memory = self.memory[-2*self.memory_window_size:] if len(self.memory) > 2*self.memory_window_size else self.memory
        
        # Format as a string
        context_str = ""
        for entry in recent_memory:
            context_str += f"{entry['role'].capitalize()}: {entry['content']}\n"
            
        return context_str
    
    def clear_memories(self):
        """
        Clear all memories.
        
        Biological analogy: Memory reset.
        Justification: Like how certain brain processes can clear working
        memory for new tasks, this method clears the agent's memory.
        """
        self.memory = []
        
        # Clear langchain memory as well
        self.langchain_memory.clear()
        
        # If using shared context, clear it too
        if self.use_shared_context:
            if self.shared_context_key in Agent.shared_context:
                Agent.shared_context[self.shared_context_key] = []
    
    def save_to_shared_context(self, context_key: str):
        """
        Save memory to shared context.
        
        Biological analogy: Collective memory formation.
        Justification: Like how social organisms contribute to collective
        knowledge, this method saves memory to a shared context.
        """
        Agent.shared_context[context_key] = self.memory.copy()
    
    def load_from_shared_context(self, context_key: str):
        """
        Load memory from shared context.
        """
        if context_key in Agent.shared_context:
            # Clear existing memory first
            self.memory = []
            self.langchain_memory.clear()
            
            # Load shared context
            self.memory = Agent.shared_context[context_key].copy()
            
            # Rebuild langchain memory from shared context
            self.langchain_memory.clear()
            for i in range(0, len(self.memory), 2):
                if i+1 < len(self.memory):  # Make sure we have both user and assistant messages
                    user_message = self.memory[i]["content"]
                    assistant_message = self.memory[i+1]["content"]
                    self.langchain_memory.save_context({"input": user_message}, {"output": assistant_message})
    
    def _register_tools(self, tools: List[Step]):
        """
        Register Step objects as tools for the LLM.
        Required only for non-OpenAI-compatible tools.
        
        """
        pass
    
    def _create_tool_from_step(self, step: Step) -> Optional[BaseTool]:
        """
        Create a LangChain tool from a Step object.
        
        Biological analogy: Mental model formation.
        Justification: Like how the brain creates mental models of tools
        to facilitate their use, this method creates a tool representation
        from a Step object.
        
        Note: Since Step now inherits from BaseTool, this method simply returns
        the step itself as it's already a BaseTool instance.
        """
        #if getattr(self, "debug_mode", False):
        #    print(f"Step {step.__class__.__name__} is already a BaseTool instance")
        
        return step
    
    def add_tool(self, step: Step):
        """
        Add a Step object as a tool for the LLM.
        """
        if step not in self.tools:
            self.tools.append(step)
            
            # Set up or refresh the agent executor with the new tools
            self._setup_agent_executor()
            
            if getattr(self, "debug_mode", False):
                print(f"Added tool: {step.name}")
    
    def remove_tool(self, step: Step):
        """
        Remove a Step object from the available tools.
        
        """
        if step in self.tools:
            self.tools.remove(step)
            
            # Set up or refresh the agent executor with the updated tools
            self._setup_agent_executor()
            
            if getattr(self, "debug_mode", False):
                print(f"Removed tool: {step.name}")
    
    def get_tools(self) -> List:
        """
        Get the list of tools available to this agent.
        
        Returns:
            List of tool objects available to the agent
        """
        return self.tools.copy()
    
    async def process_with_tools(self, inputs: List[Any]) -> Any:
        """
        Process inputs using the language model with tool calling capability.
        
        Biological analogy: Tool-assisted problem solving.
        Justification: Like how humans use tools to extend their problem-solving
        capabilities, this method uses tools to extend the agent's processing capabilities.
        """
        # Extract the input text from the inputs list
        input_text = inputs[0] if inputs else ""
        
        # Get context from memory
        context = self.get_context_history()
        
        # If agent_executor is not set up, we can't use tools - fall back to regular process
        if not self.agent_executor or not self.tools:
            print("Warning: No agent_executor or tools available. Falling back to regular process.")
            return await self.process(inputs)
        
        try:
            # Invoke the agent executor to process the input with tools
            result = await self.agent_executor.ainvoke({"input": input_text})
            
            # The agent executor result will be a dict with an 'output' key
            response = result.get("output", "")
            
            # Update memory with this interaction
            self._update_memories(input_text, response)
            
            return response
        except Exception as e:
            # If there's an error, fall back to regular processing
            if getattr(self, "debug_mode", False):
                print(f"Error processing with tools: {e}. Falling back to regular process.")
                import traceback
                traceback.print_exc()
            
            # Fall back to regular process without tools
            return await self.process(inputs)
    
    async def execute_tool(self, tool_name: str, args: List[Any]) -> Any:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: The name of the tool to execute
            args: List of arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        # Find the matching tool in the registry
        matching_tool = None
        for tool in self.tools:
            if tool.name == tool_name:
                matching_tool = tool
                break
        
        if matching_tool:
            # Execute the tool with the provided arguments
            try:
                # Special handling for CalculatorStep in testing mode
                if os.environ.get('NANOBRAIN_TESTING', '0') == '1' and tool_name == "CalculatorStep":
                    # For CalculatorStep, ensure args are properly converted
                    try:
                        # Convert numeric args from strings to floats if needed
                        if len(args) > 2:
                            operation = args[0]
                            arg1 = float(args[1]) if isinstance(args[1], str) else args[1]
                            arg2 = float(args[2]) if isinstance(args[2], str) else args[2]
                            
                            # Call the process method directly with the properly prepared arguments
                            return await matching_tool.process([operation, arg1, arg2])
                    except ValueError as e:
                        return f"Error converting arguments: {e}"
                    except Exception as e:
                        return f"Error in special handling: {e}"
                
                # For other tools, use _arun
                try:
                    result = await matching_tool._arun(args)
                    return result
                except Exception as e:
                    # Fallback to process method if _arun fails
                    try:
                        return await matching_tool.process(args)
                    except Exception as e2:
                        return f"Error in fallback process: {e2}"
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {str(e)}"
                return error_msg
        else:
            return f"Tool {tool_name} not found"

    @classmethod
    def get_shared_context(cls, context_key: str) -> List[Dict]:
        """
        Get shared context by key.
        
        Biological analogy: Accessing collective memory.
        Justification: Like how social organisms access collective knowledge,
        this method retrieves shared memory by key.
        """
        return cls.shared_context.get(context_key, [])
        
    @classmethod
    def clear_shared_context(cls, context_key: Optional[str] = None):
        """
        Clear shared context.
        """
        if context_key:
            if context_key in cls.shared_context:
                del cls.shared_context[context_key]
        else:
            cls.shared_context.clear()

    def update_workflow_context(self, workflow_path: str):
        """
        Update the agent's context with information about the current workflow.
        
        Args:
            workflow_path: Path to the current workflow file
        """
        # Store the workflow path in the agent's context
        if not hasattr(self, 'workflow_context'):
            self.workflow_context = {}
        
        self.workflow_context = {
            "path": workflow_path,
            "updated_at": datetime.now().isoformat()
        }
        
        # Add a message to the agent's memory about the workflow context
        workflow_name = os.path.basename(workflow_path).replace('.pkl', '')
        self._update_memories(
            f"Working on workflow: {workflow_name}",
            f"I'll help you with the workflow '{workflow_name}'. What would you like to do with this workflow?"
        )

    def _load_tools_from_config(self):
        """
        Load tools from the YAML configuration file specified by tools_config_path.
        """
        if not self.tools_config_path:
            return
            
        try:
            # Check if the file exists
            global_config = GlobalConfig()
            if not os.path.isfile(self.tools_config_path):
                print(f"Looking for tools config in: {self.tools_config_path}")
                print(f"Tools config file not found in: {self.tools_config_path}")
                # Try to find the config relative to the current directory
                current_dir = self.directory_tracer.get_absolute_directory_location()
                workflow_config_path = os.path.join(current_dir, "config", self.tools_config_path)
                print(f"Looking for tools config in: {workflow_config_path}")
                if os.path.exists(workflow_config_path):
                    self.tools_config_path = workflow_config_path
                    print(f"Found tools config in: {workflow_config_path}")
                else:
                    # Try with just the base directory
                    config_path = os.path.join(global_config.tracer.get_absolute_directory_location(), 'default_config', self.tools_config_path)
                    print(f"Looking for tools config in: {config_path}")
                    if os.path.exists(config_path):
                        self.tools_config_path = config_path
                        print(f"Found tools config in: {config_path}")
                    else:
                        print(f"Tools config file not found: {config_path}")
                        return
                
            # Load the YAML file
            print(f"Loading tools config from: {self.tools_config_path}")
            with open(self.tools_config_path, 'r') as file:
                tools_config = yaml.safe_load(file)
                
            # Check if 'tools' key exists
            if 'tools' not in tools_config:
                print(f"No 'tools' key found in config: {self.tools_config_path}")
                return
            # Print available tools for debugging
            print(f"Found {len(tools_config['tools'])} tools in config:")
            for i, tool_config in enumerate(tools_config['tools']):
                print(f"  {i+1}. {tool_config.get('name', 'Unnamed')}: {tool_config.get('class', 'No class specified')}")
                
            # Create tool instances
            for tool_config in tools_config['tools']:
                print(f"Creating tool instance for: {tool_config.get('name', 'Unnamed')}")
                tool_instance = self._create_tool_instance(tool_config)
                if tool_instance:
                    self.tools.append(tool_instance)
                    print(f"Successfully added tool: {tool_instance.__class__.__name__}")
                else:
                    print(f"Failed to create tool instance for: {tool_config.get('name', 'Unnamed')}")
                    
        except Exception as e:
            print(f"Error loading tools from config: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_tool_instance(self, tool_config: Dict) -> Optional[Step]:
        """
        Create a tool instance from the tool configuration.
        
        Args:
            tool_config: Dictionary containing tool configuration
            
        Returns:
            Instance of the specified tool class or None if creation fails
        """
        try:
            # Get the class path and name
            class_path = tool_config.get('class')
            if not class_path:
                print(f"No class specified in tool config: {tool_config}")
                return None
                
            # Split into module path and class name
            module_path, class_name = class_path.rsplit('.', 1)
            
            # Debug log
            print(f"Attempting to import module: {module_path} for class: {class_name}")
            
            try:
                # Import the module
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as e:
                print(f"Error importing {module_path}: {str(e)}")
                print(f"Current sys.path: {sys.path}")
                raise
            
            # Get the class
            cls = getattr(module, class_name)
            
            # Create the instance with the executor from the agent's runner
            # The executor is passed to the constructor in the Step.__init__
            instance = cls(executor=self.runner)
            
            print(f"Successfully created tool instance: {instance.__class__.__name__}")
            return instance
            
        except Exception as e:
            print(f"Error creating tool instance: {str(e)}")
            import traceback
            traceback.print_exc()
            return None 

    def _setup_agent_executor(self):
        """
        Set up the agent executor for tool calling.
        """
        # Don't set up the agent executor if there are no tools
        if not self.tools:
            if getattr(self, "debug_mode", False):
                print("Warning: No tools available for agent executor setup")
            return
        
        try:
            # Create a system message prompt based on the agent's role
            system_message_prompt = SystemMessage(
                content="You are an AI assistant designed to {role_description}. {specific_instructions}"
                .format(**self.prompt_variables)
            )
            
            # Create a chat prompt template
            prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                MessagesPlaceholder(variable_name=self.memory_key),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create the agent using LangChain's create_tool_calling_agent
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            
            # Create the agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True if getattr(self, "debug_mode", False) else False,
                memory=self.langchain_memory,
                handle_parsing_errors=True
            )
            
            # Log successful creation
            if getattr(self, "debug_mode", False):
                print(f"Successfully set up agent executor with {len(self.tools)} tools")
        except Exception as e:
            # Log error but don't re-raise
            if getattr(self, "debug_mode", False):
                print(f"Error setting up agent executor: {e}")
                import traceback
                traceback.print_exc() 