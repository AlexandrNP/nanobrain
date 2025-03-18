from typing import Any, List, Optional, Dict, ClassVar, Callable, Tuple, Union
from langchain.llms.base import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, BaseTool, StructuredTool
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
    
    def __init__(self, 
                 executor: ExecutorBase,
                 model_name: str = "gpt-3.5-turbo",
                 model_class: Optional[str] = None,
                 memory_window_size: int = 5,
                 prompt_file: str = None,
                 prompt_template: str = None,
                 prompt_variables: Optional[Dict] = None,
                 use_shared_context: bool = False,
                 shared_context_key: Optional[str] = None,
                 tools: Optional[List[Step]] = None,
                 use_custom_tool_prompt: bool = False,
                 tools_config_path: Optional[str] = None,
                 use_buffer_window_memory: bool = True,
                 memory_key: str = "chat_history",
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
        self.langchain_tools = []
        
        # Register tools from config if provided
        if self.tools_config_path:
            self._load_tools_from_config()
        
        # Register tools if provided
        if self.tools:
            self._register_tools(self.tools)
    
    def _initialize_llm(self, model_name: str, model_class: Optional[str] = None) -> Union[BaseLLM, 'BaseChatModel']:
        """
        Initialize the language model based on the specified model name and class.
        
        Args:
            model_name: Name of the model to use (e.g., "gpt-3.5-turbo", "claude-2")
            model_class: Optional class name to use (e.g., "ChatOpenAI", "ChatAnthropic")
            
        Returns:
            An instance of the language model
        """
        # If we're in testing mode, use mock models
        if TESTING_MODE:
            is_claude = model_name is not None and model_name.startswith("claude")
            if model_class == "OpenAI" or (model_class is None and not is_claude):
                return MockOpenAI()
            else:
                return MockChatOpenAI()
                
        
        # Try to get the global configuration
        
        try:
            global_config = GlobalConfig()  # This will return the singleton instance due to the __new__ method
            # Load config if not already loaded
            if not global_config.config:
                global_config.load_config()
                global_config.load_from_env()
        except Exception as e:
            # Log the error but proceed with defaults
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"Error getting global config: {e}")
            global_config = None
        
        # Default model name if None
        if model_name is None:
            model_name = "gpt-3.5-turbo"
            if hasattr(self, '_debug_mode') and self._debug_mode:
                print(f"Model name is None, defaulting to {model_name}")
        
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
            response = self.llm.invoke(messages).content
        else:
            # For completion models, use the formatted prompt directly
            response = self.llm.predict(formatted_prompt)
        
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
        
        Biological analogy: Social learning.
        Justification: Like how organisms can learn from shared knowledge,
        this method loads memory from a shared context.
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
        
        Biological analogy: Tool use acquisition.
        Justification: Like how primates learn to use tools by integrating
        motor skills with cognitive understanding, this method integrates
        Step objects as tools for the agent's cognitive processing.
        """
        self.langchain_tools = []
        from langchain_core.tools import Tool

        for step in tools:
            if type(step) == Tool:
                self.langchain_tools.append(step)
                continue
            # Create a tool from the Step object
            step_tool = self._create_tool_from_step(step)
            if step_tool:
                self.langchain_tools.append(step_tool)
        
        # Bind tools to the LLM if we have any and if the LLM supports tool binding
        if self.langchain_tools and hasattr(self.llm, 'bind_tools'):
            self.llm_with_tools = self.llm.bind_tools(self.langchain_tools)
    
    def _create_tool_from_step(self, step: Step) -> Optional[BaseTool]:
        """
        Create a LangChain tool from a Step object.
        
        Biological analogy: Mental model formation.
        Justification: Like how the brain creates mental models of tools
        to facilitate their use, this method creates a tool representation
        from a Step object.
        """
        import traceback
        
        # Skip if the step doesn't have a process method
        if not hasattr(step, 'process') or not callable(step.process):
            return None
        
        # Get the signature of the process method
        try:
            sig = inspect.signature(step.process)
            
            # Create a wrapper function that will call the step's process method
            async def tool_func(*args, **kwargs):
                # Convert args to a list for the process method
                inputs = list(args)
                result = await step.process(inputs)
                return result
            
            # Create a synchronous version for tools that don't support async
            def sync_tool_func(*args, **kwargs):
                import asyncio
                # Convert args to a list for the process method
                inputs = list(args)
                # Run the async function in a new event loop
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(step.process(inputs))
                    return result
                finally:
                    loop.close()
            
            # Get the docstring for the step
            doc = step.__doc__ or f"Execute the {step.__class__.__name__} step"
            
            # Create a tool from the function
            if TESTING_MODE:
                # In testing mode, use the mock implementation
                from test.mock_langchain import MockBaseTool
                # Create a tool directly instead of using the decorator
                step_tool = MockBaseTool(
                    name=step.__class__.__name__,
                    description=doc,
                    func=sync_tool_func
                )
            else:
                # In production mode, use the real implementation
                from langchain_core.tools import Tool
                # Create a Tool instance directly
                step_tool = Tool(
                    name=step.__class__.__name__,
                    description=doc,
                    func=sync_tool_func
                )
            
            return step_tool
            
        except Exception as e:
            print(f"Error creating tool from step {step.__class__.__name__}: {type(e).__name__} - {str(e)}")
            print("\nStack trace:")
            traceback.print_exc()
            return None
    
    def add_tool(self, step: Step):
        """
        Add a new tool to the agent.
        
        Biological analogy: Tool acquisition.
        Justification: Like how organisms can learn to use new tools over time,
        this method adds a new tool to the agent's repertoire.
        """
        if step not in self.tools:
            self.tools.append(step)
            tool_obj = self._create_tool_from_step(step)
            if tool_obj:
                self.langchain_tools.append(tool_obj)
                # Re-bind tools to the LLM
                if hasattr(self.llm, 'bind_tools'):
                    self.llm_with_tools = self.llm.bind_tools(self.langchain_tools)
    
    def remove_tool(self, step: Step):
        """
        Remove a tool from the agent.
        
        Biological analogy: Tool disuse.
        Justification: Like how organisms may stop using certain tools when they're
        no longer needed, this method removes a tool from the agent's repertoire.
        """
        if step in self.tools:
            self.tools.remove(step)
            # Recreate all tools to ensure consistency
            self._register_tools(self.tools)
    
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
        
        if self.use_custom_tool_prompt:
            # Use custom tool calling prompt
            # Create custom prompt with tool descriptions
            try:
                # We already imported create_tool_calling_prompt and parse_tool_call at the top level
                custom_prompt = create_tool_calling_prompt(self.tools)
                
                # Format the prompt with input and context
                formatted_prompt = custom_prompt.format(
                    context=context,
                    input=input_text
                )
                
                # Process with LLM based on its type
                from langchain_core.language_models.chat_models import BaseChatModel
                if issubclass(type(self.llm), BaseChatModel):
                    # For chat models, create a chat template
                    from langchain.schema import SystemMessage, HumanMessage
                    system_template = "You are an AI assistant with access to tools. {specific_instructions}"
                    system_message = SystemMessage(content=system_template.format(**self.prompt_variables))
                    human_message = HumanMessage(content=formatted_prompt)
                    messages = [system_message, human_message]
                    
                    # Use self.llm_with_tools if available, otherwise fall back to self.llm
                    if hasattr(self, 'llm_with_tools'):
                        response = self.llm_with_tools.predict_messages(messages).content
                    else:
                        response = self.llm.predict_messages(messages).content
                else:
                    # For completion models, use the formatted prompt directly
                    response = self.llm.predict(formatted_prompt)
                
                # Parse tool call from response
                tool_call = parse_tool_call(response)
                
                if tool_call:
                    tool_name, args = tool_call
                    
                    # Find the matching tool
                    matching_tool = None
                    for tool in self.tools:
                        if tool.__class__.__name__ == tool_name:
                            matching_tool = tool
                            break
                    
                    if matching_tool:
                        # Execute the tool with the provided arguments
                        try:
                            tool_result = await matching_tool.process(args)
                            
                            # Format the final response with the tool result
                            final_response = f"I used the {tool_name} tool with arguments: {', '.join(args)}\n\nResult: {tool_result}"
                            
                            # Update memory with this interaction
                            self._update_memories(input_text, final_response)
                            
                            return final_response
                        except Exception as e:
                            error_response = f"I tried to use the {tool_name} tool, but encountered an error: {str(e)}"
                            
                            # Update memory with this interaction
                            self._update_memories(input_text, error_response)
                            
                            return error_response
                    else:
                        error_response = f"I tried to use a tool called {tool_name}, but it's not available."
                        
                        # Update memory with this interaction
                        self._update_memories(input_text, error_response)
                        
                        return error_response
                else:
                    # No tool call, just a regular response
                    # Update memory with this interaction
                    self._update_memories(input_text, response)
                    
                    return response
            except Exception as e:
                print(f"Error using custom tool prompt: {e}, falling back to standard method")
                # Fall back to standard method if custom tool prompt fails
        
        # Use LangChain's built-in tool calling
        # Format the prompt with input and context
        prompt_vars = {
            "input": input_text,
            "context": context,
            **self.prompt_variables
        }
        
        # Process with LLM based on its type
        from langchain_core.language_models.chat_models import BaseChatModel
        if issubclass(type(self.llm), BaseChatModel):
            # For chat models, create a chat template with tools
            from langchain.schema import SystemMessage, HumanMessage
            system_template = "You are an AI assistant designed to {role_description}. {specific_instructions}"
            system_message = SystemMessage(content=system_template.format(**self.prompt_variables))
            human_message = HumanMessage(content=f"Context: {context}\n\nUser input: {input_text}")
            messages = [system_message, human_message]
            
            # Use self.llm_with_tools if available, otherwise fall back to self.llm
            if hasattr(self, 'llm_with_tools'):
                response = self.llm_with_tools.predict_messages(messages).content
            else:
                print("Warning: llm_with_tools not available, using llm directly")
                response = self.llm.predict_messages(messages, tools=self.langchain_tools).content
        else:
            print('Current LLM:', self.llm)
            print('Type of LLM:', type(self.llm))
            # Check if the llm has a bound attribute before trying to access it
            if hasattr(self.llm, 'bound'):
                print('Bindings of LLM:', self.llm.bound)
            # For completion models, use the formatted prompt
            formatted_prompt = self.prompt_template.format(**prompt_vars)
            response = self.llm.predict(formatted_prompt)
        
        # Update memory with this interaction
        self._update_memories(input_text, response)
        
        return response
    
    async def execute_tool(self, tool_name: str, args: List[Any]) -> Any:
        """
        Execute a specific tool by name with the given arguments.
        
        Biological analogy: Deliberate tool use.
        Justification: Like how humans can deliberately select and use specific tools
        for specific tasks, this method allows direct execution of a specific tool.
        """
        # Find the matching tool
        matching_tool = None
        for tool in self.tools:
            if tool.__class__.__name__ == tool_name:
                matching_tool = tool
                break
        
        if matching_tool:
            # Execute the tool with the provided arguments
            try:
                # Check if the tool has a process method
                if hasattr(matching_tool, 'process') and callable(matching_tool.process):
                    return await matching_tool.process(args)
                else:
                    # If no process method, try to determine the appropriate method to call
                    # based on the arguments
                    if len(args) == 2 and isinstance(args[0], str):
                        # Assume this is a file-related tool call with (path, content)
                        if hasattr(matching_tool, 'create_file') and callable(matching_tool.create_file):
                            return await matching_tool.create_file(args[0], args[1])
                        elif hasattr(matching_tool, 'write') and callable(matching_tool.write):
                            return await matching_tool.write(args[0], args[1])
                    
                    # If we couldn't determine the method, show available methods
                    available_methods = [
                        method for method in dir(matching_tool) 
                        if callable(getattr(matching_tool, method)) and not method.startswith('_')
                    ]
                    return f"Tool {tool_name} found but has no process method. Available methods: {', '.join(available_methods)}"
            except Exception as e:
                return f"Error executing tool {tool_name}: {str(e)}"
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
        
        Biological analogy: Collective memory reset.
        Justification: Like how social groups can reset collective understanding,
        this method clears shared memory.
        """
        if context_key:
            if context_key in cls.shared_context:
                del cls.shared_context[context_key]
        else:
            cls.shared_context.clear()

    def update_workflow_context(self, workflow_path: str):
        """
        Update the agent's context with information about the current workflow.
        
        Biological analogy: Contextual awareness in the prefrontal cortex.
        Justification: Like how the prefrontal cortex maintains awareness of the
        current task context, this method updates the agent's context with
        information about the current workflow.
        
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
        
        Biological analogy: Tool discovery through exploration.
        Justification: Like how organisms discover tools in their environment through
        exploration, this method discovers tools through configuration exploration.
        """
        if not self.tools_config_path:
            return
            
        try:
            # Check if the file exists
            print(f"Looking for tools config in: {self.tools_config_path}")
            if not os.path.exists(self.tools_config_path):
                print(f"Tools config file not found in: {self.tools_config_path}")
                # Try to find the config relative to the current directory
                from src.DirectoryTracer import DirectoryTracer
                tracer = DirectoryTracer()
                current_dir = tracer.get_current_directory()
                config_path = os.path.join(current_dir, "config", self.tools_config_path)
                print(f"Looking for tools config in: {config_path}")
                if os.path.exists(config_path):
                    self.tools_config_path = config_path
                    print(f"Found tools config in: {config_path}")
                else:
                    # Try with just the base directory
                    config_path = os.path.join(current_dir, self.tools_config_path)
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
        
        Biological analogy: Tool crafting from raw materials.
        Justification: Like how early humans crafted tools from raw materials based
        on mental templates, this method creates tool instances from configuration.
        
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