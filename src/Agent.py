from typing import Any, List, Optional, Dict, ClassVar, Callable, Tuple, Union
from langchain.llms.base import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, BaseTool, StructuredTool
from src.Step import Step
from src.ExecutorBase import ExecutorBase
from prompts.templates import create_chat_template, BASE_ASSISTANT, TECHNICAL_EXPERT, CREATIVE_ASSISTANT
import importlib
import yaml
import os
import inspect
from datetime import datetime
from src.DirectoryTracer import DirectoryTracer
from src.ConfigManager import ConfigManager

# Check if we're in testing mode
TESTING_MODE = os.environ.get('NANOBRAIN_TESTING', '0') == '1'

# Import mock classes if in testing mode
if TESTING_MODE:
    from test.mock_langchain import (
        MockChatOpenAI as ChatOpenAI,
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
                 **kwargs):
        """
        Initialize the agent with LLM configuration.
        
        Biological analogy: Neural circuit formation.
        Justification: Like how neural circuits form with specific connectivity
        patterns based on genetic and environmental factors, the agent initializes
        with specific configuration parameters.
        """
        super().__init__(executor, **kwargs)
        
        # Store configuration
        self.model_name = model_name
        self.model_class = model_class
        self.memory_window_size = memory_window_size
        self.use_shared_context = use_shared_context
        self.shared_context_key = shared_context_key or self.__class__.__name__
        self.use_custom_tool_prompt = use_custom_tool_prompt
        
        # Initialize LLM
        self.llm = self._initialize_llm(self.model_name, self.model_class)
        
        # Initialize memory
        self.memory = []
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template(prompt_file, prompt_template)
        self.prompt_variables = prompt_variables or {}
        
        # Load from shared context if needed
        if self.use_shared_context:
            self.load_from_shared_context(self.shared_context_key)
        
        # Store tools
        self.tools = tools or []
        self.langchain_tools = []
        
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
            if model_class == "OpenAI" or (model_class is None and not model_name.startswith("claude")):
                return MockOpenAI()
            else:
                return MockChatOpenAI()
        
        # Try to get the global configuration
        try:
            from src.GlobalConfig import GlobalConfig
            global_config = GlobalConfig()
        except ImportError:
            global_config = None
        
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
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        elif model_class == "ChatAnthropic":
            from langchain.chat_models import ChatAnthropic
            return ChatAnthropic(model_name=model_name, anthropic_api_key=api_key)
        elif model_class == "ChatGoogleGenerativeAI":
            from langchain.chat_models import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        elif model_class == "ChatMistralAI":
            from langchain.chat_models import ChatMistralAI
            return ChatMistralAI(model=model_name, mistral_api_key=api_key)
        else:
            # Default to ChatOpenAI
            from langchain.chat_models import ChatOpenAI
            return ChatOpenAI(model_name=model_name, openai_api_key=api_key)
    
    def _load_prompt_template(self, prompt_file: Optional[str], prompt_template: Optional[str]) -> PromptTemplate:
        """
        Load a prompt template from a file or use a provided template.
        
        Biological analogy: Loading cognitive schemas.
        Justification: Like how the brain loads cognitive schemas for different
        contexts, the agent loads prompt templates for different interaction types.
        """
        if prompt_file:
            try:
                # Try to load from the specified path
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        template_content = f.read()
                # If not found, try to load from a prompts directory
                elif os.path.exists(f"prompts/{prompt_file}"):
                    with open(f"prompts/{prompt_file}", 'r') as f:
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
        
        formatted_prompt = self.prompt_template.format(**prompt_vars)
        
        # Process with LLM based on its type
        from langchain_core.language_models.chat_models import BaseChatModel
        if isinstance(self.llm, BaseChatModel):
            # For chat models, create a chat template
            from langchain.schema import SystemMessage, HumanMessage
            system_template = "You are an AI assistant designed to {role_description}. {specific_instructions}"
            system_message = SystemMessage(content=system_template.format(**self.prompt_variables))
            human_message = HumanMessage(content=f"Context: {context}\n\nUser input: {input_text}")
            messages = [system_message, human_message]
            response = self.llm.predict_messages(messages).content
        else:
            # For completion models, use the formatted prompt directly
            response = self.llm.predict(formatted_prompt)
        
        # Update memory with this interaction
        self._update_memories(input_text, response)
        
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
            self.memory = Agent.shared_context[context_key].copy()
    
    def _register_tools(self, tools: List[Step]):
        """
        Register Step objects as tools for the LLM.
        
        Biological analogy: Tool use acquisition.
        Justification: Like how primates learn to use tools by integrating
        motor skills with cognitive understanding, this method integrates
        Step objects as tools for the agent's cognitive processing.
        """
        self.langchain_tools = []
        
        for step in tools:
            # Create a tool from the Step object
            step_tool = self._create_tool_from_step(step)
            if step_tool:
                self.langchain_tools.append(step_tool)
        
        # Bind tools to the LLM if we have any and if the LLM supports tool binding
        if self.langchain_tools and hasattr(self.llm, 'bind_tools'):
            self.llm = self.llm.bind_tools(self.langchain_tools)
    
    def _create_tool_from_step(self, step: Step) -> Optional[BaseTool]:
        """
        Create a LangChain tool from a Step object.
        
        Biological analogy: Mental model formation.
        Justification: Like how the brain creates mental models of tools
        to facilitate their use, this method creates a tool representation
        from a Step object.
        """
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
                from test.mock_langchain import tool
                step_tool = tool(name=step.__class__.__name__, description=doc)(sync_tool_func)
            else:
                # In production mode, use the real implementation
                from langchain_core.tools import tool
                step_tool = tool(name=step.__class__.__name__, description=doc)(sync_tool_func)
            
            return step_tool
            
        except Exception as e:
            print(f"Error creating tool from step {step.__class__.__name__}: {e}")
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
                    self.llm = self.llm.bind_tools(self.langchain_tools)
    
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
            from prompts.tool_calling_prompt import create_tool_calling_prompt, parse_tool_call
            
            # Create custom prompt with tool descriptions
            custom_prompt = create_tool_calling_prompt(self.tools)
            
            # Format the prompt with input and context
            formatted_prompt = custom_prompt.format(
                context=context,
                input=input_text
            )
            
            # Process with LLM based on its type
            from langchain_core.language_models.chat_models import BaseChatModel
            if isinstance(self.llm, BaseChatModel):
                # For chat models, create a chat template
                from langchain.schema import SystemMessage, HumanMessage
                system_template = "You are an AI assistant with access to tools. {specific_instructions}"
                system_message = SystemMessage(content=system_template.format(**self.prompt_variables))
                human_message = HumanMessage(content=formatted_prompt)
                messages = [system_message, human_message]
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
        else:
            # Use LangChain's built-in tool calling
            # Format the prompt with input and context
            prompt_vars = {
                "input": input_text,
                "context": context,
                **self.prompt_variables
            }
            
            # Process with LLM based on its type
            from langchain_core.language_models.chat_models import BaseChatModel
            if isinstance(self.llm, BaseChatModel):
                # For chat models, create a chat template with tools
                from langchain.schema import SystemMessage, HumanMessage
                system_template = "You are an AI assistant designed to {role_description}. {specific_instructions}"
                system_message = SystemMessage(content=system_template.format(**self.prompt_variables))
                human_message = HumanMessage(content=f"Context: {context}\n\nUser input: {input_text}")
                messages = [system_message, human_message]
                
                # Use the chat model with tools
                response = self.llm.predict_messages(messages, tools=self.langchain_tools).content
            else:
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
                return await matching_tool.process(args)
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