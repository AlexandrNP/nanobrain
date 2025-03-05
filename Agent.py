from typing import Any, List, Optional, Dict, ClassVar
from langchain.llms.base import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from Step import Step
from ExecutorBase import ExecutorBase
from prompts.templates import create_chat_template, BASE_ASSISTANT, TECHNICAL_EXPERT, CREATIVE_ASSISTANT
import importlib
import yaml
import os
from datetime import datetime
from DirectoryTracer import DirectoryTracer
from ConfigManager import ConfigManager

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
                 **kwargs):
        """
        Initialize the agent with LLM configuration.
        
        Biological analogy: Neural circuit formation.
        Justification: Like how neural circuits form with specific connectivity
        patterns based on their function, the agent initializes with specific
        configuration based on its intended role.
        """
        # Initialize the Step base class
        super().__init__(executor, **kwargs)
        
        # Set up directory tracer and config manager
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        
        # Load configuration
        config = self.config_manager.get_config(self.__class__.__name__)
        
        # Use config values or defaults
        self.model_name = config.get('model_name', model_name)
        self.model_class = config.get('model_class', model_class)
        self.memory_window_size = config.get('memory_window_size', memory_window_size)
        self.prompt_file = config.get('prompt_file', prompt_file)
        self.prompt_template = config.get('prompt_template', prompt_template)
        self.prompt_variables = config.get('prompt_variables', prompt_variables or {})
        
        # Context sharing settings
        self.use_shared_context = config.get('use_shared_context', use_shared_context)
        self.shared_context_key = config.get('shared_context_key', shared_context_key or self.__class__.__name__)
        
        # Initialize LLM
        self.llm = self._initialize_llm(self.model_name, self.model_class)
        
        # Initialize memories
        self.full_memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.context_memory = ConversationBufferMemory(memory_key="history", return_messages=True, k=self.memory_window_size)
        
        # Initialize prompt template
        self.prompt = self._load_prompt_template(self.prompt_file, self.prompt_template)
        
        # Load from shared context if needed
        if self.use_shared_context:
            self.load_from_shared_context(self.shared_context_key)
        
    def _initialize_llm(self, model_name: str, model_class: Optional[str] = None) -> BaseLLM:
        """
        Initialize the language model.
        
        Biological analogy: Specialized neural circuit development.
        Justification: Like how specialized neural circuits develop for specific
        cognitive functions, the agent initializes a specific language model
        for its cognitive processing.
        """
        if model_class:
            # Dynamic import of the specified model class
            module_path, class_name = model_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
            return ModelClass(model_name=model_name)
        else:
            # Default to OpenAI
            from langchain.llms import OpenAI
            return OpenAI(model_name=model_name)
            
    def _load_prompt_template(self, prompt_file: Optional[str], prompt_template: Optional[str]) -> PromptTemplate:
        """
        Load prompt template from file or use provided template.
        
        Biological analogy: Loading cognitive schemas.
        Justification: Like how the brain loads cognitive schemas to guide
        information processing, the agent loads prompt templates to guide
        language generation.
        """
        template_text = ""
        
        # Try to load from file first
        if prompt_file:
            try:
                with open(prompt_file, 'r') as file:
                    template_text = file.read()
            except Exception as e:
                print(f"Error loading prompt file: {e}")
                
        # Fall back to provided template
        if not template_text and prompt_template:
            template_text = prompt_template
            
        # Fall back to default template
        if not template_text:
            template_text = """
            You are a helpful AI assistant. 
            
            {history}
            
            Human: {input}
            AI: 
            """
            
        # Create template with input variables
        input_variables = ["history", "input"]
        if self.prompt_variables:
            # Add any additional variables from config
            for var in self.prompt_variables:
                if var not in input_variables:
                    input_variables.append(var)
                    
        return PromptTemplate(
            input_variables=input_variables,
            template=template_text
        )
            
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs using the language model.
        
        Biological analogy: Cognitive processing with memory integration.
        Justification: Like how the brain integrates new information with
        existing memories to generate responses, the agent integrates
        inputs with context memory to generate responses.
        """
        if not inputs:
            return None
            
        # Combine all inputs into a single text
        input_text = ""
        for item in inputs:
            if isinstance(item, str):
                input_text += item + "\n"
            elif hasattr(item, '__str__'):
                input_text += str(item) + "\n"
                
        # Prepare prompt variables
        prompt_vars = {"input": input_text}
        
        # Add history from context memory
        history_vars = self.context_memory.load_memory_variables({})
        prompt_vars.update(history_vars)
        
        # Add any additional variables from config
        if self.prompt_variables:
            prompt_vars.update(self.prompt_variables)
            
        # Generate response
        prompt_text = self.prompt.format(**prompt_vars)
        response = self.llm(prompt_text)
        
        # Update memories
        self._update_memories(input_text, response)
        
        # Update shared context if needed
        if self.use_shared_context:
            self.dump_to_shared_context(self.shared_context_key)
            
        return response
            
    def _update_memories(self, input_text: str, response: str):
        """
        Update both full and context memories.
        
        Biological analogy: Memory encoding and consolidation.
        Justification: Like how the brain encodes new experiences into
        both short-term and long-term memory, the agent updates both
        its full and context memories.
        """
        # Update full memory (like long-term memory)
        self.full_memory.chat_memory.add_user_message(input_text)
        self.full_memory.chat_memory.add_ai_message(response)
        
        # Update context memory (like working memory)
        self.context_memory.chat_memory.add_user_message(input_text)
        self.context_memory.chat_memory.add_ai_message(response)
        
    def get_full_history(self) -> List[Dict]:
        """
        Retrieve full conversation history.
        
        Biological analogy: Long-term memory access.
        Justification: Like how long-term memory stores complete
        experiences, this method retrieves the complete conversation history.
        """
        return self.full_memory.load_memory_variables({}).get("history", [])
        
    def get_context_history(self) -> List[Dict]:
        """
        Retrieve recent context history.
        
        Biological analogy: Working memory access.
        Justification: Like how working memory maintains recent and relevant
        information for current processing, this method retrieves recent context.
        """
        return self.context_memory.load_memory_variables({}).get("history", [])
        
    def clear_memories(self):
        """
        Clear both full and context memories.
        
        Biological analogy: Memory reset.
        Justification: Like how sleep helps clear and reorganize neural circuits,
        this method resets the agent's memory states.
        """
        self.full_memory.clear()
        self.context_memory.clear()
        
    def dump_to_shared_context(self, context_key: Optional[str] = None):
        """
        Dump current message history to shared context.
        
        Biological analogy: Memory consolidation to shared knowledge.
        Justification: Like how individual experiences can be consolidated into
        shared knowledge in social groups, this method allows individual agent
        memories to be shared with other agents.
        """
        target_key = context_key or self.shared_context_key
        if target_key not in Agent.shared_context:
            Agent.shared_context[target_key] = []
            
        # Get full history
        history = self.get_full_history()
        
        # Convert to shared context format
        shared_messages = []
        for message in history:
            shared_messages.append({
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content
            })
            
        # Update shared context
        Agent.shared_context[target_key] = shared_messages
        
    def load_from_shared_context(self, context_key: Optional[str] = None, max_messages: Optional[int] = None):
        """
        Load message history from shared context.
        
        Biological analogy: Accessing collective knowledge.
        Justification: Like how individuals in social groups can access
        shared knowledge, this method allows agents to access shared
        conversation history.
        """
        target_key = context_key or self.shared_context_key
        
        # Check if context exists
        if target_key not in Agent.shared_context:
            return
            
        # Clear existing memories
        self.clear_memories()
        
        # Get shared messages
        shared_messages = Agent.shared_context[target_key]
        
        # Limit number of messages if specified
        if max_messages:
            shared_messages = shared_messages[-max_messages:]
            
        # Load into memories
        for message in shared_messages:
            if message["role"] == "user":
                self.full_memory.chat_memory.add_user_message(message["content"])
                self.context_memory.chat_memory.add_user_message(message["content"])
            else:
                self.full_memory.chat_memory.add_ai_message(message["content"])
                self.context_memory.chat_memory.add_ai_message(message["content"])
        
    @classmethod
    def get_shared_context(cls, context_key: str) -> List[Dict]:
        """
        Get shared context by key.
        
        Biological analogy: Accessing specific shared knowledge.
        Justification: Like how individuals can access specific domains
        of shared knowledge, this method allows access to specific
        shared conversation contexts.
        """
        if context_key in cls.shared_context:
            return cls.shared_context[context_key]
        return []
        
    @classmethod
    def clear_shared_context(cls, context_key: Optional[str] = None):
        """
        Clear shared context.
        
        Biological analogy: Resetting collective knowledge.
        Justification: Like how social groups can reset or update their
        shared knowledge, this method allows clearing of shared conversation history.
        """
        if context_key:
            if context_key in cls.shared_context:
                cls.shared_context[context_key] = []
        else:
            cls.shared_context = {} 