from typing import Any, List, Optional, Dict, ClassVar
from langchain.llms.base import BaseLLM
from langchain_community.chat_models import ChatOpenAI
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

# Check if we're in testing mode
TESTING_MODE = os.environ.get('NANOBRAIN_TESTING', '0') == '1'

# Import mock classes if in testing mode
if TESTING_MODE:
    from mock_langchain import (
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
        
    def _initialize_llm(self, model_name: str, model_class: Optional[str] = None) -> BaseLLM:
        """
        Initialize the language model.
        
        Biological analogy: Specialized neural circuit development.
        Justification: Like how specialized neural circuits develop for specific
        cognitive functions, the agent initializes a specific language model
        for its cognitive processing.
        """
        # If in testing mode, use the mock implementation
        if TESTING_MODE:
            # Use the MockOpenAI that was imported at the top of the file
            return MockOpenAI(model_name=model_name)
            
        if model_class:
            # Dynamic import of the specified model class
            module_path, class_name = model_class.rsplit('.', 1)
            module = importlib.import_module(module_path)
            ModelClass = getattr(module, class_name)
            return ModelClass(model_name=model_name)
        else:
            # Default to OpenAI
            from langchain_community.llms import OpenAI
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
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as file:
                        template_text = file.read()
                else:
                    # Try to find in prompts directory
                    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
                    prompt_path = os.path.join(prompts_dir, prompt_file)
                    if os.path.exists(prompt_path):
                        with open(prompt_path, 'r') as file:
                            template_text = file.read()
            except Exception as e:
                print(f"Error loading prompt file: {e}")
                # Fall back to default template
                template_text = "You are a helpful AI assistant. {input}"
        
        # Use provided template if file loading failed or no file was specified
        if not template_text and prompt_template:
            template_text = prompt_template
        
        # Fall back to default if neither file nor template was provided
        if not template_text:
            template_text = "You are a helpful AI assistant. {input}"
        
        # Create prompt template
        return PromptTemplate.from_template(template_text)
            
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs using the language model.
        
        Biological analogy: Higher-order cognitive processing.
        Justification: Like how the prefrontal cortex integrates information
        and generates responses based on context and past experiences, this
        method processes inputs using the language model and context memory.
        """
        # Convert inputs to text
        input_text = " ".join([str(i) for i in inputs if i is not None])
        
        # Get context from memory
        context = self.get_context_history()
        
        # Format prompt with input and context
        prompt = self.prompt_template.format(input=input_text)
        
        # Generate response
        response = self.llm.predict(prompt)
        
        # Update memories
        self._update_memories(input_text, response)
        
        # Return response
        return response
            
    def _update_memories(self, input_text: str, response: str):
        """
        Update agent's memory with new interaction.
        
        Biological analogy: Memory formation.
        Justification: Like how the brain forms memories from experiences,
        the agent updates its memory with new interactions.
        """
        # Add user message
        self.memory.append({"role": "user", "content": input_text})
        
        # Add assistant message
        self.memory.append({"role": "assistant", "content": response})
        
    def get_full_history(self) -> List[Dict]:
        """
        Get the full conversation history.
        
        Biological analogy: Long-term memory retrieval.
        Justification: Like how the brain can retrieve complete memories,
        this method returns the full conversation history.
        """
        return self.memory
        
    def get_context_history(self) -> List[Dict]:
        """
        Get the recent conversation history based on memory window size.
        
        Biological analogy: Working memory retrieval.
        Justification: Like how working memory holds recent information
        for immediate use, this method returns recent conversation history.
        """
        # Return the most recent messages based on window size
        window_size = self.memory_window_size * 2  # Each exchange has 2 messages
        return self.memory[-window_size:] if len(self.memory) > window_size else self.memory
        
    def clear_memories(self):
        """
        Clear all memories.
        
        Biological analogy: Memory reset.
        Justification: Like how certain brain states can clear working memory,
        this method resets the agent's memory.
        """
        self.memory = []
        
    def dump_to_shared_context(self, context_key: Optional[str] = None):
        """
        Save current memory to shared context.
        
        Biological analogy: Social memory formation.
        Justification: Like how social organisms share memories through
        communication, this method allows sharing memory between agents.
        """
        key = context_key or self.shared_context_key
        Agent.shared_context[key] = self.get_full_history()
        
    def load_from_shared_context(self, context_key: Optional[str] = None, max_messages: Optional[int] = None):
        """
        Load memory from shared context.
        
        Biological analogy: Social learning.
        Justification: Like how social organisms learn from shared experiences,
        this method allows loading memory from other agents.
        """
        key = context_key or self.shared_context_key
        
        if key in Agent.shared_context:
            shared_memory = Agent.shared_context[key]
            
            if max_messages and len(shared_memory) > max_messages:
                # Load only the most recent messages if max_messages is specified
                self.memory = shared_memory[-max_messages:]
            else:
                # Load all messages
                self.memory = shared_memory.copy()
                
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