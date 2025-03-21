"""
Mock implementations of LangChain classes for testing.
This module provides mock implementations of LangChain classes to allow testing
without requiring actual API keys or external services.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import os
import re

class MockMessage:
    """Mock implementation of LangChain message classes."""
    
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role
        
    def __str__(self):
        return f"{self.role}: {self.content}"
        
    def to_dict(self):
        return {"role": self.role, "content": self.content}

class MockSystemMessage(MockMessage):
    """Mock implementation of SystemMessage."""
    
    def __init__(self, content: str):
        super().__init__(content, role="system")

class MockHumanMessage(MockMessage):
    """Mock implementation of HumanMessage."""
    
    def __init__(self, content: str):
        super().__init__(content, role="human")

class MockAIMessage(MockMessage):
    """Mock implementation of AIMessage."""
    
    def __init__(self, content: str):
        super().__init__(content, role="ai")

class MockChatOpenAI:
    """Mock implementation of ChatOpenAI."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.tools = []
        self._predict = None
        
    @property
    def predict(self) -> Callable:
        """Get the predict method."""
        return self._predict or self._default_predict

    @predict.setter
    def predict(self, value: Callable):
        """Set the predict method."""
        self._predict = value
        
    def _default_predict(self, text: str, **kwargs) -> str:
        """Return a mock response."""
        return f"Mock response to: {text}"
        
    async def apredict(self, text: str, **kwargs) -> str:
        """Return a mock response asynchronously."""
        return self.predict(text, **kwargs)
        
    def predict_messages(self, messages: List[MockMessage], **kwargs) -> MockAIMessage:
        """Return a mock response as an AIMessage."""
        return MockAIMessage("Mock response to messages")
    
    async def apredict_messages(self, messages: List[MockMessage], **kwargs) -> MockAIMessage:
        """Return a mock response as an AIMessage asynchronously."""
        return self.predict_messages(messages, **kwargs)
    
    def invoke(self, messages: List[MockMessage], **kwargs) -> MockAIMessage:
        """Return a mock response when invoked."""
        return self.predict_messages(messages, **kwargs)
    
    async def ainvoke(self, messages: List[MockMessage], **kwargs) -> MockAIMessage:
        """Return a mock response when invoked asynchronously."""
        return await self.apredict_messages(messages, **kwargs)
    
    def bind_tools(self, tools: List[Any]) -> 'MockChatOpenAI':
        """Bind tools to the model."""
        self.tools = tools
        return self

class MockOpenAI:
    """Mock implementation of OpenAI."""
    
    def __init__(self, model_name: str = "text-davinci-003", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.tools = []
        
    def predict(self, text: str, **kwargs) -> str:
        """Return a mock response."""
        return f"Mock response to: {text}"
    
    async def apredict(self, text: str, **kwargs) -> str:
        """Return a mock response asynchronously."""
        return self.predict(text, **kwargs)
    
    def invoke(self, text: str, **kwargs) -> str:
        """Return a mock response when invoked."""
        return self.predict(text, **kwargs)
    
    async def ainvoke(self, text: str, **kwargs) -> str:
        """Return a mock response when invoked asynchronously."""
        return await self.apredict(text, **kwargs)
    
    def bind_tools(self, tools: List[Any]) -> 'MockOpenAI':
        """Bind tools to the model."""
        self.tools = tools
        return self

class MockConversationBufferMemory:
    """Mock implementation of ConversationBufferMemory."""
    
    def __init__(self, memory_key: str = "history", return_messages: bool = False, **kwargs):
        self.chat_memory = MockChatMemory()
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.kwargs = kwargs
        
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory."""
        self.chat_memory.add_user_message(inputs.get("input", ""))
        self.chat_memory.add_ai_message(outputs.get("output", ""))
        
    def load_memory_variables(self, input_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load memory variables."""
        if input_values is None:
            input_values = {}
            
        if self.return_messages:
            return {self.memory_key: self.chat_memory.messages}
        else:
            # Convert messages to string format
            result = ""
            for message in self.chat_memory.messages:
                if isinstance(message, MockHumanMessage):
                    result += f"Human: {message.content}\n"
                elif isinstance(message, MockAIMessage):
                    result += f"AI: {message.content}\n"
                elif isinstance(message, MockSystemMessage):
                    result += f"System: {message.content}\n"
            return {self.memory_key: result.strip()}
            
    def clear(self) -> None:
        """Clear memory."""
        self.chat_memory.clear()

class MockConversationBufferWindowMemory(MockConversationBufferMemory):
    """Mock implementation of ConversationBufferWindowMemory."""
    
    def __init__(self, k: int = 5, memory_key: str = "history", return_messages: bool = False, **kwargs):
        super().__init__(memory_key=memory_key, return_messages=return_messages, **kwargs)
        self.k = k
        
    def load_memory_variables(self, input_values: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load memory variables with window."""
        if input_values is None:
            input_values = {}
            
        # Get the last k exchanges (k * 2 messages)
        messages = self.chat_memory.messages[-self.k*2:] if len(self.chat_memory.messages) > self.k*2 else self.chat_memory.messages
        
        if self.return_messages:
            return {self.memory_key: messages}
        else:
            # Convert messages to string format
            result = ""
            for message in messages:
                if isinstance(message, MockHumanMessage):
                    result += f"Human: {message.content}\n"
                elif isinstance(message, MockAIMessage):
                    result += f"AI: {message.content}\n"
                elif isinstance(message, MockSystemMessage):
                    result += f"System: {message.content}\n"
            return {self.memory_key: result.strip()}

class MockChatMemory:
    """Mock implementation of ChatMemory."""
    
    def __init__(self):
        self.messages = []
        
    def add_user_message(self, message: str) -> None:
        """Add a user message to memory."""
        self.messages.append(MockHumanMessage(message))
        
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to memory."""
        self.messages.append(MockAIMessage(message))
        
    def clear(self) -> None:
        """Clear all messages from memory."""
        self.messages = []

class MockPromptTemplate:
    """Mock implementation of PromptTemplate."""
    
    def __init__(self, template: str, input_variables: List[str], **kwargs):
        self.template = template
        self.input_variables = input_variables
        self.kwargs = kwargs
        
    @classmethod
    def from_template(cls, template_or_prompt, **kwargs):
        """Create a prompt template from a template string or PromptTemplate object."""
        # Check if template is already a PromptTemplate
        if hasattr(template_or_prompt, 'template') and hasattr(template_or_prompt, 'input_variables'):
            # If it's a PromptTemplate-like object, extract its template and input_variables
            template = template_or_prompt.template
            input_variables = template_or_prompt.input_variables
            return cls(template=template, input_variables=input_variables, **kwargs)
        
        # Otherwise, treat it as a string template
        template = template_or_prompt
        # Extract variables from the template string
        variables = re.findall(r'\{([^{}]*)\}', template)
        # Remove duplicates and create a list of input variables
        input_variables = list(set(variables))
        return cls(template=template, input_variables=input_variables, **kwargs)
        
    def format(self, **kwargs) -> str:
        """Format the template with the given values."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            # Handle missing keys by using empty strings
            missing_key = str(e).strip("'")
            kwargs[missing_key] = ""
            return self.format(**kwargs)

class MockBaseTool:
    """Mock implementation of BaseTool."""
    
    def __init__(self, name: str, description: str, func: Callable, **kwargs):
        self.name = name
        self.description = description
        self.func = func
        self.kwargs = kwargs
        self.return_direct = kwargs.get('return_direct', False)
        self.args_schema = kwargs.get('args_schema', None)
        self.coroutine = None  # For async compatibility
        
    def __call__(self, *args, **kwargs):
        """Call the tool function."""
        return self.func(*args, **kwargs)
    
    def invoke(self, input_data, **kwargs):
        """Invoke the tool with the given input."""
        if isinstance(input_data, dict):
            return self.func(**input_data)
        return self.func(input_data)
    
    async def ainvoke(self, input_data, **kwargs):
        """Invoke the tool asynchronously."""
        # For simplicity, we just call the sync version
        return self.invoke(input_data, **kwargs)
    
    def run(self, *args, **kwargs):
        """Run the tool."""
        return self.func(*args, **kwargs)
    
    async def arun(self, *args, **kwargs):
        """Run the tool asynchronously."""
        return self.run(*args, **kwargs)
        
    @property
    def args(self):
        """Get the arguments schema."""
        if self.args_schema:
            return self.args_schema
        # Try to infer from function signature
        import inspect
        sig = inspect.signature(self.func)
        args = {}
        for name, param in sig.parameters.items():
            if name == 'self' or name == 'kwargs':
                continue
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else 'string'
            args[name] = {
                'type': str(param_type).replace("<class '", "").replace("'>", ""),
                'title': name.replace('_', ' ').title()
            }
        return args

def tool(func=None, name=None, description=None, **kwargs):
    """Mock implementation of the tool decorator."""
    if func is None:
        # This is the case when the decorator is called with arguments
        def decorator(f):
            tool_name = name or f.__name__
            tool_description = description or f.__doc__ or ""
            return MockBaseTool(name=tool_name, description=tool_description, func=f, **kwargs)
        return decorator
    else:
        # This is the case when the decorator is called without arguments
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or ""
        return MockBaseTool(name=tool_name, description=tool_description, func=func, **kwargs) 