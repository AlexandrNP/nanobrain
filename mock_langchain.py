"""
Mock implementations of LangChain classes for testing.
This module provides mock implementations of LangChain classes to allow testing
without requiring actual API keys or external services.
"""

from typing import Any, Dict, List, Optional, Union
import os

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
        super().__init__(content, role="user")

class MockAIMessage(MockMessage):
    """Mock implementation of AIMessage."""
    def __init__(self, content: str):
        super().__init__(content, role="assistant")

class MockChatOpenAI:
    """Mock implementation of ChatOpenAI."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        
    def predict(self, text: str, **kwargs) -> str:
        """Return a mock response."""
        return f"Mock response to: {text}"
        
    def predict_messages(self, messages: List[MockMessage], **kwargs) -> MockAIMessage:
        """Return a mock response as an AIMessage."""
        return MockAIMessage("Mock response to messages")

class MockOpenAI:
    """Mock implementation of OpenAI."""
    
    def __init__(self, model_name: str = "text-davinci-003", **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        
    def predict(self, text: str, **kwargs) -> str:
        """Return a mock response."""
        return f"Mock response to: {text}"

class MockConversationBufferMemory:
    """Mock implementation of ConversationBufferMemory."""
    
    def __init__(self, **kwargs):
        self.chat_memory = MockChatMemory()
        self.kwargs = kwargs
        
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory."""
        self.chat_memory.add_user_message(inputs.get("input", ""))
        self.chat_memory.add_ai_message(outputs.get("output", ""))
        
    def load_memory_variables(self, **kwargs) -> Dict[str, Any]:
        """Load memory variables."""
        return {"history": self.chat_memory.messages}

class MockConversationBufferWindowMemory(MockConversationBufferMemory):
    """Mock implementation of ConversationBufferWindowMemory."""
    
    def __init__(self, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        
    def load_memory_variables(self, **kwargs) -> Dict[str, Any]:
        """Load memory variables with window."""
        messages = self.chat_memory.messages[-self.k*2:] if len(self.chat_memory.messages) > self.k*2 else self.chat_memory.messages
        return {"history": messages}

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
        """Clear memory."""
        self.messages = []

class MockPromptTemplate:
    """Mock implementation of PromptTemplate."""
    
    def __init__(self, template: str, input_variables: List[str], **kwargs):
        self.template = template
        self.input_variables = input_variables
        self.kwargs = kwargs
        
    @classmethod
    def from_template(cls, template: str, **kwargs):
        """Create a prompt template from a template string."""
        input_variables = [v.strip("{}") for v in template.split("{") if "}" in v]
        return cls(template=template, input_variables=input_variables, **kwargs)
        
    def format(self, **kwargs) -> str:
        """Format the template with the given values."""
        return self.template.format(**kwargs) 