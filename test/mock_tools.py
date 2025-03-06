"""
Mock tools for testing purposes.
"""

from unittest.mock import MagicMock
from typing import List, Dict, Any, Optional
import os


class MockStep:
    """Base mock Step class for testing."""
    
    def __init__(self, executor):
        """Initialize the mock Step."""
        self.executor = executor
        self.input_data = None
        self.output_data = None
        self.__class__.__name__ = "MockStep"
    
    async def process(self, inputs):
        """Process the inputs."""
        return "Mock result"


class StepFileWriter(MockStep):
    """Mock StepFileWriter for testing."""
    
    def __init__(self, executor):
        """Initialize the StepFileWriter."""
        super().__init__(executor)
        self.__class__.__name__ = "StepFileWriter"
    
    async def process(self, inputs):
        """Process the inputs."""
        return {"success": True, "message": "File written successfully"}
    
    async def create_file(self, file_path, content):
        """Create a file with the given content."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return {"success": True, "message": f"File created: {file_path}"}


class StepPlanner(MockStep):
    """Mock StepPlanner for testing."""
    
    def __init__(self, executor):
        """Initialize the StepPlanner."""
        super().__init__(executor)
        self.__class__.__name__ = "StepPlanner"
    
    async def process(self, inputs):
        """Process the inputs."""
        return {"success": True, "message": "Plan created successfully"}
    
    async def plan_step(self, step_name, description, base_class):
        """Plan a step implementation."""
        return {
            "success": True, 
            "plan": f"Plan for {step_name}: {description}", 
            "message": f"Created plan for {step_name}"
        }


class StepCoder(MockStep):
    """Mock StepCoder for testing."""
    
    def __init__(self, executor):
        """Initialize the StepCoder."""
        super().__init__(executor)
        self.__class__.__name__ = "StepCoder"
    
    async def process(self, inputs):
        """Process the inputs."""
        return {"success": True, "message": "Code generated successfully"}
    
    async def generate_code(self, specification, output_file=None):
        """Generate code based on the specification."""
        code = f"# Generated code for {specification}\n\ndef main():\n    return 'Hello World'\n"
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(code)
        return {"success": True, "code": code}


class StepGitInit(MockStep):
    """Mock StepGitInit for testing."""
    
    def __init__(self, executor):
        """Initialize the StepGitInit."""
        super().__init__(executor)
        self.__class__.__name__ = "StepGitInit"
    
    async def process(self, inputs):
        """Process the inputs."""
        repo_path = inputs[0] if len(inputs) > 0 else "."
        repo_name = inputs[1] if len(inputs) > 1 else os.path.basename(repo_path)
        return {"success": True, "message": f"Git repository initialized at {repo_path}"}


class StepContextSearch(MockStep):
    """Mock StepContextSearch for testing."""
    
    def __init__(self, executor):
        """Initialize the StepContextSearch."""
        super().__init__(executor)
        self.__class__.__name__ = "StepContextSearch"
    
    async def process(self, inputs):
        """Process the inputs."""
        query = inputs[0] if len(inputs) > 0 else ""
        return {"success": True, "results": [{"content": f"Mock result for query: {query}", "relevance": 0.8}]}
    
    async def search_all(self, query, context_provider=None):
        """Search all contexts."""
        return {"success": True, "results": [{"content": f"Mock result for query: {query}", "relevance": 0.8}]}


class StepWebSearch(MockStep):
    """Mock StepWebSearch for testing."""
    
    def __init__(self, executor):
        """Initialize the StepWebSearch."""
        super().__init__(executor)
        self.__class__.__name__ = "StepWebSearch"
    
    async def process(self, inputs):
        """Process the inputs."""
        query = inputs[0] if len(inputs) > 0 else ""
        return {"success": True, "results": [{"title": f"Mock result for: {query}", "snippet": "This is a mock search result", "url": "https://example.com"}]}
    
    async def search(self, query, search_engine=None, num_results=None):
        """Search the web."""
        return {"success": True, "results": [{"title": f"Mock result for: {query}", "snippet": "This is a mock search result", "url": "https://example.com"}]}


# Create mock instances for easy access
StepFileWriter_mock = MagicMock()
StepFileWriter_mock.return_value = StepFileWriter(None)
StepFileWriter_mock.return_value.__class__.__name__ = "StepFileWriter"

StepPlanner_mock = MagicMock()
StepPlanner_mock.return_value = StepPlanner(None)
StepPlanner_mock.return_value.__class__.__name__ = "StepPlanner"

StepCoder_mock = MagicMock()
StepCoder_mock.return_value = StepCoder(None)
StepCoder_mock.return_value.__class__.__name__ = "StepCoder"

StepGitInit_mock = MagicMock()
StepGitInit_mock.return_value = StepGitInit(None)
StepGitInit_mock.return_value.__class__.__name__ = "StepGitInit"

StepContextSearch_mock = MagicMock()
StepContextSearch_mock.return_value = StepContextSearch(None)
StepContextSearch_mock.return_value.__class__.__name__ = "StepContextSearch"

StepWebSearch_mock = MagicMock()
StepWebSearch_mock.return_value = StepWebSearch(None)
StepWebSearch_mock.return_value.__class__.__name__ = "StepWebSearch" 