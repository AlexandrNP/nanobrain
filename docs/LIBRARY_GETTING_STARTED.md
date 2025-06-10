# Getting Started with NanoBrain Library

This guide will help you get up and running with the NanoBrain Library quickly. We'll walk through installation, basic concepts, and build your first application step by step.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Quick Start Examples](#quick-start-examples)
- [Building Your First Application](#building-your-first-application)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Next Steps](#next-steps)

## Prerequisites

Before getting started, ensure you have:

- **Python 3.9+** installed
- **Git** for cloning the repository
- **Basic understanding** of async/await in Python
- **API keys** for AI models (OpenAI, Anthropic, etc.)
- **Optional**: Docker for containerized deployment

### System Requirements

- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: At least 2GB free space
- **Network**: Internet connection for AI model APIs

## Installation

### Option 1: Development Installation

```bash
# Clone the repository
git clone <repository-url>
cd nanobrain

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Option 2: Package Installation

```bash
# Install from PyPI (when available)
pip install nanobrain-library

# Or install from source
pip install git+<repository-url>
```

### Verify Installation

```python
# Test the installation
import asyncio
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig

async def test_installation():
    config = DataUnitConfig(data_type="memory", name="test")
    data_unit = DataUnitMemory(config)
    await data_unit.initialize()
    
    await data_unit.set("Hello, NanoBrain!")
    result = await data_unit.get()
    print(f"Installation test: {result}")
    
    await data_unit.shutdown()

# Run the test
asyncio.run(test_installation())
```

## Core Concepts

### 1. Data Units

Data Units are the foundation of data management in NanoBrain:

```python
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig

# Create a memory-based data unit
config = DataUnitConfig(data_type="memory", name="my_data")
data_unit = DataUnitMemory(config)

# Always initialize before use
await data_unit.initialize()

# Store and retrieve data
await data_unit.set({"message": "Hello World"})
data = await data_unit.get()

# Clean up when done
await data_unit.shutdown()
```

### 2. Enhanced Agents

Enhanced agents provide AI processing with advanced features:

```python
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.core.agent import AgentConfig

# Configure the agent
config = AgentConfig(
    name="my_assistant",
    model="gpt-3.5-turbo",
    temperature=0.7,
    system_prompt="You are a helpful AI assistant."
)

# Create and initialize the agent
agent = EnhancedCollaborativeAgent(config)
await agent.initialize()

# Process requests
response = await agent.process("Hello, how can you help me?")
print(response)

await agent.shutdown()
```

### 3. Parallel Processing

Process multiple requests concurrently using the ParslExecutor:

```python
from nanobrain.core.executor import ParslExecutor, ExecutorConfig
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.core.agent import AgentConfig

# Configure parallel processing with Parsl
executor_config = ExecutorConfig(
    executor_type="parsl",
    max_workers=5
)

executor = ParslExecutor(executor_config)
await executor.initialize()

# Create agents for parallel processing
agents = [
    EnhancedCollaborativeAgent(AgentConfig(name=f"agent_{i}", model="gpt-3.5-turbo"))
    for i in range(3)
]

# Initialize agents
for agent in agents:
    await agent.initialize()

# Process multiple requests in parallel
requests = [
    "What is AI?",
    "Explain machine learning",
    "How does deep learning work?"
]

# Submit tasks to Parsl executor
futures = []
for i, request in enumerate(requests):
    agent = agents[i % len(agents)]
    future = executor.submit(agent.process, request)
    futures.append(future)

# Collect results
results = []
for future in futures:
    result = await future
    results.append(result)

for request, result in zip(requests, results):
    print(f"Q: {request}")
    print(f"A: {result}")

# Cleanup
for agent in agents:
    await agent.shutdown()
await executor.shutdown()
```

### 4. Complete Workflows

Orchestrate complex workflows with all components:

```python
from nanobrain.library.workflows.chat_workflow import ChatWorkflowOrchestrator, ChatWorkflowConfig

# Configure the workflow
config = ChatWorkflowConfig(
    name="my_chat_workflow",
    agents=[
        {
            'name': 'primary_agent',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7
        }
    ],
    enable_parallel_processing=True,
    max_parallel_requests=5,
    database_config={
        'adapter': 'sqlite',
        'connection_string': 'chat.db'
    }
)

# Create and run the workflow
orchestrator = ChatWorkflowOrchestrator(config)
await orchestrator.initialize()

# Process chat messages
response = await orchestrator.process_chat(
    message="Hello, I need help with Python programming",
    user_id="user_123",
    session_id="session_456"
)

print(response.content)
await orchestrator.shutdown()
```

## Quick Start Examples

### Example 1: Simple Chat Bot

Create a basic chat bot in under 20 lines:

```python
import asyncio
from library.agents.enhanced import CollaborativeAgent
from core.agent import AgentConfig

async def simple_chatbot():
    # Configure agent
    config = AgentConfig(
        name="chatbot",
        model="gpt-3.5-turbo",
        system_prompt="You are a friendly chatbot."
    )
    
    # Create agent
    agent = CollaborativeAgent(config)
    await agent.initialize()
    
    # Chat loop
    print("Chatbot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = await agent.process(user_input)
        print(f"Chatbot: {response}")
    
    await agent.shutdown()

# Run the chatbot
asyncio.run(simple_chatbot())
```

### Example 2: Parallel Processing Demo

Process multiple requests simultaneously:

```python
import asyncio
import time
from library.infrastructure.steps import ParallelConversationalAgentStep
from library.agents.enhanced import CollaborativeAgent
from core.agent import AgentConfig

async def parallel_processing_demo():
    # Create multiple agents
    agents = []
    for i in range(3):
        config = AgentConfig(name=f"agent_{i}", model="gpt-3.5-turbo")
        agent = CollaborativeAgent(config)
        agents.append(agent)
    
    # Configure parallel processing
    from library.infrastructure.steps import ParallelConversationalAgentConfig
    config = ParallelConversationalAgentConfig(
        name="parallel_demo",
        max_parallel_requests=10
    )
    
    # Create parallel step
    step = ParallelConversationalAgentStep(config, agents)
    await step.initialize()
    
    # Prepare test requests
    requests = [
        "What is the capital of France?",
        "Explain quantum computing",
        "How do neural networks work?",
        "What is the meaning of life?",
        "Describe the solar system"
    ]
    
    # Process sequentially (for comparison)
    start_time = time.time()
    sequential_results = []
    for request in requests:
        result = await agents[0].process(request)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    # Process in parallel
    start_time = time.time()
    parallel_results = await step.process_batch(requests)
    parallel_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.2f} seconds")
    print(f"Parallel processing: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time / parallel_time:.2f}x")
    
    await step.shutdown()

asyncio.run(parallel_processing_demo())
```

### Example 3: Data Persistence

Store and retrieve conversation history:

```python
import asyncio
from datetime import datetime
from library.infrastructure.data import ConversationHistoryUnit, ConversationMessage
from library.infrastructure.interfaces.database import SQLiteAdapter

async def conversation_history_demo():
    # Setup database
    db_adapter = SQLiteAdapter("conversations.db")
    await db_adapter.initialize()
    
    # Create conversation history unit
    history_unit = ConversationHistoryUnit(
        database_adapter=db_adapter,
        table_name="conversations"
    )
    await history_unit.initialize()
    
    # Save some conversation messages
    messages = [
        ConversationMessage(
            conversation_id="conv_123",
            user_input="Hello, how are you?",
            agent_response="I'm doing well, thank you! How can I help you today?",
            timestamp=datetime.now(),
            response_time_ms=150.0
        ),
        ConversationMessage(
            conversation_id="conv_123",
            user_input="What's the weather like?",
            agent_response="I don't have access to current weather data, but I can help you find weather information.",
            timestamp=datetime.now(),
            response_time_ms=200.0
        )
    ]
    
    for message in messages:
        await history_unit.save_message(message)
    
    # Retrieve conversation history
    history = await history_unit.get_conversation_history("conv_123", limit=10)
    
    print("Conversation History:")
    for msg in history:
        print(f"User: {msg.user_input}")
        print(f"Agent: {msg.agent_response}")
        print(f"Time: {msg.response_time_ms}ms\n")
    
    await history_unit.shutdown()
    await db_adapter.shutdown()

asyncio.run(conversation_history_demo())
```

## Building Your First Application

Let's build a complete chat application with all the features:

### Step 1: Project Structure

Create the following project structure:

```
my_chat_app/
├── config/
│   ├── agents.yaml
│   ├── database.yaml
│   └── workflow.yaml
├── src/
│   └── main.py
├── requirements.txt
└── README.md
```

### Step 2: Configuration Files

**config/agents.yaml**:
```yaml
agents:
  - name: "primary_agent"
    model: "gpt-3.5-turbo"
    temperature: 0.7
    system_prompt: "You are a helpful AI assistant with expertise in various topics."
    
  - name: "technical_agent"
    model: "gpt-4"
    temperature: 0.5
    system_prompt: "You are a technical specialist focused on programming and technology."

delegation_rules:
  - keywords: ["code", "programming", "technical", "debug"]
    agent: "technical_agent"
    confidence_threshold: 0.8
```

**config/database.yaml**:
```yaml
database:
  adapter: "sqlite"
  connection_string: "chat_app.db"
  enable_wal_mode: true
  
conversation_history:
  table_name: "conversations"
  enable_search: true
  retention_days: 30
```

**config/workflow.yaml**:
```yaml
workflow:
  name: "my_chat_application"
  description: "A complete chat application with NanoBrain Library"
  
parallel_processing:
  enable: true
  max_parallel_requests: 10
  load_balancing_strategy: "fastest_response"
  
session_management:
  storage_backend: "database"
  default_timeout: 3600  # 1 hour
  enable_persistence: true
  
monitoring:
  enable_performance_tracking: true
  enable_health_monitoring: true
  metrics_collection_interval: 60.0
```

### Step 3: Main Application

**src/main.py**:
```python
import asyncio
import yaml
from pathlib import Path
from library.workflows.chat_workflow import ChatWorkflowOrchestrator, ChatWorkflowConfig
from library.infrastructure.interfaces.cli import InteractiveCLI, CLIConfig

class ChatApplication:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.orchestrator = None
        self.cli = None
    
    async def initialize(self):
        """Initialize the chat application."""
        # Load configuration
        config = await self.load_configuration()
        
        # Initialize workflow orchestrator
        self.orchestrator = ChatWorkflowOrchestrator(config)
        await self.orchestrator.initialize()
        
        # Initialize CLI
        cli_config = CLIConfig(
            app_name="My Chat App",
            prompt_style="🤖 >>> ",
            enable_colors=True,
            enable_history=True
        )
        self.cli = InteractiveCLI(cli_config)
        await self.cli.initialize()
    
    async def load_configuration(self) -> ChatWorkflowConfig:
        """Load configuration from YAML files."""
        # Load individual config files
        with open(self.config_dir / "agents.yaml") as f:
            agents_config = yaml.safe_load(f)
        
        with open(self.config_dir / "database.yaml") as f:
            database_config = yaml.safe_load(f)
        
        with open(self.config_dir / "workflow.yaml") as f:
            workflow_config = yaml.safe_load(f)
        
        # Combine into workflow configuration
        return ChatWorkflowConfig(
            name=workflow_config["workflow"]["name"],
            description=workflow_config["workflow"]["description"],
            agents=agents_config["agents"],
            delegation_rules=agents_config.get("delegation_rules", []),
            enable_parallel_processing=workflow_config["parallel_processing"]["enable"],
            max_parallel_requests=workflow_config["parallel_processing"]["max_parallel_requests"],
            load_balancing_strategy=workflow_config["parallel_processing"]["load_balancing_strategy"],
            database_config=database_config["database"],
            session_config={
                "storage_backend": workflow_config["session_management"]["storage_backend"],
                "default_timeout": workflow_config["session_management"]["default_timeout"],
                "enable_persistence": workflow_config["session_management"]["enable_persistence"]
            },
            monitoring_config=workflow_config["monitoring"]
        )
    
    async def run(self):
        """Run the main chat loop."""
        await self.cli.print_header("Welcome to My Chat App")
        await self.cli.print_info("Type 'help' for commands, 'quit' to exit")
        
        # Create user session
        user_id = "user_123"  # In real app, get from authentication
        session_id = await self.orchestrator.create_session(user_id)
        
        while True:
            try:
                # Get user input
                user_input = await self.cli.get_input("You: ")
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'help':
                    await self.show_help()
                    continue
                elif user_input.lower() == 'stats':
                    await self.show_statistics()
                    continue
                elif user_input.lower() == 'history':
                    await self.show_conversation_history(session_id)
                    continue
                
                # Process with workflow
                with self.cli.progress_context("Processing..."):
                    response = await self.orchestrator.process_chat(
                        message=user_input,
                        user_id=user_id,
                        session_id=session_id
                    )
                
                # Display response
                await self.cli.print_response(f"Assistant: {response.content}")
                
                # Show processing time if verbose
                if response.metadata.get('show_timing'):
                    await self.cli.print_info(
                        f"Response time: {response.metadata.get('processing_time', 0):.2f}ms"
                    )
                
            except KeyboardInterrupt:
                await self.cli.print_info("\nGoodbye!")
                break
            except Exception as e:
                await self.cli.print_error(f"Error: {e}")
    
    async def show_help(self):
        """Show help information."""
        help_text = """
Available commands:
  help     - Show this help message
  stats    - Show performance statistics
  history  - Show conversation history
  quit     - Exit the application
        """
        await self.cli.print_info(help_text)
    
    async def show_statistics(self):
        """Show performance statistics."""
        stats = await self.orchestrator.get_performance_stats()
        
        stats_text = f"""
Performance Statistics:
  Total requests: {stats.get('total_requests', 0)}
  Average response time: {stats.get('avg_response_time', 0):.2f}ms
  Success rate: {stats.get('success_rate', 0):.2%}
  Active sessions: {stats.get('active_sessions', 0)}
        """
        await self.cli.print_info(stats_text)
    
    async def show_conversation_history(self, session_id: str):
        """Show recent conversation history."""
        history = await self.orchestrator.get_conversation_history(session_id, limit=5)
        
        await self.cli.print_info("Recent conversation history:")
        for msg in history:
            await self.cli.print_info(f"You: {msg.user_input}")
            await self.cli.print_info(f"Assistant: {msg.agent_response}")
            await self.cli.print_info("---")
    
    async def shutdown(self):
        """Clean up resources."""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        if self.cli:
            await self.cli.shutdown()

async def main():
    """Main application entry point."""
    app = ChatApplication()
    
    try:
        await app.initialize()
        await app.run()
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 4: Requirements File

**requirements.txt**:
```
nanobrain-library
pyyaml
aiofiles
asyncio
```

### Step 5: Run Your Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run the application
python src/main.py
```

## Configuration

### Environment Variables

Set up your environment variables:

```bash
# AI Model API Keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Database Configuration
export DATABASE_URL="postgresql://user:pass@localhost/nanobrain"

# Redis Configuration (for caching/sessions)
export REDIS_URL="redis://localhost:6379/0"

# Application Settings
export NANOBRAIN_LOG_LEVEL="INFO"
export NANOBRAIN_DEBUG="false"
```

### Configuration Files

Use YAML configuration files for complex setups:

```yaml
# config/production.yaml
environment: "production"

agents:
  - name: "primary_agent"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2000
    
database:
  adapter: "postgresql"
  host: "${DATABASE_HOST}"
  database: "${DATABASE_NAME}"
  user: "${DATABASE_USER}"
  password: "${DATABASE_PASSWORD}"
  pool_size: 20
  
caching:
  backend: "redis"
  url: "${REDIS_URL}"
  ttl: 300
  
monitoring:
  enable_metrics: true
  metrics_backend: "prometheus"
  health_check_interval: 30
```

## Best Practices

### 1. Resource Management

Always properly initialize and clean up resources:

```python
# Use try/finally blocks
async def use_component():
    component = SomeComponent(config)
    try:
        await component.initialize()
        # Use component
        result = await component.process(data)
        return result
    finally:
        await component.shutdown()

# Or use context managers when available
async with SomeComponent(config) as component:
    result = await component.process(data)
```

### 2. Error Handling

Implement comprehensive error handling:

```python
from library.infrastructure.data import DataUnitError
from library.agents.enhanced import AgentError

try:
    result = await agent.process(user_input)
except AgentError as e:
    logger.error(f"Agent processing failed: {e}")
    # Implement fallback logic
    result = "I'm sorry, I'm having trouble processing your request."
except DataUnitError as e:
    logger.error(f"Data storage failed: {e}")
    # Handle data persistence issues
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected errors gracefully
```

### 3. Configuration Management

Use configuration files and environment variables:

```python
import os
from pathlib import Path

class Config:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///app.db")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Load from config file if exists
        config_file = Path("config/app.yaml")
        if config_file.exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: Path):
        import yaml
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
        
        # Override with file values
        for key, value in config_data.items():
            setattr(self, key, value)
```

### 4. Logging

Set up proper logging:

```python
import logging
from library.infrastructure.logging import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use library logger
logger = get_logger("my_app")

# Log important events
logger.info("Application starting")
logger.debug("Processing user input: %s", user_input)
logger.error("Failed to process request: %s", error)
```

### 5. Testing

Write tests for your components:

```python
import pytest
from library.agents.enhanced import CollaborativeAgent
from core.agent import AgentConfig

@pytest.fixture
async def test_agent():
    config = AgentConfig(name="test_agent", model="gpt-3.5-turbo")
    agent = CollaborativeAgent(config)
    await agent.initialize()
    yield agent
    await agent.shutdown()

async def test_agent_processing(test_agent):
    response = await test_agent.process("Hello")
    assert response is not None
    assert len(response) > 0

async def test_agent_error_handling(test_agent):
    # Test with invalid input
    with pytest.raises(ValueError):
        await test_agent.process("")
```

## Next Steps

### 1. Explore Advanced Features

- **Protocol Support**: Learn about A2A and MCP integration
- **Load Balancing**: Implement custom load balancing strategies
- **Monitoring**: Set up comprehensive monitoring and alerting
- **Security**: Add authentication and authorization

### 2. Scale Your Application

- **Horizontal Scaling**: Deploy multiple instances
- **Database Optimization**: Use PostgreSQL or other production databases
- **Caching**: Implement Redis caching for better performance
- **Container Deployment**: Use Docker and Kubernetes

### 3. Customize Components

- **Custom Agents**: Create specialized agents for your use case
- **Custom Data Units**: Implement custom storage backends
- **Custom Workflows**: Build domain-specific workflows
- **Plugins**: Develop plugins for additional functionality

### 4. Production Deployment

- **Environment Configuration**: Set up staging and production environments
- **Monitoring**: Implement comprehensive monitoring and logging
- **Security**: Add proper authentication, authorization, and encryption
- **Performance**: Optimize for your specific use case

### 5. Community and Support

- **Documentation**: Read the complete documentation
- **Examples**: Explore more examples in the examples directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the right directory and virtual environment
cd nanobrain
source venv/bin/activate
pip install -e .
```

**2. API Key Issues**
```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"

# Or create a .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

**3. Database Connection Issues**
```python
# Check database configuration
from library.infrastructure.interfaces.database import SQLiteAdapter

adapter = SQLiteAdapter("test.db")
try:
    await adapter.initialize()
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")
```

**4. Performance Issues**
```python
# Monitor performance
from library.infrastructure.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
stats = await monitor.get_current_stats()
print(f"Memory usage: {stats.memory_usage}")
print(f"CPU usage: {stats.cpu_usage}")
```

### Getting Help

- **Documentation**: Check the [complete documentation](README.md)
- **Examples**: Look at examples in the `examples/` directory
- **Issues**: Create an issue on GitHub
- **Discussions**: Join community discussions

---

Congratulations! You now have a solid foundation for building applications with the NanoBrain Library. Start with the simple examples and gradually explore more advanced features as you become comfortable with the framework.

*For more detailed information, see the [complete documentation](README.md) and [architecture guide](LIBRARY_ARCHITECTURE.md).* 