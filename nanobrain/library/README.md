# NanoBrain Library

A comprehensive collection of reusable components, workflows, and infrastructure elements built on top of the NanoBrain framework core.

## Overview

The NanoBrain Library provides a structured, extensible collection of components that demonstrate best practices for building complex AI workflows. It showcases proper interconnections between components via data units, links, and triggers, following the enhanced file structure with individual step directories.

## Directory Structure

```
library/
├── __init__.py                          # Main library entry point
├── README.md                            # This file
├── test_library_structure.py            # Library verification tests
├── agents/                              # Specialized and conversational agents
│   ├── __init__.py
│   ├── conversational/                  # Enhanced conversational agents
│   │   ├── __init__.py
│   │   ├── enhanced_collaborative_agent.py
│   │   ├── enhanced_collaborative_agent.yml
│   │   └── README.md
│   └── specialized/                     # Task-specific agents
│       ├── __init__.py
│       ├── code_writer.py               # Moved from src/agents/
│       ├── code_writer.yml              # Moved from src/agents/config/
│       ├── file_writer.py               # Moved from src/agents/
│       ├── file_writer.yml              # Moved from src/agents/config/
│       └── README.md
├── infrastructure/                      # Custom infrastructure components
│   ├── __init__.py
│   ├── data_units/                      # Specialized data units
│   │   ├── __init__.py
│   │   ├── conversation_history_unit.py
│   │   ├── conversation_history_unit.yml
│   │   └── README.md
│   ├── triggers/                        # Advanced triggers
│   │   ├── __init__.py
│   │   └── README.md
│   ├── links/                           # Specialized links
│   │   ├── __init__.py
│   │   └── README.md
│   └── steps/                           # Enhanced steps
│       ├── __init__.py
│       └── README.md
└── workflows/                           # Complete workflow implementations
    ├── __init__.py
    └── chat_workflow/                   # Enhanced chat workflow
        ├── __init__.py
        ├── chat_workflow.py             # Main workflow implementation
        ├── chat_workflow.yml            # Workflow configuration
        ├── requirements.txt             # Workflow dependencies
        ├── README.md                    # Workflow documentation
        └── steps/                       # Individual step directories
            ├── __init__.py
            ├── cli_interface_step/      # CLI interface step
            │   ├── __init__.py
            │   ├── cli_interface_step.py
            │   ├── cli_interface_step.yml
            │   ├── cli_interface_step.txt
            │   └── README.md
            ├── conversation_manager_step/
            │   ├── __init__.py
            │   ├── conversation_manager_step.py
            │   ├── conversation_manager_step.yml
            │   ├── conversation_manager_step.txt
            │   ├── README.md
            │   └── substeps/            # Nested substeps
            │       ├── __init__.py
            │       ├── history_persistence_step/
            │       └── performance_tracking_step/
            └── agent_processing_step/
                ├── __init__.py
                ├── agent_processing_step.py
                ├── agent_processing_step.yml
                ├── agent_processing_step.txt
                └── README.md
```

## Key Features

### 1. Modular Architecture
- **Individual Step Directories**: Each step has its own directory with implementation, configuration, and documentation
- **Hierarchical Composition**: Steps can contain substeps for complex processing
- **Proper Interconnections**: Components are connected via data units, links, and triggers

### 2. Enhanced Agents
- **Specialized Agents**: Task-specific agents for code generation, file management, etc.
- **Conversational Agents**: Enhanced agents with A2A and MCP protocol support
- **Performance Tracking**: Built-in metrics and monitoring capabilities

### 3. Infrastructure Components
- **Custom Data Units**: Specialized data storage with persistence and search
- **Advanced Triggers**: Event-driven processing with complex conditions
- **Specialized Links**: Custom data flow patterns and load balancing

### 4. Complete Workflows
- **Chat Workflow**: Comprehensive chat implementation with step hierarchy
- **Proper Data Flow**: Demonstrates correct use of data units, links, and triggers
- **Configuration-Driven**: YAML-based configuration for all components

## Quick Start

### 1. Basic Usage

```python
import asyncio
from library.workflows.chat_workflow.chat_workflow import create_chat_workflow

async def main():
    # Create and initialize workflow
    workflow = await create_chat_workflow()
    
    # Process user input
    response = await workflow.process_user_input("Hello, how are you?")
    print(f"Response: {response}")
    
    # Shutdown
    await workflow.shutdown()

asyncio.run(main())
```

### 2. Using Enhanced Collaborative Agent

```python
import asyncio
from library.agents.conversational.enhanced_collaborative_agent import EnhancedCollaborativeAgent
from core.agent import AgentConfig

async def main():
    # Configure agent
    config = AgentConfig(
        name="my_assistant",
        model="gpt-3.5-turbo",
        temperature=0.7,
        system_prompt="You are a helpful assistant."
    )
    
    # Create agent with delegation rules
    delegation_rules = [
        {
            'keywords': ['code', 'programming'],
            'agent': 'code_specialist',
            'description': 'Delegate coding tasks'
        }
    ]
    
    agent = EnhancedCollaborativeAgent(
        config,
        delegation_rules=delegation_rules,
        enable_metrics=True
    )
    
    await agent.initialize()
    
    # Process input
    response = await agent.process("Write a Python function to sort a list")
    print(response)
    
    # Get performance metrics
    status = agent.get_enhanced_status()
    print(f"Collaboration count: {status['collaboration_count']}")
    
    await agent.shutdown()

asyncio.run(main())
```

### 3. Using Conversation History Data Unit

```python
import asyncio
from library.infrastructure.data_units.conversation_history_unit import (
    ConversationHistoryUnit, ConversationMessage
)
from core.data_unit import DataUnitConfig
from datetime import datetime

async def main():
    # Create conversation history unit
    config = DataUnitConfig(
        name="chat_history",
        data_type="conversation_history"
    )
    
    history_unit = ConversationHistoryUnit(
        config,
        db_path="my_chat_history.db"
    )
    
    await history_unit.initialize()
    
    # Save a conversation message
    message = ConversationMessage(
        timestamp=datetime.now(),
        user_input="Hello!",
        agent_response="Hi there! How can I help you?",
        response_time_ms=150.0,
        conversation_id="conv_001",
        message_id=1
    )
    
    await history_unit.save_message(message)
    
    # Retrieve conversation history
    history = await history_unit.get_conversation_history("conv_001")
    print(f"Retrieved {len(history)} messages")
    
    # Search conversations
    results = await history_unit.search_conversations("hello")
    print(f"Found {len(results)} matching messages")
    
    await history_unit.shutdown()

asyncio.run(main())
```

## Component Interconnections

The library demonstrates proper component interconnections following NanoBrain best practices:

### Data Flow Pattern
```
User Input → Data Unit → Trigger → Step → Data Unit → Link → Next Step
```

### Example: Chat Workflow Data Flow
```
CLI Input → user_input (DataUnit) → user_input_trigger (Trigger) → 
conversation_manager_step (Step) → agent_input (DataUnit) → 
agent_input_trigger (Trigger) → agent_processing_step (Step) → 
agent_output (DataUnit) → agent_to_cli (Link) → CLI Output
```

### Hierarchical Step Composition
```
ConversationManagerStep
├── HistoryPersistenceStep (substep)
│   ├── Data persistence logic
│   └── Database operations
└── PerformanceTrackingStep (substep)
    ├── Metrics collection
    └── Performance analysis
```

## Configuration

All components support YAML-based configuration:

### Agent Configuration
```yaml
name: "EnhancedCollaborativeAgent"
description: "Advanced conversational agent"
config:
  model: "gpt-4-turbo"
  temperature: 0.7
  delegation_rules:
    - keywords: ["code", "programming"]
      agent: "code_specialist"
```

### Workflow Configuration
```yaml
name: "ChatWorkflow"
description: "Enhanced chat workflow"
steps:
  cli_interface_step:
    directory: "steps/cli_interface_step"
    class: "CLIInterfaceStep"
    config:
      prompt_prefix: "Chat> "
      show_timestamps: true
```

## Testing

Run the library verification tests:

```bash
cd nanobrain/library
python test_library_structure.py
```

This will test:
- Import functionality for all components
- Workflow creation and initialization
- Agent functionality and metrics
- Data unit operations and persistence

## Migration from src/agents

The library includes agents moved from `src/agents/` with updated import paths:

**Old imports:**
```python
from src.agents.code_writer import CodeWriterAgent
from src.agents.file_writer import FileWriterAgent
```

**New imports:**
```python
from library.agents.specialized.code_writer import CodeWriterAgent
from library.agents.specialized.file_writer import FileWriterAgent
```

## Best Practices

### 1. Step Design
- Each step should have its own directory
- Include implementation, configuration, and documentation
- Use substeps for complex hierarchical processing
- Implement proper error handling and logging

### 2. Data Flow
- Use data units for state management
- Connect components with appropriate links
- Use triggers for event-driven processing
- Maintain clear data flow documentation

### 3. Configuration
- Use YAML files for all configuration
- Include validation schemas
- Provide examples and templates
- Document all configuration options

### 4. Testing
- Create unit tests for each component
- Test component interactions
- Verify data flow and triggers
- Include integration tests for workflows

## Contributing

When adding new components to the library:

1. Follow the established directory structure
2. Include proper documentation and configuration
3. Implement comprehensive tests
4. Update the main library `__init__.py` files
5. Add examples to this README

## Dependencies

Core dependencies are managed by the main NanoBrain framework. Additional dependencies for specific components are listed in their respective `requirements.txt` files.

## License

This library is part of the NanoBrain framework and follows the same licensing terms. 