"""
NanoBrain Framework - A Comprehensive AI Agent Framework
=========================================================

NanoBrain is a production-ready, event-driven AI agent framework designed for building
sophisticated multi-agent systems with distributed execution capabilities. The framework
follows a configuration-driven architecture that enables complex AI workflows through
declarative YAML configurations.

**Core Philosophy:**
    - **Configuration-Driven**: All behavior controlled via YAML configurations
    - **Event-Driven Architecture**: Components communicate through data flows and triggers
    - **Mandatory from_config Pattern**: All components use unified creation patterns
    - **Zero Hardcoding**: Complete system flexibility through configuration
    - **Production-Ready**: Enterprise-grade reliability and scalability

**Architecture Overview:**
    The framework is built on several foundational concepts:
    
    * **Agents**: AI entities that process requests using LLM integration and tool calling
    * **Steps**: Processing units that transform data through configurable operations
    * **Workflows**: Orchestrate multi-step processing through agent and step coordination
    * **Data Units**: Type-safe data containers for inter-component communication
    * **Links**: Define data flow connections between workflow components
    * **Triggers**: Event-driven activation mechanisms for workflow execution
    * **Executors**: Pluggable execution backends (local, threaded, distributed via Parsl)
    * **Tools**: Extensible tool system for agent capabilities (LangChain compatible)

**Key Features:**
    - **Multi-Agent Collaboration**: Agent-to-Agent (A2A) protocol support
    - **Tool Integration**: Model Context Protocol (MCP) and LangChain tool compatibility
    - **Distributed Execution**: Parsl integration for high-performance computing
    - **Bioinformatics Support**: Specialized tools for computational biology workflows
    - **Web Interfaces**: Universal web interface for workflow interaction
    - **Comprehensive Logging**: Structured logging with performance monitoring
    - **Configuration Management**: Recursive YAML configuration with validation

**Quick Start Example:**
    ```python
    from nanobrain import ConversationalAgent, AgentConfig
    
    # Create agent from configuration
    agent = ConversationalAgent.from_config('config/my_agent.yml')
    
    # Process request
    response = await agent.aprocess("Hello, what can you help me with?")
    print(response.content)
    ```

**Configuration Example:**
    ```yaml
    # my_agent.yml
    name: "helpful_assistant"
    description: "A helpful AI assistant"
    model: "gpt-4"
    temperature: 0.7
    system_prompt: "You are a helpful AI assistant."
    tools:
      - class: "nanobrain.library.tools.WebSearchTool"
        config: "config/web_search.yml"
    ```

**Framework Components:**
    - **Core Framework** (:mod:`nanobrain.core`): Essential building blocks
    - **Component Library** (:mod:`nanobrain.library`): Reusable implementations
    - **Configuration System** (:mod:`nanobrain.config`): Advanced configuration management

**Getting Started:**
    1. Install: ``pip install nanobrain``
    2. Create configuration files for your components
    3. Use the ``from_config`` pattern to instantiate components
    4. Build workflows by connecting components with links and triggers
    5. Execute using appropriate executors (local, distributed, etc.)

**Advanced Usage:**
    The framework supports complex scenarios including:
    - Multi-agent collaboration with delegation and consensus
    - Distributed workflow execution across compute clusters
    - Real-time data processing with event-driven triggers
    - Integration with external systems via tool protocols
    - Custom component development following framework patterns

**Documentation:**
    - **API Reference**: Complete class and method documentation
    - **User Guide**: Step-by-step tutorials and examples  
    - **Developer Guide**: Framework extension and customization
    - **Configuration Reference**: Complete YAML configuration options

**Version:** 0.1.0  
**Authors:** NanoBrain Team  
**License:** MIT  
**Repository:** https://github.com/nanobrain/nanobrain
"""

__version__ = "0.1.0"
__author__ = "NanoBrain Team"
__email__ = "team@nanobrain.ai"

# Core framework imports
from . import core
# from . import library  # Temporarily disabled
from . import config

# Convenience imports for common use cases
from .core.agent import ConversationalAgent, AgentConfig
from .core.executor import LocalExecutor, ParslExecutor, ExecutorConfig
from .core.data_unit import DataUnitMemory, DataUnitConfig
from .core.step import Step, StepConfig
from .core.trigger import DataUpdatedTrigger, TriggerConfig
from .core.link import DirectLink, LinkConfig

# Configuration imports
from .core.config.config_manager import get_config_manager

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Modules
    'core',
    # 'library',  # Temporarily disabled
    'config',
    
    # Core classes
    'ConversationalAgent',
    'AgentConfig',
    'LocalExecutor',
    'ParslExecutor',
    'ExecutorConfig',
    'DataUnitMemory',
    'DataUnitConfig',
    'Step',
    'StepConfig',
    'DataUpdatedTrigger',
    'TriggerConfig',
    'DirectLink',
    'LinkConfig',
    
    # Configuration
    'get_config_manager',
] 