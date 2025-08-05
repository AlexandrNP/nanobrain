"""
NanoBrain Component Library - Production-Ready Implementations
==============================================================

The NanoBrain Library provides a comprehensive collection of production-ready, 
reusable components that extend the core framework with specialized implementations
for real-world applications. All library components follow the framework's
configuration-driven architecture and ``from_config`` patterns.

**Library Philosophy:**
    The library bridges the gap between core framework primitives and application
    requirements by providing:
    
    * **Production-Ready Components**: Battle-tested implementations for common use cases
    * **Specialized Domains**: Components optimized for specific domains (bioinformatics, web interfaces)
    * **Extensible Architecture**: Components designed for customization and extension
    * **Configuration-Driven**: All behavior controlled through YAML configurations
    * **Interoperability**: Components work seamlessly together and with core framework

**Component Categories:**

**Agent Systems:**
    Advanced agent implementations with specialized capabilities:
    
    * **Enhanced Collaborative Agents**: Multi-protocol agents with A2A and MCP support
      enabling sophisticated agent-to-agent collaboration and tool integration
    * **Conversational Agents**: Context-aware agents with conversation history management
    * **Specialized Agents**: Domain-specific agents (code writers, file processors, analyzers)
    * **Agent Mixins**: Reusable agent capabilities for custom agent development

**Infrastructure Components:**
    Core infrastructure services and specialized data management:
    
    * **Data Management**: Advanced data units with persistence, caching, and synchronization
    * **Step Implementations**: Parallel processing steps, load balancing, and monitoring
    * **Interface Systems**: Database adapters, CLI interfaces, and system integrations
    * **Deployment Tools**: Docker containers, port management, and service orchestration
    * **Monitoring Systems**: Performance monitoring, health checking, and metrics collection

**Workflow Orchestration:**
    Pre-built workflows for common application patterns:
    
    * **Chat Workflows**: Complete conversational AI workflows with history management
    * **Analysis Pipelines**: Data analysis workflows with parallel processing
    * **Integration Workflows**: System integration patterns for external services
    * **Bioinformatics Workflows**: Computational biology pipelines with tool integration

**Tool Ecosystem:**
    Comprehensive tool integrations for agent capabilities:
    
    * **Bioinformatics Tools**: BV-BRC, MMseqs2, MUSCLE, PubMed integration
    * **Search Tools**: Elasticsearch, web search, document retrieval
    * **Analysis Tools**: Data processors, statistical analyzers, visualization
    * **Integration Tools**: API clients, database connectors, file processors

**Web Interface System:**
    Universal web interface components for workflow interaction:
    
    * **Universal Server**: Multi-workflow server with automatic request routing
    * **Request Analysis**: Intent classification and domain-specific routing
    * **Response Processing**: Format conversion, streaming, and aggregation
    * **Frontend Integration**: React components and API client libraries

**Bioinformatics Suite:**
    Specialized components for computational biology applications:
    
    * **Sequence Management**: FASTA parsing, validation, and analysis
    * **Tool Wrappers**: Integration with popular bioinformatics software
    * **Data Structures**: Biological data units for genomic and proteomic data
    * **Analysis Pipelines**: Pre-built workflows for common bioinformatics tasks

**Key Features:**

**Production Readiness:**
    * **Comprehensive Testing**: Full test coverage with integration tests
    * **Error Handling**: Robust error handling with detailed diagnostics
    * **Performance Optimization**: Optimized for production workloads
    * **Scalability**: Designed for horizontal and vertical scaling
    * **Monitoring Integration**: Built-in metrics and health monitoring

**Configuration Management:**
    * **Schema Validation**: Comprehensive Pydantic schemas for all components
    * **Environment Support**: Development, staging, and production configurations
    * **Template System**: Reusable configuration templates and patterns
    * **Validation**: Pre-deployment configuration validation and testing

**Interoperability:**
    * **Framework Integration**: Seamless integration with core framework components
    * **Protocol Support**: A2A (Agent-to-Agent) and MCP (Model Context Protocol)
    * **Tool Compatibility**: LangChain tool compatibility and custom tool support
    * **API Standards**: RESTful APIs and standardized interfaces

**Usage Patterns:**

**Enhanced Agent Creation:**
    ```python
    from nanobrain.library import EnhancedCollaborativeAgent
    
    # Create advanced agent with multi-protocol support
    agent = EnhancedCollaborativeAgent.from_config('config/enhanced_agent.yml')
    
    # Agent automatically supports A2A collaboration and MCP tools
    response = await agent.aprocess("Analyze this data with collaboration")
    ```

**Chat Workflow Implementation:**
    ```python
    from nanobrain.library import ChatWorkflow
    
    # Create complete chat workflow
    workflow = ChatWorkflow.from_config('config/chat_workflow.yml')
    
    # Workflow includes conversation history, agent processing, and response formatting
    result = await workflow.execute(user_input="Hello, how can you help me?")
    ```

**Bioinformatics Tool Integration:**
    ```python
    from nanobrain.library.tools.bioinformatics import BVBRCTool
    
    # Create bioinformatics tool
    bvbrc = BVBRCTool.from_config('config/bvbrc_tool.yml')
    
    # Use tool for viral genome analysis
    genomes = await bvbrc.get_viral_genomes("alphavirus")
    ```

**Universal Web Interface:**
    ```python
    from nanobrain.library.interfaces.web import UniversalNanoBrainServer
    
    # Create universal server supporting multiple workflows
    server = UniversalNanoBrainServer.from_config('config/universal_server.yml')
    
    # Server automatically routes requests to appropriate workflows
    await server.start()
    ```

**Configuration Examples:**

**Enhanced Agent Configuration:**
    ```yaml
    name: "collaborative_researcher"
    description: "Research agent with collaboration capabilities"
    model: "gpt-4"
    temperature: 0.3
    
    # A2A collaboration settings
    a2a_support:
      enabled: true
      max_delegation_depth: 3
      collaboration_timeout: 300
    
    # MCP tool integration
    mcp_support:
      enabled: true
      server_configs:
        - name: "filesystem"
          config: "config/mcp_filesystem.yml"
        - name: "search"
          config: "config/mcp_search.yml"
    
    # Traditional tools
    tools:
      - class: "nanobrain.library.tools.bioinformatics.BVBRCTool"
        config: "config/bvbrc_tool.yml"
    ```

**Chat Workflow Configuration:**
    ```yaml
    name: "intelligent_chat"
    description: "Complete chat workflow with history and analysis"
    execution_strategy: "event_driven"
    
    steps:
      - class: "nanobrain.library.infrastructure.steps.ConversationManagerStep"
        config: "config/conversation_manager.yml"
      - class: "nanobrain.library.infrastructure.steps.AgentProcessingStep"
        config: "config/agent_processing.yml"
      - class: "nanobrain.library.infrastructure.steps.ResponseFormattingStep"
        config: "config/response_formatting.yml"
    
    data_units:
      conversation_history:
        class: "nanobrain.library.infrastructure.data.ConversationHistoryUnit"
        config: "config/conversation_history.yml"
    ```

**Advanced Features:**

**Multi-Agent Collaboration:**
    * **Agent-to-Agent Protocol**: Standardized communication between agents
    * **Delegation Patterns**: Intelligent task delegation based on agent capabilities
    * **Consensus Mechanisms**: Multi-agent decision making and validation
    * **Performance Tracking**: Collaboration efficiency metrics and optimization

**Real-Time Processing:**
    * **Event-Driven Architecture**: Responsive processing based on data changes
    * **Streaming Support**: Real-time data processing and response streaming
    * **Load Balancing**: Automatic load distribution across processing units
    * **Circuit Breakers**: Fault tolerance and graceful degradation

**Integration Ecosystem:**
    * **Database Integration**: PostgreSQL, MySQL, MongoDB, SQLite adapters
    * **Message Queues**: Redis, RabbitMQ, and custom queue implementations
    * **External APIs**: RESTful API clients and webhook integrations
    * **File Systems**: Local, cloud, and distributed file system support

**Performance and Scalability:**
    * **Parallel Processing**: Multi-threaded and multi-process execution
    * **Distributed Computing**: Parsl integration for HPC environments
    * **Caching Systems**: Multi-tier caching for performance optimization
    * **Resource Management**: Automatic resource allocation and cleanup

**Security and Reliability:**
    * **Secure Configuration**: Encrypted configuration storage and transmission
    * **Input Validation**: Comprehensive input sanitization and validation
    * **Audit Logging**: Detailed audit trails for compliance and debugging
    * **Health Monitoring**: Automatic health checks and failure recovery

**Development and Testing:**
    * **Test Utilities**: Comprehensive testing framework and mock services
    * **Development Tools**: CLI tools for component development and debugging
    * **Documentation Generation**: Automatic documentation from configurations
    * **Migration Tools**: Utilities for upgrading between framework versions

**Best Practices:**
    * **Configuration Management**: Environment-specific configurations with validation
    * **Error Handling**: Comprehensive error handling with user-friendly messages
    * **Performance Monitoring**: Built-in metrics collection and analysis
    * **Security**: Secure defaults and configuration validation
    * **Documentation**: Comprehensive documentation with examples and tutorials

**Community and Ecosystem:**
    The library is designed to be:
    
    * **Extensible**: Easy to add new components and capabilities
    * **Modular**: Components can be used independently or together
    * **Community-Driven**: Open architecture for community contributions
    * **Standards-Compliant**: Follows industry standards and best practices

**Version Compatibility:**
    * **Framework Version**: 2.0.0+
    * **Python Version**: 3.8+
    * **Dependencies**: Minimal external dependencies for core functionality
    * **Backward Compatibility**: Semantic versioning with migration guides

See Also:
    * :mod:`nanobrain.core`: Core framework components
    * :mod:`nanobrain.library.agents`: Agent implementations
    * :mod:`nanobrain.library.infrastructure`: Infrastructure components
    * :mod:`nanobrain.library.tools`: Tool integrations
    * :mod:`nanobrain.library.workflows`: Workflow implementations
    * :mod:`nanobrain.library.interfaces.web`: Web interface system
"""

# Agent components
from .agents import (
    EnhancedCollaborativeAgent,
)

# Workflow components  
from .workflows import (
    ChatWorkflow
)

# Infrastructure components
from .infrastructure import (
    # Data components
    ConversationHistoryUnit,
    DataUnitMemory,
    
    # Step components
    ParallelConversationalAgentStep,
    ParallelAgentStep,
)

# Bioinformatics components
from . import bioinformatics

# Try to import web interface components
try:
    from .interfaces.web import (
        WebInterface,
        WebInterfaceConfig,
        ChatRequest,
        ChatOptions,
        ChatResponse,
        ChatMetadata
    )
    WEB_INTERFACE_AVAILABLE = True
except ImportError:
    WEB_INTERFACE_AVAILABLE = False
    # Define placeholder exports
    WebInterface = None
    WebInterfaceConfig = None
    ChatRequest = None
    ChatOptions = None
    ChatResponse = None
    ChatMetadata = None

__all__ = [
    # Agents
    'EnhancedCollaborativeAgent',
    
    # Workflows
    'ChatWorkflow',
    
    # Infrastructure - Data
    'ConversationHistoryUnit',
    'DataUnitMemory',
    
    # Infrastructure - Steps
    'ParallelConversationalAgentStep',
    'ParallelAgentStep',
    
    # Bioinformatics
    'bioinformatics',
    
    # Web Interface (if available)
    'WebInterface',
    'WebInterfaceConfig',
    'ChatRequest',
    'ChatOptions',
    'ChatResponse',
    'ChatMetadata',
    
    # Utility
    'WEB_INTERFACE_AVAILABLE'
]

# Version information
__version__ = "2.0.0"
__author__ = "NanoBrain Team"
__description__ = "Comprehensive library for NanoBrain framework components"

# Library metadata
LIBRARY_INFO = {
    "version": __version__,
    "components": {
        "agents": ["conversational", "specialized"],
        "infrastructure": ["data_units", "triggers", "links", "steps"],
        "workflows": ["chat_workflow", "parsl_chat_workflow"]
    },
    "description": __description__
}

def get_library_info():
    """Get information about the library components."""
    return LIBRARY_INFO

def list_available_components():
    """List all available components in the library."""
    components = []
    for category, items in LIBRARY_INFO["components"].items():
        for item in items:
            components.append(f"{category}.{item}")
    return components 