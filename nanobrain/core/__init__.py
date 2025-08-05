"""
NanoBrain Core Framework - Foundational Components
==================================================

The core framework provides the essential building blocks for constructing sophisticated
AI agent systems. All components follow the mandatory ``from_config`` pattern and support
event-driven architecture with comprehensive configuration management.

**Architectural Patterns:**
    The core framework implements several key architectural patterns:
    
    * **Unified from_config Pattern**: All components inherit from :class:`FromConfigBase`
      and are created exclusively through configuration files
    * **Event-Driven Data Flow**: Components communicate through :class:`DataUnit` containers
      connected by :class:`Link` objects and activated by :class:`Trigger` events
    * **Configuration-First Design**: Every component behavior is controlled through
      YAML configurations with :class:`ConfigBase` validation
    * **Pluggable Execution**: :class:`ExecutorBase` allows local, threaded, or distributed
      execution via Parsl integration

**Core Component Categories:**

**Foundation Components:**
    * :class:`FromConfigBase` - Base class enforcing unified component creation patterns
    * :class:`ConfigBase` - Configuration validation with schema support and recursive resolution
    * :class:`ComponentConfigurationError` - Configuration-related error handling
    * :class:`ComponentDependencyError` - Dependency resolution error handling

**Agent System:**
    * :class:`Agent` - Abstract base for AI processing agents with LLM integration
    * :class:`ConversationalAgent` - Conversational agent with context management
    * :class:`SimpleAgent` - Stateless agent for single-request processing
    * :class:`AgentConfig` - Comprehensive agent configuration with tool integration

**Step System:**
    * :class:`BaseStep` - Abstract base for data processing steps
    * :class:`Step` - Standard step implementation with configurable processing
    * :class:`TransformStep` - Specialized step for data transformation operations
    * :class:`StepConfig` - Step configuration with input/output data unit management

**Workflow Orchestration:**
    * :class:`Workflow` - Multi-component workflow orchestration with execution strategies
    * :class:`WorkflowConfig` - Workflow configuration with step dependencies and data flow
    * :class:`WorkflowGraph` - Workflow execution graph management and validation
    * :class:`ExecutionStrategy` - Workflow execution patterns (sequential, parallel, event-driven)
    * :class:`ErrorHandlingStrategy` - Error handling policies for workflow execution

**Data Management:**
    * :class:`DataUnitBase` - Abstract base for type-safe data containers
    * :class:`DataUnitMemory` - In-memory data storage for workflow state
    * :class:`DataUnitFile` - File-based data storage with path management
    * :class:`DataUnitString` - String data containers for text processing
    * :class:`DataUnitStream` - Streaming data containers for real-time processing
    * :class:`DataUnitConfig` - Data unit configuration with validation and serialization

**Component Connectivity:**
    * :class:`LinkBase` - Abstract base for inter-component data connections
    * :class:`DirectLink` - Direct data transfer between components
    * :class:`FileLink` - File-based data transfer with persistence
    * :class:`QueueLink` - Queued data transfer for asynchronous processing
    * :class:`TransformLink` - Data transformation during transfer
    * :class:`ConditionalLink` - Conditional data transfer based on predicates
    * :class:`LinkConfig` - Link configuration with source/target specification

**Event System:**
    * :class:`TriggerBase` - Abstract base for workflow activation events
    * :class:`DataUpdatedTrigger` - Activation on data unit changes
    * :class:`AllDataReceivedTrigger` - Activation when all required data is available
    * :class:`TimerTrigger` - Time-based activation for scheduled workflows
    * :class:`ManualTrigger` - Manual activation for user-controlled workflows
    * :class:`TriggerConfig` - Trigger configuration with event specification

**Execution Backend:**
    * :class:`ExecutorBase` - Abstract base for pluggable execution backends
    * :class:`LocalExecutor` - Single-machine execution for development and testing
    * :class:`ThreadExecutor` - Multi-threaded execution for concurrent processing
    * :class:`ProcessExecutor` - Multi-process execution for CPU-intensive tasks
    * :class:`ParslExecutor` - Distributed execution via Parsl for HPC environments
    * :class:`ExecutorConfig` - Executor configuration with resource management

**Tool Integration:**
    * :class:`ToolBase` - Abstract base for agent tools and capabilities
    * :class:`FunctionTool` - Python function wrapping for agent tool use
    * :class:`AgentTool` - Agent delegation as a tool for multi-agent workflows
    * :class:`StepTool` - Step execution as a tool for agent-driven processing
    * :class:`LangChainTool` - LangChain tool compatibility layer
    * :class:`ToolRegistry` - Tool discovery and management system
    * :class:`ToolConfig` - Tool configuration with capability specification

**Bioinformatics Support:**
    * :class:`BioinformaticsConfig` - Specialized configuration for computational biology
    * :class:`CoordinateSystem` - Biological coordinate system management (0-based, 1-based)
    * :class:`SequenceType` - Biological sequence type definitions (DNA, RNA, protein)
    * :class:`SequenceCoordinate` - Coordinate representation for biological sequences
    * :class:`SequenceRegion` - Region specification for sequence analysis
    * :class:`BioinformaticsDataUnit` - Specialized data units for biological data
    * :class:`BioinformaticsStep` - Steps optimized for bioinformatics workflows
    * :class:`BioinformaticsAgent` - Agents specialized for computational biology tasks
    * :class:`BioinformaticsTool` - Tool wrappers for bioinformatics software

**Sequence Management:**
    * :class:`SequenceManager` - Comprehensive biological sequence management
    * :class:`SequenceValidator` - Validation for biological sequence integrity
    * :class:`FastaParser` - FASTA format parsing with metadata extraction
    * :class:`SequenceFormat` - Format specification for biological sequences
    * :class:`SequenceStats` - Statistical analysis for biological sequences
    * :class:`SequenceValidationError` - Sequence-specific error handling

**Logging and Monitoring:**
    * :class:`NanoBrainLogger` - Structured logging system with performance tracking
    * :class:`LogLevel` - Logging level definitions for system monitoring
    * :class:`OperationType` - Operation classification for performance analysis
    * :class:`ExecutionContext` - Execution context tracking for debugging
    * :class:`ToolCallLog` - Tool usage logging for agent behavior analysis
    * :class:`AgentConversationLog` - Conversation logging for agent interactions

**Template Management:**
    * :class:`PromptTemplateManager` - Prompt template management for agent systems

**Factory Functions:**
    The core framework provides factory functions for convenient component creation:
    
    * :func:`create_executor` - Executor creation with automatic type selection
    * :func:`create_tool` - Tool creation with registry integration
    * :func:`create_workflow` - Workflow creation with validation
    * :func:`create_agent` - Agent creation with configuration validation
    * :func:`create_bioinformatics_data_unit` - Specialized data unit creation
    * :func:`create_sequence_coordinate` - Coordinate creation with validation
    * :func:`create_sequence_region` - Region creation with coordinate validation
    * :func:`create_sequence_manager` - Sequence manager with format detection
    * :func:`create_fasta_parser` - FASTA parser with customizable options
    * :func:`create_sequence_validator` - Validator with sequence type detection

**Usage Patterns:**

**Basic Component Creation:**
    ```python
    from nanobrain.core import ConversationalAgent, AgentConfig
    
    # Create agent from configuration file
    agent = ConversationalAgent.from_config('config/agent.yml')
    
    # Process input
    result = await agent.aprocess("What is machine learning?")
    ```

**Workflow Construction:**
    ```python
    from nanobrain.core import Workflow, Step, DirectLink
    
    # Create workflow from configuration
    workflow = Workflow.from_config('config/workflow.yml')
    
    # Execute workflow
    results = await workflow.execute(input_data)
    ```

**Custom Step Implementation:**
    ```python
    from nanobrain.core import Step, StepConfig
    
    class CustomStep(Step):
        async def process(self, input_data):
            # Custom processing logic
            return processed_data
    
    # Create from configuration
    step = CustomStep.from_config('config/custom_step.yml')
    ```

**Configuration Examples:**

**Agent Configuration:**
    ```yaml
    name: "research_agent"
    description: "AI agent for research tasks"
    model: "gpt-4"
    temperature: 0.3
    tools:
      - class: "nanobrain.library.tools.WebSearchTool"
        config: "config/web_search.yml"
      - class: "nanobrain.library.tools.DocumentAnalyzer"
        config: "config/doc_analyzer.yml"
    ```

**Workflow Configuration:**
    ```yaml
    name: "data_analysis_workflow"
    execution_strategy: "event_driven"
    steps:
      - class: "nanobrain.core.step.Step"
        config: "config/data_ingestion.yml"
      - class: "nanobrain.core.step.TransformStep"
        config: "config/data_transform.yml"
    links:
      - class: "nanobrain.core.link.DirectLink"
        config:
          source: "data_ingestion.output"
          target: "data_transform.input"
    ```

**Advanced Features:**
    - **Distributed Execution**: Seamless scaling to HPC environments via Parsl
    - **Multi-Agent Collaboration**: Agent-to-Agent protocol for complex task delegation
    - **Real-Time Processing**: Event-driven triggers for responsive system behavior
    - **Bioinformatics Integration**: Specialized components for computational biology
    - **Tool Ecosystem**: Extensive tool integration including LangChain compatibility
    - **Configuration Validation**: Comprehensive schema validation with helpful error messages
    - **Performance Monitoring**: Built-in logging and metrics collection

**Error Handling:**
    The framework provides comprehensive error handling with specific exception types:
    
    * :exc:`ComponentConfigurationError` - Configuration validation failures
    * :exc:`ComponentDependencyError` - Dependency resolution issues
    * :exc:`SequenceValidationError` - Biological sequence validation problems
    
    All errors include detailed context and suggestions for resolution.

**Thread Safety:**
    All core components are designed to be thread-safe and support concurrent execution
    patterns. The framework handles synchronization automatically for shared resources.

**Performance Considerations:**
    - **Lazy Loading**: Components are instantiated only when needed
    - **Resource Management**: Automatic cleanup of temporary resources
    - **Caching**: Intelligent caching of configuration and component instances
    - **Memory Efficiency**: Streaming data processing for large datasets
"""

from .executor import (
    ExecutorBase, LocalExecutor, ThreadExecutor, ProcessExecutor, 
    ParslExecutor, ExecutorConfig, create_executor
)

from .data_unit import (
    DataUnitBase, DataUnitMemory, DataUnitFile, DataUnitString, 
    DataUnitStream, DataUnitConfig
)

from .trigger import (
    TriggerBase, DataUpdatedTrigger, AllDataReceivedTrigger, TimerTrigger, ManualTrigger, 
    TriggerConfig, TriggerType
)

from .link import (
    LinkBase, DirectLink, FileLink, QueueLink, TransformLink, 
    ConditionalLink, LinkConfig, LinkType
)

from .tool import (
    ToolBase, FunctionTool, AgentTool, StepTool, LangChainTool, 
    ToolRegistry, ToolConfig, ToolType, create_tool
)

from .component_base import (
    FromConfigBase, ComponentConfigurationError, ComponentDependencyError,
    import_class_from_path
)

from .step import (
    BaseStep, Step, TransformStep, StepConfig
)

from .workflow import (
    Workflow, WorkflowConfig, WorkflowGraph, ConfigLoader, 
    ExecutionStrategy, ErrorHandlingStrategy, create_workflow
)

from .agent import (
    Agent, SimpleAgent, ConversationalAgent, AgentConfig, create_agent
)

from .logging_system import (
    NanoBrainLogger, get_logger, set_debug_mode, trace_function_calls,
    LogLevel, OperationType, ExecutionContext, ToolCallLog, 
    AgentConversationLog
)

from .bioinformatics import (
    BioinformaticsConfig, CoordinateSystem, SequenceType,
    SequenceCoordinate, SequenceRegion, BioinformaticsDataUnit,
    ExternalToolManager, BioinformaticsStep, BioinformaticsAgent, BioinformaticsTool,
    create_bioinformatics_data_unit, create_sequence_coordinate, create_sequence_region
)

from .sequence_manager import (
    SequenceManager, SequenceValidator, FastaParser, SequenceFormat,
    SequenceStats, SequenceValidationError,
    create_sequence_manager, create_fasta_parser, create_sequence_validator
)

from .prompt_template_manager import (
    PromptTemplateManager, PromptTemplate, PromptTemplateConfig
)

__all__ = [
    # Executors
    'ExecutorBase', 'LocalExecutor', 'ThreadExecutor', 'ProcessExecutor', 
    'ParslExecutor', 'ExecutorConfig', 'create_executor',
    
    # Data Units
    'DataUnitBase', 'DataUnitMemory', 'DataUnitFile', 'DataUnitString', 
    'DataUnitStream', 'DataUnitConfig',
    
    # Triggers
    'TriggerBase', 'DataUpdatedTrigger', 'AllDataReceivedTrigger', 'TimerTrigger', 'ManualTrigger', 
    'TriggerConfig', 'TriggerType',
    
    # Links
    'LinkBase', 'DirectLink', 'FileLink', 'QueueLink', 'TransformLink', 
    'ConditionalLink', 'LinkConfig', 'LinkType',
    
    # Tools
    'ToolBase', 'FunctionTool', 'AgentTool', 'StepTool', 'LangChainTool', 
    'ToolRegistry', 'ToolConfig', 'ToolType', 'create_tool',
    
    # Component Base
    'FromConfigBase', 'ComponentConfigurationError', 'ComponentDependencyError',
    'import_class_from_path',
    
    # Steps
    'BaseStep', 'Step', 'TransformStep', 'StepConfig',
    
    # Workflows
    'Workflow', 'WorkflowConfig', 'WorkflowGraph', 'ConfigLoader', 
    'ExecutionStrategy', 'ErrorHandlingStrategy', 'create_workflow',
    
    # Agents
    'Agent', 'SimpleAgent', 'ConversationalAgent', 'AgentConfig', 'create_agent',
    
    # Logging
    'NanoBrainLogger', 'get_logger', 'set_debug_mode', 'trace_function_calls',
    'LogLevel', 'OperationType', 'ExecutionContext', 'ToolCallLog', 
    'AgentConversationLog',
    
    # Bioinformatics
    'BioinformaticsConfig', 'CoordinateSystem', 'SequenceType',
    'SequenceCoordinate', 'SequenceRegion', 'BioinformaticsDataUnit',
    'ExternalToolManager', 'BioinformaticsStep', 'BioinformaticsAgent', 'BioinformaticsTool',
    'create_bioinformatics_data_unit', 'create_sequence_coordinate', 'create_sequence_region',
    
    # Sequence Management
    'SequenceManager', 'SequenceValidator', 'FastaParser', 'SequenceFormat',
    'SequenceStats', 'SequenceValidationError',
    'create_sequence_manager', 'create_fasta_parser', 'create_sequence_validator',
    
    # Prompt Template Management
    'PromptTemplateManager', 'PromptTemplate', 'PromptTemplateConfig'
] 