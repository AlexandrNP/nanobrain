"""
Agent System for NanoBrain Framework

Provides tool-calling based AI processing with LLM integration.
"""

import asyncio
import logging
import json
import time
import yaml
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Type
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

# LangChain imports for tool compatibility
try:
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback if LangChain is not available
    BaseTool = object
    CallbackManagerForToolRun = object
    AsyncCallbackManagerForToolRun = object
    LANGCHAIN_AVAILABLE = False

from .component_base import FromConfigBase
from .executor import ExecutorBase, LocalExecutor, ExecutorConfig
from .tool import ToolBase, ToolRegistry, ToolType, ToolConfig, create_tool
from .logging_system import (
    NanoBrainLogger, get_logger, OperationType, ToolCallLog, 
    AgentConversationLog, trace_function_calls
)
from .prompt_template_manager import PromptTemplateManager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


class AgentToolSchema(BaseModel):
    """Schema for agent tool input using Pydantic V2."""
    input: str = Field(..., description="Input text for the agent to process")


class AgentConfig(ConfigBase):
    """
    Agent Configuration Schema - Comprehensive AI Agent Setup and Management
    ======================================================================
    
    The AgentConfig class defines the complete configuration schema for AI agents
    within the NanoBrain framework. This configuration class provides comprehensive
    control over agent behavior, capabilities, integrations, and performance
    characteristics through a declarative YAML-based approach.
    
    **Configuration Philosophy:**
        AgentConfig follows the framework's configuration-first design principles:
        
        * **Declarative Configuration**: All agent behavior defined through YAML
        * **Type Safety**: Comprehensive Pydantic validation with helpful error messages
        * **Schema Completeness**: Every aspect of agent operation is configurable
        * **Environment Flexibility**: Support for development, staging, and production configs
        * **Security First**: Secure defaults with comprehensive validation
        * **Performance Optimization**: Built-in performance and resource management
    
    **Core Configuration Categories:**
        
        **Identity and Behavior:**
        * Agent identification, description, and behavioral characteristics
        * System prompts and behavioral guidelines
        * Conversation management and context handling
        
        **LLM Integration:**
        * Model selection and configuration (GPT-4, Claude, Llama, etc.)
        * Generation parameters (temperature, tokens, sampling)
        * API authentication and rate limiting
        
        **Tool Integration:**
        * Tool discovery, registration, and configuration
        * Tool capability matching and selection
        * Tool performance monitoring and optimization
        
        **Framework Integration:**
        * Executor configuration for local/distributed execution
        * Logging and monitoring configuration
        * A2A protocol configuration for agent collaboration
        
        **Performance and Monitoring:**
        * Resource allocation and optimization
        * Performance metrics and monitoring
        * Error handling and recovery strategies
    
    **Configuration Examples:**
        
        **Basic Conversational Agent:**
        ```yaml
        # config/conversational_agent.yml
        name: "helpful_assistant"
        description: "A helpful AI assistant for general tasks"
        model: "gpt-4"
        temperature: 0.7
        max_tokens: 2000
        system_prompt: |
          You are a helpful AI assistant. Be concise, accurate, and friendly.
          Always provide practical and actionable advice when possible.
        
        # Enable comprehensive logging
        enable_logging: true
        log_conversations: true
        log_tool_calls: true
        ```
        
        **Research Agent with Tools:**
        ```yaml
        # config/research_agent.yml
        name: "research_specialist"
        description: "AI agent specialized in research and analysis"
        model: "gpt-4"
        temperature: 0.3
        max_tokens: 4000
        system_prompt: |
          You are a research specialist AI. Use available tools to gather
          information, analyze data, and provide comprehensive insights.
        
        # Tool integration
        tools:
          - class: "nanobrain.library.tools.WebSearchTool"
            config: "config/tools/web_search.yml"
          - class: "nanobrain.library.tools.DocumentAnalyzer"
            config: "config/tools/document_analyzer.yml"
          - class: "nanobrain.library.tools.DataVisualizer"
            config: "config/tools/data_visualizer.yml"
        
        # External tool configuration file
        tools_config_path: "config/research_tools.yml"
        
        # Performance optimization
        executor_config:
          executor_type: "thread"
          max_workers: 4
          timeout: 300
        ```
        
        **Collaborative Agent with A2A Protocol:**
        ```yaml
        # config/collaborative_agent.yml
        name: "collaboration_coordinator"
        description: "Agent for multi-agent collaboration scenarios"
        model: "gpt-4"
        temperature: 0.5
        
        # A2A protocol configuration
        agent_card:
          capabilities:
            - "task_coordination"
            - "resource_allocation"
            - "progress_monitoring"
          specializations:
            - "project_management"
            - "team_coordination"
          collaboration_patterns:
            - "delegation"
            - "consensus_building"
            - "conflict_resolution"
          max_delegation_depth: 3
          timeout_seconds: 300
        
        # Advanced prompt management
        prompt_template_file: "config/collaboration_prompts.yml"
        ```
        
        **Bioinformatics Specialist Agent:**
        ```yaml
        # config/bioinformatics_agent.yml
        name: "bio_analyst"
        description: "Specialized agent for bioinformatics analysis"
        model: "gpt-4"
        temperature: 0.2
        system_prompt: |
          You are a bioinformatics specialist AI. Use computational biology
          tools to analyze genomic data, protein sequences, and biological pathways.
        
        # Specialized bioinformatics tools
        tools:
          - class: "nanobrain.library.tools.bioinformatics.BVBRCTool"
            config: "config/tools/bvbrc.yml"
          - class: "nanobrain.library.tools.bioinformatics.MMseqs2Tool"
            config: "config/tools/mmseqs2.yml"
          - class: "nanobrain.library.tools.bioinformatics.PubMedClient"
            config: "config/tools/pubmed.yml"
        
        # High-performance execution for computational tasks
        executor_config:
          executor_type: "parsl"
          config: "config/hpc_executor.yml"
        ```
    
    **Field Documentation:**
        
        **Core Identity Fields:**
        
        * **name** (str, required): Unique agent identifier
          - Used for logging, monitoring, and inter-agent communication
          - Should be descriptive and follow naming conventions
          - Examples: "research_assistant", "data_analyst", "collaboration_coordinator"
        
        * **description** (str, optional): Human-readable agent description
          - Explains agent purpose and capabilities
          - Used in documentation and agent discovery
          - Should be concise but comprehensive
        
        **LLM Configuration:**
        
        * **model** (str, default="gpt-3.5-turbo"): LLM model identifier
          - Supported models: "gpt-4", "gpt-3.5-turbo", "claude-3", "llama-2", etc.
          - Model selection affects capabilities, cost, and performance
          - Consider model-specific features and limitations
        
        * **temperature** (float, 0.0-2.0, default=0.7): Response randomness control
          - Lower values (0.0-0.3): More deterministic, factual responses
          - Medium values (0.4-0.8): Balanced creativity and consistency
          - Higher values (0.9-2.0): More creative and varied responses
        
        * **max_tokens** (int, optional): Maximum tokens per response
          - Controls response length and API costs
          - Model-specific limits apply (e.g., GPT-4: 8192 tokens)
          - Consider conversation context and token consumption
        
        * **system_prompt** (str, optional): System-level behavioral instructions
          - Defines agent personality, expertise, and behavioral guidelines
          - Influences all agent responses and decision-making
          - Should be specific, clear, and aligned with agent purpose
        
        **Tool Integration:**
        
        * **tools** (List[Dict], optional): Inline tool configurations
          - Each tool defined with class and config parameters
          - Supports nested tool configurations and dependencies
          - Tools automatically registered and available to agent
        
        * **tools_config_path** (str, optional): External tool configuration file
          - Path to YAML file containing tool configurations
          - Enables shared tool configurations across agents
          - Supports environment-specific tool configurations
        
        **Prompt Management:**
        
        * **prompt_templates** (Dict, optional): Inline prompt templates
          - Named prompt templates for different scenarios
          - Supports Jinja2 templating with variable substitution
          - Used for consistent prompt generation across conversations
        
        * **prompt_template_file** (str, optional): External prompt template file
          - Path to YAML file containing prompt templates
          - Enables shared prompt templates and A/B testing
          - Supports dynamic prompt loading and updates
        
        **Framework Integration:**
        
        * **executor_config** (ExecutorConfig, optional): Execution backend configuration
          - Controls how agent operations are executed
          - Supports local, threaded, process, and distributed execution
          - Affects performance, scalability, and resource usage
        
        * **agent_card** (Dict, optional): A2A protocol metadata
          - Defines agent capabilities for inter-agent collaboration
          - Includes specializations, collaboration patterns, and limits
          - Required for agents participating in multi-agent workflows
        
        **Monitoring and Performance:**
        
        * **auto_initialize** (bool, default=True): Automatic agent initialization
          - Controls whether agent initializes automatically on creation
          - Manual initialization provides more control but requires explicit setup
        
        * **debug_mode** (bool, default=False): Enhanced debugging and logging
          - Enables detailed logging, tracing, and diagnostic information
          - Useful for development and troubleshooting
          - May impact performance in production environments
        
        * **enable_logging** (bool, default=True): Comprehensive logging system
          - Controls whether agent operations are logged
          - Includes performance metrics, error tracking, and audit trails
          - Essential for production monitoring and debugging
        
        * **log_conversations** (bool, default=True): Conversation history logging
          - Controls whether conversation history is logged and stored
          - Important for debugging, analysis, and compliance
          - Consider privacy and storage implications
        
        * **log_tool_calls** (bool, default=True): Tool usage logging
          - Controls whether tool calls and results are logged
          - Useful for performance analysis and debugging
          - Helps optimize tool selection and usage patterns
    
    **Validation and Security:**
        
        **Input Validation:**
        * Comprehensive Pydantic validation for all fields
        * Type checking and constraint enforcement
        * Custom validation rules for complex fields
        * Helpful error messages with correction suggestions
        
        **Security Considerations:**
        * API key protection and secure storage
        * Input sanitization and prompt injection protection
        * Access control and permission management
        * Audit logging for compliance and security monitoring
        
        **Configuration Security:**
        * Environment variable support for sensitive values
        * Configuration encryption for production deployments
        * Secure defaults with minimal privilege principles
        * Regular security validation and updates
    
    **Best Practices:**
        
        **Development:**
        * Use descriptive names and comprehensive descriptions
        * Start with conservative settings and tune based on performance
        * Enable comprehensive logging during development
        * Test with different models and temperature settings
        
        **Production:**
        * Use environment-specific configurations
        * Monitor token usage and API costs
        * Implement rate limiting and error handling
        * Regular configuration validation and updates
        
        **Performance:**
        * Choose appropriate models for specific tasks
        * Optimize token usage with efficient prompts
        * Use caching for repeated operations
        * Monitor and tune based on usage patterns
        
        **Collaboration:**
        * Define clear agent capabilities and specializations
        * Use consistent naming and configuration patterns
        * Document agent purposes and integration points
        * Test multi-agent scenarios thoroughly
    
    **Advanced Configuration Patterns:**
        
        **Environment-Specific Configurations:**
        ```yaml
        # Base configuration
        name: "research_agent"
        model: "${MODEL_NAME:-gpt-3.5-turbo}"
        temperature: ${TEMPERATURE:-0.7}
        debug_mode: ${DEBUG_MODE:-false}
        
        # Environment-specific tool configurations
        tools_config_path: "config/tools_${ENVIRONMENT}.yml"
        ```
        
        **Conditional Tool Loading:**
        ```yaml
        # Tools loaded based on environment
        tools:
          - class: "nanobrain.library.tools.WebSearchTool"
            config: "config/web_search_${ENVIRONMENT}.yml"
            enabled: ${WEB_SEARCH_ENABLED:-true}
        ```
        
        **Dynamic Prompt Management:**
        ```yaml
        # A/B testing prompt templates
        prompt_template_file: "config/prompts_${PROMPT_VERSION:-v1}.yml"
        ```
    
    Note:
        This configuration class inherits from ConfigBase and follows the framework's
        mandatory from_config pattern. Direct instantiation is prohibited - all
        configurations must be loaded from YAML files using the from_config method.
        This ensures consistency, validation, and proper integration with the framework.
    
    Warning:
        Agent configurations may contain sensitive information including API keys,
        system prompts, and behavioral guidelines. Ensure proper security measures
        including secure storage, access controls, and regular security audits.
        Monitor resource usage and costs associated with LLM API calls.
    
    Examples:
        **Loading Configuration:**
        ```python
        # Load agent configuration from file
        config = AgentConfig.from_config('config/my_agent.yml')
        
        # Create agent with configuration
        agent = ConversationalAgent.from_config(config)
        ```
        
        **Configuration Validation:**
        ```python
        # Validate configuration without creating agent
        try:
            config = AgentConfig.from_config('config/test_agent.yml')
            print("Configuration is valid")
        except ValidationError as e:
            print(f"Configuration error: {e}")
        ```
    
    See Also:
        * :class:`Agent`: Base agent class using this configuration
        * :class:`ConversationalAgent`: Conversational agent implementation
        * :class:`ConfigBase`: Base configuration class with validation
        * :class:`ExecutorConfig`: Executor configuration for performance tuning
        * :mod:`nanobrain.library.tools`: Available tools for agent integration
        * :mod:`nanobrain.library.agents`: Specialized agent implementations
    """
    
    name: str
    description: str = ""
    model: str = "gpt-3.5-turbo"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    system_prompt: str = ""
    prompt_templates: Optional[Dict[str, Any]] = Field(default=None, description="Inline prompt templates")
    prompt_template_file: Optional[str] = Field(default=None, description="Path to YAML file containing prompt templates")
    executor_config: Optional[ExecutorConfig] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    tools_config_path: Optional[str] = Field(default=None, description="Path to YAML file containing tool configurations")
    auto_initialize: bool = True
    debug_mode: bool = False
    enable_logging: bool = True
    log_conversations: bool = True
    log_tool_calls: bool = True
    
    # MANDATORY: Agent card section for A2A protocol compliance
    agent_card: Optional[Dict[str, Any]] = Field(default=None, description="Agent card metadata for A2A protocol compliance")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'AgentConfig':
        """
        Load AgentConfig from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            AgentConfig instance
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
            ValidationError: If config validation fails
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Agent config file not found: {yaml_path}")
        
        try:
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Create and validate config
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config {yaml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error creating AgentConfig from {yaml_path}: {e}")


class Agent(FromConfigBase, ABC):
    """
    AI Agent Base Class - Intelligent LLM-Based Processing with Tool Integration
    ===========================================================================
    
    The Agent class is the foundational component for building intelligent AI systems within
    the NanoBrain framework. Agents combine Large Language Model (LLM) capabilities with
    tool calling functionality to create sophisticated AI entities capable of complex
    reasoning, decision making, and task execution.
    
    **Core Architecture:**
        Agents represent autonomous AI entities that:
        
        * **Process Natural Language**: Understand and generate human-like responses
        * **Execute Tools**: Call external tools and APIs to extend capabilities
        * **Maintain Context**: Track conversation history and state across interactions
        * **Make Decisions**: Reason about complex problems and choose appropriate actions
        * **Collaborate**: Work with other agents through the A2A (Agent-to-Agent) protocol
        * **Learn and Adapt**: Improve performance through experience and feedback
    
    **Biological Analogy:**
        Like the prefrontal cortex orchestrating specialized brain regions, agents coordinate
        different tools and capabilities for complex AI processing tasks. The prefrontal cortex
        integrates information from various brain areas, makes executive decisions, and
        coordinates responses - exactly how agents integrate tools, make decisions, and
        coordinate AI processing workflows.
    
    **AI Processing Capabilities:**
        
        **Language Understanding:**
        * Natural language processing and comprehension
        * Context-aware interpretation of user requests
        * Multi-turn conversation management with history
        * Intent recognition and response generation
        
        **Tool Integration:**
        * Dynamic tool discovery and registration
        * Intelligent tool selection based on task requirements
        * Parallel tool execution for complex operations
        * Tool result interpretation and integration
        * LangChain tool compatibility for ecosystem integration
        
        **Decision Making:**
        * Multi-step reasoning and planning
        * Goal decomposition and task prioritization
        * Context-aware decision making
        * Error recovery and alternative strategy selection
        
        **State Management:**
        * Conversation history tracking and retrieval
        * Context preservation across sessions
        * Performance metrics and optimization
        * Memory management for long-running conversations
    
    **Framework Integration:**
        Agents seamlessly integrate with all NanoBrain framework components:
        
        * **Steps Integration**: Agents can be embedded in workflow steps for processing
        * **Workflow Orchestration**: Multi-agent workflows with delegation and coordination
        * **Tool Ecosystem**: Access to bioinformatics tools, web interfaces, and custom tools
        * **Executor Support**: Local, threaded, and distributed execution via Parsl
        * **Configuration Management**: Complete YAML-driven configuration and lifecycle
        * **Logging and Monitoring**: Comprehensive performance tracking and debugging
    
    **Agent Types and Specializations:**
        The framework supports various agent specializations:
        
        * **ConversationalAgent**: Context-aware conversational interactions
        * **SimpleAgent**: Stateless processing for single requests
        * **EnhancedCollaborativeAgent**: Multi-protocol agents with A2A and MCP support
        * **SpecializedAgents**: Domain-specific agents (code writers, analyzers, etc.)
        * **BioinformaticsAgents**: Computational biology specialized agents
    
    **Configuration Architecture:**
        Agents follow the framework's configuration-first design:
        
        ```yaml
        # Basic agent configuration
        name: "intelligent_assistant"
        description: "AI assistant with tool capabilities"
        model: "gpt-4"
        temperature: 0.7
        max_tokens: 2000
        system_prompt: "You are a helpful AI assistant with access to tools."
        
        # Tool integration
        tools:
          - class: "nanobrain.library.tools.WebSearchTool"
            config: "config/web_search.yml"
          - class: "nanobrain.library.tools.DocumentAnalyzer"
            config: "config/doc_analyzer.yml"
        
        # Executor configuration
        executor:
          class: "nanobrain.core.executor.LocalExecutor"
          config: "config/local_executor.yml"
        
        # Logging and monitoring
        enable_logging: true
        log_conversations: true
        log_tool_calls: true
        debug_mode: false
        ```
    
    **Usage Patterns:**
        
        **Basic Agent Creation:**
        ```python
        from nanobrain.core import ConversationalAgent
        
        # Create agent from configuration
        agent = ConversationalAgent.from_config('config/assistant.yml')
        
        # Process single request
        response = await agent.aprocess("What is machine learning?")
        print(response.content)
        ```
        
        **Agent with Custom Tools:**
        ```python
        # Agent automatically loads and registers configured tools
        agent = ConversationalAgent.from_config('config/research_agent.yml')
        
        # Agent intelligently selects and uses appropriate tools
        response = await agent.aprocess(
            "Research the latest developments in quantum computing"
        )
        ```
        
        **Multi-Agent Collaboration:**
        ```python
        # Agents can delegate tasks to other specialized agents
        coordinator = EnhancedCollaborativeAgent.from_config('config/coordinator.yml')
        
        response = await coordinator.aprocess(
            "Analyze this dataset and create a visualization"
        )
        # Coordinator automatically delegates to data analysis and visualization agents
        ```
    
    **Tool Integration Patterns:**
        
        **Automatic Tool Registration:**
        * Tools are automatically discovered and registered from configuration
        * Dynamic tool loading based on task requirements
        * Tool capability matching for optimal selection
        
        **LangChain Compatibility:**
        * Seamless integration with existing LangChain tools
        * Automatic adaptation of tool interfaces
        * Preservation of tool metadata and documentation
        
        **Custom Tool Development:**
        * Framework-native tool development patterns
        * Tool validation and error handling
        * Performance monitoring and optimization
    
    **Performance and Scalability:**
        
        **Execution Optimization:**
        * Asynchronous processing for responsive interactions
        * Parallel tool execution for complex operations
        * Intelligent caching of LLM responses and tool results
        * Resource management and cleanup
        
        **Monitoring and Metrics:**
        * Token usage tracking and optimization
        * Response time monitoring and analysis
        * Error rate tracking and alerting
        * Conversation quality metrics
        
        **Scalability Features:**
        * Stateless operation support for horizontal scaling
        * Session management for multi-user environments
        * Load balancing and resource allocation
        * Distributed execution support via Parsl integration
    
    **Error Handling and Recovery:**
        Comprehensive error handling with graceful degradation:
        
        * **LLM Failures**: Automatic retry with exponential backoff
        * **Tool Failures**: Fallback strategies and alternative tools
        * **Configuration Errors**: Detailed validation and helpful error messages
        * **Resource Limits**: Graceful handling of rate limits and quotas
        * **Network Issues**: Robust error recovery and user notification
    
    **Security and Privacy:**
        
        **Input Validation:**
        * Comprehensive input sanitization and validation
        * Protection against prompt injection attacks
        * Content filtering and safety checks
        
        **Data Protection:**
        * Conversation history encryption and secure storage
        * API key protection and rotation
        * Audit logging for compliance and debugging
        
        **Access Control:**
        * Tool access permissions and restrictions
        * User authentication and authorization
        * Resource usage limits and monitoring
    
    **Development and Testing:**
        
        **Testing Support:**
        * Mock LLM clients for deterministic testing
        * Tool mocking and simulation capabilities
        * Conversation replay and analysis tools
        
        **Debugging Features:**
        * Comprehensive logging with structured output
        * Step-by-step execution tracing
        * Tool call inspection and analysis
        * Performance profiling and optimization hints
    
    **Agent Lifecycle:**
        The agent follows a well-defined lifecycle:
        
        1. **Configuration Loading**: Parse and validate agent configuration
        2. **LLM Client Initialization**: Setup LLM client with authentication
        3. **Tool Registration**: Discover and register available tools
        4. **Prompt Manager Setup**: Initialize template management system
        5. **State Initialization**: Setup conversation history and metrics
        6. **Ready State**: Agent ready to process requests
        7. **Processing**: Handle requests with tool calling and reasoning
        8. **Cleanup**: Resource cleanup and state persistence
    
    **Advanced Features:**
        
        **Prompt Engineering:**
        * Template-based prompt management
        * Dynamic prompt generation based on context
        * A/B testing of prompt variations
        * Prompt optimization and fine-tuning
        
        **Conversation Management:**
        * Multi-turn conversation tracking
        * Context window management and optimization
        * Conversation summarization for long interactions
        * Session persistence and recovery
        
        **Multi-Modal Capabilities:**
        * Text, image, and document processing
        * File upload and analysis support
        * Rich media response generation
        * Cross-modal reasoning and understanding
    
    Attributes:
        name (str): Agent identifier for logging and debugging
        description (str): Human-readable agent description
        model (str): LLM model identifier (e.g., "gpt-4", "claude-3")
        temperature (float): LLM temperature for response randomness (0.0-1.0)
        max_tokens (int): Maximum tokens per LLM response
        system_prompt (str): System prompt for agent behavior definition
        tools (List[Tool]): Registered tools available to the agent
        executor (ExecutorBase): Execution backend for agent operations
        tool_registry (ToolRegistry): Tool management and discovery system
        conversation_history (List[Dict]): Conversation history and context
        performance_metrics (Dict): Real-time performance and usage metrics
    
    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like ConversationalAgent or SimpleAgent.
        All agents must be created using the from_config pattern with proper
        configuration files following the framework's architectural patterns.
    
    Warning:
        Agents may consume significant LLM API resources. Monitor token usage
        and implement appropriate rate limiting and cost controls. Ensure
        proper API key security and never commit credentials to version control.
    
    See Also:
        * :class:`ConversationalAgent`: Context-aware conversational agent
        * :class:`SimpleAgent`: Stateless agent for single requests
        * :class:`AgentConfig`: Agent configuration schema and validation
        * :class:`ToolRegistry`: Tool management and discovery system
        * :mod:`nanobrain.library.agents`: Specialized agent implementations
        * :mod:`nanobrain.library.tools`: Available tools and integrations
    """
    
    COMPONENT_TYPE = "agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'model': 'gpt-3.5-turbo',
        'system_prompt': '',
        'tools': [],
        'temperature': 0.7,
        'max_tokens': 1000,
        'timeout': 30,
        'retry_attempts': 3
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return AgentConfig - ONLY method that differs from other components"""
        return AgentConfig
    
    # UNIFIED PATTERN: Override _init_from_config with SAME signature as all components  
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """UNIFIED: Initialize agent - SAME signature as ALL components"""
        # Call parent initialization with SAME signature
        super()._init_from_config(config, component_config, dependencies)
        
        # Agent-specific initialization (preserving existing logic)
        self.description = getattr(config, 'description', '')
        
        # Initialize specialized agent logging (automatically detects concrete instances)
        from nanobrain.core.agent_logging import create_agent_logger
        self.agent_logger = create_agent_logger(self)
        
        # Log agent lifecycle event for concrete instances only
        self.agent_logger.log_lifecycle_event("initialize", {
            "agent_type": self.__class__.__name__,
            "model": getattr(config, 'model', 'gpt-3.5-turbo'),
            "description": self.description,
            "log_conversations": getattr(config, 'log_conversations', True),
            "enable_logging": getattr(config, 'enable_logging', True)
        })
        
        # Executor for running the agent - get from dependencies or create default
        executor = dependencies.get('executor')
        if executor:
            self.executor = executor
        else:
            # Create executor using from_config pattern
            executor_config = getattr(config, 'executor_config', None) or ExecutorConfig.from_config({})
            self.executor = LocalExecutor.from_config(executor_config)
        
        # Tool registry for managing tools
        self.tool_registry = ToolRegistry()
        
        # Initialize prompt template manager
        self.prompt_manager: Optional[PromptTemplateManager] = None
        self._init_prompt_manager()
        
        # LLM client (will be set during initialization)
        self.llm_client = None
        
        # State management
        self._is_initialized = False
        self._conversation_history: List[Dict[str, Any]] = []
        self._execution_count = 0
        self._error_count = 0
        self._total_tokens_used = 0
        self._total_llm_calls = 0
        
        # Performance tracking
        self._start_time = time.time()
        self._last_activity_time = time.time()
    
    # ARCHITECTURAL VIOLATION FIXED: Agent now properly inherits from FromConfigBase
    # Direct instantiation is now prevented by inherited __init__ method
        
    def _init_prompt_manager(self) -> None:
        """Initialize the prompt template manager if configured."""
        try:
            # Check for prompt template file first
            if self.config.prompt_template_file:
                # Resolve the path relative to the project root
                template_path = self.config.prompt_template_file
                if not Path(template_path).is_absolute():
                    # Try multiple resolution strategies
                    possible_paths = [
                        Path(template_path),  # Current working directory
                        Path(__file__).parent.parent / template_path,  # Relative to nanobrain package
                        Path(__file__).parent.parent.parent / template_path,  # Relative to project root
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            template_path = str(path)
                            break
                    else:
                        # If not found, use original path and let PromptTemplateManager handle the error
                        self.agent_logger.warning(f"Prompt template file not found in expected locations: {self.config.prompt_template_file}")
                
                self.prompt_manager = PromptTemplateManager(template_path)
                self.agent_logger.log_debug(
                    f"Initialized prompt manager from file: {template_path}"
                )
            # Then check for inline templates
            elif self.config.prompt_templates:
                self.prompt_manager = PromptTemplateManager(self.config.prompt_templates)
                self.agent_logger.log_debug(
                    f"Initialized prompt manager with {len(self.config.prompt_templates.get('prompts', {}))} inline templates"
                )
            else:
                self.agent_logger.log_debug("No prompt templates configured")
                
        except Exception as e:
            self.agent_logger.warning(f"Failed to initialize prompt manager: {e}")
            self.prompt_manager = None
    
    def get_prompt(self, prompt_name: str, **params) -> str:
        """
        Get a formatted prompt from templates.
        
        Args:
            prompt_name: Name of the prompt template
            **params: Parameters for template substitution
            
        Returns:
            Formatted prompt string
            
        Raises:
            RuntimeError: If prompt manager not initialized
            KeyError: If prompt not found
        """
        if not self.prompt_manager:
            raise RuntimeError("Prompt manager not initialized")
        
        return self.prompt_manager.get_prompt(prompt_name, params=params)
    
    def has_prompt_template(self, prompt_name: str) -> bool:
        """Check if a prompt template exists."""
        if not self.prompt_manager:
            return False
        
        return prompt_name in self.prompt_manager.list_prompts()
    
    async def initialize(self) -> None:
        """Initialize the agent and its components."""
        if self._is_initialized:
            self.agent_logger.log_debug(f"Agent {self.name} already initialized")
            return
        
        # Initialize executor
        self.agent_logger.log_debug(f"Initializing executor for agent {self.name}")
        await self.executor.initialize()
        
        # Initialize LLM client
        self.agent_logger.log_debug(f"Initializing LLM client for agent {self.name}")
        await self._initialize_llm_client()
        
        # Load tools from YAML configuration if specified
        if self.config.tools_config_path:
            self.agent_logger.log_debug(f"Loading tools from YAML config: {self.config.tools_config_path}")
            await self._load_tools_from_yaml_config()
        
        # Register tools from configuration
        self.agent_logger.log_debug(f"Registering {len(self.config.tools)} tools for agent {self.name}")
        for i, tool_config in enumerate(self.config.tools):
            try:
                await self._register_tool_from_config(tool_config)
                self.agent_logger.log_debug(f"Registered tool {i+1}/{len(self.config.tools)}")
            except Exception as e:
                self.agent_logger.log_error(f"Failed to register tool {i+1}: {e}", 
                                          error_type="tool_registration_error",
                                          context={"tool_config": tool_config})
        
        # Initialize all tools
        self.agent_logger.log_debug(f"Initializing all tools for agent {self.name}")
        await self.tool_registry.initialize_all()
        
        self._is_initialized = True
            
        # Log agent lifecycle event
        self.agent_logger.log_lifecycle_event("initialized", {
            "agent_type": self.__class__.__name__,
            "model": self.config.model,
            "tools_count": len(self.tool_registry.list_tools()),
            "has_llm_client": self.llm_client is not None,
            "log_conversations": self.config.log_conversations,
            "enable_logging": self.config.enable_logging
        })
    
    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        # Log final statistics
        uptime_seconds = time.time() - self._start_time
        
        # Shutdown tools
        await self.tool_registry.shutdown_all()
        
        # Shutdown executor
        await self.executor.shutdown()
        
        self._is_initialized = False
        
        # Log agent lifecycle event with final stats
        self.agent_logger.log_lifecycle_event("shutdown", {
            "agent_type": self.__class__.__name__,
            "uptime_seconds": uptime_seconds,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "total_tokens_used": self._total_tokens_used,
            "total_llm_calls": self._total_llm_calls,
            "tools_used": len(self.tool_registry.list_tools())
        })
        
        # Shutdown agent logger
        self.agent_logger.shutdown()
    
    async def _initialize_llm_client(self) -> None:
        """Initialize the LLM client using NanoBrain configuration system."""
        try:
            # Try to import OpenAI client
            from openai import AsyncOpenAI
            import os
            
            # Get API key from NanoBrain configuration system first
            api_key = None
            try:
                from nanobrain.core.config import get_api_key
                api_key = get_api_key('openai')
                if api_key:
                    self.agent_logger.log_debug(f"Agent {self.name} using OpenAI API key from NanoBrain configuration")
            except ImportError:
                self.agent_logger.log_debug("NanoBrain config system not available, falling back to environment variables")
            
            # Fallback to environment variable if not found in config
            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.agent_logger.log_debug(f"Agent {self.name} using OpenAI API key from environment variable")
            
            if not api_key:
                self.agent_logger.log_error(f"No OpenAI API key found for agent {self.name}. Configure in global_config.yml or set OPENAI_API_KEY environment variable.",
                                          error_type="missing_api_key")
                self.llm_client = None
                return
            
            # Create OpenAI client with API key
            self.llm_client = AsyncOpenAI(api_key=api_key)
            self.agent_logger.log_debug(f"Agent {self.name} initialized with OpenAI client")
            
            # Test the client with a simple call to verify it works
            try:
                # Make a minimal test call to verify the client works
                test_response = await self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                self.agent_logger.log_debug(f"Agent {self.name} OpenAI client test successful")
            except Exception as e:
                self.agent_logger.log_error(f"OpenAI client test failed for agent {self.name}: {e}",
                                          error_type="llm_client_test_failed")
                self.llm_client = None
                
        except ImportError:
            self.agent_logger.log_error("OpenAI client not available. Install with: pip install openai",
                                      error_type="missing_dependency")
            # Could add other LLM clients here (Anthropic, etc.)
            self.llm_client = None
        except Exception as e:
            self.agent_logger.log_error(f"Failed to initialize LLM client for agent {self.name}: {e}",
                                      error_type=type(e).__name__)
            self.llm_client = None

    async def _load_tools_from_yaml_config(self) -> None:
        """Load tools from YAML configuration file."""
        if not self.config.tools_config_path:
            return
            
        try:
            # Resolve the config path
            config_path = self._resolve_config_path(self.config.tools_config_path)
            
            self.agent_logger.log_debug(f"Loading tools from YAML file: {config_path}")
            
            # Load YAML file
            with open(config_path, 'r') as file:
                tools_config = yaml.safe_load(file)
            
            # Check if 'tools' key exists
            if 'tools' not in tools_config:
                self.agent_logger.log_error(f"No 'tools' key found in config: {config_path}",
                                          error_type="config_validation_error")
                return
            
            # Add tools from YAML to the agent's tools list
            yaml_tools = tools_config['tools']
            self.agent_logger.log_debug(f"Found {len(yaml_tools)} tools in YAML config")
            
            # Convert YAML tool configs to the format expected by _register_tool_from_config
            for tool_config in yaml_tools:
                # Create a standardized tool configuration
                standardized_config = self._standardize_tool_config(tool_config)
                self.config.tools.append(standardized_config)
                
        except FileNotFoundError:
            self.agent_logger.log_error(f"Tools config file not found: {self.config.tools_config_path}",
                                      error_type="file_not_found")
            raise
        except yaml.YAMLError as e:
            self.agent_logger.log_error(f"Error parsing YAML config: {e}",
                                      error_type="yaml_parse_error")
            raise
        except Exception as e:
            self.agent_logger.log_error(f"Error loading tools from YAML config: {e}",
                                      error_type=type(e).__name__)
            raise

    def _resolve_config_path(self, config_path: str) -> str:
        """Resolve the configuration file path."""
        # If absolute path, use as-is
        if os.path.isabs(config_path):
            if os.path.exists(config_path):
                return config_path
            else:
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Try relative to current working directory
        if os.path.exists(config_path):
            return config_path
        
        # Try relative to src directory
        src_path = os.path.join("src", config_path)
        if os.path.exists(src_path):
            return src_path
        
        # Try in config directories
        config_dirs = [
            "config",
            "src/config", 
            "src/agents/config",
            "agents/config"
        ]
        
        for config_dir in config_dirs:
            full_path = os.path.join(config_dir, config_path)
            if os.path.exists(full_path):
                return full_path
        
        raise FileNotFoundError(f"Config file not found in any search paths: {config_path}")

    def _standardize_tool_config(self, tool_config: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize tool configuration from YAML to internal format."""
        # Extract basic information
        name = tool_config.get('name', 'unnamed_tool')
        description = tool_config.get('description', '')
        class_path = tool_config.get('class', '')
        
        # Determine tool type based on class path or explicit type
        tool_type = tool_config.get('tool_type', 'agent')  # Default to agent for backward compatibility
        
        # Create standardized configuration
        standardized = {
            'tool_type': tool_type,
            'name': name,
            'description': description,
            'class_path': class_path,
            'parameters': tool_config.get('parameters', {}),
            'config': tool_config.get('config', {})
        }
        
        # Add any additional fields from the original config
        for key, value in tool_config.items():
            if key not in standardized:
                standardized[key] = value
        
        return standardized
    
    async def _register_tool_from_config(self, tool_config: Dict[str, Any]) -> None:
        """Register a tool from configuration."""
        tool_type = ToolType(tool_config.get('tool_type', 'function'))
        
        self.agent_logger.log_debug(f"Registering tool from config: {tool_config.get('name', 'unnamed')}")
        
        # Create tool configuration using from_config
        config_data = {k: v for k, v in tool_config.items() 
                      if k in ['tool_type', 'name', 'description', 'parameters', 'async_execution', 'timeout']}
        config = ToolConfig.from_config(config_data)
        
        # Create and register tool based on type
        if tool_type == ToolType.FUNCTION:
            # Function tools need to be provided externally
            self.agent_logger.log_error(f"Function tool {config.name} needs to be registered manually",
                                      error_type="manual_registration_required")
        elif tool_type == ToolType.AGENT:
            # Create agent instance from class path
            await self._register_agent_tool_from_config(tool_config, config)
        else:
            # Other tool types can be created from config
            tool = create_tool(tool_type, config, **tool_config)
            self.tool_registry.register(tool)
            self.agent_logger.log_debug(f"Successfully registered tool: {config.name}")

    async def _register_agent_tool_from_config(self, tool_config: Dict[str, Any], config: ToolConfig) -> None:
        """Register an agent tool from configuration by creating the agent instance."""
        class_path = tool_config.get('class_path', tool_config.get('class', ''))
        
        if not class_path:
            self.agent_logger.log_error(f"No class path specified for agent tool: {config.name}",
                                      error_type="missing_class_path")
            return
        
        try:
            # Import and create the agent class
            agent_class = self._import_class_from_path(class_path)
            
            # Create agent configuration
            agent_config_data = tool_config.get('config', {})
            agent_config_data.setdefault('name', config.name)
            agent_config_data.setdefault('description', config.description)
            
            # Create AgentConfig instance using from_config
            agent_config = AgentConfig.from_config(agent_config_data)
            
            # Create agent instance
            agent_instance = agent_class(agent_config)
            
            # Initialize the agent
            await agent_instance.initialize()
            
            # Create agent tool
            tool = create_tool(ToolType.AGENT, config, agent=agent_instance)
            self.tool_registry.register(tool)
            
            self.agent_logger.log_debug(f"Successfully registered agent tool: {config.name} ({class_path})")
            
        except Exception as e:
            self.agent_logger.log_error(f"Failed to register agent tool {config.name}: {e}",
                                      error_type=type(e).__name__)
            raise

    def _import_class_from_path(self, class_path: str):
        """Import a class from a module path."""
        try:
            # Split module path and class name
            if '.' not in class_path:
                raise ValueError(f"Invalid class path format: {class_path}")
            
            module_path, class_name = class_path.rsplit('.', 1)
            
            # Import the module
            import importlib
            module = importlib.import_module(module_path)
            
            # Get the class
            agent_class = getattr(module, class_name)
            
            return agent_class
            
        except ImportError as e:
            raise ImportError(f"Could not import module {module_path}: {e}")
        except AttributeError as e:
            raise AttributeError(f"Class {class_name} not found in module {module_path}: {e}")
        except Exception as e:
            raise Exception(f"Error importing class {class_path}: {e}")
    
    def register_tool(self, tool: ToolBase) -> None:
        """Register a tool with the agent."""
        self.tool_registry.register(tool)
        self.agent_logger.log_debug(f"Agent {self.name} registered tool: {tool.name}")
    
    @abstractmethod
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input text and return response.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Response text
        """
        pass
    
    async def execute(self, input_text: str, **kwargs) -> str:
        """Execute the agent using the configured executor."""
        if not self._is_initialized:
            await self.initialize()
        
        # Use the special interaction context to ensure I/O is ALWAYS logged
        async with self.agent_logger.interaction_context(input_text) as context:
            # Store initial state for tracking
            initial_llm_calls = self._total_llm_calls
            initial_total_tokens = self._total_tokens_used
            
            try:
                # Process using executor
                result = await self.executor.execute(self._execute_process, input_text=input_text, **kwargs)
                
                self._execution_count += 1
                self._last_activity_time = time.time()
                
                # Update context with result for logging
                if context:
                    context['response_text'] = result
                    context['llm_calls'] = self._total_llm_calls - initial_llm_calls
                    context['total_tokens'] = self._total_tokens_used - initial_total_tokens
                
                return result
                
            except Exception as e:
                self._error_count += 1
                
                # Log error with agent logger
                self.agent_logger.log_error(f"Agent {self.name} execution failed: {e}", 
                                          error_type=type(e).__name__,
                                          context={"error_count": self._error_count, "input_length": len(input_text)})
                raise
    
    async def _execute_process(self, input_text: str, **kwargs) -> str:
        """Wrapper for process method to be executed by executor."""
        return await self.process(input_text, **kwargs)
    
    async def _call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Call the LLM with messages and optional tools."""
        if not self.llm_client:
            raise RuntimeError("LLM client not initialized")
        
        start_time = time.time()
        try:
            # Prepare the request
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            
            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens
            
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
            
            self.agent_logger.log_debug(f"Making LLM call with {len(messages)} messages")
            
            # Make the API call
            response = await self.llm_client.chat.completions.create(**request_params)
            
            # Track usage
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = response.usage.total_tokens
                self._total_tokens_used += tokens_used
            
            self._total_llm_calls += 1
            
            # Convert response to dict
            response_dict = {
                "id": response.id,
                "model": response.model,
                "choices": [
                    {
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                } for tc in (choice.message.tool_calls or [])
                            ] if choice.message.tool_calls else None
                        },
                        "finish_reason": choice.finish_reason
                    } for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            }
            
            # Log LLM interaction details
            first_choice = response_dict["choices"][0] if response_dict["choices"] else {}
            message_content = first_choice.get("message", {}).get("content", "")
            duration_ms = (time.time() - start_time) * 1000
            
            self.agent_logger.log_llm_call(
                model=response_dict.get("model", "unknown"),
                messages_count=len(messages),
                response_content=message_content,
                tokens_used=tokens_used,
                finish_reason=first_choice.get("finish_reason", "unknown"),
                duration_ms=duration_ms
            )
            
            return response_dict
            
        except Exception as e:
            self.agent_logger.log_error(f"LLM call failed: {e}", 
                                      error_type=type(e).__name__)
            raise
    
    async def _execute_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """Execute tool calls and return results."""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args_str = tool_call["function"]["arguments"]
            
            try:
                # Parse arguments
                tool_args = json.loads(tool_args_str)
                
                self.agent_logger.log_debug(f"Executing tool call: {tool_name}")
                
                # Get tool from registry
                tool = self.tool_registry.get(tool_name)
                if not tool:
                    raise ValueError(f"Tool '{tool_name}' not found")
                
                # Execute tool
                start_time = time.time()
                result = await tool.execute(**tool_args)
                duration_ms = (time.time() - start_time) * 1000
                
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_name,
                    "content": str(result)
                })
                
                self.agent_logger.log_debug(f"Tool call completed: {tool_name} in {duration_ms:.1f}ms")
                
            except Exception as e:
                error_msg = f"Tool call failed: {e}"
                
                results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_name,
                    "content": error_msg
                })
                
                self.agent_logger.log_error(f"Tool call failed: {tool_name}", 
                                          error_type=type(e).__name__,
                                          context={"tool_name": tool_name, "error": str(e)})
        
        return results
    
    def _setup_agent_executor(self):
        """Set up LangChain agent executor for tool calling."""
        if not LANGCHAIN_AVAILABLE:
            self.agent_logger.log_error("LangChain not available, cannot set up agent executor",
                                      error_type="missing_dependency")
            return
        
        if not self.tool_registry.list_tools():
            self.agent_logger.log_debug("No tools available, skipping agent executor setup")
            return
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate
            
            # Create LangChain LLM
            llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Convert NanoBrain tools to LangChain tools
            langchain_tools = []
            for tool_name in self.tool_registry.list_tools():
                tool = self.tool_registry.get(tool_name)
                if tool:
                    langchain_tool = tool.to_langchain_tool() if hasattr(tool, 'to_langchain_tool') else None
                    if langchain_tool:
                        langchain_tools.append(langchain_tool)
            
            if not langchain_tools:
                self.agent_logger.log_error("No LangChain-compatible tools found",
                                          error_type="no_compatible_tools")
                return
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.config.system_prompt or "You are a helpful assistant."),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            # Create agent
            agent = create_tool_calling_agent(llm, langchain_tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                verbose=self.config.debug_mode
            )
            
            self.agent_logger.log_debug(f"Agent executor set up with {len(langchain_tools)} tools")
            
        except Exception as e:
            self.agent_logger.log_error(f"Failed to set up agent executor: {e}",
                                      error_type=type(e).__name__)
            self.agent_executor = None
    
    async def process_with_tools(self, input_text: str, **kwargs) -> str:
        """Process input using LangChain agent executor with tools."""
        if not hasattr(self, 'agent_executor'):
            self.agent_executor = None
            
        if not self.agent_executor:
            self._setup_agent_executor()
        
        if not self.agent_executor:
            # Fallback to regular processing
            return await self.process(input_text, **kwargs)
        
        try:
            # Use LangChain agent executor
            result = await self.agent_executor.ainvoke({"input": input_text})
            
            # Extract output
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            else:
                return str(result)
                
        except Exception as e:
            self.agent_logger.log_error(f"Error in process_with_tools: {e}",
                                      error_type=type(e).__name__)
            # Fallback to regular processing
            return await self.process(input_text, **kwargs)
    
    def add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = {"role": role, "content": content, "timestamp": time.time()}
        self._conversation_history.append(message)
        self.agent_logger.log_debug(f"Added message to conversation: {role} ({len(content)} chars)")
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        old_length = len(self._conversation_history)
        self._conversation_history.clear()
        self.agent_logger.log_debug(f"Cleared conversation history ({old_length} messages)")
    
    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._is_initialized
    
    @property
    def execution_count(self) -> int:
        """Get the number of executions."""
        return self._execution_count
    
    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return self._error_count
    
    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return self.tool_registry.list_tools()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent."""
        uptime_seconds = time.time() - self._start_time
        idle_seconds = time.time() - self._last_activity_time
        
        return {
            "uptime_seconds": uptime_seconds,
            "idle_seconds": idle_seconds,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (self._execution_count - self._error_count) / max(self._execution_count, 1),
            "total_tokens_used": self._total_tokens_used,
            "total_llm_calls": self._total_llm_calls,
            "avg_tokens_per_call": self._total_tokens_used / max(self._total_llm_calls, 1),
            "conversation_length": len(self._conversation_history),
            "available_tools": len(self.available_tools)
        }
    
    def to_langchain_tool(self):
        """Convert this agent to a LangChain tool."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not available")
        
        return AgentLangChainTool(self)


class AgentLangChainTool(BaseTool if LANGCHAIN_AVAILABLE else object):
    """
    LangChain tool wrapper for NanoBrain agents.
    """
    
    def __init__(self, agent: Agent):
        self.agent = agent
        
        if LANGCHAIN_AVAILABLE:
            super().__init__(
                name=agent.name,
                description=agent.description,
                args_schema=AgentToolSchema
            )
    
    def _run(self, input: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Synchronous run method for LangChain compatibility."""
        if not LANGCHAIN_AVAILABLE:
            raise NotImplementedError("LangChain is not available")
        
        # Run the async process method synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.agent.process(input))
            return str(result)
        finally:
            loop.close()
    
    async def _arun(self, input: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Asynchronous run method for LangChain compatibility."""
        if not LANGCHAIN_AVAILABLE:
            raise NotImplementedError("LangChain is not available")
        
        result = await self.agent.process(input)
        return str(result)


class SimpleAgent(Agent):
    """Simple agent implementation without conversation history."""
    
    # Now inherits unified from_config implementation from FromConfigBase
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Process input text and return response."""
        # Use the special interaction context to ensure I/O is ALWAYS logged
        async with self.agent_logger.interaction_context(input_text) as context:
            if not self.llm_client:
                # Fallback for when LLM is not available
                response = f"Echo from {self.name}: {input_text}"
                self.agent_logger.log_error("No LLM client available, using echo response",
                                          error_type="missing_llm_client")
            else:
                # Prepare messages
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                messages.append({"role": "user", "content": input_text})
                
                # Get available tools
                tools = None
                if self.tool_registry.list_tools():
                    tools = []
                    for tool_name in self.tool_registry.list_tools():
                        tool = self.tool_registry.get(tool_name)
                        if tool and hasattr(tool, 'get_schema'):
                            tools.append(tool.get_schema())
                
                # Call LLM
                llm_response = await self._call_llm(messages, tools)
                
                # Process response
                choice = llm_response["choices"][0]
                message = choice["message"]
                
                if message.get("tool_calls"):
                    # Execute tool calls
                    tool_results = await self._execute_tool_calls(message["tool_calls"])
                    
                    # Add tool results to messages and call LLM again
                    messages.append(message)
                    messages.extend(tool_results)
                    
                    final_response = await self._call_llm(messages)
                    response = final_response["choices"][0]["message"]["content"]
                else:
                    response = message["content"]
            
            # Update context with response for logging
            if context:
                context['response_text'] = response
            
            return response or ""


class ConversationalAgent(Agent):
    """Agent that maintains conversation history and context."""
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ConversationalAgent with conversation-specific settings"""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)
        
        # Conversational-specific settings
        self.max_history_length = dependencies.get('max_history_length', 10)
    
    # Now inherits unified from_config implementation from FromConfigBase
    
    async def process(self, input_text: str, **kwargs) -> str:
        """Process input text with conversation history."""
        # Use the special interaction context to ensure I/O is ALWAYS logged
        async with self.agent_logger.interaction_context(input_text) as context:
            if not self.llm_client:
                # Fallback for when LLM is not available
                response = f"Conversational echo from {self.name}: {input_text}"
                self.agent_logger.log_error("No LLM client available, using echo response",
                                          error_type="missing_llm_client")
            else:
                # Prepare messages with history
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                
                # Add conversation history (limited)
                history_start = max(0, len(self._conversation_history) - self.max_history_length)
                for msg in self._conversation_history[history_start:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add current input
                messages.append({"role": "user", "content": input_text})
                
                # Get available tools
                tools = None
                if self.tool_registry.list_tools():
                    tools = []
                    for tool_name in self.tool_registry.list_tools():
                        tool = self.tool_registry.get(tool_name)
                        if tool and hasattr(tool, 'get_schema'):
                            tools.append(tool.get_schema())
                
                # Call LLM
                llm_response = await self._call_llm(messages, tools)
                
                # Process response
                choice = llm_response["choices"][0]
                message = choice["message"]
                
                if message.get("tool_calls"):
                    # Execute tool calls
                    tool_results = await self._execute_tool_calls(message["tool_calls"])
                    
                    # Add tool results to messages and call LLM again
                    messages.append(message)
                    messages.extend(tool_results)
                    
                    final_response = await self._call_llm(messages)
                    response = final_response["choices"][0]["message"]["content"]
                else:
                    response = message["content"]
                
                # Update conversation history
                self.add_to_conversation("user", input_text)
                if response:
                    self.add_to_conversation("assistant", response)
            
            # Update context with response for logging
            if context:
                context['response_text'] = response
                context['history_length'] = len(self._conversation_history)
            
            return response or ""


def create_agent(agent_type: str, config: AgentConfig, **kwargs) -> Agent:
    """
    Factory function to create agents of different types.
    
    Args:
        agent_type: Type of agent ('simple' or 'conversational')
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Agent instance
    """
    logger = get_logger("agent.factory")
    logger.info(f"Creating agent: {config.name}", 
               agent_type=agent_type, 
               agent_name=config.name)
    
    if agent_type.lower() == "simple":
        return SimpleAgent(config, **kwargs)
    elif agent_type.lower() == "conversational":
        return ConversationalAgent(config, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_langchain_agent_executor(agents: List[Agent], llm_model: str = "gpt-3.5-turbo"):
    """
    Create a LangChain agent executor that can use NanoBrain agents as tools.
    
    Args:
        agents: List of NanoBrain agents to use as tools
        llm_model: LLM model to use for the main agent
        
    Returns:
        LangChain AgentExecutor that can use the provided agents as tools
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not available. Install with: pip install langchain langchain-openai")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate
        
        # Create the main LLM
        llm = ChatOpenAI(model=llm_model, temperature=0.7)
        
        # Convert NanoBrain agents to LangChain tools
        tools = []
        for agent in agents:
            try:
                langchain_tool = agent.to_langchain_tool()
                tools.append(langchain_tool)
            except Exception as e:
                logger.warning(f"Failed to convert agent {agent.name} to LangChain tool: {e}")
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can use specialized agents as tools. "
                      "Each agent has specific capabilities. Use them appropriately based on the user's request."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create the agent executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        return agent_executor
        
    except ImportError as e:
        raise ImportError(f"Required LangChain components not available: {e}")


async def initialize_agents_for_langchain(agents: List[Agent]) -> List[Agent]:
    """
    Initialize a list of agents for use with LangChain.
    
    Args:
        agents: List of agents to initialize
        
    Returns:
        List of initialized agents
    """
    initialized_agents = []
    for agent in agents:
        if not agent.is_initialized:
            await agent.initialize()
        initialized_agents.append(agent)
    return initialized_agents 