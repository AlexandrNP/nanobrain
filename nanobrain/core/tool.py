"""
Tool System for NanoBrain Framework

Provides tool interface and adapters for different tool frameworks.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
from .logging_system import get_logger
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools."""
    FUNCTION = "function"
    AGENT = "agent"
    STEP = "step"
    EXTERNAL = "external"
    LANGCHAIN = "langchain"


class ToolConfig(ConfigBase):
    """
    Configuration for tools - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: ToolConfig(name="test", tool_type="...")
    ✅ REQUIRED: ToolConfig.from_config('path/to/config.yml')
    """
    tool_type: ToolType = ToolType.FUNCTION
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    async_execution: bool = True
    timeout: Optional[float] = None
    
    # MANDATORY: Tool card section for A2A protocol compliance
    tool_card: Optional[Dict[str, Any]] = Field(default=None, description="Tool card metadata for A2A protocol compliance")


class ToolBase(FromConfigBase, ABC):
    """
    Base Tool Class - Specialized AI Agent Extensions and Framework Integrations
    ===========================================================================
    
    The ToolBase class is the foundational component for creating specialized functionality
    extensions within the NanoBrain framework. Tools provide AI agents, steps, and workflows
    with access to external services, computational capabilities, data sources, and
    specialized processing functions through a unified, type-safe interface.
    
    **Core Architecture:**
        Tools represent specialized capability providers that:
        
        * **Extend Agent Capabilities**: Provide agents with external functionality access
        * **Enable Ecosystem Integration**: Connect with external APIs, services, and libraries
        * **Ensure Type Safety**: Validate inputs and outputs with comprehensive schemas
        * **Support Async Operations**: Handle concurrent and parallel tool execution
        * **Provide Error Handling**: Robust error management with retry and fallback mechanisms
        * **Enable Protocol Compliance**: Support A2A (Agent-to-Agent) and MCP protocols
    
    **Biological Analogy:**
        Like specialized neural circuits for specific functions (visual cortex for vision,
        motor cortex for movement), tools provide specialized functionality for agents.
        These specialized circuits have dedicated architectures optimized for particular
        tasks and integrate with the broader neural network - exactly how tools provide
        specialized capabilities that integrate seamlessly with the agent ecosystem.
    
    **Tool Architecture Categories:**
        
        **Function Tools:**
        * Direct function execution with parameter validation
        * Synchronous and asynchronous function calls
        * Result caching and performance optimization
        * Type-safe parameter passing and return values
        
        **Agent Tools:**
        * Integration with other AI agents and LLM services
        * Agent delegation and collaboration patterns
        * Context sharing and conversation management
        * Multi-agent workflow coordination
        
        **Step Tools:**
        * Workflow step execution and orchestration
        * Data processing pipeline integration
        * Step chaining and dependency management
        * Result validation and error propagation
        
        **External Tools:**
        * API integration with external services
        * Database connections and queries
        * File system operations and data access
        * Network services and web scraping
        
        **LangChain Tools:**
        * Seamless LangChain ecosystem integration
        * Tool adapter patterns for existing LangChain tools
        * Metadata preservation and capability mapping
        * Protocol translation and compatibility layers
    
    **Framework Integration:**
        Tools seamlessly integrate with all framework components:
        
        * **Agent Integration**: Automatic tool discovery and registration by agents
        * **Workflow Orchestration**: Tools used within workflow steps for processing
        * **Executor Support**: Tools run on various execution backends (local, distributed)
        * **Configuration Management**: Complete YAML-driven tool configuration
        * **Monitoring Integration**: Comprehensive logging and performance tracking
        * **Protocol Compliance**: A2A and MCP protocol support for interoperability
    
    **Tool Discovery and Registration:**
        The framework supports automatic tool discovery:
        
        * **Configuration-Based**: Tools defined in agent and workflow configurations
        * **Dynamic Discovery**: Runtime tool registration and capability detection
        * **Capability Matching**: Automatic tool selection based on task requirements
        * **Protocol Adaptation**: Automatic adaptation between different tool protocols
        * **Metadata Extraction**: Tool capabilities extracted from configuration and metadata
    
    **Configuration Architecture:**
        Tools follow the framework's configuration-first design:
        
        ```yaml
        # Basic function tool
        name: "data_processor"
        description: "Processes and validates data structures"
        tool_type: "function"
        async_execution: true
        timeout: 30
        
        # Tool card for A2A protocol compliance
        tool_card:
          capabilities:
            - "data_validation"
            - "format_conversion"
            - "schema_validation"
          input_types:
            - "json"
            - "csv"
            - "xml"
          output_types:
            - "json"
            - "validated_data"
          parameters:
            validation_schema:
              type: "string"
              description: "JSON schema for validation"
              required: true
            output_format:
              type: "string"
              description: "Output format preference"
              default: "json"
              choices: ["json", "csv", "xml"]
        
        # External API tool
        name: "web_search"
        description: "Web search and information retrieval"
        tool_type: "external"
        
        # API configuration
        api_config:
          base_url: "https://api.search.com"
          api_key_env: "SEARCH_API_KEY"
          rate_limit: 100
          timeout: 10
        
        # LangChain tool integration
        name: "langchain_calculator"
        description: "Mathematical calculator via LangChain"
        tool_type: "langchain"
        
        # LangChain adapter configuration
        langchain_config:
          tool_class: "langchain.tools.Calculator"
          adapter_settings:
            preserve_metadata: true
            enable_streaming: false
        ```
    
    **Usage Patterns:**
        
        **Basic Tool Implementation:**
        ```python
        from nanobrain.core import ToolBase, ToolConfig
        
        class DataProcessor(ToolBase):
            async def execute(self, data, schema=None):
                # Validate input data against schema
                validated_data = self.validate_data(data, schema)
                
                # Process data
                result = self.process_data(validated_data)
                
                # Return structured result
                return {
                    "processed_data": result,
                    "validation_status": "passed",
                    "processing_time": self.get_execution_time()
                }
        
        # Create tool from configuration
        tool = DataProcessor.from_config('config/data_processor.yml')
        ```
        
        **Agent Tool Integration:**
        ```python
        # Tools automatically available to agents via configuration
        agent_config = {
            "name": "data_analyst",
            "tools": [
                {"class": "tools.DataProcessor", "config": "config/processor.yml"},
                {"class": "tools.WebSearch", "config": "config/search.yml"}
            ]
        }
        
        agent = ConversationalAgent.from_config(agent_config)
        
        # Agent automatically discovers and uses appropriate tools
        response = await agent.aprocess(
            "Analyze this dataset and search for related information"
        )
        ```
        
        **LangChain Tool Integration:**
        ```python
        # Seamless LangChain tool integration
        class LangChainCalculator(ToolBase):
            def __init__(self, config):
                super().__init__(config)
                # Automatic LangChain tool adapter
                self.langchain_tool = self.create_langchain_adapter()
            
            async def execute(self, expression):
                # Direct LangChain tool execution
                result = await self.langchain_tool.arun(expression)
                return {"calculation": result, "expression": expression}
        ```
        
        **External API Tool:**
        ```python
        class WebSearchTool(ToolBase):
            async def execute(self, query, max_results=10):
                # Automatic API authentication and rate limiting
                async with self.api_client() as client:
                    response = await client.search(
                        query=query,
                        limit=max_results
                    )
                
                # Structured result with metadata
                return {
                    "results": response.results,
                    "query": query,
                    "total_found": response.total_count,
                    "search_time": response.duration
                }
        ```
    
    **Advanced Features:**
        
        **Parallel Tool Execution:**
        * Concurrent execution of independent tools
        * Result aggregation and correlation
        * Error isolation and partial result handling
        * Resource pooling and optimization
        
        **Tool Chaining and Composition:**
        * Sequential tool execution with result passing
        * Conditional tool selection based on results
        * Tool pipeline creation and optimization
        * Dynamic tool workflow generation
        
        **Caching and Performance:**
        * Intelligent result caching with TTL management
        * Parameter-based cache key generation
        * Cache invalidation and refresh strategies
        * Performance monitoring and optimization
        
        **Error Handling and Resilience:**
        * Automatic retry with exponential backoff
        * Fallback tools for failed operations
        * Error classification and handling strategies
        * Circuit breaker patterns for unstable services
    
    **Protocol Compliance:**
        
        **A2A (Agent-to-Agent) Protocol:**
        * Tool card metadata for capability advertisement
        * Standardized tool interfaces and contracts
        * Inter-agent tool sharing and delegation
        * Capability negotiation and discovery
        
        **MCP (Model Context Protocol):**
        * Standardized tool description and execution
        * Context preservation across tool calls
        * Resource management and lifecycle
        * Security and access control integration
        
        **Framework Native Protocol:**
        * Full NanoBrain framework integration
        * Configuration-driven tool creation
        * Event-driven execution and monitoring
        * Complete lifecycle management
    
    **Security and Validation:**
        
        **Input Validation:**
        * Comprehensive parameter validation with schemas
        * Type checking and constraint enforcement
        * Sanitization for security-sensitive operations
        * Input size limits and resource protection
        
        **Access Control:**
        * Tool permission management and restrictions
        * API key protection and rotation
        * Rate limiting and quota enforcement
        * Audit logging for compliance and security
        
        **Output Validation:**
        * Result schema validation and verification
        * Content filtering and sanitization
        * Size limits and resource management
        * Error information sanitization
    
    **Performance and Scalability:**
        
        **Execution Optimization:**
        * Asynchronous execution for non-blocking operations
        * Connection pooling for external services
        * Resource reuse and optimization
        * Memory management and cleanup
        
        **Monitoring and Metrics:**
        * Tool execution time tracking and analysis
        * Success rate monitoring and alerting
        * Resource usage tracking and optimization
        * Error rate analysis and improvement recommendations
        
        **Scalability Features:**
        * Horizontal scaling through tool distribution
        * Load balancing across tool instances
        * Resource allocation and management
        * Auto-scaling based on demand
    
    **Development and Testing:**
        
        **Testing Support:**
        * Mock tool implementations for testing
        * Tool simulation and validation frameworks
        * Performance benchmarking and profiling
        * Integration testing with agents and workflows
        
        **Debugging Features:**
        * Comprehensive logging with structured output
        * Tool execution tracing and analysis
        * Parameter and result inspection
        * Performance profiling and optimization hints
        
        **Development Tools:**
        * Tool template generation and scaffolding
        * Configuration validation and linting
        * Documentation generation from tool cards
        * Protocol compliance verification
    
    **Tool Lifecycle:**
        Tools follow a well-defined lifecycle:
        
        1. **Configuration Loading**: Parse and validate tool configuration
        2. **Capability Registration**: Register tool capabilities and metadata
        3. **Dependency Resolution**: Setup connections and external dependencies
        4. **Authentication Setup**: Configure API keys and authentication
        5. **Resource Initialization**: Setup connection pools and caches
        6. **Ready State**: Tool ready for execution requests
        7. **Execution**: Handle tool calls with validation and processing
        8. **Cleanup**: Release resources and cleanup connections
    
    **Integration Patterns:**
        
        **Multi-Tool Coordination:**
        * Tool orchestration for complex operations
        * Result correlation and aggregation
        * Conditional tool execution based on results
        * Tool pipeline optimization and caching
        
        **Agent Tool Delegation:**
        * Automatic tool selection based on capabilities
        * Tool recommendation and suggestion systems
        * Context-aware tool parameter generation
        * Result interpretation and integration
        
        **Workflow Tool Integration:**
        * Tools as processing steps in workflows
        * Tool result validation and error handling
        * Tool performance monitoring in workflows
        * Dynamic tool configuration based on workflow state
    
    Attributes:
        name (str): Tool identifier for logging and agent discovery
        description (str): Human-readable tool description and capabilities
        tool_type (ToolType): Tool category and execution pattern
        parameters (Dict): Tool parameter schema and validation rules
        async_execution (bool): Whether tool supports asynchronous execution
        timeout (float, optional): Maximum execution time before timeout
        tool_card (Dict, optional): A2A protocol metadata for capability advertisement
        api_client (optional): Configured API client for external tool types
        cache (optional): Result cache for performance optimization
        performance_metrics (Dict): Real-time performance and usage metrics
    
    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations or create custom tools by extending this class.
        All tools must be created using the from_config pattern with proper
        configuration files following the framework's architectural patterns.
    
    Warning:
        Tools may access external services and APIs, consuming network resources
        and potentially incurring costs. Monitor API usage, implement rate limiting,
        and ensure proper authentication and security measures. Be cautious with
        tools that modify external systems or access sensitive data.
    
    See Also:
        * :class:`ToolConfig`: Tool configuration schema and validation
        * :class:`ToolType`: Available tool types and categories
        * :mod:`nanobrain.library.tools`: Specialized tool implementations
        * :class:`Agent`: AI agents that use tools for extended capabilities
        * :mod:`nanobrain.library.tools.bioinformatics`: Bioinformatics tool suite
        * :mod:`nanobrain.library.tools.web`: Web and API integration tools
    """
    
    COMPONENT_TYPE = "tool"
    REQUIRED_CONFIG_FIELDS = ['name', 'tool_type']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'parameters': {},
        'async_execution': True,
        'timeout': None
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return ToolConfig - ONLY method that differs from other components"""
        return ToolConfig
    
    @classmethod
    def extract_component_config(cls, config: ToolConfig) -> Dict[str, Any]:
        """Extract Tool configuration"""
        return {
            'name': config.name,
            'tool_type': config.tool_type,
            'description': getattr(config, 'description', ''),
            'parameters': getattr(config, 'parameters', {}),
            'async_execution': getattr(config, 'async_execution', True),
            'timeout': getattr(config, 'timeout', None)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Tool dependencies"""
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }
    
    @classmethod
    def validate_and_extract_tool_card(cls, config: Union[ToolConfig, Dict], instance_name: str = None) -> Dict[str, Any]:
        """
        Validate and extract mandatory tool_card section from configuration.
        
        Args:
            config: Tool configuration (ToolConfig object or dict)
            instance_name: Name of the tool instance for error messages
            
        Returns:
            Tool card data dictionary
            
        Raises:
            ValueError: If mandatory tool_card section is missing
        """
        logger = get_logger(f"{cls.__name__}.validate_tool_card")
        
        # Extract tool card data
        if hasattr(config, 'tool_card') and config.tool_card:
            tool_card_data = config.tool_card.model_dump() if hasattr(config.tool_card, 'model_dump') else config.tool_card
            logger.info(f"Tool {instance_name or 'unknown'} loaded with tool card metadata")
            return tool_card_data
        elif isinstance(config, dict) and 'tool_card' in config:
            tool_card_data = config['tool_card']
            logger.info(f"Tool {instance_name or 'unknown'} loaded with tool card metadata")
            return tool_card_data
        else:
            raise ValueError(
                f"Missing mandatory 'tool_card' section in configuration for {cls.__name__}. "
                f"All tools must include tool card metadata for proper discovery and usage."
            )
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Tool with resolved dependencies"""
        self.config = config
        self.name = component_config['name']
        self.description = component_config['description']
        self._is_initialized = False
        self._call_count = 0
        self._error_count = 0
        
        # Initialize logging if enabled
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            self.nb_logger = get_logger(self.name, category="tools", 
                                      debug_mode=dependencies.get('debug_mode', False))
        else:
            self.nb_logger = None
    
    # ToolBase inherits FromConfigBase.__init__ which prevents direct instantiation
        
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the tool."""
        if not self._is_initialized:
            self._is_initialized = True
            logger.debug(f"Tool {self.name} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the tool."""
        self._is_initialized = False
        logger.debug(f"Tool {self.name} shutdown")
    
    @property
    def is_initialized(self) -> bool:
        """Check if tool is initialized."""
        return self._is_initialized
    
    @property
    def call_count(self) -> int:
        """Get number of tool calls."""
        return self._call_count
    
    @property
    def error_count(self) -> int:
        """Get number of tool errors."""
        return self._error_count
    
    async def _record_call(self, success: bool = True) -> None:
        """Record tool call statistics."""
        self._call_count += 1
        if not success:
            self._error_count += 1
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.config.parameters
            }
        }
    
    def to_langchain_tool(self):
        """Convert this tool to a LangChain-compatible tool."""
        try:
            from langchain_core.tools import BaseTool
            from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
            from pydantic import BaseModel, Field
            from typing import Optional, Type
            
            # Create input schema for the tool
            class ToolInputSchema(BaseModel):
                """Input schema for the tool."""
                pass
            
            # Add fields based on parameters
            if self.config.parameters and "properties" in self.config.parameters:
                for param_name, param_info in self.config.parameters["properties"].items():
                    field_type = str  # Default to string
                    field_description = param_info.get("description", "")
                    field_default = ... if param_name in self.config.parameters.get("required", []) else None
                    
                    # Set field on the schema class
                    setattr(ToolInputSchema, param_name, Field(default=field_default, description=field_description))
            
            # If no specific parameters, use a generic input field
            if not hasattr(ToolInputSchema, '__fields__') or not ToolInputSchema.__fields__:
                setattr(ToolInputSchema, 'input', Field(..., description="Input for the tool"))
            
            # Create the LangChain tool class
            class NanoBrainLangChainTool(BaseTool):
                name: str = self.name
                description: str = self.description
                args_schema: Type[BaseModel] = ToolInputSchema
                
                def __init__(self, nanobrain_tool):
                    super().__init__()
                    self.nanobrain_tool = nanobrain_tool
                
                def _run(self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> str:
                    """Synchronous execution."""
                    try:
                        # Run async method in sync context
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(self.nanobrain_tool.execute(**kwargs))
                            return str(result)
                        finally:
                            loop.close()
                    except Exception as e:
                        return f"Error: {str(e)}"
                
                async def _arun(self, run_manager: Optional[AsyncCallbackManagerForToolRun] = None, **kwargs) -> str:
                    """Asynchronous execution."""
                    try:
                        result = await self.nanobrain_tool.execute(**kwargs)
                        return str(result)
                    except Exception as e:
                        return f"Error: {str(e)}"
            
            return NanoBrainLangChainTool(self)
            
        except ImportError:
            logger.warning("LangChain not available, cannot create LangChain tool")
            return None


class FunctionTool(ToolBase):
    """
    Tool that wraps a Python function.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    @classmethod
    def from_config(cls, config: ToolConfig, **kwargs) -> 'FunctionTool':
        """Mandatory from_config implementation for FunctionTool"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve FunctionTool dependencies"""
        func = kwargs.get('func')
        if not func:
            raise ComponentDependencyError("FunctionTool requires 'func' parameter")
        
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'func': func
        }
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize FunctionTool with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.func = dependencies['func']
    
    # FunctionTool inherits FromConfigBase.__init__ which prevents direct instantiation
        
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            if self.config.async_execution and asyncio.iscoroutinefunction(self.func):
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        self.func(**kwargs), 
                        timeout=self.config.timeout
                    )
                else:
                    result = await self.func(**kwargs)
            else:
                # Run in thread pool for sync functions
                loop = asyncio.get_event_loop()
                if self.config.timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: self.func(**kwargs)),
                        timeout=self.config.timeout
                    )
                else:
                    result = await loop.run_in_executor(None, lambda: self.func(**kwargs))
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"FunctionTool {self.name} execution failed: {e}")
            raise


class AgentTool(ToolBase):
    """
    Tool that wraps another Agent for agent-to-agent interaction.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    @classmethod
    def from_config(cls, config: ToolConfig, **kwargs) -> 'AgentTool':
        """Mandatory from_config implementation for AgentTool"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve AgentTool dependencies"""
        agent = kwargs.get('agent')
        if not agent:
            raise ComponentDependencyError("AgentTool requires 'agent' parameter")
        
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'agent': agent
        }
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AgentTool with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.agent = dependencies['agent']
    
    # AgentTool inherits FromConfigBase.__init__ which prevents direct instantiation
        
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped agent."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Extract the main input for the agent
            input_text = kwargs.get('input', kwargs.get('query', kwargs.get('text', '')))
            
            if hasattr(self.agent, 'process'):
                result = await self.agent.process(input_text, **kwargs)
            elif hasattr(self.agent, 'execute'):
                result = await self.agent.execute(input_text, **kwargs)
            else:
                raise ValueError(f"Agent {self.agent.name} has no process or execute method")
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"AgentTool {self.name} execution failed: {e}")
            raise


class StepTool(ToolBase):
    """
    Tool that wraps a Step for step-based processing.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    @classmethod
    def from_config(cls, config: ToolConfig, **kwargs) -> 'StepTool':
        """Mandatory from_config implementation for StepTool"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve StepTool dependencies"""
        step = kwargs.get('step')
        if not step:
            raise ComponentDependencyError("StepTool requires 'step' parameter")
        
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'step': step
        }
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize StepTool with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.step = dependencies['step']
    
    # StepTool inherits FromConfigBase.__init__ which prevents direct instantiation
        
    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped step."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Set input data in step's input data units
            if hasattr(self.step, 'input_data_units') and self.step.input_data_units:
                for i, (key, value) in enumerate(kwargs.items()):
                    if i < len(self.step.input_data_units):
                        await self.step.input_data_units[i].set(value)
            
            # Execute the step
            if hasattr(self.step, 'execute'):
                result = await self.step.execute()
            else:
                raise ValueError(f"Step {self.step.name} has no execute method")
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"StepTool {self.name} execution failed: {e}")
            raise


class LangChainTool(ToolBase):
    """
    Adapter for LangChain tools.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    @classmethod
    def from_config(cls, config: ToolConfig, **kwargs) -> 'LangChainTool':
        """Mandatory from_config implementation for LangChainTool"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve LangChainTool dependencies"""
        langchain_tool = kwargs.get('langchain_tool')
        if not langchain_tool:
            raise ComponentDependencyError("LangChainTool requires 'langchain_tool' parameter")
        
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'langchain_tool': langchain_tool
        }
    
    def _init_from_config(self, config: ToolConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize LangChainTool with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.langchain_tool = dependencies['langchain_tool']
    
    # LangChainTool inherits FromConfigBase.__init__ which prevents direct instantiation
        
    async def execute(self, **kwargs) -> Any:
        """Execute the LangChain tool."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # LangChain tools typically expect a single input string
            input_str = kwargs.get('input', kwargs.get('query', ''))
            
            if hasattr(self.langchain_tool, 'arun'):
                # Async LangChain tool
                result = await self.langchain_tool.arun(input_str)
            elif hasattr(self.langchain_tool, 'run'):
                # Sync LangChain tool - run in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.langchain_tool.run(input_str)
                )
            else:
                raise ValueError(f"LangChain tool {self.name} has no run method")
            
            await self._record_call(True)
            return result
            
        except Exception as e:
            await self._record_call(False)
            logger.error(f"LangChainTool {self.name} execution failed: {e}")
            raise
    
    def get_schema(self) -> Dict[str, Any]:
        """Get schema from LangChain tool."""
        if hasattr(self.langchain_tool, 'args_schema') and self.langchain_tool.args_schema:
            # Convert Pydantic schema to function calling schema
            schema = self.langchain_tool.args_schema.model_json_schema()
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": schema
                }
            }
        else:
            # Fallback to simple string input
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Input for the tool"
                            }
                        },
                        "required": ["input"]
                    }
                }
            }


class ToolRegistry:
    """
    Registry for managing tools.
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolBase] = {}
        
    def register(self, tool: ToolBase) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
    
    def get(self, name: str) -> Optional[ToolBase]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        
        return await tool.execute(**kwargs)
    
    async def initialize_all(self) -> None:
        """Initialize all registered tools."""
        for tool in self._tools.values():
            if not tool.is_initialized:
                await tool.initialize()
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered tools."""
        for tool in self._tools.values():
            if tool.is_initialized:
                await tool.shutdown()


def create_tool(config: Union[Dict[str, Any], ToolConfig], **kwargs) -> ToolBase:
    """
    MANDATORY from_config factory for all tool types
    
    Args:
        config: Tool configuration (dict or ToolConfig)
        **kwargs: Framework-provided dependencies
        
    Returns:
        ToolBase instance created via from_config
        
    Raises:
        ValueError: If tool type is unknown
        ComponentConfigurationError: If configuration is invalid
    """
    logger = get_logger("tool.factory")
    logger.info(f"Creating tool via mandatory from_config")
    
    if isinstance(config, dict):
        config = ToolConfig.from_config(config)
    
    # Handle both enum and string values (due to use_enum_values=True)
    tool_type = config.tool_type
    if isinstance(tool_type, str):
        tool_type = ToolType(tool_type)
    
    try:
        tool_classes = {
            ToolType.FUNCTION: FunctionTool,
            ToolType.AGENT: AgentTool,
            ToolType.STEP: StepTool,
            ToolType.LANGCHAIN: LangChainTool,
            # Note: EXTERNAL tools should use specific external tool classes
        }
        
        tool_class = tool_classes.get(tool_type)
        if not tool_class:
            raise ValueError(f"Unknown tool type: {tool_type}. Available types: {list(tool_classes.keys())}")
        
        # Create instance via from_config
        instance = tool_class.from_config(config, **kwargs)
        
        logger.info(f"Successfully created {tool_class.__name__} via from_config")
        return instance
        
    except Exception as e:
        raise ValueError(f"Failed to create tool '{tool_type}' via from_config: {e}")


def function_tool(name: str, description: str = "", parameters: Optional[Dict[str, Any]] = None):
    """
    Decorator to create a function tool using from_config pattern.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Parameter schema
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> ToolBase:
        config = ToolConfig.from_config({
            'tool_type': 'function',
            'name': name,
            'description': description,
            'parameters': parameters or {}
        })
        return FunctionTool.from_config(config, func=func)
    
    return decorator 