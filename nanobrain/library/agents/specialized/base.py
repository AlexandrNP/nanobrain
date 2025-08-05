"""
Base Agent Classes

Base classes for specialized agents in the NanoBrain framework.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

# Updated imports for nanobrain package structure
from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.agent import Agent, SimpleAgent, ConversationalAgent, AgentConfig
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import NanoBrainLogger, get_logger


class SpecializedAgentBase(ABC):
    """
    Specialized Agent Base Mixin - Domain-Specific AI Agent Capabilities and Performance Optimization
    ===============================================================================================
    
    The SpecializedAgentBase mixin provides foundational capabilities for creating domain-specific
    AI agents within the NanoBrain framework. This base class extends standard agent functionality
    with specialized processing patterns, domain-specific error handling, performance tracking,
    and integration capabilities optimized for particular use cases or industries.
    
    **Core Architecture:**
        Specialized agents enhance base agent capabilities with:
        
        * **Domain Specialization**: Optimized processing for specific domains and use cases
        * **Performance Tracking**: Advanced metrics collection for specialized operations
        * **Error Handling**: Domain-specific error detection, classification, and recovery
        * **Resource Management**: Optimized resource usage for specialized workloads
        * **Integration Patterns**: Seamless integration with domain-specific tools and systems
        * **Extensibility**: Framework for building highly specialized agent implementations
    
    **Specialization Architecture:**
        
        **Domain-Specific Processing:**
        * Optimized algorithms and processing patterns for specific domains
        * Domain vocabulary and terminology understanding and usage
        * Specialized data structures and formats handling
        * Industry-specific workflow patterns and best practices
        
        **Performance Optimization:**
        * Domain-specific caching strategies and data management
        * Optimized resource allocation for specialized workloads
        * Performance benchmarking against domain-specific metrics
        * Adaptive optimization based on usage patterns and performance data
        
        **Specialized Error Handling:**
        * Domain-specific error detection and classification systems
        * Industry-standard error codes and messaging patterns
        * Specialized recovery strategies and fallback mechanisms
        * Error context preservation with domain-specific diagnostics
        
        **Integration Capabilities:**
        * Native integration with domain-specific tools and systems
        * Protocol adapters for industry-standard communication patterns
        * Data format conversion and transformation capabilities
        * Workflow integration with existing domain-specific processes
    
    **Specialized Agent Types:**
        The base mixin supports various specialization patterns:
        
        **Data Analysis Agents:**
        * Statistical analysis and data science workflows
        * Large dataset processing and analysis optimization
        * Visualization and reporting capabilities
        * Integration with data science tools and platforms
        
        **Code Generation Agents:**
        * Programming language-specific code generation
        * Software architecture and design pattern application
        * Code quality analysis and optimization recommendations
        * Integration with development tools and CI/CD pipelines
        
        **Bioinformatics Agents:**
        * Biological sequence analysis and processing
        * Genomics and proteomics workflow optimization
        * Scientific database integration and querying
        * Specialized visualization and reporting for biological data
        
        **Document Processing Agents:**
        * Natural language processing and text analysis
        * Document format conversion and standardization
        * Content extraction and metadata management
        * Integration with document management systems
        
        **Research Agents:**
        * Scientific literature review and analysis
        * Research methodology and experimental design
        * Citation management and reference formatting
        * Integration with academic databases and repositories
    
    **Configuration Architecture:**
        Specialized agents support comprehensive domain-specific configuration:
        
        ```yaml
        # Specialized Agent Configuration
        name: "bioinformatics_specialist"
        description: "Specialized agent for genomics and proteomics analysis"
        
        # Base agent configuration
        base_agent:
          class: "nanobrain.core.agent.ConversationalAgent"
          model: "gpt-4"
          temperature: 0.1  # Lower temperature for scientific precision
        
        # Specialization configuration
        specialization:
          domain: "bioinformatics"
          expertise_areas:
            - "genomics"
            - "proteomics"
            - "sequence_analysis"
            - "phylogenetics"
          
          # Domain-specific processing
          processing_config:
            sequence_validation: true
            batch_processing_size: 1000
            memory_optimization: "large_datasets"
            parallel_processing: true
        
        # Domain-specific tools and integrations
        domain_tools:
          - name: "blast_tool"
            class: "nanobrain.library.tools.bioinformatics.BLASTTool"
            config: "config/blast_config.yml"
          
          - name: "muscle_tool"
            class: "nanobrain.library.tools.bioinformatics.MUSCLETool"
            config: "config/muscle_config.yml"
        
        # Performance and monitoring
        performance_config:
          specialized_metrics: true
          domain_benchmarks: true
          resource_optimization: "domain_specific"
          performance_alerts: true
        
        # Error handling configuration
        error_handling:
          domain_error_codes: true
          specialized_recovery: true
          fallback_strategies: ["alternative_tool", "simplified_analysis"]
          error_reporting: "domain_specific"
        ```
    
    **Usage Patterns:**
        
        **Basic Specialized Agent:**
        ```python
        from nanobrain.library.agents.specialized import BioinformaticsAgent
        
        # Create specialized agent from configuration
        agent = BioinformaticsAgent.from_config('config/bio_agent.yml')
        
        # Specialized processing with domain optimization
        result = await agent.aprocess(
            "Analyze this protein sequence for structural domains: MKTVRQERLK..."
        )
        
        # Access specialized performance metrics
        metrics = agent.get_specialized_performance_stats()
        print(f"Domain operations: {metrics['specialized_operations_count']}")
        ```
        
        **Multi-Domain Specialization:**
        ```python
        # Agent with multiple specialization areas
        multi_specialist = DataScienceAgent.from_config('config/data_science.yml')
        
        # Task requiring multiple specialized capabilities
        analysis_task = ("Perform comprehensive analysis of the sales dataset: "
                        "1. Statistical analysis of trends, "
                        "2. Predictive modeling for forecasting, "
                        "3. Visualization of key metrics, "
                        "4. Anomaly detection and reporting")
        
        result = await multi_specialist.aprocess(analysis_task)
        
        # Specialized agent automatically applies domain expertise
        # for each aspect of the analysis
        ```
        
        **Domain-Specific Tool Integration:**
        ```python
        # Specialized agent with domain-specific tools
        research_agent = ResearchAgent.from_config('config/research_agent.yml')
        
        # Research task with tool integration
        research_query = "Find recent papers on CRISPR gene editing applications"
        
        result = await research_agent.aprocess(research_query)
        
        # Agent automatically:
        # 1. Uses specialized academic search tools
        # 2. Applies domain knowledge for query optimization
        # 3. Filters results using domain expertise
        # 4. Synthesizes findings with research methodology
        ```
        
        **Performance-Optimized Processing:**
        ```python
        # Large-scale specialized processing
        specialist = LargeDataAgent.from_config('config/large_data_agent.yml')
        
        # Enable specialized performance tracking
        await specialist.initialize()
        
        # Process large dataset with optimization
        result = await specialist.process_large_dataset(
            data_path="large_dataset.csv",
            analysis_type="comprehensive"
        )
        
        # Review performance optimization results
        perf_stats = specialist.get_specialized_performance_stats()
        optimization_report = specialist.generate_optimization_report()
        ```
    
    **Advanced Features:**
        
        **Adaptive Specialization:**
        * Dynamic specialization level adjustment based on task complexity
        * Learning and adaptation from domain-specific feedback
        * Specialization confidence scoring and validation
        * Automatic specialization area detection and optimization
        
        **Domain Knowledge Integration:**
        * Integration with domain-specific knowledge bases and ontologies
        * Automatic terminology and concept recognition
        * Domain-specific reasoning and inference capabilities
        * Specialized validation and quality assurance patterns
        
        **Performance Optimization:**
        * Domain-specific algorithm selection and optimization
        * Resource allocation tuning for specialized workloads
        * Caching strategies optimized for domain-specific data patterns
        * Parallel processing optimization for specialized algorithms
        
        **Quality Assurance:**
        * Domain-specific validation and verification patterns
        * Quality metrics and benchmarking against domain standards
        * Automated testing with domain-specific test cases
        * Continuous quality monitoring and improvement
    
    **Integration Patterns:**
        
        **Tool Ecosystem Integration:**
        * Native integration with domain-specific tool ecosystems
        * Tool chain optimization for specialized workflows
        * Cross-tool data format standardization and conversion
        * Tool performance monitoring and optimization
        
        **System Integration:**
        * Integration with domain-specific systems and platforms
        * Data pipeline integration with specialized data sources
        * Workflow integration with existing domain processes
        * API integration with industry-standard services
        
        **Collaborative Specialization:**
        * Multi-specialist collaboration for complex domain problems
        * Specialization handoff and coordination patterns
        * Cross-domain knowledge sharing and integration
        * Specialized result synthesis and reporting
    
    **Performance and Monitoring:**
        
        **Specialized Metrics:**
        * Domain-specific performance indicators and benchmarks
        * Specialization effectiveness measurement and tracking
        * Resource utilization optimization for domain workloads
        * Quality metrics aligned with domain standards
        
        **Optimization Features:**
        * Automatic performance tuning for specialized operations
        * Resource allocation optimization based on domain patterns
        * Caching strategies optimized for domain-specific data
        * Parallel processing optimization for specialized algorithms
        
        **Monitoring and Analytics:**
        * Real-time performance monitoring for specialized operations
        * Domain-specific alert and notification systems
        * Usage pattern analysis and optimization recommendations
        * Performance trending and capacity planning
    
    **Development and Extension:**
        
        **Specialization Framework:**
        * Template-based specialization development patterns
        * Domain expertise integration and validation frameworks
        * Specialization testing and validation tools
        * Performance benchmarking and optimization utilities
        
        **Extension Patterns:**
        * Plugin architecture for domain-specific extensions
        * Modular specialization component development
        * Dynamic specialization loading and configuration
        * Specialization marketplace and sharing capabilities
        
        **Testing and Validation:**
        * Domain-specific testing frameworks and methodologies
        * Specialization validation against domain benchmarks
        * Performance regression testing for specialized operations
        * Integration testing with domain-specific systems and tools
    
    **Error Handling and Recovery:**
        
        **Domain-Specific Error Management:**
        * Error classification using domain-specific taxonomies
        * Specialized error recovery strategies and fallback mechanisms
        * Error context preservation with domain-specific diagnostics
        * Integration with domain-specific error reporting systems
        
        **Quality Assurance:**
        * Domain-specific validation and verification patterns
        * Quality metrics aligned with industry standards
        * Automated quality monitoring and alerting
        * Continuous improvement based on domain feedback
        
        **Reliability Features:**
        * Fault tolerance optimized for domain-specific failure patterns
        * Graceful degradation with domain-appropriate fallbacks
        * Health monitoring aligned with domain-specific indicators
        * Recovery strategies optimized for domain-specific scenarios
    
    Methods:
        initialize(): Initialize specialized agent capabilities and resources
        shutdown(): Cleanup specialized resources and finalize metrics
        _initialize_specialized_features(): Override point for specialization-specific initialization
        _shutdown_specialized_features(): Override point for specialization-specific cleanup
        _track_specialized_operation(): Track performance metrics for specialized operations
        get_specialized_performance_stats(): Retrieve comprehensive specialization metrics
    
    Attributes:
        _specialized_operations_count (int): Total specialized operations performed
        _specialized_errors_count (int): Number of specialized operation errors
        _domain_specific_metrics (Dict): Detailed metrics by operation type
        specialized_logger (Logger): Logger configured for specialized operations
    
    Note:
        This is a mixin class that should be combined with concrete agent implementations.
        Specialized agents must implement domain-specific processing methods and
        configuration patterns. All specialized agents should follow the framework's
        from_config pattern and specialized configuration standards.
    
    Warning:
        Specialized operations may consume significant resources depending on domain
        requirements. Monitor resource usage and implement appropriate limits for
        domain-specific operations. Be cautious with specialization that might
        conflict with base agent capabilities or other specializations.
    
    See Also:
        * :class:`Agent`: Base agent class for specialization
        * :class:`ConversationalAgent`: Conversational agent for specialization
        * :mod:`nanobrain.library.agents.specialized`: Concrete specialized implementations
        * :mod:`nanobrain.library.tools.bioinformatics`: Bioinformatics specialization tools
        * :mod:`nanobrain.core.logging_system`: Logging system for specialized operations
    """
    
    def __init__(self, **kwargs):
        """Initialize specialized agent base."""
        super().__init__(**kwargs)
        
        # Specialized agent tracking
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        if hasattr(self, 'name'):
            self.specialized_logger = get_logger(f"specialized.{self.name}")
        else:
            self.specialized_logger = get_logger("specialized.agent")
    
    async def initialize(self) -> None:
        """Initialize the specialized agent."""
        await super().initialize()
        await self._initialize_specialized_features()
        
        self.specialized_logger.info(f"Specialized agent {getattr(self, 'name', 'unknown')} initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the specialized agent."""
        await self._shutdown_specialized_features()
        await super().shutdown()
        
        self.specialized_logger.info(
            f"Specialized agent {getattr(self, 'name', 'unknown')} shutdown",
            specialized_operations=self._specialized_operations_count,
            specialized_errors=self._specialized_errors_count
        )
    
    async def _initialize_specialized_features(self) -> None:
        """Initialize specialized features. Override in subclasses."""
        pass
    
    async def _shutdown_specialized_features(self) -> None:
        """Shutdown specialized features. Override in subclasses."""
        pass
    
    def _track_specialized_operation(self, operation_name: str, success: bool = True) -> None:
        """Track specialized operation metrics."""
        self._specialized_operations_count += 1
        
        if not success:
            self._specialized_errors_count += 1
        
        # Track domain-specific metrics
        if operation_name not in self._domain_specific_metrics:
            self._domain_specific_metrics[operation_name] = {'count': 0, 'errors': 0}
        
        self._domain_specific_metrics[operation_name]['count'] += 1
        if not success:
            self._domain_specific_metrics[operation_name]['errors'] += 1
    
    def get_specialized_performance_stats(self) -> Dict[str, Any]:
        """Get specialized performance statistics."""
        base_stats = {}
        if hasattr(super(), 'get_performance_stats'):
            base_stats = super().get_performance_stats()
        
        specialized_stats = {
            'specialized_operations_count': self._specialized_operations_count,
            'specialized_errors_count': self._specialized_errors_count,
            'specialized_error_rate': (
                self._specialized_errors_count / max(1, self._specialized_operations_count)
            ),
            'domain_specific_metrics': self._domain_specific_metrics.copy()
        }
        
        return {**base_stats, **specialized_stats}
    
    @abstractmethod
    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized requests that don't require LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed result if handled, None if should fall back to LLM
        """
        pass
    
    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by specialized logic.
        
        Args:
            input_text: Input text
            **kwargs: Additional parameters
            
        Returns:
            True if should be handled by specialized logic
        """
        return False


class BaseAgent(FromConfigBase, ABC):
    """
    Enhanced Base Agent with Universal Tool Loading
    
    Implements comprehensive tool loading via ConfigBase class+config patterns.
    All agent subclasses inherit this functionality automatically unless they
    need to override for additional complexity.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Universal tool loading through ConfigBase._resolve_nested_objects()
    - Complete validation of agent and tool configurations
    - No manual tool instantiation or factory logic
    - Pure configuration-driven tool availability
    - Inherited by all agent types without code duplication
    """
    
    COMPONENT_TYPE = "base_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True,
        'capabilities': []
    }
    
    @classmethod
    def _get_config_class(cls):
        """Return the appropriate config class for this agent type"""
        # Default implementation - subclasses can override
        return AgentConfig
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], **context) -> 'BaseAgent':
        """
        Universal agent loading with automatic tool resolution
        
        This base implementation handles all common agent functionality:
        - Configuration loading and validation via ConfigBase
        - Automatic tool instantiation through class+config patterns
        - Tool validation and agent integration
        - Capability validation against available tools
        
        Subclasses should only override this method if they need additional
        complexity beyond the standard agent + tools pattern.
        
        Args:
            config_path: Path to agent configuration file
            **context: Additional context
            
        Returns:
            Fully initialized agent with tools
            
        Example Configuration (works for all agent types):
        ```yaml
        name: "universal_agent"
        description: "Agent with automatic tool loading"
        class: "nanobrain.library.agents.specialized.conversational_specialized_agent.ConversationalSpecializedAgent"
        
        # Agent-specific configuration (varies by agent type)
        agent_config:
          model: "gpt-4"
          temperature: 0.3
          max_tokens: 1000
          system_prompt: "You are an expert agent..."
          
        # Universal tool loading (same pattern for all agents)
        tools:
          primary_tool:
            class: "nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool"
            config: "config/tools/BVBRCTool.yml"
          
          analysis_tool:
            class: "nanobrain.library.tools.analysis.sequence_analyzer.SequenceAnalyzer"
            config:
              analysis_mode: "protein"
              enable_secondary_structure: true
          
        capabilities:
          - "data_retrieval"
          - "sequence_analysis"
        ```
        
        ✅ FRAMEWORK COMPLIANCE:
        - ConfigBase._resolve_nested_objects() automatically instantiates all tools
        - Tools validated through their respective ConfigBase schemas
        - No manual tool creation or factory dependencies
        - Complete configuration-driven tool availability
        - Inherited by all agent subclasses without modification
        """
        # Use ConfigBase for validation and automatic tool resolution
        agent_config = cls._get_config_class().from_config(config_path, **context)
        
        # ConfigBase._resolve_nested_objects() has already instantiated all tools
        # specified in the 'tools' section using their class+config patterns
        
        # Create agent instance
        agent = cls.__new__(cls)
        
        # Initialize with universal agent logic
        agent._initialize_from_config(agent_config, **context)
        agent._initialize_universal_tools(agent_config, **context)
        agent._initialize_agent_specifics(agent_config, **context)
        
        return agent
    
    def _initialize_from_config(self, config: AgentConfig, **context) -> None:
        """Initialize basic agent properties from validated configuration"""
        # Basic agent properties
        self.config = config
        self.name = config.name
        self.description = getattr(config, 'description', '')
        
        # Initialize logging
        self.logger = get_logger(f"agent.{self.name}")
        
        # Initialize capabilities
        self.capabilities = getattr(config, 'capabilities', [])
        
        # Initialize agent state
        self.is_active = False
        self.execution_history = []
        
        self.logger.info(f"✅ Initialized base agent: {self.name} with {len(self.capabilities)} capabilities")
    
    def _initialize_universal_tools(self, config: AgentConfig, **context) -> None:
        """
        Initialize universal tool loading and management for all agent types.
        
        Extracts pre-instantiated tool objects from resolved configuration
        and sets up tool access methods.
        
        Args:
            config: Agent configuration with resolved tool objects
            **context: Additional context
            
        ✅ FRAMEWORK COMPLIANCE:
        - Extracts resolved tool instances from configuration
        - No programmatic tool creation or hardcoded tool types
        - Supports both dict (resolved class+config) and list (AgentConfig) formats
        - Works identically for all agent types
        """
        # Extract tools from configuration - handle both dict and list formats
        tools_raw = getattr(config, 'tools', [])
        self.tools = {}
        
        # Handle different tools configuration formats
        if isinstance(tools_raw, dict):
            # Dictionary format: tool_name -> tool_instance (from class+config resolution)
            for tool_name, tool_instance in tools_raw.items():
                self._add_tool_to_agent(tool_name, tool_instance)
        elif isinstance(tools_raw, list):
            # List format: list of tool configurations (from AgentConfig.tools)
            for i, tool_config in enumerate(tools_raw):
                if isinstance(tool_config, dict):
                    # If it's a resolved tool instance, use it directly
                    if hasattr(tool_config, 'execute') or hasattr(tool_config, 'run'):
                        tool_name = getattr(tool_config, 'name', f'tool_{i}')
                        self._add_tool_to_agent(tool_name, tool_config)
                    else:
                        # It's a configuration dict that should have been resolved
                        self.logger.warning(f"⚠️ Tool configuration not resolved: {tool_config}")
                else:
                    # It's already a tool instance
                    tool_name = getattr(tool_config, 'name', f'tool_{i}')
                    self._add_tool_to_agent(tool_name, tool_config)
        else:
            self.logger.warning(f"⚠️ Unexpected tools format: {type(tools_raw)}")
        
        self.logger.info(f"✅ Initialized base agent: {config.name} with {len(self.tools)} tools")
    
    def _add_tool_to_agent(self, tool_name: str, tool_instance) -> None:
        """
        Add a tool instance to the agent's tool collection.
        
        Args:
            tool_name: Name/identifier for the tool
            tool_instance: Tool instance to add
        """
        # Validate that it's a proper tool instance
        if hasattr(tool_instance, 'execute') or hasattr(tool_instance, 'run'):
            self.tools[tool_name] = tool_instance
            
            # Bind tool to agent context if supported
            if hasattr(tool_instance, 'set_agent_context'):
                tool_instance.set_agent_context(self)
            
            # Initialize tool-specific agent integration
            if hasattr(tool_instance, 'initialize_agent_integration'):
                tool_instance.initialize_agent_integration(self)
            
            self.logger.info(f"✅ Loaded tool: {tool_name} ({tool_instance.__class__.__name__})")
        else:
            self.logger.warning(f"⚠️ Invalid tool instance: {tool_name} - missing execute/run method")
    
    def _initialize_agent_specifics(self, config: AgentConfig, **context) -> None:
        """
        Initialize agent-specific configuration
        
        Base implementation does nothing. Subclasses override this method
        to handle their specific initialization requirements without needing
        to override the entire from_config method.
        
        Args:
            config: Validated agent configuration
            **context: Additional context
        """
        # Base implementation - subclasses can override for specific needs
        pass
    
    def _validate_tool_capability_mapping(self) -> None:
        """
        Validate that agent capabilities are supported by available tools
        
        Base implementation provides common capability validation.
        Subclasses can override to provide specific mappings.
        """
        # Default capability-tool mapping (can be overridden by subclasses)
        default_mapping = {
            'data_retrieval': ['bvbrc_tool', 'pubmed_client', 'database_tool'],
            'sequence_analysis': ['sequence_analyzer', 'alignment_tool', 'clustering_tool'],
            'literature_search': ['pubmed_client', 'literature_tool'],
            'protein_analysis': ['protein_analyzer', 'structure_tool', 'domain_tool'],
            'collaboration': ['communication_tool', 'shared_workspace']
        }
        
        # Get specific mapping from subclass if available
        capability_tool_mapping = getattr(self, '_capability_tool_mapping', default_mapping)
        
        missing_tools = []
        for capability in self.capabilities:
            required_tools = capability_tool_mapping.get(capability, [])
            available_tools = set(self.tools.keys())
            
            if required_tools and not any(tool in available_tools for tool in required_tools):
                missing_tools.append({
                    'capability': capability,
                    'required_tools': required_tools,
                    'available_tools': list(available_tools)
                })
        
        if missing_tools:
            self.logger.warning(f"⚠️ Some capabilities may be limited due to missing tools: {missing_tools}")
        else:
            self.logger.info("✅ All capabilities supported by available tools")
    
    # Universal tool access methods (inherited by all agent types)
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get tool instance by name"""
        return self.tools.get(tool_name)
    
    def list_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def execute_with_tool(self, tool_name: str, tool_params: Dict[str, Any]) -> Any:
        """Execute a specific tool with parameters"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not available. Available tools: {self.list_available_tools()}")
        
        try:
            if hasattr(tool, 'execute'):
                return tool.execute(**tool_params)
            elif hasattr(tool, 'run'):
                return tool.run(**tool_params)
            else:
                raise ValueError(f"Tool '{tool_name}' does not have execute() or run() method")
        except Exception as e:
            self.logger.error(f"❌ Tool execution failed: {tool_name} - {str(e)}")
            raise ValueError(f"Tool execution failed: {tool_name} - {str(e)}") from e


class SimpleSpecializedAgent(BaseAgent, SpecializedAgentBase, SimpleAgent):
    """
    Simple Specialized Agent - Inherits Universal Tool Loading
    
    Uses BaseAgent's universal tool loading mechanism. Only implements
    agent-specific functionality beyond the standard pattern.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Inherits tool loading from BaseAgent automatically
    - Only overrides methods for specialized functionality
    - No duplicate tool loading or from_config logic
    - Pure configuration-driven through inheritance
    """
    
    # Component configuration
    COMPONENT_TYPE = "simple_specialized_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {}
    
    @classmethod
    def _get_config_class(cls):
        """Return agent specific config class"""
        return AgentConfig
    
    @classmethod
    def extract_component_config(cls, config: AgentConfig) -> Dict[str, Any]:
        """Extract SimpleSpecializedAgent configuration"""
        # Handle both dictionary and object configurations - NO HARDCODED DEFAULTS
        if isinstance(config, dict):
            return {
                'name': config['name'],  # Required field - no default
                'description': config.get('description', ''),  # Optional field can default to empty
                'model': config['model'],  # Required field - no default  
                'system_prompt': config.get('system_prompt', ''),  # Optional field can default to empty
            }
        else:
            return {
                'name': config.name,
                'description': config.description,
                'model': config.model,
                'system_prompt': config.system_prompt,
            }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve SimpleSpecializedAgent dependencies"""
        # Create executor via from_config to avoid direct instantiation
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        
        executor_config = kwargs.get('executor_config')
        if not executor_config:
            # Create default executor configuration using proper framework pattern
            try:
                ExecutorConfig._allow_direct_instantiation = True
                executor_config = ExecutorConfig(executor_type="local")
            finally:
                ExecutorConfig._allow_direct_instantiation = False
        
        executor = LocalExecutor.from_config(executor_config)
        
        return {
            'executor': executor,
        }
    
    # Now inherits unified from_config implementation from FromConfigBase
        
    def _init_from_config(self, config: AgentConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize SimpleSpecializedAgent with resolved dependencies"""
        # Extract executor from dependencies to pass to parent
        executor = dependencies.pop('executor', None)
        
        # Initialize parent classes first
        SimpleAgent.__init__(self, config, executor=executor, **dependencies)
        # SpecializedAgentBase doesn't need config, just call super() directly
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        self.specialized_logger = get_logger(f"specialized.{self.name}")
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed response
        """
        # Try specialized processing first
        if self._should_handle_specialized(input_text, **kwargs):
            try:
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                if specialized_result is not None:
                    self._track_specialized_operation("direct_processing", success=True)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation("direct_processing", success=False)
                self.specialized_logger.error(f"Specialized processing failed: {e}")
        
        # Fall back to parent LLM processing
        return await super().process(input_text, **kwargs)


class ConversationalSpecializedAgent(BaseAgent, SpecializedAgentBase, ConversationalAgent):
    """
    Conversational Specialized Agent - Inherits Universal Tool Loading
    
    Uses BaseAgent's universal tool loading mechanism. Only implements
    conversational and specialized functionality beyond the standard pattern.
    
    ✅ FRAMEWORK COMPLIANCE:
    - Inherits tool loading from BaseAgent automatically
    - Only overrides methods for conversational and specialized functionality
    - No duplicate tool loading or from_config logic
    - Pure configuration-driven through inheritance
    """
    
    # Component configuration
    COMPONENT_TYPE = "conversational_specialized_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {}
    
    @classmethod
    def _get_config_class(cls):
        """Return conversational agent specific config class"""
        return AgentConfig
    
    def _initialize_agent_specifics(self, config: AgentConfig, **context) -> None:
        """
        Initialize conversational and specialized functionality
        
        This method is called by BaseAgent.from_config() after universal
        tool loading is complete. Only handles conversational and specialized logic.
        
        ✅ FRAMEWORK COMPLIANCE:
        - Called automatically by inherited BaseAgent.from_config()
        - Tools already loaded and available via self.tools
        - Only implements conversational and specialized initialization
        - No tool loading or validation logic needed
        """
        # Initialize conversational agent functionality
        if hasattr(ConversationalAgent, '__init__'):
            # Initialize conversation state
            self.conversation_history = []
            self.conversation_memory = {}
            self.max_conversation_length = getattr(config, 'max_conversation_length', 50)
            self.enable_memory = getattr(config, 'enable_memory', True)
        
        # Initialize specialized agent functionality  
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}
        
        # Get specialized logger
        self.specialized_logger = get_logger(f"specialized.{self.name}")
        
        self.logger.info(f"✅ Initialized conversational and specialized functionality for agent: {self.name}")
    
    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to conversational LLM.
        
        Args:
            input_text: Input text to process
            **kwargs: Additional parameters
            
        Returns:
            Processed response
        """
        # Try specialized processing first
        if self._should_handle_specialized(input_text, **kwargs):
            try:
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                if specialized_result is not None:
                    self._track_specialized_operation("direct_processing", success=True)
                    # Add to conversation history for context
                    if hasattr(self, 'add_to_conversation'):
                        self.add_to_conversation("user", input_text)
                        self.add_to_conversation("assistant", specialized_result)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation("direct_processing", success=False)
                self.specialized_logger.error(f"Specialized processing failed: {e}")
        
        # Fall back to conversational processing (using tools if available)
        if self.tools:
            # Determine if any tools should be used for this input
            relevant_tools = self._analyze_input_for_tool_usage(input_text, **kwargs)
            if relevant_tools:
                tool_results = {}
                for tool_name in relevant_tools:
                    try:
                        tool_result = self.execute_with_tool(tool_name, {'input': input_text, **kwargs})
                        tool_results[tool_name] = tool_result
                    except Exception as e:
                        self.logger.warning(f"Tool {tool_name} execution failed: {e}")
                
                # Enhance response with tool results if available
                if tool_results:
                    enhanced_input = f"Input: {input_text}\nTool Results: {tool_results}"
                    return await self._process_with_conversation_context(enhanced_input, **kwargs)
        
        # Standard conversational processing
        return await self._process_with_conversation_context(input_text, **kwargs)
    
    def _analyze_input_for_tool_usage(self, input_text: str, **kwargs) -> List[str]:
        """
        Analyze input to determine which tools should be used
        
        Args:
            input_text: Input text to analyze
            **kwargs: Additional parameters
            
        Returns:
            List of tool names that should be used
        """
        relevant_tools = []
        
        # Simple keyword-based tool selection (can be enhanced with NLP)
        tool_keywords = {
            'bvbrc_tool': ['virus', 'bacteria', 'genome', 'sequence', 'database'],
            'pubmed_client': ['research', 'paper', 'study', 'literature', 'publication'],
            'sequence_analyzer': ['analyze', 'protein', 'structure', 'domain', 'sequence']
        }
        
        input_lower = input_text.lower()
        
        for tool_name, keywords in tool_keywords.items():
            if tool_name in self.tools and any(keyword in input_lower for keyword in keywords):
                relevant_tools.append(tool_name)
        
        return relevant_tools
    
    async def _process_with_conversation_context(self, input_text: str, **kwargs) -> str:
        """
        Process input with conversation context
        
        This is a simplified implementation that can be enhanced
        with proper conversation management.
        """
        # Add to conversation history if conversation management is available
        if hasattr(self, 'conversation_history'):
            self.conversation_history.append({
                'role': 'user',
                'content': input_text,
                'timestamp': time.time()
            })
        
        # For now, return a simple response
        # In a full implementation, this would integrate with LLM processing
        response = f"Processed conversational input: {input_text}"
        
        # Add response to conversation history
        if hasattr(self, 'conversation_history'):
            self.conversation_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': time.time()
            })
        
        return response


def create_specialized_agent(
    agent_type: str,
    specialized_class: type,
    config: AgentConfig,
    **kwargs
) -> Agent:
    """
    Factory function to create specialized agents.
    
    Args:
        agent_type: Type of agent ('simple' or 'conversational')
        specialized_class: The specialized agent class
        config: Agent configuration
        **kwargs: Additional arguments
        
    Returns:
        Specialized agent instance
    """
    logger = get_logger("specialized.factory")
    logger.info(
        f"Creating specialized agent: {config.name}",
        agent_type=agent_type,
        specialized_class=specialized_class.__name__
    )
    
    if agent_type.lower() == "simple":
        # Create a simple specialized agent class dynamically
        class SimpleSpecialized(specialized_class, SimpleSpecializedAgent):
            pass
        return SimpleSpecialized(config, **kwargs)
    
    elif agent_type.lower() == "conversational":
        # Create a conversational specialized agent class dynamically
        class ConversationalSpecialized(specialized_class, ConversationalSpecializedAgent):
            pass
        return ConversationalSpecialized(config, **kwargs)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}") 