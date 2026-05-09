"""
Base Agent Classes

Base classes for specialized agents in the NanoBrain framework.
"""

import time
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

# Updated imports for nanobrain package structure
from nanobrain.core.agent import Agent, SimpleAgent, ConversationalAgent, AgentConfig
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import get_logger


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
        multi_specialist = DataScienceAgent.from_config(
            'config/data_science.yml')

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

        self.specialized_logger.info(
            f"Specialized agent {getattr(self, 'name', 'unknown')} initialized")

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
            self._domain_specific_metrics[operation_name] = {
                'count': 0, 'errors': 0}

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
                self._specialized_errors_count /
                max(1, self._specialized_operations_count)
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


class SimpleSpecializedAgent(SimpleAgent, SpecializedAgentBase):
    """
    Simple Specialized Agent with proper core agent inheritance

    Inherits from SimpleAgent (which provides core agent functionality)
    and SpecializedAgentBase (which provides specialized agent capabilities).

    ✅ FRAMEWORK COMPLIANCE:
    - Inherits core agent functionality from SimpleAgent
    - Uses unified from_config pattern from FromConfigBase
    - Proper initialization chain ensures all attributes are set
    - Clean single inheritance with mixin for specialization
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
                # Optional field can default to empty
                'description': config.get('description', ''),
                'model': config['model'],  # Required field - no default
                # Optional field can default to empty
                'system_prompt': config.get('system_prompt', ''),
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
        """Initialize SimpleSpecializedAgent with proper parent chain"""
        # Call SimpleAgent initialization (which properly calls Agent._init_from_config)
        super()._init_from_config(config, component_config, dependencies)

        # Initialize specialized agent attributes
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
                    self._track_specialized_operation(
                        "direct_processing", success=True)
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation(
                    "direct_processing", success=False)
                self.specialized_logger.error(
                    f"Specialized processing failed: {e}")

        # Fall back to parent LLM processing
        return await super().process(input_text, **kwargs)


class ConversationalSpecializedAgent(ConversationalAgent, SpecializedAgentBase):
    """
    Conversational Specialized Agent with proper core agent inheritance

    Inherits from ConversationalAgent (which provides core agent and conversation functionality)
    and SpecializedAgentBase (which provides specialized agent capabilities).

    ✅ FRAMEWORK COMPLIANCE:
    - Inherits core agent functionality from ConversationalAgent
    - Uses unified from_config pattern from FromConfigBase
    - Proper initialization chain ensures all attributes are set
    - Clean single inheritance with mixin for specialization
    """

    # Component configuration
    COMPONENT_TYPE = "conversational_specialized_agent"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {}

    @classmethod
    def _get_config_class(cls):
        """Return conversational agent specific config class"""
        return AgentConfig

    def _init_from_config(self, config: AgentConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize ConversationalSpecializedAgent with proper parent chain"""
        # Call ConversationalAgent initialization (which properly calls Agent._init_from_config)
        super()._init_from_config(config, component_config, dependencies)

        # Initialize specialized agent attributes
        self._specialized_operations_count = 0
        self._specialized_errors_count = 0
        self._domain_specific_metrics = {}

        # Get specialized logger
        self.specialized_logger = get_logger(f"specialized.{self.name}")

    async def process(self, input_text: str, **kwargs) -> str:
        """
        Process input with specialized logic first, then fall back to conversational LLM.

        Args:
            input_text: Input text to process
            **kwargs: Additional parameters

        Returns:
            Processed response
        """
        import time
        start_time = time.time()

        try:
            # PHASE 1 DIAGNOSTICS: Enhanced logging for agent execution tracing
            input_preview = str(input_text)[:50] if input_text else "None"
            self.specialized_logger.info(
                f"🔍 [AGENT-TRACE] ConversationalSpecializedAgent.process() ENTRY - Agent: {getattr(self, 'name', 'unknown')}")
            self.specialized_logger.info(
                f"🔍 [AGENT-TRACE] Input preview: {input_preview}...")
            self.specialized_logger.info(
                f"🔍 [AGENT-TRACE] Agent state - LLM client: {hasattr(self, 'llm_client') and self.llm_client is not None}")
            self.specialized_logger.info(
                f"🔍 [AGENT-TRACE] Agent state - Initialized: {getattr(self, '_initialized', False)}")
        except Exception as e:
            print(f"ERROR: Failed to log in process method: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

        # PHASE 1 DIAGNOSTICS: Enhanced specialized processing tracing
        self.specialized_logger.info("🔍 [AGENT-TRACE] Checking specialized processing...")
        try:
            should_handle = self._should_handle_specialized(input_text, **kwargs)
            self.specialized_logger.info(f"🔍 [AGENT-TRACE] _should_handle_specialized() returned: {should_handle}")
        except Exception as e:
            self.specialized_logger.error(f"❌ [AGENT-TRACE] Error in _should_handle_specialized: {e}")
            should_handle = False

        if should_handle:
            try:
                self.specialized_logger.info("🔍 [AGENT-TRACE] Calling _process_specialized_request...")
                specialized_result = await self._process_specialized_request(input_text, **kwargs)
                self.specialized_logger.info(f"🔍 [AGENT-TRACE] _process_specialized_request returned: {specialized_result is not None}")

                if specialized_result is not None:
                    self._track_specialized_operation(
                        "direct_processing", success=True)
                    # Add to conversation history for context
                    if hasattr(self, 'add_to_conversation'):
                        self.add_to_conversation("user", input_text)
                        self.add_to_conversation(
                            "assistant", specialized_result)

                    elapsed_time = time.time() - start_time
                    self.specialized_logger.info(f"✅ [AGENT-TRACE] Specialized processing completed in {elapsed_time:.2f}s")
                    return specialized_result
            except Exception as e:
                self._track_specialized_operation(
                    "direct_processing", success=False)
                self.specialized_logger.error(
                    f"❌ [AGENT-TRACE] Specialized processing failed: {e}")

        # PHASE 1 DIAGNOSTICS: Log transition to LLM processing
        self.specialized_logger.info("🔍 [AGENT-TRACE] Transitioning to LLM processing...")

        # Fall back to conversational processing (using tools if available)
        if hasattr(self, 'tool_registry') and self.tool_registry.list_tools():
            # Determine if any tools should be used for this input
            relevant_tools = self._analyze_input_for_tool_usage(
                input_text, **kwargs)
            if relevant_tools:
                tool_results = {}
                for tool_name in relevant_tools:
                    try:
                        tool_result = self.execute_with_tool(
                            tool_name, {'input': input_text, **kwargs})
                        tool_results[tool_name] = tool_result
                    except Exception as e:
                        self.specialized_logger.warning(
                            f"Tool {tool_name} execution failed: {e}")

                # Enhance response with tool results if available
                if tool_results:
                    enhanced_input = f"Input: {input_text}\nTool Results: {tool_results}"
                    return await self._process_with_conversation_context(enhanced_input, **kwargs)

        # Standard conversational processing
        input_preview = str(input_text)[:50] if input_text else "None"
        self.specialized_logger.info(
            f"🔍 Calling _process_with_conversation_context() for: {input_preview}...")
        try:
            result = await self._process_with_conversation_context(input_text, **kwargs)
            result_preview = str(result)[:50] if result else "None"
            self.specialized_logger.info(
                f"🔍 _process_with_conversation_context() returned: {result_preview}...")
            return result
        except Exception as e:
            self.specialized_logger.error(
                f"❌ Exception in _process_with_conversation_context(): {e}", exc_info=True)
            return f"I encountered an error while processing your request: {str(e)}"

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
        # CRITICAL FIX: Initialize timing variables at the very beginning
        start_time = time.time()

        # Add to conversation history if conversation management is available
        if hasattr(self, 'conversation_history'):
            self.conversation_history.append({
                'role': 'user',
                'content': input_text,
                'timestamp': time.time()
            })

        # PHASE 1 DIAGNOSTICS: Enhanced LLM processing tracing
        try:
            # PHASE 1 DIAGNOSTICS: Comprehensive LLM client state logging
            self.specialized_logger.info("🔍 [AGENT-TRACE] Starting LLM processing...")
            self.specialized_logger.info(
                f"🔍 [AGENT-TRACE] LLM client check: hasattr={hasattr(self, 'llm_client')}, client={getattr(self, 'llm_client', None) is not None}")

            if hasattr(self, 'llm_client') and self.llm_client:
                self.specialized_logger.info(f"🔍 [AGENT-TRACE] LLM client type: {type(self.llm_client)}")

            # Use the parent ConversationalAgent's LLM processing
            if hasattr(self, 'llm_client') and self.llm_client:
                self.specialized_logger.info("🔍 [AGENT-TRACE] Building message context...")

                # Create conversation context with system prompt
                messages = []

                # Add system prompt if available - ensure it's a string
                if hasattr(self, 'system_prompt') and self.system_prompt:
                    system_content = str(self.system_prompt)  # Ensure string conversion
                    messages.append(
                        {"role": "system", "content": system_content})
                    self.specialized_logger.info(f"🔍 [AGENT-TRACE] Added system prompt ({len(system_content)} chars)")

                # Add conversation history if available
                if hasattr(self, 'conversation_history') and self.conversation_history:
                    # Add recent conversation history (last 10 messages to avoid token limits)
                    recent_history = self.conversation_history[-10:]
                    for msg in recent_history:
                        if msg.get('role') in ['user', 'assistant']:
                            msg_content = str(msg['content'])  # Ensure string conversion
                            messages.append({
                                "role": msg['role'],
                                "content": msg_content
                            })
                    self.specialized_logger.info(f"🔍 [AGENT-TRACE] Added {len(recent_history)} history messages")

                # Add current user input - ensure it's a string
                user_content = str(input_text)  # Ensure string conversion
                messages.append({"role": "user", "content": user_content})
                self.specialized_logger.info(f"🔍 [AGENT-TRACE] Total messages prepared: {len(messages)}")

                # PHASE 1 DIAGNOSTICS: Critical LLM call tracing
                try:
                    self.specialized_logger.info("🔍 [AGENT-TRACE] CALLING _call_llm() - THIS IS THE CRITICAL POINT")
                    llm_call_start = time.time()
                    llm_response = await self._call_llm(messages)
                    llm_call_duration = time.time() - llm_call_start
                    self.specialized_logger.info(f"✅ [AGENT-TRACE] _call_llm() completed in {llm_call_duration:.2f}s")

                    # PHASE 1 DIAGNOSTICS: Trace response processing
                    self.specialized_logger.info("🔍 [AGENT-TRACE] Processing LLM response...")
                    self.specialized_logger.info(f"🔍 [AGENT-TRACE] Response type: {type(llm_response)}")
                    self.specialized_logger.info(f"🔍 [AGENT-TRACE] Response has choices: {'choices' in llm_response if llm_response else False}")

                    # Extract response content from the LLM response
                    if llm_response and "choices" in llm_response and llm_response["choices"]:
                        choice = llm_response["choices"][0]
                        message = choice.get("message", {})
                        response = message.get("content", "")

                        self.specialized_logger.info(f"🔍 [AGENT-TRACE] Extracted response length: {len(response) if response else 0}")

                        if response:
                            total_elapsed = time.time() - start_time
                            self.specialized_logger.info(f"✅ [AGENT-TRACE] Agent processing completed successfully in {total_elapsed:.2f}s")
                            return response

                    self.specialized_logger.error("❌ [AGENT-TRACE] No valid response content extracted from LLM response")

                except Exception as e:
                    # CRITICAL FIX: Safe variable access in exception handler
                    current_time = time.time()
                    llm_call_duration = current_time - llm_call_start if 'llm_call_start' in locals() else 0
                    self.specialized_logger.error(f"❌ [AGENT-TRACE] _call_llm() failed after {llm_call_duration:.2f}s: {e}", exc_info=True)
                    # DEVELOPMENT MODE: Re-raise the exception instead of falling back
                    raise RuntimeError(f"LLM API call failed: {e}") from e

            # DEVELOPMENT MODE: No fallbacks - fail immediately with detailed error
            error_msg = f"LLM client is not available or not properly configured. Cannot process request: '{input_text[:100]}...'"
            self.specialized_logger.error(f"CRITICAL ERROR: {error_msg}")
            raise RuntimeError(error_msg)

        except Exception as e:
            self.specialized_logger.error(f"Error in LLM processing: {e}")
            # DEVELOPMENT MODE: Re-raise instead of providing fallback response
            raise

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
