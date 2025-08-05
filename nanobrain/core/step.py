"""
Step System for NanoBrain Framework

Provides event-driven data processing with DataUnit integration.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import importlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from .component_base import (
    FromConfigBase, ComponentConfigurationError, ComponentDependencyError,
    import_class_from_path
)
from .executor import ExecutorBase, LocalExecutor, ExecutorConfig
from .data_unit import DataUnitBase, DataUnitMemory, DataUnitConfig
from .trigger import TriggerBase, DataUpdatedTrigger, TriggerConfig
from .link import LinkBase, DirectLink, LinkConfig
from .logging_system import (
    NanoBrainLogger, get_logger, OperationType, trace_function_calls
)
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


class StepConfig(ConfigBase):
    """
    Configuration for steps - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: StepConfig(name="test", class="...")
    ✅ REQUIRED: StepConfig.from_config('path/to/config.yml')
    """
    
    name: str
    description: str = ""
    executor_config: Optional[ExecutorConfig] = None
    input_configs: Dict[str, DataUnitConfig] = Field(default_factory=dict)
    output_config: Optional[DataUnitConfig] = None
    trigger_config: Optional[TriggerConfig] = None
    auto_initialize: bool = True
    debug_mode: bool = False
    enable_logging: bool = True
    log_data_transfers: bool = True
    log_executions: bool = True
    
    # NEW: Step-level tool configuration
    tools: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)
    
    # Step-specific agent configurations (optional, step-dependent)
    extraction_agent: Optional[Any] = Field(default=None, description="Extraction agent for specialized processing")
    agents: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Multiple agents for complex processing steps")
    
    # EVENT-DRIVEN ARCHITECTURE: Step-level data units and triggers  
    input_data_units: Optional[Dict[str, Union[Dict[str, Any], 'DataUnitBase']]] = Field(default_factory=dict)
    output_data_units: Optional[Dict[str, Union[Dict[str, Any], 'DataUnitBase']]] = Field(default_factory=dict)
    # ✅ UNIFIED RESOLUTION: Accept both dict configs and resolved trigger objects (workflows ARE steps)
    triggers: Optional[List[Union[Dict[str, Any], 'TriggerBase']]] = Field(default_factory=list)


class BaseStep(FromConfigBase, ABC):
    """
    Base Step Class - Event-Driven Data Processing and Workflow Building Blocks
    ==========================================================================
    
    The BaseStep class is the foundational component for creating data processing
    units within the NanoBrain framework. Steps represent discrete processing
    operations that transform data, execute computations, and coordinate with
    other components through event-driven architecture patterns.
    
    **Core Architecture:**
        Steps represent autonomous processing units that:
        
        * **Process Data**: Transform input data through configurable operations
        * **Manage State**: Track processing state and maintain operation history
        * **Coordinate Events**: Respond to triggers and emit events for workflow orchestration
        * **Handle Resources**: Manage computational resources and cleanup operations
        * **Integrate Components**: Seamlessly work with agents, tools, and other steps
        * **Execute Asynchronously**: Support concurrent and parallel processing patterns
    
    **Biological Analogy:**
        Like functional neural circuits that process specific types of information
        and pass results to other circuits, steps process specific operations and
        pass results to other steps. Neural circuits are specialized for particular
        functions (visual processing, motor control, etc.) and coordinate through
        complex signaling - exactly how steps specialize for specific operations
        and coordinate through data units, links, and triggers.
    
    **Data Processing Architecture:**
        
        **Input Management:**
        * Multiple input data units with type validation
        * Streaming data support for real-time processing
        * Batch processing capabilities for large datasets
        * Input dependency tracking and resolution
        
        **Processing Patterns:**
        * Synchronous processing for immediate results
        * Asynchronous processing for non-blocking operations
        * Parallel processing for computationally intensive tasks
        * Pipeline processing for multi-stage transformations
        
        **Output Generation:**
        * Multiple output data units with structured results
        * Result validation and quality assurance
        * Output routing to downstream components
        * Error handling with detailed diagnostics
        
        **State Management:**
        * Processing state tracking and persistence
        * Progress monitoring and reporting
        * Resource usage tracking and optimization
        * Recovery mechanisms for interrupted processing
    
    **Event-Driven Integration:**
        Steps operate within an event-driven architecture:
        
        * **Trigger Activation**: Steps respond to various trigger types
            - Data availability triggers when input data is ready
            - Timer triggers for scheduled processing
            - Manual triggers for user-initiated operations
            - Conditional triggers based on processing state
        
        * **Data Flow Coordination**: Steps coordinate through data units
            - Input data units provide processing inputs
            - Output data units store and share results
            - Shared data units enable cross-step communication
            - Data persistence for workflow continuity
        
        * **Link Integration**: Steps connect through configurable links
            - Direct links for immediate data transfer
            - Transform links for data format conversion
            - Conditional links for dynamic routing
            - Queue links for asynchronous processing
    
    **Framework Integration:**
        Steps seamlessly integrate with all framework components:
        
        * **Agent Integration**: Steps can embed agents for AI-driven processing
        * **Tool Utilization**: Steps can use tools for specialized operations
        * **Workflow Orchestration**: Steps compose complex multi-stage workflows
        * **Executor Support**: Steps run on various execution backends
        * **Configuration Management**: Complete YAML-driven configuration
        * **Monitoring Integration**: Comprehensive logging and performance tracking
    
    **Step Specializations:**
        The framework supports various step specializations:
        
        * **Step**: Standard processing step with configurable operations
        * **TransformStep**: Specialized for data transformation operations
        * **ParallelStep**: Parallel processing across multiple execution units
        * **BioinformaticsStep**: Computational biology specialized processing
        * **AgentStep**: Steps that integrate AI agents for processing
        * **ConversationalAgentStep**: Multi-turn conversation processing
    
    **Configuration Architecture:**
        Steps follow the framework's configuration-first design:
        
        ```yaml
        # Basic step configuration
        name: "data_processor"
        description: "Processes input data with validation"
        auto_initialize: true
        enable_logging: true
        
        # Input data units
        input_data_units:
          raw_data:
            class: "nanobrain.core.data_unit.DataUnitFile"
            config:
              file_path: "data/input.json"
              encoding: "utf-8"
          parameters:
            class: "nanobrain.core.data_unit.DataUnitMemory"
            config:
              initial_value: {"threshold": 0.5}
        
        # Output data units
        output_data_units:
          processed_data:
            class: "nanobrain.core.data_unit.DataUnitMemory"
            config:
              persistent: true
          results:
            class: "nanobrain.core.data_unit.DataUnitFile"
            config:
              file_path: "data/output.json"
        
        # Agent integration (optional)
        agent:
          class: "nanobrain.core.agent.ConversationalAgent"
          config: "config/processing_agent.yml"
        
        # Tools integration (optional)
        tools:
          validator:
            class: "nanobrain.library.tools.DataValidator"
            config: "config/validator.yml"
        
        # Triggers
        triggers:
          - class: "nanobrain.core.trigger.DataUpdatedTrigger"
            config:
              watch_data_units: ["raw_data", "parameters"]
        
        # Executor configuration
        executor:
          class: "nanobrain.core.executor.LocalExecutor"
          config: "config/local_executor.yml"
        ```
    
    **Usage Patterns:**
        
        **Basic Step Processing:**
        ```python
        from nanobrain.core import Step
        
        # Create step from configuration
        step = Step.from_config('config/data_processor.yml')
        
        # Execute step processing
        results = await step.execute()
        print(f"Processing complete: {results}")
        ```
        
        **Step with Agent Integration:**
        ```python
        # Step automatically creates and uses configured agent
        step = Step.from_config('config/ai_processor.yml')
        
        # Step processes data using embedded agent
        results = await step.execute()
        # Agent provides AI-driven processing within step
        ```
        
        **Multi-Step Workflow:**
        ```python
        # Steps coordinate through data units and triggers
        ingestion = Step.from_config('config/data_ingestion.yml')
        processing = Step.from_config('config/data_processing.yml')
        output = Step.from_config('config/data_output.yml')
        
        # Steps automatically coordinate through configured links
        await ingestion.execute()  # Triggers processing step
        # Processing automatically starts when ingestion completes
        # Output automatically starts when processing completes
        ```
    
    **Data Flow Patterns:**
        
        **Input Processing:**
        * Multiple data sources with different types and formats
        * Data validation and quality checks before processing
        * Dependency resolution and waiting for required inputs
        * Streaming data support for real-time processing
        
        **Processing Execution:**
        * Configurable processing logic through agents or tools
        * Error handling with retry mechanisms and fallbacks
        * Progress monitoring and intermediate result storage
        * Resource management and optimization
        
        **Output Management:**
        * Multiple output destinations with format conversion
        * Result validation and quality assurance
        * Output routing to downstream processing steps
        * Result persistence and retrieval mechanisms
    
    **Performance and Scalability:**
        
        **Execution Optimization:**
        * Asynchronous processing for responsive operations
        * Parallel execution for computationally intensive tasks
        * Streaming processing for large datasets
        * Resource pooling and reuse for efficiency
        
        **Monitoring and Metrics:**
        * Processing time tracking and optimization
        * Resource usage monitoring and alerting
        * Error rate tracking and analysis
        * Throughput measurement and optimization
        
        **Scalability Features:**
        * Horizontal scaling through parallel step instances
        * Vertical scaling through resource allocation
        * Load balancing across execution backends
        * Distributed processing via Parsl integration
    
    **Error Handling and Recovery:**
        Comprehensive error handling with graceful degradation:
        
        * **Input Validation**: Data format and content validation
        * **Processing Errors**: Exception handling with detailed diagnostics
        * **Resource Failures**: Automatic resource recovery and reallocation
        * **Network Issues**: Retry mechanisms with exponential backoff
        * **State Recovery**: Checkpoint and resume capabilities for long-running operations
    
    **Integration Patterns:**
        
        **Agent Integration:**
        * Embed agents for AI-driven processing logic
        * Agent tool calling within step processing
        * Multi-agent coordination for complex operations
        * Agent state management and conversation tracking
        
        **Tool Integration:**
        * Tool selection based on processing requirements
        * Parallel tool execution for complex operations
        * Tool result validation and integration
        * Tool performance monitoring and optimization
        
        **Workflow Integration:**
        * Step composition into complex workflows
        * Dynamic step routing based on processing results
        * Conditional step execution with branching logic
        * Loop and iteration patterns for repetitive processing
    
    **Step Lifecycle:**
        Steps follow a well-defined processing lifecycle:
        
        1. **Configuration Loading**: Parse and validate step configuration
        2. **Dependency Resolution**: Initialize data units, agents, and tools
        3. **Trigger Registration**: Setup event listeners and activation conditions
        4. **Initialization**: Prepare processing resources and state
        5. **Activation**: Respond to triggers and begin processing
        6. **Processing**: Execute configured processing logic
        7. **Output Generation**: Produce results and update output data units
        8. **Cleanup**: Release resources and update processing state
    
    **Advanced Features:**
        
        **Dynamic Configuration:**
        * Runtime configuration updates and reloading
        * Parameter tuning based on processing performance
        * Adaptive resource allocation based on workload
        * Dynamic tool selection based on data characteristics
        
        **State Persistence:**
        * Processing checkpoint creation and restoration
        * State synchronization across distributed execution
        * Result caching for performance optimization
        * Audit trail maintenance for debugging and compliance
        
        **Quality Assurance:**
        * Input data validation and sanitization
        * Processing result verification and validation
        * Performance benchmarking and optimization
        * Error detection and automated correction
    
    Attributes:
        name (str): Step identifier for logging and workflow coordination
        description (str): Human-readable step description and purpose
        input_data_units (Dict[str, DataUnitBase]): Input data sources and containers
        output_data_units (Dict[str, DataUnitBase]): Output data destinations and storage
        triggers (List[TriggerBase]): Event triggers that activate step processing
        agent (Agent, optional): Embedded agent for AI-driven processing
        tools (Dict[str, Tool], optional): Available tools for processing operations
        executor (ExecutorBase): Execution backend for step operations
        processing_state (Dict): Current processing state and progress information
        performance_metrics (Dict): Real-time performance and usage metrics
    
    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like Step or TransformStep. All steps
        must be created using the from_config pattern with proper configuration
        files following the framework's event-driven architecture patterns.
    
    Warning:
        Steps may consume significant computational resources depending on
        processing complexity. Monitor resource usage and implement appropriate
        limits and cleanup mechanisms. Ensure proper error handling for
        long-running or resource-intensive operations.
    
    See Also:
        * :class:`Step`: Standard step implementation with configurable processing
        * :class:`TransformStep`: Specialized step for data transformation
        * :class:`StepConfig`: Step configuration schema and validation
        * :class:`DataUnitBase`: Data container and management system
        * :class:`TriggerBase`: Event trigger system for step activation
        * :class:`LinkBase`: Step connectivity and data flow management
        * :mod:`nanobrain.library.infrastructure.steps`: Specialized step implementations
    """
    
    COMPONENT_TYPE = "base_step"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True,
        'log_data_transfers': True,
        'log_executions': True
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return StepConfig - ONLY method that differs from other components"""
        return StepConfig
    
    # Now inherits unified from_config implementation from FromConfigBase
    
    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """
        Extract BaseStep-specific configuration including resolved objects
        
        ✅ FRAMEWORK COMPLIANCE: Includes resolved objects from class+config patterns
        so steps can access instantiated agents, tools, and other components.
        """
        # Use model_dump() to get all field values, including resolved objects
        all_config = config.model_dump()
        
        # Start with core configuration fields
        component_config = {
            'name': config.name,
            'description': getattr(config, 'description', ''),
            'auto_initialize': getattr(config, 'auto_initialize', True),
            'debug_mode': getattr(config, 'debug_mode', False),
            'enable_logging': getattr(config, 'enable_logging', True),
            'log_data_transfers': getattr(config, 'log_data_transfers', True),
            'log_executions': getattr(config, 'log_executions', True),
        }
        
        # Add all other fields from model_dump() for resolved objects
        # This captures resolved agents, tools, etc. from class+config patterns
        for field_name, field_value in all_config.items():
            if field_name not in component_config:
                component_config[field_name] = field_value
        
        return component_config
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve BaseStep dependencies"""
        executor = kwargs.get('executor')
        if executor is None:
            # Import here to avoid circular imports
            from .executor import ExecutorConfig
            
            # Create default LocalExecutor using proper framework pattern
            # ExecutorConfig doesn't support inline dict config, so use direct instantiation
            try:
                ExecutorConfig._allow_direct_instantiation = True
                default_config = ExecutorConfig(executor_type="local")
            finally:
                ExecutorConfig._allow_direct_instantiation = False
                
            executor = LocalExecutor.from_config(default_config)
            
        return {
            'executor': executor
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize BaseStep with proper sequence"""
        # ✅ FRAMEWORK COMPLIANT - Call parent initialization first to set up enable_logging and other base attributes
        super()._init_from_config(config, component_config, dependencies)
        
        # StepBase-specific initialization (parent already sets self.config, self.name, self.description)
        # Override with step-specific logger that includes debug_mode
        self.nb_logger = get_logger(f"step.{self.name}", debug_mode=component_config['debug_mode'])
        self.nb_logger.info(f"Initializing step: {self.name}", step_name=self.name, config=config.model_dump())
        
        # Executor for running the step
        self.executor = dependencies['executor']
        
        # Data management (legacy step data units)
        self.input_data_units: Dict[str, DataUnitBase] = {}
        self.output_data_unit: Optional[DataUnitBase] = None
        self.links: Dict[str, LinkBase] = {}
        
        # Trigger for activation (legacy)
        self.trigger: Optional[TriggerBase] = None
        
        # Initialize tool registry
        self.tools: Dict[str, Any] = {}
        
        # ✅ PHASE 1: Initialize step-level data unit containers
        self.step_input_data_units = {}
        self.step_output_data_units = {}
        
        # ✅ PHASE 1: Create step-level data units via from_config
        self._create_step_data_units(config)
        
        # ✅ ARCHITECTURAL FIX: Store trigger configurations for later processing in initialize()
        # Triggers will be created AFTER data units exist, ensuring proper resolution
        self.step_trigger_configs = getattr(config, 'triggers', [])
        self.step_triggers = {}  # Will be populated during resolution phase

        # Load tools from external configuration files
        self.step_tools = {}
        
        # ENHANCED: Check if tools were already resolved by ConfigBase._resolve_nested_objects()
        resolved_tools = getattr(config, '_resolved_tools', {})
        if resolved_tools:
            # Use already-instantiated tools from enhanced class+config resolution
            self.step_tools.update(resolved_tools)
            for tool_name in resolved_tools:
                self.nb_logger.info(f"Using resolved tool: {tool_name}")
        else:
            # Fallback: Load tools from external configuration files (legacy)
            tools_config = getattr(config, 'tools', {})
            for tool_name, tool_ref in tools_config.items():
                config_file = tool_ref.get('config_file')
                if config_file:
                    tool_config = self._load_config_file(config_file)
                    from .config.component_factory import create_component
                    tool = create_component(tool_config['class'], tool_config)
                    self.step_tools[tool_name] = tool
                    self.nb_logger.info(f"Loaded step tool: {tool_name}")
        
        # Skip legacy tool loading if tools were already resolved by enhanced system
        # Load tools from legacy configuration only if no enhanced tools were found
        if not resolved_tools and hasattr(config, 'tools') and config.tools:
            self._load_tools_from_config(config.tools)
        
        # State management
        self._is_initialized = False
        self._execution_count = 0
        self._error_count = 0
        self._last_result = None
        
        # Performance tracking
        self._start_time = time.time()
        self._last_activity_time = time.time()
        self._total_processing_time = 0.0
    
    # BaseStep inherits FromConfigBase.__init__ which prevents direct instantiation
    
    def _create_step_data_units(self, config: StepConfig) -> None:
        """
        ✅ FRAMEWORK COMPLIANT: Create step data units via from_config
        Phase 1: Component creation without binding/resolution
        """
        # Create input data units
        input_configs = getattr(config, 'input_data_units', {})
        for unit_name, unit_config in input_configs.items():
            from .data_unit import DataUnitConfig
            
            # ✅ from_config COMPLIANCE: All creation via proper pattern
            if isinstance(unit_config, DataUnitBase):
                # Already instantiated DataUnit object from class+config resolution
                data_unit = unit_config
            elif isinstance(unit_config, dict) and 'class' in unit_config:
                # Dictionary configuration with class field - determine class and call its from_config
                data_unit = self._create_data_unit_from_class_config(unit_config)
            elif isinstance(unit_config, dict):
                # Dictionary configuration without class field - use default
                class_path = unit_config.get('class', 'nanobrain.core.data_unit.DataUnitMemory')
                enhanced_config = unit_config.copy()
                enhanced_config['class'] = class_path
                data_unit = self._create_data_unit_from_class_config(enhanced_config)
            else:
                # DataUnitConfig object - use proper from_config pattern
                data_unit = self._create_data_unit(unit_config)
            
            self.step_input_data_units[unit_name] = data_unit
            self.nb_logger.debug(f"Created step input data unit: {unit_name}")
        
        # Create output data units  
        output_configs = getattr(config, 'output_data_units', {})
        for unit_name, unit_config in output_configs.items():
            from .data_unit import DataUnitConfig
            
            # ✅ from_config COMPLIANCE: All creation via proper pattern
            if isinstance(unit_config, DataUnitBase):
                # Already instantiated DataUnit object from class+config resolution
                data_unit = unit_config
            elif isinstance(unit_config, dict) and 'class' in unit_config:
                # Dictionary configuration with class field - determine class and call its from_config
                data_unit = self._create_data_unit_from_class_config(unit_config)
            elif isinstance(unit_config, dict):
                # Dictionary configuration without class field - use default
                class_path = unit_config.get('class', 'nanobrain.core.data_unit.DataUnitMemory')
                enhanced_config = unit_config.copy()
                enhanced_config['class'] = class_path
                data_unit = self._create_data_unit_from_class_config(enhanced_config)
            else:
                # DataUnitConfig object - use proper from_config pattern
                data_unit = self._create_data_unit(unit_config)
                
            self.step_output_data_units[unit_name] = data_unit
            self.nb_logger.debug(f"Created step output data unit: {unit_name}")
    
    def _create_data_unit_from_class_config(self, unit_config: Dict[str, Any]) -> DataUnitBase:
        """
        ✅ FRAMEWORK COMPLIANT: Create data unit from class+config dictionary
        """
        class_path = unit_config.get('class')
        if not class_path:
            raise ValueError("Data unit configuration missing 'class' field")
        
        # Import the DataUnit class and call its from_config method directly
        module_path, class_name = class_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        data_unit_class = getattr(module, class_name)
        
        # DataUnit classes support inline dict configs
        return data_unit_class.from_config(unit_config)
        
    async def initialize(self) -> None:
        """
        ✅ THREE-PHASE INITIALIZATION: Proper sequence for trigger resolution
        Phase 1: Component creation (already done in _init_from_config)
        Phase 2: Data unit initialization 
        Phase 3: Trigger resolution and binding
        """
        if self._is_initialized:
            self.nb_logger.debug(f"Step {self.name} already initialized")
            return
        
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE, 
            f"{self.name}.initialize"
        ) as context:
            
            # Phase 2: Initialize data units (make them ready for binding)
            await self._initialize_step_data_units()
            
            # Phase 3: Resolve and bind triggers to data units
            await self._resolve_and_bind_step_triggers()
            
            # Initialize other components (executor, legacy components)
            await self._initialize_other_components()
            
            self._is_initialized = True
            context.metadata['input_count'] = len(self.input_data_units)
            context.metadata['step_input_count'] = len(self.step_input_data_units)
            context.metadata['step_output_count'] = len(self.step_output_data_units)
            context.metadata['step_trigger_count'] = len(self.step_triggers)
            context.metadata['has_output'] = self.output_data_unit is not None
            context.metadata['has_trigger'] = self.trigger is not None
            
        self.nb_logger.info(f"Step {self.name} initialized successfully", 
                           input_count=len(self.input_data_units),
                           step_input_count=len(self.step_input_data_units),
                           step_output_count=len(self.step_output_data_units),
                           step_trigger_count=len(self.step_triggers),
                           has_output=self.output_data_unit is not None,
                           has_trigger=self.trigger is not None)
    
    async def _initialize_step_data_units(self) -> None:
        """
        ✅ PHASE 2: Initialize data units to make them ready for trigger binding
        """
        # Initialize step input data units
        for unit_name, data_unit in self.step_input_data_units.items():
            await data_unit.initialize()
            self.nb_logger.debug(f"Initialized step input data unit: {unit_name}")
        
        # Initialize step output data units
        for unit_name, data_unit in self.step_output_data_units.items():
            await data_unit.initialize()
            self.nb_logger.debug(f"Initialized step output data unit: {unit_name}")
        
        self.nb_logger.info(f"✅ Phase 2 Complete: All step data units initialized")
    
    async def _resolve_and_bind_step_triggers(self) -> None:
        """
        ✅ PHASE 3: Resolve trigger data unit references and bind to actual objects
        
        ARCHITECTURAL COMPLIANCE:
        - Only resolves within step scope (no cross-step references)
        - Uses actual data unit objects created in previous phases
        - Maintains from_config pattern for trigger creation
        """
        step_context = self._create_step_resolution_context()
        
        for trigger_config in self.step_trigger_configs:
            # ✅ FRAMEWORK COMPLIANT: Create trigger via from_config with step context
            trigger_instance = await self._create_and_resolve_step_trigger(
                trigger_config, step_context
            )
            
            if trigger_instance:
                # Bind trigger to step execution
                trigger_instance.bind_action(self._execute_on_trigger)
                
                # Store resolved trigger
                trigger_id = getattr(trigger_instance, 'trigger_id', f'trigger_{len(self.step_triggers)}')
                self.step_triggers[trigger_id] = trigger_instance
                
                # Start monitoring
                await trigger_instance.start_monitoring()
                
                self.nb_logger.info(f"✅ Resolved and bound step trigger: {trigger_id}")
        
        self.nb_logger.info(f"✅ Phase 3 Complete: All triggers resolved and monitoring")
    
    async def _initialize_other_components(self) -> None:
        """
        ✅ LEGACY SUPPORT: Initialize other components (executor, legacy data units, triggers)
        """
        # Initialize executor
        self.nb_logger.debug(f"Initializing executor for step {self.name}")
        await self.executor.initialize()
        
        # Initialize legacy input data units (if any)
        self.nb_logger.debug(f"Initializing {len(self.config.input_configs)} legacy input data units")
        for input_id, input_config in self.config.input_configs.items():
            data_unit = self._create_data_unit(input_config)
            await data_unit.initialize()
            self.input_data_units[input_id] = data_unit
            self.nb_logger.debug(f"Initialized legacy input data unit: {input_id}")
        
        # Initialize legacy output data unit (if any)
        if self.config.output_config:
            self.nb_logger.debug(f"Initializing legacy output data unit")
            self.output_data_unit = self._create_data_unit(self.config.output_config)
            await self.output_data_unit.initialize()
        
        # Initialize legacy trigger (if any)
        if self.config.trigger_config:
            self.nb_logger.debug(f"Initializing legacy trigger")
            self.trigger = self._create_trigger(self.config.trigger_config)
            
            # Set up trigger callback
            await self.trigger.add_callback(self._on_trigger_activated)
            
            # Start monitoring
            await self.trigger.start_monitoring()
    
    def _create_step_resolution_context(self) -> Dict[str, Any]:
        """
        ✅ STEP-SCOPE ONLY: Create resolution context with step-local data units
        
        ARCHITECTURAL COMPLIANCE:
        - Only includes data units within this step scope
        - No workflow-level or cross-step data units
        - Ensures trigger isolation within step boundaries
        """
        return {
            'step_input_data_units': self.step_input_data_units,
            'step_output_data_units': self.step_output_data_units,
            'step_name': self.name,
            'step_scope_only': True  # Enforce step isolation
        }
    
    async def _create_and_resolve_step_trigger(self, trigger_config: Any, 
                                             step_context: Dict[str, Any]) -> Optional[TriggerBase]:
        """
        ✅ FRAMEWORK COMPLIANT: Create trigger via from_config with step-scope resolution
        """
        try:
            # Phase 3A: Create trigger instance via from_config
            if hasattr(trigger_config, '__class__') and hasattr(trigger_config, 'bind_action'):
                # Already instantiated trigger from ConfigBase resolution
                trigger_instance = trigger_config
            else:
                # Create trigger via from_config pattern
                trigger_class = self._get_trigger_class(trigger_config)
                trigger_instance = trigger_class.from_config(
                    trigger_config, 
                    step_context=step_context  # Pass step-local context
                )
            
            # Phase 3B: Resolve data unit references within step scope
            if hasattr(trigger_instance, 'data_unit') and isinstance(trigger_instance.data_unit, str):
                resolved_data_unit = self._resolve_step_data_unit_reference(
                    trigger_instance.data_unit, step_context
                )
                
                if resolved_data_unit:
                    trigger_instance.data_unit = resolved_data_unit
                    self.nb_logger.debug(f"✅ Resolved trigger data unit: {trigger_instance.data_unit.name}")
                else:
                    raise ValueError(f"❌ Data unit '{trigger_instance.data_unit}' not found in step scope")
            
            return trigger_instance
            
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to create/resolve step trigger: {e}")
            return None
    
    def _resolve_step_data_unit_reference(self, data_unit_ref: str, 
                                        step_context: Dict[str, Any]) -> Optional[DataUnitBase]:
        """
        ✅ STEP-SCOPE ONLY: Resolve data unit reference within step boundaries
        
        ARCHITECTURAL COMPLIANCE:
        - Only searches within step input/output data units
        - No cross-step or workflow-level resolution
        - Returns None if not found in step scope (proper behavior)
        """
        # Check step input data units
        step_input_units = step_context.get('step_input_data_units', {})
        if data_unit_ref in step_input_units:
            return step_input_units[data_unit_ref]
        
        # Check step output data units
        step_output_units = step_context.get('step_output_data_units', {})
        if data_unit_ref in step_output_units:
            return step_output_units[data_unit_ref]
        
        # Not found in step scope - this is expected behavior for cross-scope references
        available_units = list(step_input_units.keys()) + list(step_output_units.keys())
        self.nb_logger.warning(f"Data unit '{data_unit_ref}' not found in step scope. "
                              f"Available: {available_units}")
        return None
    
    def _get_trigger_class(self, trigger_config: Any):
        """
        ✅ FRAMEWORK COMPLIANT: Get trigger class for from_config creation
        """
        if isinstance(trigger_config, dict):
            class_path = trigger_config.get('class', 'nanobrain.core.trigger.DataUnitChangeTrigger')
        elif hasattr(trigger_config, 'class_path'):
            class_path = trigger_config.class_path
        else:
            class_path = 'nanobrain.core.trigger.DataUnitChangeTrigger'
        
        # Import the trigger class
        module_path, class_name = class_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    async def shutdown(self) -> None:
        """Shutdown the step and cleanup resources."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE, 
            f"{self.name}.shutdown"
        ) as context:
            # Log final statistics
            uptime_seconds = time.time() - self._start_time
            self.nb_logger.info(f"Step {self.name} shutting down", 
                               uptime_seconds=uptime_seconds,
                               execution_count=self._execution_count,
                               error_count=self._error_count,
                               total_processing_time=self._total_processing_time)
            
            # Shutdown trigger
            if self.trigger:
                await self.trigger.stop_monitoring()
            
            # EVENT-DRIVEN ARCHITECTURE: Shutdown step-level triggers
            for trigger_id, trigger in self.step_triggers.items():
                await trigger.stop_monitoring()
                self.nb_logger.debug(f"Stopped step trigger: {trigger_id}")
            
            # Shutdown data units
            for data_unit in self.input_data_units.values():
                await data_unit.shutdown()
            
            if self.output_data_unit:
                await self.output_data_unit.shutdown()
            
            # Shutdown executor
            await self.executor.shutdown()
            
            self._is_initialized = False
            context.metadata['final_stats'] = {
                'uptime_seconds': uptime_seconds,
                'execution_count': self._execution_count,
                'error_count': self._error_count,
                'total_processing_time': self._total_processing_time
            }
    
    def _create_data_unit(self, config: DataUnitConfig) -> DataUnitBase:
        """Create a data unit from configuration using proper from_config pattern."""
        # Import here to avoid circular imports
        class_path = config.class_path
        
        # Import the DataUnit class and call its from_config method directly
        module_path, class_name = class_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        data_unit_class = getattr(module, class_name)
        
        # Use the DataUnit class's from_config method
        return data_unit_class.from_config(config)
    
    def _create_trigger(self, config: TriggerConfig) -> TriggerBase:
        """Create a trigger from configuration using proper from_config pattern."""
        # Import here to avoid circular imports
        trigger_type = config.trigger_type
        
        # Import the appropriate trigger class and call its from_config method
        if trigger_type == 'data_updated':
            from .trigger import DataUpdatedTrigger
            return DataUpdatedTrigger.from_config(config)
        elif trigger_type == 'all_data_received':
            from .trigger import AllDataReceivedTrigger
            return AllDataReceivedTrigger.from_config(config)
        elif trigger_type == 'timer':
            from .trigger import TimerTrigger
            return TimerTrigger.from_config(config)
        elif trigger_type == 'manual':
            from .trigger import ManualTrigger
            return ManualTrigger.from_config(config)
        else:
            # Default to manual trigger
            from .trigger import ManualTrigger
            return ManualTrigger.from_config(config)
    
    async def _execute_on_trigger(self, trigger_event: Dict[str, Any]) -> None:
        """Execute step when triggered by data unit change (EVENT-DRIVEN EXECUTION)"""
        try:
            self.nb_logger.info(f"🔥 Step {self.name} triggered by {trigger_event['trigger_id']}")
            
            # Get input data from triggered data unit
            input_data = {}
            for unit_name, data_unit in self.step_input_data_units.items():
                input_data[unit_name] = await data_unit.get()
            
            # Execute step business logic
            result = await self.process(input_data)
            
            # Update output data units
            for unit_name, data_unit in self.step_output_data_units.items():
                if unit_name in result:
                    await data_unit.set(result[unit_name])
                    self.nb_logger.info(f"📤 Updated output data unit: {unit_name}")
            
            # Update execution statistics
            self._execution_count += 1
            self._last_result = result
            self._last_activity_time = time.time()
            
        except Exception as e:
            self._error_count += 1
            self.nb_logger.error(f"❌ Step execution failed: {e}")
            raise
    
    async def _on_trigger_activated(self, trigger_data: Dict[str, Any]) -> None:
        """Handle trigger activation."""
        async with self.nb_logger.async_execution_context(
            OperationType.TRIGGER_ACTIVATE, 
            f"{self.name}.trigger_activated",
            trigger_type=type(self.trigger).__name__ if self.trigger else "unknown"
        ) as context:
            self.nb_logger.log_trigger_activation(
                trigger_name=f"{self.name}.trigger",
                trigger_type=type(self.trigger).__name__ if self.trigger else "unknown",
                conditions=trigger_data,
                activated=True
            )
            
            context.metadata['trigger_data'] = trigger_data
            
            # Execute the step
            await self.execute()
    
    def register_input_data_unit(self, input_id: str, data_unit: DataUnitBase) -> None:
        """Register an input data unit."""
        self.input_data_units[input_id] = data_unit
        self.nb_logger.info(f"Registered input data unit: {input_id}", 
                           input_id=input_id, 
                           data_unit_type=type(data_unit).__name__)
        
        # If we have a trigger, register the data unit with it
        if self.trigger and hasattr(self.trigger, 'register_data_unit'):
            self.trigger.register_data_unit(input_id, data_unit)
    
    def register_output_data_unit(self, data_unit: DataUnitBase) -> None:
        """Register the output data unit."""
        self.output_data_unit = data_unit
        self.nb_logger.info(f"Registered output data unit", 
                           data_unit_type=type(data_unit).__name__)
    
    def add_link(self, link_id: str, link: LinkBase) -> None:
        """Add a link to another step."""
        self.links[link_id] = link
        self.nb_logger.info(f"Added link: {link_id}", 
                           link_id=link_id, 
                           link_type=type(link).__name__)
    
    async def execute(self, **kwargs) -> Any:
        """Execute the step using the configured executor."""
        if not self._is_initialized:
            await self.initialize()
        
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE, 
            f"{self.name}.execute",
            kwargs_keys=list(kwargs.keys())
        ) as context:
            try:
                # Collect input data
                input_data = {}
                for input_id, data_unit in self.input_data_units.items():
                    data = await data_unit.read()
                    input_data[input_id] = data
                    
                    if self.config.log_data_transfers:
                        self.nb_logger.log_data_transfer(
                            source=f"{input_id}.data_unit",
                            destination=f"{self.name}.input",
                            data_type=type(data).__name__,
                            size_bytes=len(str(data)) if data else 0
                        )
                
                context.metadata['input_data_keys'] = list(input_data.keys())
                context.metadata['input_data_sizes'] = {
                    k: len(str(v)) if v else 0 for k, v in input_data.items()
                }
                
                self.nb_logger.debug(f"Collected input data for step {self.name}", 
                                   input_keys=list(input_data.keys()))
                
                # Process using executor
                start_time = time.time()
                # Create a wrapper function that properly handles the input_data argument
                async def execute_wrapper():
                    return await self._execute_process(input_data, **kwargs)
                result = await self.executor.execute(execute_wrapper)
                processing_time = time.time() - start_time
                
                self._execution_count += 1
                self._last_activity_time = time.time()
                self._total_processing_time += processing_time
                self._last_result = result
                
                # Store result in output data unit
                if self.output_data_unit and result is not None:
                    await self.output_data_unit.write(result)
                    
                    if self.config.log_data_transfers:
                        self.nb_logger.log_data_transfer(
                            source=f"{self.name}.output",
                            destination=f"{self.name}.output_data_unit",
                            data_type=type(result).__name__,
                            size_bytes=len(str(result)) if result else 0
                        )
                
                # Propagate data through links
                if self.links and result is not None:
                    await self._propagate_through_links(result)
                
                # Log execution
                if self.config.log_executions:
                    self.nb_logger.log_step_execution(
                        step_name=self.name,
                        inputs=input_data,
                        outputs=result,
                        duration_ms=processing_time * 1000,
                        success=True
                    )
                
                context.metadata['result_type'] = type(result).__name__ if result else None
                context.metadata['processing_time_ms'] = processing_time * 1000
                context.metadata['execution_count'] = self._execution_count
                
                self.nb_logger.debug(f"Step {self.name} executed successfully", 
                                   execution_count=self._execution_count,
                                   processing_time_ms=processing_time * 1000,
                                   result_type=type(result).__name__ if result else None)
                
                return result
                
            except Exception as e:
                self._error_count += 1
                
                # Log failed execution
                if self.config.log_executions:
                    self.nb_logger.log_step_execution(
                        step_name=self.name,
                        inputs=input_data if 'input_data' in locals() else {},
                        success=False,
                        error=str(e)
                    )
                
                context.metadata['error_count'] = self._error_count
                self.nb_logger.error(f"Step {self.name} execution failed: {e}", 
                                   error_type=type(e).__name__,
                                   error_count=self._error_count)
                raise
    
    async def _execute_process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Wrapper for process method to be executed by executor."""
        return await self.process(input_data, **kwargs)
    
    async def _propagate_through_links(self, data: Any) -> None:
        """Propagate data through all links."""
        for link_id, link in self.links.items():
            try:
                async with self.nb_logger.async_execution_context(
                    OperationType.DATA_TRANSFER, 
                    f"{self.name}.link.{link_id}",
                    link_type=type(link).__name__
                ) as context:
                    await link.transfer(data)
                    
                    if self.config.log_data_transfers:
                        self.nb_logger.log_data_transfer(
                            source=f"{self.name}",
                            destination=f"link.{link_id}",
                            data_type=type(data).__name__,
                            size_bytes=len(str(data)) if data else 0
                        )
                    
                    context.metadata['data_type'] = type(data).__name__
                    
            except Exception as e:
                self.nb_logger.error(f"Failed to propagate data through link {link_id}: {e}", 
                                   link_id=link_id,
                                   error_type=type(e).__name__)
    
    def _load_tools_from_config(self, tools_config: Dict[str, Dict[str, Any]]) -> None:
        """Load tools configuration from step-level YAML configuration"""
        logger = get_logger(f"step.{self.name}.tools")
        
        for tool_name, tool_config in tools_config.items():
            try:
                tool_instance = self._create_tool_from_config(tool_name, tool_config)
                self._register_tool(tool_instance, tool_name)
                logger.info(f"Loaded tool: {tool_name}")
            except Exception as e:
                logger.error(f"Failed to load tool {tool_name}: {e}")
                raise

    def _create_tool_from_config(self, tool_name: str, tool_config: Dict[str, Any]) -> Any:
        """Create tool instance from step-level YAML configuration"""
        from .config.component_factory import create_component
        
        # Start with a copy of the local tool config
        merged_config = tool_config.copy()
        
        # Get tool class from config
        tool_class = tool_config.get('class')
        
        # If no class specified or config_file is present, load from referenced config file
        config_file = tool_config.get('config_file')
        if config_file:
            try:
                tool_config_data = self._load_config_file(config_file)
                # Merge external config with local config, prioritizing local config
                external_config = tool_config_data.copy()
                # Remove the tool_config section temporarily if it exists in external config
                external_tool_config = external_config.pop('tool_config', {})
                # Merge external config first, then local config overrides it
                merged_config = {**external_config, **merged_config, **external_tool_config}
                
                # Get class from external config if not specified locally
                if not tool_class:
                    tool_class = tool_config_data.get('class')
            except Exception as e:
                logger = get_logger(f"step.{self.name}.tools")
                logger.warning(f"Failed to load external config file {config_file}: {e}")
        
        if not tool_class:
            raise ValueError(f"Tool '{tool_name}' configuration missing 'class' field")
        
        # Update the class in merged config
        merged_config['class'] = tool_class
        
        # Merge tool_config section if present in local config
        if 'tool_config' in tool_config:
            tool_config_section = merged_config.pop('tool_config', {})
            merged_config = {**merged_config, **tool_config_section, **tool_config['tool_config']}
        
        # Ensure name field is present (required by many components)
        # For external tools, use 'tool_name' instead of 'name'
        if 'name' not in merged_config and 'tool_name' not in merged_config and tool_name:
            # Check if this is an external tool by checking the class path
            if 'bioinformatics.bv_brc_tool' in tool_class or 'external_tool' in tool_class.lower():
                merged_config['tool_name'] = tool_name
            else:
                merged_config['name'] = tool_name
        
        # Create tool using pure component factory
        return create_component(tool_class, merged_config)

    def _load_config_file(self, config_file_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        import yaml
        from pathlib import Path
        
        config_path = Path(config_file_path)
        if not config_path.is_absolute():
            # Make relative to step's workflow directory if available
            workflow_dir = getattr(self.config, 'workflow_directory', '.')
            config_path = Path(workflow_dir) / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _register_tool(self, tool: Any, name: str) -> None:
        """Register tool with step for easy access"""
        self.tools[name] = tool

    def get_tool(self, name: str) -> Optional[Any]:
        """Get tool by name for step usage"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process input data and return result.
        
        Args:
            input_data: Dictionary of input data from registered data units
            **kwargs: Additional parameters
            
        Returns:
            Processing result
        """
        pass
    
    def get_result(self) -> Any:
        """Get the last execution result."""
        return self._last_result
    
    @property
    def is_initialized(self) -> bool:
        """Check if the step is initialized."""
        return self._is_initialized
    
    @property
    def execution_count(self) -> int:
        """Get the number of executions."""
        return self._execution_count
    
    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return self._error_count
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this step."""
        uptime_seconds = time.time() - self._start_time
        idle_seconds = time.time() - self._last_activity_time
        avg_processing_time = self._total_processing_time / max(self._execution_count, 1)
        
        return {
            "uptime_seconds": uptime_seconds,
            "idle_seconds": idle_seconds,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "success_rate": (self._execution_count - self._error_count) / max(self._execution_count, 1),
            "total_processing_time": self._total_processing_time,
            "avg_processing_time": avg_processing_time,
            "input_data_units": len(self.input_data_units),
            "has_output_data_unit": self.output_data_unit is not None,
            "links_count": len(self.links),
            "has_trigger": self.trigger is not None
        }
    
    async def set_input(self, data: Any, input_id: str = "input_0") -> None:
        """Convenience method to set input data."""
        if input_id not in self.input_data_units:
            # Create a default input data unit if it doesn't exist
            from .data_unit import DataUnitConfig, DataUnitType
            from .config.component_factory import create_component
            config = DataUnitConfig(data_type=DataUnitType.MEMORY, name=input_id)
            data_unit = create_component(
                "nanobrain.core.data_unit.DataUnitMemory", 
                config, 
                name=input_id
            )
            await data_unit.initialize()
            self.input_data_units[input_id] = data_unit
        
        await self.input_data_units[input_id].set(data)
    
    async def get_output(self) -> Any:
        """Convenience method to get output data."""
        if not self.output_data_unit:
            return self._last_result
        return await self.output_data_unit.get()
    
    @property
    def is_running(self) -> bool:
        """Check if the step is currently running."""
        # For now, just return False as we don't track running state
        # This could be enhanced to track actual execution state
        return False


class Step(BaseStep):
    """Step (formerly SimpleStep) with mandatory from_config implementation."""
    
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'processing_mode': 'combine',
        'add_metadata': True,
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }
    
    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract Step configuration"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'processing_mode': getattr(config, 'processing_mode', 'combine'),
            'add_metadata': getattr(config, 'add_metadata', True),
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Step with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.processing_mode = component_config['processing_mode']
        self.add_metadata = component_config['add_metadata']
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Process input data with simple transformation."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE, 
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="SimpleStep"
        ) as context:
            # Simple processing: combine all inputs
            if not input_data:
                result = "No input data"
            elif len(input_data) == 1:
                result = list(input_data.values())[0]
            else:
                result = {
                    "processed_at": time.time(),
                    "inputs": input_data,
                    "step_name": self.name
                }
            
            self.nb_logger.debug(f"Simple step {self.name} processed data", 
                               input_count=len(input_data),
                               result_type=type(result).__name__)
            
            context.metadata['result_type'] = type(result).__name__
            context.metadata['input_count'] = len(input_data)
            
            return result


class TransformStep(BaseStep):
    """Step that applies a transformation function to input data."""
    
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': '',
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }
    
    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract TransformStep configuration"""
        base_config = super().extract_component_config(config)
        return base_config
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve TransformStep dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        
        # Get transform function from kwargs
        transform_func = kwargs.get('transform_func')
        
        return {
            **base_deps,
            'transform_func': transform_func
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize TransformStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.transform_func = dependencies.get('transform_func') or self._default_transform
        
        self.nb_logger.debug(f"Transform step {self.name} initialized", 
                           has_custom_transform=dependencies.get('transform_func') is not None)
    
    def _default_transform(self, data: Any) -> Any:
        """Default transformation: convert to string and add metadata."""
        return {
            "original": data,
            "transformed": str(data).upper(),
            "timestamp": time.time(),
            "step": self.name
        }
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """Process input data using the transformation function."""
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE, 
            f"{self.name}.process",
            input_keys=list(input_data.keys()),
            step_type="TransformStep"
        ) as context:
            if not input_data:
                result = None
            elif len(input_data) == 1:
                # Single input: apply transform directly
                input_value = list(input_data.values())[0]
                result = self.transform_func(input_value)
            else:
                # Multiple inputs: apply transform to each
                result = {}
                for key, value in input_data.items():
                    result[key] = self.transform_func(value)
            
            self.nb_logger.debug(f"Transform step {self.name} processed data", 
                               input_count=len(input_data),
                               result_type=type(result).__name__)
            
            context.metadata['result_type'] = type(result).__name__
            context.metadata['input_count'] = len(input_data)
            context.metadata['transform_func'] = self.transform_func.__name__
            
            return result


"""
FRAMEWORK CHANGE: Pure Configuration-Driven Step Loading

Steps are now loaded EXCLUSIVELY through configuration files using the
from_config pattern. The create_step factory has been eliminated.

✅ CORRECT USAGE:
   # In workflow configuration (YAML):
   steps:
     - step_id: my_step
       config_file: "config/MyStep/MyStep.yml"
   
   # In workflow code:
   step_config = manager.load_config(config_path, StepConfig)
   step_class = import_class(step_config.class)
   step = step_class.from_config(config_path, executor=executor)

❌ DEPRECATED USAGE:
   step = create_step('module.StepClass', step_config, executor=executor)

REASON: Enforces pure configuration-driven architecture without
        programmatic component creation.
"""


# Legacy factory functions removed as per Phase 3: Legacy Component Removal
# These functions are no longer needed as all step creation is now handled
# via class-specific from_config methods leveraging ConfigBase._resolve_nested_objects()
#
# ✅ FRAMEWORK COMPLIANCE:
# - All component creation uses class-specific from_config methods
# - ConfigBase._resolve_nested_objects() handles automatic instantiation
# - No factory functions or redundant creation logic
# - Pure configuration-driven component creation 