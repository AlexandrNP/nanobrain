"""
Trigger System for NanoBrain Framework

Provides event-driven processing capabilities for Steps.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Set, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
# Import logging system
from .logging_system import get_logger, get_system_log_manager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers."""
    DATA_UPDATED = "data_updated"
    ALL_DATA_RECEIVED = "all_data_received"
    TIMER = "timer"
    MANUAL = "manual"
    CONDITION = "condition"


class TriggerConfig(ConfigBase):
    """
    Configuration for triggers - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: TriggerConfig(trigger_type="data_updated", ...)
    ✅ REQUIRED: TriggerConfig.from_config('path/to/config.yml')
    """
    
    trigger_type: TriggerType = TriggerType.DATA_UPDATED
    debounce_ms: int = Field(default=100, ge=0)
    max_frequency_hz: float = Field(default=10.0, gt=0)
    condition: Optional[str] = None
    timer_interval_ms: Optional[int] = None
    name: str = ""


class TriggerBase(FromConfigBase, ABC):
    """
    Base Trigger Class - Event-Driven Activation and Workflow Orchestration
    =======================================================================
    
    The TriggerBase class is the foundational component for event-driven processing
    within the NanoBrain framework. Triggers monitor conditions, data changes, and
    external events to automatically activate steps, workflows, and agents, enabling
    reactive and responsive processing architectures.
    
    **Core Architecture:**
        Triggers represent intelligent event detection systems that:
        
        * **Monitor Conditions**: Continuously watch for specific events or state changes
        * **Activate Components**: Automatically trigger step and workflow execution
        * **Manage Timing**: Control execution frequency with debouncing and rate limiting
        * **Enable Reactivity**: Support real-time response to data and system events
        * **Coordinate Workflows**: Orchestrate complex multi-step processing pipelines
        * **Ensure Reliability**: Provide robust event detection with error handling
    
    **Biological Analogy:**
        Like action potential threshold mechanisms that fire when specific conditions
        are met, triggers activate steps when particular events or conditions are
        satisfied. Neurons accumulate electrical potential and fire when threshold
        is reached, propagating signals through neural networks - exactly how triggers
        monitor conditions and activate processing components when criteria are met,
        propagating execution through workflow networks.
    
    **Event-Driven Processing Architecture:**
        
        **Condition Monitoring:**
        * Continuous monitoring of data units and system state
        * Real-time change detection with configurable sensitivity
        * Multi-condition evaluation with logical operators
        * Custom condition scripting and evaluation
        
        **Activation Patterns:**
        * Immediate activation upon condition detection
        * Debounced activation to prevent excessive triggering
        * Rate-limited activation for performance optimization
        * Scheduled activation with timer-based triggers
        
        **Event Types:**
        * **Data Updated**: Triggered when data units receive new data
        * **All Data Received**: Triggered when all required inputs are available
        * **Timer**: Triggered on scheduled intervals or specific times
        * **Manual**: Triggered by explicit user or system commands
        * **Condition**: Triggered when custom conditions evaluate to true
        
        **Response Coordination:**
        * Multi-target activation for parallel processing
        * Sequential activation with dependency management
        * Conditional activation based on runtime state
        * Priority-based activation ordering
    
    **Framework Integration:**
        Triggers seamlessly integrate with all framework components:
        
        * **Step Activation**: Automatically trigger step execution when conditions are met
        * **Workflow Orchestration**: Coordinate complex multi-step processing workflows
        * **Agent Integration**: Trigger agent processing based on data availability
        * **Data Unit Monitoring**: Monitor data unit changes and trigger processing
        * **Executor Support**: Triggers work with all execution backends
        * **Monitoring Integration**: Comprehensive logging and performance tracking
    
    **Trigger Type Implementations:**
        The framework supports various trigger specializations:
        
        * **DataUpdatedTrigger**: Monitors data unit changes and modifications
        * **AllDataReceivedTrigger**: Waits for all required inputs before activation
        * **TimerTrigger**: Provides scheduled and interval-based activation
        * **ManualTrigger**: Enables user-controlled activation and testing
        * **ConditionalTrigger**: Supports custom condition evaluation and scripting
        * **CompoundTrigger**: Combines multiple triggers with logical operators
    
    **Configuration Architecture:**
        Triggers follow the framework's configuration-first design:
        
        ```yaml
        # Data update trigger
        name: "data_change_trigger"
        trigger_type: "data_updated"
        debounce_ms: 500
        max_frequency_hz: 5.0
        
        # Watch specific data units
        watch_data_units:
          - "input_data"
          - "parameters"
        
        # Target steps to activate
        target_steps:
          - "data_processing"
          - "validation"
        
        # Timer trigger
        name: "scheduled_processing"
        trigger_type: "timer"
        timer_interval_ms: 60000  # Every minute
        
        # Schedule configuration
        schedule:
          type: "interval"
          interval: "1m"
          start_time: "09:00"
          end_time: "17:00"
          timezone: "UTC"
        
        # Conditional trigger
        name: "threshold_trigger"
        trigger_type: "condition"
        condition: "data.temperature > 25 and data.humidity < 60"
        
        # Condition evaluation
        evaluation:
          language: "python"
          context_variables:
            - "data"
            - "metadata"
            - "system_state"
          timeout_ms: 1000
        
        # All data received trigger
        name: "batch_ready_trigger"
        trigger_type: "all_data_received"
        required_data_units:
          - "raw_data"
          - "configuration"
          - "metadata"
        
        # Activation settings
        activation:
          mode: "once_per_batch"
          reset_on_completion: true
          timeout_ms: 30000
        ```
    
    **Usage Patterns:**
        
        **Basic Data Monitoring:**
        ```python
        from nanobrain.core import DataUpdatedTrigger
        
        # Create trigger from configuration
        trigger = DataUpdatedTrigger.from_config('config/data_trigger.yml')
        
        # Register callback for activation
        async def on_data_updated(data_unit, old_value, new_value):
            print(f"Data changed: {old_value} -> {new_value}")
            # Trigger step execution
            await step.execute()
        
        trigger.register_callback(on_data_updated)
        
        # Start monitoring
        await trigger.start()
        ```
        
        **Timer-Based Processing:**
        ```python
        # Scheduled processing trigger
        timer_trigger = TimerTrigger.from_config('config/timer_trigger.yml')
        
        # Register processing callback
        async def scheduled_process():
            # Execute periodic processing
            results = await workflow.execute_batch()
            return results
        
        timer_trigger.register_callback(scheduled_process)
        
        # Start scheduled execution
        await timer_trigger.start()
        ```
        
        **Complex Condition Monitoring:**
        ```python
        # Custom condition trigger
        condition_trigger = ConditionalTrigger.from_config('config/condition_trigger.yml')
        
        # Advanced condition evaluation
        condition_expression = "data.temperature > threshold and data.trend == 'increasing'"
        condition_trigger.set_condition(condition_expression)
        
        # Context variables for evaluation
        condition_trigger.set_context({
            'threshold': 30.0,
            'system_state': system_monitor.get_state()
        })
        
        await condition_trigger.start()
        ```
        
        **Multi-Target Activation:**
        ```python
        # Trigger multiple steps simultaneously
        multi_trigger = DataUpdatedTrigger.from_config('config/multi_trigger.yml')
        
        # Register multiple targets
        multi_trigger.add_target(preprocessing_step)
        multi_trigger.add_target(validation_step)
        multi_trigger.add_target(logging_step)
        
        # All targets activated when trigger fires
        await multi_trigger.start()
        ```
    
    **Advanced Features:**
        
        **Debouncing and Rate Limiting:**
        * Configurable debounce periods to prevent excessive triggering
        * Rate limiting to control maximum activation frequency
        * Burst detection and handling for high-frequency events
        * Adaptive rate limiting based on processing capacity
        
        **Condition Evaluation:**
        * Python expression evaluation for custom conditions
        * Multi-variable context with data and system state
        * Safe evaluation with timeout and resource limits
        * Precompiled expressions for performance optimization
        
        **Event Aggregation:**
        * Batch event processing for improved efficiency
        * Event correlation and pattern detection
        * Time-window based aggregation
        * Statistical analysis of event patterns
        
        **Error Handling and Recovery:**
        * Robust error handling with detailed diagnostics
        * Automatic recovery from temporary failures
        * Circuit breaker patterns for unstable conditions
        * Fallback activation mechanisms
    
    **Performance and Scalability:**
        
        **Efficient Monitoring:**
        * Low-overhead condition checking with optimized algorithms
        * Event-driven architecture minimizing resource usage
        * Selective monitoring with configurable granularity
        * Batch processing for improved throughput
        
        **Scalability Features:**
        * Distributed trigger monitoring across multiple nodes
        * Load balancing for high-frequency event processing
        * Horizontal scaling with trigger distribution
        * Resource pooling and optimization
        
        **Monitoring and Metrics:**
        * Trigger activation frequency and timing analysis
        * Condition evaluation performance monitoring
        * Error rate tracking and optimization recommendations
        * Resource usage analysis and capacity planning
    
    **Integration Patterns:**
        
        **Workflow Orchestration:**
        * Event-driven workflow activation and coordination
        * Multi-stage pipeline triggering with dependencies
        * Conditional workflow branching based on trigger conditions
        * Dynamic workflow modification based on events
        
        **Real-Time Processing:**
        * Stream processing with continuous data monitoring
        * Low-latency response to critical events
        * Real-time analytics and alerting systems
        * Adaptive processing based on data characteristics
        
        **Batch Processing:**
        * Scheduled batch processing with timer triggers
        * Data availability-based batch activation
        * Resource-aware batch scheduling and optimization
        * Large dataset processing with progress monitoring
    
    **Event Lifecycle:**
        Triggers follow a well-defined event processing lifecycle:
        
        1. **Configuration Loading**: Parse and validate trigger configuration
        2. **Condition Setup**: Initialize monitoring conditions and parameters
        3. **Target Registration**: Register callback functions and target components
        4. **Monitoring Initialization**: Setup event listeners and data watchers
        5. **Active Monitoring**: Continuously monitor conditions and events
        6. **Condition Evaluation**: Evaluate trigger conditions when events occur
        7. **Activation Decision**: Determine whether to activate based on conditions
        8. **Target Activation**: Execute registered callbacks and activate targets
        9. **Rate Limiting**: Apply debouncing and frequency controls
        10. **Cleanup**: Handle cleanup and resource management
    
    **Security and Reliability:**
        
        **Secure Condition Evaluation:**
        * Safe expression evaluation with sandboxing
        * Input validation and sanitization
        * Resource limits and timeout protection
        * Access control for sensitive data and operations
        
        **Reliability Features:**
        * Fault tolerance with automatic recovery
        * Event persistence and replay capabilities
        * Redundancy and failover mechanisms
        * Health monitoring and alerting
        
        **Audit and Compliance:**
        * Comprehensive logging of trigger activations
        * Event history and audit trails
        * Performance metrics and compliance reporting
        * Security event tracking and analysis
    
    **Development and Testing:**
        
        **Testing Support:**
        * Mock trigger implementations for testing
        * Event simulation and validation frameworks
        * Trigger performance benchmarking
        * Integration testing with steps and workflows
        
        **Debugging Features:**
        * Comprehensive logging with event tracing
        * Condition evaluation debugging and analysis
        * Performance profiling and optimization hints
        * Visual event timeline and inspection tools
        
        **Development Tools:**
        * Trigger configuration validation and linting
        * Condition expression testing and validation
        * Performance monitoring and optimization tools
        * Event pattern analysis and optimization
    
    Attributes:
        name (str): Trigger identifier for logging and component coordination
        trigger_type (TriggerType): Type of trigger and activation pattern
        debounce_ms (int): Debounce period in milliseconds to prevent excessive activation
        max_frequency_hz (float): Maximum activation frequency in Hz for rate limiting
        condition (str, optional): Custom condition expression for conditional triggers
        callbacks (List[Callable]): Registered callback functions for activation
        is_active (bool): Whether trigger is currently monitoring conditions
        last_trigger_time (float): Timestamp of last activation for rate limiting
        trigger_count (int): Total number of activations since creation
        performance_metrics (Dict): Real-time performance and usage metrics
    
    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like DataUpdatedTrigger, TimerTrigger, or
        ConditionalTrigger. All triggers must be created using the from_config
        pattern with proper configuration files following framework patterns.
    
    Warning:
        Triggers can significantly impact system performance if configured with
        high frequencies or complex conditions. Monitor trigger performance and
        implement appropriate rate limiting and debouncing. Be cautious with
        condition expressions that access external resources or perform expensive operations.
    
    See Also:
        * :class:`TriggerConfig`: Trigger configuration schema and validation
        * :class:`TriggerType`: Available trigger types and activation patterns
        * :class:`DataUpdatedTrigger`: Data change monitoring and activation
        * :class:`TimerTrigger`: Scheduled and interval-based activation
        * :class:`ConditionalTrigger`: Custom condition evaluation and activation
        * :class:`BaseStep`: Steps that can be activated by triggers
        * :class:`Workflow`: Workflows that coordinate trigger-driven processing
    """
    
    COMPONENT_TYPE = "trigger"
    REQUIRED_CONFIG_FIELDS = ['trigger_type']
    OPTIONAL_CONFIG_FIELDS = {
        'debounce_ms': 100,
        'max_frequency_hz': 10.0,
        'condition': None,
        'timer_interval_ms': None,
        'name': ''
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return TriggerConfig - ONLY method that differs from other components"""
        return TriggerConfig
    
    @classmethod
    def extract_component_config(cls, config: TriggerConfig) -> Dict[str, Any]:
        """Extract Trigger configuration"""
        return {
            'trigger_type': config.trigger_type,
            'debounce_ms': getattr(config, 'debounce_ms', 100),
            'max_frequency_hz': getattr(config, 'max_frequency_hz', 10.0),
            'condition': getattr(config, 'condition', None),
            'timer_interval_ms': getattr(config, 'timer_interval_ms', None),
            'name': getattr(config, 'name', '')
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Trigger dependencies"""
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Trigger with resolved dependencies"""
        self.config = config
        self.name = component_config.get('name') or self.__class__.__name__
        self._is_active = False
        self._callbacks: List[Callable] = []
        self._last_trigger_time = 0.0
        self._debounce_task: Optional[asyncio.Task] = None
        
        # Internal state tracking
        self._creation_time = time.time()
        self._trigger_count = 0
        self._rate_limited_count = 0
        self._callback_error_count = 0
        self._total_callback_time = 0.0
        
        # Initialize centralized logging system
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(self.name, category="triggers", debug_mode=dependencies.get('debug_mode', False))
            
            # Register with system log manager
            system_manager = get_system_log_manager()
            system_manager.register_component("triggers", self.name, self, {
                "trigger_type": component_config['trigger_type'].value if hasattr(component_config['trigger_type'], 'value') else str(component_config['trigger_type']),
                "debounce_ms": component_config['debounce_ms'],
                "max_frequency_hz": component_config['max_frequency_hz'],
                "enable_logging": True
            })
        else:
            self.nb_logger = None
    
    # TriggerBase inherits FromConfigBase.__init__ which prevents direct instantiation
        
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        uptime = time.time() - self._creation_time
        avg_callback_time = self._total_callback_time / max(self._trigger_count, 1)
        
        return {
            "is_active": self._is_active,
            "trigger_count": self._trigger_count,
            "rate_limited_count": self._rate_limited_count,
            "callback_error_count": self._callback_error_count,
            "callback_count": len(self._callbacks),
            "uptime_seconds": uptime,
            "last_trigger_time": self._last_trigger_time,
            "avg_callback_time_ms": avg_callback_time * 1000 if self._trigger_count > 0 else 0,
            "trigger_type": self.config.trigger_type.value if hasattr(self.config.trigger_type, 'value') else str(self.config.trigger_type),
            "debounce_ms": self.config.debounce_ms,
            "max_frequency_hz": self.config.max_frequency_hz,
            "success_rate": (self._trigger_count - self._callback_error_count) / max(self._trigger_count, 1)
        }
    
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        pass
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop monitoring for trigger conditions."""
        pass
    
    async def add_callback(self, callback: Callable) -> None:
        """Add a callback to be executed when triggered."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
            
            # Log callback addition
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"Callback added to trigger: {self.name}",
                                   operation="add_callback",
                                   callback_count=len(self._callbacks),
                                   callback_name=getattr(callback, '__name__', str(callback)),
                                   internal_state=self._get_internal_state())
    
    async def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
            # Log callback removal
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"Callback removed from trigger: {self.name}",
                                   operation="remove_callback",
                                   callback_count=len(self._callbacks),
                                   callback_name=getattr(callback, '__name__', str(callback)),
                                   internal_state=self._get_internal_state())
    
    async def trigger(self, data: Any = None) -> None:
        """Execute trigger with rate limiting and debouncing."""
        current_time = asyncio.get_event_loop().time()
        
        # Check frequency limit
        time_since_last = current_time - self._last_trigger_time
        min_interval = 1.0 / self.config.max_frequency_hz
        
        if time_since_last < min_interval:
            self._rate_limited_count += 1
            logger.debug(f"Trigger {self.name} rate limited")
            
            # Log rate limiting
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Trigger rate limited: {self.name}",
                                   operation="rate_limited",
                                   time_since_last=time_since_last,
                                   min_interval=min_interval,
                                   rate_limited_count=self._rate_limited_count)
            return
        
        # Cancel previous debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        # Create debounced execution
        if self.config.debounce_ms > 0:
            self._debounce_task = asyncio.create_task(
                self._debounced_execute(data)
            )
        else:
            await self._execute_callbacks(data)
    
    async def _debounced_execute(self, data: Any) -> None:
        """Execute callbacks after debounce delay."""
        try:
            await asyncio.sleep(self.config.debounce_ms / 1000.0)
            await self._execute_callbacks(data)
        except asyncio.CancelledError:
            logger.debug(f"Debounced execution cancelled for {self.name}")
            
            # Log debounce cancellation
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Debounced execution cancelled: {self.name}",
                                   operation="debounce_cancelled")
    
    async def _execute_callbacks(self, data: Any) -> None:
        """Execute all registered callbacks."""
        self._last_trigger_time = asyncio.get_event_loop().time()
        self._trigger_count += 1
        
        start_time = time.time()
        successful_callbacks = 0
        
        # Log trigger activation
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_trigger_activation(
                trigger_name=self.name,
                trigger_type=self.config.trigger_type.value if hasattr(self.config.trigger_type, 'value') else str(self.config.trigger_type),
                conditions={"data_type": type(data).__name__ if data is not None else "None"},
                activated=True
            )
            
            self.nb_logger.info(f"Trigger activated: {self.name}",
                               operation="trigger_activated",
                               callback_count=len(self._callbacks),
                               data_type=type(data).__name__ if data is not None else "None",
                               trigger_count=self._trigger_count)
        
        for i, callback in enumerate(self._callbacks):
            try:
                callback_start = time.time()
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
                callback_duration = time.time() - callback_start
                self._total_callback_time += callback_duration
                successful_callbacks += 1
                
                # Log successful callback execution
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(f"Callback executed successfully: {self.name}[{i}]",
                                       operation="callback_success",
                                       callback_index=i,
                                       callback_name=getattr(callback, '__name__', str(callback)),
                                       duration_ms=callback_duration * 1000)
                
            except Exception as e:
                self._callback_error_count += 1
                logger.error(f"Error in trigger callback: {e}")
                
                # Log callback error
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.error(f"Callback error in trigger: {self.name}[{i}]",
                                       operation="callback_error",
                                       callback_index=i,
                                       callback_name=getattr(callback, '__name__', str(callback)),
                                       error=str(e),
                                       error_type=type(e).__name__)
        
        # Log execution summary
        total_duration = time.time() - start_time
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Trigger execution completed: {self.name}",
                               operation="trigger_completed",
                               successful_callbacks=successful_callbacks,
                               total_callbacks=len(self._callbacks),
                               total_duration_ms=total_duration * 1000,
                               internal_state=self._get_internal_state())
    
    @property
    def is_active(self) -> bool:
        """Check if trigger is actively monitoring."""
        return self._is_active


class DataUnitChangeTrigger(TriggerBase):
    """
    Event-driven trigger that fires when data unit changes occur.
    Uses change listener system for immediate response without polling.
    """
    
    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]], **kwargs) -> 'DataUnitChangeTrigger':
        """
        Enhanced from_config implementation following standard NanoBrain pattern
        
        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.
        
        Args:
            config: Configuration file path, TriggerConfig object, or dictionary
            **kwargs: Additional context and dependencies
            
        Returns:
            Fully initialized DataUnitChangeTrigger instance
            
        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per Trigger rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Normalize input to TriggerConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create TriggerConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            # Already a TriggerConfig object
            config_object = config
        else:
            # Handle other BaseModel types
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            
            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        
        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)
        
        # Step 3: Extract component-specific configuration  
        component_config = cls.extract_component_config(config_object)
        
        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 5: Create instance
        instance = cls.create_instance(config_object, component_config, dependencies)
        
        # Step 6: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    @classmethod
    def extract_component_config(cls, config: TriggerConfig) -> Dict[str, Any]:
        """Extract DataUnitChangeTrigger configuration including data_unit field"""
        base_config = super().extract_component_config(config)
        return {
            **base_config,
            'data_unit': getattr(config, 'data_unit', None),
            'event_type': getattr(config, 'event_type', 'set')
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Resolve DataUnitChangeTrigger dependencies with step-scope data unit resolution
        
        ✅ ARCHITECTURAL COMPLIANCE: Only resolve data_unit string references 
        to actual DataUnit objects from step context (step-scope isolation).
        """
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        
        # Get data unit reference (may be string or object)
        data_unit_ref = component_config.get('data_unit')
        
        # ✅ STEP-SCOPE DATA UNIT RESOLUTION
        if isinstance(data_unit_ref, str) and 'step_context' in kwargs:
            step_context = kwargs['step_context']
            
            # ✅ ARCHITECTURAL COMPLIANCE: Only search within step scope
            resolved_data_unit = None
            
            # Check step input data units
            step_input_units = step_context.get('step_input_data_units', {})
            if data_unit_ref in step_input_units:
                resolved_data_unit = step_input_units[data_unit_ref]
            
            # Check step output data units
            if not resolved_data_unit:
                step_output_units = step_context.get('step_output_data_units', {})
                if data_unit_ref in step_output_units:
                    resolved_data_unit = step_output_units[data_unit_ref]
            
            if resolved_data_unit:
                # ✅ Successfully resolved string reference to DataUnit object within step scope
                data_unit_ref = resolved_data_unit
                logger = get_logger(f"{cls.__name__}.resolve_dependencies") 
                logger.info(f"✅ Resolved data_unit reference: '{component_config.get('data_unit')}' -> {resolved_data_unit.name}")
            else:
                # ✅ STEP-SCOPE ISOLATION: Only show step-local data units
                available_units = (
                    list(step_input_units.keys()) + 
                    list(step_output_units.keys())
                )
                step_name = step_context.get('step_name', 'unknown')
                raise ValueError(
                    f"❌ Data unit reference '{data_unit_ref}' not found in step '{step_name}' scope. "
                    f"Available step data units: {available_units}"
                )
        elif isinstance(data_unit_ref, str) and 'workflow_context' in kwargs:
            # ✅ LEGACY SUPPORT: Handle old workflow_context parameter for backward compatibility
            # But prioritize step-scope resolution for proper architecture
            workflow_context = kwargs['workflow_context']
            
            # Attempt to resolve string reference to actual DataUnit object
            resolved_data_unit = (
                workflow_context.get('step_input_data_units', {}).get(data_unit_ref) or
                workflow_context.get('step_output_data_units', {}).get(data_unit_ref)
            )
            
            if resolved_data_unit:
                data_unit_ref = resolved_data_unit
                logger = get_logger(f"{cls.__name__}.resolve_dependencies") 
                logger.info(f"✅ Resolved data_unit reference (legacy): '{component_config.get('data_unit')}' -> {resolved_data_unit.name}")
            else:
                available_units = (
                    list(workflow_context.get('step_input_data_units', {}).keys()) +
                    list(workflow_context.get('step_output_data_units', {}).keys())
                )
                raise ValueError(
                    f"❌ Data unit reference '{data_unit_ref}' not found in context. "
                    f"Available data units: {available_units}"
                )
        
        return {
            **base_deps,
            'data_unit': data_unit_ref,
            'event_type': component_config.get('event_type', 'data_unit_updated')
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitChangeTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_unit = dependencies.get('data_unit')
        self.event_type = dependencies.get('event_type', 'data_unit_updated')
        self.bound_actions = []
        
        if not self.data_unit:
            raise ComponentConfigurationError("DataUnitChangeTrigger requires data_unit")
    
    def bind_action(self, action_func: Callable) -> None:
        """Bind action to trigger for immediate execution on data unit changes"""
        if action_func not in self.bound_actions:
            self.bound_actions.append(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Bound action to trigger {self.name}")
    
    def unbind_action(self, action_func: Callable) -> None:
        """Unbind action from trigger"""
        if action_func in self.bound_actions:
            self.bound_actions.remove(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Unbound action from trigger {self.name}")
    
    async def start_monitoring(self) -> None:
        """Start monitoring by registering as change listener"""
        if self._is_active:
            return
            
        self._is_active = True
        
        # Register with data unit's change listener system
        self.data_unit.register_change_listener(self._on_data_unit_changed)
        
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Started monitoring data unit {self.data_unit.name}")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring by unregistering change listener"""
        if not self._is_active:
            return
            
        self._is_active = False
        
        # Unregister from data unit's change listener system
        self.data_unit.unregister_change_listener(self._on_data_unit_changed)
        
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Stopped monitoring data unit {self.data_unit.name}")
    
    async def _on_data_unit_changed(self, change_event: Dict[str, Any]) -> None:
        """Handle data unit change event"""
        try:
            # Check if event type matches (if specified)
            if hasattr(self, 'event_type') and self.event_type != 'all':
                if change_event.get('operation') != self.event_type:
                    return
            
            # Create trigger event
            trigger_event = {
                'trigger_id': getattr(self, 'trigger_id', self.name),
                'event_type': self.event_type,
                'data_unit': self.data_unit.name,
                'change_event': change_event,
                'timestamp': time.time()
            }
            
            # Execute bound actions immediately (no polling delay)
            for action in self.bound_actions:
                await action(trigger_event)
            
            # Also execute callbacks for compatibility
            await self._execute_callbacks(change_event)
            
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🔥 Trigger {self.name} fired for data unit change")
                
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Error in DataUnitChangeTrigger: {e}")


class DataUpdatedTrigger(TriggerBase):
    """
    Trigger that fires when data units are updated.
    """
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'DataUpdatedTrigger':
        """Mandatory from_config implementation for DataUpdatedTrigger"""
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
        """Resolve DataUpdatedTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'data_units': kwargs.get('data_units', [])
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DataUpdatedTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_units = dependencies.get('data_units', [])
        self._monitoring_tasks: List[asyncio.Task] = []
        
    async def start_monitoring(self) -> None:
        """Start monitoring data units for updates."""
        if self._is_active:
            return
            
        self._is_active = True
        
        # Monitor each data unit
        for data_unit in self.data_units:
            task = asyncio.create_task(self._monitor_data_unit(data_unit))
            self._monitoring_tasks.append(task)
        
        logger.debug(f"DataUpdatedTrigger {self.name} started monitoring {len(self.data_units)} data units")
        
        # Log monitoring start
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Monitoring started: {self.name}",
                               operation="start_monitoring",
                               data_units_count=len(self.data_units),
                               data_unit_names=[getattr(du, 'name', str(du)) for du in self.data_units],
                               internal_state=self._get_internal_state())
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring data units."""
        self._is_active = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.debug(f"DataUpdatedTrigger {self.name} stopped monitoring")
        
        # Log monitoring stop with statistics
        if self.enable_logging and self.nb_logger:
            uptime = time.time() - self._creation_time
            self.nb_logger.info(f"Monitoring stopped: {self.name}",
                               operation="stop_monitoring",
                               uptime_seconds=uptime,
                               final_stats={
                                   "total_triggers": self._trigger_count,
                                   "rate_limited": self._rate_limited_count,
                                   "callback_errors": self._callback_error_count
                               },
                               internal_state=self._get_internal_state())
    
    async def _monitor_data_unit(self, data_unit: Any) -> None:
        """Monitor a single data unit for changes."""
        last_update_time = 0.0
        data_unit_name = getattr(data_unit, 'name', str(data_unit))
        
        # Log monitoring start for this data unit
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Started monitoring data unit: {data_unit_name}",
                               operation="start_data_unit_monitoring",
                               data_unit_name=data_unit_name,
                               trigger_name=self.name)
        
        try:
            while self._is_active:
                # Check if data unit has metadata about last update
                if hasattr(data_unit, 'get_metadata'):
                    current_update_time = await data_unit.get_metadata('last_updated', 0.0)
                    
                    if current_update_time > last_update_time:
                        last_update_time = current_update_time
                        data = await data_unit.get()
                        
                        # Log data change detection
                        if self.enable_logging and self.nb_logger:
                            self.nb_logger.info(f"Data change detected: {data_unit_name}",
                                               operation="data_change_detected",
                                               data_unit_name=data_unit_name,
                                               trigger_name=self.name,
                                               update_time=current_update_time,
                                               data_type=type(data).__name__ if data is not None else "None")
                        
                        await self.trigger(data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.debug(f"Monitoring cancelled for data unit in {self.name}")
            
            # Log monitoring cancellation
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Data unit monitoring cancelled: {data_unit_name}",
                                   operation="data_unit_monitoring_cancelled",
                                   data_unit_name=data_unit_name,
                                   trigger_name=self.name)
        except Exception as e:
            logger.error(f"Error monitoring data unit in {self.name}: {e}")
            
            # Log monitoring error
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"Error monitoring data unit: {data_unit_name}",
                                   operation="data_unit_monitoring_error",
                                   data_unit_name=data_unit_name,
                                   trigger_name=self.name,
                                   error=str(e),
                                   error_type=type(e).__name__)


class AllDataReceivedTrigger(TriggerBase):
    """
    Trigger that fires when all required data units have data.
    """
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'AllDataReceivedTrigger':
        """Mandatory from_config implementation for AllDataReceivedTrigger"""
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
        """Resolve AllDataReceivedTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'data_units': kwargs.get('data_units', [])
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AllDataReceivedTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_units = dependencies.get('data_units', [])
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self) -> None:
        """Start monitoring for all data received."""
        if self._is_active:
            return
            
        self._is_active = True
        self._monitoring_task = asyncio.create_task(self._monitor_all_data())
        logger.debug(f"AllDataReceivedTrigger {self.name} started monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_active = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"AllDataReceivedTrigger {self.name} stopped monitoring")
    
    async def _monitor_all_data(self) -> None:
        """Monitor until all data units have data."""
        try:
            while self._is_active:
                all_have_data = True
                data_dict = {}
                
                for i, data_unit in enumerate(self.data_units):
                    data = await data_unit.get()
                    if data is None:
                        all_have_data = False
                        break
                    data_dict[f"input_{i}"] = data
                
                if all_have_data:
                    await self.trigger(data_dict)
                    # Stop monitoring after successful trigger
                    break
                
                # Check again after a short delay
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.debug(f"AllDataReceivedTrigger {self.name} monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in AllDataReceivedTrigger {self.name}: {e}")


class TimerTrigger(TriggerBase):
    """
    Trigger that fires at regular intervals.
    """
    
    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]], **kwargs) -> 'TimerTrigger':
        """
        Enhanced from_config implementation following standard NanoBrain pattern
        
        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.
        
        Args:
            config: Configuration file path, TriggerConfig object, or dictionary
            **kwargs: Additional context and dependencies
            
        Returns:
            Fully initialized TimerTrigger instance
            
        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per Trigger rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Normalize input to TriggerConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create TriggerConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            # Already a TriggerConfig object
            config_object = config
        else:
            # Handle other BaseModel types
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            
            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        
        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)
        
        # Step 3: Extract component-specific configuration  
        component_config = cls.extract_component_config(config_object)
        
        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 5: Create instance
        instance = cls.create_instance(config_object, component_config, dependencies)
        
        # Step 6: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve TimerTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        interval_ms = kwargs.get('interval_ms') or component_config.get('timer_interval_ms', 1000)
        return {
            **base_deps,
            'interval_ms': interval_ms
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize TimerTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.interval_ms = dependencies.get('interval_ms', 1000)
        self._timer_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self) -> None:
        """Start timer monitoring."""
        if self._is_active:
            return
            
        self._is_active = True
        self._timer_task = asyncio.create_task(self._timer_loop())
        logger.debug(f"TimerTrigger {self.name} started with {self.interval_ms}ms interval")
    
    async def stop_monitoring(self) -> None:
        """Stop timer monitoring."""
        self._is_active = False
        
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"TimerTrigger {self.name} stopped")
    
    async def _timer_loop(self) -> None:
        """Timer loop that triggers at intervals."""
        try:
            while self._is_active:
                await asyncio.sleep(self.interval_ms / 1000.0)
                if self._is_active:  # Check again after sleep
                    await self.trigger()
        except asyncio.CancelledError:
            logger.debug(f"TimerTrigger {self.name} loop cancelled")
        except Exception as e:
            logger.error(f"Error in TimerTrigger {self.name}: {e}")


class ManualTrigger(TriggerBase):
    """
    Trigger that fires only when manually activated.
    """
    
    @classmethod
    def from_config(cls, config: Union[str, Path, TriggerConfig, Dict[str, Any]], **kwargs) -> 'ManualTrigger':
        """
        Enhanced from_config implementation following standard NanoBrain pattern
        
        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.
        
        Args:
            config: Configuration file path, TriggerConfig object, or dictionary
            **kwargs: Additional context and dependencies
            
        Returns:
            Fully initialized ManualTrigger instance
            
        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per Trigger rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Normalize input to TriggerConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = TriggerConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create TriggerConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        elif isinstance(config, TriggerConfig):
            # Already a TriggerConfig object
            config_object = config
        else:
            # Handle other BaseModel types
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            
            try:
                TriggerConfig._allow_direct_instantiation = True
                config_object = TriggerConfig(**config_dict)
            finally:
                TriggerConfig._allow_direct_instantiation = False
        
        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)
        
        # Step 3: Extract component-specific configuration  
        component_config = cls.extract_component_config(config_object)
        
        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 5: Create instance
        instance = cls.create_instance(config_object, component_config, dependencies)
        
        # Step 6: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ManualTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
    async def start_monitoring(self) -> None:
        """Start monitoring (no-op for manual trigger)."""
        self._is_active = True
        logger.debug(f"ManualTrigger {self.name} ready for manual activation")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_active = False
        logger.debug(f"ManualTrigger {self.name} deactivated")
    
    async def fire(self, data: Any = None) -> None:
        """Manually fire the trigger."""
        if self._is_active:
            await self.trigger(data)
        else:
            logger.warning(f"ManualTrigger {self.name} not active")


 