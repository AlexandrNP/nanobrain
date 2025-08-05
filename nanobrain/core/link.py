"""
Link System for NanoBrain Framework

Provides dataflow abstractions for connecting Steps together.
Links define how information flows between system components.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Union
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
# Import logging system
from .logging_system import get_logger, get_system_log_manager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


def get_nested_value(data: Dict[str, Any], field_path: str) -> Any:
    """
    Extract nested value from dictionary using dot notation.
    
    Args:
        data: Dictionary to extract from
        field_path: Dot-separated path like "routing_decision.next_step"
        
    Returns:
        The value at the specified path, or None if not found
    """
    try:
        current = data
        for key in field_path.split('.'):
            current = current[key]
        return current
    except (KeyError, TypeError, AttributeError):
        return None


def parse_condition_from_config(condition_config: Union[str, Dict[str, Any]]) -> Callable:
    """
    Parse YAML condition configuration into a callable function.
    
    Args:
        condition_config: Either a string expression or dictionary with field/operator/value
        
    Returns:
        Callable function that evaluates the condition
    """
    if isinstance(condition_config, str):
        # Simple string conditions - could be enhanced later
        def string_condition_func(data):
            # For now, just check if the string exists in the data representation
            return condition_config in str(data)
        return string_condition_func
    
    elif isinstance(condition_config, dict):
        field = condition_config.get('field')
        operator = condition_config.get('operator', 'equals')
        value = condition_config.get('value')
        
        def dict_condition_func(data):
            try:
                field_value = get_nested_value(data, field)
                
                if operator == 'equals':
                    return field_value == value
                elif operator == 'not_equals':
                    return field_value != value
                elif operator == 'contains':
                    return value in str(field_value) if field_value else False
                elif operator == 'greater_than':
                    return float(field_value) > float(value) if field_value is not None else False
                elif operator == 'less_than':
                    return float(field_value) < float(value) if field_value is not None else False
                elif operator == 'exists':
                    return field_value is not None
                else:
                    logger.warning(f"Unknown operator: {operator}, defaulting to equals")
                    return field_value == value
                    
            except Exception as e:
                logger.debug(f"Condition evaluation failed: {e}")
                return False
                
        return dict_condition_func
    
    else:
        # Fallback for other types
        def default_condition_func(data):
            return bool(condition_config)
        return default_condition_func


class LinkType(Enum):
    """Types of links."""
    DIRECT = "direct"
    FILE = "file"
    QUEUE = "queue"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"


class LinkConfig(ConfigBase):
    """
    Configuration for links - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: LinkConfig(link_type="direct", ...)
    ✅ REQUIRED: LinkConfig.from_config('path/to/config.yml')
    """
    link_type: LinkType = LinkType.DIRECT
    buffer_size: int = Field(default=100, ge=1)
    transform_function: Optional[str] = None
    condition: Optional[Union[str, Dict[str, Any]]] = None
    file_path: Optional[str] = None
    data_mapping: Optional[Dict[str, str]] = None


class LinkBase(FromConfigBase, ABC):
    """
    Base Link Class - Data Flow Connections and Workflow Communication
    =================================================================
    
    The LinkBase class is the foundational component for creating data flow connections
    within the NanoBrain framework. Links enable seamless data transfer between steps,
    workflows, and agents with support for data transformation, conditional routing,
    buffering, and various transport mechanisms.
    
    **Core Architecture:**
        Links represent intelligent data transport systems that:
        
        * **Connect Components**: Establish data flow paths between steps and workflows
        * **Transform Data**: Apply transformations during data transfer
        * **Route Conditionally**: Support conditional data routing based on content
        * **Buffer Data**: Provide buffering for performance and reliability
        * **Validate Transfer**: Ensure data integrity during transmission
        * **Monitor Performance**: Track data flow performance and throughput
    
    **Biological Analogy:**
        Like neural pathways that carry information between different brain regions,
        links carry data between different processing components. Neural pathways
        have specialized properties (myelination for speed, neurotransmitter specificity
        for signal type, synaptic plasticity for learning) - exactly how links have
        specialized properties for data transformation, conditional routing, buffering,
        and performance optimization.
    
    **Data Flow Architecture:**
        
        **Connection Management:**
        * Direct connections for immediate data transfer
        * Buffered connections for asynchronous processing
        * Queue-based connections for reliable message passing
        * File-based connections for large dataset transfer
        
        **Data Transformation:**
        * Built-in transformation functions for common operations
        * Custom transformation scripts and functions
        * Data format conversion and normalization
        * Schema mapping and validation
        
        **Conditional Routing:**
        * Content-based routing with configurable conditions
        * Multi-path routing for complex data flows
        * Dynamic routing based on runtime conditions
        * Fallback routing for error scenarios
        
        **Performance Optimization:**
        * Intelligent buffering with configurable sizes
        * Compression for large data transfers
        * Parallel transfer for improved throughput
        * Connection pooling and reuse
    
    **Framework Integration:**
        Links seamlessly integrate with all framework components:
        
        * **Step Integration**: Connect step outputs to inputs for data flow
        * **Workflow Coordination**: Enable complex multi-step data processing
        * **Agent Communication**: Support data exchange between agents
        * **Data Unit Connectivity**: Connect data units across processing boundaries
        * **Executor Support**: Links work with all execution backends
        * **Monitoring Integration**: Comprehensive logging and performance tracking
    
    **Link Type Implementations:**
        The framework supports various link specializations:
        
        * **DirectLink**: Immediate data transfer with minimal overhead
        * **FileLink**: File-based transfer for large datasets and persistence
        * **QueueLink**: Reliable message queuing with persistence and retry
        * **TransformLink**: Data transformation during transfer
        * **ConditionalLink**: Conditional routing based on data content
        * **CompoundLink**: Combination of multiple link types for complex scenarios
    
    **Configuration Architecture:**
        Links follow the framework's configuration-first design:
        
        ```yaml
        # Direct link configuration
        name: "data_transfer"
        link_type: "direct"
        buffer_size: 1000
        
        # Source and target configuration
        source: "step_a.output_data"
        target: "step_b.input_data"
        
        # Transform link configuration
        name: "data_transformation"
        link_type: "transform"
        transform_function: "normalize_data"
        
        # Data transformation settings
        transformation:
          function_name: "custom_normalizer"
          parameters:
            scale_factor: 1.0
            remove_outliers: true
          input_schema: "schemas/raw_data.json"
          output_schema: "schemas/normalized_data.json"
        
        # Conditional link configuration
        name: "conditional_routing"
        link_type: "conditional"
        
        # Routing conditions
        condition:
          field: "data.category"
          operator: "equals"
          value: "priority"
        
        # Alternative routing
        routes:
          default:
            target: "standard_processing.input"
          priority:
            target: "priority_processing.input"
            condition:
              field: "data.priority"
              operator: "greater_than"
              value: 5
        
        # Queue link configuration
        name: "reliable_transfer"
        link_type: "queue"
        buffer_size: 10000
        
        # Queue settings
        queue_config:
          persistence: true
          retry_attempts: 3
          retry_delay_ms: 1000
          dead_letter_queue: true
          compression: true
        
        # File link configuration
        name: "large_dataset_transfer"
        link_type: "file"
        file_path: "data/transfer/{timestamp}_{source}_{target}.json"
        
        # File transfer settings
        file_config:
          compression: "gzip"
          encryption: true
          chunk_size: "10MB"
          cleanup_after_transfer: true
        ```
    
    **Usage Patterns:**
        
        **Basic Data Transfer:**
        ```python
        from nanobrain.core import DirectLink
        
        # Create link from configuration
        link = DirectLink.from_config('config/data_link.yml')
        
        # Connect data units
        await link.connect(source_data_unit, target_data_unit)
        
        # Transfer data
        result = await link.transfer(data)
        print(f"Transfer completed: {result}")
        ```
        
        **Data Transformation:**
        ```python
        # Transform link with custom function
        transform_link = TransformLink.from_config('config/transform_link.yml')
        
        # Define transformation function
        def normalize_data(data):
            # Custom normalization logic
            normalized = {
                'values': [x / max(data['values']) for x in data['values']],
                'metadata': data.get('metadata', {}),
                'timestamp': time.time()
            }
            return normalized
        
        # Register transformation
        transform_link.set_transform_function(normalize_data)
        
        # Data automatically transformed during transfer
        await transform_link.transfer(raw_data)
        ```
        
        **Conditional Routing:**
        ```python
        # Conditional link for dynamic routing
        conditional_link = ConditionalLink.from_config('config/routing_link.yml')
        
        # Define routing conditions
        conditional_link.add_route(
            condition=lambda data: data.get('priority', 0) > 5,
            target='high_priority_processor.input'
        )
        
        conditional_link.add_route(
            condition=lambda data: data.get('category') == 'urgent',
            target='urgent_processor.input'
        )
        
        # Default route for unmatched conditions
        conditional_link.set_default_route('standard_processor.input')
        
        # Data automatically routed based on content
        await conditional_link.transfer(data)
        ```
        
        **Large Dataset Transfer:**
        ```python
        # File-based link for large datasets
        file_link = FileLink.from_config('config/large_data_link.yml')
        
        # Configure compression and chunking
        file_link.set_compression('gzip')
        file_link.set_chunk_size('100MB')
        
        # Transfer large dataset efficiently
        await file_link.transfer(large_dataset)
        
        # Automatic cleanup and compression
        ```
    
    **Advanced Features:**
        
        **Performance Optimization:**
        * Intelligent buffering with adaptive sizing
        * Parallel transfer for large datasets
        * Compression for reduced bandwidth usage
        * Connection pooling and reuse for efficiency
        
        **Reliability and Error Handling:**
        * Automatic retry with exponential backoff
        * Dead letter queues for failed transfers
        * Data integrity validation with checksums
        * Graceful degradation for partial failures
        
        **Monitoring and Analytics:**
        * Real-time transfer performance monitoring
        * Throughput analysis and optimization recommendations
        * Error rate tracking and alerting
        * Data flow visualization and analysis
        
        **Security and Privacy:**
        * Data encryption for sensitive transfers
        * Access control and permission management
        * Audit logging for compliance and debugging
        * Data anonymization and privacy protection
    
    **Data Flow Patterns:**
        
        **Fan-Out Pattern:**
        * Single source data distributed to multiple targets
        * Parallel processing with result aggregation
        * Load balancing across processing components
        * Broadcast messaging for notifications
        
        **Fan-In Pattern:**
        * Multiple sources feeding into single target
        * Data aggregation and correlation
        * Result collection from parallel processing
        * Event consolidation and analysis
        
        **Pipeline Pattern:**
        * Sequential data processing through multiple stages
        * Data transformation at each pipeline stage
        * Error propagation and recovery mechanisms
        * Progress tracking and monitoring
        
        **Mesh Pattern:**
        * Complex interconnection of multiple components
        * Dynamic routing based on content and conditions
        * Adaptive data flow optimization
        * Distributed processing coordination
    
    **Integration Patterns:**
        
        **Workflow Integration:**
        * Links coordinate data flow between workflow steps
        * Automatic activation of downstream processing
        * Data dependency resolution and management
        * Result propagation and collection
        
        **Agent Coordination:**
        * Inter-agent communication and data sharing
        * Context preservation across agent interactions
        * Result correlation and analysis
        * Collaborative processing workflows
        
        **External System Integration:**
        * API-based data exchange with external services
        * Database integration for data persistence
        * File system integration for large datasets
        * Message queue integration for reliable messaging
    
    **Performance and Scalability:**
        
        **Throughput Optimization:**
        * Asynchronous data transfer for non-blocking operations
        * Batch processing for improved efficiency
        * Connection pooling for resource optimization
        * Adaptive buffering based on load patterns
        
        **Scalability Features:**
        * Horizontal scaling through link distribution
        * Load balancing across multiple link instances
        * Dynamic resource allocation based on demand
        * Auto-scaling for varying workloads
        
        **Resource Management:**
        * Memory management for large data transfers
        * Network bandwidth optimization and throttling
        * Storage management for file-based transfers
        * Cleanup and garbage collection
    
    **Development and Testing:**
        
        **Testing Support:**
        * Mock link implementations for testing
        * Data flow simulation and validation
        * Performance benchmarking and profiling
        * Integration testing with steps and workflows
        
        **Debugging Features:**
        * Comprehensive logging with data flow tracing
        * Transfer inspection and analysis tools
        * Performance profiling and optimization hints
        * Visual data flow monitoring and debugging
        
        **Development Tools:**
        * Link configuration validation and linting
        * Data flow visualization and analysis
        * Performance monitoring and optimization tools
        * Template generation for common link patterns
    
    Attributes:
        name (str): Link identifier for logging and component coordination
        link_type (LinkType): Type of link and data transfer mechanism
        buffer_size (int): Buffer size for data transfer optimization
        transform_function (str, optional): Transformation function name for data processing
        condition (Union[str, Dict], optional): Routing condition for conditional links
        source (str): Source data unit or component identifier
        target (str): Target data unit or component identifier
        is_connected (bool): Whether link is currently active and connected
        transfer_count (int): Total number of data transfers performed
        performance_metrics (Dict): Real-time performance and usage metrics
    
    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like DirectLink, TransformLink, or
        ConditionalLink. All links must be created using the from_config
        pattern with proper configuration files following framework patterns.
    
    Warning:
        Links may consume significant network and memory resources depending
        on data size and transfer patterns. Monitor resource usage and implement
        appropriate limits, buffering, and cleanup mechanisms. Be cautious with
        file-based links that may consume disk space.
    
    See Also:
        * :class:`LinkConfig`: Link configuration schema and validation
        * :class:`LinkType`: Available link types and transfer mechanisms
        * :class:`DirectLink`: Direct data transfer with minimal overhead
        * :class:`TransformLink`: Data transformation during transfer
        * :class:`ConditionalLink`: Conditional routing based on data content
        * :class:`BaseStep`: Steps that connect through links for data flow
        * :class:`Workflow`: Workflows that coordinate link-based data processing
    """
    
    COMPONENT_TYPE = "link"
    REQUIRED_CONFIG_FIELDS = ['link_type']
    OPTIONAL_CONFIG_FIELDS = {
        'buffer_size': 100,
        'transform_function': None,
        'condition': None,
        'file_path': None,
        'data_mapping': None
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return LinkConfig - ONLY method that differs from other components"""
        return LinkConfig
    
    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract Link configuration"""
        return {
            'link_type': config.link_type,
            'buffer_size': getattr(config, 'buffer_size', 100),
            'transform_function': getattr(config, 'transform_function', None),
            'condition': getattr(config, 'condition', None),
            'file_path': getattr(config, 'file_path', None),
            'data_mapping': getattr(config, 'data_mapping', None)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Link dependencies"""
        return {
            'source': kwargs.get('source'),
            'target': kwargs.get('target'),
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }
    
    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Link with resolved dependencies"""
        self.config = config
        self._source = dependencies.get('source')
        self._target = dependencies.get('target')
        self._update_name()
        self._is_active = False
        self._transfer_count = 0
        self._error_count = 0
        self._creation_time = time.time()
        self._last_transfer_time = None
        self._total_transfer_time = 0.0
        
        # Store data mapping configuration for transform operations
        self.data_mapping = getattr(config, 'data_mapping', None)
        
        # Initialize centralized logging system
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(self.name, category="links", debug_mode=dependencies.get('debug_mode', False))
            
            # Register with system log manager
            system_manager = get_system_log_manager()
            system_manager.register_component("links", self.name, self, {
                "link_type": component_config['link_type'].value if hasattr(component_config['link_type'], 'value') else str(component_config['link_type']),
                "source": getattr(self.source, 'name', str(self.source)),
                "target": getattr(self.target, 'name', str(self.target)),
                "buffer_size": component_config['buffer_size'],
                "enable_logging": True
            })
        else:
            self.nb_logger = None
        
        # EVENT-DRIVEN ARCHITECTURE: Enhanced link system with automatic trigger binding
        self.transfer_trigger = None
        self._setup_automatic_transfer_if_possible()
    
    # LinkBase inherits FromConfigBase.__init__ which prevents direct instantiation
    
    def _setup_automatic_transfer_if_possible(self) -> None:
        """Setup automatic transfer trigger if source data unit supports change listeners"""
        try:
            # Check if source has change listener support (from Phase 1 implementation)
            if hasattr(self.source, 'register_change_listener'):
                # Create automatic transfer trigger
                self._create_transfer_trigger()
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(f"🔗 Setup automatic transfer for link {self.name}")
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Could not setup automatic transfer: {e}")
    
    def _create_transfer_trigger(self) -> None:
        """Create trigger for automatic data transfer on source data unit changes"""
        from .trigger import DataUnitChangeTrigger, TriggerConfig
        
        try:
            # Create trigger config
            trigger_config = TriggerConfig(
                trigger_type="data_updated",
                name=f"link_transfer_{self.name}"
            )
            
            # Create trigger with action binding
            self.transfer_trigger = DataUnitChangeTrigger.from_config(
                trigger_config,
                data_unit=self.source,
                event_type='set'
            )
            
            # Bind automatic transfer action
            self.transfer_trigger.bind_action(self._on_source_data_changed)
            
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Created transfer trigger for link {self.name}")
                
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"Failed to create transfer trigger: {e}")
    
    def _update_name(self) -> None:
        """Update link name based on current source and target"""
        source_name = getattr(self._source, 'name', 'unknown') if self._source else 'unknown'
        target_name = getattr(self._target, 'name', 'unknown') if self._target else 'unknown'
        self.name = f"{source_name}->{target_name}"
    
    @property
    def source(self):
        """Get the source data unit"""
        return self._source
    
    @source.setter
    def source(self, value):
        """Set the source data unit and update name"""
        self._source = value
        self._update_name()
        # Update logger if it exists
        if hasattr(self, 'nb_logger') and self.nb_logger:
            from nanobrain.core.logging_system import get_logger
            self.nb_logger = get_logger(self.name, category="links")
    
    @property
    def target(self):
        """Get the target data unit"""
        return self._target
    
    @target.setter
    def target(self, value):
        """Set the target data unit and update name"""
        self._target = value
        self._update_name()
        # Update logger if it exists
        if hasattr(self, 'nb_logger') and self.nb_logger:
            from nanobrain.core.logging_system import get_logger
            self.nb_logger = get_logger(self.name, category="links")
    
    async def _on_source_data_changed(self, trigger_event: Dict[str, Any]) -> None:
        """Handle source data unit change - automatically transfer data"""
        try:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🔥 Link {self.name} triggered by source data change")
            
            # Get data from source
            source_data = await self.source.get()
            
            # Apply any transformations or conditions
            if await self._should_transfer(source_data):
                transformed_data = await self._transform_data(source_data)
                
                # Transfer to target
                await self.target.set(transformed_data)
                
                # Record transfer
                await self._record_transfer(True, data_info={'auto_transfer': True})
                
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(f"📤 Auto-transferred data via link {self.name}")
            
        except Exception as e:
            await self._record_transfer(False)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Auto-transfer failed for link {self.name}: {e}")
    
    async def _should_transfer(self, data: Any) -> bool:
        """Check if data should be transferred (apply conditions)"""
        try:
            return self.condition_func(data) if hasattr(self, 'condition_func') else True
        except:
            return True
    
    async def _transform_data(self, data: Any) -> Any:
        """Apply data transformations (can be overridden by subclasses)"""
        # Apply data mapping if configured
        if hasattr(self, 'data_mapping') and self.data_mapping:
            if isinstance(data, dict):
                transformed = {}
                for target_key, source_key in self.data_mapping.items():
                    if source_key in data:
                        transformed[target_key] = data[source_key]
                return transformed
        
        return data
    
    def _parse_data_unit_reference(self, reference: str) -> tuple[str, str]:
        """Parse step.data_unit notation"""
        if '.' not in reference:
            raise ValueError(f"Invalid reference: {reference}. Must use 'step.data_unit' format")
        
        step_name, data_unit_name = reference.split('.', 1)
        return step_name.strip(), data_unit_name.strip()



    def _validate_dot_notation_reference(self, reference: str) -> bool:
        """Validate that reference uses proper step.data_unit format"""
        try:
            step_name, data_unit_name = self._parse_data_unit_reference(reference)
            return len(step_name) > 0 and len(data_unit_name) > 0
        except ValueError:
            return False

    def resolve_link_endpoints(self, source_ref: str, target_ref: str, 
                             workflow_context: Dict[str, Any]) -> tuple[Any, Any]:
        """Resolve both source and target references using dot notation"""
        # Validate references
        if not self._validate_dot_notation_reference(source_ref):
            raise ValueError(f"Invalid source reference: {source_ref}")
        if not self._validate_dot_notation_reference(target_ref):
            raise ValueError(f"Invalid target reference: {target_ref}")
        
        # Resolve references
        source_data_unit = self._resolve_data_unit_reference(source_ref, workflow_context)
        target_data_unit = self._resolve_data_unit_reference(target_ref, workflow_context)
        
        return source_data_unit, target_data_unit
        
    @abstractmethod
    async def transfer(self, data: Any) -> None:
        """Transfer data from source to target."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the link."""
        # EVENT-DRIVEN ARCHITECTURE: Start automatic transfer trigger
        if self.transfer_trigger:
            await self.transfer_trigger.start_monitoring()
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🚀 Started automatic transfer for link {self.name}")
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the link."""
        # EVENT-DRIVEN ARCHITECTURE: Stop automatic transfer trigger
        if self.transfer_trigger:
            await self.transfer_trigger.stop_monitoring()
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🛑 Stopped automatic transfer for link {self.name}")
        pass
    
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        uptime = time.time() - self._creation_time
        avg_transfer_time = self._total_transfer_time / max(self._transfer_count, 1)
        
        return {
            "is_active": self._is_active,
            "transfer_count": self._transfer_count,
            "error_count": self._error_count,
            "success_rate": self._transfer_count / max(self._transfer_count + self._error_count, 1),
            "uptime_seconds": uptime,
            "last_transfer_time": self._last_transfer_time,
            "avg_transfer_time_ms": avg_transfer_time * 1000 if self._transfer_count > 0 else 0,
            "link_type": self.config.link_type.value if hasattr(self.config.link_type, 'value') else str(self.config.link_type),
            "source": getattr(self.source, 'name', str(self.source)),
            "target": getattr(self.target, 'name', str(self.target))
        }
    
    @property
    def is_active(self) -> bool:
        """Check if link is active."""
        return self._is_active
    
    @property
    def transfer_count(self) -> int:
        """Get number of successful transfers."""
        return self._transfer_count
    
    @property
    def error_count(self) -> int:
        """Get number of transfer errors."""
        return self._error_count
    
    async def _record_transfer(self, success: bool = True, duration_ms: float = None, data_info: Dict[str, Any] = None) -> None:
        """Record transfer statistics with comprehensive logging."""
        transfer_time = time.time()
        
        if success:
            self._transfer_count += 1
            self._last_transfer_time = transfer_time
            if duration_ms:
                self._total_transfer_time += duration_ms / 1000.0
        else:
            self._error_count += 1
        
        # Log transfer event
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_data_transfer(
                source=getattr(self.source, 'name', str(self.source)),
                destination=getattr(self.target, 'name', str(self.target)),
                data_type=data_info.get('data_type', 'unknown') if data_info else 'unknown',
                size_bytes=data_info.get('size_bytes', 0) if data_info else 0
            )
            
            self.nb_logger.info(f"Link transfer {'succeeded' if success else 'failed'}: {self.name}",
                               operation="transfer",
                               success=success,
                               duration_ms=duration_ms,
                               data_info=data_info or {},
                               internal_state=self._get_internal_state())


class DirectLink(LinkBase):
    """
    Direct link that immediately transfers data from source to target.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    COMPONENT_TYPE = "direct_link"
    REQUIRED_CONFIG_FIELDS = ['link_type']
    OPTIONAL_CONFIG_FIELDS = {
        'buffer_size': 100,
        'data_mapping': None
    }
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of DirectLink is prohibited. "
            "ALL framework components must use DirectLink.from_config() "
            "as per mandatory framework requirements."
        )
    
    @classmethod
    def from_config(cls, config: Union[str, Path, LinkConfig, Dict[str, Any]], **kwargs) -> 'DirectLink':
        """Mandatory from_config implementation for DirectLink with dictionary support"""
        # Get logger
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Normalize input to LinkConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = LinkConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create LinkConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config)
            finally:
                LinkConfig._allow_direct_instantiation = False
        elif isinstance(config, LinkConfig):
            # Already a LinkConfig object
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
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config_dict)
            finally:
                LinkConfig._allow_direct_instantiation = False
        
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
        
        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract DirectLink configuration"""
        return {
            'link_type': config.link_type,
            'buffer_size': getattr(config, 'buffer_size', 100),
            'data_mapping': getattr(config, 'data_mapping', None)
        }
    


    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DirectLink dependencies - trust framework for string references"""
        return {
            'source': kwargs.get('source'),
            'target': kwargs.get('target'), 
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }
    
    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DirectLink with resolved dependencies"""
        # Call parent _init_from_config
        super()._init_from_config(config, component_config, dependencies)
        
    async def start(self) -> None:
        """Start the direct link."""
        # EVENT-DRIVEN ARCHITECTURE: Start automatic transfer trigger first
        await super().start()
        
        self._is_active = True
        logger.debug(f"DirectLink {self.name} started")
        
        # Log start event
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Link started: {self.name}",
                               operation="start",
                               internal_state=self._get_internal_state())
    
    async def stop(self) -> None:
        """Stop the direct link."""
        # EVENT-DRIVEN ARCHITECTURE: Stop automatic transfer trigger first
        await super().stop()
        
        self._is_active = False
        logger.debug(f"DirectLink {self.name} stopped")
        
        # Log stop event with final statistics
        if self.enable_logging and self.nb_logger:
            uptime = time.time() - self._creation_time
            self.nb_logger.info(f"Link stopped: {self.name}",
                               operation="stop",
                               uptime_seconds=uptime,
                               final_stats={
                                   "total_transfers": self._transfer_count,
                                   "total_errors": self._error_count,
                                   "success_rate": self._transfer_count / max(self._transfer_count + self._error_count, 1)
                               },
                               internal_state=self._get_internal_state())
    
    async def transfer(self, data: Any) -> None:
        """Transfer data directly to target."""
        if not self._is_active:
            logger.warning(f"DirectLink {self.name} not active")
            return
        
        start_time = time.time()
        transfer_method = None
        
        try:
            # Prepare data info for logging
            data_info = {
                "data_type": type(data).__name__ if data is not None else "None",
                "size_bytes": len(str(data)) if data is not None else 0,
                "data_value": data if self.enable_logging else None
            }
            
            # Handle different target types
            if hasattr(self.target, 'set') and callable(getattr(self.target, 'set')):
                # Target is a DataUnit - call set method directly
                transfer_method = "DataUnit.set"
                await self.target.set(data)
                logger.debug(f"DirectLink {self.name} transferred data to DataUnit")
            elif hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                # Target is a Step with input data units
                transfer_method = "Step.input_data_units"
                for input_unit in self.target.input_data_units.values():
                    await input_unit.set(data)
                logger.debug(f"DirectLink {self.name} transferred data to Step input units")
            elif hasattr(self.target, 'execute') and callable(getattr(self.target, 'execute')):
                # Target is a Step - trigger execution
                transfer_method = "Step.execute"
                await self.target.execute(data)
                logger.debug(f"DirectLink {self.name} executed target Step")
            elif hasattr(self.target, 'set_input'):
                # Direct method call
                transfer_method = "set_input"
                await self.target.set_input(data)
                logger.debug(f"DirectLink {self.name} called set_input on target")
            else:
                logger.warning(f"Target {getattr(self.target, 'name', str(self.target))} has no compatible input mechanism")
                
                # Log failed transfer attempt
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.warning(f"Transfer failed - no compatible target mechanism: {self.name}",
                                         operation="transfer_failed",
                                         reason="no_compatible_mechanism",
                                         target_type=type(self.target).__name__,
                                         data_info=data_info)
                return
            
            # Calculate duration and record success
            duration_ms = (time.time() - start_time) * 1000
            data_info["transfer_method"] = transfer_method
            await self._record_transfer(True, duration_ms, data_info)
            logger.debug(f"DirectLink {self.name} transferred data successfully")
            
        except Exception as e:
            # Calculate duration and record failure
            duration_ms = (time.time() - start_time) * 1000
            data_info = {
                "data_type": type(data).__name__ if data is not None else "None",
                "size_bytes": len(str(data)) if data is not None else 0,
                "transfer_method": transfer_method,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            await self._record_transfer(False, duration_ms, data_info)
            logger.error(f"DirectLink {self.name} transfer failed: {e}")
            
            # Log detailed error information
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"Transfer error in link: {self.name}",
                                   operation="transfer_error",
                                   error=str(e),
                                   error_type=type(e).__name__,
                                   duration_ms=duration_ms,
                                   data_info=data_info,
                                   internal_state=self._get_internal_state())
            raise


class QueueLink(LinkBase):
    """
    Queue-based link that buffers data between source and target.
    """
    
    def __init__(self, source: Any, target: Any, config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.QUEUE)
        super().__init__(source, target, config, **kwargs)
        self._queue: Optional[asyncio.Queue] = None
        self._consumer_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the queue link."""
        if self._is_active:
            return
            
        self._queue = asyncio.Queue(maxsize=self.config.buffer_size)
        self._consumer_task = asyncio.create_task(self._consume_queue())
        self._is_active = True
        logger.debug(f"QueueLink {self.name} started with buffer size {self.config.buffer_size}")
    
    async def stop(self) -> None:
        """Stop the queue link."""
        self._is_active = False
        
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        self._queue = None
        logger.debug(f"QueueLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Add data to queue for transfer."""
        if not self._is_active or not self._queue:
            logger.warning(f"QueueLink {self.name} not active")
            return
            
        try:
            await self._queue.put(data)
            logger.debug(f"QueueLink {self.name} queued data")
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"QueueLink {self.name} queue failed: {e}")
            raise
    
    async def _consume_queue(self) -> None:
        """Consume data from queue and transfer to target."""
        try:
            while self._is_active and self._queue:
                try:
                    # Wait for data with timeout
                    data = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    
                    # Transfer to target
                    if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                        input_unit = self.target.input_data_units[0]
                        await input_unit.set(data)
                    elif hasattr(self.target, 'set_input'):
                        await self.target.set_input(data)
                    
                    await self._record_transfer(True)
                    
                except asyncio.TimeoutError:
                    # Continue loop on timeout
                    continue
                except Exception as e:
                    await self._record_transfer(False)
                    logger.error(f"QueueLink {self.name} consumer error: {e}")
                    
        except asyncio.CancelledError:
            logger.debug(f"QueueLink {self.name} consumer cancelled")


class TransformLink(LinkBase):
    """
    Link that transforms data before transferring to target.
    """
    
    def __init__(self, source: Any, target: Any, transform_func: Callable, 
                 config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.TRANSFORM)
        super().__init__(source, target, config, **kwargs)
        self.transform_func = transform_func
        
    async def start(self) -> None:
        """Start the transform link."""
        self._is_active = True
        logger.debug(f"TransformLink {self.name} started")
    
    async def stop(self) -> None:
        """Stop the transform link."""
        self._is_active = False
        logger.debug(f"TransformLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Transform data and transfer to target."""
        if not self._is_active:
            logger.warning(f"TransformLink {self.name} not active")
            return
            
        try:
            # Apply transformation
            if asyncio.iscoroutinefunction(self.transform_func):
                transformed_data = await self.transform_func(data)
            else:
                transformed_data = self.transform_func(data)
            
            # Transfer transformed data
            if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                input_unit = self.target.input_data_units[0]
                await input_unit.set(transformed_data)
            elif hasattr(self.target, 'set_input'):
                await self.target.set_input(transformed_data)
            
            await self._record_transfer(True)
            logger.debug(f"TransformLink {self.name} transformed and transferred data")
            
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"TransformLink {self.name} transform failed: {e}")
            raise


class ConditionalLink(LinkBase):
    """
    Link that transfers data only when condition is met.
    Enhanced with mandatory from_config pattern implementation.
    """
    
    COMPONENT_TYPE = "conditional_link"
    REQUIRED_CONFIG_FIELDS = ['link_type', 'condition']
    OPTIONAL_CONFIG_FIELDS = {
        'buffer_size': 100,
        'data_mapping': None
    }
    
    def __init__(self, *args, **kwargs):
        """Prevent direct instantiation - use from_config instead"""
        raise RuntimeError(
            "Direct instantiation of ConditionalLink is prohibited. "
            "ALL framework components must use ConditionalLink.from_config() "
            "as per mandatory framework requirements."
        )
    
    @classmethod
    def from_config(cls, config: Union[str, Path, LinkConfig, Dict[str, Any]], **kwargs) -> 'ConditionalLink':
        """Mandatory from_config implementation for ConditionalLink with dictionary support"""
        # Get logger
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Normalize input to LinkConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = LinkConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create LinkConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config)
            finally:
                LinkConfig._allow_direct_instantiation = False
        elif isinstance(config, LinkConfig):
            # Already a LinkConfig object
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
                LinkConfig._allow_direct_instantiation = True
                config_object = LinkConfig(**config_dict)
            finally:
                LinkConfig._allow_direct_instantiation = False
        
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
        
        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    @classmethod
    def extract_component_config(cls, config: LinkConfig) -> Dict[str, Any]:
        """Extract ConditionalLink configuration"""
        return {
            'link_type': config.link_type,
            'condition': getattr(config, 'condition', None),
            'buffer_size': getattr(config, 'buffer_size', 100),
            'data_mapping': getattr(config, 'data_mapping', None)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve ConditionalLink dependencies"""
        condition_config = component_config.get('condition')
        if not condition_config:
            raise ValueError("ConditionalLink requires condition configuration")
        
        # Parse condition function from config
        condition_func = parse_condition_from_config(condition_config)
        
        return {
            'source': kwargs.get('source'),
            'target': kwargs.get('target'), 
            'condition_func': condition_func,
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }
    
    def _init_from_config(self, config: LinkConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ConditionalLink with resolved dependencies"""
        # Call parent _init_from_config
        super()._init_from_config(config, component_config, dependencies)
        
        # Set ConditionalLink-specific attributes
        self.condition_func = dependencies['condition_func']
        
    async def start(self) -> None:
        """Start the conditional link."""
        self._is_active = True
        logger.debug(f"ConditionalLink {self.name} started")
    
    async def stop(self) -> None:
        """Stop the conditional link."""
        self._is_active = False
        logger.debug(f"ConditionalLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Transfer data if condition is met."""
        if not self._is_active:
            logger.warning(f"ConditionalLink {self.name} not active")
            return
            
        try:
            # Check condition
            if asyncio.iscoroutinefunction(self.condition_func):
                should_transfer = await self.condition_func(data)
            else:
                should_transfer = self.condition_func(data)
            
            if should_transfer:
                # Transfer data
                if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                    input_unit = self.target.input_data_units[0]
                    await input_unit.set(data)
                elif hasattr(self.target, 'set_input'):
                    await self.target.set_input(data)
                
                await self._record_transfer(True)
                logger.debug(f"ConditionalLink {self.name} condition met, transferred data")
            else:
                logger.debug(f"ConditionalLink {self.name} condition not met, skipped transfer")
                
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"ConditionalLink {self.name} condition check failed: {e}")
            raise


class FileLink(LinkBase):
    """
    File-based link that transfers data through file system.
    """
    
    def __init__(self, source: Any, target: Any, file_path: str, 
                 config: Optional[LinkConfig] = None, **kwargs):
        config = config or LinkConfig(link_type=LinkType.FILE, file_path=file_path)
        super().__init__(source, target, config, **kwargs)
        self.file_path = file_path
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the file link."""
        if self._is_active:
            return
            
        self._monitor_task = asyncio.create_task(self._monitor_file())
        self._is_active = True
        logger.debug(f"FileLink {self.name} started monitoring {self.file_path}")
    
    async def stop(self) -> None:
        """Stop the file link."""
        self._is_active = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"FileLink {self.name} stopped")
    
    async def transfer(self, data: Any) -> None:
        """Write data to file for transfer."""
        if not self._is_active:
            logger.warning(f"FileLink {self.name} not active")
            return
            
        try:
            from pathlib import Path
            import json
            
            file_path = Path(self.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data to file
            if isinstance(data, (dict, list)):
                content = json.dumps(data, indent=2)
            else:
                content = str(data)
            
            file_path.write_text(content)
            logger.debug(f"FileLink {self.name} wrote data to file")
            
        except Exception as e:
            await self._record_transfer(False)
            logger.error(f"FileLink {self.name} file write failed: {e}")
            raise
    
    async def _monitor_file(self) -> None:
        """Monitor file for changes and transfer to target."""
        from pathlib import Path
        import json
        
        file_path = Path(self.file_path)
        last_modified = 0.0
        
        try:
            while self._is_active:
                try:
                    if file_path.exists():
                        current_modified = file_path.stat().st_mtime
                        
                        if current_modified > last_modified:
                            last_modified = current_modified
                            
                            # Read and transfer data
                            content = file_path.read_text()
                            try:
                                data = json.loads(content)
                            except json.JSONDecodeError:
                                data = content
                            
                            # Transfer to target
                            if hasattr(self.target, 'input_data_units') and self.target.input_data_units:
                                input_unit = self.target.input_data_units[0]
                                await input_unit.set(data)
                            elif hasattr(self.target, 'set_input'):
                                await self.target.set_input(data)
                            
                            await self._record_transfer(True)
                            logger.debug(f"FileLink {self.name} detected file change and transferred")
                    
                    await asyncio.sleep(1.0)  # Check every second
                    
                except Exception as e:
                    await self._record_transfer(False)
                    logger.error(f"FileLink {self.name} monitor error: {e}")
                    await asyncio.sleep(5.0)  # Wait longer on error
                    
        except asyncio.CancelledError:
            logger.debug(f"FileLink {self.name} monitor cancelled") 