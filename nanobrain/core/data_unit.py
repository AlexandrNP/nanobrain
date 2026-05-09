"""
Data Unit System for NanoBrain Framework

Provides data interfaces and ingestion capabilities for Steps.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Callable
from pydantic import BaseModel, Field, ConfigDict, field_validator
from pathlib import Path
import json
import time

# Async file operations
import aiofiles

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
# Import logging system
from .logging_system import get_logger, get_system_log_manager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase
# Import event types for proper enum-based event handling
from .event_types import DataUnitEventType, validate_event_type, DEFAULT_DATA_UNIT_EVENT

logger = logging.getLogger(__name__)


class DataUnitConfig(ConfigBase):
    """
    Configuration for data units - INHERITS constructor prohibition.

    ❌ FORBIDDEN: DataUnitConfig(name="test", class="...")
    ✅ REQUIRED: DataUnitConfig.from_config('path/to/config.yml')
    """

    # MANDATORY class field for data unit type specification
    class_field: str = Field(
        alias="class", description="Full class path for data unit type")

    # Keep existing fields
    name: str = ""
    description: str = ""
    persistent: bool = False
    cache_size: int = Field(default=1000, ge=1)
    file_path: Optional[str] = None
    encoding: str = "utf-8"
    initial_value: Optional[str] = None

    # G3 — DataUnitProxyRef configuration (all optional; only meaningful for
    # DataUnitProxyRef class). See `nanobrain_capability_gaps.md G3`.
    proxystore_connector: Optional[str] = Field(
        default=None,
        description="ProxyStore connector kind: 'file' | 'redis' | 'globus' | 'endpoint'. "
                    "Required when class is DataUnitProxyRef; ignored otherwise."
    )
    proxystore_store_name: Optional[str] = Field(
        default=None,
        description="ProxyStore store name (used as the registration key)."
    )
    proxystore_store_dir: Optional[str] = Field(
        default=None,
        description="Filesystem directory for the file connector. Required when "
                    "proxystore_connector='file'."
    )
    proxystore_redis_addr: Optional[str] = Field(
        default=None,
        description="Redis address (host:port) for the redis connector. Required "
                    "when proxystore_connector='redis'."
    )
    proxystore_namespace_prefix: Optional[str] = Field(
        default=None,
        description="Optional namespace prefix prepended to every key (G13 "
                    "multi-tenant ProxyStore namespacing). Default: no prefix."
    )
    proxystore_metadata_mime: Optional[str] = Field(
        default=None,
        description="Descriptive mime type recorded in the ref's metadata block. "
                    "Not used for routing; descriptive only."
    )
    proxystore_metadata_max_size_bytes: Optional[int] = Field(
        default=None,
        description="Descriptive size hint in the ref's metadata block. "
                    "Not enforced — purely informational for downstream consumers."
    )

    @field_validator('class_field')
    @classmethod
    def validate_class_field(cls, v):
        """Validate class field is properly specified"""
        if not v or not v.strip():
            raise ValueError("Data unit class must be specified")
        if not v.startswith('nanobrain.core.data_unit.'):
            raise ValueError(
                "Data unit class must be from nanobrain.core.data_unit module")
        return v.strip()

    @property
    def class_path(self) -> str:
        """Get the class path for component factory"""
        return self.class_field


class DataUnitBase(FromConfigBase, ABC):
    """
    Base Data Unit Class - Type-Safe Data Containers for Event-Driven Workflows
    ===========================================================================

    The DataUnitBase class is the foundational component for data management within
    the NanoBrain framework. Data units provide type-safe, event-driven data containers
    that enable seamless communication between workflow components while maintaining
    data integrity, persistence, and performance optimization.

    **Core Architecture:**
        Data units represent intelligent data containers that:

        * **Store Data Safely**: Type-safe data storage with validation and serialization
        * **Enable Communication**: Facilitate data flow between steps, agents, and workflows
        * **Trigger Events**: Emit events when data changes to activate downstream processing
        * **Manage Persistence**: Handle data persistence, caching, and retrieval strategies
        * **Ensure Consistency**: Maintain data consistency across concurrent operations
        * **Track Metadata**: Store rich metadata about data provenance and lineage

    **Biological Analogy:**
        Like synaptic vesicles that store and release neurotransmitters for neural
        communication, data units store and provide data for component communication.
        Synaptic vesicles are specialized organelles that package neurotransmitters,
        respond to cellular signals for release, and enable precise information transfer
        between neurons - exactly how data units package information, respond to workflow
        events, and enable precise data transfer between framework components.

    **Data Management Architecture:**

        **Storage Patterns:**
        * In-memory storage for fast access and temporary data
        * File-based storage for persistent data and large datasets
        * Streaming storage for real-time data processing
        * String storage for text and configuration data
        * Binary storage for complex data structures and media

        **Data Types and Validation:**
        * Strong typing with Pydantic schema validation
        * Custom data type definitions and extensions
        * Automatic serialization and deserialization
        * Data format conversion and normalization
        * Content validation and integrity checking

        **Event-Driven Communication:**
        * Change notifications for data updates and modifications
        * Listener registration for downstream component activation
        * Event filtering based on data properties and conditions
        * Batch event processing for performance optimization
        * Event history tracking for debugging and analysis

        **Concurrency and Thread Safety:**
        * Atomic operations for data consistency
        * Lock-free data structures for high-performance access
        * Async/await support for non-blocking operations
        * Thread-safe data access and modification
        * Deadlock prevention and resolution mechanisms

    **Framework Integration:**
        Data units seamlessly integrate with all framework components:

        * **Step Integration**: Provide input/output data for step processing
        * **Workflow Coordination**: Enable data flow between workflow stages
        * **Agent Communication**: Store conversation history and context
        * **Tool Data Exchange**: Manage tool inputs, outputs, and intermediate results
        * **Trigger Activation**: Trigger workflow events based on data changes
        * **Monitoring Integration**: Comprehensive data access and modification logging

    **Data Unit Specializations:**
        The framework supports various data unit specializations:

        * **DataUnitMemory**: High-performance in-memory data storage
        * **DataUnitFile**: File-based persistent data storage with path management
        * **DataUnitString**: Optimized string data containers for text processing
        * **DataUnitStream**: Real-time streaming data containers
        * **ConversationHistoryUnit**: Specialized for conversation and interaction history
        * **BioinformaticsDataUnit**: Optimized for biological data and sequences

    **Configuration Architecture:**
        Data units follow the framework's configuration-first design:

        ```yaml
        # Memory-based data unit
        name: "processing_results"
        description: "Stores processing results in memory"
        class: "nanobrain.core.data_unit.DataUnitMemory"
        persistent: false
        cache_size: 1000
        initial_value: null

        # File-based data unit
        name: "dataset_storage"
        description: "Persistent file-based data storage"
        class: "nanobrain.core.data_unit.DataUnitFile"
        file_path: "data/dataset.json"
        encoding: "utf-8"
        persistent: true

        # Streaming data unit
        name: "realtime_feed"
        description: "Real-time data streaming"
        class: "nanobrain.core.data_unit.DataUnitStream"
        buffer_size: 1000
        stream_timeout: 30
        auto_flush: true

        # Conversation history
        name: "chat_history"
        description: "Conversation history storage"
        class: "nanobrain.library.infrastructure.data.ConversationHistoryUnit"
        max_history_length: 100
        persistence_backend: "sqlite"
        encryption_enabled: true
        ```

    **Usage Patterns:**

        **Basic Data Storage and Retrieval:**
        ```python
        from nanobrain.core import DataUnitMemory

        # Create data unit from configuration
        data_unit = DataUnitMemory.from_config('config/results_storage.yml')

        # Store data
        await data_unit.set({"results": [1, 2, 3], "status": "complete"})

        # Retrieve data
        data = await data_unit.get()
        print(f"Stored data: {data}")

        # Check if data exists
        if await data_unit.has_data():
            print("Data is available")
        ```

        **Event-Driven Data Processing:**
        ```python
        # Register change listener for automatic processing
        def on_data_change(data_unit, old_value, new_value):
            print(f"Data changed from {old_value} to {new_value}")

        data_unit.register_change_listener(on_data_change)

        # Data changes automatically trigger listeners
        await data_unit.set({"new": "data"})
        # Listener automatically called with change information
        ```

        **File-Based Persistent Storage:**
        ```python
        from nanobrain.core import DataUnitFile

        # Create file-based data unit
        file_unit = DataUnitFile.from_config('config/dataset_storage.yml')

        # Store data to file
        await file_unit.set({
            "dataset": "large_dataset.csv",
            "metadata": {"rows": 10000, "columns": 50}
        })

        # Data automatically persisted to configured file
        # Data survives process restarts and system reboots
        ```

        **Streaming Data Processing:**
        ```python
        from nanobrain.core import DataUnitStream

        # Create streaming data unit
        stream_unit = DataUnitStream.from_config('config/realtime_feed.yml')

        # Stream data in real-time
        async for data_chunk in stream_unit.stream():
            # Process each chunk as it arrives
            await process_chunk(data_chunk)

        # Or append data to stream
        await stream_unit.append({"timestamp": time.time(), "value": 42})
        ```

    **Data Flow and Communication:**

        **Inter-Component Communication:**
        * Data units serve as communication channels between components
        * Type-safe data exchange with validation and conversion
        * Event-driven notifications for data availability and changes
        * Shared data units for cross-component state management

        **Workflow Coordination:**
        * Data units coordinate workflow execution through data availability
        * Steps wait for required input data before processing
        * Output data units trigger downstream step activation
        * Data dependency tracking and resolution

        **Persistence Strategies:**
        * Configurable persistence backends (memory, file, database)
        * Automatic data backup and recovery mechanisms
        * Version control and change tracking for data evolution
        * Data compression and optimization for storage efficiency

    **Performance and Scalability:**

        **Memory Management:**
        * Intelligent caching with configurable cache sizes
        * Automatic memory cleanup and garbage collection
        * Memory usage monitoring and optimization
        * Large dataset handling with streaming and pagination

        **I/O Optimization:**
        * Asynchronous I/O operations for non-blocking access
        * Batch operations for improved throughput
        * Connection pooling for database and network operations
        * Compression and serialization optimization

        **Scalability Features:**
        * Distributed data units for cluster environments
        * Data sharding and partitioning strategies
        * Load balancing across storage backends
        * Horizontal scaling with data replication

    **Data Integrity and Validation:**

        **Type Safety:**
        * Strong typing with Pydantic schema validation
        * Automatic type conversion and normalization
        * Custom validation rules and constraints
        * Data format verification and error reporting

        **Consistency Guarantees:**
        * Atomic operations for data modifications
        * Transaction support for complex data updates
        * Conflict resolution for concurrent modifications
        * Data consistency checks and validation

        **Error Handling:**
        * Comprehensive error handling with detailed diagnostics
        * Data recovery mechanisms for corruption scenarios
        * Validation error reporting with correction suggestions
        * Graceful degradation for partial data availability

    **Security and Privacy:**

        **Data Protection:**
        * Encryption at rest and in transit for sensitive data
        * Access control and permission management
        * Data anonymization and privacy protection
        * Secure key management and rotation

        **Audit and Compliance:**
        * Comprehensive audit logging for data access and modifications
        * Data lineage tracking for compliance requirements
        * Privacy controls and data retention policies
        * Compliance reporting and data governance

    **Data Unit Lifecycle:**
        Data units follow a well-defined lifecycle:

        1. **Configuration Loading**: Parse and validate data unit configuration
        2. **Storage Backend Initialization**: Setup storage backend and connections
        3. **Schema Validation**: Validate data schemas and types
        4. **Event System Setup**: Register change listeners and event handlers
        5. **Data Loading**: Load existing data from persistent storage
        6. **Active State**: Ready for data operations and event processing
        7. **Data Operations**: Handle get, set, and stream operations
        8. **Cleanup**: Persist data and release resources

    **Advanced Features:**

        **Data Transformation:**
        * Automatic data format conversion and normalization
        * Custom transformation pipelines and processors
        * Data enrichment and annotation capabilities
        * Schema evolution and migration support

        **Monitoring and Analytics:**
        * Real-time data access and modification metrics
        * Performance monitoring and optimization recommendations
        * Data usage patterns and trend analysis
        * Capacity planning and resource optimization

        **Integration Capabilities:**
        * Database integration (PostgreSQL, MongoDB, Redis)
        * Cloud storage integration (S3, GCS, Azure Blob)
        * Message queue integration (RabbitMQ, Apache Kafka)
        * API integration for external data sources

    Attributes:
        name (str): Data unit identifier for logging and component coordination
        data (Any): Stored data content with type validation and serialization
        metadata (Dict[str, Any]): Rich metadata about data content and provenance
        persistent (bool): Whether data persists across component restarts
        cache_size (int): Maximum cache size for performance optimization
        encoding (str): Text encoding for string-based data units
        file_path (str, optional): File path for file-based data units
        change_listeners (List[Callable]): Registered listeners for data change events
        access_count (Dict): Statistics about data access patterns and frequency
        performance_metrics (Dict): Real-time performance and usage metrics

    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like DataUnitMemory, DataUnitFile, or
        DataUnitStream. All data units must be created using the from_config
        pattern with proper configuration files following framework patterns.

    Warning:
        Data units may consume significant memory or storage resources depending
        on data size and persistence requirements. Monitor resource usage and
        implement appropriate limits and cleanup mechanisms. Ensure proper
        data validation and security for sensitive information.

    See Also:
        * :class:`DataUnitMemory`: High-performance in-memory data storage
        * :class:`DataUnitFile`: File-based persistent data storage
        * :class:`DataUnitStream`: Real-time streaming data containers
        * :class:`DataUnitConfig`: Data unit configuration schema and validation
        * :mod:`nanobrain.library.infrastructure.data`: Specialized data unit implementations
        * :class:`TriggerBase`: Event trigger system that responds to data changes
    """

    COMPONENT_TYPE = "data_unit"
    REQUIRED_CONFIG_FIELDS = ['class_field']
    OPTIONAL_CONFIG_FIELDS = {
        'persistent': False,
        'cache_size': 1000,
        'file_path': None,
        'encoding': 'utf-8',
        'initial_value': None,
        'name': '',
        'description': ''
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return DataUnitConfig - ONLY method that differs from other components"""
        return DataUnitConfig

    @classmethod
    def extract_component_config(cls, config: DataUnitConfig) -> Dict[str, Any]:
        """Extract DataUnit configuration"""
        return {
            'class_path': config.class_path,
            'persistent': getattr(config, 'persistent', False),
            'cache_size': getattr(config, 'cache_size', 1000),
            'file_path': getattr(config, 'file_path', None),
            'encoding': getattr(config, 'encoding', 'utf-8'),
            'initial_value': getattr(config, 'initial_value', None),
            'name': getattr(config, 'name', ''),
            'description': getattr(config, 'description', '')
        }

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DataUnit dependencies"""
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False),
            'parent_scope': kwargs.get('parent_scope', 'global')
        }

    def _init_from_config(self, config: DataUnitConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnit with resolved dependencies"""
        self.config = config
        self.name = component_config.get('name') or self.__class__.__name__
        self._data: Any = None
        self._metadata: Dict[str, Any] = {}
        self._is_initialized = False
        self._lock = asyncio.Lock()

        # Initialize centralized logging system
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(
                self.name, category="data_units", debug_mode=dependencies.get('debug_mode', False))

            # Register with system log manager using scoped component name
            system_manager = get_system_log_manager()
            # Create hierarchical component name with parent scope
            parent_scope = dependencies.get('parent_scope', 'global')
            scoped_name = f"{parent_scope}.{self.name}" if parent_scope != 'global' else self.name
            system_manager.register_component("data_units", scoped_name, self, {
                "class_path": component_config['class_path'],
                "persistent": component_config['persistent'],
                "enable_logging": True,
                "parent_scope": parent_scope
            })
        else:
            self.nb_logger = None

        # Internal state tracking
        self._operation_count = 0
        self._last_operation = None
        self._creation_time = time.time()
        self._access_count = {"get": 0, "set": 0, "clear": 0}

        # Event-driven architecture: Change listeners for triggers
        self._change_listeners: List[Callable] = []

        # Automatic trigger management
        self._auto_input_triggers: Dict[str, Any] = {}
        self._auto_output_triggers: Dict[str, Any] = {}
        self._associated_steps: List[Any] = []
        self._associated_links: List[Any] = []
        self._trigger_creation_enabled = True

    # DataUnitBase inherits FromConfigBase.__init__ which prevents direct instantiation

    def register_change_listener(self, listener: Callable) -> None:
        """Register a change listener for event-driven triggers"""
        if listener not in self._change_listeners:
            self._change_listeners.append(listener)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(
                    f"Registered change listener for {self.name}")

    def unregister_change_listener(self, listener: Callable) -> None:
        """Unregister a change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(
                    f"Unregistered change listener for {self.name}")

    # ============================================================================
    # DATA ENCAPSULATION METHODS - PROPER OOP IMPLEMENTATION
    # ============================================================================

    def _get_internal_data(self) -> Any:
        """
        Get internal data with proper encapsulation.

        BRUTAL TRUTH: This replaces direct _data access throughout the framework.
        """
        return getattr(self, '_data', None)

    async def _set_internal_data(self, data: Any, operation: DataUnitEventType = DEFAULT_DATA_UNIT_EVENT) -> None:
        """
        Set internal data with proper encapsulation and event notification.

        BRUTAL TRUTH: This replaces direct _data manipulation and ensures
        proper change event notification.

        Args:
            data: Data to set
            operation: Type of operation being performed
        """
        print(f"🔗 BRUTAL TRUTH: _set_internal_data ENTRY for {self.name}")  # Force print
        if not self.is_initialized:
            await self.initialize()

        async with self._lock:
            old_data = self._get_internal_data()
            self._data = data
            self._metadata['last_updated'] = time.time()

            # Increment operation count
            self._operation_count = getattr(self, '_operation_count', 0) + 1

            # Create change event with proper enum-based operation
            change_event = {
                'data_unit_name': self.name,
                'operation': operation.value,  # Use enum value
                'old_data': old_data,
                'new_data': data,
                'timestamp': time.time(),
                'operation_count': self._operation_count
            }

            # Notify change listeners for event-driven execution
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🔗 BRUTAL TRUTH: Notifying {len(self._change_listeners)} change listeners for {self.name}")

            await self._notify_change_listeners(change_event)

    def _validate_data(self, data: Any) -> bool:
        """
        Validate data before setting.

        BRUTAL TRUTH: This provides a hook for subclasses to implement
        data validation without breaking encapsulation.
        """
        # Base implementation accepts any data
        # Subclasses can override for specific validation
        return True

    def _transform_data_on_set(self, data: Any) -> Any:
        """
        Transform data before setting.

        BRUTAL TRUTH: This provides a hook for subclasses to implement
        data transformation without breaking encapsulation.
        """
        # Base implementation returns data unchanged
        # Subclasses can override for specific transformations
        return data

    # ============================================================================
    # AUTOMATIC TRIGGER SYSTEM - NEW IMPLEMENTATION
    # ============================================================================

    @property
    def has_automatic_triggers(self) -> bool:
        """Check if this data unit has automatic triggers."""
        return len(self._auto_input_triggers) > 0 or len(self._auto_output_triggers) > 0

    @property
    def automatic_trigger_count(self) -> int:
        """Get total count of automatic triggers."""
        return len(self._auto_input_triggers) + len(self._auto_output_triggers)

    async def enable_automatic_triggers(self) -> None:
        """Enable automatic trigger creation for this data unit."""
        self._trigger_creation_enabled = True
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Enabled automatic triggers for {self.name}")

    async def disable_automatic_triggers(self) -> None:
        """Disable automatic trigger creation for this data unit."""
        self._trigger_creation_enabled = False

        # Stop existing automatic triggers
        for trigger in list(self._auto_input_triggers.values()):
            if hasattr(trigger, 'stop_monitoring'):
                await trigger.stop_monitoring()
        for trigger in list(self._auto_output_triggers.values()):
            if hasattr(trigger, 'stop_monitoring'):
                await trigger.stop_monitoring()

        self._auto_input_triggers.clear()
        self._auto_output_triggers.clear()

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"🚫 Disabled automatic triggers for {self.name}")

    async def register_as_input_for_step(self, step: Any, trigger_config: Dict[str, Any] = None) -> bool:
        """
        Automatically register this data unit as input for a step and create trigger.

        Args:
            step: The step that will be triggered when this data unit receives data
            trigger_config: Optional custom trigger configuration

        Returns:
            bool: True if trigger was created successfully
        """
        # Validate input parameters
        if step is None:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning("❌ Cannot register automatic trigger: step is None")
            return False

        if not hasattr(step, 'name'):
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning("❌ Cannot register automatic trigger: step has no name attribute")
            return False

        if not self._trigger_creation_enabled:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"⚠️ Trigger creation disabled for {self.name}")
            return False

        if step in self._associated_steps:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"✅ Step {step.name} already registered with {self.name}")
            return True  # Already registered

        try:
            self._associated_steps.append(step)

            # Create automatic input trigger
            trigger_id = f"auto_input_{step.name}_{self.name}"
            auto_trigger = await self._create_automatic_input_trigger(
                step, trigger_id, trigger_config or {}
            )

            if auto_trigger:
                self._auto_input_triggers[trigger_id] = auto_trigger
                await auto_trigger.start_monitoring()

                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(
                        f"✅ Created automatic input trigger {trigger_id} for step {step.name}")
                return True

        except Exception as e:
            step_name = getattr(step, 'name', 'unknown') if step is not None else 'None'
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"❌ Failed to create automatic input trigger for step {step_name}: {e}")
            return False

        return False

    async def register_as_output_for_step(self, step: Any) -> bool:
        """
        Register this data unit as output for a step.

        Args:
            step: The step that will write to this data unit

        Returns:
            bool: True if registration was successful
        """
        if step not in self._associated_steps:
            self._associated_steps.append(step)

            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(
                    f"Registered data unit {self.name} as output for step {step.name}")
        return True

    async def register_with_link(self, link: Any, role: str, trigger_config: Dict[str, Any] = None) -> bool:
        """
        Register this data unit with a link and create automatic triggers if needed.

        Args:
            link: The link to register with
            role: Either "source" or "target"
            trigger_config: Optional custom trigger configuration

        Returns:
            bool: True if registration was successful
        """
        # Validate input parameters
        if link is None:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning("❌ Cannot register with link: link is None")
            return False

        if not hasattr(link, 'name'):
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning("❌ Cannot register with link: link has no name attribute")
            return False

        if role not in ["source", "target"]:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.warning(f"❌ Invalid role '{role}': must be 'source' or 'target'")
            return False

        if not self._trigger_creation_enabled:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"⚠️ Trigger creation disabled for {self.name}")
            return False

        if link in self._associated_links:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"✅ Link {link.name} already registered with {self.name}")
            return True  # Already registered

        try:
            self._associated_links.append(link)

            if role == "source":
                # Create automatic output trigger for link activation
                trigger_id = f"auto_link_{link.name}_{self.name}"
                auto_trigger = await self._create_automatic_link_trigger(
                    link, trigger_id, trigger_config or {}
                )

                if auto_trigger:
                    self._auto_output_triggers[trigger_id] = auto_trigger
                    await auto_trigger.start_monitoring()

                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.info(
                            f"✅ Created automatic link trigger {trigger_id} for link {link.name}")
                    return True

        except Exception as e:
            link_name = getattr(link, 'name', 'unknown') if link is not None else 'None'
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"❌ Failed to register with link {link_name}: {e}")
            return False

        return True

    async def _create_automatic_input_trigger(self, step: Any, trigger_id: str,
                                            config: Dict[str, Any]) -> Any:
        """Create automatic trigger for step execution when this data unit receives data."""
        try:
            from .trigger import DataUnitChangeTrigger

            # Merge default config with custom config - FIXED: Use enum-based event types
            trigger_config = {
                'name': trigger_id,
                'trigger_type': 'data_updated',
                'data_unit': self,
                'event_type': validate_event_type(
                    config.get('event_type', DEFAULT_DATA_UNIT_EVENT),
                    DataUnitEventType
                ).value,  # Convert enum to string for backward compatibility
                'description': config.get('description',
                    f"Auto-generated trigger for {step.name} when {self.name} receives data")
            }
            trigger_config.update(config)

            # Create trigger instance using from_config pattern
            trigger = DataUnitChangeTrigger.from_config(trigger_config)

            # Bind trigger to step execution with error handling
            async def safe_step_execution(trigger_event):
                """Safely execute step when trigger fires."""
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🔥 BRUTAL TRUTH: safe_step_execution called for step {step.name}")
                try:
                    if hasattr(step, '_execute_on_trigger'):
                        logger.info(f"🔥 BRUTAL TRUTH: Calling step._execute_on_trigger for {step.name}")
                        await step._execute_on_trigger(trigger_event)
                        logger.info(f"🔥 BRUTAL TRUTH: step._execute_on_trigger completed for {step.name}")
                    elif hasattr(step, 'execute'):
                        logger.info(f"🔥 BRUTAL TRUTH: Calling step.execute for {step.name}")
                        await step.execute()
                        logger.info(f"🔥 BRUTAL TRUTH: step.execute completed for {step.name}")
                    else:
                        logger.warning(f"⚠️ Step {step.name} has no execute method for automatic trigger")
                except Exception as e:
                    logger.error(f"❌ Automatic trigger execution failed for step {step.name}: {e}")
                    logger.error(f"🔥 BRUTAL TRUTH: Exception type: {type(e).__name__}")
                    import traceback
                    logger.error(f"🔥 BRUTAL TRUTH: Traceback: {traceback.format_exc()}")
                    # CRITICAL: Re-raise the exception so we can see what's failing
                    raise

            trigger.bind_action(safe_step_execution)
            return trigger

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create automatic input trigger: {e}")
            return None

    async def _create_automatic_link_trigger(self, link: Any, trigger_id: str,
                                           config: Dict[str, Any]) -> Any:
        """Create automatic trigger for link activation when this data unit is updated."""
        try:
            from .trigger import DataUnitChangeTrigger

            # Merge default config with custom config - FIXED: Use enum-based event types
            trigger_config = {
                'name': trigger_id,
                'trigger_type': 'data_updated',
                'data_unit': self,
                'event_type': validate_event_type(
                    config.get('event_type', DEFAULT_DATA_UNIT_EVENT),
                    DataUnitEventType
                ).value,  # Convert enum to string for backward compatibility
                'description': config.get('description',
                    f"Auto-generated trigger for link {link.name} when {self.name} is updated")
            }
            trigger_config.update(config)

            # Create trigger instance
            trigger = DataUnitChangeTrigger.from_config(trigger_config)

            # Bind trigger to link transfer with error handling
            async def safe_link_activation(trigger_event):
                """Safely activate link when trigger fires."""
                try:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.info(f"🔗 BRUTAL TRUTH: Link trigger fired for {self.name} -> {getattr(link, 'name', 'unknown')}")
                        self.nb_logger.info(f"🔗 Trigger event: {trigger_event}")

                    # Check if link should be activated
                    should_activate = await self._should_activate_link(link)
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.info(f"🔗 Should activate: {should_activate}")

                    if not should_activate:
                        return

                    source_data = await self.get()
                    if source_data is not None:
                        if hasattr(link, 'transfer'):
                            if self.enable_logging and self.nb_logger:
                                self.nb_logger.info(f"🔗 BRUTAL TRUTH: Transferring data via link {getattr(link, 'name', 'unknown')}")

                            await link.transfer(source_data)

                            if self.enable_logging and self.nb_logger:
                                self.nb_logger.info(
                                    f"🔗 BRUTAL TRUTH: Link {getattr(link, 'name', 'unknown')} activated successfully with {type(source_data).__name__}")
                        else:
                            if self.enable_logging and self.nb_logger:
                                self.nb_logger.warning(
                                    f"⚠️ Link {getattr(link, 'name', 'unknown')} has no transfer method")
                    else:
                        if self.enable_logging and self.nb_logger:
                            self.nb_logger.warning(
                                f"⚠️ Link {getattr(link, 'name', 'unknown')} not activated - no data available")

                except Exception as e:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.error(
                            f"❌ Automatic link activation failed for {link.name}: {e}")

            trigger.bind_action(safe_link_activation)
            return trigger

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Failed to create automatic link trigger: {e}")
            return None

    async def _should_activate_link(self, link: Any) -> bool:
        """Check if link should be activated based on current conditions."""
        try:
            # BRUTAL TRUTH: Add debug logging to catch link activation issues
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"🔍 Checking link activation for {getattr(link, 'name', 'unknown')}")

            # Check if link is active
            if hasattr(link, 'is_active'):
                is_active = link.is_active
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(f"🔍 Link is_active: {is_active}")
                if not is_active:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.warning(f"⚠️ Link {getattr(link, 'name', 'unknown')} not activated - link is not active")
                    return False

            # Check if target is ready
            if hasattr(link, 'target') and hasattr(link.target, 'is_ready'):
                target_ready = await link.target.is_ready()
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(f"🔍 Target ready: {target_ready}")
                if not target_ready:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.warning(f"⚠️ Link {getattr(link, 'name', 'unknown')} not activated - target not ready")
                    return False

            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"✅ Link {getattr(link, 'name', 'unknown')} activation approved")
            return True
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Error checking link activation conditions: {e}")
            return False

    async def cleanup_automatic_triggers(self) -> None:
        """Clean up all automatic triggers when data unit is destroyed."""
        try:
            # Stop and clean up input triggers
            for trigger_id, trigger in list(self._auto_input_triggers.items()):
                if hasattr(trigger, 'stop_monitoring'):
                    await trigger.stop_monitoring()
                del self._auto_input_triggers[trigger_id]

            # Stop and clean up output triggers
            for trigger_id, trigger in list(self._auto_output_triggers.items()):
                if hasattr(trigger, 'stop_monitoring'):
                    await trigger.stop_monitoring()
                del self._auto_output_triggers[trigger_id]

            # Clear associations
            self._associated_steps.clear()
            self._associated_links.clear()

            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🧹 Cleaned up automatic triggers for {self.name}")

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Error cleaning up automatic triggers: {e}")

    async def _notify_change_listeners(self, change_event: Dict[str, Any]) -> None:
        """Notify all registered change listeners of data unit changes using async execution."""
        print(f"🔗 BRUTAL TRUTH: _notify_change_listeners ENTRY for {self.name}")  # Force print to bypass logging

        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"🔗 BRUTAL TRUTH: _notify_change_listeners called for {self.name} with {len(self._change_listeners)} listeners")

        if self._change_listeners:
            # ✅ DEADLOCK FIX: Check for actual data changes to prevent infinite loops
            old_data = change_event.get('old_data')
            new_data = change_event.get('new_data')

            # Skip notification if data hasn't actually changed
            if self._data_unchanged(old_data, new_data):
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(
                        f"Skipping change notification for {self.name} - data unchanged",
                        operation="change_notification_skipped"
                    )
                return

            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🔗 BRUTAL TRUTH: About to notify {len(self._change_listeners)} change listeners for {self.name}")

            # Import AsyncTriggerExecutor here to avoid circular imports
            from .trigger import AsyncTriggerExecutor

            try:
                # Get async executor instance
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(f"🔗 BRUTAL TRUTH: Getting AsyncTriggerExecutor instance for {self.name}")

                async_executor = await AsyncTriggerExecutor.get_instance()

                if self.enable_logging and self.nb_logger:
                    self.nb_logger.info(f"🔗 BRUTAL TRUTH: Got AsyncTriggerExecutor, executing {len(self._change_listeners)} listeners")

                # Execute all listeners asynchronously and AWAIT completion
                tasks = []
                for i, listener in enumerate(self._change_listeners):
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.info(f"🔗 BRUTAL TRUTH: Processing listener {i+1}/{len(self._change_listeners)}: {getattr(listener, '__name__', str(listener))}")
                    try:
                        # Create async task for listener execution
                        task = asyncio.create_task(
                            self._execute_listener_async(
                                listener, change_event)
                        )

                        # Add to executor's background tasks for tracking
                        async_executor.background_tasks.add(task)
                        task.add_done_callback(
                            async_executor.background_tasks.discard)

                        # BRUTAL TRUTH: Collect tasks to await them
                        tasks.append(task)

                        # Log async listener execution
                        if self.enable_logging and self.nb_logger:
                            self.nb_logger.info(
                                f"🚀 BRUTAL TRUTH: Async listener execution initiated for {self.name}",
                                operation="async_listener_start",
                                listener_name=getattr(
                                    listener, '__name__', str(listener))
                            )

                    except Exception as e:
                        if self.enable_logging and self.nb_logger:
                            self.nb_logger.error(
                                f"Failed to initiate async listener for {self.name}: {e}",
                                operation="async_listener_error",
                                error=str(e)
                            )

                # BRUTAL TRUTH: Wait for ALL listener tasks to complete
                if tasks:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.info(f"🔗 BRUTAL TRUTH: Awaiting {len(tasks)} listener tasks for {self.name}")

                    await asyncio.gather(*tasks, return_exceptions=True)

                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.info(f"🔗 BRUTAL TRUTH: All {len(tasks)} listener tasks completed for {self.name}")

            except Exception as e:
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.error(
                        f"Failed to get AsyncTriggerExecutor for {self.name}: {e}",
                        operation="async_executor_error",
                        error=str(e)
                    )

    async def _execute_listener_async(self, listener: Callable, change_event: Dict[str, Any]) -> None:
        """Execute listener in async context without blocking DataUnit operations."""
        try:
            # Yield control to allow DataUnit.set() to complete
            await asyncio.sleep(0)

            # Execute the listener
            if asyncio.iscoroutinefunction(listener):
                await listener(change_event)
            else:
                listener(change_event)

            # Log successful completion
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(
                    f"✅ BRUTAL TRUTH: Async listener execution completed for {self.name}",
                    operation="async_listener_complete",
                    listener_name=getattr(listener, '__name__', str(listener))
                )

        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(
                    f"❌ Async listener execution error for {self.name}: {e}",
                    operation="async_listener_execution_error",
                    listener_name=getattr(listener, '__name__', str(listener)),
                    error=str(e)
                )

    @abstractmethod
    async def get(self) -> Any:
        """Get data from the unit."""
        pass

    @abstractmethod
    async def set(self, data: Any) -> None:
        """Set data in the unit."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear data from the unit."""
        pass

    def _data_unchanged(self, old_data: Any, new_data: Any) -> bool:
        """Check if data has actually changed to prevent unnecessary triggers."""
        try:
            # Handle None values
            if old_data is None and new_data is None:
                return True
            if old_data is None or new_data is None:
                return False

            # ✅ CRITICAL FIX: Handle DataUnit objects (prevent self-triggering)
            if hasattr(old_data, '__class__') and hasattr(new_data, '__class__'):
                if 'DataUnit' in old_data.__class__.__name__ and 'DataUnit' in new_data.__class__.__name__:
                    # DataUnit objects should be compared by their data, not the objects themselves
                    if hasattr(old_data, '_data') and hasattr(new_data, '_data'):
                        return self._data_unchanged(old_data._data, new_data._data)
                    return old_data is new_data

            # Handle basic types
            if type(old_data) != type(new_data):
                return False

            # For dictionaries, compare contents
            if isinstance(old_data, dict) and isinstance(new_data, dict):
                return old_data == new_data

            # For lists, compare contents
            if isinstance(old_data, list) and isinstance(new_data, list):
                return old_data == new_data

            # For other types, use equality comparison
            return old_data == new_data

        except Exception:
            # If comparison fails, assume data has changed
            return False

    async def initialize(self) -> None:
        """Initialize the data unit."""
        if not self._is_initialized:
            self._is_initialized = True
            self._last_operation = "initialize"
            self._operation_count += 1
            logger.debug(f"DataUnit {self.name} initialized")

            # Log initialization with comprehensive state
            if self.enable_logging and self.nb_logger:
                self.nb_logger.log_data_unit_operation(
                    operation="initialize",
                    data_unit_name=self.name,
                    metadata={
                        "data_unit_type": type(self).__name__,
                        "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else str(self.config),
                        "creation_time": self._creation_time,
                        "internal_state": self._get_internal_state()
                    }
                )

    async def shutdown(self) -> None:
        """Shutdown the data unit."""
        # Log shutdown with final state
        if self.enable_logging and self.nb_logger:
            uptime = time.time() - self._creation_time
            self.nb_logger.log_data_unit_operation(
                operation="shutdown",
                data_unit_name=self.name,
                metadata={
                    "data_unit_type": type(self).__name__,
                    "final_metadata": self._metadata,
                    "uptime_seconds": uptime,
                    "total_operations": self._operation_count,
                    "access_counts": self._access_count.copy(),
                    "final_state": self._get_internal_state()
                }
            )

        await self.clear()
        self._is_initialized = False
        logger.debug(f"DataUnit {self.name} shutdown")

    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        return {
            "is_initialized": self._is_initialized,
            "operation_count": self._operation_count,
            "last_operation": self._last_operation,
            "access_counts": self._access_count.copy(),
            "metadata_keys": list(self._metadata.keys()),
            "has_data": self._data is not None,
            "data_type": type(self._data).__name__ if self._data is not None else "None",
            "uptime_seconds": time.time() - self._creation_time
        }

    @property
    def is_initialized(self) -> bool:
        """Check if data unit is initialized."""
        return self._is_initialized

    def has_data(self) -> bool:
        """Check if data unit contains data."""
        return hasattr(self, '_data') and self._data is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata."""
        return self._metadata.copy()

    async def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata."""
        async with self._lock:
            old_value = self._metadata.get(key)
            self._metadata[key] = value

            # Log metadata change
            if self.enable_logging and self.nb_logger:
                self.nb_logger.log_data_unit_operation(
                    operation="set_metadata",
                    data_unit_name=self.name,
                    metadata={
                        "key": key,
                        "old_value": old_value,
                        "new_value": value,
                        "internal_state": self._get_internal_state()
                    }
                )

    async def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        value = self._metadata.get(key, default)

        # Log metadata access
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Metadata accessed: {key}",
                                 key=key,
                                 value=value,
                                 data_unit=self.name)

        return value

    async def read(self) -> Any:
        """Read data from the unit (alias for get)."""
        self._access_count["get"] += 1
        self._last_operation = "read"
        self._operation_count += 1

        data = await self.get()

        # Log the read operation with state
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_data_unit_operation(
                operation="read",
                data_unit_name=self.name,
                data=data,
                metadata={
                    "data_unit_type": type(self).__name__,
                    "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else str(self.config),
                    "metadata": self._metadata.copy(),
                    "internal_state": self._get_internal_state()
                }
            )

        return data

    async def write(self, data: Any) -> None:
        """Write data to the unit (alias for set)."""
        self._access_count["set"] += 1
        self._last_operation = "write"
        self._operation_count += 1

        # Log the write operation with state change
        if self.enable_logging and self.nb_logger:
            old_data_type = type(
                self._data).__name__ if self._data is not None else "None"
            new_data_type = type(data).__name__ if data is not None else "None"

            self.nb_logger.log_data_unit_operation(
                operation="write",
                data_unit_name=self.name,
                data=data,
                metadata={
                    "data_unit_type": type(self).__name__,
                    "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else str(self.config),
                    "metadata": self._metadata.copy(),
                    "old_data_type": old_data_type,
                    "new_data_type": new_data_type,
                    "state_change": {
                        "before": self._get_internal_state(),
                    }
                }
            )

        # Store old data for change event
        old_data = self._data if hasattr(self, '_data') else None

        await self.set(data)

        # Notify change listeners (EVENT-DRIVEN ARCHITECTURE)
        change_event = {
            'data_unit_name': self.name,
            'operation': 'write',
            'old_data': old_data,
            'new_data': data,
            'timestamp': time.time(),
            'operation_count': self._operation_count
        }
        await self._notify_change_listeners(change_event)

        # Log state after change
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Write completed for {self.name}",
                                 operation="write_complete",
                                 internal_state=self._get_internal_state())


class DataUnitMemory(DataUnitBase):
    """
    In-memory data unit for fast access.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, DataUnitConfig, Dict[str, Any]], **kwargs) -> 'DataUnitMemory':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, DataUnitConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized DataUnitMemory instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per DataUnit rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to DataUnitConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = DataUnitConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create DataUnitConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config)
            finally:
                DataUnitConfig._allow_direct_instantiation = False
        elif isinstance(config, DataUnitConfig):
            # Already a DataUnitConfig object
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
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config_dict)
            finally:
                DataUnitConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    def _init_from_config(self, config: DataUnitConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitMemory with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._data = None

    async def get(self) -> Any:
        """
        Get data from memory with proper encapsulation.

        BRUTAL TRUTH: This now uses proper encapsulation instead of
        direct _data access.
        """
        if not self.is_initialized:
            await self.initialize()
        return self._get_internal_data()

    async def set(self, data: Any) -> None:
        """
        Set data in memory with proper encapsulation.

        BRUTAL TRUTH: This now uses proper encapsulation methods instead of
        direct _data manipulation that was causing framework bugs.
        """
        print(f"🔗 BRUTAL TRUTH: DataUnitMemory.set() ENTRY for {self.name}")  # Force print

        # Validate data before setting
        if not self._validate_data(data):
            raise ValueError(f"Invalid data for {self.name}: {data}")

        print(f"🔗 BRUTAL TRUTH: DataUnitMemory.set() validation passed for {self.name}")  # Force print

        # Transform data if needed
        transformed_data = self._transform_data_on_set(data)

        print(f"🔗 BRUTAL TRUTH: DataUnitMemory.set() about to call _set_internal_data for {self.name}")  # Force print

        # Use proper encapsulated setter with enum-based event type
        await self._set_internal_data(transformed_data, DataUnitEventType.SET)

    async def clear(self) -> None:
        """
        Clear data from memory with proper encapsulation.

        BRUTAL TRUTH: This now uses proper encapsulation instead of
        direct _data manipulation.
        """
        # Use proper encapsulated setter with enum-based event type
        await self._set_internal_data(None, DataUnitEventType.CLEAR)

        # Clear metadata
        async with self._lock:
            self._metadata.clear()

        # Log after clearing
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Clear completed for {self.name}",
                                 operation="clear_complete",
                                 internal_state=self._get_internal_state())


class DataUnitFile(DataUnitBase):
    """
    File-based data unit for persistent storage.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, DataUnitConfig, Dict[str, Any]], **kwargs) -> 'DataUnitFile':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, DataUnitConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized DataUnitFile instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per DataUnit rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to DataUnitConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = DataUnitConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create DataUnitConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config)
            finally:
                DataUnitConfig._allow_direct_instantiation = False
        elif isinstance(config, DataUnitConfig):
            # Already a DataUnitConfig object
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
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config_dict)
            finally:
                DataUnitConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DataUnitFile dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'file_path': kwargs.get('file_path') or component_config.get('file_path')
        }

    def _init_from_config(self, config: DataUnitConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitFile with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        file_path = dependencies.get(
            'file_path') or component_config.get('file_path')
        if not file_path:
            raise ComponentConfigurationError(
                "DataUnitFile requires file_path")
        self.file_path = Path(file_path)

    async def get(self) -> Any:
        """Get data from file."""
        if not self.is_initialized:
            await self.initialize()

        if not self.file_path.exists():
            return None

        try:
            async with self._lock:
                # NON-BLOCKING file read using aiofiles
                async with aiofiles.open(self.file_path, 'r', encoding=self.config.encoding) as f:
                    content = await f.read()

                # Try to parse as JSON if possible
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content

        except Exception as e:
            logger.error(
                f"Error reading file {self.file_path}: {e}", exc_info=True)
            raise

    async def set(self, data: Any) -> None:
        """Set data to file."""
        if not self.is_initialized:
            await self.initialize()

        try:
            async with self._lock:
                # Get old data for change event (NON-BLOCKING)
                old_data = None
                if self.file_path.exists():
                    try:
                        async with aiofiles.open(self.file_path, 'r', encoding=self.config.encoding) as f:
                            old_content = await f.read()
                        try:
                            old_data = json.loads(old_content)
                        except json.JSONDecodeError:
                            old_data = old_content
                    except:
                        old_data = None

                # Ensure parent directory exists
                self.file_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert data to string
                if isinstance(data, (dict, list)):
                    content = json.dumps(data, indent=2)
                else:
                    content = str(data)

                # NON-BLOCKING file write using aiofiles
                async with aiofiles.open(self.file_path, 'w', encoding=self.config.encoding) as f:
                    await f.write(content)
                self._metadata['last_updated'] = time.time()

                # Notify change listeners for event-driven execution
                change_event = {
                    'data_unit_name': self.name,
                    'operation': 'set',
                    'old_data': old_data,
                    'new_data': data,
                    'timestamp': time.time(),
                    'operation_count': self._operation_count
                }
                await self._notify_change_listeners(change_event)

        except Exception as e:
            logger.error(
                f"Error writing file {self.file_path}: {e}", exc_info=True)
            raise

    async def clear(self) -> None:
        """Clear file data."""
        async with self._lock:
            if self.file_path.exists():
                self.file_path.unlink()
            self._metadata.clear()


class DataUnitString(DataUnitBase):
    """
    String-based data unit for text data.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, DataUnitConfig, Dict[str, Any]], **kwargs) -> 'DataUnitString':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, DataUnitConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized DataUnitString instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per DataUnit rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to DataUnitConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = DataUnitConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create DataUnitConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config)
            finally:
                DataUnitConfig._allow_direct_instantiation = False
        elif isinstance(config, DataUnitConfig):
            # Already a DataUnitConfig object
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
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config_dict)
            finally:
                DataUnitConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DataUnitString dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'initial_value': kwargs.get('initial_value', component_config.get('initial_value', ''))
        }

    def _init_from_config(self, config: DataUnitConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitString with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._data = dependencies.get('initial_value', '')

    async def get(self) -> str:
        """Get string data."""
        if not self.is_initialized:
            await self.initialize()
        return self._data or ""

    async def set(self, data: Any) -> None:
        """Set string data."""
        if not self.is_initialized:
            await self.initialize()
        async with self._lock:
            old_data = self._data
            self._data = str(data) if data is not None else ""
            self._metadata['last_updated'] = time.time()

            # Notify change listeners for event-driven execution
            change_event = {
                'data_unit_name': self.name,
                'operation': 'set',
                'old_data': old_data,
                'new_data': self._data,
                'timestamp': time.time(),
                'operation_count': self._operation_count
            }
            await self._notify_change_listeners(change_event)

    async def append(self, data: str) -> None:
        """Append to string data."""
        async with self._lock:
            current = await self.get()
            await self.set(current + str(data))

    async def clear(self) -> None:
        """Clear string data."""
        async with self._lock:
            self._data = ""
            self._metadata.clear()


class DataUnitStream(DataUnitBase):
    """
    Stream-based data unit for continuous data flow.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, DataUnitConfig, Dict[str, Any]], **kwargs) -> 'DataUnitStream':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, DataUnitConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized DataUnitStream instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per DataUnit rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to DataUnitConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = DataUnitConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create DataUnitConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config)
            finally:
                DataUnitConfig._allow_direct_instantiation = False
        elif isinstance(config, DataUnitConfig):
            # Already a DataUnitConfig object
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
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config_dict)
            finally:
                DataUnitConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    def _init_from_config(self, config: DataUnitConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitStream with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._queue: Optional[asyncio.Queue] = None
        self._subscribers: List[asyncio.Queue] = []

    async def initialize(self) -> None:
        """Initialize the stream."""
        if not self._is_initialized:
            self._queue = asyncio.Queue(maxsize=self.config.cache_size)
            await super().initialize()

    async def get(self) -> Any:
        """Get next item from stream."""
        if not self.is_initialized:
            await self.initialize()
        return await self._queue.get()

    async def set(self, data: Any) -> None:
        """Add data to stream."""
        if not self.is_initialized:
            await self.initialize()

        # Add to main queue
        try:
            await self._queue.put(data)
        except asyncio.QueueFull:
            # Remove oldest item and add new one
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            await self._queue.put(data)

        # Notify subscribers
        for subscriber_queue in self._subscribers:
            try:
                await subscriber_queue.put(data)
            except asyncio.QueueFull:
                # Skip if subscriber queue is full
                pass

    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to stream updates."""
        if not self.is_initialized:
            await self.initialize()
        subscriber_queue = asyncio.Queue(maxsize=self.config.cache_size)
        self._subscribers.append(subscriber_queue)
        return subscriber_queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from stream updates."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def clear(self) -> None:
        """Clear stream data."""
        if self._queue:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Clear subscriber queues
        for subscriber_queue in self._subscribers:
            while not subscriber_queue.empty():
                try:
                    subscriber_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._metadata.clear()


# ---------------------------------------------------------------------------
# G3 — DataUnitProxyRef (added 2026-05-09)
# ---------------------------------------------------------------------------
#
# Per `apecx-mcp-integration/docs/nanobrain_capability_gaps.md G3`: a DataUnit
# whose payload is a ProxyStore reference rather than the bytes themselves.
# Producers write to the ProxyStore and the data unit holds only the key.
# Consumers can resolve the key on demand (`.get()`), pass the proxy onward
# without materializing (`.as_proxy()`), or read the key directly (`.key()`).
#
# The change-event payload is the KEY, not the bytes — so
# AllDataReceivedTrigger fires on key-set, not on bytes-materialization.
# This is necessary for HPC-scale tool I/O where multi-GB payloads cannot
# ride a Python dict between steps.
#
# Equality + hashing (G3 spec): two DataUnitProxyRef instances are equal iff
# their (namespace, key) tuples are equal. metadata is descriptive, not
# identity-bearing.
#
# proxystore is an OPTIONAL dependency: a workflow that doesn't use
# DataUnitProxyRef does not need it installed. The ImportError is deferred
# to first use, with a clear message.
# ---------------------------------------------------------------------------

# Lazy-import marker — populated on first use.
_PROXYSTORE_IMPORT_ERROR: Optional[ImportError] = None


def _import_proxystore():
    """Lazy import of proxystore primitives. Returns the (Store, register_store,
    get_store, FileConnector, RedisConnector) tuple. RedisConnector may be None
    if the redis extra is not installed.

    Raises ComponentConfigurationError with a clear remediation hint if
    proxystore itself is missing.
    """
    global _PROXYSTORE_IMPORT_ERROR
    try:
        from proxystore.store import Store, register_store, get_store
        from proxystore.connectors.file import FileConnector
    except ImportError as e:
        _PROXYSTORE_IMPORT_ERROR = e
        raise ComponentConfigurationError(
            f"FAIL-FAST: DataUnitProxyRef requires proxystore. "
            f"Install with: pip install proxystore. "
            f"Original ImportError: {e}"
        ) from e
    try:
        from proxystore.connectors.redis import RedisConnector  # type: ignore
    except ImportError:
        RedisConnector = None  # type: ignore
    return Store, register_store, get_store, FileConnector, RedisConnector


class DataUnitProxyRef(DataUnitBase):
    """DataUnit whose payload is a ProxyStore reference.

    Guarantees per `nanobrain_capability_gaps.md G3`:

    - ``set(value)`` writes to the configured ProxyStore and keeps only the
      key. The trigger cascade fires on key-set (the change-event payload
      is the key string, not the bytes).
    - ``get()`` materializes lazily — one round-trip on first call;
      subsequent calls reuse the resolved value.
    - ``as_proxy()`` returns the proxy for fan-out without materialization.
    - ``key()`` returns the raw key string (for provenance recording).
    - ``namespace()`` returns the configured namespace prefix.
    - ``__eq__`` / ``__hash__`` use the ``(namespace, key)`` tuple.

    The ``proxystore_connector`` config field selects the backend:
    ``file`` (filesystem-backed; ideal for tests + single-host runs) or
    ``redis`` (multi-host, multi-tenant; required for HPC-scale).
    The full connector list will grow as the framework adds support
    (per the gap proposal: also ``globus``, ``endpoint``).
    """

    @classmethod
    def from_config(
        cls,
        config: Union[str, Path, DataUnitConfig, Dict[str, Any]],
        **kwargs,
    ) -> 'DataUnitProxyRef':
        """Standard from_config path mirroring DataUnitMemory."""
        nb_logger = get_logger(f"{cls.__name__}.from_config")
        nb_logger.info(f"Creating {cls.__name__} from configuration")

        # Normalize to DataUnitConfig (same shape as DataUnitMemory).
        if isinstance(config, (str, Path)):
            config_object = DataUnitConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            try:
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config)
            finally:
                DataUnitConfig._allow_direct_instantiation = False
        elif isinstance(config, DataUnitConfig):
            config_object = config
        else:
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            try:
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config_dict)
            finally:
                DataUnitConfig._allow_direct_instantiation = False

        cls.validate_config_schema(config_object)
        component_config = cls.extract_component_config(config_object)
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        instance = cls.create_instance(
            config_object, component_config, dependencies)
        instance._post_config_initialization()

        nb_logger.info(f"Successfully created {cls.__name__}")
        return instance

    @classmethod
    def extract_component_config(cls, config: DataUnitConfig) -> Dict[str, Any]:
        """Pull out the proxystore-specific fields plus the standard ones."""
        base = super().extract_component_config(config)
        base.update({
            "proxystore_connector": getattr(config, "proxystore_connector", None),
            "proxystore_store_name": getattr(config, "proxystore_store_name", None),
            "proxystore_store_dir": getattr(config, "proxystore_store_dir", None),
            "proxystore_redis_addr": getattr(config, "proxystore_redis_addr", None),
            "proxystore_namespace_prefix": getattr(
                config, "proxystore_namespace_prefix", None),
            "proxystore_metadata_mime": getattr(
                config, "proxystore_metadata_mime", None),
            "proxystore_metadata_max_size_bytes": getattr(
                config, "proxystore_metadata_max_size_bytes", None),
        })
        return base

    def _init_from_config(
        self,
        config: DataUnitConfig,
        component_config: Dict[str, Any],
        dependencies: Dict[str, Any],
    ) -> None:
        super()._init_from_config(config, component_config, dependencies)

        connector_kind = component_config.get("proxystore_connector")
        if not connector_kind:
            raise ComponentConfigurationError(
                f"FAIL-FAST: DataUnitProxyRef {self.name!r} requires "
                f"proxystore_connector ('file' | 'redis' | 'globus' | 'endpoint')"
            )

        store_name = component_config.get("proxystore_store_name") or self.name
        if not store_name:
            raise ComponentConfigurationError(
                f"FAIL-FAST: DataUnitProxyRef {self.name!r} requires "
                f"proxystore_store_name (or a non-empty data unit name)"
            )

        # Build connector + store — lazy import keeps proxystore optional.
        Store, register_store, get_store, FileConnector, RedisConnector = _import_proxystore()

        if connector_kind == "file":
            store_dir = component_config.get("proxystore_store_dir")
            if not store_dir:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: DataUnitProxyRef {self.name!r} with "
                    f"proxystore_connector='file' requires proxystore_store_dir"
                )
            connector = FileConnector(store_dir)
        elif connector_kind == "redis":
            if RedisConnector is None:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: DataUnitProxyRef {self.name!r} with "
                    f"proxystore_connector='redis' requires the proxystore[redis] "
                    f"extra. Install with: pip install 'proxystore[redis]'"
                )
            redis_addr = component_config.get("proxystore_redis_addr")
            if not redis_addr or ":" not in redis_addr:
                raise ComponentConfigurationError(
                    f"FAIL-FAST: DataUnitProxyRef {self.name!r} with "
                    f"proxystore_connector='redis' requires proxystore_redis_addr "
                    f"in 'host:port' form"
                )
            host, port = redis_addr.rsplit(":", 1)
            connector = RedisConnector(hostname=host, port=int(port))
        else:
            # globus / endpoint deferred to future tasks.
            raise ComponentConfigurationError(
                f"FAIL-FAST: DataUnitProxyRef {self.name!r} connector "
                f"{connector_kind!r} not yet supported "
                f"(supported: 'file', 'redis')"
            )

        # Idempotent registration: get_store returns the existing one if any.
        existing = get_store(store_name)
        if existing is None:
            store = Store(store_name, connector)
            register_store(store)
        else:
            store = existing
        self._store = store
        self._connector_kind = connector_kind

        # Namespace prefix per G13 (multi-tenant). Captured for
        # __eq__/__hash__ identity.
        self._namespace_prefix = component_config.get(
            "proxystore_namespace_prefix") or ""

        # Metadata — descriptive only, not identity-bearing.
        self._metadata_mime = component_config.get("proxystore_metadata_mime")
        self._metadata_max_size_bytes = component_config.get(
            "proxystore_metadata_max_size_bytes")

        # Reference state — populated at first set().
        self._key: Optional[str] = None
        self._materialized_value: Any = None
        self._materialized_for_key: Optional[str] = None

    def key(self) -> Optional[str]:
        """Return the raw key string (or None if no value has been set yet)."""
        return self._key

    def namespace(self) -> str:
        """Return the namespace prefix used for key construction (G13)."""
        return self._namespace_prefix

    def as_proxy(self) -> Any:
        """Return a Proxy<T> over the current key for fan-out.

        Materialization is lazy on the proxy: attribute access materializes;
        passing the proxy through another step does not.
        """
        if self._key is None:
            raise ComponentConfigurationError(
                f"FAIL-FAST: DataUnitProxyRef {self.name!r} has no value yet "
                f"(set() has not been called)"
            )
        return self._store.proxy_from_key(self._key)

    async def set(self, value: Any) -> None:
        """Write ``value`` to the ProxyStore and keep only the key.

        The change-event payload is the **key** (a typed proxystore Key
        object — FileKey/RedisKey/etc — NOT a string), so
        AllDataReceivedTrigger fires on key-set, before any consumer
        materializes the bytes.

        Note on namespacing (G13): the configured ``proxystore_namespace_prefix``
        is tracked SEPARATELY from the proxystore Key (which is a typed
        opaque value owned by proxystore). The prefix is used for equality,
        hashing, and provenance recording — NOT for prefixing the Key
        itself, because proxystore Keys are not strings. Multi-tenant
        isolation at the storage layer requires either a per-tenant Store
        name or a per-tenant FileConnector path; the framework-level
        runtime work for that is gap G13's full implementation.
        """
        if not self._validate_data(value):
            raise ValueError(f"Invalid data for {self.name}: {value!r}")

        # put() returns a typed Key object (FileKey, RedisKey, etc).
        # We DO NOT mangle it — we preserve the typed object as-is.
        proxy_key = self._store.put(value)

        self._key = proxy_key
        self._materialized_value = value
        self._materialized_for_key = proxy_key

        # Fire the change event with the KEY as payload (G3 contract).
        # Downstream consumers see the typed Key and can call store.get(key).
        await self._set_internal_data(proxy_key, DataUnitEventType.SET)

    async def get(self) -> Any:
        """Resolve the key to its value. One round-trip on first call;
        cached thereafter for the lifetime of the current key.
        """
        if not self.is_initialized:
            await self.initialize()
        if self._key is None:
            return None  # never set; matches DataUnitMemory's pre-set semantics

        # If we already materialized for this key, reuse the cached value.
        # Equality is delegated to the proxystore Key's own __eq__.
        if self._materialized_for_key == self._key:
            return self._materialized_value

        # Real round-trip through the store. This is the cross-process path:
        # a downstream consumer in a different process gets the Key from
        # the trigger payload, instantiates its OWN Store with the same
        # connector config, and calls .get(key) — that path is exercised
        # in tests/integration/test_proxy_ref_redis.py.
        value = self._store.get(self._key)
        self._materialized_value = value
        self._materialized_for_key = self._key
        return value

    async def clear(self) -> None:
        """Drop the local reference. Does NOT delete the underlying store
        entry — the store's eviction policy is workflow-level (per the gap
        proposal's ownership boundary)."""
        self._key = None
        self._materialized_value = None
        self._materialized_for_key = None
        await self._set_internal_data(None, DataUnitEventType.CLEAR)
        async with self._lock:
            self._metadata.clear()

    # G3 spec — equality + hashing on (namespace, key). Two refs to the
    # same key with different mime hints are still the same ref.

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataUnitProxyRef):
            return NotImplemented
        return (self._namespace_prefix, self._key) == (
            other._namespace_prefix, other._key)

    def __hash__(self) -> int:
        return hash((self._namespace_prefix, self._key))


class DataUnit(DataUnitBase):
    """
    Object-based data unit for efficient transportation of objects by reference.

    This data unit is designed to pass Python objects between workflow steps
    without serialization, maintaining object references for efficient memory usage.
    """

    @classmethod
    def from_config(cls, config: Union[str, Path, DataUnitConfig, Dict[str, Any]], **kwargs) -> 'DataUnit':
        """
        Enhanced from_config implementation following standard NanoBrain pattern

        Supports both file paths and inline dictionary configurations as per
        NanoBrain framework standards for DataUnit, Link, and Trigger classes.

        Args:
            config: Configuration file path, DataUnitConfig object, or dictionary
            **kwargs: Additional context and dependencies

        Returns:
            Fully initialized DataUnit instance

        ✅ FRAMEWORK COMPLIANCE:
        - Follows standard Union[str, Path, ConfigClass, Dict] pattern
        - Supports inline dict config as per DataUnit rules
        - No hardcoding or simplified solutions
        - Pure configuration-driven instantiation
        """
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Step 1: Normalize input to DataUnitConfig object
        if isinstance(config, (str, Path)):
            # File path input - use standard config loading
            config_object = DataUnitConfig.from_config(config, **kwargs)
        elif isinstance(config, dict):
            # Dictionary input - create DataUnitConfig from dict (inline config support)
            # This is specifically allowed for DataUnit, Link, Trigger classes
            try:
                # Enable direct instantiation for config creation
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config)
            finally:
                DataUnitConfig._allow_direct_instantiation = False
        elif isinstance(config, DataUnitConfig):
            # Already a DataUnitConfig object
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
                DataUnitConfig._allow_direct_instantiation = True
                config_object = DataUnitConfig(**config_dict)
            finally:
                DataUnitConfig._allow_direct_instantiation = False

        # Step 2: Validate configuration schema
        cls.validate_config_schema(config_object)

        # Step 3: Extract component-specific configuration
        component_config = cls.extract_component_config(config_object)

        # Step 4: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 5: Create instance
        instance = cls.create_instance(
            config_object, component_config, dependencies)

        # Step 6: Post-creation initialization
        instance._post_config_initialization()

        logger.info(f"Successfully created {cls.__name__}")
        return instance

    def _init_from_config(self, config: DataUnitConfig, component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnit with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._object_ref = None
        self._object_type = None
        self._object_id = None

    async def get(self) -> Any:
        """Get the object reference."""
        if not self.is_initialized:
            await self.initialize()
        return self._object_ref

    async def set(self, data: Any) -> None:
        """Set the object reference."""
        if not self.is_initialized:
            await self.initialize()
        async with self._lock:
            self._object_ref = data
            self._object_type = type(data).__name__
            self._object_id = id(data)
            self._metadata['last_updated'] = time.time()
            self._metadata['object_type'] = self._object_type
            self._metadata['object_id'] = self._object_id

    async def clear(self) -> None:
        """Clear the object reference."""
        self._access_count["clear"] += 1
        self._last_operation = "clear"
        self._operation_count += 1

        # Log before clearing
        if self.enable_logging and self.nb_logger:
            had_object = self._object_ref is not None
            self.nb_logger.log_data_unit_operation(
                operation="clear",
                data_unit_name=self.name,
                metadata={
                    "had_object": had_object,
                    "previous_object_type": self._object_type,
                    "previous_object_id": self._object_id,
                    "metadata_count": len(self._metadata),
                    "state_before": self._get_internal_state()
                }
            )

        async with self._lock:
            self._object_ref = None
            self._object_type = None
            self._object_id = None
            self._metadata.clear()

        # Log after clearing
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Clear completed for {self.name}",
                                 operation="clear_complete",
                                 internal_state=self._get_internal_state())

    def get_object_info(self) -> Dict[str, Any]:
        """Get information about the stored object."""
        return {
            'has_object': self._object_ref is not None,
            'object_type': self._object_type,
            'object_id': self._object_id,
            'metadata': self.metadata
        }
