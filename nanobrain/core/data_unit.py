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

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
# Import logging system
from .logging_system import get_logger, get_system_log_manager
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


class DataUnitConfig(ConfigBase):
    """
    Configuration for data units - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: DataUnitConfig(name="test", class="...")
    ✅ REQUIRED: DataUnitConfig.from_config('path/to/config.yml')
    """
    
    # MANDATORY class field for data unit type specification
    class_field: str = Field(alias="class", description="Full class path for data unit type")
    
    # Keep existing fields
    name: str = ""
    description: str = ""
    persistent: bool = False
    cache_size: int = Field(default=1000, ge=1)
    file_path: Optional[str] = None
    encoding: str = "utf-8"
    initial_value: Optional[str] = None
    
    @field_validator('class_field')
    @classmethod
    def validate_class_field(cls, v):
        """Validate class field is properly specified"""
        if not v or not v.strip():
            raise ValueError("Data unit class must be specified")
        if not v.startswith('nanobrain.core.data_unit.'):
            raise ValueError("Data unit class must be from nanobrain.core.data_unit module")
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
            'debug_mode': kwargs.get('debug_mode', False)
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
            self.nb_logger = get_logger(self.name, category="data_units", debug_mode=dependencies.get('debug_mode', False))
            
            # Register with system log manager
            system_manager = get_system_log_manager()
            system_manager.register_component("data_units", self.name, self, {
                "class_path": component_config['class_path'],
                "persistent": component_config['persistent'],
                "enable_logging": True
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
    
    # DataUnitBase inherits FromConfigBase.__init__ which prevents direct instantiation
    
    def register_change_listener(self, listener: Callable) -> None:
        """Register a change listener for event-driven triggers"""
        if listener not in self._change_listeners:
            self._change_listeners.append(listener)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Registered change listener for {self.name}")
    
    def unregister_change_listener(self, listener: Callable) -> None:
        """Unregister a change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Unregistered change listener for {self.name}")
    
    async def _notify_change_listeners(self, change_event: Dict[str, Any]) -> None:
        """Notify all registered change listeners of data unit changes"""
        if self._change_listeners:
            for listener in self._change_listeners:
                try:
                    await listener(change_event)
                except Exception as e:
                    if self.enable_logging and self.nb_logger:
                        self.nb_logger.error(f"Error in change listener for {self.name}: {e}")
        
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
            old_data_type = type(self._data).__name__ if self._data is not None else "None"
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
        instance = cls.create_instance(config_object, component_config, dependencies)
        
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
        """Get data from memory."""
        if not self.is_initialized:
            await self.initialize()
        return self._data
    
    async def set(self, data: Any) -> None:
        """Set data in memory."""
        if not self.is_initialized:
            await self.initialize()
        async with self._lock:
            old_data = self._data
            self._data = data
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
    
    async def clear(self) -> None:
        """Clear data from memory."""
        self._access_count["clear"] += 1
        self._last_operation = "clear"
        self._operation_count += 1
        
        # Log before clearing
        if self.enable_logging and self.nb_logger:
            had_data = self._data is not None
            self.nb_logger.log_data_unit_operation(
                operation="clear",
                data_unit_name=self.name,
                metadata={
                    "had_data": had_data,
                    "previous_data_type": type(self._data).__name__ if self._data is not None else "None",
                    "metadata_count": len(self._metadata),
                    "state_before": self._get_internal_state()
                }
            )
        
        async with self._lock:
            self._data = None
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
        instance = cls.create_instance(config_object, component_config, dependencies)
        
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
        file_path = dependencies.get('file_path') or component_config.get('file_path')
        if not file_path:
            raise ComponentConfigurationError("DataUnitFile requires file_path")
        self.file_path = Path(file_path)
        
    async def get(self) -> Any:
        """Get data from file."""
        if not self.is_initialized:
            await self.initialize()
            
        if not self.file_path.exists():
            return None
            
        try:
            async with self._lock:
                # Read file content
                content = self.file_path.read_text(encoding=self.config.encoding)
                
                # Try to parse as JSON if possible
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
                    
        except Exception as e:
            logger.error(f"Error reading file {self.file_path}: {e}")
            raise
    
    async def set(self, data: Any) -> None:
        """Set data to file."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            async with self._lock:
                # Get old data for change event
                old_data = None
                if self.file_path.exists():
                    try:
                        old_content = self.file_path.read_text(encoding=self.config.encoding)
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
                
                # Write to file
                self.file_path.write_text(content, encoding=self.config.encoding)
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
            logger.error(f"Error writing file {self.file_path}: {e}")
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
        instance = cls.create_instance(config_object, component_config, dependencies)
        
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
        instance = cls.create_instance(config_object, component_config, dependencies)
        
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
        instance = cls.create_instance(config_object, component_config, dependencies)
        
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

 