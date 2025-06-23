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

logger = logging.getLogger(__name__)


class StepConfig(BaseModel):
    """Configuration for steps."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
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


class BaseStep(FromConfigBase, ABC):
    """
    Enhanced base class for Steps with mandatory from_config implementation.
    
    Biological analogy: Functional neural circuit.
    Justification: Like how functional neural circuits process specific
    types of information and pass results to other circuits, steps process
    specific operations and pass results to other steps.
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
    def from_config(cls, config: StepConfig, **kwargs) -> 'BaseStep':
        """Mandatory from_config implementation for BaseStep components"""
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
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract BaseStep-specific configuration"""
        return {
            'name': config.name,
            'description': getattr(config, 'description', ''),
            'auto_initialize': getattr(config, 'auto_initialize', True),
            'debug_mode': getattr(config, 'debug_mode', False),
            'enable_logging': getattr(config, 'enable_logging', True),
            'log_data_transfers': getattr(config, 'log_data_transfers', True),
            'log_executions': getattr(config, 'log_executions', True),
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve BaseStep dependencies"""
        executor = kwargs.get('executor') or LocalExecutor()
        return {
            'executor': executor
        }
    
    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize BaseStep with resolved dependencies"""
        self.config = config
        self.name = component_config['name']
        self.description = component_config['description']
        
        # Initialize logging system
        self.nb_logger = get_logger(f"step.{self.name}", debug_mode=component_config['debug_mode'])
        self.nb_logger.info(f"Initializing step: {self.name}", step_name=self.name, config=config.model_dump())
        
        # Executor for running the step
        self.executor = dependencies['executor']
        
        # Data management
        self.input_data_units: Dict[str, DataUnitBase] = {}
        self.output_data_unit: Optional[DataUnitBase] = None
        self.links: Dict[str, LinkBase] = {}
        
        # Trigger for activation
        self.trigger: Optional[TriggerBase] = None
        
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
        
    async def initialize(self) -> None:
        """Initialize the step and its components."""
        if self._is_initialized:
            self.nb_logger.debug(f"Step {self.name} already initialized")
            return
        
        async with self.nb_logger.async_execution_context(
            OperationType.STEP_EXECUTE, 
            f"{self.name}.initialize"
        ) as context:
            # Initialize executor
            self.nb_logger.debug(f"Initializing executor for step {self.name}")
            await self.executor.initialize()
            
            # Initialize input data units
            self.nb_logger.debug(f"Initializing {len(self.config.input_configs)} input data units")
            for input_id, input_config in self.config.input_configs.items():
                data_unit = self._create_data_unit(input_config)
                await data_unit.initialize()
                self.input_data_units[input_id] = data_unit
                self.nb_logger.debug(f"Initialized input data unit: {input_id}")
            
            # Initialize output data unit
            if self.config.output_config:
                self.nb_logger.debug(f"Initializing output data unit")
                self.output_data_unit = self._create_data_unit(self.config.output_config)
                await self.output_data_unit.initialize()
            
            # Initialize trigger
            if self.config.trigger_config:
                self.nb_logger.debug(f"Initializing trigger")
                self.trigger = self._create_trigger(self.config.trigger_config)
                
                # Set up trigger callback
                await self.trigger.add_callback(self._on_trigger_activated)
                
                # Start monitoring
                await self.trigger.start_monitoring()
            
            self._is_initialized = True
            context.metadata['input_count'] = len(self.input_data_units)
            context.metadata['has_output'] = self.output_data_unit is not None
            context.metadata['has_trigger'] = self.trigger is not None
            
        self.nb_logger.info(f"Step {self.name} initialized successfully", 
                           input_count=len(self.input_data_units),
                           has_output=self.output_data_unit is not None,
                           has_trigger=self.trigger is not None)
    
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
        """Create a data unit from configuration using ComponentFactory."""
        # Import here to avoid circular imports
        from .config.component_factory import import_and_create_from_config
        from .data_unit import DataUnitType
        
        # Map data unit type to full class path
        data_type = config.data_type
        if isinstance(data_type, str):
            data_type = DataUnitType(data_type)
        
        class_path_map = {
            DataUnitType.MEMORY: "nanobrain.core.data_unit.DataUnitMemory",
            DataUnitType.FILE: "nanobrain.core.data_unit.DataUnitFile", 
            DataUnitType.STRING: "nanobrain.core.data_unit.DataUnitString",
            DataUnitType.STREAM: "nanobrain.core.data_unit.DataUnitStream"
        }
        
        class_path = class_path_map.get(data_type)
        if not class_path:
            raise ValueError(f"Unknown data unit type: {data_type}")
        
        return import_and_create_from_config(class_path, config)
    
    def _create_trigger(self, config: TriggerConfig) -> TriggerBase:
        """Create a trigger from configuration."""
        # Import here to avoid circular imports
        from .trigger import create_trigger
        return create_trigger(config)
    
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
            from .config.component_factory import import_and_create_from_config
            config = DataUnitConfig(data_type=DataUnitType.MEMORY, name=input_id)
            data_unit = import_and_create_from_config(
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


def create_step(step_type: str, config: StepConfig, **kwargs) -> BaseStep:
    """
    Direct import path factory - NO BACKWARD COMPATIBILITY
    
    Args:
        step_type: MUST be full class import path (e.g., 'module.submodule.ClassName')
        config: Step configuration object
        **kwargs: Framework-provided dependencies
        
    Returns:
        BaseStep instance created via from_config
        
    Raises:
        ValueError: If step_type is not a full import path or class doesn't implement from_config
        ImportError: If class cannot be imported
    """
    logger = get_logger("step.factory")
    logger.info(f"Creating step via mandatory from_config: {step_type}")
    
    # ENFORCE full import path requirement
    if '.' not in step_type:
        raise ValueError(
            f"Step type '{step_type}' must be a full import path. "
            f"Short class names and built-in types are no longer supported. "
            f"Use format: 'module.submodule.ClassName'"
        )
    
    try:
        # Direct import - only supported method
        module_path, class_name = step_type.rsplit('.', 1)
        module = importlib.import_module(module_path)
        step_class = getattr(module, class_name)
        
        # MANDATORY: Validate from_config implementation
        if not hasattr(step_class, 'from_config'):
            raise ValueError(f"Step class '{step_type}' must implement from_config method")
        
        # Create instance via from_config
        instance = step_class.from_config(config, **kwargs)
        
        # Validate returned instance
        if not isinstance(instance, BaseStep):
            raise ValueError(
                f"from_config for '{step_type}' must return BaseStep instance, "
                f"got {type(instance)}"
            )
        
        logger.info(f"Successfully created {step_type} via from_config")
        return instance
        
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_path}': {e}")
    except AttributeError as e:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}': {e}")
    except Exception as e:
        raise ValueError(f"Failed to create step '{step_type}': {e}") 