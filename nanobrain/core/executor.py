"""
Executor System for NanoBrain Framework

Provides configurable execution backends including local and Parsl-based HPC execution.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError

logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Types of available executors."""
    LOCAL = "local"
    PARSL = "parsl"
    THREAD = "thread"
    PROCESS = "process"


class ExecutorConfig(BaseModel):
    """Configuration for executors."""
    model_config = ConfigDict(use_enum_values=True)
    
    executor_type: ExecutorType = ExecutorType.LOCAL
    max_workers: int = Field(default=4, ge=1)
    timeout: Optional[float] = None
    parsl_config: Optional[Dict[str, Any]] = None


class ExecutorBase(FromConfigBase, ABC):
    """
    Base executor class for running tasks.
    Enhanced with mandatory from_config pattern implementation.
    
    Biological analogy: Neurotransmitter systems controlling neural activation.
    Justification: Like how different neurotransmitter systems control the 
    activation of different neural circuits, executor classes control the 
    activation of different types of tasks.
    """
    
    COMPONENT_TYPE = "executor"
    REQUIRED_CONFIG_FIELDS = ['executor_type']
    OPTIONAL_CONFIG_FIELDS = {
        'max_workers': 4,
        'timeout': None,
        'parsl_config': None
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return ExecutorConfig - ONLY method that differs from other components"""
        return ExecutorConfig
    
    @classmethod
    def extract_component_config(cls, config: ExecutorConfig) -> Dict[str, Any]:
        """Extract Executor configuration"""
        return {
            'executor_type': config.executor_type,
            'max_workers': getattr(config, 'max_workers', 4),
            'timeout': getattr(config, 'timeout', None),
            'parsl_config': getattr(config, 'parsl_config', None)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Executor dependencies"""
        return {}
    
    def _init_from_config(self, config: ExecutorConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Executor with resolved dependencies"""
        self.config = config
        self.name = self.__class__.__name__
        self._energy_level = 1.0
        self._is_initialized = False
    
    # ExecutorBase inherits FromConfigBase.__init__ which prevents direct instantiation
        
    @abstractmethod
    async def execute(self, task: Any, **kwargs) -> Any:
        """Execute a task asynchronously."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the executor."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the executor and cleanup resources."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if executor is initialized."""
        return self._is_initialized
    
    @property
    def energy_level(self) -> float:
        """Get current energy level (0.0-1.0)."""
        return self._energy_level
    
    def can_execute(self, task_type: str) -> bool:
        """Check if this executor can handle the task type."""
        return self.is_initialized and self.energy_level > 0.1


class LocalExecutor(ExecutorBase):
    """
    Local async executor for lightweight tasks.
    """
    
    @classmethod
    def from_config(cls, config: ExecutorConfig, **kwargs) -> 'LocalExecutor':
        """Mandatory from_config implementation for LocalExecutor"""
        logger = logging.getLogger(f"{cls.__name__}.from_config")
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
    
    def _init_from_config(self, config: ExecutorConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize LocalExecutor with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._semaphore: Optional[asyncio.Semaphore] = None
        
    async def initialize(self) -> None:
        """Initialize the local executor."""
        if not self._is_initialized:
            self._semaphore = asyncio.Semaphore(self.config.max_workers)
            self._is_initialized = True
            logger.info(f"LocalExecutor initialized with {self.config.max_workers} workers")
    
    async def execute(self, task: Any, **kwargs) -> Any:
        """Execute a task locally."""
        if not self.is_initialized:
            await self.initialize()
            
        async with self._semaphore:
            try:
                # If task is a coroutine, await it
                if asyncio.iscoroutine(task):
                    result = await task
                # If task is callable, call it
                elif callable(task):
                    result = task(**kwargs)
                    # If the result is a coroutine, await it
                    if asyncio.iscoroutine(result):
                        result = await result
                else:
                    result = task
                    
                return result
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                raise
    
    async def shutdown(self) -> None:
        """Shutdown the local executor."""
        self._is_initialized = False
        self._semaphore = None
        logger.info("LocalExecutor shutdown complete")


class ThreadExecutor(ExecutorBase):
    """
    Thread-based executor for CPU-bound tasks.
    """
    
    @classmethod
    def from_config(cls, config: ExecutorConfig, **kwargs) -> 'ThreadExecutor':
        """Mandatory from_config implementation for ThreadExecutor"""
        logger = logging.getLogger(f"{cls.__name__}.from_config")
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
    
    def _init_from_config(self, config: ExecutorConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ThreadExecutor with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._executor: Optional[Any] = None
        
    async def initialize(self) -> None:
        """Initialize the thread executor."""
        if not self._is_initialized:
            import concurrent.futures
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.max_workers
            )
            self._is_initialized = True
            logger.info(f"ThreadExecutor initialized with {self.config.max_workers} workers")
    
    async def execute(self, task: Any, **kwargs) -> Any:
        """Execute a task in a thread."""
        if not self.is_initialized:
            await self.initialize()
            
        loop = asyncio.get_event_loop()
        
        try:
            if callable(task):
                result = await loop.run_in_executor(
                    self._executor, 
                    lambda: task(**kwargs)
                )
            else:
                result = task
                
            return result
            
        except Exception as e:
            logger.error(f"Thread task execution failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the thread executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._is_initialized = False
        logger.info("ThreadExecutor shutdown complete")


class ProcessExecutor(ExecutorBase):
    """
    Process-based executor for CPU-intensive tasks requiring isolation.
    """
    
    @classmethod
    def from_config(cls, config: ExecutorConfig, **kwargs) -> 'ProcessExecutor':
        """Mandatory from_config implementation for ProcessExecutor"""
        logger = logging.getLogger(f"{cls.__name__}.from_config")
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
    
    def _init_from_config(self, config: ExecutorConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ProcessExecutor with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._executor: Optional[Any] = None
        
    async def initialize(self) -> None:
        """Initialize the process executor."""
        if not self._is_initialized:
            import concurrent.futures
            self._executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            )
            self._is_initialized = True
            logger.info(f"ProcessExecutor initialized with {self.config.max_workers} workers")
    
    async def execute(self, task: Any, **kwargs) -> Any:
        """Execute a task in a separate process."""
        if not self.is_initialized:
            await self.initialize()
            
        loop = asyncio.get_event_loop()
        
        try:
            if callable(task):
                # For process execution, we need to ensure the task is pickleable
                result = await loop.run_in_executor(
                    self._executor, 
                    lambda: task(**kwargs)
                )
            else:
                result = task
                
            return result
            
        except Exception as e:
            logger.error(f"Process task execution failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the process executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._is_initialized = False
        logger.info("ProcessExecutor shutdown complete")


# Parsl executor will be implemented in a separate module to avoid heavy dependencies
class ParslExecutor(ExecutorBase):
    """
    Parsl-based executor for HPC workloads.
    
    This executor integrates with Parsl for distributed and HPC execution.
    """
    
    @classmethod
    def from_config(cls, config: ExecutorConfig, **kwargs) -> 'ParslExecutor':
        """Mandatory from_config implementation for ParslExecutor"""
        logger = logging.getLogger(f"{cls.__name__}.from_config")
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
    
    def _init_from_config(self, config: ExecutorConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ParslExecutor with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self._parsl_dfk = None
        self._parsl_config = None
        

        
    async def initialize(self) -> None:
        """Initialize Parsl executor."""
        try:
            import parsl
            from parsl.config import Config
            from parsl.executors import HighThroughputExecutor
            from parsl.providers import LocalProvider
            import os
            import sys
            import logging
            
            # Get project root path for worker initialization
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            # Worker initialization script to set up Python path (shell commands)
            worker_init = f'export PYTHONPATH="{project_root}:$PYTHONPATH"'
            
            # Create Parsl configuration
            if self.config.parsl_config:
                # Process provided Parsl configuration
                parsl_config_dict = self.config.parsl_config.copy()
                
                # Convert executor dictionaries to actual executor objects
                if 'executors' in parsl_config_dict:
                    executors = []
                    for executor_config in parsl_config_dict['executors']:
                        if isinstance(executor_config, dict):
                            # Make a copy to avoid modifying the original
                            exec_config = executor_config.copy()
                            
                            # Extract class name and parameters
                            executor_class_name = exec_config.pop('class', 'parsl.executors.HighThroughputExecutor')
                            
                            # Import the executor class
                            if executor_class_name == 'parsl.executors.HighThroughputExecutor':
                                executor_class = HighThroughputExecutor
                            else:
                                # Dynamic import for other executor types
                                module_name, class_name = executor_class_name.rsplit('.', 1)
                                module = __import__(module_name, fromlist=[class_name])
                                executor_class = getattr(module, class_name)
                            
                            # Handle provider configuration
                            if 'provider_config' in exec_config:
                                provider_config = exec_config.pop('provider_config')
                                provider_class_name = provider_config.pop('class', 'parsl.providers.LocalProvider')
                                
                                # Import provider class
                                if provider_class_name == 'parsl.providers.LocalProvider':
                                    provider_class = LocalProvider
                                else:
                                    # Dynamic import for other provider types
                                    module_name, class_name = provider_class_name.rsplit('.', 1)
                                    module = __import__(module_name, fromlist=[class_name])
                                    provider_class = getattr(module, class_name)
                                
                                # Add worker initialization
                                provider_config['worker_init'] = worker_init
                                
                                # Create provider instance
                                provider = provider_class(**provider_config)
                                exec_config['provider'] = provider
                            
                            elif 'provider' in exec_config:
                                # Handle existing provider configuration
                                if isinstance(exec_config['provider'], dict):
                                    provider_config = exec_config['provider']
                                    provider_config['worker_init'] = worker_init
                                    # Create LocalProvider as default
                                    provider = LocalProvider(**provider_config)
                                    exec_config['provider'] = provider
                                elif hasattr(exec_config['provider'], 'worker_init'):
                                    exec_config['provider'].worker_init = worker_init
                            
                            else:
                                # Create default provider if none specified
                                provider = LocalProvider(
                                    worker_init=worker_init,
                                    init_blocks=1,
                                    max_blocks=1,
                                    min_blocks=0
                                )
                                exec_config['provider'] = provider
                            
                            # Create executor instance
                            executor = executor_class(**exec_config)
                            executors.append(executor)
                        else:
                            # Already an executor object
                            executors.append(executor_config)
                    
                    parsl_config_dict['executors'] = executors
                
                self._parsl_config = Config(**parsl_config_dict)
            else:
                # Default configuration with worker initialization
                provider = LocalProvider(
                    worker_init=worker_init,
                    init_blocks=1,
                    max_blocks=1,
                    min_blocks=0
                )
                
                self._parsl_config = Config(
                    executors=[
                        HighThroughputExecutor(
                            label="htex_local",
                            max_workers_per_node=self.config.max_workers,
                            provider=provider
                        )
                    ]
                )
            
            # Load Parsl configuration
            try:
                self._parsl_dfk = parsl.load(self._parsl_config)
            except Exception as e:
                if "Config has already been loaded" in str(e):
                    # Parsl is already loaded, get the existing DataFlowKernel
                    self._parsl_dfk = parsl.dfk()
                    logger.info("ParslExecutor using existing Parsl configuration")
                else:
                    raise
            
            self._is_initialized = True
            logger.info("ParslExecutor initialized")
            
        except ImportError:
            logger.error("Parsl not available. Install with: pip install parsl")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ParslExecutor: {e}")
            raise
    
    def submit(self, task: Any, *args, **kwargs):
        """Submit a task for asynchronous execution using Parsl."""
        if not self.is_initialized:
            raise RuntimeError("ParslExecutor not initialized")
            
        try:
            import parsl
            import asyncio
            
            # Convert task to Parsl app if needed
            if not hasattr(task, 'func'):
                # Wrap function as Parsl app
                @parsl.python_app
                def parsl_task(*task_args, **task_kwargs):
                    if callable(task):
                        return task(*task_args, **task_kwargs)
                    return task
                
                app_task = parsl_task
            else:
                app_task = task
            
            # Submit task and return a future-like object
            parsl_future = app_task(*args, **kwargs)
            
            # Create an asyncio future that wraps the Parsl future
            async def await_parsl_future():
                loop = asyncio.get_event_loop()
                # Run the blocking result() call in a thread pool
                return await loop.run_in_executor(None, parsl_future.result)
            
            return await_parsl_future()
            
        except Exception as e:
            logger.error(f"Parsl task submission failed: {e}")
            raise

    async def execute(self, task: Any, **kwargs) -> Any:
        """Execute a task using Parsl."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            # Use submit and await the result
            future = self.submit(task, **kwargs)
            result = await future
            return result
            
        except Exception as e:
            logger.error(f"Parsl task execution failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown Parsl executor."""
        try:
            if self._parsl_dfk:
                import parsl
                parsl.clear()
                self._parsl_dfk = None
            self._is_initialized = False
            logger.info("ParslExecutor shutdown complete")
        except Exception as e:
            logger.error(f"Error during ParslExecutor shutdown: {e}")


def create_executor(executor_type: Union[ExecutorType, str], 
                   config: Optional[ExecutorConfig] = None, **kwargs) -> ExecutorBase:
    """
    MANDATORY from_config factory for all executor types
    
    Args:
        executor_type: Type of executor to create
        config: Executor configuration
        **kwargs: Framework-provided dependencies
        
    Returns:
        ExecutorBase instance created via from_config
        
    Raises:
        ValueError: If executor type is unknown
        ComponentConfigurationError: If configuration is invalid
    """
    logger = logging.getLogger("executor.factory")
    logger.info(f"Creating executor via mandatory from_config")
    
    if isinstance(executor_type, str):
        executor_type = ExecutorType(executor_type)
    
    config = config or ExecutorConfig(executor_type=executor_type)
    
    try:
        if executor_type == ExecutorType.LOCAL:
            executor_class = LocalExecutor
        elif executor_type == ExecutorType.THREAD:
            executor_class = ThreadExecutor
        elif executor_type == ExecutorType.PROCESS:
            executor_class = ProcessExecutor
        elif executor_type == ExecutorType.PARSL:
            executor_class = ParslExecutor
        else:
            raise ValueError(f"Unknown executor type: {executor_type}")
        
        # Create instance via from_config
        instance = executor_class.from_config(config, **kwargs)
        
        logger.info(f"Successfully created {executor_class.__name__} via from_config")
        return instance
        
    except Exception as e:
        raise ValueError(f"Failed to create executor '{executor_type}' via from_config: {e}") 