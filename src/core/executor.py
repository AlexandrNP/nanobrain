"""
Executor System for NanoBrain Framework

Provides configurable execution backends including local and Parsl-based HPC execution.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

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


class ExecutorBase(ABC):
    """
    Base executor class for running tasks.
    
    Biological analogy: Neurotransmitter systems controlling neural activation.
    Justification: Like how different neurotransmitter systems control the 
    activation of different neural circuits, executor classes control the 
    activation of different types of tasks.
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        self.config = config or ExecutorConfig()
        self.name = self.__class__.__name__
        self._energy_level = 1.0
        self._is_initialized = False
        
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
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        super().__init__(config)
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
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        super().__init__(config)
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
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        super().__init__(config)
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
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        super().__init__(config)
        self._parsl_dfk = None
        self._parsl_config = None
        
    async def initialize(self) -> None:
        """Initialize Parsl executor."""
        try:
            import parsl
            from parsl.config import Config
            from parsl.executors import HighThroughputExecutor
            
            # Create Parsl configuration
            if self.config.parsl_config:
                # Process provided Parsl configuration
                parsl_config_dict = self.config.parsl_config.copy()
                
                # Convert executor dictionaries to actual executor objects
                if 'executors' in parsl_config_dict:
                    executors = []
                    for executor_config in parsl_config_dict['executors']:
                        if isinstance(executor_config, dict):
                            # Extract class name and parameters
                            executor_class_name = executor_config.pop('class', 'parsl.executors.HighThroughputExecutor')
                            
                            # Import the executor class
                            if executor_class_name == 'parsl.executors.HighThroughputExecutor':
                                executor_class = HighThroughputExecutor
                            else:
                                # Dynamic import for other executor types
                                module_name, class_name = executor_class_name.rsplit('.', 1)
                                module = __import__(module_name, fromlist=[class_name])
                                executor_class = getattr(module, class_name)
                            
                            # Create executor instance
                            executor = executor_class(**executor_config)
                            executors.append(executor)
                        else:
                            # Already an executor object
                            executors.append(executor_config)
                    
                    parsl_config_dict['executors'] = executors
                
                self._parsl_config = Config(**parsl_config_dict)
            else:
                # Default configuration
                self._parsl_config = Config(
                    executors=[
                        HighThroughputExecutor(
                            label="htex_local",
                            max_workers=self.config.max_workers,
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
    
    async def execute(self, task: Any, **kwargs) -> Any:
        """Execute a task using Parsl."""
        if not self.is_initialized:
            await self.initialize()
            
        try:
            import parsl
            
            # Convert task to Parsl app if needed
            if not hasattr(task, 'func'):
                # Wrap function as Parsl app
                @parsl.python_app
                def parsl_task(**task_kwargs):
                    if callable(task):
                        return task(**task_kwargs)
                    return task
                
                app_task = parsl_task
            else:
                app_task = task
            
            # Execute and wait for result
            future = app_task(**kwargs)
            result = future.result()  # This blocks until completion
            
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
                   config: Optional[ExecutorConfig] = None) -> ExecutorBase:
    """
    Factory function to create executors.
    
    Args:
        executor_type: Type of executor to create
        config: Optional configuration
        
    Returns:
        Configured executor instance
    """
    if isinstance(executor_type, str):
        executor_type = ExecutorType(executor_type)
    
    config = config or ExecutorConfig(executor_type=executor_type)
    
    if executor_type == ExecutorType.LOCAL:
        return LocalExecutor(config)
    elif executor_type == ExecutorType.THREAD:
        return ThreadExecutor(config)
    elif executor_type == ExecutorType.PROCESS:
        return ProcessExecutor(config)
    elif executor_type == ExecutorType.PARSL:
        return ParslExecutor(config)
    else:
        raise ValueError(f"Unknown executor type: {executor_type}") 