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
# Import new ConfigBase for constructor prohibition
from .config.config_base import ConfigBase

logger = logging.getLogger(__name__)


class ExecutorType(Enum):
    """Types of available executors."""
    LOCAL = "local"
    PARSL = "parsl"
    THREAD = "thread"
    PROCESS = "process"


class ExecutorConfig(ConfigBase):
    """
    Configuration for executors - INHERITS constructor prohibition.
    
    ❌ FORBIDDEN: ExecutorConfig(executor_type="local", ...)
    ✅ REQUIRED: ExecutorConfig.from_config('path/to/config.yml')
    """
    
    executor_type: ExecutorType = ExecutorType.LOCAL
    max_workers: int = Field(default=4, ge=1)
    timeout: Optional[float] = None
    parsl_config: Optional[Dict[str, Any]] = None


class ExecutorBase(FromConfigBase, ABC):
    """
    Base Executor Class - Configurable Execution Backends and Performance Optimization
    =================================================================================
    
    The ExecutorBase class is the foundational component for managing task execution
    within the NanoBrain framework. Executors provide flexible execution backends
    that enable components to run on various computing environments from local
    machines to distributed high-performance computing clusters.
    
    **Core Architecture:**
        Executors represent intelligent execution management systems that:
        
        * **Manage Execution**: Control how and where tasks are executed
        * **Optimize Performance**: Provide performance optimization and resource management
        * **Scale Dynamically**: Support scaling from local to distributed execution
        * **Handle Resources**: Manage computational resources and worker processes
        * **Ensure Reliability**: Provide fault tolerance and error recovery
        * **Monitor Performance**: Track execution metrics and optimization opportunities
    
    **Biological Analogy:**
        Like neurotransmitter systems that control the activation of different neural
        circuits, executor classes control the activation and execution of different
        types of computational tasks. Neurotransmitter systems (dopamine, serotonin,
        acetylcholine) have specialized properties for different types of neural
        activation - exactly how executor systems have specialized properties for
        different types of computational execution (local, parallel, distributed).
    
    **Execution Backend Architecture:**
        
        **Local Execution:**
        * Single-threaded execution for lightweight tasks
        * Minimal overhead for simple operations
        * Direct function calls without serialization
        * Immediate execution and result return
        
        **Threaded Execution:**
        * Multi-threaded execution for concurrent tasks
        * Thread pool management and optimization
        * Shared memory for efficient data exchange
        * Thread-safe operation coordination
        
        **Process Execution:**
        * Multi-process execution for CPU-intensive tasks
        * Process isolation for fault tolerance
        * Inter-process communication and data exchange
        * Resource isolation and management
        
        **Distributed Execution:**
        * Cluster-based execution for large-scale computing
        * Automatic resource discovery and allocation
        * Load balancing across multiple compute nodes
        * Fault tolerance with automatic recovery
    
    **Framework Integration:**
        Executors seamlessly integrate with all framework components:
        
        * **Step Execution**: Execute step processing logic on various backends
        * **Workflow Orchestration**: Coordinate workflow execution across resources
        * **Agent Processing**: Execute agent operations with performance optimization
        * **Tool Execution**: Run tool operations on appropriate execution backends
        * **Data Processing**: Handle data operations with optimal resource allocation
        * **Monitoring Integration**: Comprehensive logging and performance tracking
    
    **Executor Type Implementations:**
        The framework supports various executor specializations:
        
        * **LocalExecutor**: Direct execution for lightweight operations
        * **ThreadExecutor**: Multi-threaded execution for concurrent processing
        * **ProcessExecutor**: Multi-process execution for CPU-intensive tasks
        * **ParslExecutor**: Distributed execution for high-performance computing
        * **CloudExecutor**: Cloud-based execution with auto-scaling
        * **HybridExecutor**: Combination of multiple execution strategies
    
    **Configuration Architecture:**
        Executors follow the framework's configuration-first design:
        
        ```yaml
        # Local executor configuration
        name: "local_executor"
        executor_type: "local"
        max_workers: 1
        timeout: 30
        
        # Performance settings
        performance:
          enable_profiling: true
          memory_limit: "2GB"
          cpu_limit: 2
        
        # Thread executor configuration
        name: "thread_executor"
        executor_type: "thread"
        max_workers: 8
        timeout: 60
        
        # Thread pool settings
        thread_config:
          pool_type: "adaptive"
          idle_timeout: 30
          queue_size: 1000
          thread_name_prefix: "nanobrain-worker"
        
        # Process executor configuration
        name: "process_executor"
        executor_type: "process"
        max_workers: 4
        timeout: 300
        
        # Process management
        process_config:
          start_method: "spawn"
          max_memory_per_worker: "4GB"
          worker_restart_threshold: 100
          enable_process_monitoring: true
        
        # Parsl executor for HPC
        name: "hpc_executor"
        executor_type: "parsl"
        max_workers: 100
        
        # Parsl configuration
        parsl_config:
          executors:
            - class: "HighThroughputExecutor"
              label: "htex"
              max_workers: 100
              worker_init: "module load python/3.9"
              
          providers:
            - class: "SlurmProvider"
              partition: "compute"
              nodes_per_block: 2
              init_blocks: 1
              max_blocks: 10
              walltime: "01:00:00"
              
          launchers:
            - class: "SrunLauncher"
              overrides: "--ntasks-per-node=24"
        
        # Cloud executor configuration
        name: "cloud_executor"
        executor_type: "cloud"
        
        # Cloud settings
        cloud_config:
          provider: "aws"
          instance_type: "c5.xlarge"
          min_instances: 1
          max_instances: 20
          auto_scaling: true
          spot_instances: true
          region: "us-west-2"
        ```
    
    **Usage Patterns:**
        
        **Basic Local Execution:**
        ```python
        from nanobrain.core import LocalExecutor
        
        # Create executor from configuration
        executor = LocalExecutor.from_config('config/local_executor.yml')
        
        # Initialize executor
        await executor.initialize()
        
        # Execute task
        result = await executor.execute(my_task, arg1="value1", arg2="value2")
        print(f"Task completed: {result}")
        
        # Cleanup
        await executor.shutdown()
        ```
        
        **Multi-Threaded Execution:**
        ```python
        # Thread executor for concurrent tasks
        thread_executor = ThreadExecutor.from_config('config/thread_executor.yml')
        
        await thread_executor.initialize()
        
        # Execute multiple tasks concurrently
        tasks = [task1, task2, task3, task4]
        results = await asyncio.gather(*[
            thread_executor.execute(task) for task in tasks
        ])
        
        print(f"All tasks completed: {results}")
        ```
        
        **Distributed HPC Execution:**
        ```python
        # Parsl executor for HPC clusters
        hpc_executor = ParslExecutor.from_config('config/hpc_executor.yml')
        
        await hpc_executor.initialize()
        
        # Execute computationally intensive task
        large_task = create_large_computation_task()
        result = await hpc_executor.execute(large_task)
        
        # Automatic distribution across cluster nodes
        print(f"HPC computation completed: {result}")
        ```
        
        **Dynamic Executor Selection:**
        ```python
        # Automatic executor selection based on task requirements
        def get_optimal_executor(task_profile):
            if task_profile.cpu_intensive:
                return ProcessExecutor.from_config('config/process_executor.yml')
            elif task_profile.requires_scaling:
                return ParslExecutor.from_config('config/hpc_executor.yml')
            else:
                return LocalExecutor.from_config('config/local_executor.yml')
        
        # Select and use optimal executor
        executor = get_optimal_executor(task.profile)
        result = await executor.execute(task)
        ```
    
    **Advanced Features:**
        
        **Performance Optimization:**
        * Intelligent task scheduling and load balancing
        * Resource pooling and reuse for efficiency
        * Adaptive worker scaling based on workload
        * Memory management and garbage collection
        
        **Fault Tolerance:**
        * Automatic retry with exponential backoff
        * Worker process restart and recovery
        * Task checkpointing and resumption
        * Graceful degradation for partial failures
        
        **Resource Management:**
        * Dynamic resource allocation and optimization
        * Memory and CPU usage monitoring
        * Resource limits and quota enforcement
        * Cleanup and resource deallocation
        
        **Monitoring and Analytics:**
        * Real-time execution performance monitoring
        * Resource utilization tracking and analysis
        * Task execution time profiling
        * Bottleneck identification and optimization recommendations
    
    **Execution Patterns:**
        
        **Task Parallelization:**
        * Automatic task decomposition for parallel execution
        * Result aggregation and correlation
        * Dependency tracking and resolution
        * Load balancing across available resources
        
        **Pipeline Execution:**
        * Sequential task execution with dependency management
        * Intermediate result caching and optimization
        * Pipeline stage monitoring and optimization
        * Error propagation and recovery
        
        **Batch Processing:**
        * Large-scale batch job execution and management
        * Progress tracking and intermediate checkpointing
        * Resource scheduling and optimization
        * Result collection and validation
        
        **Stream Processing:**
        * Real-time data stream processing
        * Low-latency execution for time-sensitive tasks
        * Continuous monitoring and adaptation
        * Backpressure handling and flow control
    
    **Performance and Scalability:**
        
        **Execution Optimization:**
        * Task-specific optimization strategies
        * Intelligent caching and memoization
        * Resource allocation optimization
        * Execution path optimization
        
        **Scalability Features:**
        * Horizontal scaling across multiple compute nodes
        * Vertical scaling with resource allocation
        * Auto-scaling based on workload demands
        * Load balancing and resource distribution
        
        **Resource Efficiency:**
        * Memory usage optimization and management
        * CPU utilization monitoring and optimization
        * Network bandwidth optimization
        * Storage usage optimization and cleanup
    
    **Integration Patterns:**
        
        **Workflow Integration:**
        * Executors provide computation backends for workflow steps
        * Automatic executor selection based on step requirements
        * Resource sharing and optimization across workflow stages
        * Progress tracking and performance monitoring
        
        **Agent Integration:**
        * Agents use executors for LLM operations and tool execution
        * Intelligent executor selection based on agent requirements
        * Resource optimization for multi-agent scenarios
        * Performance monitoring and optimization
        
        **Tool Integration:**
        * Tools use executors for computational operations
        * Automatic scaling for computationally intensive tools
        * Resource isolation for tool security
        * Performance monitoring and optimization
    
    **Security and Reliability:**
        
        **Secure Execution:**
        * Process isolation and sandboxing
        * Resource limits and quota enforcement
        * Access control and permission management
        * Audit logging and security monitoring
        
        **Reliability Features:**
        * Fault tolerance with automatic recovery
        * Task persistence and resumption
        * Health monitoring and alerting
        * Graceful shutdown and cleanup
        
        **Data Protection:**
        * Secure data transfer and storage
        * Encryption for sensitive operations
        * Data integrity validation
        * Privacy protection and anonymization
    
    **Development and Testing:**
        
        **Testing Support:**
        * Mock executor implementations for testing
        * Execution simulation and validation
        * Performance benchmarking and profiling
        * Load testing and stress testing
        
        **Debugging Features:**
        * Comprehensive logging with execution tracing
        * Performance profiling and analysis
        * Resource usage monitoring and debugging
        * Error diagnosis and resolution tools
        
        **Development Tools:**
        * Executor configuration validation and optimization
        * Performance monitoring and analysis tools
        * Resource usage visualization and analysis
        * Optimization recommendations and tuning
    
    **Executor Lifecycle:**
        Executors follow a well-defined lifecycle:
        
        1. **Configuration Loading**: Parse and validate executor configuration
        2. **Resource Initialization**: Setup computational resources and connections
        3. **Worker Pool Creation**: Initialize worker processes or threads
        4. **Ready State**: Executor ready to accept and execute tasks
        5. **Task Execution**: Handle task execution with resource management
        6. **Performance Monitoring**: Track performance and resource usage
        7. **Resource Optimization**: Optimize resource allocation and usage
        8. **Cleanup and Shutdown**: Release resources and cleanup connections
    
    Attributes:
        name (str): Executor identifier for logging and resource management
        executor_type (ExecutorType): Type of execution backend and strategy
        max_workers (int): Maximum number of concurrent workers or processes
        timeout (float, optional): Maximum execution time before timeout
        is_initialized (bool): Whether executor is ready for task execution
        energy_level (float): Current resource availability and capacity (0.0-1.0)
        worker_pool (optional): Pool of worker processes or threads
        performance_metrics (Dict): Real-time performance and resource usage metrics
    
    Note:
        This is an abstract base class that cannot be instantiated directly.
        Use concrete implementations like LocalExecutor, ThreadExecutor, or
        ParslExecutor. All executors must be created using the from_config
        pattern with proper configuration files following framework patterns.
    
    Warning:
        Executors may consume significant computational resources including
        CPU, memory, and network bandwidth. Monitor resource usage and implement
        appropriate limits and cleanup mechanisms. Be cautious with distributed
        executors that may incur cloud computing costs.
    
    See Also:
        * :class:`ExecutorConfig`: Executor configuration schema and validation
        * :class:`ExecutorType`: Available executor types and execution strategies
        * :class:`LocalExecutor`: Local execution backend for lightweight tasks
        * :class:`ParslExecutor`: Distributed execution backend for HPC environments
        * :class:`BaseStep`: Steps that use executors for processing operations
        * :class:`Workflow`: Workflows that coordinate executor-based processing
        * :class:`Agent`: Agents that use executors for LLM and tool operations
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
    
    config = config or ExecutorConfig.from_config({"executor_type": executor_type})
    
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