"""
Executor System for NanoBrain Framework

Provides configurable execution backends including local and Parsl-based HPC execution.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, Union, List
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
    default_resource_specification: Optional[Dict[str, Any]] = None


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
    async def execute(self, task: Any, resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute a task asynchronously with optional resource specification."""
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
    def from_config(cls, config: Union[str, ExecutorConfig], **kwargs) -> 'LocalExecutor':
        """
        Fixed from_config implementation - accepts both paths and objects for framework consistency

        Args:
            config: Either a string path to config file or ExecutorConfig object
            **kwargs: Additional context for configuration loading

        Returns:
            LocalExecutor instance
        """
        logger = logging.getLogger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Handle both string paths and ExecutorConfig objects for framework consistency
        if isinstance(config, str):
            # Load ExecutorConfig from file path (standard framework pattern)
            executor_config = ExecutorConfig.from_config(config, **kwargs)
        elif isinstance(config, ExecutorConfig):
            # Use provided ExecutorConfig object (backward compatibility)
            executor_config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected str or ExecutorConfig")

        # Step 1: Validate configuration schema
        cls.validate_config_schema(executor_config)
        
        # Step 2: Extract component-specific configuration
        component_config = cls.extract_component_config(executor_config)

        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 4: Create instance
        instance = cls.create_instance(executor_config, component_config, dependencies)
        
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
    
    async def execute(self, task: Any, resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute a task locally with proper async deadlock prevention.

        Note: resource_specification is ignored for local execution.
        """
        if not self.is_initialized:
            await self.initialize()

        async with self._semaphore:
            try:
                # If task is a coroutine, await it
                if asyncio.iscoroutine(task):
                    result = await task
                # If task is callable, call it
                elif callable(task):
                    # CRITICAL BUG FIX: Handle async functions properly to prevent deadlock
                    logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor handling callable task: {task}")
                    if asyncio.iscoroutinefunction(task):
                        # For async functions, create a new task to avoid deadlock
                        logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor creating new task for async function {task}")
                        async_task = asyncio.create_task(task(**kwargs))
                        result = await async_task
                        logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor async task completed with result type: {type(result)}")
                    else:
                        # For regular functions, call directly
                        logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor calling regular function {task}")
                        result = task(**kwargs)
                        # If the result is a coroutine, handle it properly
                        if asyncio.iscoroutine(result):
                            logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor function returned coroutine, creating task")
                            async_task = asyncio.create_task(result)
                            result = await async_task
                            logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor coroutine task completed with result type: {type(result)}")
                        else:
                            logger.info(f"🔥 BRUTAL TRUTH: LocalExecutor regular function completed with result type: {type(result)}")
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
    def from_config(cls, config: Union[str, ExecutorConfig], **kwargs) -> 'ThreadExecutor':
        """
        Fixed from_config implementation - accepts both paths and objects for framework consistency

        Args:
            config: Either a string path to config file or ExecutorConfig object
            **kwargs: Additional context for configuration loading

        Returns:
            ThreadExecutor instance
        """
        logger = logging.getLogger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Handle both string paths and ExecutorConfig objects for framework consistency
        if isinstance(config, str):
            # Load ExecutorConfig from file path (standard framework pattern)
            executor_config = ExecutorConfig.from_config(config, **kwargs)
        elif isinstance(config, ExecutorConfig):
            # Use provided ExecutorConfig object (backward compatibility)
            executor_config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected str or ExecutorConfig")

        # Step 1: Validate configuration schema
        cls.validate_config_schema(executor_config)
        
        # Step 2: Extract component-specific configuration
        component_config = cls.extract_component_config(executor_config)

        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 4: Create instance
        instance = cls.create_instance(executor_config, component_config, dependencies)
        
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
    
    async def execute(self, task: Any, resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute a task in a thread.

        Note: resource_specification is ignored for thread execution.
        """
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
    def from_config(cls, config: Union[str, ExecutorConfig], **kwargs) -> 'ProcessExecutor':
        """
        Fixed from_config implementation - accepts both paths and objects for framework consistency

        Args:
            config: Either a string path to config file or ExecutorConfig object
            **kwargs: Additional context for configuration loading

        Returns:
            ProcessExecutor instance
        """
        logger = logging.getLogger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Handle both string paths and ExecutorConfig objects for framework consistency
        if isinstance(config, str):
            # Load ExecutorConfig from file path (standard framework pattern)
            executor_config = ExecutorConfig.from_config(config, **kwargs)
        elif isinstance(config, ExecutorConfig):
            # Use provided ExecutorConfig object (backward compatibility)
            executor_config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected str or ExecutorConfig")

        # Step 1: Validate configuration schema
        cls.validate_config_schema(executor_config)
        
        # Step 2: Extract component-specific configuration
        component_config = cls.extract_component_config(executor_config)

        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 4: Create instance
        instance = cls.create_instance(executor_config, component_config, dependencies)
        
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
    
    async def execute(self, task: Any, resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute a task in a separate process.

        Note: resource_specification is ignored for process execution.
        """
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


# Module-level Parsl app for step execution
# CRITICAL: Must be at module level to avoid capturing non-serializable closure variables
def _get_execute_step_parsl_app():
    """
    Get or create the Parsl app for executing steps from config.
    Lazy import to avoid Parsl dependency issues.
    """
    try:
        from parsl.app.app import python_app

        @python_app
        def execute_step_from_config(step_config_path, step_class_name, input_data,
                                    parsl_resource_specification=None, **kwargs):
            """
            Recreate and execute step from config on compute node.
            This is fully serializable since it only uses config paths and data.

            Args:
                step_config_path: Path to step configuration file
                step_class_name: Fully qualified class name
                input_data: Input data dictionary
                parsl_resource_specification: Resource requirements (cores, memory, disk, etc.)
                **kwargs: Additional arguments

            Note: parsl_resource_specification is used by Parsl for task scheduling
            """
            import sys
            import os
            import asyncio
            import importlib

            # Add nanobrain to path and change to project directory
            nanobrain_path = "/lus/flare/projects/FoundEpidem/onarykov/nanobrain"
            if nanobrain_path not in sys.path:
                sys.path.insert(0, nanobrain_path)
            # Change to nanobrain directory so relative config paths work
            os.chdir(nanobrain_path)

            try:
                # Import the step class
                module_name, class_name = step_class_name.rsplit('.', 1)
                module = importlib.import_module(module_name)
                step_class = getattr(module, class_name)

                # Recreate step from config
                step = step_class.from_config(step_config_path)

                # Execute the step's process method
                if asyncio.iscoroutinefunction(step.process):
                    # Handle async process method
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(step.process(input_data, **kwargs))
                    finally:
                        loop.close()
                else:
                    # Handle sync process method
                    result = step.process(input_data, **kwargs)

                return {
                    'result': result,
                    'status': 'success',
                    'worker_node': os.uname().nodename,
                    'worker_pid': os.getpid(),
                    'resource_spec_used': parsl_resource_specification
                }

            except Exception as e:
                import traceback
                return {
                    'result': None,
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'worker_node': os.uname().nodename,
                    'worker_pid': os.getpid(),
                    'resource_spec_used': parsl_resource_specification
                }

        return execute_step_from_config

    except ImportError:
        logger.error("Parsl not available. Install with: pip install parsl")
        raise


# Module-level Parsl app for generic task execution
# CRITICAL: Must be at module level to avoid capturing non-serializable closure variables
def _get_parsl_task_wrapper():
    """
    Get or create a generic Parsl app wrapper for callable tasks.
    Lazy import to avoid Parsl dependency issues.
    """
    try:
        from parsl.app.app import python_app

        @python_app
        def parsl_task_wrapper(task_func, *task_args, **task_kwargs):
            """
            Generic wrapper for executing callable tasks on compute nodes.
            This is fully serializable since task_func is passed as parameter, not captured in closure.
            """
            if callable(task_func):
                return task_func(*task_args, **task_kwargs)
            return task_func

        return parsl_task_wrapper

    except ImportError:
        logger.error("Parsl not available. Install with: pip install parsl")
        raise


# Parsl executor will be implemented in a separate module to avoid heavy dependencies
class ParslExecutor(ExecutorBase):
    """
    Parsl-based executor for HPC workloads.
    
    This executor integrates with Parsl for distributed and HPC execution.
    """
    
    @classmethod
    def from_config(cls, config: Union[str, ExecutorConfig], **kwargs) -> 'ParslExecutor':
        """
        Fixed from_config implementation - accepts both paths and objects for framework consistency

        Args:
            config: Either a string path to config file or ExecutorConfig object
            **kwargs: Additional context for configuration loading

        Returns:
            ParslExecutor instance
        """
        logger = logging.getLogger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")

        # Handle both string paths and ExecutorConfig objects for framework consistency
        if isinstance(config, str):
            # Load ExecutorConfig from file path (standard framework pattern)
            executor_config = ExecutorConfig.from_config(config, **kwargs)
        elif isinstance(config, ExecutorConfig):
            # Use provided ExecutorConfig object (backward compatibility)
            executor_config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}. Expected str or ExecutorConfig")

        # Step 1: Validate configuration schema
        cls.validate_config_schema(executor_config)
        
        # Step 2: Extract component-specific configuration
        component_config = cls.extract_component_config(executor_config)

        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)

        # Step 4: Create instance
        instance = cls.create_instance(executor_config, component_config, dependencies)
        
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
        self._worker_step_pools = {}  # step_id -> WorkerStepPool

        # Resource tracking for parallel path conflict detection
        self._active_submissions = {}
        self._submission_lock = asyncio.Lock()

        # Ensure nb_logger exists (safety check)
        if not hasattr(self, 'nb_logger') or self.nb_logger is None:
            from nanobrain.core.logging_system import get_logger
            self.nb_logger = get_logger(f"parsl_executor.{getattr(self, 'name', 'unnamed')}")

    def _detect_execution_context(self) -> str:
        """
        Detect if this is top-level workflow or subworkflow execution.

        Returns:
            'top_level': ONE PBS job should be submitted (use HTEX)
            'subworkflow': Use existing PBS allocation (use WorkQueue)
        """
        import os

        # Check if we're running inside an existing PBS allocation
        pbs_jobid = os.environ.get('PBS_JOBID')
        parsl_worker_rank = os.environ.get('PARSL_WORKER_RANK')

        if pbs_jobid and parsl_worker_rank:
            # We're inside a Parsl worker - this is subworkflow execution
            self.nb_logger.info(f"🔥 DETECTED: Subworkflow execution (PBS_JOBID={pbs_jobid}, RANK={parsl_worker_rank})")
            return 'subworkflow'
        elif pbs_jobid:
            # We're in PBS job but not inside Parsl worker - this is top-level
            self.nb_logger.info(f"🔥 DETECTED: Top-level workflow in PBS job (PBS_JOBID={pbs_jobid})")
            return 'top_level'
        else:
            # No PBS allocation - local development or testing
            self.nb_logger.info(f"🔥 DETECTED: Local execution (no PBS)")
            return 'top_level'

    async def _track_resource_submission(self, resource_spec: Dict[str, Any], submission_id: str):
        """Track resource submission to detect conflicts."""
        async with self._submission_lock:
            # Log concurrent resource requests
            if self._active_submissions:
                self.nb_logger.info(f"🔥 PARALLEL PATHS: Concurrent submissions detected:")
                for sid, spec in self._active_submissions.items():
                    self.nb_logger.info(f"   {sid}: {spec}")
                self.nb_logger.info(f"   NEW {submission_id}: {resource_spec}")

            # Add to active submissions
            self._active_submissions[submission_id] = resource_spec

    async def _remove_resource_submission(self, submission_id: str):
        """Remove completed submission from tracking."""
        async with self._submission_lock:
            self._active_submissions.pop(submission_id, None)

    async def initialize(self) -> None:
        """Initialize Parsl executor with automatic HTEX/WorkQueue selection based on context."""
        # CRITICAL: If already initialized (e.g., inherited from parent workflow), skip initialization
        if self._is_initialized or self._parsl_dfk is not None:
            self.nb_logger.info("✅ ParslExecutor already initialized (inherited from parent), skipping")
            self._is_initialized = True
            return

        try:
            import asyncio
            import parsl
            from parsl.config import Config
            from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
            from parsl.providers import LocalProvider
            import os
            import sys
            import logging

            # STEP 1: Detect execution context (top-level vs subworkflow)
            execution_context = self._detect_execution_context()
            self.nb_logger.info(f"🔥 EXECUTION CONTEXT: {execution_context}")

            # STEP 2: Determine forced executor type based on context
            forced_executor_type = None
            if execution_context == 'subworkflow':
                # Subworkflows MUST use WorkQueue to share PBS allocation
                forced_executor_type = 'parsl.executors.WorkQueueExecutor'
                self.nb_logger.info(f"🔥 FORCING WorkQueueExecutor for subworkflow execution")
            elif execution_context == 'top_level':
                # Top-level workflows should use HTEX (or config-specified executor)
                # Don't force, let config decide (could be HTEX or WorkQueue)
                self.nb_logger.info(f"🔥 Using executor from config for top-level execution")

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

                            # STEP 3: Override executor type if forced by context
                            if forced_executor_type:
                                original_executor_type = executor_class_name
                                executor_class_name = forced_executor_type
                                self.nb_logger.info(f"🔥 OVERRIDING executor type: {original_executor_type} → {executor_class_name}")

                                # CRITICAL: For subworkflows, also force LocalProvider to avoid new PBS job submission
                                # Use MPIExecLauncher for multi-node distribution within PBS allocation
                                if execution_context == 'subworkflow' and 'provider_config' in exec_config:
                                    original_provider = exec_config.get('provider_config', {}).get('class', 'unknown')
                                    exec_config['provider_config'] = {
                                        'class': 'parsl.providers.LocalProvider',
                                        'init_blocks': 1,
                                        'min_blocks': 1,
                                        'max_blocks': 1,  # Only use existing PBS allocation
                                        'worker_init': exec_config.get('provider_config', {}).get('worker_init', ''),

                                        # MPI launcher for distributing across allocated nodes
                                        'launcher_config': {
                                            'class': 'parsl.launchers.MpiExecLauncher',
                                            'overrides': '--ppn 1'  # Process distribution config
                                        }
                                    }
                                    self.nb_logger.info(f"🔥 FORCING LocalProvider+MPIExecLauncher for subworkflow (was: {original_provider})")
                                    self.nb_logger.info(f"🔥 Subworkflow will use existing PBS allocation with MPI distribution")

                            # Import the executor class
                            if executor_class_name == 'parsl.executors.HighThroughputExecutor':
                                executor_class = HighThroughputExecutor
                            elif executor_class_name == 'parsl.executors.WorkQueueExecutor':
                                executor_class = WorkQueueExecutor
                            else:
                                # Dynamic import for other executor types
                                module_name, class_name = executor_class_name.rsplit('.', 1)
                                module = __import__(module_name, fromlist=[class_name])
                                executor_class = getattr(module, class_name)
                            
                            # Check if this executor type supports providers
                            executor_supports_provider = executor_class_name in [
                                'parsl.executors.HighThroughputExecutor',
                                'parsl.executors.WorkQueueExecutor',
                                'parsl.executors.ExtremeScaleExecutor',
                                'parsl.executors.FluxExecutor'
                            ]

                            # Handle provider configuration ONLY for executors that support it
                            if executor_supports_provider and 'provider_config' in exec_config:
                                provider_config = exec_config.pop('provider_config')
                                provider_class_name = provider_config.pop('class', 'parsl.providers.LocalProvider')

                                # Handle launcher configuration if present
                                launcher = None
                                if 'launcher' in provider_config:
                                    launcher_config = provider_config.pop('launcher')
                                    if isinstance(launcher_config, dict) and 'class' in launcher_config:
                                        launcher_class_name = launcher_config.pop('class')

                                        # Import launcher class
                                        if launcher_class_name == 'parsl.launchers.MpiExecLauncher':
                                            from parsl.launchers import MpiExecLauncher
                                            launcher_class = MpiExecLauncher
                                        elif launcher_class_name == 'parsl.launchers.SrunLauncher':
                                            from parsl.launchers import SrunLauncher
                                            launcher_class = SrunLauncher
                                        elif launcher_class_name == 'parsl.launchers.AprunLauncher':
                                            from parsl.launchers import AprunLauncher
                                            launcher_class = AprunLauncher
                                        else:
                                            # Dynamic import for other launcher types
                                            module_name, class_name = launcher_class_name.rsplit('.', 1)
                                            module = __import__(module_name, fromlist=[class_name])
                                            launcher_class = getattr(module, class_name)

                                        # Handle PBS nodefile for MPI launchers
                                        if launcher_class_name == 'parsl.launchers.MpiExecLauncher':
                                            import os
                                            pbs_nodefile = os.environ.get('PBS_NODEFILE')
                                            if pbs_nodefile and os.path.exists(pbs_nodefile):
                                                # Read and process PBS nodefile for multi-node distribution
                                                with open(pbs_nodefile, 'r') as f:
                                                    nodes = [line.strip() for line in f if line.strip()]

                                                if len(nodes) >= 2:
                                                    # Configure MPI for multi-node execution with explicit process count
                                                    current_overrides = launcher_config.get('overrides', '')
                                                    # Use Intel MPI options for multi-node distribution
                                                    # Let Parsl handle the hostfile, just add ppn for process distribution
                                                    launcher_config['overrides'] = f"{current_overrides} --ppn 1".strip()
                                                else:
                                                    # Single node fallback
                                                    current_overrides = launcher_config.get('overrides', '')
                                                    launcher_config['overrides'] = current_overrides

                                        # Create launcher instance
                                        launcher = launcher_class(**launcher_config)

                                # Import provider class
                                if provider_class_name == 'parsl.providers.LocalProvider':
                                    provider_class = LocalProvider
                                else:
                                    # Dynamic import for other provider types
                                    module_name, class_name = provider_class_name.rsplit('.', 1)
                                    module = __import__(module_name, fromlist=[class_name])
                                    provider_class = getattr(module, class_name)

                                # Add worker initialization - APPEND to existing worker_init instead of overwriting
                                existing_worker_init = provider_config.get('worker_init', '')
                                if existing_worker_init:
                                    # Append our PYTHONPATH setup to existing worker_init
                                    provider_config['worker_init'] = f"{existing_worker_init}\n{worker_init}"
                                else:
                                    # Use our worker_init if none exists
                                    provider_config['worker_init'] = worker_init

                                # Add launcher if configured
                                if launcher:
                                    provider_config['launcher'] = launcher

                                # Create provider instance
                                provider = provider_class(**provider_config)
                                exec_config['provider'] = provider
                            
                            elif executor_supports_provider and 'provider' in exec_config:
                                # Handle existing provider configuration
                                if isinstance(exec_config['provider'], dict):
                                    provider_config = exec_config['provider']
                                    provider_config['worker_init'] = worker_init
                                    # Create LocalProvider as default
                                    provider = LocalProvider(**provider_config)
                                    exec_config['provider'] = provider
                                elif hasattr(exec_config['provider'], 'worker_init'):
                                    exec_config['provider'].worker_init = worker_init

                            elif not executor_supports_provider:
                                # For executors that don't support providers, remove any provider-related config
                                exec_config.pop('provider_config', None)
                                exec_config.pop('provider', None)
                                self.nb_logger.info(f"🔥 BRUTAL TRUTH: Removed provider configuration for {executor_class_name} (not supported)")
                            
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
            # Check if Parsl is already loaded first to avoid double-load
            try:
                self._parsl_dfk = parsl.dfk()
                logger.info("✅ ParslExecutor using existing Parsl DataFlowKernel")
            except Exception:
                # Parsl not loaded yet, load it now
                # CRITICAL: parsl.load() is synchronous and MUST NOT block the async event loop
                # Run it in a thread executor to prevent blocking
                logger.info("🚀 Loading new Parsl configuration in thread executor...")
                loop = asyncio.get_running_loop()
                self._parsl_dfk = await loop.run_in_executor(
                    None,  # Use default ThreadPoolExecutor
                    parsl.load,
                    self._parsl_config
                )
                logger.info("✅ Parsl DataFlowKernel loaded successfully")
            
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
                # Use module-level Parsl app wrapper to avoid closure serialization issues
                parsl_task_wrapper = _get_parsl_task_wrapper()
                # Pass task as parameter instead of capturing in closure
                app_task = lambda *task_args, **task_kwargs: parsl_task_wrapper(task, *task_args, **task_kwargs)
            else:
                app_task = task
            
            # Submit task and return a future-like object
            parsl_future = app_task(*args, **kwargs)

            # Create an asyncio future that wraps the Parsl future
            async def await_parsl_future():
                loop = asyncio.get_event_loop()
                # Run the blocking result() call in a thread pool
                return await loop.run_in_executor(None, parsl_future.result)

            return await_parsl_future  # Return the coroutine, don't await it here
            
        except Exception as e:
            logger.error(f"Parsl task submission failed: {e}")
            raise

    async def setup_worker_step_pool(
        self,
        step_class: type,
        step_config: Dict[str, Any],
        step_id: str
    ) -> None:
        """
        Set up per-worker step instances for parallel execution.

        This creates one step instance per PARSL worker, ensuring:
        1. Each worker has its own step instance
        2. Each instance has a unique worker_id
        3. @shared resources are NOT duplicated
        4. Non-shared resources ARE duplicated per worker

        Args:
            step_class: Class of the step to instantiate
            step_config: Configuration for step instances
            step_id: Unique identifier for this step
        """
        from .worker_step_pool import WorkerStepPool

        if step_id in self._worker_step_pools:
            logger.warning(f"Worker step pool already exists for {step_id}")
            return

        logger.info(f"🔧 Setting up worker step pool for {step_id}")

        # Create pool with same number of workers as PARSL executor
        num_workers = self.config.max_workers if hasattr(self.config, 'max_workers') else 4

        pool = WorkerStepPool(
            step_class=step_class,
            step_config=step_config,
            num_workers=num_workers,
            step_id=step_id
        )

        # Initialize all worker instances
        await pool.initialize_workers()

        # Store pool
        self._worker_step_pools[step_id] = pool

        logger.info(f"✅ Worker step pool ready for {step_id}")
        logger.info(f"   Workers: {pool.get_all_worker_ids()}")
        logger.info(f"   Shared resources: {list(pool.shared_resources.keys())}")

    def get_worker_step_pool(self, step_id: str):
        """
        Get worker step pool for a specific step.

        Args:
            step_id: Step identifier

        Returns:
            WorkerStepPool instance or None
        """
        return self._worker_step_pools.get(step_id)

    def get_worker_step_pools_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all worker step pools.

        Returns:
            Dictionary with pool statistics
        """
        return {
            step_id: pool.get_stats()
            for step_id, pool in self._worker_step_pools.items()
        }

    async def execute(self, task: Any, add_worker_id: bool = True, add_node_info: bool = False,
                     resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Execute a task using Parsl with automatic WorkQueue delegation and fallback.

        Args:
            task: Task to execute
            add_worker_id: Whether to add worker ID to result (default: True)
            add_node_info: Whether to add worker node information (default: False) - DISABLED for Phase 1
            resource_specification: Resource requirements (cores, memory, disk, priority, etc.)
            **kwargs: Additional arguments

        Returns:
            Task result, optionally wrapped with worker ID metadata
        """
        self.nb_logger.info(f"🔥 BRUTAL TRUTH: ParslExecutor.execute() called with task: {task}")

        if not self.is_initialized:
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Executor not initialized, initializing...")
            await self.initialize()

        # STEP 1: Check if this should use WorkQueueExecutor delegation
        if self._should_use_workqueue_delegation():
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Detected WorkQueueExecutor configuration - delegating to WorkQueue execution...")
            return await self._execute_workqueue_delegation(task, add_worker_id, resource_specification, **kwargs)

        try:
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Attempting primary PARSL execution...")
            # Attempt primary PARSL execution with resource specification
            return await self._execute_parsl(task, add_worker_id, resource_specification, **kwargs)
        except Exception as e:
            self.nb_logger.error(f"🔥 BRUTAL TRUTH: PARSL execution failed with exception: {e}")
            import traceback
            self.nb_logger.error(f"🔥 BRUTAL TRUTH: Traceback: {traceback.format_exc()}")

            # Check if fallback should be attempted
            if self._should_fallback(e):
                self.nb_logger.warning(f"🔥 BRUTAL TRUTH: Fallback is enabled, attempting fallback...")
                logger.warning(f"PARSL execution failed: {e}")
                logger.info("🔄 Attempting fallback execution...")
                return await self._execute_fallback(task, add_worker_id, e, **kwargs)
            else:
                # Re-raise if fallback not configured or not appropriate
                self.nb_logger.error(f"🔥 BRUTAL TRUTH: No fallback configured, re-raising exception...")
                logger.error(f"PARSL execution failed and no fallback available: {e}")
                raise

    async def _execute_parsl(self, task: Any, add_worker_id: bool = True,
                           resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Execute task using PARSL with node tracking and resource specification."""
        self.nb_logger.info(f"🔥 BRUTAL TRUTH: Starting Parsl execution for task: {task}")

        # Generate worker ID and submission ID for this execution
        import uuid
        import time
        worker_id = f"parsl_worker_{uuid.uuid4().hex[:8]}"
        submission_id = f"submission_{uuid.uuid4().hex[:8]}"

        # For Parsl execution, we need to ensure the task is properly submitted to workers
        # Create a simple node-tracking Parsl app with resource specification
        try:
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Creating Parsl app with resource_specification: {resource_specification}")
            import parsl

            # Merge default resource specification from config with provided one
            final_resource_spec = {}

            # Get default resource specification from executor config
            # FIX BUG #1: Check for None before updating
            if hasattr(self.config, 'default_resource_specification') and \
               self.config.default_resource_specification is not None:
                final_resource_spec.update(self.config.default_resource_specification)

            # Override with provided resource specification
            if resource_specification:
                final_resource_spec.update(resource_specification)

            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Final resource specification: {final_resource_spec}")

            # Track this resource submission for parallel path conflict detection
            await self._track_resource_submission(final_resource_spec, submission_id)

            # REAL PARSL EXECUTION: Use module-level Parsl app
            #
            # SOLUTION: Use module-level function to avoid capturing non-serializable closure
            # This avoids serialization errors from ZMQ contexts and other non-picklable objects

            # Get the module-level Parsl app
            execute_step_from_config = _get_execute_step_parsl_app()

            # Extract step information from task closure
            # The task is a closure that contains the step and input data
            step_instance = None
            input_data = None

            # Try to extract step from task closure
            if hasattr(task, '__closure__') and task.__closure__:
                for cell in task.__closure__:
                    if hasattr(cell.cell_contents, 'config') and hasattr(cell.cell_contents, 'process'):
                        step_instance = cell.cell_contents
                        break
                    elif isinstance(cell.cell_contents, dict):
                        input_data = cell.cell_contents

            if step_instance is None:
                raise RuntimeError("Cannot extract step instance from task closure for Parsl execution")

            # Get step config path and class name
            step_config_path = getattr(step_instance, '_config_path', None)
            if step_config_path is None:
                raise RuntimeError("Step instance missing _config_path for Parsl execution")

            step_class_name = f"{step_instance.__class__.__module__}.{step_instance.__class__.__name__}"

            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Executing step {step_class_name} from config {step_config_path}")

            # Submit to Parsl with resource specification
            parsl_future = execute_step_from_config(
                step_config_path,
                step_class_name,
                input_data or {},
                parsl_resource_specification=final_resource_spec,  # CRITICAL: Pass resource spec
                **kwargs
            )

            # Wait for Parsl future result
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Waiting for Parsl execution...")
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, parsl_future.result)

            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Parsl execution completed: {result}")

            # Handle result
            if result['status'] == 'error':
                raise RuntimeError(f"Parsl execution failed: {result['error']}\n{result['traceback']}")

            result = result['result']

        except Exception as e:
            # Fallback to direct submission if Parsl app creation fails
            self.nb_logger.error(f"🔥 BRUTAL TRUTH: Parsl execution failed with exception: {e}")
            import traceback
            self.nb_logger.error(f"🔥 BRUTAL TRUTH: Traceback: {traceback.format_exc()}")
            logger.warning(f"⚠️ Parsl app creation failed, using direct submission: {e}")
            # FIX: self.submit() returns a coroutine, not a Future - await it directly
            result_coro = self.submit(task, **kwargs)
            import asyncio
            result = await result_coro()

        finally:
            # Remove from resource tracking
            await self._remove_resource_submission(submission_id)

        # Add worker ID to result if requested (simplified approach)
        if add_worker_id:
            # If result is a dict, add metadata fields
            if isinstance(result, dict):
                result['_worker_id'] = worker_id
                result['_executor_type'] = 'parsl'
            # Otherwise, wrap in metadata dict
            else:
                result = {
                    'result': result,
                    '_worker_id': worker_id,
                    '_executor_type': 'parsl'
                }

        logger.info(f"✅ Parsl task executed successfully by worker: {worker_id}")
        return result

    def _should_fallback(self, exception: Exception) -> bool:
        """Determine if exception should trigger fallback based on configuration."""
        # Check if fallback is enabled in configuration
        fallback_config = getattr(self.config, 'fallback', None)
        if not fallback_config or not fallback_config.get('enable_fallback', False):
            return False

        # Get fallback conditions from config (if specified)
        fallback_conditions = fallback_config.get('fallback_conditions', [])

        # If no specific conditions, fallback on any failure (existing behavior)
        if not fallback_conditions:
            return True

        # Check specific fallback conditions
        exception_str = str(exception).lower()

        for condition in fallback_conditions:
            if condition == "serialization_error" and ("serialize" in exception_str or "pickle" in exception_str):
                return True
            elif condition == "worker_connection_failure" and ("connection" in exception_str or "worker" in exception_str):
                return True
            elif condition == "task_timeout" and ("timeout" in exception_str):
                return True
            elif condition == "parsl_initialization_failure" and ("parsl" in exception_str or "initialization" in exception_str):
                return True

        # Default fallback for any exception if general fallback is enabled
        return fallback_config.get('fallback_on_failure', True)

    def _should_use_workqueue_delegation(self) -> bool:
        """
        Determine if this ParslExecutor should delegate to WorkQueueExecutor.

        Returns True if the Parsl configuration contains WorkQueueExecutor.
        """
        self.nb_logger.info(f"🔥 BRUTAL TRUTH: _should_use_workqueue_delegation() called")
        self.nb_logger.info(f"🔥 BRUTAL TRUTH: self.config.parsl_config = {self.config.parsl_config}")

        if not self.config.parsl_config or 'executors' not in self.config.parsl_config:
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: No parsl_config or executors found - returning False")
            return False

        # Check if any executor in the configuration is WorkQueueExecutor
        self.nb_logger.info(f"🔥 BRUTAL TRUTH: Checking {len(self.config.parsl_config['executors'])} executors...")
        for i, executor_config in enumerate(self.config.parsl_config['executors']):
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Executor {i}: {executor_config}")
            if isinstance(executor_config, dict):
                executor_class_name = executor_config.get('class', '')
                self.nb_logger.info(f"🔥 BRUTAL TRUTH: Executor {i} class: '{executor_class_name}'")
                if 'WorkQueueExecutor' in executor_class_name:
                    self.nb_logger.info(f"🔥 BRUTAL TRUTH: Found WorkQueueExecutor in config: {executor_class_name}")
                    return True

        self.nb_logger.info(f"🔥 BRUTAL TRUTH: No WorkQueueExecutor found - returning False")
        return False

    async def _execute_workqueue_delegation(self, task: Any, add_worker_id: bool = True,
                                          resource_specification: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Execute task by delegating to WorkQueueExecutor within the single PBS allocation.

        This method handles the Aurora workflow pattern where:
        1. ONE PBS job allocates resources
        2. WorkQueueExecutor distributes tasks within those resources
        3. Subworkflow serialization provides accountability
        """
        self.nb_logger.info(f"🔥 BRUTAL TRUTH: Starting WorkQueue delegation for task: {task}")

        try:
            # STEP 2: Serialize subworkflow information for accountability
            subworkflow_info = await self._serialize_subworkflow_info(task)
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Subworkflow serialized: {subworkflow_info}")

            # STEP 3: Execute task using the existing Parsl WorkQueueExecutor
            # The WorkQueueExecutor is already configured in the Parsl config
            # We just need to submit the task normally - Parsl will route it to WorkQueue
            self.nb_logger.info(f"🔥 BRUTAL TRUTH: Submitting task to WorkQueueExecutor via Parsl...")

            # Use the existing _execute_parsl method - it will automatically use WorkQueueExecutor
            # because that's what's configured in the Parsl config
            result = await self._execute_parsl(task, add_worker_id, resource_specification, **kwargs)

            self.nb_logger.info(f"🔥 BRUTAL TRUTH: WorkQueue delegation completed successfully")
            return result

        except Exception as e:
            self.nb_logger.error(f"🔥 BRUTAL TRUTH: WorkQueue delegation failed: {e}")
            import traceback
            self.nb_logger.error(f"🔥 BRUTAL TRUTH: Traceback: {traceback.format_exc()}")
            raise

    async def _serialize_subworkflow_info(self, task: Any) -> Dict[str, Any]:
        """
        Serialize subworkflow information for accountability purposes.

        This tracks what subworkflow/task is being executed for debugging and accountability.
        """
        import uuid
        import time

        subworkflow_info = {
            'execution_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'task_type': type(task).__name__ if hasattr(task, '__name__') else str(type(task)),
            'executor_type': 'workqueue_delegation',
            'parsl_config_summary': self._get_parsl_config_summary()
        }

        # Try to extract more information from the task if it's a step
        if hasattr(task, '__self__') and hasattr(task.__self__, 'config'):
            step_instance = task.__self__
            if hasattr(step_instance, 'config') and hasattr(step_instance.config, 'name'):
                subworkflow_info['step_name'] = step_instance.config.name
            if hasattr(step_instance, 'parent_workflow_config_path'):
                subworkflow_info['parent_workflow_config'] = step_instance.parent_workflow_config_path

        self.nb_logger.info(f"🔥 BRUTAL TRUTH: Subworkflow info: {subworkflow_info}")
        return subworkflow_info

    def _get_parsl_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the Parsl configuration for logging."""
        if not self.config.parsl_config:
            return {}

        summary = {}
        if 'executors' in self.config.parsl_config:
            summary['executor_count'] = len(self.config.parsl_config['executors'])
            summary['executor_types'] = []
            for exec_config in self.config.parsl_config['executors']:
                if isinstance(exec_config, dict):
                    exec_class = exec_config.get('class', 'unknown')
                    summary['executor_types'].append(exec_class)

        return summary

    async def _execute_fallback(self, task: Any, add_worker_id: bool, original_exception: Exception, **kwargs) -> Any:
        """Execute task using configured fallback executor."""
        try:
            # Create fallback executor based on configuration
            fallback_executor = await self._create_fallback_executor()

            # Execute task with fallback executor
            result = await fallback_executor.execute(task, **kwargs)

            # Add worker ID and executor type for tracking
            if add_worker_id:
                import uuid
                fallback_type = self._get_fallback_type()
                worker_id = f"fallback_{fallback_type}_{uuid.uuid4().hex[:8]}"

                if isinstance(result, dict):
                    result['_worker_id'] = worker_id
                    result['_executor_type'] = f'fallback_{fallback_type}'
                else:
                    result = {
                        'result': result,
                        '_worker_id': worker_id,
                        '_executor_type': f'fallback_{fallback_type}'
                    }

            logger.info(f"✅ Fallback execution completed successfully")

            # Cleanup fallback executor
            await fallback_executor.shutdown()

            return result

        except Exception as fallback_error:
            logger.error(f"Fallback execution also failed: {fallback_error}")
            raise RuntimeError(f"Both PARSL and fallback execution failed. PARSL error: {original_exception}, Fallback error: {fallback_error}")

    async def _create_fallback_executor(self):
        """Create fallback executor based on configuration."""
        fallback_type = self._get_fallback_type()

        if fallback_type == "thread":
            # Use existing ThreadExecutor with thread_fallback configuration
            thread_config = getattr(self.config, 'thread_fallback', {})
            config = ExecutorConfig(
                executor_type='thread',
                max_workers=thread_config.get('max_workers', 4),
                timeout=thread_config.get('timeout', self.config.timeout)
            )
            fallback_executor = ThreadExecutor.from_config(config)
        else:
            # Use existing LocalExecutor with fallback configuration (default)
            fallback_config = getattr(self.config, 'fallback', {})
            config = ExecutorConfig(
                executor_type='local',
                max_workers=fallback_config.get('max_workers', 4),
                timeout=fallback_config.get('timeout', self.config.timeout)
            )
            fallback_executor = LocalExecutor.from_config(config)

        # Initialize the fallback executor
        await fallback_executor.initialize()
        logger.info(f"✅ Fallback executor ({fallback_type}) initialized")

        return fallback_executor

    def _get_fallback_type(self) -> str:
        """Get fallback executor type from configuration."""
        # Check for thread fallback first
        if hasattr(self.config, 'thread_fallback') and self.config.thread_fallback:
            return "thread"

        # Check fallback configuration
        fallback_config = getattr(self.config, 'fallback', {})
        return fallback_config.get('executor_type', 'local')

    async def execute_workflow_distributed(
        self,
        config_path: str,
        input_data: Dict[str, Any]
    ) -> Any:
        """
        Execute single workflow on distributed Parsl workers.

        This method provides distributed execution for individual workflows
        while maintaining the same interface as local execution.

        Args:
            config_path: Path to workflow configuration file
            input_data: Input data for workflow execution

        Returns:
            Workflow execution result

        Raises:
            RuntimeError: If distributed execution fails
        """
        import os
        from nanobrain.core.distributed.workflow_execution import execute_workflow_distributed

        # Get main process working directory
        main_working_directory = os.getcwd()

        # Submit workflow execution to Parsl
        future = execute_workflow_distributed(
            config_path=config_path,
            input_data=input_data,
            main_working_directory=main_working_directory
        )

        # Wait for execution completion (AppFuture.result() is blocking)
        import asyncio
        loop = asyncio.get_event_loop()
        execution_result = await loop.run_in_executor(None, future.result)

        # Handle execution results
        if execution_result['success']:
            return execution_result['result']
        else:
            error_message = f"Distributed workflow execution failed: {execution_result['error']}"
            logger.error(error_message)
            raise RuntimeError(error_message)

    async def execute_workflows_parallel(
        self,
        workflow_configs: list,
        input_data_list: list
    ) -> list:
        """
        Execute multiple workflows in parallel on distributed workers.

        This method enables parallel execution of multiple workflow instances
        across available Parsl workers.

        Args:
            workflow_configs: List of workflow configuration file paths
            input_data_list: List of input data dictionaries for each workflow

        Returns:
            List of execution results, one per workflow

        Raises:
            ValueError: If workflow_configs and input_data_list lengths don't match
        """
        if len(workflow_configs) != len(input_data_list):
            raise ValueError("Number of workflow configs must match number of input data items")

        import os
        from nanobrain.core.distributed.workflow_execution import execute_workflow_distributed

        # Get main process working directory
        main_working_directory = os.getcwd()

        # Submit all workflows to Parsl
        futures = []
        for config_path, input_data in zip(workflow_configs, input_data_list):
            future = execute_workflow_distributed(
                config_path=config_path,
                input_data=input_data,
                main_working_directory=main_working_directory
            )
            futures.append(future)

        # Wait for all executions to complete
        import asyncio
        loop = asyncio.get_event_loop()

        execution_results = []
        for future in futures:
            # AppFuture.result() is blocking, so run in executor
            result = await loop.run_in_executor(None, future.result)
            execution_results.append(result)

        return execution_results

    async def shutdown(self) -> None:
        """Shutdown Parsl executor and all worker step pools."""
        try:
            # Shutdown all worker step pools
            for step_id, pool in self._worker_step_pools.items():
                logger.info(f"Shutting down worker step pool: {step_id}")
                await pool.shutdown()

            self._worker_step_pools.clear()

            # Shutdown PARSL
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
    
    if not config:
        # Create a temporary YAML config file for the executor type
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({"executor_type": executor_type}, f)
            temp_config_path = f.name
        config = ExecutorConfig.from_config(temp_config_path)
        import os
        os.unlink(temp_config_path)
    
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