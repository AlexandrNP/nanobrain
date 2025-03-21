from typing import Any, List, Optional, Dict, Union, Type, ClassVar
from src.ExecutorBase import ExecutorBase
from src.LinkBase import LinkBase
from src.PackageBase import PackageBase
from src.interfaces import IRunnable
from src.WorkingMemory import WorkingMemory
from src.enums import ComponentState
from src.concurrency import CircuitBreaker
from src.DataUnitBase import DataUnitBase
from src.TriggerAllDataReceived import TriggerAllDataReceived
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import Field, BaseModel
from src.DirectoryTracer import DirectoryTracer
from src.ConfigManager import ConfigManager
from src.Runner import Runner

class Step(BaseTool, PackageBase, IRunnable):
    """
    Base class for workflow steps.
    
    Biological analogy: Functional neural circuit.
    Justification: Like how functional neural circuits process specific
    types of information and pass results to other circuits, steps process
    specific operations and pass results to other steps.
    """
    # Fields required by BaseTool
    name: str = Field(default="")
    description: str = Field(default="")
    return_direct: bool = Field(default=False)
    
    # Fields from PackageBase that need to be defined for Pydantic
    directory_tracer: Any = Field(default=None, exclude=True)
    config_manager: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    dependencies: List = Field(default_factory=list, exclude=True)
    dependency_resolution_state: Any = Field(default=None, exclude=True)
    
    # Fields specific to Step
    executor: Any = Field(default=None, exclude=True)
    input_sources: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    output: Any = Field(default=None, exclude=True)
    result: Any = Field(default=None, exclude=True)
    working_memory: Any = Field(default=None, exclude=True)
    circuit_breaker: Any = Field(default=None, exclude=True)
    specialization: float = Field(default=0.0, exclude=True)
    adaptive_network: Any = Field(default=None, exclude=True)
    state_internal: Any = Field(default=None, exclude=True)
    running_internal: bool = Field(default=False, exclude=True)
    trigger: Any = Field(default=None, exclude=True)
    
    # Pydantic model configuration to allow arbitrary types
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
    }
    
    def __init__(self, executor: ExecutorBase, 
                 input_sources: Dict[str, DataUnitBase] = None, 
                 output: DataUnitBase = None, **kwargs):
        """
        Initialize the step with an executor and optional input sources and output.
        
        Args:
            executor: ExecutorBase instance for running steps
            input_sources: Dictionary mapping IDs to input data sources
            output: Output data unit
            **kwargs: Additional keyword arguments
        """
        # Initialize BaseTool with name and description from kwargs or class info
        tool_name = kwargs.get('name', self.__class__.__name__)
        tool_description = kwargs.get('description', self.__doc__ or f"Execute the {self.__class__.__name__} step")
        
        # Initialize BaseTool first (Pydantic model)
        BaseTool.__init__(
            self,
            name=tool_name,
            description=tool_description,
            return_direct=kwargs.get('return_direct', False)
        )
        
        # Initialize PackageBase components manually
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        self.runner = Runner(executor, **kwargs)
        self.dependencies = []
        self.dependency_resolution_state = ComponentState.INACTIVE
        
        # Store the executor directly
        self.executor = executor
        
        # Step-specific attributes
        self.input_sources = input_sources or {}
        self.output = output
        self.result = None
        self.working_memory = WorkingMemory(**kwargs)
        self.circuit_breaker = CircuitBreaker()
        self.specialization = 0.0  # How specialized for current task (0.0-1.0)
        self.adaptive_network = None  # Network for adaptive processing
        self.state_internal = ComponentState.INACTIVE
        self.running_internal = False
        
        # Create TriggerAllDataReceived for input synchronization
        self.trigger = TriggerAllDataReceived(self)
        
    def register_input_source(self, link_id: str, data_unit: DataUnitBase):
        """
        Register a new input source data unit with a specific ID.
        
        Biological analogy: Synaptic connection formation.
        Justification: Like how neurons form new synaptic connections,
        steps can register new input sources.
        
        Args:
            link_id: The identifier for this input source
            data_unit: The data unit that will provide input
        """
        self.input_sources[link_id] = data_unit
            
    def register_output(self, data_unit: DataUnitBase):
        """
        Register an output data unit.
        
        Biological analogy: Axon terminal formation.
        Justification: Like how neurons form axon terminals to connect
        with target neurons, steps can register output data units.
        
        Args:
            data_unit: The data unit that will receive output
        """
        self.output = data_unit
    
    def execute(self) -> Any:
        """
        Execute the step's processing logic on the inputs.
        
        This method is synchronous and will block until processing is complete.
        It handles triggering, collecting inputs, and distributing outputs.
        
        Biological analogy: Neural activation cycle.
        Justification: Like how neurons collect inputs, process them, and 
        produce outputs in a coordinated cycle, this method orchestrates 
        the complete processing cycle of the step.
        
        Returns:
            The result of the processing
        """
        # Check if already running
        if self.running_internal:
            return None
            
        # Set running state
        self.running_internal = True
        self.state_internal = ComponentState.ACTIVE
        
        # Check if circuit breaker allows execution
        if not self.circuit_breaker.allow_execution():
            self.state_internal = ComponentState.BLOCKED
            self.running_internal = False
            return None
            
        try:
            # Wait for trigger condition (all inputs received)
            if self.trigger and not self.trigger.monitor():
                self.state_internal = ComponentState.WAITING
                self.running_internal = False
                return None
            
            # Collect inputs from all sources
            inputs = {}
            for link_id, data_unit in self.input_sources.items():
                data = data_unit.get()
                if data is not None:
                    inputs[link_id] = data
            
            # Process inputs using executor
            if self.executor:
                self.result = self.executor.execute(self, inputs)
            else:
                # Handle potentially asynchronous process method
                import asyncio
                import inspect
                
                # Check if process is a coroutine function or returns a coroutine
                process_result = self.process(inputs)
                if inspect.iscoroutine(process_result):
                    # It's an async process method, run it with asyncio
                    try:
                        # Try to run in a new event loop
                        self.result = asyncio.run(process_result)
                    except RuntimeError:
                        # If there's already a running event loop (e.g. in tests),
                        # use get_event_loop instead
                        loop = asyncio.get_event_loop()
                        self.result = loop.run_until_complete(process_result)
                else:
                    # It's a synchronous process method
                    self.result = process_result
            
            # Store in working memory
            self.working_memory.store('last_result', self.result)
            
            # Increase specialization for this type of input
            self.specialization = min(1.0, self.specialization + 0.01)
            
            # Send result to output if available
            if self.output and self.result is not None:
                self.output.set(self.result)
                
            # Record success in circuit breaker
            self.circuit_breaker.record_success()
            self.state_internal = ComponentState.INACTIVE
            return self.result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            
            # Decrease specialization due to failure
            self.specialization = max(0.0, self.specialization - 0.05)
            
            self.state_internal = ComponentState.ERROR
            raise e
            
        finally:
            self.running_internal = False
    
    async def process(self, inputs: Dict[str, Any]) -> Any:
        """
        Process inputs to produce a result.
        
        Biological analogy: Information processing in neural circuits.
        Justification: Like how neural circuits transform inputs into
        outputs through specific processing operations, this method
        transforms input data into output results.
        
        Args:
            inputs: A dictionary mapping link IDs to input data
            
        Returns:
            The processed result
        """
        # Base implementation just returns the first input value if available
        return next(iter(inputs.values())) if inputs else None
    
    def _run(self, *args: Any, run_manager: Optional[CallbackManagerForToolRun] = None) -> Any:
        """
        Use the tool synchronously. Implementation required by BaseTool.
        
        Biological analogy: Direct neural circuit activation.
        Justification: Like how neural circuits can be directly activated
        through specific input patterns, this method provides direct
        activation of the step's processing logic.
        """
        # Convert args to the format expected by process
        if len(args) == 1 and isinstance(args[0], list):
            # If a single list is passed, use it as inputs
            inputs = args[0]
        else:
            # Otherwise, use args as inputs
            inputs = args
            
        # Create a dictionary of inputs with auto-generated keys
        inputs_dict = {f"input_{i}": input_val for i, input_val in enumerate(inputs)}
        
        # Run the process method synchronously
        return self.process(inputs_dict)
    
    async def _arun(self, *args: Any, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> Any:
        """
        Use the tool asynchronously. Implementation required by BaseTool.
        
        Biological analogy: Asynchronous neural activation.
        Justification: Like how neural circuits can process information
        asynchronously in parallel with other neural activities, this method
        provides asynchronous activation of the step's processing logic.
        """
        # Convert args to the format expected by process
        if len(args) == 1 and isinstance(args[0], list):
            # If a single list is passed, use it as inputs
            inputs = args[0]
        else:
            # Otherwise, use args as inputs
            inputs = args
            
        # Create a dictionary of inputs with auto-generated keys
        inputs_dict = {f"input_{i}": input_val for i, input_val in enumerate(inputs)}
        
        # Run the process method asynchronously
        return await self.process(inputs_dict)
    
    def get_result(self) -> Any:
        """
        Get the most recent result.
        
        Biological analogy: Neural circuit output.
        Justification: Like how neural circuits maintain their output
        state until new processing occurs, steps maintain their result
        until new execution occurs.
        """
        return self.result
    
    async def invoke(self):
        """
        Alias for execute to conform to IRunnable interface.
        
        Biological analogy: Alternative pathway activation.
        Justification: Like how neural circuits can be activated through
        different pathways, steps can be executed through different methods.
        """
        return await self.execute()
    
    def check_runnable_config(self) -> bool:
        """
        Check if the step is properly configured to run.
        
        Biological analogy: Neural circuit readiness check.
        Justification: Like how neural circuits must have proper connections
        to function, steps must have proper configuration to execute.
        """
        return self.executor is not None and self.executor.can_execute(self.__class__.__name__)
    
    @property
    def adaptability(self) -> float:
        """
        Get the adaptability level of the step.
        
        Biological analogy: Neural plasticity.
        Justification: Like how neural circuits have varying levels of
        plasticity for adaptation, steps have varying levels of adaptability.
        """
        return self.config_manager.adaptability
    
    @adaptability.setter
    def adaptability(self, value: float):
        """
        Set the adaptability level of the step.
        
        Biological analogy: Modulation of neural plasticity.
        Justification: Like how neuromodulators can adjust plasticity levels,
        this setter adjusts the adaptability of the step.
        """
        self.config_manager.adaptability = value
    
    @property
    def state(self):
        """
        Get the current state of the step.
        
        Biological analogy: Neural circuit state.
        Justification: Like how neural circuits have different activation states,
        steps have different operational states.
        """
        return self.state_internal
    
    @state.setter
    def state(self, value):
        """
        Set the current state of the step.
        
        Biological analogy: Neural circuit state transition.
        Justification: Like how neural circuits transition between states,
        steps transition between operational states.
        """
        self.state_internal = value
    
    @property
    def running(self) -> bool:
        """
        Get whether the step is currently running.
        
        Biological analogy: Neural circuit activity.
        Justification: Like how neural circuits can be active or inactive,
        steps can be running or not running.
        """
        return self.running_internal
    
    @running.setter
    def running(self, value: bool):
        """
        Set whether the step is currently running.
        
        Biological analogy: Neural circuit activation control.
        Justification: Like how neural circuits can be activated or deactivated,
        steps can be set to running or not running.
        """
        self.running_internal = value
        
    # PackageBase methods delegated to properties
    def check_dependencies(self) -> bool:
        """
        Checks if all dependencies are satisfied.
        """
        return all(dependency.check_availability() for dependency in self.dependencies)
    
    def check_availability(self) -> bool:
        """
        Checks if this package is available for use by others.
        """
        return (self.runner.state == ComponentState.INACTIVE or 
                self.runner.state == ComponentState.ENHANCED) and not self.runner.is_running
                
    def get_relative_path(self) -> str:
        """Delegate to directory tracer."""
        return self.directory_tracer.get_relative_path()
    
    def get_absolute_path(self) -> str:
        """Delegate to directory tracer."""
        return self.directory_tracer.get_absolute_path()
    
    def get_config(self, class_dir: str = None) -> dict:
        """
        Get configuration for this class.
        """
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)