from typing import Any, List, Optional
from src.ExecutorBase import ExecutorBase
from src.LinkBase import LinkBase
from src.PackageBase import PackageBase
from src.interfaces import IRunnable
from src.WorkingMemory import WorkingMemory
from src.enums import ComponentState
from src.concurrency import CircuitBreaker
from src.DataUnitBase import DataUnitBase

class Step(PackageBase, IRunnable):
    """
    Base class for workflow steps.
    
    Biological analogy: Functional neural circuit.
    Justification: Like how functional neural circuits process specific
    types of information and pass results to other circuits, steps process
    specific operations and pass results to other steps.
    """
    def __init__(self, executor: ExecutorBase, input_sources: List[LinkBase] = None, 
                 output_sink: LinkBase = None, **kwargs):
        # Initialize the base class
        super().__init__(executor, **kwargs)
        
        # Step-specific attributes
        self.input_sources = input_sources or []
        self.output_sink = output_sink
        self.result = None
        self.working_memory = WorkingMemory(**kwargs)
        self.circuit_breaker = CircuitBreaker()
        self.specialization = 0.0  # How specialized for current task (0.0-1.0)
        self.adaptive_network = None  # Network for adaptive processing
        self._state = ComponentState.INACTIVE
        self._running = False
    
    async def execute(self):
        """
        Execute the step's operation.
        
        Biological analogy: Neural circuit activation.
        Justification: Like how neural circuits activate to process
        specific information, steps execute to process specific operations.
        """
        if self._running:
            return self.result
            
        self._running = True
        self._state = ComponentState.ACTIVE
        
        if not self.circuit_breaker.can_execute():
            # Area is inhibited
            self._state = ComponentState.INACTIVE
            self._running = False
            return None
            
        try:
            # Collect inputs from all sources
            inputs = []
            for source in self.input_sources:
                await source.transfer()
                data = source.output.get()
                if data is not None:
                    inputs.append(data)
            
            # Process inputs
            self.result = await self.process(inputs)
            
            # Store in working memory
            self.working_memory.store('last_result', self.result)
            
            # Increase specialization for this type of input
            self.specialization = min(1.0, self.specialization + 0.01)
            
            # Send result to output sink if available
            if self.output_sink and self.result is not None:
                self.output_sink.input.set(self.result)
                await self.output_sink.transfer()
                
            # Record success in circuit breaker
            self.circuit_breaker.record_success()
            self._state = ComponentState.INACTIVE
            return self.result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            
            # Decrease specialization due to failure
            self.specialization = max(0.0, self.specialization - 0.05)
            
            self._state = ComponentState.ERROR
            raise e
            
        finally:
            self._running = False
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process inputs to produce a result.
        
        Biological analogy: Information processing in neural circuits.
        Justification: Like how neural circuits transform inputs into
        outputs through specific processing operations, this method
        transforms input data into output results.
        """
        # Base implementation just returns the first input
        return inputs[0] if inputs else None
    
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
        return self._state
    
    @state.setter
    def state(self, value):
        """
        Set the current state of the step.
        
        Biological analogy: Neural circuit state transition.
        Justification: Like how neural circuits transition between states,
        steps transition between operational states.
        """
        self._state = value
    
    @property
    def running(self) -> bool:
        """
        Get whether the step is currently running.
        
        Biological analogy: Neural circuit activity.
        Justification: Like how neural circuits can be active or inactive,
        steps can be running or not running.
        """
        return self._running
    
    @running.setter
    def running(self, value: bool):
        """
        Set whether the step is currently running.
        
        Biological analogy: Neural circuit activation control.
        Justification: Like how neural circuits can be activated or deactivated,
        steps can be set to running or not running.
        """
        self._running = value