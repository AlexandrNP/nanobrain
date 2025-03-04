from enums import ExecutorBase, CircuitBreaker, WorkingMemory
from typing import List, Any
from PackageBase import PackageBase
from LinkBase import LinkBase


class Step(PackageBase):
    """
    Represents a step in the workflow.
    
    Biological analogy: Cortical processing area.
    Justification: Like how specialized cortical areas process specific aspects
    of information (e.g., visual features in visual cortex), steps process
    specific aspects of the workflow data.
    """
    def __init__(self, executor: ExecutorBase, input_sources: List[LinkBase] = None, 
                 output_sink: LinkBase = None, **kwargs):
        super().__init__(executor, **kwargs)
        self.input_sources = input_sources or []
        self.output_sink = output_sink
        self.circuit_breaker = CircuitBreaker()
        self.result = None
        self.working_memory = WorkingMemory()  # Working memory for computations
        self.adaptability = 0.5  # Ability to adapt processing (0.0-1.0)
        self.specialization = 0.0  # How specialized for current task (0.0-1.0)
        self.adaptive_network = None  # Network for adaptive processing
    
    async def execute(self):
        """
        Executes the step if circuit breaker allows.
        
        Biological analogy: Activation of a cortical processing area.
        Justification: Like how cortical areas activate to process specific
        inputs when not inhibited, steps execute to process workflow data
        when not blocked by the circuit breaker.
        """
        if not self.circuit_breaker.can_execute():
            # Area is inhibited
            return None
            
        try:
            # Collect inputs from all sources
            inputs = []
            for source in self.input_sources:
                await source.transfer()
                inputs.append(source.output.get())
            
            # Process inputs
            self.result = await self.process(inputs)
            
            # Store in working memory
            self.working_memory.store('last_result', self.result)
            
            # Increase specialization for this type of input
            self.specialization = min(1.0, self.specialization + 0.01)
            
            # Transfer to output if exists
            if self.output_sink:
                await self.output_sink.transfer()
                
            self.circuit_breaker.record_success()
            return self.result
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            
            # Decrease specialization due to failure
            self.specialization = max(0.0, self.specialization - 0.05)
            
            raise e
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Processes inputs to produce result.
        
        Biological analogy: Specific computation of a cortical area.
        Justification: Like how different cortical areas perform specific
        computations on their inputs (e.g., edge detection in V1),
        step subclasses implement specific processing on workflow data.
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def get_result(self) -> Any:
        """
        Returns the result of the step.
        
        Biological analogy: Output of a cortical processing area.
        Justification: Like how cortical areas produce processed outputs that
        can be accessed by other brain regions, steps produce processed results
        that can be accessed by other workflow components.
        """
        return self.result