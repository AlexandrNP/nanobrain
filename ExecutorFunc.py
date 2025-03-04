import random
from typing import Any, Callable
from ExecutorBase import ExecutorBase
from enums import RunnableMixin


class ExecutorFunc(ExecutorBase):
    """
    Simple executor implementation that runs functions.
    
    Biological analogy: Specialized neuron with specific function.
    Justification: Like how specialized neurons perform specific operations
    (e.g., orientation-selective cells in visual cortex), this executor
    performs specific functional operations.
    """
    def __init__(self, function: Callable = None, **kwargs):
        super().__init__(**kwargs)
        self.function = function
        self.reliability_threshold = 0.3  # Minimum reliability to execute
    
    def execute(self, runnable: 'RunnableMixin') -> Any:
        """
        Executes the function on the runnable.
        
        Biological analogy: Specialized neural computation.
        Justification: Like how specialized neurons transform their inputs in
        specific ways (e.g., edge detection), this method transforms inputs
        through a specific function.
        """
        # Check reliability level
        reliability = self.get_modulator_effect("reliability")
        if reliability < self.reliability_threshold:
            # Low reliability - execution may fail or be suboptimal
            execution_probability = reliability / self.reliability_threshold
            if random.random() > execution_probability:
                # Execution failed due to low reliability
                self.system_modulators.update_from_event("failure", 0.1)
                return None
        
        # Consume energy
        self.energy_level -= self.energy_per_execution
        
        # Execute function if provided, otherwise just return None
        if self.function:
            try:
                result = self.function(runnable)
                # Successful execution updates system state
                self.system_modulators.update_from_event("success", 0.05)
                return result
            except Exception as e:
                # Failed execution updates system state
                self.system_modulators.update_from_event("failure", 0.1)
                raise e
        
        return None