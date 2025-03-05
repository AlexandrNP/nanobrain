import random
from typing import Any, Callable
from ExecutorBase import ExecutorBase

class ExecutorFunc:
    """
    Simple executor implementation that runs functions.
    
    Biological analogy: Specialized neuron with specific function.
    Justification: Like how specialized neurons perform specific operations
    (e.g., orientation-selective cells in visual cortex), this executor
    performs specific functional operations.
    """
    def __init__(self, function: Callable = None, **kwargs):
        self.base_executor = ExecutorBase(**kwargs)
        self.function = function
        self.reliability_threshold = 0.3  # Minimum reliability to execute
    
    def execute(self, runnable: Any) -> Any:
        """
        Executes the function on the runnable.
        
        Biological analogy: Specialized neural computation.
        Justification: Like how specialized neurons transform their inputs in
        specific ways (e.g., edge detection), this method transforms inputs
        through a specific function.
        """
        # Check reliability level
        reliability = self.base_executor.get_modulator_effect("reliability")
        if reliability < self.reliability_threshold:
            # Low reliability - execution may fail or be suboptimal
            execution_probability = reliability / self.reliability_threshold
            if random.random() > execution_probability:
                # Execution failed due to low reliability
                self.base_executor.system_modulators.update_from_event("failure", 0.1)
                return None
        
        # Check and consume energy through base executor
        if not self.base_executor.can_execute(runnable.__class__.__name__):
            return None
            
        # Execute function if provided
        if self.function:
            try:
                result = self.function(runnable)
                # Successful execution updates system state
                self.base_executor.system_modulators.update_from_event("success", 0.05)
                return result
            except Exception as e:
                # Failed execution updates system state
                self.base_executor.system_modulators.update_from_event("failure", 0.1)
                raise e
        
        return None
        
    # Delegate methods to base executor
    def can_execute(self, runnable_type: str) -> bool:
        return self.base_executor.can_execute(runnable_type)
    
    def recover_energy(self):
        self.base_executor.recover_energy()
        
    def get_modulator_effect(self, name: str) -> float:
        return self.base_executor.get_modulator_effect(name)
        
    def get_config(self, class_dir: str = None) -> dict:
        return self.base_executor.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        return self.base_executor.update_config(updates, adaptability_threshold)
        
    @property
    def energy_level(self):
        return self.base_executor.energy_level
        
    @energy_level.setter
    def energy_level(self, value):
        self.base_executor.energy_level = value
        
    @property
    def energy_per_execution(self):
        return self.base_executor.energy_per_execution
        
    @energy_per_execution.setter
    def energy_per_execution(self, value):
        self.base_executor.energy_per_execution = value
        
    @property
    def recovery_rate(self):
        return self.base_executor.recovery_rate
        
    @recovery_rate.setter
    def recovery_rate(self, value):
        self.base_executor.recovery_rate = value
        
    @property
    def runnable_types(self):
        return self.base_executor.runnable_types