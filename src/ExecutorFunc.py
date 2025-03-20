import random
import asyncio
from typing import Any, Callable
from src.ExecutorBase import ExecutorBase

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
        Execute the function with the runnable as input.
        
        Biological analogy: Specialized neural computation.
        Justification: Like how specialized neural circuits perform specific
        computations, this executor performs a specific function computation.
        
        Args:
            runnable: The input to the function
            
        Returns:
            The result of the function
        """
        # Check if we have enough energy
        if self.energy_level < self.energy_per_execution:
            raise ValueError("Insufficient energy for execution")
            
        # Check if we have enough reliability
        if self.reliability < self.reliability_threshold:
            raise ValueError("Insufficient reliability for execution")
            
        try:
            # Consume energy
            self.energy_level -= self.energy_per_execution
            
            # Execute the function
            result = self.function(runnable)
            
            # Increase reliability on success
            self.reliability = min(1.0, self.reliability + 0.01)
            
            return result
            
        except Exception as e:
            # Decrease reliability on failure
            self.reliability = max(0.0, self.reliability - 0.05)
            raise e
    
    async def execute_async(self, runnable: Any) -> Any:
        """
        Async wrapper for the execute method.
        
        This method allows the executor to be used in async contexts.
        
        Biological analogy: Neural adaptation to different signaling speeds.
        Justification: Like how neurons can adapt to different signaling speeds
        in various brain regions, this method adapts the execution to async contexts.
        
        Args:
            runnable: The input to process
            
        Returns:
            The result of execution
        """
        # Use run_in_executor to run the synchronous execute method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, runnable)
        
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