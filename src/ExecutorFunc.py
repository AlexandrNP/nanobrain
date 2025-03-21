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
        self._reliability = 0.5  # Default reliability
    
    def execute(self, runnable: Any, *args, **kwargs) -> Any:
        """
        Execute the function with the runnable and additional parameters.
        
        Biological analogy: Specialized neural computation.
        Justification: Like how specialized neural circuits perform specific
        computations, this executor performs a specific function computation.
        
        Args:
            runnable: The primary input to the function
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
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
            if callable(getattr(runnable, 'process', None)):
                # If runnable has a process method, call it with args
                if args and 'inputs' not in kwargs:
                    # Assume first arg is inputs if not provided in kwargs
                    kwargs['inputs'] = args[0] if len(args) == 1 else args
                result = runnable.process(**kwargs)
            elif self.function:
                # Use the executor's function with runnable and args
                result = self.function(runnable, *args, **kwargs)
            else:
                # Default: try to call the runnable directly
                result = runnable(*args, **kwargs) if callable(runnable) else runnable
            
            # Increase reliability on success
            self.reliability = min(1.0, self.reliability + 0.01)
            
            return result
            
        except Exception as e:
            # Decrease reliability on failure
            self.reliability = max(0.0, self.reliability - 0.05)
            raise e
    
    async def execute_async(self, runnable: Any, *args, **kwargs) -> Any:
        """
        Async wrapper for the execute method.
        
        This method allows the executor to be used in async contexts.
        
        Biological analogy: Neural adaptation to different signaling speeds.
        Justification: Like how neurons can adapt to different signaling speeds
        in various brain regions, this method adapts the execution to async contexts.
        
        Args:
            runnable: The primary input to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            The result of execution
        """
        # Use run_in_executor to run the synchronous execute method in a thread pool
        loop = asyncio.get_event_loop()
        # Create a lambda to capture all arguments
        func = lambda: self.execute(runnable, *args, **kwargs)
        return await loop.run_in_executor(None, func)
        
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

    # Add reliability property
    @property
    def reliability(self):
        return self._reliability
        
    @reliability.setter
    def reliability(self, value):
        self._reliability = value