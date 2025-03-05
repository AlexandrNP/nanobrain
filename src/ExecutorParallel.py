import asyncio
from typing import Any, List
from ExecutorBase import ExecutorBase

class ExecutorParallel:
    """
    Executor implementation that runs tasks in parallel.
    
    Biological analogy: Parallel processing in neural networks.
    Justification: Like how different neural pathways can process information
    simultaneously (e.g., parallel visual pathways), this executor can run
    multiple tasks concurrently.
    """
    def __init__(self, max_concurrent: int = 5, **kwargs):
        self.base_executor = ExecutorBase(**kwargs)
        self.max_concurrent = max_concurrent
        self.active_tasks: List[asyncio.Task] = []
        self.base_executor.energy_per_execution *= 1.2  # Higher energy cost for parallel
        self.reliability_threshold = 0.4  # Higher reliability needed
        
    async def execute(self, runnable: Any) -> Any:
        """
        Executes the runnable in parallel if possible.
        
        Biological analogy: Distributed neural computation.
        Justification: Like how the brain processes information through
        multiple parallel pathways, this executor distributes computation
        across concurrent tasks.
        """
        # Check reliability level
        reliability = self.base_executor.get_modulator_effect("reliability")
        if reliability < self.reliability_threshold:
            # Low reliability - execution may fail
            self.base_executor.system_modulators.update_from_event("failure", 0.1)
            return None
            
        # Clean up completed tasks
        self.active_tasks = [task for task in self.active_tasks if not task.done()]
        
        # Check if we can start new task
        if len(self.active_tasks) >= self.max_concurrent:
            # Wait for a task to complete
            if self.active_tasks:
                await asyncio.wait(self.active_tasks, return_when=asyncio.FIRST_COMPLETED)
                
        # Check and consume energy through base executor
        if not self.base_executor.can_execute(runnable.__class__.__name__):
            return None
            
        try:
            # Create and start new task
            task = asyncio.create_task(runnable.invoke())
            self.active_tasks.append(task)
            
            result = await task
            
            # Successful execution
            self.base_executor.system_modulators.update_from_event("success", 0.05)
            return result
            
        except Exception as e:
            # Failed execution
            self.base_executor.system_modulators.update_from_event("failure", 0.1)
            raise e
            
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