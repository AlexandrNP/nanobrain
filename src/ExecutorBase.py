from typing import Any, Set
from src.ConfigManager import ConfigManager
from src.regulations import SystemModulator
from src.DirectoryTracer import DirectoryTracer


class ExecutorBase:
    """
    Base executor class to execute Runnable objects.
    
    Biological analogy: Neurotransmitter systems controlling neural activation.
    Justification: Like how different neurotransmitter systems (e.g., glutamatergic,
    cholinergic) control the activation of different neural circuits, executor
    classes control the activation of different types of runnables.
    """
    def __init__(self, **kwargs):
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        self.runnable_types: Set[str] = set()
        self.energy_level = 1.0  # Metabolic energy available
        self.energy_per_execution = 0.1  # Energy cost per execution
        self.recovery_rate = 0.05  # Energy recovery rate
        self.system_modulators = SystemModulator()
    
    def can_execute(self, runnable_type: str) -> bool:
        """
        Checks if this executor can handle the specified runnable type.
        
        Biological analogy: Receptor specificity in neural signaling.
        Justification: Like how neurons have specific receptors that respond
        only to certain neurotransmitters, executors have specific types of
        runnables they can process.
        """
        # Check type compatibility
        type_compatible = runnable_type in self.runnable_types
        
        # Check energy level
        energy_sufficient = self.energy_level >= self.energy_per_execution
        
        return type_compatible and energy_sufficient
    
    def execute(self, runnable: 'RunnableMixin') -> Any:
        """
        Execute the runnable.
        
        Biological analogy: Neural computation.
        Justification: Like how neurons compute outputs from inputs,
        executors compute results from runnables.
        
        Args:
            runnable: The runnable to execute
            
        Returns:
            The result of execution
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def recover_energy(self):
        """
        Recovers energy over time.
        
        Biological analogy: Metabolic recovery processes.
        Justification: Like how neurons recover their energy reserves after
        activity through metabolic processes, executors recover their
        computational resources over time.
        """
        self.energy_level = min(1.0, self.energy_level + self.recovery_rate)
        
    def get_modulator_effect(self, name: str) -> float:
        """
        Gets the effect of a system modulator on execution.
        
        Biological analogy: Neuromodulatory effects on neural circuits.
        Justification: Like how different neuromodulators (dopamine, serotonin)
        affect neural circuits in different ways, system modulators affect
        execution parameters differently.
        """
        modulator_level = self.system_modulators.get_modulator(name)
        
        if name == "performance":
            # Performance affects execution speed
            return modulator_level
        elif name == "reliability":
            # Reliability affects error handling
            return modulator_level
        elif name == "adaptability":
            # Adaptability affects learning and adjustment
            return modulator_level
        
        return 0.5  # Neutral effect
        
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration for this class."""
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)