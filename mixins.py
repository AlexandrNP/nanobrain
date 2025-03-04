import os
import time
import asyncio
from typing import Any, Dict
from pydantic import BaseModel
from ExecutorBase import ExecutorBase
from concurrency import ActivationGate
from DataUnitBase import DataUnitBase
from enums import ComponentState
from pathlib import Path
import yaml


class DirectoryTracerMixin:
    """
    Tracks and provides the path of a class relative to the framework package root.
    
    Biological analogy: Place cells in the hippocampus that encode spatial location.
    Justification: Just as place cells allow an organism to know its location in physical space,
    this mixin allows components to know their location in the codebase's structure.
    """
    def __init__(self):
        self.relative_path = self.__class__.__module__.replace('.', os.path.sep)
    
    def get_relative_path(self) -> str:
        """Returns the saved relative path."""
        return self.relative_path
    
    def get_absolute_path(self) -> str:
        """
        Returns the absolute path by finding the package root and combining with relative path.
        
        Biological analogy: Integration of egocentric and allocentric reference frames in navigation.
        Justification: Similar to how the brain integrates relative positional information with
        absolute map-like representations to determine precise locations.
        """
        # Find the package root
        package_root = Path(__file__).parent.parent
        return os.path.join(str(package_root), self.relative_path)
    

class ConfigurableMixin(DirectoryTracerMixin, BaseModel):
    """
    Handles configuration via YAML frontmatter with Pydantic validation.
    
    Biological analogy: Epigenetic mechanisms that control gene expression.
    Justification: Like how epigenetic modifications determine which genes are expressed
    without changing the DNA sequence, configuration parameters determine component
    behavior without changing the underlying code.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}
        self.adaptability = 0.5  # Ability to reconfigure (0.0-1.0)
    
    def get_config(self, class_dir: str = None) -> Dict:
        """
        Looks for <class_name>.yml in the specified directory and returns parameter dictionary.
        Each parameter is supplemented with the property 'type' that references the class name.
        
        Biological analogy: Cellular response to environmental cues.
        Justification: Like how cells read their environment to determine appropriate
        protein expression, components read configuration files to determine behavior.
        """
        if class_dir is None:
            class_dir = self.get_absolute_path()
        
        class_name = self.__class__.__name__
        config_path = os.path.join(class_dir, f"{class_name}.yml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # In real implementation, validate using pydantic
            
            return self.config
        else:
            return {}
    
    def update_config(self, updates: Dict, adaptability_threshold: float = 0.3) -> bool:
        """
        Updates configuration parameters if adaptability is high enough.
        
        Biological analogy: Cellular plasticity - ability to change in response to stimuli.
        Justification: Components with higher adaptability should be more responsive to
        configuration changes, similar to how more plastic neural circuits adapt more readily.
        """
        if self.adaptability >= adaptability_threshold:
            self.config.update(updates)
            return True
        return False
    


class RunnableMixin(ConfigurableMixin):
    """
    Base class for objects that can be executed.
    
    Biological analogy: Neuron with activation potential.
    Justification: Like how neurons can be activated to process and transmit
    signals, runnable components can be executed to process and transmit data.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(**kwargs)
        self.executor = executor
        self.running = False
        self.activation_gate = ActivationGate()  # Neural membrane analog
        self.input_channels = []  # Input connections
        self.output_channels = []  # Output connections
        self.state = ComponentState.INACTIVE
        self.activation_history = []  # Record of activations for plasticity
    
    def check_runnable_config(self) -> bool:
        """
        Checks that class name is in executor's runnable_types set.
        
        Biological analogy: Receptor-ligand specificity checking.
        Justification: Like how a neurotransmitter binds only to neurons with
        the matching receptors, executors can only execute runnables of
        compatible types.
        """
        return self.executor.can_execute(self.__class__.__name__)
    
    async def invoke(self) -> Any:
        """
        Sets running to True, calls executor's execute method, 
        sets running to False, and returns result.
        
        Biological analogy: Neural activation and firing.
        Justification: Like how a neuron integrates inputs, fires if threshold
        is reached, and then enters a refractory period, runnables process
        inputs, execute if conditions are met, and then recover.
        """
        if not self.check_runnable_config():
            return None
        
        # Check activation threshold
        if not self.activation_gate.receive_signal(1.0):  # Input signal
            # Below threshold - no activation
            return None
        
        # Above threshold - activation occurs
        self.running = True
        self.state = ComponentState.ACTIVE
        self.activation_history.append(time.time())
        
        try:
            # Run the executor
            result = await asyncio.coroutine(self.executor.execute)(self)
            
            # Propagate result to all output channels
            for channel in self.output_channels:
                channel.transmit(result)
                
            return result
        except Exception as e:
            # Handle error - enter recovery period
            self.state = ComponentState.RECOVERING
            raise e
        finally:
            # Enter recovery period
            self.running = False
            self.state = ComponentState.RECOVERING
            
            # Schedule recovery
            asyncio.create_task(self._recover())
    
    async def _recover(self):
        """
        Recovers from activation.
        
        Biological analogy: Neural refractory period.
        Justification: Like how neurons have a refractory period after firing
        during which they cannot be activated again, runnables have a recovery
        period after execution.
        """
        await asyncio.sleep(1.0)  # Recovery period duration
        self.state = ComponentState.INACTIVE
        
        # Recover resources in output channels
        for channel in self.output_channels:
            channel.recover()