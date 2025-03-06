from typing import Any
from enums import ComponentState
from ExecutorBase import ExecutorBase
from ActivationGate import ActivationGate
from interfaces import IRunnable
from DirectoryTracer import DirectoryTracer
from ConfigManager import ConfigManager
import asyncio
import time


class Runner(IRunnable):
    """
    Base class for objects that can be executed.
    
    Biological analogy: Neuron with activation potential.
    Justification: Like how neurons can be activated to process and transmit
    signals, runnable components can be executed to process and transmit data.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        self.executor = executor
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        
        # Load configuration
        config = self.config_manager.get_config(self.__class__.__name__)
        
        self._running = False
        self.activation_gate = ActivationGate()  # Neural membrane analog
        self.input_channels = config.get('input_channels', [])  # Input connections
        self.output_channels = config.get('output_channels', [])  # Output connections
        self._state = ComponentState.INACTIVE
        self.activation_history = []  # Record of activations for plasticity
        self._adaptability = config.get('adaptability', 0.5)  # Default adaptability
        self.activation_threshold = config.get('activation_threshold', 1.0)
        self.recovery_period = config.get('recovery_period', 1.0)
    
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
        if not self.activation_gate.receive_signal(self.activation_threshold):  # Input signal
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
        await asyncio.sleep(self.recovery_period)  # Recovery period duration
        self.state = ComponentState.INACTIVE
        
        # Recover resources in output channels
        for channel in self.output_channels:
            channel.recover()
            
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration using ConfigManager."""
        class_name = class_dir or self.__class__.__name__
        return self.config_manager.get_config(class_name)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Update configuration using ConfigManager."""
        if self._adaptability >= adaptability_threshold:
            return self.config_manager.update_config(self.__class__.__name__, updates)
        return False
        
    @property
    def adaptability(self) -> float:
        """Get the adaptability level."""
        return self._adaptability
        
    @adaptability.setter
    def adaptability(self, value: float):
        """Set the adaptability level."""
        self._adaptability = max(0.0, min(1.0, value))
        
    @property
    def state(self):
        """Get the current state."""
        return self._state
        
    @state.setter
    def state(self, value):
        """Set the current state."""
        self._state = value
        
    @property
    def running(self) -> bool:
        """Get whether the component is running."""
        return self._running
        
    @running.setter
    def running(self, value: bool):
        """Set whether the component is running."""
        self._running = value 