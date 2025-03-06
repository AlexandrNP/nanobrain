from typing import Any
from src.DataUnitBase import DataUnitBase
from src.ConfigManager import ConfigManager
from src.DirectoryTracer import DirectoryTracer
from src.regulations import ConnectionStrength
import random

class LinkBase:
    """
    Base class for connections between data units.
    
    Biological analogy: Synaptic connection between neurons.
    Justification: Like how synapses connect neurons and transmit signals
    with varying strengths and reliability, links connect data units and
    transfer data with configurable characteristics.
    """
    def __init__(self, input_data: DataUnitBase, output_data: DataUnitBase, **kwargs):
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        self.input = input_data
        self.output = output_data
        self.connection_strength = ConnectionStrength()
        self.reliability = 0.95  # Probability of successful transfer
        self.adaptability = 0.5  # How quickly connection strength changes
    
    def process_signal(self) -> float:
        """
        Processes the signal passing through the link.
        
        Biological analogy: Synaptic signal processing.
        Justification: Like how synapses modify signals based on their
        strength and recent activity, links process data based on their
        connection strength and characteristics.
        """
        # Get input data
        data = self.input.get()
        
        if data is None:
            return 0.0
            
        # Calculate signal strength based on connection strength
        # and input data characteristics
        signal_strength = 0.0
        
        if isinstance(data, (int, float)):
            # For numerical data, use the value directly
            signal_strength = abs(data) * self.connection_strength.get_value()
        elif isinstance(data, str):
            # For string data, use the length
            signal_strength = min(1.0, len(data) / 100) * self.connection_strength.get_value()
        elif isinstance(data, (list, dict)):
            # For collections, use the size
            signal_strength = min(1.0, len(data) / 10) * self.connection_strength.get_value()
        else:
            # For other types, use a default value
            signal_strength = 0.5 * self.connection_strength.get_value()
            
        return signal_strength
    
    async def transfer(self):
        """
        Transfers data from input to output.
        
        Biological analogy: Synaptic transmission.
        Justification: Like how synapses transmit signals from pre-synaptic
        to post-synaptic neurons with varying reliability, links transfer
        data from input to output units with configurable reliability.
        """
        # Get input data
        data = self.input.get()
        
        if data is None:
            return False
            
        # Check reliability
        if random.random() > self.reliability:
            # Transfer failed
            return False
            
        # Transfer data
        self.output.set(data)
        
        # Strengthen connection based on successful transfer
        self.connection_strength.increase(0.01 * self.adaptability)
        
        return True
    
    def recover(self):
        """
        Recovers from failed transfers.
        
        Biological analogy: Synaptic recovery.
        Justification: Like how synapses recover from neurotransmitter
        depletion, links recover from failed transfers.
        """
        # Increase reliability slightly
        self.reliability = min(0.99, self.reliability + 0.01)
    
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration for this class."""
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)