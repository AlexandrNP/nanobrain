from typing import Any
from enums import DataUnitBase
from regulations import ConnectionStrength, ActivationGate
from ConfigManager import ConfigManager
import random

class LinkBase:
    """
    Base link class that connects two steps in the workflow.
    
    Biological analogy: Combined dendritic and axonal functionality.
    Justification: Like how neural processes handle both input processing (dendrites)
    and output transmission (axons), this class handles both input processing and
    output transmission.
    """
    def __init__(self, input_data: DataUnitBase, output_data: DataUnitBase, **kwargs):
        self.input = input_data
        self.output = output_data
        self.config_manager = ConfigManager(**kwargs)
        self.connection_strength = ConnectionStrength()  # Connection strength
        self.activation_gate = ActivationGate()  # For input processing
        self.adaptability = 0.5  # How easily the connection changes (0.0-1.0)
        self.reliability = 0.95  # Probability of successful transmission
        self.transmission_delay = 0.1  # Base transmission delay
        self.resource_level = 1.0  # Resource capacity for transmission
        self.recovery_rate = 0.1  # Rate of resource recovery
    
    def process_signal(self) -> float:
        """
        Process input to produce a weighted signal value.
        
        Biological analogy: Dendritic integration and scaling.
        Justification: Like how dendrites apply weights to incoming signals before
        integration at the cell body, this method processes and scales input signals.
        """
        # Get raw data
        data = self.input.get()
        
        # Convert to a signal value between 0 and 1
        if data is None:
            signal = 0.0
        elif isinstance(data, (int, float)):
            # Normalize between 0 and 1
            signal = min(1.0, max(0.0, float(data)))
        elif isinstance(data, bool):
            signal = 1.0 if data else 0.0
        else:
            # Complex data type, just indicate presence
            signal = 1.0 if data else 0.0
        
        # Apply connection strength
        weighted_signal = signal * self.connection_strength.strength
        
        # Pass through activation gate
        self.activation_gate.receive_signal(weighted_signal)
        
        return weighted_signal
    
    async def transfer(self):
        """
        Transfers data from input to output with resource management.
        
        Biological analogy: Combined dendritic and axonal transmission.
        Justification: Like how neural processes handle both input integration
        and output transmission with resource constraints, this method handles
        both processing and transmission with resource management.
        """
        # Check reliability and resources
        if random.random() > self.reliability or self.resource_level < 0.1:
            # Transmission failed
            return False
            
        # Process the input signal
        weighted_signal = self.process_signal()
        
        # Consume resources for transmission
        self.resource_level = max(0.0, self.resource_level - 0.1)
        
        # Send processed data to output
        self.output.set(weighted_signal if isinstance(self.input.get(), (int, float)) 
                       else self.input.get())
        
        # Apply Hebbian-like learning if transmission successful
        if weighted_signal > 0:
            self.connection_strength.increase(0.01 * self.adaptability)
        
        return True
    
    def recover(self):
        """
        Recover transmission resources.
        
        Biological analogy: Neurotransmitter reuptake and synthesis.
        Justification: Like how neural synapses recover their transmission
        resources over time, this method restores transmission capacity.
        """
        self.resource_level = min(1.0, self.resource_level + self.recovery_rate)
    
    def get_config(self, class_dir: str = None) -> dict:
        """Delegate to config manager."""
        return self.config_manager.get_config(class_dir)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)