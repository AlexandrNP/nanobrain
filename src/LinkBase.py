from typing import Any, Optional
from src.DataUnitBase import DataUnitBase
from src.ConfigManager import ConfigManager
from src.DirectoryTracer import DirectoryTracer
from src.regulations import ConnectionStrength
from src.ActivationGate import ActivationGate
from src.TriggerBase import TriggerBase
# Remove direct imports of specific trigger classes
import random
import asyncio

class LinkBase:
    """
    Base class for connections between data units.
    
    Biological analogy: Synaptic connection between neurons.
    Justification: Like how synapses connect neurons and transmit signals
    with varying strengths and reliability, links connect data units and
    transfer data with configurable characteristics.
    """
    def __init__(self, 
                 source_step: Any,  # Should be Step but would create circular import
                 sink_step: Any,    # Should be Step but would create circular import
                 link_id: str = None,
                 trigger_type: str = "data_changed",
                 trigger: Optional[TriggerBase] = None,
                 auto_setup_trigger: bool = True,
                 **kwargs):
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        
        # Store steps
        self.source_step = source_step
        self.sink_step = sink_step
        self.link_id = link_id or f"link_{id(self)}"
        
        # Debug mode - accept both 'debug' and 'debug_mode' parameters
        self._debug_mode = kwargs.get('debug_mode', kwargs.get('debug', False))
        self.debug_mode = self._debug_mode  # Add a public attribute for compatibility
        
        # Verify steps have output and input_sources
        if not hasattr(source_step, 'output') or source_step.output is None:
            raise ValueError("Source step must have an output data unit")
            
        if not hasattr(sink_step, 'input_sources'):
            raise ValueError("Sink step must have input_sources dictionary attribute")
            
        # Register this link with the sink step
        self.sink_step.register_input_source(self.link_id, self.source_step.output)
        
        # Configure transfer characteristics
        self.connection_strength = ConnectionStrength()
        self.reliability = kwargs.get('reliability', 0.95)  # Probability of successful transfer
        self.adaptability = kwargs.get('adaptability', 0.5)  # How quickly connection strength changes
        self.activation_gate = ActivationGate()
        
        # Set up trigger
        self._trigger_type = trigger_type
        self.trigger = trigger  # Allow passing a pre-configured trigger
        
        # Set up trigger if auto_setup_trigger is True and no trigger was provided
        if auto_setup_trigger and self.trigger is None:
            self._setup_trigger(**kwargs)
        
        # Last transferred data for comparison
        self._last_transferred_data = None
            
        # Track if this link is being monitored
        self._monitoring = False
        # Track if data has been transferred
        self._data_transferred = False
        
        # Start monitoring for data changes if we have a trigger
        if self.trigger is not None:
            asyncio.create_task(self.start_monitoring())
    
    def _setup_trigger(self, **kwargs):
        """
        Set up the trigger for this link.
        
        Args:
            **kwargs: Additional arguments to pass to the trigger constructor.
        """
        # Import specific trigger classes here to avoid circular imports
        from src.TriggerDataUpdated import TriggerDataUpdated
        from src.TriggerDataHashChanged import TriggerDataHashChanged
        
        if self._trigger_type == "data_changed":
            # Use TriggerDataUpdated for DataUnitBase classes with 'updated' flag
            self.trigger = TriggerDataUpdated(source_step=self.source_step, runnable=self, **kwargs)
        elif self._trigger_type == "data_hash_changed":
            # Use TriggerDataHashChanged for monitoring based on data hash
            self.trigger = TriggerDataHashChanged(source_step=self.source_step, runnable=self, **kwargs) 
        else:
            # Default to TriggerDataUpdated
            if self._debug_mode:
                print(f"LinkBase: Unknown trigger type '{self._trigger_type}', using TriggerDataUpdated")
            self.trigger = TriggerDataUpdated(source_step=self.source_step, runnable=self, **kwargs)
        
        if self._debug_mode:
            print(f"LinkBase: Created trigger of type {self.trigger.__class__.__name__}")
    
    def process_signal(self) -> float:
        """
        Processes the signal passing through the link.
        
        Biological analogy: Synaptic signal processing.
        Justification: Like how synapses modify signals based on their
        strength and recent activity, links process data based on their
        connection strength and characteristics.
        """
        # Get input data from source step
        data = self.source_step.output.get()
        
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
        Transfers data from source to sink.
        
        Biological analogy: Synaptic transmission.
        Justification: Like how synapses transfer signals from pre-synaptic
        to post-synaptic neurons with varying reliability, links transfer
        data with configurable reliability.
        """
        # Check if we're allowed to execute the transfer
        if not self.activation_gate.receive_signal(0.5):
            return False
            
        # Process the signal
        signal_strength = self.process_signal()
        
        # If signal is too weak, transfer fails
        if signal_strength < 0.1:
            return False
            
        # Determine if transfer succeeds based on reliability
        transfer_succeeds = random.random() < self.reliability
        
        if transfer_succeeds:
            # Get data from source output
            data = self.source_step.output.get()
            
            # Skip if the data is the same as the last transferred data
            if data == self._last_transferred_data:
                if self._debug_mode:
                    print("LinkBase: Data unchanged, skipping transfer")
                return False
                
            # Store the current data as the last transferred data
            self._last_transferred_data = data
            
            # If data exists, transfer it to sink input
            if data is not None:
                # The data is already accessible to the sink step via the registered data unit
                # Mark successful transfer
                self._data_transferred = True
                
                # Strengthen connection due to successful transfer
                self.connection_strength.strengthen(self.adaptability)
                
                # Trigger the sink step to process the new data
                try:
                    # If the sink step has process method, execute it directly 
                    if hasattr(self.sink_step, 'process'):
                        if isinstance(data, str):
                            asyncio.create_task(self.sink_step.process([data]))
                        else:
                            asyncio.create_task(self.sink_step.process([data]))
                    else:
                        # Otherwise use the general execute method
                        asyncio.create_task(self.sink_step.execute())
                except Exception as e:
                    print(f"Error in transfer execution: {e}")
                
                return True
        else:
            # Weaken connection due to failed transfer
            self.connection_strength.weaken(self.adaptability)
            
            # Recover from failure
            await self.recover()
            return False
            
    async def recover(self):
        """
        Recovers from failed transfers.
        
        Biological analogy: Synaptic recovery.
        Justification: Like how synapses recover from neurotransmitter
        depletion, links recover from failed transfers.
        """
        # Give the activation gate time to recover
        await asyncio.sleep(0.1)
        
    async def start_monitoring(self):
        """
        Start monitoring the trigger condition.
        
        Biological analogy: Synaptic vigilance.
        Justification: Like how synapses remain vigilant for signals,
        links monitor for conditions to trigger transfer.
        """
        if self._monitoring or self.trigger is None:
            return
            
        self._monitoring = True
        
        # Create a task to monitor in the background
        asyncio.create_task(self._monitor_loop())
    
    async def _monitor_loop(self):
        """
        Internal method to continuously monitor for trigger condition.
        """
        # Continuously monitor for trigger condition
        while self._monitoring and self.trigger is not None:
            try:
                # Check if the trigger is the enhanced TriggerDataUpdated with wait_for_data_change method
                if hasattr(self.trigger, 'wait_for_data_change'):
                    # Use the more efficient wait_for_data_change method
                    data_changed = await self.trigger.wait_for_data_change(timeout=0.5)
                    if data_changed:
                        # Transfer data when trigger condition is met
                        await self.transfer()
                else:
                    # Wait for the trigger condition the traditional way
                    if await self.trigger.monitor():
                        # Transfer data when trigger condition is met
                        await self.transfer()
                
                # Brief pause to prevent CPU overuse
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                await asyncio.sleep(0.1)  # Longer pause after error
            
    async def stop_monitoring(self):
        """
        Stop monitoring the trigger condition.
        
        Biological analogy: Synaptic dormancy.
        Justification: Like how synapses can become dormant when not needed,
        links can stop monitoring when no longer active.
        """
        self._monitoring = False
        
    def has_data_transferred(self) -> bool:
        """
        Check if data has been transferred through this link.
        
        Biological analogy: Synaptic transmission status.
        Justification: Like how synapses can indicate whether they have
        recently transmitted signals, links can indicate if they have
        transferred data.
        """
        return self._data_transferred
        
    def reset_transfer_status(self):
        """
        Reset the data transfer status.
        
        Biological analogy: Synaptic reset.
        Justification: Like how synapses reset after signal transmission,
        links can reset their transfer status.
        """
        self._data_transferred = False
    
    def get_config(self, class_dir: str = None) -> dict:
        """Get configuration for this class."""
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)

    def start_monitoring_sync(self):
        """
        Start monitoring for data changes synchronously (non-awaitable version).
        
        This method is used when we need to start monitoring from a non-async context.
        """
        asyncio.create_task(self.start_monitoring())