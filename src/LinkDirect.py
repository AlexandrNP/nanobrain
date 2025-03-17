from typing import Optional, Any
from src.DataUnitBase import DataUnitBase
from src.LinkBase import LinkBase
from src.TriggerBase import TriggerBase
import asyncio

class LinkDirect(LinkBase):
    """
    Direct link implementation that transfers data directly.
    
    Biological analogy: Fast, direct neural pathway.
    Justification: Like how some neural pathways offer rapid, direct transmission
    (e.g., reflexes), direct links provide immediate data transfer between components.
    """
    def __init__(self, 
                 source_step: Any,
                 sink_step: Any,
                 link_id: str = None,
                 trigger_type: str = "data_changed",
                 trigger: Optional[TriggerBase] = None,
                 auto_setup_trigger: bool = True,
                 **kwargs):
        # Set direct link-specific defaults
        kwargs.setdefault('reliability', 0.98)  # Higher reliability than base class
        kwargs.setdefault('transmission_delay', 0.01)  # Faster transmission than base class
        
        # Handle debug mode - accept both 'debug' and 'debug_mode' parameters
        self._debug_mode = kwargs.get('debug_mode', kwargs.get('debug', False))
        kwargs['debug_mode'] = self._debug_mode
        
        # Initialize the base class with source and sink steps
        super().__init__(
            source_step, 
            sink_step, 
            link_id, 
            trigger_type=trigger_type,
            trigger=trigger,
            auto_setup_trigger=auto_setup_trigger,
            **kwargs
        )
        
        # Direct link-specific attributes
        self._test_mode = kwargs.get('test_mode', False)  # Special mode for testing
        
    async def transfer(self):
        """
        Transfers data directly from source to sink.
        
        Biological analogy: Fast, direct synaptic transmission.
        Justification: Like how some synapses use fast ionotropic receptors
        for immediate signal transmission, direct links provide immediate
        data transfer with minimal processing.
        """
        # Get data from source first
        data = None
        if hasattr(self.source_step, 'output') and self.source_step.output is not None:
            data = self.source_step.output.get()
            
        if self._debug_mode:
            print(f"LinkDirect: Transferring data: {data}")
            
        # Check if we have actual data to transfer
        if data is None:
            if self._debug_mode:
                print("LinkDirect: No data to transfer")
            return False
        
        # Get the appropriate input_sources key
        link_id = self.link_id
        
        # Check if we have a valid input source
        if not hasattr(self.sink_step, 'input_sources') or link_id not in self.sink_step.input_sources:
            # Try regular input source registration if possible
            if hasattr(self.sink_step, 'register_input_source') and hasattr(self.source_step, 'output'):
                if self._debug_mode:
                    print(f"LinkDirect: Registering input source with ID {link_id}")
                self.sink_step.register_input_source(link_id, self.source_step.output)
            else:
                if self._debug_mode:
                    print("LinkDirect: Could not transfer data, no valid input source")
                return False
        
        # Use the base class transfer method for the actual data transfer
        return await super().transfer()