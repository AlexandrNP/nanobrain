import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
import traceback

class FlowMonitor:
    """
    A utility class for monitoring data flow between components.
    """
    def __init__(self, debug_mode=True):
        """
        Initialize the flow monitor.
        
        Args:
            debug_mode: Whether to print debug messages
        """
        self.debug_mode = debug_mode
        self.monitored_components = {}
        self.data_flow_history = []
        self.max_history_entries = 100
        self._monitor_task = None
        self._monitoring = False
        
    def register_component(self, component_name: str, component: Any, 
                          data_getter: Optional[Callable] = None,
                          input_getter: Optional[Callable] = None):
        """
        Register a component to be monitored.
        
        Args:
            component_name: Name of the component
            component: The component object
            data_getter: Optional function to get data from the component
            input_getter: Optional function to get input from the component
        """
        self.monitored_components[component_name] = {
            'component': component,
            'data_getter': data_getter or (lambda c: getattr(c, 'output', None) and c.output.get()),
            'input_getter': input_getter or (lambda c: getattr(c, 'input_sources', {}) and 
                                          {k: v.get() for k, v in c.input_sources.items()}),
            'last_data': None,
            'last_input': None,
            'last_check_time': time.time()
        }
        
        if self.debug_mode:
            print(f"FlowMonitor: Registered component {component_name}")
            
    def log_data_flow(self, source: str, destination: str, data: Any):
        """
        Log a data flow event.
        
        Args:
            source: Source component name
            destination: Destination component name
            data: The data that flowed
        """
        entry = {
            'timestamp': time.time(),
            'source': source,
            'destination': destination,
            'data': data
        }
        
        self.data_flow_history.append(entry)
        
        # Trim history if needed
        if len(self.data_flow_history) > self.max_history_entries:
            self.data_flow_history = self.data_flow_history[-self.max_history_entries:]
            
        if self.debug_mode:
            print(f"FlowMonitor: Data flow from {source} to {destination}: {data}")
            
    async def check_components(self):
        """
        Check all registered components for changes.
        """
        for name, info in self.monitored_components.items():
            try:
                component = info['component']
                
                # Check if component still exists and is valid
                if component is None:
                    continue
                    
                # Get current data and input
                current_data = info['data_getter'](component)
                current_input = info['input_getter'](component)
                
                # Check for data changes
                if current_data != info['last_data']:
                    if self.debug_mode:
                        print(f"FlowMonitor: Component {name} data changed: {current_data}")
                    info['last_data'] = current_data
                    
                # Check for input changes
                if current_input != info['last_input']:
                    if self.debug_mode:
                        print(f"FlowMonitor: Component {name} input changed: {current_input}")
                    info['last_input'] = current_input
                    
                # Update last check time
                info['last_check_time'] = time.time()
                
            except Exception as e:
                if self.debug_mode:
                    print(f"FlowMonitor: Error checking component {name}: {e}")
                    traceback.print_exc()
                    
    async def _monitor_loop(self):
        """
        Monitoring loop that checks components at regular intervals.
        """
        try:
            while self._monitoring:
                await self.check_components()
                await asyncio.sleep(0.5)  # Check every half second
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.debug_mode:
                print(f"FlowMonitor: Error in monitor loop: {e}")
                traceback.print_exc()
                
    def start_monitoring(self):
        """
        Start monitoring all registered components.
        """
        if self._monitoring:
            return  # Already monitoring
            
        self._monitoring = True
        if self.debug_mode:
            print("FlowMonitor: Starting monitoring")
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    def stop_monitoring(self):
        """
        Stop monitoring components.
        """
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self.debug_mode:
            print("FlowMonitor: Stopped monitoring")
            
    def get_data_flow_history(self):
        """
        Get the data flow history.
        
        Returns:
            List of data flow events
        """
        return self.data_flow_history
        
    def apply_patch(self, create_step):
        """
        Apply monitoring patches to the CreateStep components.
        
        Args:
            create_step: The CreateStep instance to patch
        """
        if create_step.command_line:
            self.register_component("CommandLine", create_step.command_line)
            
        if create_step.agent_builder:
            self.register_component("AgentBuilder", create_step.agent_builder)
            
        if create_step.link:
            # Patch the transfer method to log data flow
            original_transfer = create_step.link.transfer
            link = create_step.link
            flow_monitor = self
            
            async def patched_transfer():
                try:
                    if flow_monitor.debug_mode:
                        print("FlowMonitor: LinkDirect.transfer called")
                    
                    # Get the data before transfer
                    source_data = None
                    if hasattr(link.source_step, 'output') and link.source_step.output is not None:
                        source_data = link.source_step.output.get()
                    
                    # Call the original transfer method
                    result = await original_transfer()
                    
                    # Log the data flow
                    if result:
                        flow_monitor.log_data_flow(
                            link.source_step.name,
                            link.sink_step.name,
                            source_data
                        )
                    
                    return result
                except Exception as e:
                    if flow_monitor.debug_mode:
                        print(f"FlowMonitor: Error in patched transfer: {e}")
                        traceback.print_exc()
                    raise
            
            # Replace the original method with our patched version
            create_step.link.transfer = patched_transfer
            
            if flow_monitor.debug_mode:
                print("FlowMonitor: Patched LinkDirect.transfer method")

# Create a global instance
flow_monitor = FlowMonitor(debug_mode=True)

def apply_monitoring(create_step):
    """
    Apply monitoring to a CreateStep instance.
    
    Args:
        create_step: The CreateStep instance to monitor
    """
    flow_monitor.apply_patch(create_step)
    flow_monitor.start_monitoring()
    return flow_monitor 