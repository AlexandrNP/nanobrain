from typing import Any, Optional, Dict
from src.TriggerBase import TriggerBase
import asyncio
import copy
import time
import json
import hashlib

class TriggerDataHashChanged(TriggerBase):
    """
    Trigger that activates when data has changed.
    
    Biological analogy: Change detection neurons.
    Justification: Similar to how some neurons in the visual system respond specifically
    to changes in the visual field, this trigger responds specifically to changes in data.
    """

    def __init__(self, source_step, runnable=None, **kwargs):
        """
        Initialize a data change trigger.
        
        Args:
            source_step: The step whose output is monitored for changes
            runnable: The entity to run when triggered (typically a link)
        """
        # Only pass runnable to TriggerBase.__init__
        super().__init__(runnable, **kwargs)
        self.source_step = source_step  # Store source_step as a class attribute
        self._last_data = None
        self._last_data_hash = None  # Store a hash for more reliable change detection
        self._sensitivity = kwargs.get('sensitivity', 0.1)  # How sensitive to changes
        self.check_interval = kwargs.get('check_interval', 0.5)  # How often to check for changes
        self._monitor_task = None
        self._monitoring = False
        self._debug_mode = kwargs.get('debug', False)  # Enable debug printing

    def _hash_data(self, data):
        """
        Create a hash of the data to detect changes in complex objects.
        
        Args:
            data: The data to hash
            
        Returns:
            str: Hash of the data
        """
        if data is None:
            return None
            
        try:
            # Try to get a stable representation for hashing
            if isinstance(data, (str, int, float, bool)):
                data_str = str(data)
            elif isinstance(data, (list, tuple, dict)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                # For other objects use their string representation
                data_str = str(data)
                
            # Return a hash of the string representation
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            if self._debug_mode:
                print(f"TriggerDataHashChanged: Error hashing data: {e}")
            return str(data)

    async def check_condition(self) -> bool:
        """
        Check if the data in the source step has changed.
        
        Biological analogy: Novelty detection.
        Justification: Like how the brain's novelty detection system identifies new
        or changed stimuli, this method detects when data has changed from its previous state.
        
        Returns:
            bool: True if the data has changed, False otherwise
        """
        if not self.runnable:
            if self._debug_mode:
                print("TriggerDataHashChanged: No runnable found")
            return False
            
        if not self.source_step:
            if self._debug_mode:
                print("TriggerDataHashChanged: No source step found")
            return False
            
        # Check if the source step has an output
        if not hasattr(self.source_step, 'output') or self.source_step.output is None:
            if self._debug_mode:
                print("TriggerDataHashChanged: Source step has no output")
            return False
            
        # Get current data
        current_data = self.source_step.output.get()
        
        # Hash the current data
        current_hash = self._hash_data(current_data)
        
        # Check if data has changed by comparing hashes
        if current_hash != self._last_data_hash:
            if self._debug_mode:
                print(f"TriggerDataHashChanged: Data changed from {self._last_data} to {current_data}")
            
            # Update our saved data and hash
            self._last_data = copy.deepcopy(current_data) if current_data is not None else None
            self._last_data_hash = current_hash
            return True
            
        return False

    async def monitor(self):
        """
        Continuously monitor for data changes and trigger the runnable if detected.
        
        Biological analogy: Continuous sensory monitoring.
        Justification: Like how sensory systems continuously monitor the environment
        for changes, this method continuously monitors data for changes.
        """
        if await self.check_condition():
            if self._debug_mode:
                print("TriggerDataHashChanged: Condition met, triggering runnable")
                
            # If the runnable has a transfer method, call it
            if hasattr(self.runnable, 'transfer'):
                try:
                    if self._debug_mode:
                        print("TriggerDataHashChanged: Calling transfer on runnable")
                    asyncio.create_task(self.runnable.transfer())
                except Exception as e:
                    print(f"TriggerDataHashChanged: Error triggering runnable: {e}")

    async def _monitor_loop(self):
        """Internal monitoring loop."""
        try:
            while self._monitoring:
                await self.monitor()
                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"TriggerDataHashChanged: Error in monitor loop: {e}")

    def start_monitoring(self):
        """
        Start continuous monitoring for data changes.
        
        Biological analogy: Activating attention system.
        Justification: Like how the brain's attention system activates to monitor
        for specific stimuli, this method activates continuous monitoring for data changes.
        """
        if self._monitoring:
            return  # Already monitoring
            
        self._monitoring = True
        if self._debug_mode:
            print("TriggerDataHashChanged: Starting monitoring")
        
        # Create the task but don't await it - it's meant to run in the background
        # Store the task so it can be properly cancelled later
        loop = asyncio.get_event_loop()
        self._monitor_task = loop.create_task(self._monitor_loop())

    def stop_monitoring(self):
        """
        Stop continuous monitoring for data changes.
        
        Biological analogy: Deactivating attention system.
        Justification: Like how the brain's attention system deactivates when no longer
        needed, this method deactivates continuous monitoring for data changes.
        """
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self._debug_mode:
            print("TriggerDataHashChanged: Stopped monitoring")
            
    # Override runnable property to handle data resets
    @property
    def runnable(self):
        """Get the runnable object."""
        return self._runnable if hasattr(self, '_runnable') else None
        
    @runnable.setter
    def runnable(self, value):
        """
        Set the runnable object.
        
        This also resets the last_data to ensure proper change detection.
        """
        self._runnable = value
        # Reset the last data when the runnable changes
        self._last_data = None
        self._last_data_hash = None

    @property
    def sensitivity(self):
        """Get the sensitivity level."""
        return self._sensitivity
        
    @sensitivity.setter
    def sensitivity(self, value):
        """Set the sensitivity level."""
        self._sensitivity = value
        
    @property
    def adaptation_rate(self):
        """Get the adaptation rate."""
        return self._adaptation_rate if hasattr(self, '_adaptation_rate') else 0.05
        
    @adaptation_rate.setter
    def adaptation_rate(self, value):
        """Set the adaptation rate."""
        self._adaptation_rate = value
        
    @property
    def activation_gate(self):
        """Get the activation gate."""
        return self._activation_gate if hasattr(self, '_activation_gate') else None
        
    @activation_gate.setter
    def activation_gate(self, value):
        """Set the activation gate."""
        self._activation_gate = value 