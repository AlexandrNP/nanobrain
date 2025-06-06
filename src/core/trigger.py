"""
Trigger System for NanoBrain Framework

Provides event-driven processing capabilities for Steps.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Set, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of triggers."""
    DATA_UPDATED = "data_updated"
    ALL_DATA_RECEIVED = "all_data_received"
    TIMER = "timer"
    MANUAL = "manual"
    CONDITION = "condition"


class TriggerConfig(BaseModel):
    """Configuration for triggers."""
    model_config = ConfigDict(use_enum_values=True)
    
    trigger_type: TriggerType = TriggerType.DATA_UPDATED
    debounce_ms: int = Field(default=100, ge=0)
    max_frequency_hz: float = Field(default=10.0, gt=0)
    condition: Optional[str] = None
    timer_interval_ms: Optional[int] = None


class TriggerBase(ABC):
    """
    Base class for triggers that control when Steps execute.
    
    Biological analogy: Action potential threshold mechanisms.
    Justification: Like how neurons fire when threshold conditions are met,
    triggers activate steps when specific conditions are satisfied.
    """
    
    def __init__(self, config: Optional[TriggerConfig] = None, **kwargs):
        self.config = config or TriggerConfig()
        self.name = kwargs.get('name', self.__class__.__name__)
        self._is_active = False
        self._callbacks: List[Callable] = []
        self._last_trigger_time = 0.0
        self._debounce_task: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        pass
    
    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop monitoring for trigger conditions."""
        pass
    
    async def add_callback(self, callback: Callable) -> None:
        """Add a callback to be executed when triggered."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    async def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    async def trigger(self, data: Any = None) -> None:
        """Execute trigger with rate limiting and debouncing."""
        current_time = asyncio.get_event_loop().time()
        
        # Check frequency limit
        time_since_last = current_time - self._last_trigger_time
        min_interval = 1.0 / self.config.max_frequency_hz
        
        if time_since_last < min_interval:
            logger.debug(f"Trigger {self.name} rate limited")
            return
        
        # Cancel previous debounce task
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        
        # Create debounced execution
        if self.config.debounce_ms > 0:
            self._debounce_task = asyncio.create_task(
                self._debounced_execute(data)
            )
        else:
            await self._execute_callbacks(data)
    
    async def _debounced_execute(self, data: Any) -> None:
        """Execute callbacks after debounce delay."""
        try:
            await asyncio.sleep(self.config.debounce_ms / 1000.0)
            await self._execute_callbacks(data)
        except asyncio.CancelledError:
            logger.debug(f"Debounced execution cancelled for {self.name}")
    
    async def _execute_callbacks(self, data: Any) -> None:
        """Execute all registered callbacks."""
        self._last_trigger_time = asyncio.get_event_loop().time()
        
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in trigger callback: {e}")
    
    @property
    def is_active(self) -> bool:
        """Check if trigger is actively monitoring."""
        return self._is_active


class DataUpdatedTrigger(TriggerBase):
    """
    Trigger that fires when data units are updated.
    """
    
    def __init__(self, data_units: List[Any], config: Optional[TriggerConfig] = None, **kwargs):
        config = config or TriggerConfig(trigger_type=TriggerType.DATA_UPDATED)
        super().__init__(config, **kwargs)
        self.data_units = data_units
        self._monitoring_tasks: List[asyncio.Task] = []
        
    async def start_monitoring(self) -> None:
        """Start monitoring data units for updates."""
        if self._is_active:
            return
            
        self._is_active = True
        
        # Monitor each data unit
        for data_unit in self.data_units:
            task = asyncio.create_task(self._monitor_data_unit(data_unit))
            self._monitoring_tasks.append(task)
        
        logger.debug(f"DataUpdatedTrigger {self.name} started monitoring {len(self.data_units)} data units")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring data units."""
        self._is_active = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.debug(f"DataUpdatedTrigger {self.name} stopped monitoring")
    
    async def _monitor_data_unit(self, data_unit: Any) -> None:
        """Monitor a single data unit for changes."""
        last_update_time = 0.0
        
        try:
            while self._is_active:
                # Check if data unit has metadata about last update
                if hasattr(data_unit, 'get_metadata'):
                    current_update_time = await data_unit.get_metadata('last_updated', 0.0)
                    
                    if current_update_time > last_update_time:
                        last_update_time = current_update_time
                        data = await data_unit.get()
                        await self.trigger(data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.debug(f"Monitoring cancelled for data unit in {self.name}")
        except Exception as e:
            logger.error(f"Error monitoring data unit in {self.name}: {e}")


class AllDataReceivedTrigger(TriggerBase):
    """
    Trigger that fires when all required data units have data.
    """
    
    def __init__(self, data_units: List[Any], config: Optional[TriggerConfig] = None, **kwargs):
        config = config or TriggerConfig(trigger_type=TriggerType.ALL_DATA_RECEIVED)
        super().__init__(config, **kwargs)
        self.data_units = data_units
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self) -> None:
        """Start monitoring for all data received."""
        if self._is_active:
            return
            
        self._is_active = True
        self._monitoring_task = asyncio.create_task(self._monitor_all_data())
        logger.debug(f"AllDataReceivedTrigger {self.name} started monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_active = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"AllDataReceivedTrigger {self.name} stopped monitoring")
    
    async def _monitor_all_data(self) -> None:
        """Monitor until all data units have data."""
        try:
            while self._is_active:
                all_have_data = True
                data_dict = {}
                
                for i, data_unit in enumerate(self.data_units):
                    data = await data_unit.get()
                    if data is None:
                        all_have_data = False
                        break
                    data_dict[f"input_{i}"] = data
                
                if all_have_data:
                    await self.trigger(data_dict)
                    # Stop monitoring after successful trigger
                    break
                
                # Check again after a short delay
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.debug(f"AllDataReceivedTrigger {self.name} monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in AllDataReceivedTrigger {self.name}: {e}")


class TimerTrigger(TriggerBase):
    """
    Trigger that fires at regular intervals.
    """
    
    def __init__(self, interval_ms: int, config: Optional[TriggerConfig] = None, **kwargs):
        config = config or TriggerConfig(
            trigger_type=TriggerType.TIMER,
            timer_interval_ms=interval_ms
        )
        super().__init__(config, **kwargs)
        self.interval_ms = interval_ms
        self._timer_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self) -> None:
        """Start timer monitoring."""
        if self._is_active:
            return
            
        self._is_active = True
        self._timer_task = asyncio.create_task(self._timer_loop())
        logger.debug(f"TimerTrigger {self.name} started with {self.interval_ms}ms interval")
    
    async def stop_monitoring(self) -> None:
        """Stop timer monitoring."""
        self._is_active = False
        
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        
        logger.debug(f"TimerTrigger {self.name} stopped")
    
    async def _timer_loop(self) -> None:
        """Timer loop that triggers at intervals."""
        try:
            while self._is_active:
                await asyncio.sleep(self.interval_ms / 1000.0)
                if self._is_active:  # Check again after sleep
                    await self.trigger()
        except asyncio.CancelledError:
            logger.debug(f"TimerTrigger {self.name} loop cancelled")
        except Exception as e:
            logger.error(f"Error in TimerTrigger {self.name}: {e}")


class ManualTrigger(TriggerBase):
    """
    Trigger that fires only when manually activated.
    """
    
    def __init__(self, config: Optional[TriggerConfig] = None, **kwargs):
        config = config or TriggerConfig(trigger_type=TriggerType.MANUAL)
        super().__init__(config, **kwargs)
        
    async def start_monitoring(self) -> None:
        """Start monitoring (no-op for manual trigger)."""
        self._is_active = True
        logger.debug(f"ManualTrigger {self.name} ready for manual activation")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_active = False
        logger.debug(f"ManualTrigger {self.name} deactivated")
    
    async def fire(self, data: Any = None) -> None:
        """Manually fire the trigger."""
        if self._is_active:
            await self.trigger(data)
        else:
            logger.warning(f"ManualTrigger {self.name} not active")


def create_trigger(config: Union[Dict[str, Any], TriggerConfig]) -> TriggerBase:
    """
    Factory function to create triggers.
    
    Args:
        config: Trigger configuration (dict or TriggerConfig)
        
    Returns:
        Configured trigger instance
    """
    if isinstance(config, dict):
        config = TriggerConfig(**config)
    
    # Handle both enum and string values (due to use_enum_values=True)
    trigger_type = config.trigger_type
    if isinstance(trigger_type, str):
        trigger_type = TriggerType(trigger_type)
    
    if trigger_type == TriggerType.DATA_UPDATED:
        return DataUpdatedTrigger([], config)
    elif trigger_type == TriggerType.ALL_DATA_RECEIVED:
        return AllDataReceivedTrigger([], config)
    elif trigger_type == TriggerType.TIMER:
        interval_ms = config.timer_interval_ms or 1000
        return TimerTrigger(interval_ms, config)
    elif trigger_type == TriggerType.MANUAL:
        return ManualTrigger(config)
    else:
        raise ValueError(f"Unknown trigger type: {trigger_type}") 