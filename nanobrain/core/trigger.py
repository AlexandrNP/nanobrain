"""
Trigger System for NanoBrain Framework

Provides event-driven processing capabilities for Steps.
Enhanced with mandatory from_config pattern implementation.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, Set, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from .component_base import FromConfigBase, ComponentConfigurationError, ComponentDependencyError
# Import logging system
from .logging_system import get_logger, get_system_log_manager

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
    name: str = ""


class TriggerBase(FromConfigBase, ABC):
    """
    Base class for triggers that control when Steps execute.
    Enhanced with mandatory from_config pattern implementation.
    
    Biological analogy: Action potential threshold mechanisms.
    Justification: Like how neurons fire when threshold conditions are met,
    triggers activate steps when specific conditions are satisfied.
    """
    
    COMPONENT_TYPE = "trigger"
    REQUIRED_CONFIG_FIELDS = ['trigger_type']
    OPTIONAL_CONFIG_FIELDS = {
        'debounce_ms': 100,
        'max_frequency_hz': 10.0,
        'condition': None,
        'timer_interval_ms': None,
        'name': ''
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return TriggerConfig - ONLY method that differs from other components"""
        return TriggerConfig
    
    @classmethod
    def extract_component_config(cls, config: TriggerConfig) -> Dict[str, Any]:
        """Extract Trigger configuration"""
        return {
            'trigger_type': config.trigger_type,
            'debounce_ms': getattr(config, 'debounce_ms', 100),
            'max_frequency_hz': getattr(config, 'max_frequency_hz', 10.0),
            'condition': getattr(config, 'condition', None),
            'timer_interval_ms': getattr(config, 'timer_interval_ms', None),
            'name': getattr(config, 'name', '')
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve Trigger dependencies"""
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False)
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize Trigger with resolved dependencies"""
        self.config = config
        self.name = component_config.get('name') or self.__class__.__name__
        self._is_active = False
        self._callbacks: List[Callable] = []
        self._last_trigger_time = 0.0
        self._debounce_task: Optional[asyncio.Task] = None
        
        # Internal state tracking
        self._creation_time = time.time()
        self._trigger_count = 0
        self._rate_limited_count = 0
        self._callback_error_count = 0
        self._total_callback_time = 0.0
        
        # Initialize centralized logging system
        self.enable_logging = dependencies.get('enable_logging', True)
        if self.enable_logging:
            # Use centralized logging system
            self.nb_logger = get_logger(self.name, category="triggers", debug_mode=dependencies.get('debug_mode', False))
            
            # Register with system log manager
            system_manager = get_system_log_manager()
            system_manager.register_component("triggers", self.name, self, {
                "trigger_type": component_config['trigger_type'].value if hasattr(component_config['trigger_type'], 'value') else str(component_config['trigger_type']),
                "debounce_ms": component_config['debounce_ms'],
                "max_frequency_hz": component_config['max_frequency_hz'],
                "enable_logging": True
            })
        else:
            self.nb_logger = None
    
    # TriggerBase inherits FromConfigBase.__init__ which prevents direct instantiation
        
    def _get_internal_state(self) -> Dict[str, Any]:
        """Get comprehensive internal state for logging."""
        uptime = time.time() - self._creation_time
        avg_callback_time = self._total_callback_time / max(self._trigger_count, 1)
        
        return {
            "is_active": self._is_active,
            "trigger_count": self._trigger_count,
            "rate_limited_count": self._rate_limited_count,
            "callback_error_count": self._callback_error_count,
            "callback_count": len(self._callbacks),
            "uptime_seconds": uptime,
            "last_trigger_time": self._last_trigger_time,
            "avg_callback_time_ms": avg_callback_time * 1000 if self._trigger_count > 0 else 0,
            "trigger_type": self.config.trigger_type.value if hasattr(self.config.trigger_type, 'value') else str(self.config.trigger_type),
            "debounce_ms": self.config.debounce_ms,
            "max_frequency_hz": self.config.max_frequency_hz,
            "success_rate": (self._trigger_count - self._callback_error_count) / max(self._trigger_count, 1)
        }
    
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
            
            # Log callback addition
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"Callback added to trigger: {self.name}",
                                   operation="add_callback",
                                   callback_count=len(self._callbacks),
                                   callback_name=getattr(callback, '__name__', str(callback)),
                                   internal_state=self._get_internal_state())
    
    async def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            
            # Log callback removal
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"Callback removed from trigger: {self.name}",
                                   operation="remove_callback",
                                   callback_count=len(self._callbacks),
                                   callback_name=getattr(callback, '__name__', str(callback)),
                                   internal_state=self._get_internal_state())
    
    async def trigger(self, data: Any = None) -> None:
        """Execute trigger with rate limiting and debouncing."""
        current_time = asyncio.get_event_loop().time()
        
        # Check frequency limit
        time_since_last = current_time - self._last_trigger_time
        min_interval = 1.0 / self.config.max_frequency_hz
        
        if time_since_last < min_interval:
            self._rate_limited_count += 1
            logger.debug(f"Trigger {self.name} rate limited")
            
            # Log rate limiting
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Trigger rate limited: {self.name}",
                                   operation="rate_limited",
                                   time_since_last=time_since_last,
                                   min_interval=min_interval,
                                   rate_limited_count=self._rate_limited_count)
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
            
            # Log debounce cancellation
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Debounced execution cancelled: {self.name}",
                                   operation="debounce_cancelled")
    
    async def _execute_callbacks(self, data: Any) -> None:
        """Execute all registered callbacks."""
        self._last_trigger_time = asyncio.get_event_loop().time()
        self._trigger_count += 1
        
        start_time = time.time()
        successful_callbacks = 0
        
        # Log trigger activation
        if self.enable_logging and self.nb_logger:
            self.nb_logger.log_trigger_activation(
                trigger_name=self.name,
                trigger_type=self.config.trigger_type.value if hasattr(self.config.trigger_type, 'value') else str(self.config.trigger_type),
                conditions={"data_type": type(data).__name__ if data is not None else "None"},
                activated=True
            )
            
            self.nb_logger.info(f"Trigger activated: {self.name}",
                               operation="trigger_activated",
                               callback_count=len(self._callbacks),
                               data_type=type(data).__name__ if data is not None else "None",
                               trigger_count=self._trigger_count)
        
        for i, callback in enumerate(self._callbacks):
            try:
                callback_start = time.time()
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
                callback_duration = time.time() - callback_start
                self._total_callback_time += callback_duration
                successful_callbacks += 1
                
                # Log successful callback execution
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.debug(f"Callback executed successfully: {self.name}[{i}]",
                                       operation="callback_success",
                                       callback_index=i,
                                       callback_name=getattr(callback, '__name__', str(callback)),
                                       duration_ms=callback_duration * 1000)
                
            except Exception as e:
                self._callback_error_count += 1
                logger.error(f"Error in trigger callback: {e}")
                
                # Log callback error
                if self.enable_logging and self.nb_logger:
                    self.nb_logger.error(f"Callback error in trigger: {self.name}[{i}]",
                                       operation="callback_error",
                                       callback_index=i,
                                       callback_name=getattr(callback, '__name__', str(callback)),
                                       error=str(e),
                                       error_type=type(e).__name__)
        
        # Log execution summary
        total_duration = time.time() - start_time
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Trigger execution completed: {self.name}",
                               operation="trigger_completed",
                               successful_callbacks=successful_callbacks,
                               total_callbacks=len(self._callbacks),
                               total_duration_ms=total_duration * 1000,
                               internal_state=self._get_internal_state())
    
    @property
    def is_active(self) -> bool:
        """Check if trigger is actively monitoring."""
        return self._is_active


class DataUnitChangeTrigger(TriggerBase):
    """
    Event-driven trigger that fires when data unit changes occur.
    Uses change listener system for immediate response without polling.
    """
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'DataUnitChangeTrigger':
        """Mandatory from_config implementation for DataUnitChangeTrigger"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DataUnitChangeTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'data_unit': kwargs.get('data_unit'),
            'event_type': kwargs.get('event_type', 'data_unit_updated')
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DataUnitChangeTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_unit = dependencies.get('data_unit')
        self.event_type = dependencies.get('event_type', 'data_unit_updated')
        self.bound_actions = []
        
        if not self.data_unit:
            raise ComponentConfigurationError("DataUnitChangeTrigger requires data_unit")
    
    def bind_action(self, action_func: Callable) -> None:
        """Bind action to trigger for immediate execution on data unit changes"""
        if action_func not in self.bound_actions:
            self.bound_actions.append(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Bound action to trigger {self.name}")
    
    def unbind_action(self, action_func: Callable) -> None:
        """Unbind action from trigger"""
        if action_func in self.bound_actions:
            self.bound_actions.remove(action_func)
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Unbound action from trigger {self.name}")
    
    async def start_monitoring(self) -> None:
        """Start monitoring by registering as change listener"""
        if self._is_active:
            return
            
        self._is_active = True
        
        # Register with data unit's change listener system
        self.data_unit.register_change_listener(self._on_data_unit_changed)
        
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Started monitoring data unit {self.data_unit.name}")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring by unregistering change listener"""
        if not self._is_active:
            return
            
        self._is_active = False
        
        # Unregister from data unit's change listener system
        self.data_unit.unregister_change_listener(self._on_data_unit_changed)
        
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Stopped monitoring data unit {self.data_unit.name}")
    
    async def _on_data_unit_changed(self, change_event: Dict[str, Any]) -> None:
        """Handle data unit change event"""
        try:
            # Check if event type matches (if specified)
            if hasattr(self, 'event_type') and self.event_type != 'all':
                if change_event.get('operation') != self.event_type:
                    return
            
            # Create trigger event
            trigger_event = {
                'trigger_id': getattr(self, 'trigger_id', self.name),
                'event_type': self.event_type,
                'data_unit': self.data_unit.name,
                'change_event': change_event,
                'timestamp': time.time()
            }
            
            # Execute bound actions immediately (no polling delay)
            for action in self.bound_actions:
                await action(trigger_event)
            
            # Also execute callbacks for compatibility
            await self._execute_callbacks(change_event)
            
            if self.enable_logging and self.nb_logger:
                self.nb_logger.info(f"🔥 Trigger {self.name} fired for data unit change")
                
        except Exception as e:
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"❌ Error in DataUnitChangeTrigger: {e}")


class DataUpdatedTrigger(TriggerBase):
    """
    Trigger that fires when data units are updated.
    """
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'DataUpdatedTrigger':
        """Mandatory from_config implementation for DataUpdatedTrigger"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve DataUpdatedTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'data_units': kwargs.get('data_units', [])
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize DataUpdatedTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_units = dependencies.get('data_units', [])
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
        
        # Log monitoring start
        if self.enable_logging and self.nb_logger:
            self.nb_logger.info(f"Monitoring started: {self.name}",
                               operation="start_monitoring",
                               data_units_count=len(self.data_units),
                               data_unit_names=[getattr(du, 'name', str(du)) for du in self.data_units],
                               internal_state=self._get_internal_state())
    
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
        
        # Log monitoring stop with statistics
        if self.enable_logging and self.nb_logger:
            uptime = time.time() - self._creation_time
            self.nb_logger.info(f"Monitoring stopped: {self.name}",
                               operation="stop_monitoring",
                               uptime_seconds=uptime,
                               final_stats={
                                   "total_triggers": self._trigger_count,
                                   "rate_limited": self._rate_limited_count,
                                   "callback_errors": self._callback_error_count
                               },
                               internal_state=self._get_internal_state())
    
    async def _monitor_data_unit(self, data_unit: Any) -> None:
        """Monitor a single data unit for changes."""
        last_update_time = 0.0
        data_unit_name = getattr(data_unit, 'name', str(data_unit))
        
        # Log monitoring start for this data unit
        if self.enable_logging and self.nb_logger:
            self.nb_logger.debug(f"Started monitoring data unit: {data_unit_name}",
                               operation="start_data_unit_monitoring",
                               data_unit_name=data_unit_name,
                               trigger_name=self.name)
        
        try:
            while self._is_active:
                # Check if data unit has metadata about last update
                if hasattr(data_unit, 'get_metadata'):
                    current_update_time = await data_unit.get_metadata('last_updated', 0.0)
                    
                    if current_update_time > last_update_time:
                        last_update_time = current_update_time
                        data = await data_unit.get()
                        
                        # Log data change detection
                        if self.enable_logging and self.nb_logger:
                            self.nb_logger.info(f"Data change detected: {data_unit_name}",
                                               operation="data_change_detected",
                                               data_unit_name=data_unit_name,
                                               trigger_name=self.name,
                                               update_time=current_update_time,
                                               data_type=type(data).__name__ if data is not None else "None")
                        
                        await self.trigger(data)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            logger.debug(f"Monitoring cancelled for data unit in {self.name}")
            
            # Log monitoring cancellation
            if self.enable_logging and self.nb_logger:
                self.nb_logger.debug(f"Data unit monitoring cancelled: {data_unit_name}",
                                   operation="data_unit_monitoring_cancelled",
                                   data_unit_name=data_unit_name,
                                   trigger_name=self.name)
        except Exception as e:
            logger.error(f"Error monitoring data unit in {self.name}: {e}")
            
            # Log monitoring error
            if self.enable_logging and self.nb_logger:
                self.nb_logger.error(f"Error monitoring data unit: {data_unit_name}",
                                   operation="data_unit_monitoring_error",
                                   data_unit_name=data_unit_name,
                                   trigger_name=self.name,
                                   error=str(e),
                                   error_type=type(e).__name__)


class AllDataReceivedTrigger(TriggerBase):
    """
    Trigger that fires when all required data units have data.
    """
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'AllDataReceivedTrigger':
        """Mandatory from_config implementation for AllDataReceivedTrigger"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve AllDataReceivedTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        return {
            **base_deps,
            'data_units': kwargs.get('data_units', [])
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize AllDataReceivedTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.data_units = dependencies.get('data_units', [])
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
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'TimerTrigger':
        """Mandatory from_config implementation for TimerTrigger"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
        
    @classmethod
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve TimerTrigger dependencies"""
        base_deps = super().resolve_dependencies(component_config, **kwargs)
        interval_ms = kwargs.get('interval_ms') or component_config.get('timer_interval_ms', 1000)
        return {
            **base_deps,
            'interval_ms': interval_ms
        }
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize TimerTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        self.interval_ms = dependencies.get('interval_ms', 1000)
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
    
    @classmethod
    def from_config(cls, config: TriggerConfig, **kwargs) -> 'ManualTrigger':
        """Mandatory from_config implementation for ManualTrigger"""
        logger = get_logger(f"{cls.__name__}.from_config")
        logger.info(f"Creating {cls.__name__} from configuration")
        
        # Step 1: Validate configuration schema
        cls.validate_config_schema(config)
        
        # Step 2: Extract component-specific configuration  
        component_config = cls.extract_component_config(config)
        
        # Step 3: Resolve dependencies
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        
        # Step 4: Create instance
        instance = cls.create_instance(config, component_config, dependencies)
        
        # Step 5: Post-creation initialization
        instance._post_config_initialization()
        
        logger.info(f"Successfully created {cls.__name__}")
        return instance
    
    def _init_from_config(self, config: TriggerConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ManualTrigger with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)
        
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


def create_trigger(config: Union[Dict[str, Any], TriggerConfig], **kwargs) -> TriggerBase:
    """
    MANDATORY from_config factory for all trigger types
    
    Args:
        config: Trigger configuration (dict or TriggerConfig)
        **kwargs: Framework-provided dependencies
        
    Returns:
        TriggerBase instance created via from_config
        
    Raises:
        ValueError: If trigger type is unknown
        ComponentConfigurationError: If configuration is invalid
    """
    logger = get_logger("trigger.factory")
    logger.info(f"Creating trigger via mandatory from_config")
    
    if isinstance(config, dict):
        config = TriggerConfig(**config)
    
    # Handle both enum and string values (due to use_enum_values=True)
    trigger_type = config.trigger_type
    if isinstance(trigger_type, str):
        trigger_type = TriggerType(trigger_type)
    
    try:
        if trigger_type == TriggerType.DATA_UPDATED:
            trigger_class = DataUpdatedTrigger
        elif trigger_type == TriggerType.ALL_DATA_RECEIVED:
            trigger_class = AllDataReceivedTrigger
        elif trigger_type == TriggerType.TIMER:
            trigger_class = TimerTrigger
        elif trigger_type == TriggerType.MANUAL:
            trigger_class = ManualTrigger
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
        
        # Create instance via from_config
        instance = trigger_class.from_config(config, **kwargs)
        
        logger.info(f"Successfully created {trigger_class.__name__} via from_config")
        return instance
        
    except Exception as e:
        raise ValueError(f"Failed to create trigger '{trigger_type}' via from_config: {e}") 