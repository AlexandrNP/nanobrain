"""
Trigger Integration for ContextVar Guards

Integrates existing Nanobrain triggers with the ContextVar guard system
to enable proper via_trigger wrapping when calling protected methods.

✅ FRAMEWORK COMPLIANCE: No modifications to existing trigger base classes
✅ NON-INTRUSIVE: Uses composition and enhancement patterns
✅ EXACT PATTERN: Implements via_trigger wrapping from reference implementation
"""

from typing import Any, Optional, Callable, Dict
from ..trigger import TriggerBase, DataUnitChangeTrigger
from .contextvar_core import via_trigger
from .nanobrain_guard_mixin import TriggerGuardMixin


class TriggerGuardIntegration:
    """
    ✅ FRAMEWORK COMPLIANCE: External integration for existing triggers
    ✅ EXACT PATTERN: Implements via_trigger wrapping from reference
    
    Provides integration utilities for existing trigger classes to work
    with ContextVar-protected methods without modifying base classes.
    """
    
    @staticmethod
    def wrap_trigger_execution(trigger: TriggerBase, target_method: callable, scope: str = 'step') -> callable:
        """
        ✅ EXACT PATTERN: Wrap trigger execution with via_trigger
        
        Direct implementation of the TriggerManager pattern from reference.
        Wraps target method calls with via_trigger to enable ContextVar protection.
        
        Args:
            trigger: The trigger instance
            target_method: Method to be called by the trigger
            scope: ContextVar scope for protection
            
        Returns:
            Via_trigger wrapped method that can be called by the trigger
        """
        return via_trigger(scope)(target_method)
    
    @staticmethod
    def enhance_data_unit_change_trigger(trigger: DataUnitChangeTrigger) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Enhance existing trigger without modification
        
        Wraps the trigger's execution method with via_trigger protection.
        Works with existing DataUnitChangeTrigger instances.
        
        Args:
            trigger: DataUnitChangeTrigger instance to enhance
        """
        if not hasattr(trigger, 'bound_step') or not trigger.bound_step:
            return  # Nothing to enhance if no bound step
        
        bound_step = trigger.bound_step
        
        # Check if step has guard functionality
        if isinstance(bound_step, TriggerGuardMixin):
            # Get via_trigger wrapped methods for protected methods
            protected_methods = bound_step.get_protected_methods() if hasattr(bound_step, 'get_protected_methods') else []
            scope = bound_step.get_guard_scope() if hasattr(bound_step, 'get_guard_scope') else 'step'
            
            # Store original wrapped methods for trigger use
            trigger._wrapped_methods = {}
            for method_name in protected_methods:
                if hasattr(bound_step, method_name):
                    wrapped_method = bound_step.get_via_trigger_wrapper(method_name)
                    if wrapped_method:
                        trigger._wrapped_methods[method_name] = wrapped_method
        
        elif hasattr(bound_step, 'process'):
            # For steps without explicit guard mixin, wrap the process method
            scope = getattr(bound_step, '_guard_scope', 'step')
            wrapped_process = via_trigger(scope)(bound_step.process)
            trigger._wrapped_process = wrapped_process
    
    @staticmethod
    def get_protected_method_wrapper(trigger: TriggerBase, method_name: str) -> Optional[callable]:
        """
        Get via_trigger wrapped version of a protected method for trigger use.
        
        Args:
            trigger: Trigger instance
            method_name: Name of the method to wrap
            
        Returns:
            Via_trigger wrapped method or None if not available
        """
        # Check for stored wrapped methods
        if hasattr(trigger, '_wrapped_methods'):
            return trigger._wrapped_methods.get(method_name)
        
        # Check for specific wrapped method
        wrapped_attr = f'_wrapped_{method_name}'
        if hasattr(trigger, wrapped_attr):
            return getattr(trigger, wrapped_attr)
        
        return None


class EnhancedDataUnitChangeTrigger(DataUnitChangeTrigger):
    """
    ✅ FRAMEWORK COMPLIANCE: Enhanced trigger with automatic guard integration
    
    Extended version of DataUnitChangeTrigger that automatically integrates
    with ContextVar guards when bound to protected steps.
    
    Inherits all functionality from base DataUnitChangeTrigger while adding
    guard integration capabilities.
    """
    
    def bind_action(self, target_step):
        """
        ✅ FRAMEWORK COMPLIANCE: Enhanced bind_action with guard integration
        
        Binds the trigger to a target step and automatically sets up
        via_trigger wrapping if the step has guard protection.
        """
        # Call parent bind_action
        super().bind_action(target_step)
        
        # Enhance with guard integration
        TriggerGuardIntegration.enhance_data_unit_change_trigger(self)
    
    def _execute_callback(self, data: Any) -> None:
        """
        ✅ EXACT PATTERN: Execute callback with via_trigger wrapping
        
        Executes the bound action using via_trigger wrapped methods
        if the target step has guard protection.
        """
        if not self.bound_step:
            return
        
        # Try to use wrapped method first
        wrapped_process = TriggerGuardIntegration.get_protected_method_wrapper(self, 'process')
        if wrapped_process:
            # ✅ EXACT PATTERN: Use via_trigger wrapped method
            try:
                wrapped_process(data)
            except Exception as e:
                self.nb_logger.error(f"Error in via_trigger execution: {e}")
                raise
        else:
            # Fallback to original method (unprotected or manual wrapping)
            try:
                if hasattr(self.bound_step, 'process'):
                    self.bound_step.process(data)
            except RuntimeError as e:
                if "may only be invoked via" in str(e):
                    # This is a guard protection error - wrap and retry
                    scope = getattr(self.bound_step, '_guard_scope', 'step')
                    wrapped_method = via_trigger(scope)(self.bound_step.process)
                    wrapped_method(data)
                else:
                    raise


def integrate_trigger_with_guards(trigger: TriggerBase, target_step: Any) -> None:
    """
    ✅ FRAMEWORK COMPLIANCE: General purpose trigger-guard integration
    
    Integrates any trigger with a target step that has guard protection.
    Works with existing trigger instances without requiring inheritance.
    
    Args:
        trigger: Trigger instance to integrate
        target_step: Target step that may have guard protection
    """
    # Bind the trigger to the target step if not already bound
    if hasattr(trigger, 'bind_action') and not getattr(trigger, 'bound_step', None):
        trigger.bind_action(target_step)
    
    # Apply guard integration enhancement
    if isinstance(trigger, DataUnitChangeTrigger):
        TriggerGuardIntegration.enhance_data_unit_change_trigger(trigger)


def create_guard_aware_trigger(trigger_class, config: Dict[str, Any], target_step: Any) -> TriggerBase:
    """
    ✅ FRAMEWORK COMPLIANCE: Factory for creating guard-aware triggers
    
    Creates a trigger instance with automatic guard integration.
    
    Args:
        trigger_class: Trigger class to instantiate
        config: Trigger configuration
        target_step: Target step for the trigger
        
    Returns:
        Trigger instance with guard integration applied
    """
    # Create trigger using framework patterns
    if hasattr(trigger_class, 'from_config'):
        # Use from_config if available
        if isinstance(config, dict):
            import tempfile
            import yaml
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
                yaml.dump(config, temp_file, default_flow_style=False)
                temp_file_path = temp_file.name
            
            try:
                trigger = trigger_class.from_config(temp_file_path)
            finally:
                os.unlink(temp_file_path)
        else:
            trigger = trigger_class.from_config(config)
    else:
        # Fallback to constructor
        trigger = trigger_class(**config)
    
    # Integrate with guards
    integrate_trigger_with_guards(trigger, target_step)
    
    return trigger


# ✅ CONVENIENCE FUNCTIONS: Helper functions for common integration patterns
def enhance_existing_trigger(trigger: TriggerBase) -> None:
    """
    Enhance an existing trigger instance with guard integration.
    
    Args:
        trigger: Existing trigger instance to enhance
    """
    if isinstance(trigger, DataUnitChangeTrigger):
        TriggerGuardIntegration.enhance_data_unit_change_trigger(trigger)


def is_trigger_guard_aware(trigger: TriggerBase) -> bool:
    """
    Check if a trigger has guard integration capabilities.
    
    Args:
        trigger: Trigger instance to check
        
    Returns:
        True if trigger has guard integration, False otherwise
    """
    return (hasattr(trigger, '_wrapped_methods') or 
            hasattr(trigger, '_wrapped_process') or
            isinstance(trigger, EnhancedDataUnitChangeTrigger))


def get_trigger_wrapped_methods(trigger: TriggerBase) -> Dict[str, callable]:
    """
    Get all via_trigger wrapped methods available to a trigger.
    
    Args:
        trigger: Trigger instance
        
    Returns:
        Dictionary of method name to wrapped method mappings
    """
    wrapped_methods = {}
    
    # Check for stored wrapped methods
    if hasattr(trigger, '_wrapped_methods'):
        wrapped_methods.update(trigger._wrapped_methods)
    
    # Check for specific wrapped methods
    for attr_name in dir(trigger):
        if attr_name.startswith('_wrapped_') and callable(getattr(trigger, attr_name)):
            method_name = attr_name[9:]  # Remove '_wrapped_' prefix
            wrapped_methods[method_name] = getattr(trigger, attr_name)
    
    return wrapped_methods 