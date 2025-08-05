"""
✅ FRAMEWORK COMPLIANCE: Mixin approach - NO base class modifications
✅ COMPOSITION PATTERN: Add guard functionality without inheritance changes

Provides ContextVar guard functionality through composition rather than
modifying existing Nanobrain base classes.
"""

from typing import Dict, Any, Optional, List, Union
from .contextvar_core import via_trigger, only_via_trigger, is_trigger_active
from .nanobrain_guard_config import NanobrainTriggerGuardConfig


class TriggerGuardMixin:
    """
    ✅ FRAMEWORK COMPLIANCE: Mixin for adding ContextVar guard functionality
    ✅ ZERO INTRUSION: Can be added to any class without breaking inheritance
    
    This mixin provides ContextVar-based trigger protection without modifying
    existing Nanobrain class hierarchies. Uses pure composition patterns.
    """
    
    def __init_guard_system__(self, guard_config: Optional[Union[Dict[str, Any], str, NanobrainTriggerGuardConfig]] = None):
        """
        Initialize guard system for this component.
        
        ✅ FRAMEWORK COMPLIANCE: Uses standard configuration patterns
        ✅ OPTIONAL: Only activated if guard_config is provided
        
        Args:
            guard_config: Guard configuration (NanobrainTriggerGuardConfig object, dict, file path, or None)
        """
        if guard_config:
            try:
                if isinstance(guard_config, NanobrainTriggerGuardConfig):
                    # ✅ FRAMEWORK COMPLIANCE: Already a proper ConfigBase object
                    self._guard_config = guard_config
                    if self._guard_config.should_apply_guards():
                        self._apply_contextvar_guards()
                elif isinstance(guard_config, dict) and guard_config.get('enabled', False):
                    # ✅ FRAMEWORK COMPLIANCE: Create temporary file for ConfigBase compliance
                    import tempfile
                    import yaml
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
                        yaml.dump(guard_config, temp_file, default_flow_style=False)
                        temp_file_path = temp_file.name
                    
                    try:
                        self._guard_config = NanobrainTriggerGuardConfig.from_config(temp_file_path)
                        if self._guard_config.should_apply_guards():
                            self._apply_contextvar_guards()
                    finally:
                        os.unlink(temp_file_path)
                        
                elif isinstance(guard_config, str):
                    # ✅ FRAMEWORK COMPLIANCE: Load from file path
                    self._guard_config = NanobrainTriggerGuardConfig.from_config(guard_config)
                    if self._guard_config.should_apply_guards():
                        self._apply_contextvar_guards()
                else:
                    self._guard_config = None
            except Exception as e:
                # ✅ FRAMEWORK COMPLIANCE: Graceful degradation
                import warnings
                warnings.warn(
                    f"Failed to initialize guard system for {self.__class__.__name__}: {e}. "
                    f"Component will function without guard protection.",
                    RuntimeWarning
                )
                self._guard_config = None
        else:
            self._guard_config = None
    
    def _apply_contextvar_guards(self) -> None:
        """
        ✅ EXACT PATTERN: Apply only_via_trigger decorators to configured methods
        Runtime application without modifying class definitions.
        
        Wraps configured methods with ContextVar protection at runtime.
        """
        if not self._guard_config or not self._guard_config.should_apply_guards():
            return
            
        scope = self._guard_config.scope
        protected_methods = self._guard_config.protected_methods
        
        # Store original methods for introspection and potential restoration
        if not hasattr(self, '_original_methods'):
            self._original_methods = {}
        
        for method_name in protected_methods:
            if hasattr(self, method_name):
                original_method = getattr(self, method_name)
                
                # Skip if already protected
                if getattr(original_method, '_nanobrain_guard_protected', False):
                    continue
                
                # Store original method
                self._original_methods[method_name] = original_method
                
                # ✅ EXACT PATTERN: Wrap with only_via_trigger
                protected_method = only_via_trigger(scope, f"{self.__class__.__name__}.{method_name}")(original_method)
                
                # Add metadata for introspection
                protected_method._nanobrain_guard_original = original_method
                protected_method._nanobrain_guard_scope = scope
                protected_method._nanobrain_guard_protected = True
                protected_method._nanobrain_guard_method_name = method_name
                
                # Apply protection
                setattr(self, method_name, protected_method)
                
                # Log guard application if configured
                if self._guard_config.log_guard_events:
                    self._log_guard_event(f"Applied ContextVar protection to {method_name}")
    
    def get_via_trigger_wrapper(self, method_name: str) -> Optional[callable]:
        """
        ✅ FRAMEWORK COMPLIANCE: Get via_trigger wrapped version of method
        
        For use by triggers when calling protected methods.
        Provides the properly wrapped method for trigger execution.
        
        Args:
            method_name: Name of the method to wrap
            
        Returns:
            Via_trigger wrapped method or None if not available
        """
        if not self._guard_config:
            return None
            
        # Get original method (unwrapped version)
        original_method = self._original_methods.get(method_name)
        if not original_method:
            # Method might not be protected, try to get current method
            if hasattr(self, method_name):
                current_method = getattr(self, method_name)
                original_method = getattr(current_method, '_nanobrain_guard_original', current_method)
            else:
                return None
        
        # ✅ EXACT PATTERN: Return via_trigger wrapped version
        return via_trigger(self._guard_config.scope)(original_method)
    
    def is_method_protected(self, method_name: str) -> bool:
        """
        Check if a method is protected by ContextVar guards.
        
        Args:
            method_name: Name of the method to check
            
        Returns:
            True if method is protected, False otherwise
        """
        if not hasattr(self, method_name):
            return False
            
        method = getattr(self, method_name)
        return getattr(method, '_nanobrain_guard_protected', False)
    
    def get_guard_scope(self) -> Optional[str]:
        """
        Get the guard scope for this component.
        
        Returns:
            Guard scope ('step', 'workflow', 'agent') or None if no guards
        """
        return self._guard_config.scope if self._guard_config else None
    
    def get_protected_methods(self) -> List[str]:
        """
        Get list of methods protected by guards.
        
        Returns:
            List of protected method names
        """
        if not self._guard_config:
            return []
        return self._guard_config.protected_methods.copy()
    
    def is_trigger_currently_active(self) -> bool:
        """
        Check if trigger is currently active for this component's scope.
        
        Returns:
            True if trigger is active, False otherwise
        """
        if not self._guard_config:
            return False
        return is_trigger_active(self._guard_config.scope)
    
    def remove_guard_protection(self, method_name: str) -> bool:
        """
        Remove guard protection from a specific method.
        
        ✅ FRAMEWORK COMPLIANCE: Allows dynamic guard management
        
        Args:
            method_name: Name of the method to unprotect
            
        Returns:
            True if protection was removed, False if method wasn't protected
        """
        if not self.is_method_protected(method_name):
            return False
        
        # Restore original method
        original_method = self._original_methods.get(method_name)
        if original_method:
            setattr(self, method_name, original_method)
            
            # Log guard removal if configured
            if self._guard_config and self._guard_config.log_guard_events:
                self._log_guard_event(f"Removed ContextVar protection from {method_name}")
            
            return True
        
        return False
    
    def restore_all_methods(self) -> None:
        """
        Remove all guard protection and restore original methods.
        
        ✅ FRAMEWORK COMPLIANCE: Complete cleanup capability
        """
        if not hasattr(self, '_original_methods'):
            return
        
        for method_name, original_method in self._original_methods.items():
            setattr(self, method_name, original_method)
        
        # Log restoration if configured
        if self._guard_config and self._guard_config.log_guard_events:
            self._log_guard_event("Restored all methods to original state")
        
        # Clear stored original methods
        self._original_methods.clear()
    
    def _log_guard_event(self, message: str) -> None:
        """
        Log guard-related events if logging is enabled.
        
        Args:
            message: Event message to log
        """
        if hasattr(self, 'nb_logger') and self.nb_logger:
            self.nb_logger.debug(f"[TRIGGER_GUARD] {message}")
        elif self._guard_config and self._guard_config.debug_mode:
            print(f"[TRIGGER_GUARD] {self.__class__.__name__}: {message}")
    
    def get_guard_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the guard system for this component.
        
        Returns:
            Dictionary containing guard configuration and status information
        """
        if not self._guard_config:
            return {
                'enabled': False,
                'scope': None,
                'protected_methods': [],
                'currently_active': False
            }
        
        return {
            'enabled': self._guard_config.enabled,
            'scope': self._guard_config.scope,
            'protected_methods': self.get_protected_methods(),
            'enforcement_level': self._guard_config.enforcement_level,
            'currently_active': self.is_trigger_currently_active(),
            'config': self._guard_config.to_dict()
        }


class AutoGuardMixin(TriggerGuardMixin):
    """
    ✅ FRAMEWORK COMPLIANCE: Automatic guard application via configuration
    
    Enhanced mixin that automatically applies guards based on class-level
    configuration without requiring manual initialization.
    """
    
    def __init__(self, *args, **kwargs):
        """
        ✅ FRAMEWORK COMPLIANCE: Automatic guard initialization
        
        Automatically applies guards if _guard_config is defined at class level.
        """
        super().__init__(*args, **kwargs)
        
        # Check for class-level guard configuration
        guard_config = getattr(self.__class__, '_guard_config', None)
        if guard_config:
            self.__init_guard_system__(guard_config)


# ✅ FRAMEWORK COMPLIANCE: Convenience functions for common patterns
def add_step_guards(component, protected_methods: List[str] = None, **kwargs):
    """
    Add step-level guards to a component instance.
    
    Args:
        component: Component instance to protect
        protected_methods: Methods to protect (defaults to ['process'])
        **kwargs: Additional guard configuration options
    """
    if not isinstance(component, TriggerGuardMixin):
        raise TypeError("Component must inherit from TriggerGuardMixin to add guards")
    
    from .nanobrain_guard_config import create_step_guard_config
    guard_config = create_step_guard_config(protected_methods, **kwargs)
    component.__init_guard_system__(guard_config)  # ✅ Pass ConfigBase object directly


def add_workflow_guards(component, protected_methods: List[str] = None, **kwargs):
    """
    Add workflow-level guards to a component instance.
    
    Args:
        component: Component instance to protect  
        protected_methods: Methods to protect (defaults to ['execute'])
        **kwargs: Additional guard configuration options
    """
    if not isinstance(component, TriggerGuardMixin):
        raise TypeError("Component must inherit from TriggerGuardMixin to add guards")
    
    from .nanobrain_guard_config import create_workflow_guard_config
    guard_config = create_workflow_guard_config(protected_methods, **kwargs)
    component.__init_guard_system__(guard_config)  # ✅ Pass ConfigBase object directly


def add_agent_guards(component, protected_methods: List[str] = None, **kwargs):
    """
    Add agent-level guards to a component instance.
    
    Args:
        component: Component instance to protect
        protected_methods: Methods to protect (defaults to ['_process_specialized_request'])
        **kwargs: Additional guard configuration options
    """
    if not isinstance(component, TriggerGuardMixin):
        raise TypeError("Component must inherit from TriggerGuardMixin to add guards")
    
    from .nanobrain_guard_config import create_agent_guard_config
    guard_config = create_agent_guard_config(protected_methods, **kwargs)
    component.__init_guard_system__(guard_config)  # ✅ Pass ConfigBase object directly 