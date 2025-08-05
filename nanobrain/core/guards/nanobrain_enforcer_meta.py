"""
✅ EXACT PATTERN: TriggerEnforcerMeta from Trigger-based execution guard.md
✅ FRAMEWORK COMPLIANCE: Works with existing Nanobrain metaclass chains

Implements automatic method wrapping without disrupting existing inheritance.
Direct implementation of the metaclass pattern with Nanobrain integration.
"""

from abc import ABCMeta
from types import FunctionType
from typing import Dict, Any, Optional, List, Callable
from .nanobrain_guard_config import NanobrainTriggerGuardConfig
from .contextvar_core import only_via_trigger, via_trigger, is_trigger_active


class NanobrainTriggerEnforcerMeta(type):
    """
    ✅ EXACT PATTERN: Metaclass auto-wrapping from reference implementation
    ✅ FRAMEWORK COMPLIANCE: Respects existing Nanobrain metaclass usage
    
    Direct implementation of TriggerEnforcerMeta pattern adapted for Nanobrain.
    Only applies when _guard_config is present and enabled.
    """
    
    def __new__(mcs, name, bases, namespace):
        """
        ✅ EXACT PATTERN: Class creation with automatic method wrapping
        
        Scans namespace for function definitions and wraps configured methods
        with only_via_trigger decorator, exactly as shown in reference.
        
        Args:
            name: Name of the class being created
            bases: Base classes
            namespace: Class namespace containing attributes and methods
            
        Returns:
            New class with trigger guards applied to configured methods
        """
        # ✅ FRAMEWORK COMPLIANCE: Only apply if guard configuration exists
        guard_config_data = namespace.get('_guard_config')
        
        if guard_config_data and guard_config_data.get('enabled', False):
            try:
                # ✅ FRAMEWORK COMPLIANCE: Handle both dict and ConfigBase objects
                if isinstance(guard_config_data, NanobrainTriggerGuardConfig):
                    guard_config = guard_config_data
                elif isinstance(guard_config_data, dict):
                    # Create temporary file for ConfigBase compliance
                    import tempfile
                    import yaml
                    import os
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
                        yaml.dump(guard_config_data, temp_file, default_flow_style=False)
                        temp_file_path = temp_file.name
                    
                    try:
                        guard_config = NanobrainTriggerGuardConfig.from_config(temp_file_path)
                    finally:
                        os.unlink(temp_file_path)
                else:
                    # Invalid guard config type
                    guard_config = None
                
                if guard_config and guard_config.should_apply_guards():
                    # ✅ EXACT PATTERN: Method scanning and wrapping from reference
                    new_ns = {}
                    for attr_name, attr_value in namespace.items():
                        if (isinstance(attr_value, FunctionType) and 
                            attr_name in guard_config.protected_methods):
                            
                            # ✅ EXACT PATTERN: Wrap with only_via_trigger decorator
                            wrapped_method = only_via_trigger(
                                guard_config.scope, 
                                f"{name}.{attr_name}"
                            )(attr_value)
                            
                            # Add metadata for introspection
                            wrapped_method._nanobrain_guard_wrapped = True
                            wrapped_method._nanobrain_guard_scope = guard_config.scope
                            wrapped_method._nanobrain_guard_original = attr_value
                            wrapped_method._nanobrain_guard_method_name = attr_name
                            
                            new_ns[attr_name] = wrapped_method
                        else:
                            # Keep original attribute unchanged
                            new_ns[attr_name] = attr_value
                    
                    # Store guard information in class for runtime access
                    new_ns['_trigger_guard_config'] = guard_config
                    new_ns['_protected_methods'] = guard_config.protected_methods
                    new_ns['_guard_scope'] = guard_config.scope
                    
                    return super().__new__(mcs, name, bases, new_ns)
                    
            except Exception as e:
                # ✅ FRAMEWORK COMPLIANCE: Graceful degradation
                import warnings
                warnings.warn(
                    f"Failed to apply trigger guards to {name}: {e}. "
                    f"Component will function without guard protection.",
                    RuntimeWarning
                )
        
        # ✅ FRAMEWORK COMPLIANCE: No guard config = normal class creation
        return super().__new__(mcs, name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        """
        ✅ FRAMEWORK ENHANCEMENT: Class instantiation with guard setup
        
        Called when creating instances of classes using this metaclass.
        Sets up runtime guard access and validation.
        """
        instance = super().__call__(*args, **kwargs)
        
        # Attach guard to instance for runtime access
        if hasattr(cls, '_trigger_guard_config'):
            instance._trigger_guard_config = cls._trigger_guard_config
            instance._protected_methods = cls._protected_methods
            instance._guard_scope = cls._guard_scope
            
            # Add runtime guard methods if they don't exist
            if not hasattr(instance, 'is_method_protected'):
                instance.is_method_protected = lambda method_name: _check_method_protection(instance, method_name)
            if not hasattr(instance, 'get_guard_scope'):
                instance.get_guard_scope = lambda: getattr(instance, '_guard_scope', None)
            if not hasattr(instance, 'get_via_trigger_wrapper'):
                instance.get_via_trigger_wrapper = lambda method_name: _get_via_trigger_wrapper(instance, method_name)
        
        return instance


class ConfigurableGuardMeta(NanobrainTriggerEnforcerMeta):
    """
    ✅ FRAMEWORK EXTENSION: Enhanced metaclass with dynamic configuration support
    
    Extends the base enforcer metaclass to support:
    - Runtime guard configuration updates
    - Multiple guard configurations per class
    - Conditional guard activation
    """
    
    def __new__(mcs, name, bases, namespace):
        """Enhanced class creation with dynamic guard configuration support"""
        
        # Check for multiple guard configurations
        guard_configs = []
        
        # Look for _guard_config (single config)
        if '_guard_config' in namespace:
            guard_configs.append(namespace['_guard_config'])
        
        # Look for _guard_configs (multiple configs)
        if '_guard_configs' in namespace:
            guard_configs.extend(namespace['_guard_configs'])
        
        # If no configurations found, check parent classes
        if not guard_configs:
            for base in bases:
                if hasattr(base, '_guard_config'):
                    guard_configs.append(base._guard_config)
                if hasattr(base, '_guard_configs'):
                    guard_configs.extend(base._guard_configs)
        
        # Apply first enabled configuration
        for config in guard_configs:
            if isinstance(config, dict) and config.get('enabled', False):
                namespace['_guard_config'] = config
                break
            elif isinstance(config, NanobrainTriggerGuardConfig) and config.should_apply_guards():
                namespace['_guard_config'] = config
                break
        else:
            # No enabled configurations found - disable guards
            namespace.pop('_guard_config', None)
        
        return super().__new__(mcs, name, bases, namespace)


# ✅ CONVENIENCE FUNCTIONS: Helper functions for guard introspection
def is_method_protected(method) -> bool:
    """
    Check if a method has been wrapped with trigger protection.
    
    Args:
        method: Method to check
        
    Returns:
        True if method is protected by trigger guard, False otherwise
    """
    return getattr(method, '_nanobrain_guard_wrapped', False)


def get_method_guard_scope(method) -> Optional[str]:
    """
    Get the guard scope for a protected method.
    
    Args:
        method: Method to check
        
    Returns:
        Guard scope ('step', 'workflow', 'agent') or None if not protected
    """
    return getattr(method, '_nanobrain_guard_scope', None)


def get_original_method(method):
    """
    Get the original unwrapped method from a protected method.
    
    Args:
        method: Protected method
        
    Returns:
        Original method before guard wrapping, or the method itself if not protected
    """
    return getattr(method, '_nanobrain_guard_original', method)


def get_class_protected_methods(cls) -> List[str]:
    """
    Get list of protected method names for a class.
    
    Args:
        cls: Class to inspect
        
    Returns:
        List of method names that are protected by trigger guards
    """
    return getattr(cls, '_protected_methods', [])


def get_class_guard_scope(cls) -> Optional[str]:
    """
    Get the guard scope for a class.
    
    Args:
        cls: Class to inspect
        
    Returns:
        Guard scope or None if class doesn't use trigger guards
    """
    return getattr(cls, '_guard_scope', None)


# ✅ RUNTIME HELPER FUNCTIONS: For instances created with metaclass
def _check_method_protection(instance, method_name: str) -> bool:
    """Helper function for runtime method protection checking"""
    if not hasattr(instance, method_name):
        return False
    method = getattr(instance, method_name)
    return getattr(method, '_nanobrain_guard_protected', False)


def _get_via_trigger_wrapper(instance, method_name: str) -> Optional[callable]:
    """Helper function for getting via_trigger wrapped methods"""
    if not hasattr(instance, method_name):
        return None
    
    method = getattr(instance, method_name)
    original_method = getattr(method, '_nanobrain_guard_original', method)
    
    # Get guard scope
    guard_scope = getattr(instance, '_guard_scope', 'step')
    
    # Return via_trigger wrapped version
    from .contextvar_core import via_trigger
    return via_trigger(guard_scope)(original_method)


# ✅ FRAMEWORK COMPLIANCE: Enhanced decorator for applying ContextVar protection
def contextvar_protected(scope: str = 'step', protected_methods: List[str] = None, enabled: bool = True):
    """
    ✅ PRIMARY APPROACH: Enhanced class decorator for ContextVar protection
    
    Resolves metaclass conflicts with BaseStep while providing complete
    ContextVar guard functionality. Recommended approach for Nanobrain integration.
    
    Args:
        scope: Protection scope ('step', 'workflow', 'agent')
        protected_methods: List of methods to protect
        enabled: Whether protection is enabled
        
    Returns:
        Decorated class with ContextVar protection and enhanced functionality
    """
    def decorator(cls):
        # ✅ FRAMEWORK COMPLIANCE: Enhanced guard configuration
        cls._guard_config = {
            'enabled': enabled,
            'scope': scope,
            'protected_methods': protected_methods or ['process'],
            'auto_apply': True,
            'enforcement_level': 'strict',
            'debug_mode': False,
            'log_guard_events': True,
            'error_message_template': '{method} may only be invoked via {scope}-level trigger'
        }
        
        # ✅ ENHANCED FUNCTIONALITY: Add guard management methods to class
        if enabled:
            _add_guard_methods_to_class(cls, scope, protected_methods or ['process'])
        
        return cls
    
    return decorator


def _add_guard_methods_to_class(cls, scope: str, protected_methods: List[str]):
    """
    ✅ ENHANCED APPROACH: Add guard functionality to class without metaclass conflicts
    
    Adds all necessary guard methods and protections to a class while avoiding
    metaclass inheritance issues with BaseStep.
    
    Args:
        cls: Class to enhance
        scope: Protection scope
        protected_methods: List of methods to protect
    """
    # Store original methods and apply protection
    original_methods = {}
    
    for method_name in protected_methods:
        if hasattr(cls, method_name):
            original_method = getattr(cls, method_name)
            if callable(original_method):
                # Store original method
                original_methods[method_name] = original_method
                
                # ✅ EXACT PATTERN: Apply only_via_trigger protection
                wrapped_method = only_via_trigger(scope, f"{cls.__name__}.{method_name}")(original_method)
                
                # Add metadata for introspection
                wrapped_method._nanobrain_guard_wrapped = True
                wrapped_method._nanobrain_guard_scope = scope
                wrapped_method._nanobrain_guard_original = original_method
                wrapped_method._nanobrain_guard_method_name = method_name
                
                # Apply the protection
                setattr(cls, method_name, wrapped_method)
    
    # ✅ ENHANCED FUNCTIONALITY: Add guard management methods
    def is_method_protected(self, method_name: str) -> bool:
        """Check if a method is protected by guards"""
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return getattr(method, '_nanobrain_guard_wrapped', False)
        return False
    
    def get_guard_scope(self) -> str:
        """Get the guard scope for this component"""
        return scope
    
    def get_protected_methods(self) -> List[str]:
        """Get list of protected method names"""
        return protected_methods.copy()
    
    def get_via_trigger_wrapper(self, method_name: str) -> Optional[callable]:
        """Get via_trigger wrapped version of a method"""
        if method_name in original_methods:
            original_method = original_methods[method_name]
            return via_trigger(scope)(original_method)
        return None
    
    def is_trigger_currently_active(self) -> bool:
        """Check if trigger is currently active for this scope"""
        return is_trigger_active(scope)
    
    def get_guard_status(self) -> Dict[str, Any]:
        """Get comprehensive guard status"""
        return {
            'enabled': True,
            'scope': scope,
            'protected_methods': protected_methods.copy(),
            'currently_active': is_trigger_active(scope),
            'total_protected': len(protected_methods),
            'guard_type': 'contextvar_decorator'
        }
    
    # ✅ FRAMEWORK COMPLIANCE: Add methods to class
    cls.is_method_protected = is_method_protected
    cls.get_guard_scope = get_guard_scope  
    cls.get_protected_methods = get_protected_methods
    cls.get_via_trigger_wrapper = get_via_trigger_wrapper
    cls.is_trigger_currently_active = is_trigger_currently_active
    cls.get_guard_status = get_guard_status
    
    # Store metadata for class introspection
    cls._protected_methods = protected_methods
    cls._guard_scope = scope
    cls._original_methods = original_methods
    cls._guard_type = 'contextvar_decorator'


# ✅ RESOLUTION: ABCMeta-based composite metaclass for BaseStep compatibility
class BaseStepCompatibleMeta(ABCMeta):
    """
    ✅ METACLASS RESOLUTION: ABCMeta-based composite metaclass compatible with BaseStep
    
    Resolves metaclass conflicts by properly inheriting from ABCMeta (which BaseStep uses)
    while adding trigger guard functionality. This is the correct approach for BaseStep compatibility.
    """
    
    def __new__(mcs, name, bases, namespace):
        """
        ✅ CONFLICT RESOLUTION: Handle BaseStep metaclass inheritance
        
        Creates classes that are compatible with BaseStep's ABCMeta inheritance
        while adding trigger guard auto-wrapping functionality.
        """
        # Handle guard configuration if present
        guard_config = namespace.get('_guard_config')
        
        if guard_config and guard_config.get('enabled', False):
            # ✅ EXACT PATTERN: Apply guard enhancements to namespace
            try:
                from .nanobrain_guard_config import NanobrainTriggerGuardConfig
                import tempfile
                import yaml
                import os
                
                # Create guard config object if needed
                if isinstance(guard_config, dict):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
                        yaml.dump(guard_config, temp_file, default_flow_style=False)
                        temp_file_path = temp_file.name
                    
                    try:
                        guard_obj = NanobrainTriggerGuardConfig.from_config(temp_file_path)
                    finally:
                        os.unlink(temp_file_path)
                else:
                    guard_obj = guard_config
                
                if guard_obj and guard_obj.should_apply_guards():
                    # Apply method wrapping
                    new_ns = {}
                    for attr_name, attr_value in namespace.items():
                        if (isinstance(attr_value, FunctionType) and 
                            attr_name in guard_obj.protected_methods):
                            
                            # ✅ EXACT PATTERN: Wrap with only_via_trigger decorator
                            wrapped_method = only_via_trigger(
                                guard_obj.scope, 
                                f"{name}.{attr_name}"
                            )(attr_value)
                            
                            # Add metadata for introspection
                            wrapped_method._nanobrain_guard_wrapped = True
                            wrapped_method._nanobrain_guard_scope = guard_obj.scope
                            wrapped_method._nanobrain_guard_original = attr_value
                            wrapped_method._nanobrain_guard_method_name = attr_name
                            
                            new_ns[attr_name] = wrapped_method
                        else:
                            new_ns[attr_name] = attr_value
                    
                    # Store guard information in class
                    new_ns['_trigger_guard_config'] = guard_obj
                    new_ns['_protected_methods'] = guard_obj.protected_methods
                    new_ns['_guard_scope'] = guard_obj.scope
                    
                    # Create class with enhanced namespace
                    return super().__new__(mcs, name, bases, new_ns)
                    
            except Exception as e:
                # ✅ FRAMEWORK COMPLIANCE: Graceful degradation
                import warnings
                warnings.warn(
                    f"Failed to apply trigger guards to {name}: {e}. "
                    f"Component will function without guard protection.",
                    RuntimeWarning
                )
        
        # ✅ FRAMEWORK COMPLIANCE: Standard ABCMeta creation
        return super().__new__(mcs, name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        """Enhanced instance creation with guard setup"""
        instance = super().__call__(*args, **kwargs)
        
        # Apply guard configuration if present
        if hasattr(cls, '_trigger_guard_config'):
            # Add runtime guard access to instance
            instance._trigger_guard_config = cls._trigger_guard_config
            instance._protected_methods = getattr(cls, '_protected_methods', [])
            instance._guard_scope = getattr(cls, '_guard_scope', 'step')
            
            # Add runtime guard methods
            def is_method_protected(method_name: str) -> bool:
                if hasattr(instance, method_name):
                    method = getattr(instance, method_name)
                    return getattr(method, '_nanobrain_guard_wrapped', False)
                return False
            
            def get_guard_scope() -> str:
                return getattr(instance, '_guard_scope', 'step')
            
            def get_protected_methods() -> List[str]:
                return getattr(instance, '_protected_methods', []).copy()
            
            def get_via_trigger_wrapper(method_name: str) -> Optional[callable]:
                if hasattr(instance, method_name):
                    method = getattr(instance, method_name)
                    original = getattr(method, '_nanobrain_guard_original', method)
                    scope = getattr(instance, '_guard_scope', 'step')
                    return via_trigger(scope)(original)
                return None
            
            # Attach methods to instance
            instance.is_method_protected = is_method_protected
            instance.get_guard_scope = get_guard_scope
            instance.get_protected_methods = get_protected_methods  
            instance.get_via_trigger_wrapper = get_via_trigger_wrapper
        
        return instance


# ✅ COMPLETE SOLUTION: Enhanced metaclass with full BaseStep compatibility
class NanobrainGuardMeta(BaseStepCompatibleMeta):
    """
    ✅ COMPLETE SOLUTION: Enhanced metaclass with full BaseStep compatibility
    
    Provides complete trigger guard auto-wrapping functionality while maintaining
    full compatibility with BaseStep and all Nanobrain framework patterns.
    """
    
    def __new__(mcs, name, bases, namespace):
        """
        ✅ ENHANCED CREATION: Complete guard integration with BaseStep compatibility
        
        Provides all the functionality of the original NanobrainTriggerEnforcerMeta
        while being fully compatible with BaseStep's ABCMeta inheritance.
        """
        return super().__new__(mcs, name, bases, namespace) 