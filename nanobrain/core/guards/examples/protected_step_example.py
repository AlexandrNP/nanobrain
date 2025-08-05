"""
Protected Step Example - Option A: Pure Mixin Integration

Demonstrates how to add ContextVar guard protection to existing step classes
without modifying the base StepBase class or existing step implementations.

✅ FRAMEWORK COMPLIANCE: No changes to existing base classes
✅ PURE MIXIN: Composition-based guard integration
✅ ZERO INTRUSION: Existing steps continue to work unchanged
"""

from typing import Dict, Any, Optional
from ...step import BaseStep  
from ..nanobrain_guard_mixin import TriggerGuardMixin


class ProtectedStepExample(BaseStep, TriggerGuardMixin):
    """
    ✅ OPTION A: Pure Mixin Integration Example
    
    Shows how to create a protected step by inheriting from both:
    - BaseStep (existing Nanobrain step base class) 
    - TriggerGuardMixin (new guard functionality)
    
    No modifications needed to existing BaseStep class.
    """
    
    # ✅ CLASS-LEVEL GUARD CONFIGURATION
    _guard_config = {
        'enabled': True,
        'scope': 'step',
        'protected_methods': ['process', 'execute', '_process_internal'],
        'enforcement_level': 'strict',
        'debug_mode': False,
        'log_guard_events': True
    }
    
    @classmethod
    def _get_config_class(cls):
        """✅ FRAMEWORK COMPLIANCE: Required by BaseStep"""
        from ...config.config_base import ConfigBase
        
        class ProtectedStepConfig(ConfigBase):
            def __init__(self, **data):
                super().__init__(**data)
                self.name = data.get('name', 'protected_step_example')
                self.description = data.get('description', 'Example protected step')
                
        return ProtectedStepConfig
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any], 
                         dependencies: Dict[str, Any]) -> None:
        """✅ FRAMEWORK COMPLIANCE: Standard step initialization with guard setup"""
        # Initialize base step functionality
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ GUARD INTEGRATION: Initialize guard system
        guard_config = dependencies.get('guard_config', self._guard_config)
        self.__init_guard_system__(guard_config)
    
    def process(self, data: Any) -> Any:
        """
        ✅ PROTECTED METHOD: This method is automatically protected by guards
        
        Can only be called via triggers due to ContextVar protection.
        """
        self.nb_logger.info(f"Processing data in protected step: {type(data)}")
        
        # Simulate processing
        processed_data = {
            'step_name': self.name,
            'original_data': data,
            'processed_timestamp': '2025-01-28T10:00:00Z',
            'protection_status': 'guard_protected'
        }
        
        return processed_data
    
    def execute(self) -> Dict[str, Any]:
        """
        ✅ PROTECTED METHOD: Another protected method example
        
        Can only be called via triggers due to ContextVar protection.
        """
        self.nb_logger.info(f"Executing protected step: {self.name}")
        
        return {
            'step_name': self.name,
            'execution_status': 'completed',
            'protection_status': 'guard_protected'
        }
    
    def _process_internal(self, internal_data: Any) -> Any:
        """
        ✅ PROTECTED METHOD: Internal processing method
        
        Protected from direct access, can only be called via triggers.
        """
        return f"Internal processing of {internal_data} in {self.name}"
    
    def get_status(self) -> Dict[str, Any]:
        """
        ✅ UNPROTECTED METHOD: Regular method that can be called directly
        
        Not in protected_methods list, so no ContextVar protection applied.
        """
        return {
            'step_name': self.name,
            'guard_enabled': bool(self._guard_config),
            'protected_methods': self.get_protected_methods() if hasattr(self, 'get_protected_methods') else [],
            'guard_scope': self.get_guard_scope() if hasattr(self, 'get_guard_scope') else None,
            'currently_protected': self.is_method_protected('process') if hasattr(self, 'is_method_protected') else False
        }


# ✅ EXAMPLE: Creating protected step from existing step class
def create_protected_step_from_existing(existing_step_class):
    """
    ✅ OPTION A: Convert existing step to protected step via mixin
    
    Shows how to add protection to an existing step class without modification.
    
    Args:
        existing_step_class: Existing step class that inherits from BaseStep
        
    Returns:
        New class with ContextVar protection added
    """
    
    class ProtectedStepWrapper(existing_step_class, TriggerGuardMixin):
        """Dynamic protected wrapper for existing step"""
        
        _guard_config = {
            'enabled': True,
            'scope': 'step',
            'protected_methods': ['process'],  # Protect main process method
            'enforcement_level': 'strict'
        }
        
        def _init_from_config(self, config: Any, component_config: Dict[str, Any], 
                             dependencies: Dict[str, Any]) -> None:
            """Initialize with guard protection"""
            super()._init_from_config(config, component_config, dependencies)
            
            # Apply guards
            guard_config = dependencies.get('guard_config', self._guard_config)
            self.__init_guard_system__(guard_config)
    
    # Set proper class name and documentation
    ProtectedStepWrapper.__name__ = f"Protected{existing_step_class.__name__}"
    ProtectedStepWrapper.__qualname__ = f"Protected{existing_step_class.__qualname__}"
    ProtectedStepWrapper.__doc__ = f"Protected version of {existing_step_class.__name__} with ContextVar guards"
    
    return ProtectedStepWrapper


# ✅ EXAMPLE: Factory function for creating protected steps
def create_protected_step_with_config(step_name: str, protected_methods: Optional[list] = None, 
                                     scope: str = 'step', **kwargs) -> 'ProtectedStepExample':
    """
    ✅ OPTION A: Factory function for creating configured protected steps
    
    Args:
        step_name: Name for the step
        protected_methods: List of methods to protect (defaults to ['process'])
        scope: Protection scope (defaults to 'step')
        **kwargs: Additional guard configuration options
        
    Returns:
        Configured protected step instance
    """
    from ..nanobrain_guard_config import create_step_guard_config
    
    # Create custom guard configuration
    guard_config = create_step_guard_config(
        protected_methods or ['process'],
        **kwargs
    )
    
    # Create temporary config for step
    import tempfile
    import yaml
    import os
    
    step_config_data = {
        'name': step_name,
        'description': f'Protected step: {step_name}',
        'enable_logging': True
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
        yaml.dump(step_config_data, temp_file, default_flow_style=False)
        temp_file_path = temp_file.name
    
    try:
        # Create step with guard configuration
        dependencies = {'guard_config': guard_config}
        return ProtectedStepExample.from_config(temp_file_path, **dependencies)
    finally:
        os.unlink(temp_file_path) 