"""
✅ FRAMEWORK COMPLIANCE: ConfigBase-derived trigger guard configuration
✅ HIERARCHY PRESERVATION: Maintains standard Nanobrain configuration patterns

Configuration for ContextVar-based trigger guards while maintaining
complete compatibility with existing framework patterns.
"""

from typing import List, Dict, Any
from ..config.config_base import ConfigBase


class NanobrainTriggerGuardConfig(ConfigBase):
    """
    ✅ FRAMEWORK COMPLIANCE: Standard ConfigBase inheritance
    ✅ PATTERN ADHERENCE: Follows exact Nanobrain configuration structure
    
    Configuration for ContextVar-based trigger guards while maintaining
    complete compatibility with existing framework patterns.
    
    This class follows the mandatory ConfigBase hierarchy and supports
    file-based configuration loading as required by the framework.
    """
    
    def __init__(self, **data):
        # ✅ FRAMEWORK COMPLIANCE: Standard ConfigBase constructor
        super().__init__(**data)
        
        # Configuration fields with framework-compliant defaults
        self.enabled: bool = data.get('enabled', True)
        self.scope: str = data.get('scope', 'step')
        self.protected_methods: List[str] = data.get('protected_methods', ['process'])
        self.error_message_template: str = data.get(
            'error_message_template', 
            "{method} may only be invoked via {scope}-level trigger"
        )
        self.auto_apply: bool = data.get('auto_apply', True)
        self.enforcement_level: str = data.get('enforcement_level', 'strict')
        self.debug_mode: bool = data.get('debug_mode', False)
        self.log_guard_events: bool = data.get('log_guard_events', True)
        
        # ✅ VALIDATION: Framework-compliant validation
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate guard configuration against framework requirements"""
        valid_scopes = {'step', 'workflow', 'agent'}
        if self.scope not in valid_scopes:
            raise ValueError(f"Invalid scope '{self.scope}'. Must be one of: {valid_scopes}")
        
        valid_enforcement_levels = {'strict', 'warning', 'disabled'}
        if self.enforcement_level not in valid_enforcement_levels:
            raise ValueError(f"Invalid enforcement_level '{self.enforcement_level}'. Must be one of: {valid_enforcement_levels}")
            
        if not isinstance(self.protected_methods, list):
            raise ValueError("protected_methods must be a list of method names")
            
        if not all(isinstance(method, str) for method in self.protected_methods):
            raise ValueError("All protected_methods must be strings")
    
    def get_context_var_name(self) -> str:
        """Get the ContextVar name for this scope"""
        return f"{self.scope}_trigger_active"
    
    def is_method_protected(self, method_name: str) -> bool:
        """Check if a method is configured for protection"""
        return method_name in self.protected_methods
    
    def should_apply_guards(self) -> bool:
        """Check if guards should be applied based on configuration"""
        return self.enabled and self.enforcement_level != 'disabled'
    
    def get_error_message(self, method_name: str) -> str:
        """Generate error message for unauthorized method call"""
        return self.error_message_template.format(
            method=method_name,
            scope=self.scope
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Export configuration as dictionary
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            'enabled': self.enabled,
            'scope': self.scope,
            'protected_methods': self.protected_methods,
            'error_message_template': self.error_message_template,
            'auto_apply': self.auto_apply,
            'enforcement_level': self.enforcement_level,
            'debug_mode': self.debug_mode,
            'log_guard_events': self.log_guard_events
        }
    
    def __repr__(self) -> str:
        return (f"NanobrainTriggerGuardConfig("
                f"scope='{self.scope}', "
                f"enabled={self.enabled}, "
                f"protected_methods={self.protected_methods}, "
                f"enforcement_level='{self.enforcement_level}')")


# ✅ FRAMEWORK COMPLIANCE: Factory functions using file-based configuration
import os
import tempfile
import yaml
from pathlib import Path


def create_step_guard_config(protected_methods: List[str] = None, **kwargs) -> NanobrainTriggerGuardConfig:
    """
    Create a standard step-level guard configuration.
    
    ✅ FRAMEWORK COMPLIANCE: Uses from_config with default template
    
    Args:
        protected_methods: List of methods to protect (defaults to ['process'])
        **kwargs: Additional configuration options
        
    Returns:
        Configured NanobrainTriggerGuardConfig for step-level protection
    """
    # Get default config file path
    default_config_path = Path(__file__).parent / "config" / "step_guard_default.yml"
    
    if protected_methods is None and not kwargs:
        # Use default configuration as-is
        return NanobrainTriggerGuardConfig.from_config(str(default_config_path))
    
    # Create custom configuration based on defaults
    config_data = {
        'enabled': True,
        'scope': 'step',
        'protected_methods': protected_methods or ['process'],
        'auto_apply': True,
        'enforcement_level': 'strict',
        'debug_mode': False,
        'log_guard_events': True,
        'error_message_template': '{method} may only be invoked via {scope}-level trigger',
        **kwargs
    }
    
    # ✅ FRAMEWORK COMPLIANCE: Create temporary YAML file for from_config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file_path = temp_file.name
    
    try:
        return NanobrainTriggerGuardConfig.from_config(temp_file_path)
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def create_workflow_guard_config(protected_methods: List[str] = None, **kwargs) -> NanobrainTriggerGuardConfig:
    """
    Create a standard workflow-level guard configuration.
    
    ✅ FRAMEWORK COMPLIANCE: Uses from_config with default template
    
    Args:
        protected_methods: List of methods to protect (defaults to ['execute'])
        **kwargs: Additional configuration options
        
    Returns:
        Configured NanobrainTriggerGuardConfig for workflow-level protection
    """
    # Get default config file path
    default_config_path = Path(__file__).parent / "config" / "workflow_guard_default.yml"
    
    if protected_methods is None and not kwargs:
        # Use default configuration as-is
        return NanobrainTriggerGuardConfig.from_config(str(default_config_path))
    
    # Create custom configuration based on defaults
    config_data = {
        'enabled': True,
        'scope': 'workflow',
        'protected_methods': protected_methods or ['execute'],
        'auto_apply': True,
        'enforcement_level': 'strict',
        'debug_mode': False,
        'log_guard_events': True,
        'error_message_template': '{method} may only be invoked via {scope}-level trigger',
        **kwargs
    }
    
    # ✅ FRAMEWORK COMPLIANCE: Create temporary YAML file for from_config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file_path = temp_file.name
    
    try:
        return NanobrainTriggerGuardConfig.from_config(temp_file_path)
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def create_agent_guard_config(protected_methods: List[str] = None, **kwargs) -> NanobrainTriggerGuardConfig:
    """
    Create a standard agent-level guard configuration.
    
    ✅ FRAMEWORK COMPLIANCE: Uses from_config with default template
    
    Args:
        protected_methods: List of methods to protect (defaults to ['_process_specialized_request'])
        **kwargs: Additional configuration options
        
    Returns:
        Configured NanobrainTriggerGuardConfig for agent-level protection
    """
    # Get default config file path
    default_config_path = Path(__file__).parent / "config" / "agent_guard_default.yml"
    
    if protected_methods is None and not kwargs:
        # Use default configuration as-is
        return NanobrainTriggerGuardConfig.from_config(str(default_config_path))
    
    # Create custom configuration based on defaults
    config_data = {
        'enabled': True,
        'scope': 'agent',
        'protected_methods': protected_methods or ['_process_specialized_request'],
        'auto_apply': True,
        'enforcement_level': 'strict',
        'debug_mode': False,
        'log_guard_events': True,
        'error_message_template': '{method} may only be invoked via {scope}-level trigger',
        **kwargs
    }
    
    # ✅ FRAMEWORK COMPLIANCE: Create temporary YAML file for from_config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
        yaml.dump(config_data, temp_file, default_flow_style=False)
        temp_file_path = temp_file.name
    
    try:
        return NanobrainTriggerGuardConfig.from_config(temp_file_path)
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path) 