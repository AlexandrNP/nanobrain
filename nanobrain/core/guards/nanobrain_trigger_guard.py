"""
NanoBrain ContextVar Trigger Guard System

Direct implementation of the ContextVar + decorator pattern from Trigger-based execution guard.md
adapted for the NanoBrain framework with complete from_config compliance.

✅ EXACT PATTERN: ContextVar flags for thread-safe execution tracking
✅ FRAMEWORK COMPLIANCE: FromConfigBase integration for configuration-driven behavior
✅ SCOPE AWARENESS: Multiple ContextVars for different Nanobrain component types
"""

import contextvars
import functools
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path

from ..component_base import FromConfigBase
from ..config.config_base import ConfigBase


# ✅ EXACT PATTERN: ContextVar flags for different Nanobrain scopes
# Based on Trigger-based execution guard.md but adapted for Nanobrain architecture
_step_trigger_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "step_trigger_active", default=False
)
_workflow_trigger_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "workflow_trigger_active", default=False  
)
_agent_trigger_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_trigger_active", default=False
)


class TriggerGuardConfig(ConfigBase):
    """
    ✅ FRAMEWORK COMPLIANCE: ConfigBase-derived configuration for trigger guards
    ✅ HIERARCHY PRESERVATION: Maintains standard Nanobrain configuration patterns
    
    Configuration schema for NanoBrain trigger guard behavior.
    Follows exact Nanobrain framework hierarchy requirements.
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


class NanobrainTriggerGuard(FromConfigBase):
    """
    ✅ EXACT PATTERN: ContextVar-based trigger guard with Nanobrain framework integration
    
    Direct implementation of the trigger guard pattern from Trigger-based execution guard.md:
    - ContextVar flags for execution state tracking
    - via_trigger decorator for trigger call wrapping  
    - only_via_trigger decorator for method protection
    
    ✅ FRAMEWORK COMPLIANCE: Full from_config pattern with ConfigBase integration
    """
    
    # Mapping of scope names to their ContextVar instances
    SCOPE_CONTEXT_VARS = {
        'step': _step_trigger_active,
        'workflow': _workflow_trigger_active, 
        'agent': _agent_trigger_active
    }
    
    @classmethod
    def _get_config_class(cls):
        """✅ FRAMEWORK COMPLIANCE: Return the config class for this component"""
        return TriggerGuardConfig
    
    def _init_from_config(self, config: TriggerGuardConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """✅ FRAMEWORK COMPLIANCE: Initialize from resolved configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Store configuration
        self.enabled = config.enabled
        self.scope = config.scope
        self.protected_methods = config.protected_methods
        self.error_message_template = config.error_message_template
        
        # Get the ContextVar for this scope
        self.context_var = self.SCOPE_CONTEXT_VARS.get(self.scope)
        if not self.context_var:
            raise ValueError(f"Unknown scope: {self.scope}")
    
    def via_trigger(self, scope: Optional[str] = None) -> Callable:
        """
        ✅ EXACT PATTERN: via_trigger decorator implementation from reference
        
        Decorator that wraps trigger calls to set the ContextVar flag.
        Exact implementation from Trigger-based execution guard.md with scope support.
        
        Args:
            scope: Override scope for this specific decorator (optional)
            
        Returns:
            Decorator function that wraps calls with ContextVar token management
        """
        # Use provided scope or fall back to instance scope
        target_scope = scope or self.scope
        context_var = self.SCOPE_CONTEXT_VARS.get(target_scope)
        
        if not context_var:
            raise ValueError(f"Unknown scope: {target_scope}")
        
        def decorator(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                # ✅ EXACT PATTERN: Token set/reset from reference implementation
                token = context_var.set(True)
                try:
                    return fn(*args, **kwargs)
                finally:
                    context_var.reset(token)
            return wrapped
        return decorator
    
    def only_via_trigger(self, scope: Optional[str] = None) -> Callable:
        """
        ✅ EXACT PATTERN: only_via_trigger decorator implementation from reference
        
        Decorator that protects methods from direct calls by checking ContextVar flag.
        Exact implementation from Trigger-based execution guard.md with scope support.
        
        Args:
            scope: Override scope for this specific decorator (optional)
            
        Returns:
            Decorator function that enforces trigger-only execution
        """
        # Use provided scope or fall back to instance scope
        target_scope = scope or self.scope
        context_var = self.SCOPE_CONTEXT_VARS.get(target_scope)
        
        if not context_var:
            raise ValueError(f"Unknown scope: {target_scope}")
        
        def decorator(fn):
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                # ✅ EXACT PATTERN: ContextVar check and RuntimeError from reference
                if not context_var.get():
                    error_message = self.error_message_template.format(
                        method=fn.__qualname__,
                        scope=target_scope
                    )
                    raise RuntimeError(error_message)
                return fn(*args, **kwargs)
            return wrapped
        return decorator
    
    @classmethod
    def create_scope_guard(cls, scope: str, protected_methods: Optional[List[str]] = None) -> 'NanobrainTriggerGuard':
        """
        ✅ FRAMEWORK COMPLIANCE: Factory method for creating scope-specific guards
        
        Convenience method for creating guards with specific scope configuration.
        Uses from_config pattern internally.
        
        Args:
            scope: Target scope ('step', 'workflow', 'agent')
            protected_methods: List of method names to protect (optional)
            
        Returns:
            Configured NanobrainTriggerGuard instance
        """
        config_dict = {
            'enabled': True,
            'scope': scope,
            'protected_methods': protected_methods or ['process']
        }
        
        return cls.from_config(config_dict)
    
    def is_trigger_active(self, scope: Optional[str] = None) -> bool:
        """
        Check if trigger is currently active for the specified scope.
        
        Args:
            scope: Scope to check (uses instance scope if not provided)
            
        Returns:
            True if trigger is active, False otherwise
        """
        target_scope = scope or self.scope
        context_var = self.SCOPE_CONTEXT_VARS.get(target_scope)
        
        if not context_var:
            raise ValueError(f"Unknown scope: {target_scope}")
            
        return context_var.get()


# ✅ CONVENIENCE FUNCTIONS: Global access to guard functionality
def get_step_guard() -> NanobrainTriggerGuard:
    """Get a pre-configured step-level trigger guard"""
    return NanobrainTriggerGuard.create_scope_guard('step', ['process', 'execute', '_process_internal'])

def get_workflow_guard() -> NanobrainTriggerGuard:
    """Get a pre-configured workflow-level trigger guard"""
    return NanobrainTriggerGuard.create_scope_guard('workflow', ['execute', 'orchestrate'])

def get_agent_guard() -> NanobrainTriggerGuard:
    """Get a pre-configured agent-level trigger guard"""
    return NanobrainTriggerGuard.create_scope_guard('agent', ['_process_specialized_request', 'generate_response'])


# ✅ GLOBAL SCOPE CHECKS: Direct ContextVar access for performance-critical code
def is_step_trigger_active() -> bool:
    """Check if step-level trigger is currently active"""
    return _step_trigger_active.get()

def is_workflow_trigger_active() -> bool:
    """Check if workflow-level trigger is currently active"""
    return _workflow_trigger_active.get()

def is_agent_trigger_active() -> bool:
    """Check if agent-level trigger is currently active"""
    return _agent_trigger_active.get() 