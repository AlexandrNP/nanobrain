"""
NanoBrain ContextVar Trigger Guard System

✅ EXACT PATTERN: Direct implementation from Trigger-based execution guard.md
✅ FRAMEWORK COMPLIANCE: Zero modifications to existing hierarchies

Implements the ContextVar-based trigger guard pattern for framework-wide execution enforcement.
Based on the pattern from Trigger-based execution guard.md with Nanobrain framework integration.
"""

# Export core ContextVar functionality
from .contextvar_core import (
    _step_trigger_active,
    _workflow_trigger_active, 
    _agent_trigger_active,
    via_trigger,
    only_via_trigger,
    is_trigger_active,
    get_all_trigger_states,
    reset_all_triggers,
    SCOPE_CONTEXT_VARS
)

from .nanobrain_guard_config import (
    NanobrainTriggerGuardConfig,
    create_step_guard_config,
    create_workflow_guard_config,
    create_agent_guard_config
)

from .nanobrain_guard_mixin import (
    TriggerGuardMixin,
    AutoGuardMixin,
    add_step_guards,
    add_workflow_guards,
    add_agent_guards
)

# Legacy imports for backward compatibility
from .nanobrain_trigger_guard import (
    NanobrainTriggerGuard,
    get_step_guard,
    get_workflow_guard,
    get_agent_guard,
    is_step_trigger_active,
    is_workflow_trigger_active,
    is_agent_trigger_active
)

from .nanobrain_enforcer_meta import (
    NanobrainTriggerEnforcerMeta,
    ConfigurableGuardMeta,
    BaseStepCompatibleMeta,
    NanobrainGuardMeta,
    is_method_protected,
    get_method_guard_scope,
    get_original_method,
    get_class_protected_methods,
    get_class_guard_scope,
    contextvar_protected
)

from .trigger_integration import (
    TriggerGuardIntegration,
    EnhancedDataUnitChangeTrigger,
    integrate_trigger_with_guards,
    create_guard_aware_trigger,
    enhance_existing_trigger,
    is_trigger_guard_aware,
    get_trigger_wrapped_methods
)

__all__ = [
    # Core ContextVar system
    'via_trigger',
    'only_via_trigger', 
    'is_trigger_active',
    'get_all_trigger_states',
    'reset_all_triggers',
    'SCOPE_CONTEXT_VARS',
    '_step_trigger_active',
    '_workflow_trigger_active', 
    '_agent_trigger_active',
    
    # Configuration system
    'NanobrainTriggerGuardConfig',
    'create_step_guard_config',
    'create_workflow_guard_config',
    'create_agent_guard_config',
    
    # Mixin system
    'TriggerGuardMixin',
    'AutoGuardMixin',
    'add_step_guards',
    'add_workflow_guards',
    'add_agent_guards',
    
    # Metaclass system
    'NanobrainTriggerEnforcerMeta',
    'ConfigurableGuardMeta',
    'BaseStepCompatibleMeta',
    'NanobrainGuardMeta',
    'is_method_protected',
    'get_method_guard_scope',
    'get_original_method',
    'get_class_protected_methods',
    'get_class_guard_scope',
    'contextvar_protected',
    
    # Trigger integration system
    'TriggerGuardIntegration',
    'EnhancedDataUnitChangeTrigger',
    'integrate_trigger_with_guards',
    'create_guard_aware_trigger',
    'enhance_existing_trigger',
    'is_trigger_guard_aware',
    'get_trigger_wrapped_methods',
    
    # Legacy components
    'NanobrainTriggerGuard',
    'get_step_guard',
    'get_workflow_guard', 
    'get_agent_guard',
    'is_step_trigger_active',
    'is_workflow_trigger_active',
    'is_agent_trigger_active'
] 