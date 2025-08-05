"""
✅ EXACT PATTERN: Core ContextVar implementation from reference document
Direct translation of the pattern with Nanobrain scope awareness.

Implements the exact ContextVar + decorator pattern from Trigger-based execution guard.md
adapted for Nanobrain framework with multiple scope support.
"""

import contextvars
import functools
from typing import Callable, Optional

# ✅ EXACT PATTERN: ContextVar flags from Trigger-based execution guard.md
_step_trigger_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "step_trigger_active", default=False
)
_workflow_trigger_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "workflow_trigger_active", default=False  
)
_agent_trigger_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agent_trigger_active", default=False
)

# Scope to ContextVar mapping
SCOPE_CONTEXT_VARS = {
    'step': _step_trigger_active,
    'workflow': _workflow_trigger_active,
    'agent': _agent_trigger_active
}

def via_trigger(scope: str = 'step') -> Callable:
    """
    ✅ EXACT PATTERN: via_trigger decorator from reference implementation
    
    Decorator for trigger calls to set the ContextVar flag.
    Direct implementation from Trigger-based execution guard.md
    
    Args:
        scope: The scope level ('step', 'workflow', 'agent')
        
    Returns:
        Decorator function that wraps calls with ContextVar token management
    """
    context_var = SCOPE_CONTEXT_VARS.get(scope, _step_trigger_active)
    
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # ✅ EXACT PATTERN: Token set/reset from reference
            token = context_var.set(True)
            try:
                return fn(*args, **kwargs)
            finally:
                context_var.reset(token)
        return wrapped
    return decorator

def only_via_trigger(scope: str = 'step', method_name: Optional[str] = None) -> Callable:
    """
    ✅ EXACT PATTERN: only_via_trigger decorator from reference implementation
    
    Decorator for method protection by checking ContextVar flag.
    Direct implementation from Trigger-based execution guard.md
    
    Args:
        scope: The scope level to check ('step', 'workflow', 'agent')
        method_name: Optional custom method name for error messages
        
    Returns:
        Decorator function that enforces trigger-only execution
    """
    context_var = SCOPE_CONTEXT_VARS.get(scope, _step_trigger_active)
    
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            # ✅ EXACT PATTERN: ContextVar check and RuntimeError from reference
            if not context_var.get():
                method_qual_name = method_name or fn.__qualname__
                
                # ✅ INFORMATIVE ERROR MESSAGES: Provide clear guidance to users
                error_msg = _create_informative_error_message(method_qual_name, scope, fn)
                raise RuntimeError(error_msg)
            return fn(*args, **kwargs)
        return wrapped
    return decorator


def _create_informative_error_message(method_name: str, scope: str, original_function: Callable) -> str:
    """
    ✅ USER GUIDANCE: Create informative error messages for guard violations
    
    Provides clear, actionable guidance when users try to call protected methods incorrectly.
    
    Args:
        method_name: Name of the protected method
        scope: Protection scope ('step', 'workflow', 'agent')
        original_function: The original function for additional context
        
    Returns:
        Comprehensive error message with usage guidance
    """
    base_msg = f"🛡️ PROTECTED METHOD: {method_name} may only be invoked via {scope}-level trigger"
    
    # Add scope-specific guidance
    scope_guidance = {
        'step': {
            'trigger_type': 'DataUnitChangeTrigger or enhanced trigger',
            'example': '@via_trigger("step")\ndef trigger_function():\n    return step.process(data)',
            'integration': 'integrate_trigger_with_guards(trigger, step)'
        },
        'workflow': {
            'trigger_type': 'workflow-level trigger or orchestration',
            'example': '@via_trigger("workflow")\ndef workflow_trigger():\n    return workflow.execute()',
            'integration': 'integrate_trigger_with_guards(trigger, workflow)'
        },
        'agent': {
            'trigger_type': 'agent-level trigger or request handler',
            'example': '@via_trigger("agent")\ndef agent_trigger():\n    return agent._process_specialized_request(request)',
            'integration': 'integrate_trigger_with_guards(trigger, agent)'
        }
    }
    
    guidance = scope_guidance.get(scope, scope_guidance['step'])
    
    detailed_msg = f"""
{base_msg}

📋 HOW TO FIX THIS:

Option 1: Use via_trigger decorator
{guidance['example']}

Option 2: Use trigger integration
# Create and integrate trigger
trigger = {guidance['trigger_type']}.from_config(config)
{guidance['integration']}

Option 3: Call via trigger wrapper
# Get wrapper from component
if hasattr(component, 'get_via_trigger_wrapper'):
    wrapped_method = component.get_via_trigger_wrapper('{method_name.split('.')[-1]}')
    if wrapped_method:
        result = wrapped_method(*args, **kwargs)

📚 REASON: This method is protected by ContextVar guards to ensure it's only
called in the correct execution context. This maintains proper workflow
orchestration and prevents unauthorized direct access.

🔧 NANOBRAIN FRAMEWORK: All protected methods must be called via triggers
to maintain event-driven architecture and ensure proper data flow.
"""
    
    return detailed_msg.strip()

def is_trigger_active(scope: str = 'step') -> bool:
    """
    Check if trigger is currently active for the specified scope.
    
    Args:
        scope: The scope level to check ('step', 'workflow', 'agent')
        
    Returns:
        True if trigger is active for the scope, False otherwise
    """
    context_var = SCOPE_CONTEXT_VARS.get(scope, _step_trigger_active)
    return context_var.get()

def get_all_trigger_states() -> dict:
    """
    Get current trigger state for all scopes.
    
    Returns:
        Dictionary mapping scope names to their current trigger states
    """
    return {
        scope: context_var.get() 
        for scope, context_var in SCOPE_CONTEXT_VARS.items()
    }

def reset_all_triggers() -> None:
    """
    Reset all trigger states to False.
    
    Utility function for testing and cleanup.
    """
    for context_var in SCOPE_CONTEXT_VARS.values():
        try:
            # Only reset if currently set to True
            if context_var.get():
                context_var.set(False)
        except LookupError:
            # ContextVar not set in current context
            pass 