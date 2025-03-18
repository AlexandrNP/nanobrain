from enum import Enum


class ComponentState(Enum):
    """Represents different states of framework components."""
    INACTIVE = "INACTIVE"       # Component is not active (like neural resting state)
    ACTIVE = "ACTIVE"           # Currently processing (like neural firing)
    RECOVERING = "RECOVERING"   # Temporary recovery period after activation
    BLOCKED = "BLOCKED"         # Component is prevented from activating
    ENHANCED = "ENHANCED"       # Enhanced sensitivity/performance
    DEGRADED = "DEGRADED"       # Reduced sensitivity/performance
    CONFIGURING = "CONFIGURING" # Component is being reconfigured
    ERROR = "ERROR"             # Component encountered an error during execution


class AgentType(Enum):
    """Represents different types of agents in the system."""
    ASSISTANT = "ASSISTANT"     # Default assistant agent
    WORKFLOW = "WORKFLOW"       # Workflow building agent
    CODE_WRITER = "CODE_WRITER" # Code generation agent
    PLANNER = "PLANNER"         # Planning agent
    EXECUTOR = "EXECUTOR"       # Execution agent
    EVALUATOR = "EVALUATOR"     # Evaluation agent




