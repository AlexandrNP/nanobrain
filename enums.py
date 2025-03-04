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




