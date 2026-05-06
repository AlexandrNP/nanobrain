"""
Event Type Enums for Nanobrain Framework

BRUTAL TRUTH: This module eliminates the string-based event type disaster
that was causing silent trigger failures throughout the framework.

All event types are now strongly typed enums to prevent configuration bugs.
"""

from enum import Enum
from typing import Union, Set


class DataUnitEventType(Enum):
    """
    Enumeration of all data unit operation event types.
    
    BRUTAL TRUTH: These replace the inconsistent string-based event types
    that were causing trigger activation failures.
    """
    # Core data operations
    SET = "set"                    # Data unit set() operation
    GET = "get"                    # Data unit get() operation  
    WRITE = "write"                # Data unit write() operation
    READ = "read"                  # Data unit read() operation
    APPEND = "append"              # Data unit append() operation
    DELETE = "delete"              # Data unit delete() operation
    CLEAR = "clear"                # Data unit clear() operation
    
    # Lifecycle events
    INITIALIZE = "initialize"      # Data unit initialization
    CLEANUP = "cleanup"            # Data unit cleanup
    
    # Special events
    ALL = "all"                    # Match all event types
    
    @classmethod
    def get_write_operations(cls) -> Set['DataUnitEventType']:
        """Get all event types that represent write operations."""
        return {cls.SET, cls.WRITE, cls.APPEND, cls.DELETE, cls.CLEAR}
    
    @classmethod
    def get_read_operations(cls) -> Set['DataUnitEventType']:
        """Get all event types that represent read operations."""
        return {cls.GET, cls.READ}
    
    @classmethod
    def from_string(cls, value: str) -> 'DataUnitEventType':
        """
        Convert string to enum with validation.
        
        BRUTAL TRUTH: This provides backward compatibility while enforcing
        proper enum usage going forward.
        """
        # Handle legacy string values
        legacy_mapping = {
            'data_unit_updated': cls.SET,  # Fix the inconsistent default
            'data_updated': cls.SET,       # Another legacy variant
        }
        
        if value in legacy_mapping:
            return legacy_mapping[value]
        
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid DataUnitEventType: '{value}'. Valid values: {[e.value for e in cls]}")


class TriggerEventType(Enum):
    """
    Enumeration of trigger event types.
    
    BRUTAL TRUTH: These replace the inconsistent trigger type strings
    that were scattered throughout the framework.
    """
    # Data-driven triggers
    DATA_UPDATED = "data_updated"           # Data unit change trigger
    DATA_RECEIVED = "data_received"         # Data unit receives new data
    ALL_DATA_RECEIVED = "all_data_received" # All required data units have data
    
    # Time-based triggers
    SCHEDULED = "scheduled"                 # Time-based scheduled trigger
    INTERVAL = "interval"                   # Recurring interval trigger
    
    # Event-driven triggers
    STEP_COMPLETED = "step_completed"       # Step execution completed
    WORKFLOW_STARTED = "workflow_started"   # Workflow execution started
    WORKFLOW_COMPLETED = "workflow_completed" # Workflow execution completed
    
    # Manual triggers
    MANUAL = "manual"                       # Manually triggered
    
    @classmethod
    def from_string(cls, value: str) -> 'TriggerEventType':
        """Convert string to enum with validation."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid TriggerEventType: '{value}'. Valid values: {[e.value for e in cls]}")


class LinkEventType(Enum):
    """
    Enumeration of link event types.
    
    BRUTAL TRUTH: These define when links should be activated
    based on source data unit changes.
    """
    # Source data events
    SOURCE_UPDATED = "source_updated"       # Source data unit updated
    SOURCE_SET = "source_set"               # Source data unit set
    SOURCE_WRITE = "source_write"           # Source data unit write
    
    # Conditional events
    CONDITION_MET = "condition_met"         # Link condition satisfied
    
    # Manual events
    MANUAL_TRANSFER = "manual_transfer"     # Manual link activation
    
    @classmethod
    def from_data_unit_event(cls, data_event: DataUnitEventType) -> 'LinkEventType':
        """Convert data unit event to corresponding link event."""
        mapping = {
            DataUnitEventType.SET: cls.SOURCE_SET,
            DataUnitEventType.WRITE: cls.SOURCE_WRITE,
            DataUnitEventType.APPEND: cls.SOURCE_UPDATED,
        }
        return mapping.get(data_event, cls.SOURCE_UPDATED)
    
    @classmethod
    def from_string(cls, value: str) -> 'LinkEventType':
        """Convert string to enum with validation."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid LinkEventType: '{value}'. Valid values: {[e.value for e in cls]}")


# Type aliases for convenience
EventType = Union[DataUnitEventType, TriggerEventType, LinkEventType]


def validate_event_type(event_type: Union[str, EventType], expected_type: type) -> EventType:
    """
    Validate and convert event type to proper enum.
    
    BRUTAL TRUTH: This function enforces type safety while providing
    backward compatibility for existing string-based configurations.
    
    Args:
        event_type: String or enum value to validate
        expected_type: Expected enum class (DataUnitEventType, TriggerEventType, LinkEventType)
        
    Returns:
        Validated enum value
        
    Raises:
        ValueError: If event_type is invalid for the expected type
    """
    if isinstance(event_type, expected_type):
        return event_type
    
    if isinstance(event_type, str):
        return expected_type.from_string(event_type)
    
    raise ValueError(f"Invalid event type: {event_type}. Expected {expected_type.__name__} or string.")


# BRUTAL TRUTH: Default event types that fix the inconsistent framework defaults
DEFAULT_DATA_UNIT_EVENT = DataUnitEventType.SET
DEFAULT_TRIGGER_EVENT = TriggerEventType.DATA_UPDATED  
DEFAULT_LINK_EVENT = LinkEventType.SOURCE_SET
