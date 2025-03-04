from typing import Any, List, Optional, Set, Dict, Union, Callable, Type
from pydantic import BaseModel
import os
import yaml
import asyncio
import time
import uuid
from pathlib import Path
from enum import Enum
import random


class ComponentState(Enum):
    """Represents different states of framework components."""
    INACTIVE = "INACTIVE"       # Component is not active (like neural resting state)
    ACTIVE = "ACTIVE"           # Currently processing (like neural firing)
    RECOVERING = "RECOVERING"   # Temporary recovery period after activation
    BLOCKED = "BLOCKED"         # Component is prevented from activating
    ENHANCED = "ENHANCED"       # Enhanced sensitivity/performance
    DEGRADED = "DEGRADED"       # Reduced sensitivity/performance
    CONFIGURING = "CONFIGURING" # Component is being reconfigured




