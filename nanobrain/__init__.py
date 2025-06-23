"""
NanoBrain Framework

A comprehensive AI agent framework with distributed execution capabilities.
"""

__version__ = "0.1.0"
__author__ = "NanoBrain Team"
__email__ = "team@nanobrain.ai"

# Core framework imports
from . import core
# from . import library  # Temporarily disabled
from . import config

# Convenience imports for common use cases
from .core.agent import ConversationalAgent, AgentConfig
from .core.executor import LocalExecutor, ParslExecutor, ExecutorConfig
from .core.data_unit import DataUnitMemory, DataUnitConfig
from .core.step import Step, StepConfig
from .core.trigger import DataUpdatedTrigger, TriggerConfig
from .core.link import DirectLink, LinkConfig

# Configuration imports
from .core.config.config_manager import get_config_manager

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Modules
    'core',
    # 'library',  # Temporarily disabled
    'config',
    
    # Core classes
    'ConversationalAgent',
    'AgentConfig',
    'LocalExecutor',
    'ParslExecutor',
    'ExecutorConfig',
    'DataUnitMemory',
    'DataUnitConfig',
    'Step',
    'StepConfig',
    'DataUpdatedTrigger',
    'TriggerConfig',
    'DirectLink',
    'LinkConfig',
    
    # Configuration
    'get_config_manager',
] 