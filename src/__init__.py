"""
NanoBrain Framework - Simplified and Refactored

A modern, async-first framework for orchestrating AI agents, data processing steps,
and HPC workflows with unified YAML configuration.

Key Components:
- Step: DataUnit-based processing with triggers
- Agent: Tool-calling based AI processing  
- Executor: Configurable execution (local, Parsl HPC)
- DataUnit: Data interfaces and ingestion
- Link: Dataflow abstractions for Steps
"""

import logging
import os
from pathlib import Path

from .core.executor import ExecutorBase, LocalExecutor
from .core.step import Step
from .core.agent import Agent
from .core.data_unit import DataUnitBase
from .core.link import LinkBase
from .core.trigger import TriggerBase
from .core.tool import ToolBase

# Import configuration management
try:
    # Try to import from the config directory at the same level as src
    import sys
    config_path = Path(__file__).parent.parent / "config"
    sys.path.insert(0, str(config_path))
    
    from config_manager import (
        get_config_manager, 
        get_api_key, 
        get_provider_config, 
        get_default_model,
        initialize_config,
        ProviderConfig
    )
    
    # Initialize global configuration on import
    _config_manager = get_config_manager()
    
    # Log framework initialization
    framework_info = _config_manager.get_framework_info()
    logger = logging.getLogger(__name__)
    logger.info(f"Initialized {framework_info.get('name', 'NanoBrain')} v{framework_info.get('version', '2.0.0')}")
    
    # Validate API keys if configured to do so
    if _config_manager._config.get('security', {}).get('validate_keys_on_startup', False):
        validation_results = _config_manager.validate_api_keys()
        valid_providers = [name for name, valid in validation_results.items() if valid]
        if valid_providers:
            logger.info(f"Valid API keys found for: {', '.join(valid_providers)}")
    
    # Add configuration functions to exports
    __all__ = [
        "ExecutorBase", "LocalExecutor",
        "Step", "Agent", 
        "DataUnitBase", "LinkBase", "TriggerBase",
        "ToolBase",
        # Configuration management
        "get_config_manager", "get_api_key", "get_provider_config", 
        "get_default_model", "initialize_config", "ProviderConfig"
    ]
    
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not load configuration manager: {e}")
    logger.warning("API key management will not be available")
    
    __all__ = [
        "ExecutorBase", "LocalExecutor",
        "Step", "Agent", 
        "DataUnitBase", "LinkBase", "TriggerBase",
        "ToolBase"
    ]

__version__ = "2.0.0" 