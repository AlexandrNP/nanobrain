"""
NanoBrain Configuration Framework

Enhanced from_config pattern system providing:
- Enhanced ConfigBase with integrated loading, schema support, and protocol integration
- Recursive reference resolution for complex configurations
- Comprehensive schema extraction and validation
- Pure from_config pattern without additional managers
"""

from .config_base import ConfigBase
from .yaml_config import YAMLConfig, YAMLWorkflowConfig as WorkflowConfig
# Recursive resolution integrated into ConfigBase.from_config()
# Schema management integrated into ConfigBase

# Legacy components removed as per Phase 3: Legacy Component Removal
# DeprecatedConfigManager and related deprecation helpers removed
# All configuration loading now uses ConfigBase.from_config() exclusively
#
# ✅ FRAMEWORK COMPLIANCE:
# - Pure ConfigBase.from_config() pattern throughout framework
# - No deprecated managers or factory functions
# - Complete configuration-driven component creation
# - Unified interface without legacy compatibility layers

# Export unified interface
__all__ = [
    'ConfigBase',
    'YAMLConfig',
    'WorkflowConfig',
    # Recursive resolution integrated into ConfigBase
    # Schema management integrated into ConfigBase
]

# Version information
__version__ = '2.0.0'
__consolidation_date__ = '2025-01-28'
__framework_pattern__ = 'enhanced_from_config' 