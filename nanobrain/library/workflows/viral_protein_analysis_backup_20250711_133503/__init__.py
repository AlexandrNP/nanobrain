"""
Viral Protein Analysis Workflow Package

This package implements comprehensive viral protein boundary identification workflows
using BV-BRC data integration, sequence analysis, and literature-supported PSSM generation.

Enhanced cache management with BaseCacheManager and VirusSpecificCacheManager.
"""

from .alphavirus_workflow import AlphavirusWorkflow
from .base_cache_manager import BaseCacheManager
from .virus_specific_cache_manager import VirusSpecificCacheManager

__all__ = [
    'AlphavirusWorkflow',
    'BaseCacheManager',
    'VirusSpecificCacheManager'
]

__version__ = "1.0.0"
__author__ = "NanoBrain Team"
__description__ = "Viral protein boundary identification workflows with enhanced cache management" 