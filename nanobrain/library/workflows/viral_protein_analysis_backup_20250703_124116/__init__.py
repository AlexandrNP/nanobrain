"""
Viral Protein Analysis Workflow Package

This package implements comprehensive viral protein boundary identification workflows
using BV-BRC data integration, sequence analysis, and literature-supported PSSM generation.

Based on the detailed implementation plans in docs/PHASE2_IMPLEMENTATION_PLAN.md
and docs/VIRAL_PROTEIN_WORKFLOW_DETAILED_IMPLEMENTATION_PLAN.md
"""

from .alphavirus_workflow import AlphavirusWorkflow
from .eeev_workflow import EEEVWorkflow

__all__ = [
    'AlphavirusWorkflow',
    'EEEVWorkflow'
]

__version__ = "1.0.0"
__author__ = "NanoBrain Team"
__description__ = "Viral protein boundary identification workflows" 