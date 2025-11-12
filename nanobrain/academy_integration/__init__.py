"""
Academy Integration Module for Nanobrain Framework

This module provides integration components for connecting Nanobrain workflows
with the Academy framework for distributed execution.

Components:
- AcademyLink: Communication bridge extending LinkBase
- AcademyOrchestrator: Workflow coordination agent
- AcademyStepWrapper: Step-as-agent wrapper
- AcademyDataUnitWrapper: DataUnit-as-agent wrapper
"""

from .academy_link import AcademyLink  # 🔥 ENABLED FOR DISTRIBUTED PROCESSING!

__all__ = [
    'AcademyLink',  # 🔥 RE-ENABLED WITH PROXYSTORE SUPPORT!
]
