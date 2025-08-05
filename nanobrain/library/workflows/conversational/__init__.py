"""
Conversational Workflows Package

Contains workflows specialized for conversational interactions, particularly
focused on expert discussions about scientific topics.

✅ FRAMEWORK COMPLIANCE: All workflows use from_config pattern
✅ CLEAN ARCHITECTURE: Extracted from chatbot_viral_integration
"""

from .viral_expert_workflow import ViralExpertWorkflow

__all__ = [
    'ViralExpertWorkflow'
]

__version__ = "1.0.0"
__author__ = "NanoBrain Team"
__description__ = "Conversational workflows for expert scientific discussions" 