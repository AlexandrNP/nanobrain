"""
Viral Protein Analysis Web Package

Web-optimized components for viral protein analysis workflows including
progress tracking, result formatting, and client/server optimization.

✅ FRAMEWORK COMPLIANCE: All components use from_config pattern
✅ WEB OPTIMIZATION: Designed for client/server architectures
"""

from .viral_analysis_web_workflow import ViralAnalysisWebWorkflow, WebProgressTracker, WebResultFormatter

__all__ = [
    'ViralAnalysisWebWorkflow',
    'WebProgressTracker', 
    'WebResultFormatter'
]

__version__ = "1.0.0"
__author__ = "NanoBrain Team"
__description__ = "Web-optimized viral protein analysis workflows" 