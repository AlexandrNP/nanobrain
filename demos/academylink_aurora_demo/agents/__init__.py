"""
Real Academy Agents for Aurora Demo

This package contains real Academy agents that provide distributed execution
capabilities for the Aurora computation workflow via AcademyLink.
"""

from .aurora_computation_agent import AuroraComputationAgent
from .aurora_results_agent import AuroraResultsAgent

__all__ = [
    'AuroraComputationAgent',
    'AuroraResultsAgent'
]
