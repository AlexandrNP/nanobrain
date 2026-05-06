"""
Viral Protein Analysis Specialized Agents

Agents for viral protein analysis workflows including parameter optimization
and PSSM generation.
"""

from nanobrain.library.agents.specialized.viral_protein_analysis.parameter_proposal_agent import (
    ParameterProposalAgent,
    ParameterProposalAgentConfig
)

__all__ = [
    'ParameterProposalAgent',
    'ParameterProposalAgentConfig'
]
