#!/usr/bin/env python3
"""
Query Enhancement Step for RAG systems.
AgentStep wrapper for QueryEnhancementAgent.
"""

from typing import Dict, Any

from nanobrain.core.step import AgentStep, StepConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class QueryEnhancementStep(AgentStep):
    """
    AgentStep wrapper for QueryEnhancementAgent.

    This is the only step in the RAG pipeline that uses an AgentStep wrapper
    because it wraps the single LLM-based component (QueryEnhancementAgent).
    All other steps extend BaseStep directly for deterministic processing.
    """

    COMPONENT_TYPE = "query_enhancement_step"

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize QueryEnhancementStep."""
        if config is None:
            config = {}

        # Set agent class and config for parent AgentStep
        config['agent_class'] = 'nanobrain.library.workflows.rag.agents.query_enhancement_agent.QueryEnhancementAgent'

        # Set agent config path if not provided
        if 'agent_config' not in config:
            config['agent_config'] = 'nanobrain/library/workflows/rag/config/agents/query_enhancement_agent.yml'

        super().__init__(config)

        logger.info("🔄 QueryEnhancementStep initialized with AgentStep wrapper")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the agent and format output for RAG pipeline.

        This method handles the application-specific output formatting,
        mapping the agent response to the expected output data unit names.
        """
        # Call parent AgentStep process method to get raw agent response
        agent_response_dict = await super().process(input_data)

        # Format output for RAG pipeline - map agent response to output data units
        formatted_output = {}

        # Map the main agent response to the expected output data unit names
        if hasattr(self, 'step_output_data_units') and self.step_output_data_units:
            for unit_name in self.step_output_data_units.keys():
                # For query enhancement, map the agent response to the output unit
                formatted_output[unit_name] = agent_response_dict.get('response', '')
        else:
            # Fallback: return raw response if no output data units defined
            formatted_output = agent_response_dict

        logger.debug(f"🔄 Formatted output for {len(formatted_output)} data units")
        return formatted_output
