#!/usr/bin/env python3
"""
Response Enhancement Step for RAG systems.
AgentStep wrapper for ResponseSynthesisAgent - intelligent information synthesis.
"""

from typing import Dict, Any
from nanobrain.core.step import AgentStep, StepConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class ResponseEnhancementStep(AgentStep):
    """
    AgentStep wrapper for ResponseSynthesisAgent.

    This step leverages agentic LLM capabilities for intelligent information synthesis:
    - Analyzes relationships between retrieved document chunks
    - Identifies key themes and concepts
    - Resolves contradictions or inconsistencies
    - Creates coherent narrative flow
    - Implements multi-turn LLM strategy for comprehensive responses

    Key Features:
    - Intelligent synthesis using ResponseSynthesisAgent
    - Multi-turn LLM strategy for complex responses
    - Theme analysis and relationship mapping
    - Contradiction resolution and consistency checking
    - Configurable synthesis parameters via YAML
    """

    COMPONENT_TYPE = "response_enhancement_step"

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ResponseEnhancementStep."""
        if config is None:
            config = {}

        # Set agent class and config for parent AgentStep
        config['agent_class'] = 'nanobrain.library.workflows.rag.agents.response_synthesis_agent.ResponseSynthesisAgent'

        # Set agent config path if not provided
        if 'agent_config' not in config:
            config['agent_config'] = 'nanobrain/library/workflows/rag/config/agents/response_synthesis_agent.yml'

        super().__init__(config)

        logger.info("🔄 ResponseEnhancementStep initialized with AgentStep wrapper for ResponseSynthesisAgent")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the ResponseSynthesisAgent and format output for RAG pipeline.

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
                # For response synthesis, map the agent response to the output unit
                formatted_output[unit_name] = agent_response_dict.get('response', '')
        else:
            # Fallback: return raw response if no output data units defined
            formatted_output = agent_response_dict

        logger.debug(f"🔄 Formatted synthesis output for {len(formatted_output)} data units")
        return formatted_output




