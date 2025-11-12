"""
Viral Expert Conversational Agent
=================================

Concrete implementation of ConversationalSpecializedAgent for viral biology expertise.
Provides expert-level conversational responses about viral biology, bioinformatics,
vaccine design, and alphaviruses.

✅ FRAMEWORK COMPLIANCE: Implements required abstract methods
✅ DOMAIN EXPERTISE: Specialized for viral biology and bioinformatics
✅ CONVERSATIONAL: Natural language interaction capabilities
"""

from typing import Optional
from nanobrain.library.agents.specialized.base import ConversationalSpecializedAgent


class ViralExpertConversationalAgent(ConversationalSpecializedAgent):
    """
    Concrete implementation of ConversationalSpecializedAgent for viral expertise.

    This agent provides expert conversational responses about viral biology,
    bioinformatics, vaccine design, and alphaviruses. Implements the required
    abstract methods from the specialized agent base.
    """

    def __init__(self, config=None, **kwargs):
        """Initialize the viral expert conversational agent."""
        super().__init__(config, **kwargs)
        # Ensure _is_initialized is set
        if not hasattr(self, '_is_initialized'):
            self._is_initialized = False

        # Ensure required attributes are initialized (in case parent init didn't run properly)
        if not hasattr(self, '_total_tokens_used'):
            self._total_tokens_used = 0
        if not hasattr(self, '_total_llm_calls'):
            self._total_llm_calls = 0
        if not hasattr(self, '_execution_count'):
            self._execution_count = 0
        if not hasattr(self, '_error_count'):
            self._error_count = 0

    @property
    def logger(self):
        """Compatibility property for logger access"""
        return getattr(self, 'agent_logger', None)

    async def _process_specialized_request(self, input_text: str, **kwargs) -> Optional[str]:
        """
        Process specialized viral biology requests that don't require LLM.

        For comprehensive viral biology expertise, we primarily rely on LLM responses
        to provide detailed, accurate, and contextual information. This method could
        be enhanced to handle simple factual queries directly from databases.
        """
        # For viral biology expertise, we primarily rely on LLM responses
        # to provide comprehensive, contextual, and accurate information
        # This method could be enhanced to handle simple factual queries directly
        return None

    def _should_handle_specialized(self, input_text: str, **kwargs) -> bool:
        """
        Determine if this request should be handled by specialized logic.

        For viral biology expertise, we let all requests go to the LLM
        for comprehensive, expert-level responses with proper context.
        """
        # Let all requests go to the LLM for comprehensive expert responses
        # Could implement keyword-based routing for simple queries in the future
        return False
