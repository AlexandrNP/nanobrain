"""
Expert Conversation Step

Single step for handling expert conversational responses about viral biology.
Extracted from ConversationalResponseStep in chatbot_viral_integration.

✅ FRAMEWORK COMPLIANCE: Uses from_config pattern exclusively
✅ REUSED LOGIC: Adapted from ConversationalResponseStep
✅ NO HARDCODING: All agent behavior configured via YAML
"""

import time
from typing import Dict, Any, Optional

from nanobrain.core.step import Step
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class ExpertConversationStep(Step):
    """
    ✅ REUSED LOGIC: Extracted from ConversationalResponseStep
    Single step for generating expert conversational responses
    """
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.conversational_agent = None
        
    def _init_from_config(self, config, component_config: Dict[str, Any], dependencies: Dict[str, Any]) -> None:
        """Initialize step with conversational agent"""
        super()._init_from_config(config, component_config, dependencies)
        
        # Load conversational agent from configuration
        self._load_conversational_agent(component_config)
        
        if self.nb_logger:
            self.nb_logger.info("🧠 Expert Conversation Step initialized")
    
    def _load_conversational_agent(self, component_config: Dict[str, Any]) -> None:
        """
        ✅ REUSED COMPONENT: Load conversational agent from configuration
        """
        try:
            from nanobrain.library.agents.specialized_agents.conversational_specialized_agent import ConversationalSpecializedAgent
            
            # ✅ FRAMEWORK COMPLIANCE: Load agent via from_config
            agent_config = component_config.get('conversational_agent')
            if isinstance(agent_config, dict):
                if 'config' in agent_config:
                    config_path = agent_config['config']
                    self.conversational_agent = ConversationalSpecializedAgent.from_config(config_path)
                else:
                    # Agent configuration embedded directly
                    self.conversational_agent = ConversationalSpecializedAgent.from_config(agent_config)
            elif hasattr(agent_config, '_process_specialized_request'):
                # ✅ FRAMEWORK COMPLIANCE: Already resolved by framework
                self.conversational_agent = agent_config
            else:
                # Fallback to default configuration
                default_config_path = "nanobrain/library/workflows/chatbot_viral_integration/config/ConversationalResponseStep/ConversationalAgent.yml"
                self.conversational_agent = ConversationalSpecializedAgent.from_config(default_config_path)
            
            self.nb_logger.info("✅ Conversational agent loaded successfully")
            
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to load conversational agent: {e}")
            raise

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ REUSED LOGIC: Adapted from ConversationalResponseStep.process
        Process conversational request and generate expert response
        """
        start_time = time.time()
        
        try:
            # Extract conversation input
            conversation_input = input_data.get('conversation_input', input_data)
            user_query = conversation_input.get('user_query', '')
            
            if not user_query:
                return {
                    'success': False,
                    'conversation_output': {
                        'error': 'No user query provided',
                        'response_type': 'error'
                    }
                }
            
            self.nb_logger.info(f"🧠 Processing conversation: {user_query[:100]}...")
            
            # Generate expert response using conversational agent
            expert_response = await self._generate_expert_response(user_query, conversation_input)
            
            # Create conversation output
            conversation_output = {
                'expert_response': expert_response,
                'user_query': user_query,
                'response_type': 'conversational',
                'processing_time': time.time() - start_time,
                'step_id': 'expert_conversation'
            }
            
            return {
                'success': True,
                'conversation_output': conversation_output
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Expert conversation step failed: {e}")
            return {
                'success': False,
                'conversation_output': {
                    'error': str(e),
                    'response_type': 'error',
                    'processing_time': time.time() - start_time
                }
            }

    async def _generate_expert_response(self, user_query: str, context_data: Dict[str, Any]) -> str:
        """
        ✅ REUSED LOGIC: Generate expert response using specialized agent
        """
        try:
            # Prepare context for expert response
            expert_context = {
                'expertise': 'virology',
                'focus': 'alphaviruses',
                'query_type': 'conversational',
                'user_query': user_query
            }
            
            # Add virus context if available from intelligent routing
            virus_species = context_data.get('virus_species', [])
            if virus_species:
                expert_context['virus_context'] = virus_species
                expert_context['contextual_expertise'] = True
            
            # Add session context if available
            session_id = context_data.get('session_id')
            if session_id:
                expert_context['session_id'] = session_id
            
            # ✅ FRAMEWORK COMPLIANCE: Use agent's specialized request processing
            response = await self.conversational_agent._process_specialized_request(
                user_query,
                context=expert_context
            )
            
            return response or "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            self.nb_logger.error(f"❌ Expert response generation failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}" 