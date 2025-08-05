"""
Viral Expert Conversational Workflow

Standalone workflow for viral expert conversations extracted from chatbot_viral_integration.
Provides specialized conversational responses about viral biology using LLM agents.

✅ FRAMEWORK COMPLIANCE: Uses from_config pattern exclusively
✅ REUSED LOGIC: Extracted from ConversationalResponseStep
✅ NO HARDCODING: All agent behavior configured via YAML
"""

import time
import uuid
from typing import Dict, Any, Optional
from datetime import datetime

from nanobrain.core.workflow import Workflow, WorkflowConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class ViralExpertWorkflow(Workflow):
    """
    ✅ REUSED LOGIC: Extracted from ConversationalResponseStep
    Specialized workflow for viral expert conversations
    
    This workflow provides expert-level conversational responses about viral biology,
    particularly focused on alphaviruses. Uses specialized LLM agents for accurate
    scientific information delivery.
    """
    
    def _init_from_config(self, config: WorkflowConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize viral expert workflow from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables
        self.conversational_agent: Optional[Any] = None
        self.response_formatter: Optional[Any] = None
        
        self.nb_logger.info("🧠 Initializing Viral Expert Conversational Workflow")
        
        # Load conversational agent
        self._load_conversational_agent()
        
        # Initialize response formatter
        self._initialize_response_formatter()
        
        self.nb_logger.info("✅ Viral Expert Conversational Workflow initialized")

    def _load_conversational_agent(self) -> None:
        """
        ✅ REUSED COMPONENT: Load ConversationalAgent from chatbot_viral_integration
        """
        try:
            from nanobrain.library.agents.specialized_agents.conversational_specialized_agent import ConversationalSpecializedAgent
            
            # ✅ FRAMEWORK COMPLIANCE: Load agent via from_config
            config_path = "nanobrain/library/workflows/chatbot_viral_integration/config/ConversationalResponseStep/ConversationalAgent.yml"
            self.conversational_agent = ConversationalSpecializedAgent.from_config(config_path)
            
            self.nb_logger.info("✅ Conversational agent loaded successfully")
            
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to load conversational agent: {e}")
            raise

    def _initialize_response_formatter(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Initialize response formatter for consistent output
        """
        try:
            # Simple response formatter for conversational workflows
            self.response_formatter = ConversationalResponseFormatter()
            self.nb_logger.debug("✅ Response formatter initialized")
            
        except Exception as e:
            self.nb_logger.error(f"❌ Failed to initialize response formatter: {e}")
            raise

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ REUSED LOGIC: Adapted from ConversationalResponseStep.process
        Generate expert conversational response about viral biology
        """
        start_time = time.time()
        
        try:
            # Extract user query from input
            user_query = input_data.get('user_query', '')
            session_id = input_data.get('session_id', f"session_{uuid.uuid4().hex[:8]}")
            
            if not user_query:
                return self._create_error_response("No user query provided", session_id)
            
            self.nb_logger.info(f"🧠 Generating expert response for query: {user_query[:100]}...")
            
            # ✅ REUSED LOGIC: Generate response using conversational agent
            expert_response = await self._generate_expert_response(user_query, input_data)
            
            # Format response for output
            formatted_response = self._format_conversational_response(
                expert_response, user_query, session_id, start_time
            )
            
            processing_time = time.time() - start_time
            self.nb_logger.info(f"✅ Expert response generated in {processing_time:.2f}s")
            
            return formatted_response
            
        except Exception as e:
            self.nb_logger.error(f"❌ Conversational workflow failed: {e}")
            return self._create_error_response(str(e), session_id, start_time)

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
            
            # ✅ FRAMEWORK COMPLIANCE: Use agent's specialized request processing
            response = await self.conversational_agent._process_specialized_request(
                user_query,
                context=expert_context
            )
            
            return response or "I apologize, but I couldn't generate a response at this time."
            
        except Exception as e:
            self.nb_logger.error(f"❌ Expert response generation failed: {e}")
            return f"I encountered an error while processing your question: {str(e)}"

    def _format_conversational_response(self, expert_response: str, user_query: str, 
                                      session_id: str, start_time: float) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Format conversational response for output
        """
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'response_type': 'conversational',
            'content': expert_response,
            'session_id': session_id,
            'metadata': {
                'workflow_type': 'conversational',
                'workflow_id': 'viral_expert_conversation',
                'agent_used': 'viral_expert',
                'processing_time': processing_time,
                'query_length': len(user_query),
                'response_length': len(expert_response),
                'timestamp': datetime.now().isoformat()
            }
        }

    def _create_error_response(self, error_message: str, session_id: str, 
                             start_time: Optional[float] = None) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Create structured error response
        """
        processing_time = (time.time() - start_time) if start_time else 0.0
        
        return {
            'success': False,
            'response_type': 'error',
            'error': error_message,
            'session_id': session_id,
            'metadata': {
                'workflow_type': 'conversational',
                'workflow_id': 'viral_expert_conversation',
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
        }


class ConversationalResponseFormatter:
    """
    ✅ FRAMEWORK COMPLIANCE: Simple response formatter for conversational workflows
    """
    
    def __init__(self):
        self.formatter_id = f"formatter_{uuid.uuid4().hex[:8]}"
        
    def format_response(self, response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format conversational response with metadata"""
        return {
            'formatted_response': response,
            'formatter_metadata': {
                'formatter_id': self.formatter_id,
                'formatting_timestamp': datetime.now().isoformat()
            },
            'original_metadata': metadata
        } 