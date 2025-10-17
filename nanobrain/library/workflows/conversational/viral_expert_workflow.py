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

        self.nb_logger.info(
            "🧠 Initializing Viral Expert Conversational Workflow")

        # Initialize response formatter (sync)
        self._initialize_response_formatter()

        self.nb_logger.info(
            "✅ Viral Expert Conversational Workflow basic initialization completed - agent will be loaded in initialize()")

    async def initialize(self) -> None:
        """
        ✅ ASYNC INITIALIZATION: Initialize workflow without redundant agent creation
        The step will handle agent creation and initialization
        """
        # Initialize parent first
        await super().initialize()

        # ✅ CRITICAL FIX: Initialize and start all workflow links
        await self._initialize_workflow_links()

        # ✅ CRITICAL FIX: Connect workflow-level triggers to link callbacks
        await self._setup_trigger_callbacks()

        # PHASE 2 DEBUG: Add debugging to see what data is received by the workflow
        if hasattr(self, 'data_units') and 'user_query' in self.data_units:
            user_query_data_unit = self.data_units['user_query']
            original_set_data = user_query_data_unit.set_data

            def debug_set_data(data):
                self.nb_logger.info(f"🔍 [WORKFLOW-DEBUG] user_query received data type: {type(data)}")
                self.nb_logger.info(f"🔍 [WORKFLOW-DEBUG] user_query received data: {data}")
                return original_set_data(data)

            user_query_data_unit.set_data = debug_set_data

        self.nb_logger.info(
            "✅ Viral Expert Workflow initialized - agents handled by steps")

    async def _initialize_workflow_links(self) -> None:
        """
        ✅ CRITICAL FIX: Verify workflow links are properly set up
        The framework already initializes and starts links automatically
        """
        try:
            # Get workflow links
            if not hasattr(self, 'step_links'):
                self.nb_logger.warning("⚠️ No workflow links found")
                return

            self.nb_logger.info(f"🔗 Verifying {len(self.step_links)} workflow links")

            # Verify all links are properly configured
            for link_id, link in self.step_links.items():
                if hasattr(link, '_is_active') and link._is_active:
                    self.nb_logger.info(f"✅ Link active: {link_id}")
                else:
                    self.nb_logger.warning(f"⚠️ Link not active: {link_id}")

            self.nb_logger.info("✅ All workflow links verified")

        except Exception as e:
            self.nb_logger.error(f"❌ Failed to verify workflow links: {e}", exc_info=True)
            raise

    async def _setup_trigger_callbacks(self) -> None:
        """
        ✅ CRITICAL FIX: Connect workflow-level triggers to link transfer methods
        This is the missing piece that enables data flow through the workflow
        """
        try:
            # Get workflow-level triggers
            if not hasattr(self, '_workflow_triggers'):
                self.nb_logger.warning("⚠️ No workflow triggers found")
                return

            # Get workflow links
            if not hasattr(self, 'step_links'):
                self.nb_logger.warning("⚠️ No workflow links found")
                return

            # Connect conversation_start trigger to input_to_conversation link
            conversation_start_trigger = None
            input_to_conversation_link = None

            # Find the conversation_start trigger
            for trigger_id, trigger in self._workflow_triggers.items():
                if hasattr(trigger, 'trigger_id') and trigger.trigger_id == 'conversation_start':
                    conversation_start_trigger = trigger
                    break
                elif trigger_id == 'conversation_start':
                    conversation_start_trigger = trigger
                    break

            # Find the input_to_conversation link
            if 'input_to_conversation' in self.step_links:
                input_to_conversation_link = self.step_links['input_to_conversation']

            # Connect trigger to link
            if conversation_start_trigger and input_to_conversation_link:
                await conversation_start_trigger.add_callback(input_to_conversation_link.transfer)
                self.nb_logger.info(
                    "✅ Connected conversation_start trigger to input_to_conversation link")
            else:
                self.nb_logger.warning(
                    f"⚠️ Failed to connect trigger to link: "
                    f"trigger={conversation_start_trigger is not None}, "
                    f"link={input_to_conversation_link is not None}")

        except Exception as e:
            self.nb_logger.error(f"❌ Failed to setup trigger callbacks: {e}", exc_info=True)
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
            self.nb_logger.error(
                f"❌ Failed to initialize response formatter: {e}", exc_info=True)
            raise

    # ✅ STEP-BASED PROCESSING: Workflow now delegates to steps
    # The ExpertConversationStep handles all agent processing

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
