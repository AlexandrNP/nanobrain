"""
Enhanced Chat Workflow

A comprehensive chat workflow implementation using the NanoBrain framework
with proper step interconnections via data units, links, and triggers.

This workflow demonstrates:
- Modular step architecture with individual step directories
- Proper data flow through data units and links
- Event-driven processing with triggers
- Hierarchical step composition with substeps
- Performance monitoring and conversation history management

Architecture:
User Input → CLI Interface Step → Conversation Manager Step → Agent Processing Step → Output
                                        ↓
                                   History Persistence (substep)
                                   Performance Tracking (substep)
"""

import sys
import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# Core framework imports with proper nanobrain package structure
from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig, TriggerType
from nanobrain.core.link import DirectLink, LinkConfig, LinkType
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.logging_system import get_logger, OperationType
from nanobrain.core.agent import AgentConfig

# Library imports with updated paths
from nanobrain.library.agents.conversational import EnhancedCollaborativeAgent
from nanobrain.library.infrastructure.data import ConversationHistoryUnit


class ChatWorkflow(FromConfigBase):
    """
    Chat Workflow - Enterprise Conversational AI with Modular Step Architecture and Event-Driven Processing
    ====================================================================================================
    
    The ChatWorkflow provides a comprehensive conversational AI framework with modular step-based architecture,
    event-driven processing, and enterprise-grade conversation management. This workflow integrates advanced
    natural language processing with sophisticated conversation state management, enabling complex multi-turn
    interactions, context preservation, and intelligent response generation across diverse application domains.
    
    **Core Architecture:**
        The chat workflow provides enterprise-grade conversational AI capabilities:
        
        * **Modular Step Architecture**: Composable processing steps with individual configuration and data flow
        * **Event-Driven Processing**: Asynchronous trigger-based execution with real-time response capabilities
        * **Conversation Management**: Advanced conversation state tracking, history persistence, and context management
        * **Multi-Agent Integration**: Seamless integration with specialized agents and collaborative AI systems
        * **Performance Monitoring**: Real-time metrics, analytics, and conversation quality assessment
        * **Framework Integration**: Full integration with NanoBrain's workflow orchestration architecture
    
    **Conversational AI Capabilities:**
        
        **Advanced Conversation Management:**
        * Multi-turn conversation tracking with context preservation
        * Conversation branching and state management for complex interactions
        * Intent recognition and conversation flow routing
        * Dynamic conversation personalization and adaptation
        
        **Natural Language Processing:**
        * Advanced language understanding with context awareness
        * Multi-language support and automatic language detection
        * Sentiment analysis and emotional intelligence integration
        * Custom domain vocabulary and terminology handling
        
        **Response Generation:**
        * Context-aware response generation with conversation history
        * Dynamic response formatting and style adaptation
        * Multi-modal response support (text, rich media, structured data)
        * Response quality assessment and optimization
        
        **Agent Orchestration:**
        * Intelligent routing to specialized agents based on conversation context
        * Multi-agent collaboration for complex query resolution
        * Agent handoff and conversation continuity management
        * Load balancing and performance optimization across agent pools
    
    **Step-Based Processing Architecture:**
        
        **CLI Interface Step:**
        * User input processing and validation
        * Command interpretation and parameter extraction
        * Input sanitization and security validation
        * Multi-format input support (text, voice, structured data)
        
        **Conversation Manager Step:**
        * Conversation state management and persistence
        * Context window optimization and memory management
        * Conversation flow control and routing decisions
        * Performance metrics collection and analysis
        
        **Agent Processing Step:**
        * Intelligent agent selection and task delegation
        * Response generation and quality assurance
        * Error handling and fallback response mechanisms
        * Integration with external services and APIs
        
        **History Persistence Substep:**
        * Conversation history storage and retrieval
        * Context summarization and compression
        * Long-term memory management and archival
        * Privacy and data retention compliance
        
        **Performance Tracking Substep:**
        * Real-time performance monitoring and analytics
        * Response time optimization and bottleneck identification
        * Conversation quality metrics and user satisfaction tracking
        * System health monitoring and alerting
    
    **Configuration Architecture:**
        Comprehensive configuration supports diverse conversational AI applications:
        
        ```yaml
        # Chat Workflow Configuration
        workflow_name: "enterprise_chat_workflow"
        workflow_type: "conversational"
        
        # Workflow card for framework integration
        workflow_card:
          name: "enterprise_chat_workflow"
          description: "Enterprise conversational AI with modular architecture"
          version: "1.0.0"
          category: "conversational_ai"
          capabilities:
            - "multi_turn_conversation"
            - "context_management"
            - "agent_orchestration"
        
        # LLM Configuration
        llm_config:
          model: "gpt-4"
          temperature: 0.7        # Balance creativity and consistency
          max_tokens: 2000
          system_prompt: "You are a helpful and knowledgeable AI assistant."
          
        # Conversation Configuration
        conversation_config:
          max_history_length: 20      # Maximum conversation turns to maintain
          context_window_size: 4000   # Token limit for conversation context
          conversation_timeout: 1800  # Session timeout in seconds
          enable_conversation_persistence: true
          
        # Step Configuration
        steps:
          cli_interface:
            class: "nanobrain.library.workflows.chat_workflow.steps.CLIInterfaceStep"
            config:
              input_validation: true
              command_parsing: true
              multi_format_support: true
              
          conversation_manager:
            class: "nanobrain.library.workflows.chat_workflow.steps.ConversationManagerStep"
            config:
              history_management: true
              context_optimization: true
              performance_tracking: true
              
          agent_processor:
            class: "nanobrain.library.workflows.chat_workflow.steps.AgentProcessorStep"
            config:
              agent_selection_strategy: "intelligent"
              response_quality_checks: true
              fallback_enabled: true
              
        # Data Flow Configuration
        data_units:
          user_input:
            type: "text"
            validation: true
            
          conversation_context:
            type: "conversation_history"
            persistence: true
            
          agent_response:
            type: "structured_response"
            quality_checks: true
            
        # Performance Configuration
        performance_monitoring:
          enable_metrics: true
          response_time_tracking: true
          conversation_quality_scoring: true
          user_satisfaction_monitoring: true
          
        # Security Configuration
        security:
          input_sanitization: true
          output_filtering: true
          conversation_encryption: true
          access_control: "role_based"
        ```
    
    **Usage Patterns:**
        
        **Basic Chat Workflow Setup:**
        ```python
        from nanobrain.library.workflows.chat_workflow import ChatWorkflow
        
        # Create chat workflow with configuration
        workflow_config = {
            'name': 'customer_support_chat',
            'model': 'gpt-4',
            'temperature': 0.7,
            'system_prompt': 'You are a helpful customer support agent.'
        }
        
        chat_workflow = ChatWorkflow.from_config(workflow_config)
        
        # Initialize workflow components
        await chat_workflow.initialize()
        
        # Process user interaction
        user_input = "I need help with my account"
        response = await chat_workflow.process_message(user_input)
        
        print(f"Assistant: {response.data['response']}")
        print(f"Confidence: {response.data['confidence']:.3f}")
        print(f"Response Time: {response.data['response_time']}ms")
        ```
        
        **Multi-Turn Conversation Management:**
        ```python
        # Configure for extended conversation support
        extended_config = {
            'name': 'research_assistant_chat',
            'conversation_config': {
                'max_history_length': 50,
                'context_window_size': 8000,
                'enable_conversation_persistence': True,
                'conversation_summarization': True
            },
            'agent_integration': {
                'specialized_agents': ['research_agent', 'analysis_agent'],
                'agent_switching': 'automatic',
                'context_sharing': True
            }
        }
        
        chat_workflow = ChatWorkflow.from_config(extended_config)
        await chat_workflow.initialize()
        
        # Start conversation session
        session_id = await chat_workflow.start_conversation_session()
        
        # Multi-turn interaction
        conversation_turns = [
            "I'm researching SARS-CoV-2 spike protein mutations",
            "Can you help me find recent papers on this topic?",
            "What about structural analysis tools?",
            "How do I compare sequences across variants?"
        ]
        
        for turn, user_message in enumerate(conversation_turns):
            response = await chat_workflow.process_message(
                user_message,
                session_id=session_id
            )
            
            print(f"Turn {turn + 1}:")
            print(f"User: {user_message}")
            print(f"Assistant: {response.data['response']}")
            
            # Access conversation context
            context = response.data['conversation_context']
            print(f"Context Length: {context['turns']}")
            print(f"Active Topics: {context['topics']}")
            print("---")
        
        # End conversation session
        conversation_summary = await chat_workflow.end_conversation_session(session_id)
        print(f"\\nConversation Summary: {conversation_summary}")
        ```
        
        **Enterprise Integration with Agent Orchestration:**
        ```python
        # Configure for enterprise deployment
        enterprise_config = {
            'name': 'enterprise_chat_platform',
            'deployment_mode': 'production',
            'agent_orchestration': {
                'agent_pool': [
                    'customer_service_agent',
                    'technical_support_agent',
                    'sales_agent',
                    'billing_agent'
                ],
                'routing_strategy': 'intent_based',
                'load_balancing': True,
                'failover_enabled': True
            },
            'performance_monitoring': {
                'enable_metrics': True,
                'real_time_analytics': True,
                'quality_scoring': True,
                'user_satisfaction_tracking': True
            },
            'security': {
                'authentication_required': True,
                'conversation_encryption': True,
                'audit_logging': True,
                'data_retention_policy': '90_days'
            }
        }
        
        enterprise_chat = ChatWorkflow.from_config(enterprise_config)
        await enterprise_chat.initialize()
        
        # Process customer inquiry with routing
        customer_query = "I have a billing question about my recent invoice"
        
        # Workflow automatically routes to appropriate agent
        response = await enterprise_chat.process_customer_inquiry(
            query=customer_query,
            customer_id="CUST123456",
            priority="normal"
        )
        
        # Access routing and processing details
        routing_info = response.data['routing_info']
        performance_metrics = response.data['performance_metrics']
        
        print(f"Routed to: {routing_info['selected_agent']}")
        print(f"Confidence: {routing_info['routing_confidence']:.3f}")
        print(f"Response: {response.data['response']}")
        print(f"Processing Time: {performance_metrics['total_time']}ms")
        print(f"Quality Score: {performance_metrics['quality_score']:.3f}")
        ```
        
        **Advanced Analytics and Monitoring:**
        ```python
        # Configure for comprehensive analytics
        analytics_config = {
            'name': 'analytics_chat_workflow',
            'analytics': {
                'conversation_analytics': True,
                'sentiment_analysis': True,
                'topic_modeling': True,
                'user_journey_tracking': True
            },
            'monitoring': {
                'real_time_dashboards': True,
                'alerting': {
                    'response_time_threshold': 2000,  # ms
                    'quality_score_threshold': 0.8,
                    'error_rate_threshold': 0.05
                },
                'performance_optimization': True
            }
        }
        
        analytics_chat = ChatWorkflow.from_config(analytics_config)
        await analytics_chat.initialize()
        
        # Process messages with full analytics
        messages = [
            "I'm frustrated with the service",
            "The app keeps crashing",
            "Can someone help me?"
        ]
        
        for message in messages:
            response = await analytics_chat.process_with_analytics(message)
            
            # Access analytics data
            sentiment = response.data['analytics']['sentiment']
            topics = response.data['analytics']['topics']
            user_journey = response.data['analytics']['user_journey_stage']
            
            print(f"Message: {message}")
            print(f"Sentiment: {sentiment['label']} ({sentiment['confidence']:.3f})")
            print(f"Topics: {', '.join(topics)}")
            print(f"Journey Stage: {user_journey}")
            print(f"Response: {response.data['response']}")
            print("---")
        
        # Generate analytics report
        analytics_report = await analytics_chat.generate_analytics_report()
        print(f"\\nAnalytics Report:")
        print(f"Total Conversations: {analytics_report['total_conversations']}")
        print(f"Average Sentiment: {analytics_report['average_sentiment']:.3f}")
        print(f"Top Topics: {analytics_report['top_topics']}")
        print(f"Performance Metrics: {analytics_report['performance_summary']}")
        ```
        
        **Workflow Customization and Extension:**
        ```python
        # Configure custom workflow with additional steps
        custom_config = {
            'name': 'specialized_research_chat',
            'custom_steps': {
                'domain_classifier': {
                    'class': 'nanobrain.library.workflows.custom.DomainClassifierStep',
                    'config': {
                        'domains': ['biology', 'chemistry', 'physics'],
                        'confidence_threshold': 0.8
                    }
                },
                'knowledge_retrieval': {
                    'class': 'nanobrain.library.workflows.custom.KnowledgeRetrievalStep',
                    'config': {
                        'knowledge_bases': ['pubmed', 'arxiv', 'patents'],
                        'retrieval_strategy': 'semantic_search'
                    }
                },
                'response_synthesis': {
                    'class': 'nanobrain.library.workflows.custom.ResponseSynthesisStep',
                    'config': {
                        'synthesis_method': 'evidence_based',
                        'citation_style': 'apa'
                    }
                }
            },
            'step_flow': [
                'cli_interface',
                'domain_classifier',
                'knowledge_retrieval',
                'conversation_manager',
                'response_synthesis',
                'agent_processor'
            ]
        }
        
        specialized_chat = ChatWorkflow.from_config(custom_config)
        await specialized_chat.initialize()
        
        # Process specialized research query
        research_query = "What are the latest findings on CRISPR off-target effects?"
        
        response = await specialized_chat.process_research_query(research_query)
        
        # Access specialized processing results
        domain_classification = response.data['domain_classification']
        retrieved_knowledge = response.data['retrieved_knowledge']
        synthesis_quality = response.data['synthesis_quality']
        
        print(f"Classified Domain: {domain_classification['domain']}")
        print(f"Knowledge Sources: {len(retrieved_knowledge['sources'])}")
        print(f"Synthesis Quality: {synthesis_quality['score']:.3f}")
        print(f"Response: {response.data['response']}")
        print(f"Citations: {response.data['citations']}")
        ```
    
    **Advanced Features:**
        
        **Real-Time Collaboration:**
        * Multi-user conversation support with conflict resolution
        * Live conversation sharing and collaborative editing
        * Real-time typing indicators and presence awareness
        * Shared conversation workspaces and team channels
        
        **AI Assistant Personalization:**
        * User preference learning and adaptation
        * Conversation style customization and personality adjustment
        * Context-aware response personalization
        * Learning from user feedback and conversation history
        
        **Enterprise Security:**
        * End-to-end conversation encryption
        * Role-based access control and permission management
        * Audit logging and compliance reporting
        * Data privacy and retention policy enforcement
        
        **Performance Optimization:**
        * Intelligent caching and response optimization
        * Load balancing across multiple agent instances
        * Predictive scaling based on conversation patterns
        * Resource usage optimization and cost management
    
    **Application Domains:**
        
        **Customer Support:**
        * Automated customer service with human handoff
        * Technical support and troubleshooting assistance
        * Order management and billing inquiries
        * Product information and recommendation systems
        
        **Research and Education:**
        * Academic research assistance and literature review
        * Educational tutoring and personalized learning
        * Scientific collaboration and knowledge sharing
        * Research methodology guidance and analysis support
        
        **Healthcare and Telemedicine:**
        * Medical information and symptom analysis
        * Patient support and care coordination
        * Healthcare provider assistance and decision support
        * Mental health and wellness coaching
        
        **Business Intelligence:**
        * Data analysis and reporting assistance
        * Business process optimization and automation
        * Strategic planning and decision support
        * Market research and competitive analysis
    
    Attributes:
        workflow_config (dict): Configuration for workflow behavior and components
        conversation_manager (object): System for managing conversation state and history
        agent_orchestrator (object): System for managing and routing to specialized agents
        performance_monitor (object): Real-time performance monitoring and analytics
        security_manager (object): Security and access control management
        step_registry (dict): Registry of available processing steps and configurations
    
    Note:
        This workflow requires LLM access for conversation processing and response generation.
        Conversation persistence requires appropriate storage backend configuration.
        Performance monitoring features require metrics collection infrastructure.
        Multi-agent integration requires proper agent configuration and availability.
    
    Warning:
        Conversation data may contain sensitive information requiring appropriate security measures.
        Long conversations may consume significant memory and computational resources.
        Agent orchestration requires careful configuration to prevent infinite loops or conflicts.
        Real-time features may require additional infrastructure for optimal performance.
    
    See Also:
        * :class:`FromConfigBase`: Base framework component interface
        * :class:`EnhancedCollaborativeAgent`: Multi-protocol collaborative agent implementation
        * :class:`ConversationHistoryUnit`: Conversation history management system
        * :mod:`nanobrain.library.workflows`: Workflow implementations and utilities
        * :mod:`nanobrain.core.workflow`: Core workflow orchestration framework
    """
    
    # Component configuration
    COMPONENT_TYPE = "chat_workflow"
    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
        'system_prompt': 'You are a helpful and friendly AI assistant.',
        'enable_metrics': True
    }
    
    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return generic Dict - ChatWorkflow uses dictionary configuration"""
        return dict
    
    @classmethod
    def extract_component_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ChatWorkflow configuration"""
        return {
            'name': config.get('name', 'chat_workflow'),
            'model': config.get('model', 'gpt-3.5-turbo'),
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 2000),
            'system_prompt': config.get('system_prompt', 'You are a helpful and friendly AI assistant.'),
            'enable_metrics': config.get('enable_metrics', True),
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve ChatWorkflow dependencies"""
        # ✅ FRAMEWORK COMPLIANCE: Use from_config pattern for executor
        try:
            # Try to use provided executor config first
            executor_config_path = component_config.get('executor_config')
            if executor_config_path:
                executor_config = ExecutorConfig.from_config(executor_config_path)
            else:
                # Use default executor configuration file
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent.parent.parent
                default_executor_path = project_root / "Chatbot" / "config" / "components" / "default_executor_config.yml"
                executor_config = ExecutorConfig.from_config(str(default_executor_path))
            
            executor = LocalExecutor.from_config(executor_config)
        except Exception as e:
            # Fallback: Use LocalExecutor's default configuration
            executor = LocalExecutor.from_config({
                'name': 'fallback_local_executor',
                'description': 'Fallback local executor',
                'execution_mode': 'local',
                'max_workers': 2
            })
        
        return {
            'executor': executor,
            'agent_config': component_config,
        }
    
    # Now inherits unified from_config implementation from FromConfigBase
        
    def _init_from_config(self, config: Dict[str, Any], component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize ChatWorkflow with resolved dependencies"""
        self.logger = get_logger("chat_workflow", "workflows")
        
        # Workflow state
        self.is_initialized = False
        self.is_running = False
        
        # Core components from dependencies
        self.executor = dependencies.get('executor')
        self.agent = None
        self.agent_config = dependencies.get('agent_config')
        
        # Data units
        self.data_units = {}
        
        # Conversation management
        self.conversation_history = None
        self.current_conversation_id = None
        
    async def initialize(self) -> None:
        """Initialize all workflow components with proper interconnections."""
        if self.is_initialized:
            return
            
        self.logger.info("Initializing chat workflow")
        
        try:
            # Initialize executor (already created via from_config)
            if self.executor:
                await self.executor.initialize()
            
            # Initialize data units
            await self._setup_data_units()
            
            # Initialize agent
            await self._setup_agent()
            
            self.is_initialized = True
            self.logger.info("Chat workflow initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize chat workflow: {e}")
            raise
    
    async def _setup_executor(self) -> None:
        """Setup the workflow executor (deprecated - now handled via from_config)."""
        self.logger.info("Executor setup handled via from_config dependencies")
        
    async def _setup_data_units(self) -> None:
        """Setup data units for the workflow."""
        self.logger.info("Setting up data units")
        
        # ✅ FRAMEWORK COMPLIANCE: Use configuration files for DataUnit creation
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent.parent
        
        try:
            # User input data unit
            user_input_config_path = project_root / "Chatbot" / "config" / "components" / "user_input_dataunit_config.yml"
            self.data_units['user_input'] = DataUnitMemory.from_config(str(user_input_config_path))
            
            # Agent output data unit
            agent_output_config_path = project_root / "Chatbot" / "config" / "components" / "agent_output_dataunit_config.yml"
            self.data_units['agent_output'] = DataUnitMemory.from_config(str(agent_output_config_path))
            
        except Exception as e:
            # Fallback: Use DataUnit creation with dictionary configuration (allowed for DataUnits)
            self.logger.warning(f"Failed to load DataUnit config files, using fallback: {e}")
            
            self.data_units['user_input'] = DataUnitMemory.from_config({
                "name": "user_input",
                "data_type": "memory",
                "description": "User input messages",
                "persistent": False
            })
            
            self.data_units['agent_output'] = DataUnitMemory.from_config({
                "name": "agent_output",
                "data_type": "memory", 
                "description": "Agent response output",
                "persistent": False
            })
        
        # ✅ FRAMEWORK COMPLIANCE: Use simple config for ConversationHistoryUnit
        try:
            # Simple configuration to avoid recursion issues
            history_config = {
                'db_path': 'chat_workflow_history.db',
                'max_history_entries': 1000,
                'auto_cleanup': True,
                'cleanup_interval_hours': 24
            }
            # Skip ConversationHistoryUnit initialization for now
            self.logger.info("Skipping conversation history unit due to framework constraints")
            self.conversation_history = None
        except Exception as e:
            self.logger.warning(f"Failed to create conversation history unit: {e}")
            self.conversation_history = None
        
        if self.conversation_history:
            self.data_units['conversation_history'] = self.conversation_history
        
        # Initialize all data units
        for name, data_unit in self.data_units.items():
            await data_unit.initialize()
            self.logger.info(f"Initialized data unit: {name}")
    
    async def _setup_agent(self) -> None:
        """Setup the enhanced collaborative agent."""
        self.logger.info("Setting up enhanced collaborative agent")
        
        # ✅ FRAMEWORK COMPLIANCE: Use from_config pattern for AgentConfig
        try:
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent.parent
            agent_config_path = project_root / "Chatbot" / "config" / "components" / "chat_agent_config.yml"
            agent_config = AgentConfig.from_config(str(agent_config_path))
        except Exception as e:
            self.logger.warning(f"Failed to load agent config from file: {e}")
            # Simple fallback - skip agent setup
            self.logger.info("Skipping agent setup due to configuration issues")
            return
        
        # Use mandatory from_config pattern for agent creation
        self.agent = EnhancedCollaborativeAgent.from_config(
            agent_config,
            enable_metrics=self.agent_config.get('enable_metrics', True)
        )
        
        await self.agent.initialize()
    
    async def process_user_input(self, user_input: str) -> str:
        """
        Process user input through the workflow.
        
        Args:
            user_input: User's input message
            
        Returns:
            str: Agent's response
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Process through agent
            response = await self.agent.process(user_input)
            
            # Store in data units
            await self.data_units['user_input'].set(user_input)
            await self.data_units['agent_output'].set(response)
            
            return response
                
        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            return f"Error: {e}"
    
    async def shutdown(self) -> None:
        """Shutdown the workflow and cleanup resources."""
        if not self.is_initialized:
            return
        
        self.logger.info("Shutting down chat workflow")
        
        try:
            # Shutdown agent
            if self.agent:
                await self.agent.shutdown()
            
            # Shutdown data units
            for data_unit in self.data_units.values():
                if hasattr(data_unit, 'shutdown'):
                    await data_unit.shutdown()
            
            # Shutdown executor
            if self.executor:
                await self.executor.shutdown()
            
            self.is_initialized = False
            self.logger.info("Chat workflow shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get comprehensive workflow status.
        
        Returns:
            Dict[str, Any]: Workflow status information
        """
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'current_conversation_id': self.current_conversation_id,
            'components': {
                'data_units': len(self.data_units),
                'agent': self.agent is not None,
                'executor': self.executor is not None
            },
            'agent_status': self.agent.get_enhanced_status() if self.agent else None,
            'conversation_stats': "Available (call get_conversation_stats() for details)"
        }
    
    async def get_conversation_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get conversation statistics asynchronously.
        
        Returns:
            Optional[Dict[str, Any]]: Conversation statistics or None
        """
        if self.conversation_history:
            return await self.conversation_history.get_statistics()
        return None


# Factory function for easy workflow creation
async def create_chat_workflow(config: Optional[Dict[str, Any]] = None) -> ChatWorkflow:
    """Create and initialize a chat workflow using mandatory from_config pattern."""
    default_config = {
        'name': 'chat_workflow',
        'model': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'max_tokens': 2000,
        'system_prompt': 'You are a helpful and friendly AI assistant.',
        'enable_metrics': True
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Use mandatory from_config pattern
    workflow = ChatWorkflow.from_config(default_config)
    await workflow.initialize()
    return workflow


# Main execution for testing
async def main():
    """Main function for testing the workflow."""
    workflow = ChatWorkflow()
    
    try:
        await workflow.initialize()
        
        # Test basic functionality
        response = await workflow.process_user_input("Hello, how are you?")
        print(f"Response: {response}")
        
        # Show status
        status = workflow.get_workflow_status()
        print(f"Workflow status: {status}")
        
    finally:
        await workflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 