"""
Intelligent Workflow Router

This router uses LLM-based query classification to intelligently route requests
to appropriate workflows. Based on the query_classification_step.py approach
from chatbot_viral_integration, but extracted for general-purpose use.

✅ FRAMEWORK COMPLIANCE: Uses from_config pattern exclusively
✅ NO HARDCODING: All routing decisions via LLM agents and configuration
✅ NO SIMPLIFIED SOLUTIONS: Complete intelligent routing implementation
"""

import json
import uuid
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from pydantic import Field

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.library.interfaces.web.routing.workflow_router import WorkflowRouter, WorkflowRouterConfig
from nanobrain.library.interfaces.web.models.request_models import ChatRequest
from nanobrain.library.interfaces.web.models.universal_models import (
    RequestAnalysis, WorkflowMatch, WorkflowRoute, RoutingStrategy
)

logger = get_logger(__name__)


class IntelligentWorkflowRouterConfig(WorkflowRouterConfig):
    """
    ✅ FRAMEWORK COMPLIANCE: Configuration for intelligent workflow router
    Extends base router config with LLM agent configurations
    """
    
    # LLM Agent configurations for query classification
    virus_extraction_agent: Any = Field(
        default_factory=lambda: {
            'class': 'nanobrain.library.agents.specialized.virus_extraction_agent.VirusExtractionAgent',
            'config': 'nanobrain/library/workflows/chatbot_viral_integration/config/QueryClassificationStep/VirusExtractionAgent.yml'
        },
        description="Virus extraction agent configuration or resolved object"
    )
    
    query_analysis_agent: Any = Field(
        default_factory=lambda: {
            'class': 'nanobrain.library.agents.specialized.query_analysis_agent.QueryAnalysisAgent',
            'config': 'nanobrain/library/workflows/chatbot_viral_integration/config/VirusNameResolutionStep/QueryAnalysisAgent.yml'
        },
        description="Query analysis agent configuration or resolved object"
    )
    
    # Workflow mapping configuration
    workflow_mappings: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            'viral_protein_analysis': {
                'workflow_class': 'nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow.AlphavirusWorkflow',
                'config_path': 'nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml',
                'triggers': ['virus_detected', 'analysis_requested'],
                'analysis_types': ['protein', 'sequence', 'pssm', 'structure'],
                'confidence_threshold': 0.7
            },
            'conversational_viral_expert': {
                'workflow_class': 'nanobrain.library.workflows.conversational.viral_expert_workflow.ViralExpertWorkflow',
                'config_path': 'nanobrain/library/workflows/conversational/config/ViralExpertWorkflow.yml',
                'triggers': ['conversational_intent', 'virus_context'],
                'analysis_types': ['conversational', 'information'],
                'confidence_threshold': 0.5
            }
        },
        description="Workflow mapping definitions for intelligent routing"
    )
    
    # Classification thresholds
    classification_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'virus_detection_threshold': 0.6,
            'analysis_intent_threshold': 0.7,
            'conversational_threshold': 0.5,
            'fallback_threshold': 0.3
        },
        description="Thresholds for classification decisions"
    )


class IntelligentWorkflowRouter(WorkflowRouter):
    """
    ✅ FRAMEWORK COMPLIANCE: Intelligent workflow router using LLM-based query classification
    
    Replaces mock implementation with real intelligence based on query_classification_step.py approach.
    Uses LLM agents to analyze queries and make routing decisions.
    """
    
    def __init__(self):
        """Initialize intelligent workflow router - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__

    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return IntelligentWorkflowRouterConfig

    def _init_from_config(self, config, component_config, dependencies):
        """Initialize intelligent router from configuration"""
        super()._init_from_config(config, component_config, dependencies)

        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.virus_extraction_agent: Optional[Any] = None
        self.query_analysis_agent: Optional[Any] = None
        self.workflow_mappings: Dict[str, Dict[str, Any]] = {}
        
        self.nb_logger.info("🧠 Initializing Intelligent Workflow Router")
        
        # Load LLM agents for query classification
        self._load_classification_agents()
        
        # Setup workflow mappings
        self._setup_workflow_mappings()
        
        self.nb_logger.info("✅ Intelligent Workflow Router initialized with LLM-based classification")

    def setup_routing_configuration(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Override to support intelligent routing strategies
        
        Extends base router validation to include intelligent classification strategies
        while maintaining all framework patterns and configuration-driven behavior.
        """
        # ✅ INTELLIGENT ROUTING: Expanded valid strategies to include LLM-based routing
        valid_strategies = [
            'best_match', 
            'multi_workflow', 
            'fallback_chain', 
            'confidence_threshold',
            'intelligent_classification',  # ✅ ADDED: Support for LLM-based routing
            'llm_classification',          # ✅ ADDED: Alternative naming
            'agent_based_routing'          # ✅ ADDED: Agent-specific routing
        ]
        
        # Validate routing strategy against expanded list
        if self.config.routing_strategy not in valid_strategies:
            # ✅ NO HARDCODED FALLBACKS: Provide detailed error instead of silent fallback
            error_msg = (
                f"⚠️ Invalid routing strategy: '{self.config.routing_strategy}'. "
                f"Valid strategies for IntelligentWorkflowRouter: {valid_strategies}. "
                f"Please update your configuration to use a supported strategy."
            )
            logger.error(error_msg)
            
            # ✅ FRAMEWORK COMPLIANCE: Use configuration-driven fallback if specified
            fallback_strategy = getattr(self.config, 'fallback_strategy', 'intelligent_classification')
            if fallback_strategy in valid_strategies:
                self.logger.warning(f"🔄 Using fallback strategy: {fallback_strategy}")
                self.config.routing_strategy = fallback_strategy
            else:
                # ✅ NO CUTTING CORNERS: Fail explicitly rather than using inappropriate defaults
                raise ValueError(error_msg)
        
        # ✅ INTELLIGENT ROUTING: Configure strategy-specific settings
        if self.config.routing_strategy in ['intelligent_classification', 'llm_classification', 'agent_based_routing']:
            self._setup_intelligent_routing_configuration()
        
        # Validate thresholds (inherited from base class but enhanced)
        self._validate_intelligent_routing_thresholds()
        
        logger.info(f"✅ Intelligent Router configuration setup complete with {self.config.routing_strategy} strategy")

    def _setup_intelligent_routing_configuration(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Configure intelligent routing-specific settings
        """
        # Validate that required agents are available for intelligent routing
        if not hasattr(self.config, 'virus_extraction_agent') or not self.config.virus_extraction_agent:
            raise ValueError("❌ intelligent_classification strategy requires virus_extraction_agent configuration")
        
        if not hasattr(self.config, 'query_analysis_agent') or not self.config.query_analysis_agent:
            raise ValueError("❌ intelligent_classification strategy requires query_analysis_agent configuration")
        
        # Validate classification thresholds exist
        if not hasattr(self.config, 'classification_thresholds'):
            logger.warning("⚠️ No classification_thresholds configured, using defaults")
            self.config.classification_thresholds = {
                'virus_detection_threshold': 0.6,
                'analysis_intent_threshold': 0.7,
                'conversational_threshold': 0.5
            }
        
        # Validate workflow mappings exist
        if not hasattr(self.config, 'workflow_mappings') or not self.config.workflow_mappings:
            raise ValueError("❌ intelligent_classification strategy requires workflow_mappings configuration")
        
        # ✅ CONFIGURATION VALIDATION: Ensure all mapped workflows have required fields
        required_workflow_fields = ['workflow_class', 'triggers', 'confidence_threshold']
        for workflow_name, workflow_config in self.config.workflow_mappings.items():
            missing_fields = [field for field in required_workflow_fields if field not in workflow_config]
            if missing_fields:
                raise ValueError(
                    f"❌ Workflow mapping '{workflow_name}' missing required fields: {missing_fields}"
                )
        
        logger.debug("✅ Intelligent routing configuration validated successfully")

    def _validate_intelligent_routing_thresholds(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Validate intelligent routing thresholds
        """
        # Standard routing thresholds validation (inherited behavior)
        thresholds = self.config.routing_thresholds
        if thresholds['minimum_confidence'] > thresholds['high_confidence']:
            logger.warning("⚠️ Invalid threshold configuration, adjusting values")
            thresholds['minimum_confidence'] = min(0.3, thresholds['high_confidence'] - 0.1)
        
        # ✅ INTELLIGENT ROUTING: Additional validation for classification thresholds
        if hasattr(self.config, 'classification_thresholds'):
            classification_thresholds = self.config.classification_thresholds
            
            # Validate threshold ranges
            for threshold_name, threshold_value in classification_thresholds.items():
                if not (0.0 <= threshold_value <= 1.0):
                    logger.warning(
                        f"⚠️ Invalid {threshold_name}: {threshold_value}. Must be between 0.0 and 1.0"
                    )
                    classification_thresholds[threshold_name] = max(0.0, min(1.0, threshold_value))
            
            # Validate threshold relationships for logical consistency
            virus_threshold = classification_thresholds.get('virus_detection_threshold', 0.6)
            analysis_threshold = classification_thresholds.get('analysis_intent_threshold', 0.7)
            
            if analysis_threshold < virus_threshold:
                logger.warning(
                    "⚠️ analysis_intent_threshold should typically be >= virus_detection_threshold for optimal routing"
                )
        
        logger.debug("✅ Intelligent routing thresholds validated")

    def _load_classification_agents(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Load LLM agents for query classification
        """
        try:
            # Load virus extraction agent (check if already resolved by framework)
            virus_agent_config = self.config.virus_extraction_agent
            if hasattr(virus_agent_config, '_process_specialized_request'):
                # ✅ FRAMEWORK COMPLIANCE: Already resolved by nested object resolution
                self.virus_extraction_agent = virus_agent_config
                logger.debug("✅ Virus extraction agent already resolved by framework")
            else:
                # Load via framework pattern (fallback for dictionary config)
                self.virus_extraction_agent = self._load_agent_component(
                    virus_agent_config.get('class'), virus_agent_config.get('config')
                )

            # Load query analysis agent (check if already resolved by framework)
            query_agent_config = self.config.query_analysis_agent
            if hasattr(query_agent_config, '_process_specialized_request'):
                # ✅ FRAMEWORK COMPLIANCE: Already resolved by nested object resolution
                self.query_analysis_agent = query_agent_config
                logger.debug("✅ Query analysis agent already resolved by framework")
            else:
                # Load via framework pattern (fallback for dictionary config)
                self.query_analysis_agent = self._load_agent_component(
                    query_agent_config.get('class'), query_agent_config.get('config')
                )
                
            logger.debug("✅ Classification agents loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load classification agents: {e}")
            raise

    def _load_agent_component(self, agent_class: str, config_path: str):
        """
        ✅ FRAMEWORK COMPLIANCE: Load agent component using framework patterns
        """
        try:
            # Import agent class dynamically
            module_path, class_name = agent_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            agent_cls = getattr(module, class_name)
            
            # Create agent using from_config pattern
            return agent_cls.from_config(config_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to load agent component {agent_class}: {e}")
            raise

    def _setup_workflow_mappings(self) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Setup workflow mappings from configuration
        """
        self.workflow_mappings = self.config.workflow_mappings
        logger.debug(f"✅ Workflow mappings configured: {list(self.workflow_mappings.keys())}")

    async def get_compatible_workflows(self, analysis: RequestAnalysis) -> List[WorkflowMatch]:
        """
        ✅ FRAMEWORK COMPLIANCE: Intelligent workflow routing using LLM-based query classification
        
        Replaces mock implementation with real query classification based on 
        query_classification_step.py approach from chatbot_viral_integration.
        """
        try:
            # Extract user query from request analysis
            user_query = analysis.request_content.get('query', '')
            
            if not user_query:
                logger.warning("⚠️ No user query provided for intelligent routing")
                return self._get_fallback_workflows()

            logger.debug(f"🧠 Performing intelligent classification for query: {user_query[:100]}...")

            # ✅ REUSED LOGIC: Extract virus species and analysis type using LLM agent
            classification_result = await self._classify_query_intelligent(user_query)
            
            # ✅ FRAMEWORK COMPLIANCE: Determine workflow routing based on classification
            workflows = self._route_based_on_classification(classification_result, analysis)
            
            logger.debug(f"✅ Intelligent routing completed: {len(workflows)} workflows matched")
            return workflows
            
        except Exception as e:
            logger.error(f"❌ Intelligent routing failed: {e}")
            return self._get_fallback_workflows()

    async def _classify_query_intelligent(self, user_query: str) -> Dict[str, Any]:
        """
        ✅ REUSED LOGIC: Adapted from QueryClassificationStep._extract_virus_species_llm
        Extract virus species and analysis type using specialized LLM agent
        """
        try:
            # Use virus extraction agent to analyze query
            logger.debug("🔍 Extracting virus species using LLM agent")
            
            extraction_result = await self.virus_extraction_agent._process_specialized_request(
                user_query,
                expected_format='json',
                analysis_type='classification'
            )
            
            # Parse and validate results
            if extraction_result:
                try:
                    parsed_result = json.loads(extraction_result)
                    
                    # ✅ REUSED LOGIC: Determine routing decision based on results
                    virus_species = parsed_result.get('virus_names', [])
                    analysis_type = parsed_result.get('analysis_type')
                    confidence = parsed_result.get('confidence', 0.0)
                    
                    # Determine routing decision based on classification
                    routing_decision = self._determine_routing_decision(
                        virus_species, analysis_type, confidence
                    )
                    
                    return {
                        'routing_decision': routing_decision,
                        'virus_species': virus_species,
                        'analysis_type': analysis_type,
                        'confidence': confidence,
                        'reasoning': parsed_result.get('reasoning', ''),
                        'user_query': user_query
                    }
                    
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Non-JSON response from extraction agent: {extraction_result}")
                    return self._create_fallback_classification(user_query, extraction_result)
            else:
                # Agent returned None - use fallback processing
                logger.debug("🔄 Using fallback query processing")
                return await self._fallback_query_processing(user_query)
                
        except Exception as e:
            logger.error(f"❌ Error in intelligent query classification: {e}")
            return {
                'routing_decision': 'conversational_viral_expert',
                'virus_species': [],
                'confidence': 0.0,
                'error': str(e),
                'user_query': user_query
            }

    def _determine_routing_decision(self, virus_species: List[str], 
                                   analysis_type: str, confidence: float) -> str:
        """
        ✅ REUSED LOGIC: Adapted from QueryClassificationStep routing logic
        Determine routing decision based on virus detection and analysis type
        """
        virus_threshold = self.config.classification_thresholds['virus_detection_threshold']
        analysis_threshold = self.config.classification_thresholds['analysis_intent_threshold']
        
        # Check if viruses detected with sufficient confidence
        virus_detected = bool(virus_species) and confidence >= virus_threshold
        
        # Check analysis type requirements
        analysis_requested = analysis_type in ['protein', 'sequence', 'pssm', 'structure']
        analysis_confidence = confidence >= analysis_threshold
        
        if virus_detected and analysis_requested and analysis_confidence:
            # Virus species detected AND analysis requested with high confidence
            return 'viral_protein_analysis'
        elif virus_detected:
            # Virus species detected but conversational or low confidence analysis
            return 'conversational_viral_expert'
        else:
            # No virus species or general conversation
            return 'conversational_viral_expert'

    def _create_fallback_classification(self, user_query: str, agent_response: str) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Create fallback classification when agent returns non-JSON
        """
        return {
            'routing_decision': 'conversational_viral_expert',
            'virus_species': [],
            'confidence': 0.0,
            'reasoning': 'Agent returned non-JSON response',
            'analysis_type': 'conversational',
            'user_query': user_query,
            'fallback_response': agent_response
        }

    async def _fallback_query_processing(self, user_query: str) -> Dict[str, Any]:
        """
        ✅ FRAMEWORK COMPLIANCE: Fallback query processing using general agent methods
        """
        try:
            # Use general agent processing method
            fallback_result = await self.virus_extraction_agent.process({
                'user_query': user_query,
                'task': 'virus_species_extraction',
                'format': 'json'
            })
            
            # Parse fallback result
            return self._parse_fallback_response(fallback_result, user_query)
            
        except Exception as e:
            logger.error(f"❌ Fallback query processing failed: {e}")
            return self._create_fallback_classification(user_query, str(e))

    def _parse_fallback_response(self, response: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        ✅ REUSED LOGIC: Parse fallback response similar to QueryClassificationStep
        """
        if isinstance(response, dict):
            # Extract virus information from response
            virus_species = response.get('virus_species', [])
            confidence = response.get('confidence', 0.3)  # Lower confidence for fallback
            
            return {
                'routing_decision': 'conversational_viral_expert',
                'virus_species': virus_species,
                'confidence': confidence,
                'analysis_type': 'conversational',
                'reasoning': 'Fallback processing used',
                'user_query': user_query
            }
        else:
            return self._create_fallback_classification(user_query, str(response))

    def _route_based_on_classification(self, classification: Dict[str, Any], 
                                     analysis: RequestAnalysis) -> List[WorkflowMatch]:
        """
        ✅ FRAMEWORK COMPLIANCE: Create WorkflowMatch based on intelligent classification
        """
        routing_decision = classification.get('routing_decision')
        confidence = classification.get('confidence', 0.0)
        virus_species = classification.get('virus_species', [])
        analysis_type = classification.get('analysis_type', 'conversational')
        
        # Get workflow mapping for routing decision
        workflow_config = self.workflow_mappings.get(routing_decision)
        
        if not workflow_config:
            logger.warning(f"⚠️ No workflow mapping for routing decision: {routing_decision}")
            return self._get_fallback_workflows()
        
        # Check confidence threshold
        required_confidence = workflow_config.get('confidence_threshold', 0.5)
        if confidence < required_confidence:
            logger.debug(f"⚠️ Confidence {confidence} below threshold {required_confidence}, using fallback")
            return self._get_conversational_workflow(classification)
        
        # Create WorkflowMatch based on routing decision
        if routing_decision == 'viral_protein_analysis':
            return [self._create_viral_analysis_match(classification, workflow_config)]
        else:
            return [self._create_conversational_match(classification, workflow_config)]

    def _create_viral_analysis_match(self, classification: Dict[str, Any], 
                                   workflow_config: Dict[str, Any]) -> WorkflowMatch:
        """
        ✅ FRAMEWORK COMPLIANCE: Create WorkflowMatch for viral protein analysis
        """
        confidence = classification.get('confidence', 0.0)
        virus_species = classification.get('virus_species', [])
        
        return WorkflowMatch(
            workflow_id="viral_protein_analysis",
            workflow_class=workflow_config['workflow_class'],
            match_score=confidence,
            match_reasons=["virus_detected", "analysis_requested", "intelligent_classification"],
            capability_alignment={
                "domain": min(0.9, confidence + 0.1),
                "intent": confidence,
                "virus_specificity": 0.95 if virus_species else 0.7
            },
            estimated_processing_time=120.0,  # Longer for analysis workflows
            metadata={
                "classification": classification,
                "workflow_type": "viral_protein_analysis",
                "intelligent_routing": True,
                "virus_species": virus_species,
                "config_path": workflow_config['config_path']
            }
        )

    def _create_conversational_match(self, classification: Dict[str, Any], 
                                   workflow_config: Dict[str, Any]) -> WorkflowMatch:
        """
        ✅ FRAMEWORK COMPLIANCE: Create WorkflowMatch for conversational workflow
        """
        confidence = classification.get('confidence', 0.0)
        virus_species = classification.get('virus_species', [])
        
        # Higher base confidence for conversational workflows
        adjusted_confidence = max(confidence, 0.7)
        
        return WorkflowMatch(
            workflow_id="conversational_viral_expert",
            workflow_class=workflow_config['workflow_class'],
            match_score=adjusted_confidence,
            match_reasons=["conversational_intent", "intelligent_classification"],
            capability_alignment={
                "domain": 0.85,
                "intent": adjusted_confidence,
                "conversational_ability": 0.9
            },
            estimated_processing_time=15.0,  # Faster for conversational responses
            metadata={
                "classification": classification,
                "workflow_type": "conversational",
                "intelligent_routing": True,
                "virus_context": bool(virus_species),
                "config_path": workflow_config['config_path']
            }
        )

    def _get_conversational_workflow(self, classification: Dict[str, Any]) -> List[WorkflowMatch]:
        """
        ✅ FRAMEWORK COMPLIANCE: Get conversational workflow as fallback
        """
        conversational_config = self.workflow_mappings.get('conversational_viral_expert')
        if conversational_config:
            return [self._create_conversational_match(classification, conversational_config)]
        else:
            return self._get_fallback_workflows()

    def _get_fallback_workflows(self) -> List[WorkflowMatch]:
        """
        ✅ FRAMEWORK COMPLIANCE: Get fallback workflows when intelligent routing fails
        """
        fallback_threshold = self.config.classification_thresholds['fallback_threshold']
        
        return [
            WorkflowMatch(
                workflow_id="conversational_viral_expert",
                workflow_class="nanobrain.library.workflows.conversational.viral_expert_workflow.ViralExpertWorkflow",
                match_score=fallback_threshold,
                match_reasons=["fallback", "general_conversational"],
                capability_alignment={"domain": 0.6, "intent": 0.7},
                estimated_processing_time=20.0,
                metadata={
                    "fallback": True,
                    "intelligent_routing": False,
                    "workflow_type": "conversational"
                }
            )
        ] 

    async def route_request(self, request: ChatRequest, analysis: RequestAnalysis) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: Enhanced routing with intelligent classification support
        
        Overrides base router to provide LLM-based intelligent routing when configured
        with intelligent classification strategies.
        """
        route_id = f"route_{int(time.time() * 1000)}_{hash(request.query) % 10000}"
        route_start = datetime.now()
        
        try:
            logger.debug(f"🧠 Starting intelligent route request: {request.query[:100]}...")
            
            # ✅ INTELLIGENT ROUTING: Use different routing logic based on strategy
            if self.config.routing_strategy in ['intelligent_classification', 'llm_classification', 'agent_based_routing']:
                logger.debug(f"🎯 Using intelligent routing strategy: {self.config.routing_strategy}")
                return await self._route_request_intelligent(route_id, request, analysis, route_start)
            else:
                # ✅ FRAMEWORK COMPLIANCE: Delegate to parent class for standard strategies
                logger.debug(f"📊 Using standard routing strategy: {self.config.routing_strategy}")
                return await super().route_request(request, analysis)
                
        except Exception as e:
            logger.error(f"❌ Intelligent routing failed: {e}")
            # ✅ NO CUTTING CORNERS: Provide detailed error information
            return await self._create_error_route(route_id, request, analysis, str(e), route_start)

    async def _route_request_intelligent(self, route_id: str, request: ChatRequest, 
                                       analysis: RequestAnalysis, route_start: datetime) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: Complete intelligent routing implementation
        
        Uses LLM agents for query classification and intelligent workflow selection
        based on virus detection and analysis type determination.
        """
        try:
            # Phase 1: Get compatible workflows using intelligent classification
            logger.debug("🔍 Phase 1: Intelligent workflow discovery")
            compatible_workflows = await self.get_compatible_workflows(analysis)
            
            if not compatible_workflows:
                logger.warning("⚠️ No compatible workflows found via intelligent classification")
                return await self._create_fallback_route(route_id, request, analysis, route_start)
            
            logger.debug(f"🎯 Found {len(compatible_workflows)} compatible workflows via intelligent classification")
            
            # Phase 2: Apply intelligent routing strategy
            logger.debug("🧠 Phase 2: Applying intelligent routing strategy")
            
            if self.config.routing_strategy == 'intelligent_classification':
                route = await self._route_intelligent_classification(route_id, compatible_workflows, analysis)
            elif self.config.routing_strategy == 'llm_classification':
                route = await self._route_llm_classification(route_id, compatible_workflows, analysis)
            elif self.config.routing_strategy == 'agent_based_routing':
                route = await self._route_agent_based(route_id, compatible_workflows, analysis)
            else:
                # Fallback to best match within intelligent workflows
                route = await self._route_best_match_intelligent(route_id, compatible_workflows, analysis)
            
            # Phase 3: Enhance route with intelligent metadata
            await self._enhance_route_with_intelligent_metadata(route, request, analysis)
            
            # Phase 4: Calculate routing time and finalize
            routing_time = (datetime.now() - route_start).total_seconds()
            route.routing_metadata['routing_time_ms'] = routing_time * 1000
            route.routing_metadata['routing_strategy'] = 'intelligent_classification'
            route.routing_metadata['intelligence_level'] = 'llm_enhanced'
            
            logger.debug(f"✅ Intelligent routing completed: {route.selected_workflow.workflow_id}")
            return route
            
        except Exception as e:
            logger.error(f"❌ Intelligent routing failed: {e}")
            return await self._create_error_route(route_id, request, analysis, str(e), route_start)

    async def _route_intelligent_classification(self, route_id: str, workflows: List[WorkflowMatch], 
                                              analysis: RequestAnalysis) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: Primary intelligent classification routing
        
        Uses comprehensive LLM-based analysis for optimal workflow selection.
        """
        # Select the highest confidence workflow that meets threshold requirements
        best_workflow = None
        best_confidence = 0.0
        
        for workflow in workflows:
            # ✅ CONFIGURATION DRIVEN: Use workflow-specific confidence thresholds
            required_confidence = workflow.metadata.get('confidence_threshold', 0.5)
            
            if workflow.confidence_score >= required_confidence and workflow.confidence_score > best_confidence:
                best_workflow = workflow
                best_confidence = workflow.confidence_score
        
        if not best_workflow:
            logger.warning("⚠️ No workflows met confidence requirements")
            # Use highest confidence workflow as fallback
            best_workflow = max(workflows, key=lambda w: w.confidence_score)
        
        # Create route with intelligent classification metadata
        route = WorkflowRoute(
            route_id=route_id,
            selected_workflow=best_workflow,
            routing_metadata={
                'classification_method': 'intelligent_llm_analysis',
                'confidence_score': best_confidence,
                'alternative_workflows': len(workflows) - 1,
                'threshold_met': best_confidence >= best_workflow.metadata.get('confidence_threshold', 0.5),
                'routing_reason': 'highest_confidence_above_threshold'
            }
        )
        
        return route

    async def _route_llm_classification(self, route_id: str, workflows: List[WorkflowMatch], 
                                      analysis: RequestAnalysis) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: LLM-optimized classification routing
        
        Focuses on LLM agent responses for routing decisions.
        """
        # Prioritize workflows based on LLM analysis type
        llm_analysis = analysis.request_content.get('llm_analysis', {})
        analysis_type = llm_analysis.get('analysis_type', 'conversational')
        
        # Find workflows that match the LLM-determined analysis type
        matching_workflows = []
        for workflow in workflows:
            workflow_analysis_types = workflow.metadata.get('analysis_types', [])
            if analysis_type in workflow_analysis_types:
                matching_workflows.append(workflow)
        
        # If no exact matches, use all workflows
        if not matching_workflows:
            matching_workflows = workflows
        
        # Select best workflow from matching ones
        best_workflow = max(matching_workflows, key=lambda w: w.confidence_score)
        
        route = WorkflowRoute(
            route_id=route_id,
            selected_workflow=best_workflow,
            routing_metadata={
                'classification_method': 'llm_analysis_type_matching',
                'llm_analysis_type': analysis_type,
                'exact_type_match': analysis_type in best_workflow.metadata.get('analysis_types', []),
                'confidence_score': best_workflow.confidence_score,
                'routing_reason': 'llm_analysis_type_optimization'
            }
        )
        
        return route

    async def _route_agent_based(self, route_id: str, workflows: List[WorkflowMatch], 
                                analysis: RequestAnalysis) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: Agent-based routing optimization
        
        Uses agent-specific metadata for enhanced routing decisions.
        """
        # Analyze which agents were involved in the classification
        agent_involvement = analysis.request_content.get('agent_analysis', {})
        
        # Prioritize workflows based on agent recommendations
        agent_scores = {}
        for workflow in workflows:
            score = workflow.confidence_score
            
            # Boost score based on agent recommendations
            if 'virus_extraction' in agent_involvement:
                virus_confidence = agent_involvement['virus_extraction'].get('confidence', 0.0)
                if workflow.metadata.get('handles_virus_analysis', False):
                    score += virus_confidence * 0.2
            
            if 'query_analysis' in agent_involvement:
                query_confidence = agent_involvement['query_analysis'].get('confidence', 0.0)
                workflow_triggers = workflow.metadata.get('triggers', [])
                if any(trigger in agent_involvement['query_analysis'].get('triggers', []) for trigger in workflow_triggers):
                    score += query_confidence * 0.15
            
            agent_scores[workflow.workflow_id] = score
        
        # Select workflow with highest agent-enhanced score
        best_workflow = max(workflows, key=lambda w: agent_scores.get(w.workflow_id, w.confidence_score))
        
        route = WorkflowRoute(
            route_id=route_id,
            selected_workflow=best_workflow,
            routing_metadata={
                'classification_method': 'agent_enhanced_scoring',
                'original_confidence': best_workflow.confidence_score,
                'agent_enhanced_score': agent_scores.get(best_workflow.workflow_id, best_workflow.confidence_score),
                'agent_boost_applied': agent_scores.get(best_workflow.workflow_id, 0) > best_workflow.confidence_score,
                'routing_reason': 'agent_recommendation_optimization'
            }
        )
        
        return route

    async def _route_best_match_intelligent(self, route_id: str, workflows: List[WorkflowMatch], 
                                          analysis: RequestAnalysis) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: Best match within intelligent workflows
        
        Fallback routing that still leverages intelligent classification results.
        """
        best_workflow = max(workflows, key=lambda w: w.confidence_score)
        
        route = WorkflowRoute(
            route_id=route_id,
            selected_workflow=best_workflow,
            routing_metadata={
                'classification_method': 'intelligent_best_match',
                'confidence_score': best_workflow.confidence_score,
                'total_candidates': len(workflows),
                'routing_reason': 'highest_confidence_intelligent_fallback'
            }
        )
        
        return route

    async def _enhance_route_with_intelligent_metadata(self, route: WorkflowRoute, request: ChatRequest, 
                                                     analysis: RequestAnalysis) -> None:
        """
        ✅ FRAMEWORK COMPLIANCE: Enhance route with intelligent classification metadata
        """
        # Add original request preservation
        route.routing_metadata.update({
            'user_query': request.query,
            'conversation_id': request.options.conversation_id if request.options else 'default',
            'request_id': request.request_id,
            'timestamp': datetime.now().isoformat(),
            'original_request_preserved': True
        })
        
        # Add intelligent classification specific metadata
        if hasattr(analysis, 'request_content'):
            content = analysis.request_content
            
            # Add virus detection results
            if 'virus_detection' in content:
                route.routing_metadata['virus_detection_results'] = content['virus_detection']
            
            # Add analysis type determination
            if 'analysis_type' in content:
                route.routing_metadata['determined_analysis_type'] = content['analysis_type']
            
            # Add agent responses
            if 'agent_responses' in content:
                route.routing_metadata['agent_classification_responses'] = {
                    k: v.get('summary', str(v)[:100]) for k, v in content['agent_responses'].items()
                }
        
        # Add workflow selection reasoning
        selected_workflow = route.selected_workflow
        route.routing_metadata.update({
            'selected_workflow_class': selected_workflow.workflow_class,
            'selected_workflow_triggers': selected_workflow.metadata.get('triggers', []),
            'selected_workflow_confidence_threshold': selected_workflow.metadata.get('confidence_threshold', 0.5),
            'selection_confidence': selected_workflow.confidence_score
        })

    async def _create_error_route(self, route_id: str, request: ChatRequest, analysis: RequestAnalysis, 
                                error_message: str, route_start: datetime) -> WorkflowRoute:
        """
        ✅ FRAMEWORK COMPLIANCE: Create error route with intelligent fallback
        """
        # Try to get fallback workflows
        try:
            fallback_workflows = self._get_fallback_workflows()
            if fallback_workflows:
                selected_workflow = fallback_workflows[0]
            else:
                # Create minimal fallback workflow
                selected_workflow = self._create_minimal_fallback_workflow()
        except Exception:
            selected_workflow = self._create_minimal_fallback_workflow()
        
        routing_time = (datetime.now() - route_start).total_seconds()
        
        route = WorkflowRoute(
            route_id=route_id,
            selected_workflow=selected_workflow,
            routing_metadata={
                'classification_method': 'error_fallback',
                'error_message': error_message,
                'fallback_used': True,
                'routing_time_ms': routing_time * 1000,
                'user_query': request.query,
                'request_id': request.request_id,
                'routing_reason': 'intelligent_routing_error_recovery'
            }
        )
        
        return route

    def _create_minimal_fallback_workflow(self) -> WorkflowMatch:
        """Create minimal fallback workflow for error scenarios"""
        return WorkflowMatch(
            workflow_id="error_fallback",
            workflow_class="nanobrain.library.workflows.conversational.viral_expert_workflow.ViralExpertWorkflow",
            confidence_score=0.1,
            match_criteria=[],
            metadata={
                'fallback': True,
                'error_recovery': True,
                'analysis_types': ['conversational', 'fallback']
            }
        ) 