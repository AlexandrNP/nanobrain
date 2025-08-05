#!/usr/bin/env python3
"""
Workflow Router for NanoBrain Framework
Intelligent routing system for directing requests to appropriate workflows with multiple strategies.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.request_models import ChatRequest
from nanobrain.library.interfaces.web.models.universal_models import (
    RequestAnalysis, WorkflowMatch, WorkflowRoute, MultiWorkflowRoute,
    UniversalResponse, RoutingStrategy
)
from nanobrain.library.interfaces.web.models.workflow_models import WorkflowExecutionPlan
from pydantic import Field

# Router logger
logger = logging.getLogger(__name__)


class WorkflowRouterConfig(ConfigBase):
    """Configuration for workflow router"""
    
    # Routing strategy
    routing_strategy: str = Field(
        default="best_match",
        description="Routing strategy to use"
    )
    
    # Routing strategy configuration
    routing_strategy_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'class': 'nanobrain.library.interfaces.web.routing.routing_strategies.BestMatchStrategy',
            'config': {}
        },
        description="Routing strategy implementation configuration"
    )
    
    # Routing thresholds
    routing_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'minimum_confidence': 0.2,  # Lower threshold for development
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.3
        },
        description="Routing confidence thresholds"
    )
    
    # Workflow registry integration
    workflow_registry: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Workflow registry configuration"
    )
    
    # Execution configuration
    execution_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,
            'parallel_routing': False
        },
        description="Execution settings"
    )
    
    # Fallback configuration
    fallback_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_fallback': True,
            'fallback_strategy': 'default_workflow',
            'fallback_workflow': None
        },
        description="Fallback routing configuration"
    )
    
    # Performance settings
    performance_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_parallel_routing': True,
            'routing_timeout': 30.0,
            'max_concurrent_routes': 5
        },
        description="Performance optimization settings"
    )


class WorkflowRouter(FromConfigBase):
    """
    Configurable workflow routing with multiple strategies.
    Routes requests to appropriate workflows based on analysis.
    """
    
    def __init__(self):
        """Initialize workflow router - use from_config for creation"""
        super().__init__()
        # Instance variables moved to _init_from_config since framework uses __new__ and bypasses __init__
        
    @classmethod
    def _get_config_class(cls):
        """Return the configuration class for this component"""
        return WorkflowRouterConfig
    
    def _init_from_config(self, config, component_config, dependencies):
        """Initialize router from configuration"""
        super()._init_from_config(config, component_config, dependencies)
        
        # ✅ FRAMEWORK COMPLIANCE: Initialize instance variables here since __init__ is bypassed
        self.config: Optional[WorkflowRouterConfig] = None
        self.route_cache: Dict[str, WorkflowRoute] = {}
        self.execution_cache: Dict[str, UniversalResponse] = {}
        self.workflow_instances: Dict[str, Any] = {}
        
        logger.info("🎯 Initializing Workflow Router")
        self.config = config
        
        # Setup router configuration
        self.setup_routing_configuration()
        
        logger.info(f"✅ Workflow Router initialized with {self.config.routing_strategy} strategy")
    
    def setup_routing_configuration(self) -> None:
        """Setup routing configuration and validation"""
        # Validate routing strategy
        valid_strategies = ['best_match', 'multi_workflow', 'fallback_chain', 'confidence_threshold']
        if self.config.routing_strategy not in valid_strategies:
            logger.warning(f"⚠️ Unknown routing strategy: {self.config.routing_strategy}, using 'best_match'")
            self.config.routing_strategy = 'best_match'
        
        # Validate thresholds
        thresholds = self.config.routing_thresholds
        if thresholds['minimum_confidence'] > thresholds['high_confidence']:
            logger.warning("⚠️ Invalid threshold configuration, adjusting values")
            thresholds['minimum_confidence'] = min(0.3, thresholds['high_confidence'] - 0.1)
        
        logger.debug("✅ Router configuration setup complete")
    
    async def route_request(self, request: ChatRequest, analysis: RequestAnalysis) -> WorkflowRoute:
        """
        Route request using configured strategy.
        
        Args:
            request: Original chat request
            analysis: Request analysis result
            
        Returns:
            WorkflowRoute with selected workflow and routing metadata
        """
        try:
            route_start = datetime.now()
            route_id = f"route_{uuid.uuid4().hex[:8]}"
            
            logger.debug(f"🎯 Routing request {request.request_id} with strategy: {self.config.routing_strategy}")
            
            # Check route cache if enabled
            if self.config.performance_config.get('enable_parallel_routing', True):
                cached_route = self.get_cached_route(request.query)
                if cached_route:
                    logger.debug(f"✅ Using cached route for request: {request.request_id}")
                    return cached_route
            
            # Get compatible workflows (this would need to be injected or accessed)
            # For now, we'll simulate this - in real implementation, this would come from WorkflowRegistry
            compatible_workflows = await self.get_compatible_workflows(analysis)
            
            if not compatible_workflows:
                # No compatible workflows found, use fallback
                return await self.create_fallback_route(route_id, request, analysis)
            
            # Apply routing strategy
            if self.config.routing_strategy == 'best_match':
                route = await self.route_best_match(route_id, compatible_workflows, analysis)
            elif self.config.routing_strategy == 'multi_workflow':
                route = await self.route_multi_workflow(route_id, compatible_workflows, analysis)
            elif self.config.routing_strategy == 'fallback_chain':
                route = await self.route_fallback_chain(route_id, compatible_workflows, analysis)
            elif self.config.routing_strategy == 'confidence_threshold':
                route = await self.route_confidence_threshold(route_id, compatible_workflows, analysis)
            else:
                # Default to best match
                route = await self.route_best_match(route_id, compatible_workflows, analysis)
            
            # ✅ FRAMEWORK COMPLIANCE: Add original request data to routing metadata
            route.routing_metadata.update({
                'user_query': request.query,
                'conversation_id': request.options.conversation_id if request.options else 'default',
                'request_id': request.request_id,
                'timestamp': datetime.now().isoformat(),
                'original_request_preserved': True
            })
            
            # Cache route if enabled
            if self.config.performance_config.get('enable_parallel_routing', True):
                self.cache_route(request.query, route)
            
            routing_time = (datetime.now() - route_start).total_seconds()
            route.routing_metadata['routing_time_ms'] = routing_time * 1000
            
            logger.debug(f"✅ Request routed to: {route.selected_workflow.workflow_id}")
            return route
            
        except Exception as e:
            logger.error(f"❌ Request routing failed: {e}")
            return await self.create_error_route(f"route_error_{uuid.uuid4().hex[:8]}", request, analysis, str(e))
    
    async def get_compatible_workflows(self, analysis: RequestAnalysis) -> List[WorkflowMatch]:
        """Get compatible workflows - this would be injected from WorkflowRegistry in real implementation"""
        # This is a placeholder - in real implementation, this would be provided by dependency injection
        # or the WorkflowRegistry would be passed to the router
        logger.debug("📋 Getting compatible workflows (placeholder implementation)")
        
        # Create mock compatible workflows for demonstration
        # In real implementation, this would come from the WorkflowRegistry
        mock_workflows = [
            WorkflowMatch(
                workflow_id="chatbot_viral_integration",
                workflow_class="nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow.ChatbotViralWorkflow",
                match_score=0.85,
                match_reasons=["domain_match", "intent_match"],
                capability_alignment={"domain": 0.9, "intent": 0.8},
                estimated_processing_time=30.0,
                metadata={"mock": True}
            ),
            WorkflowMatch(
                workflow_id="viral_protein_analysis",
                workflow_class="nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow.AlphavirusWorkflow",
                match_score=0.75,
                match_reasons=["domain_match"],
                capability_alignment={"domain": 0.8, "intent": 0.7},
                estimated_processing_time=60.0,
                metadata={"mock": True}
            )
        ]
        
        # Filter based on analysis (simple mock filtering)
        if analysis.domain_classification.domain_type.value == 'virology':
            return mock_workflows
        else:
            return mock_workflows[:1]  # Return only the first one for other domains
    
    async def route_best_match(self, route_id: str, workflows: List[WorkflowMatch], 
                             analysis: RequestAnalysis) -> WorkflowRoute:
        """Route using best match strategy"""
        try:
            # Sort workflows by match score
            sorted_workflows = sorted(workflows, key=lambda w: w.match_score, reverse=True)
            best_workflow = sorted_workflows[0]
            
            # Check if best match meets minimum confidence
            if best_workflow.match_score < self.config.routing_thresholds['minimum_confidence']:
                logger.warning(f"⚠️ Best match score {best_workflow.match_score} below minimum threshold")
                return await self.create_fallback_route(route_id, None, analysis)
            
            # Create route
            route = WorkflowRoute(
                route_id=route_id,
                selected_workflow=best_workflow,
                routing_strategy=RoutingStrategy.BEST_MATCH,
                route_confidence=best_workflow.match_score,
                fallback_workflows=sorted_workflows[1:3] if len(sorted_workflows) > 1 else [],
                routing_metadata={
                    'strategy': 'best_match',
                    'total_candidates': len(workflows),
                    'selection_criteria': 'highest_match_score',
                    'confidence_threshold_met': best_workflow.match_score >= self.config.routing_thresholds['minimum_confidence']
                }
            )
            
            logger.debug(f"✅ Best match route created: {best_workflow.workflow_id}")
            return route
            
        except Exception as e:
            logger.error(f"❌ Best match routing failed: {e}")
            raise
    
    async def route_multi_workflow(self, route_id: str, workflows: List[WorkflowMatch], 
                                 analysis: RequestAnalysis) -> WorkflowRoute:
        """Route using multi-workflow strategy"""
        try:
            if not self.config.multi_workflow_support:
                logger.warning("⚠️ Multi-workflow routing requested but not enabled")
                return await self.route_best_match(route_id, workflows, analysis)
            
            # Check if multiple workflows have high scores
            high_score_workflows = [
                w for w in workflows 
                if w.match_score >= self.config.routing_thresholds['multi_workflow_threshold']
            ]
            
            if len(high_score_workflows) < 2:
                # Fall back to best match if not enough high-scoring workflows
                return await self.route_best_match(route_id, workflows, analysis)
            
            # Select primary workflow (highest score)
            primary_workflow = max(high_score_workflows, key=lambda w: w.match_score)
            
            # Create multi-workflow route
            route = WorkflowRoute(
                route_id=route_id,
                selected_workflow=primary_workflow,
                routing_strategy=RoutingStrategy.MULTI_WORKFLOW,
                route_confidence=primary_workflow.match_score * 0.9,  # Slight penalty for complexity
                fallback_workflows=high_score_workflows[1:] if len(high_score_workflows) > 1 else [],
                routing_metadata={
                    'strategy': 'multi_workflow',
                    'primary_workflow': primary_workflow.workflow_id,
                    'secondary_workflows': [w.workflow_id for w in high_score_workflows[1:]],
                    'multi_workflow_execution': True,
                    'execution_plan': 'sequential'  # Could be configured
                }
            )
            
            logger.debug(f"✅ Multi-workflow route created with {len(high_score_workflows)} workflows")
            return route
            
        except Exception as e:
            logger.error(f"❌ Multi-workflow routing failed: {e}")
            raise
    
    async def route_fallback_chain(self, route_id: str, workflows: List[WorkflowMatch], 
                                 analysis: RequestAnalysis) -> WorkflowRoute:
        """Route using fallback chain strategy"""
        try:
            if not workflows:
                return await self.create_fallback_route(route_id, None, analysis)
            
            # Sort workflows by match score
            sorted_workflows = sorted(workflows, key=lambda w: w.match_score, reverse=True)
            
            # Select primary workflow
            primary_workflow = sorted_workflows[0]
            
            # Create fallback chain
            fallback_workflows = sorted_workflows[1:5]  # Up to 4 fallbacks
            
            route = WorkflowRoute(
                route_id=route_id,
                selected_workflow=primary_workflow,
                routing_strategy=RoutingStrategy.FALLBACK_CHAIN,
                route_confidence=primary_workflow.match_score,
                fallback_workflows=fallback_workflows,
                routing_metadata={
                    'strategy': 'fallback_chain',
                    'fallback_chain_length': len(fallback_workflows),
                    'retry_on_failure': True,
                    'max_fallback_attempts': len(fallback_workflows)
                }
            )
            
            logger.debug(f"✅ Fallback chain route created with {len(fallback_workflows)} fallbacks")
            return route
            
        except Exception as e:
            logger.error(f"❌ Fallback chain routing failed: {e}")
            raise
    
    async def route_confidence_threshold(self, route_id: str, workflows: List[WorkflowMatch], 
                                       analysis: RequestAnalysis) -> WorkflowRoute:
        """Route using confidence threshold strategy"""
        try:
            high_confidence_threshold = self.config.routing_thresholds['high_confidence']
            medium_confidence_threshold = self.config.routing_thresholds['minimum_confidence']
            
            # Filter workflows by confidence levels
            high_confidence_workflows = [w for w in workflows if w.match_score >= high_confidence_threshold]
            medium_confidence_workflows = [w for w in workflows if w.match_score >= medium_confidence_threshold]
            
            if high_confidence_workflows:
                # Use best high-confidence workflow
                selected_workflow = max(high_confidence_workflows, key=lambda w: w.match_score)
                confidence_level = 'high'
            elif medium_confidence_workflows:
                # Use best medium-confidence workflow
                selected_workflow = max(medium_confidence_workflows, key=lambda w: w.match_score)
                confidence_level = 'medium'
            else:
                # No workflows meet minimum threshold, use fallback
                return await self.create_fallback_route(route_id, None, analysis)
            
            route = WorkflowRoute(
                route_id=route_id,
                selected_workflow=selected_workflow,
                routing_strategy=RoutingStrategy.CONFIDENCE_THRESHOLD,
                route_confidence=selected_workflow.match_score,
                fallback_workflows=medium_confidence_workflows[:3] if confidence_level == 'high' else [],
                routing_metadata={
                    'strategy': 'confidence_threshold',
                    'confidence_level': confidence_level,
                    'threshold_used': high_confidence_threshold if confidence_level == 'high' else medium_confidence_threshold,
                    'workflows_above_threshold': len(high_confidence_workflows) if confidence_level == 'high' else len(medium_confidence_workflows)
                }
            )
            
            logger.debug(f"✅ Confidence threshold route created: {confidence_level} confidence")
            return route
            
        except Exception as e:
            logger.error(f"❌ Confidence threshold routing failed: {e}")
            raise
    
    async def create_fallback_route(self, route_id: str, request: Optional[ChatRequest], 
                                  analysis: RequestAnalysis) -> WorkflowRoute:
        """Create fallback route when no suitable workflows found"""
        try:
            # Create a fallback workflow match
            fallback_workflow = WorkflowMatch(
                workflow_id="general_conversation_fallback",
                workflow_class="nanobrain.library.workflows.general.conversation_workflow.ConversationWorkflow",
                match_score=0.3,
                match_reasons=["fallback_strategy"],
                capability_alignment={"fallback": 1.0},
                estimated_processing_time=5.0,
                metadata={"is_fallback": True, "fallback_reason": "no_compatible_workflows"}
            )
            
            route = WorkflowRoute(
                route_id=route_id,
                selected_workflow=fallback_workflow,
                routing_strategy=RoutingStrategy.FALLBACK_CHAIN,
                route_confidence=0.3,
                fallback_workflows=[],
                routing_metadata={
                    'strategy': 'fallback',
                    'fallback_reason': 'no_compatible_workflows',
                    'original_strategy': self.config.routing_strategy,
                    'is_fallback_route': True
                }
            )
            
            logger.debug(f"✅ Fallback route created: {fallback_workflow.workflow_id}")
            return route
            
        except Exception as e:
            logger.error(f"❌ Fallback route creation failed: {e}")
            raise
    
    async def create_error_route(self, route_id: str, request: Optional[ChatRequest], 
                               analysis: RequestAnalysis, error_message: str) -> WorkflowRoute:
        """Create error route when routing fails"""
        try:
            error_workflow = WorkflowMatch(
                workflow_id="error_handling_workflow",
                workflow_class="nanobrain.library.workflows.error.error_handler.ErrorHandlerWorkflow",
                match_score=0.1,
                match_reasons=["error_fallback"],
                capability_alignment={"error_handling": 1.0},
                estimated_processing_time=1.0,
                metadata={"is_error": True, "error_message": error_message}
            )
            
            route = WorkflowRoute(
                route_id=route_id,
                selected_workflow=error_workflow,
                routing_strategy=RoutingStrategy.FALLBACK_CHAIN,
                route_confidence=0.1,
                fallback_workflows=[],
                routing_metadata={
                    'strategy': 'error_fallback',
                    'error_occurred': True,
                    'error_message': error_message,
                    'original_strategy': self.config.routing_strategy
                }
            )
            
            logger.debug(f"✅ Error route created: {error_message}")
            return route
            
        except Exception as e:
            logger.error(f"❌ Error route creation failed: {e}")
            raise
    
    async def execute_route(self, route: WorkflowRoute) -> UniversalResponse:
        """
        Execute workflow route and return standardized response.
        
        Args:
            route: WorkflowRoute to execute
            
        Returns:
            UniversalResponse with execution results
        """
        try:
            execution_start = datetime.now()
            response_id = f"resp_{uuid.uuid4().hex[:8]}"
            
            logger.debug(f"⚡ Executing route: {route.route_id}")
            
            # Check execution cache if enabled
            if self.config.execution_config.get('enable_caching', True):
                cached_response = self.get_cached_execution(route)
                if cached_response:
                    logger.debug(f"✅ Using cached execution result for route: {route.route_id}")
                    return cached_response
            
            # Execute primary workflow
            try:
                response = await self.execute_workflow(route.selected_workflow, route)
                
                if response.success:
                    # Cache successful response
                    if self.config.execution_config.get('enable_caching', True):
                        self.cache_execution(route, response)
                    
                    execution_time = (datetime.now() - execution_start).total_seconds()
                    response.processing_time = execution_time
                    response.metadata['route_id'] = route.route_id
                    response.metadata['execution_strategy'] = 'primary_workflow'
                    
                    logger.debug(f"✅ Route execution successful: {route.selected_workflow.workflow_id}")
                    return response
                    
            except Exception as e:
                logger.warning(f"⚠️ Primary workflow execution failed: {e}")
                
                # Try fallback workflows if configured
                if self.config.error_handling.get('retry_failed_routes', True) and route.fallback_workflows:
                    return await self.execute_fallback_workflows(route, str(e))
            
            # If all executions failed, return error response
            return await self.create_error_response(route, "All workflow executions failed")
            
        except Exception as e:
            logger.error(f"❌ Route execution failed: {e}")
            return await self.create_error_response(route, str(e))
    
    async def execute_workflow(self, workflow_match: WorkflowMatch, route: WorkflowRoute) -> UniversalResponse:
        """Execute a specific workflow using real NanoBrain framework patterns"""
        try:
            # Get or create workflow instance
            workflow_instance = await self.get_workflow_instance(workflow_match)
            
            # ✅ FRAMEWORK COMPLIANCE: Execute real workflow using proper process method
            logger.debug(f"🔄 Executing real workflow: {workflow_match.workflow_id}")
            
            # ✅ NEW: Prepare request data for workflow
            request_data = self.convert_universal_request_to_workflow_format(route)
            
            # ✅ FRAMEWORK COMPLIANCE: Call workflow's process method instead of direct data unit manipulation
            logger.debug(f"🚀 Calling workflow process method: {workflow_match.workflow_id}")
            
            # Execute the workflow using its process method (proper framework pattern)
            result = await workflow_instance.process(request_data)
            
            # ✅ FRAMEWORK COMPLIANCE: Convert workflow result to UniversalResponse
            logger.debug(f"✅ Workflow executed successfully: {workflow_match.workflow_id}")
            return await self.convert_workflow_result_to_universal_response(
                result, workflow_match, route
            )
            
        except Exception as e:
            logger.error(f"❌ Real workflow execution failed: {e}")
            return self.create_workflow_error_response(workflow_match, route, str(e))
    
    def convert_universal_request_to_workflow_format(self, route: WorkflowRoute) -> Dict[str, Any]:
        """Convert Universal Interface request to workflow-expected format"""
        # ✅ FRAMEWORK COMPLIANCE: Extract request data from routing metadata
        routing_metadata = route.routing_metadata
        
        # Extract request data that should have been stored during routing
        user_query = routing_metadata.get('user_query', 'Unknown query')
        conversation_id = routing_metadata.get('conversation_id', 'default')
        request_id = routing_metadata.get('request_id', route.route_id)
        
        return {
            "user_query": user_query,
            "conversation_id": conversation_id,
            "session_id": conversation_id,  # Legacy compatibility
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "routing_score": route.route_confidence,
                "routing_reasons": route.routing_metadata.get("match_reasons", [])
            }
        }
    
    async def wait_for_workflow_completion(self, output_data_unit, timeout: float = 30.0):
        """Wait for workflow completion with timeout"""
        import asyncio
        import time
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if output data unit has result
            current_value = await output_data_unit.get()
            if current_value is not None:
                logger.debug(f"📤 Workflow completed, received result")
                return current_value
            
            # Wait briefly before checking again
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"Workflow execution timed out after {timeout} seconds")
    
    async def convert_workflow_result_to_universal_response(self, result, workflow_match, route):
        """Convert workflow execution result to UniversalResponse format"""
        try:
            # Extract response content from workflow result
            response_content = self.extract_response_content(result)
            
            return UniversalResponse(
                response_id=f"resp_{uuid.uuid4().hex[:8]}",
                workflow_id=workflow_match.workflow_id,
                response_type="workflow_execution",
                content=response_content,
                success=True,
                error_details=None,
                processing_time=None,  # Will be set by caller
                metadata={
                    "workflow_class": workflow_match.workflow_class,
                    "match_score": workflow_match.match_score,
                    "real_execution": True,  # ✅ FLAG: Real execution, not mock
                    "execution_method": "data_unit_communication"
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Response conversion failed: {e}")
            raise
    
    def extract_response_content(self, result):
        """Extract response content from workflow result"""
        try:
            # Handle different result formats
            if isinstance(result, dict):
                # Extract the actual response message
                if 'chatbot_response' in result:
                    response_text = result['chatbot_response']
                elif 'response' in result:
                    response_text = result['response'] 
                elif 'message' in result:
                    response_text = result['message']
                else:
                    response_text = str(result)
                
                return {
                    "message": response_text,
                    "original_result": result,
                    "format": "conversational_response"
                }
            else:
                return {
                    "message": str(result),
                    "format": "string_response"
                }
                
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return {
                "message": "Response processing completed.",
                "error": f"Content extraction failed: {e}",
                "format": "error_fallback"
            }
    
    def create_workflow_error_response(self, workflow_match, route, error_message):
        """Create comprehensive error response for workflow execution failures"""
        return UniversalResponse(
            response_id=f"resp_error_{uuid.uuid4().hex[:8]}",
            workflow_id=workflow_match.workflow_id,
            response_type="execution_error",
            content={
                "error": "Workflow execution failed",
                "details": error_message,
                "suggestions": [
                    "Check workflow configuration",
                    "Verify input data format",
                    "Check workflow dependencies"
                ]
            },
            success=False,
            error_details={
                "error_type": "workflow_execution_error",
                "error_message": error_message,
                "workflow_class": workflow_match.workflow_class,
                "timestamp": datetime.now().isoformat()
            },
            processing_time=0.0,
            metadata={"execution_failed": True, "error_context": "universal_interface"},
            timestamp=datetime.now()
        )
    
    async def get_workflow_instance(self, workflow_match: WorkflowMatch) -> Any:
        """Get or create real workflow instance using NanoBrain framework patterns"""
        try:
            workflow_id = workflow_match.workflow_id
            
            # Check if we already have an instance
            if workflow_id in self.workflow_instances:
                logger.debug(f"🔄 Using cached workflow instance: {workflow_id}")
                return self.workflow_instances[workflow_id]
            
            # ✅ FRAMEWORK COMPLIANCE: Create real workflow instance
            logger.debug(f"🏭 Creating new workflow instance: {workflow_id}")
            
            # ✅ FRAMEWORK COMPLIANCE: Dynamic import of workflow class
            workflow_class = self.import_workflow_class(workflow_match.workflow_class)
            
            # ✅ FRAMEWORK COMPLIANCE: Get workflow configuration path
            workflow_config_path = self.get_workflow_config_path(workflow_match.workflow_id)
            
            # ✅ FRAMEWORK COMPLIANCE: Create workflow instance using from_config
            logger.debug(f"📋 Loading workflow from config: {workflow_config_path}")
            workflow_instance = workflow_class.from_config(workflow_config_path)
            
            # ✅ FRAMEWORK COMPLIANCE: Cache workflow instance for reuse
            self.workflow_instances[workflow_id] = workflow_instance
            
            logger.debug(f"✅ Real workflow instance created: {workflow_id}")
            return workflow_instance
            
        except Exception as e:
            logger.error(f"❌ Workflow instance creation failed: {e}")
            raise
    
    def import_workflow_class(self, workflow_class_path: str):
        """Dynamically import workflow class from module path"""
        try:
            # Split module path and class name
            module_path, class_name = workflow_class_path.rsplit('.', 1)
            
            # Import module
            import importlib
            module = importlib.import_module(module_path)
            
            # Get class from module
            workflow_class = getattr(module, class_name)
            
            logger.debug(f"✅ Workflow class imported: {class_name}")
            return workflow_class
            
        except Exception as e:
            logger.error(f"❌ Workflow class import failed: {e}")
            raise ImportError(f"Failed to import workflow class: {workflow_class_path}")
    
    def get_workflow_config_path(self, workflow_id: str) -> str:
        """Get configuration file path for workflow"""
        # ✅ FRAMEWORK COMPLIANCE: Map workflow IDs to config paths
        workflow_config_mapping = {
            "chatbot_viral_integration": "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml",
            "chatbot_viral_workflow": "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml",  # Alternative ID
            "viral_protein_analysis": "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml",
            "alphavirus_workflow": "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml",  # Alternative ID
        }
        
        if workflow_id not in workflow_config_mapping:
            logger.warning(f"⚠️ No explicit config mapping for workflow: {workflow_id}")
            # Try to construct default path
            default_path = f"nanobrain/library/workflows/{workflow_id}/config/{workflow_id.replace('_', '').title()}Workflow.yml"
            logger.debug(f"🔍 Trying default config path: {default_path}")
            return default_path
        
        return workflow_config_mapping[workflow_id]
    
    async def execute_fallback_workflows(self, route: WorkflowRoute, primary_error: str) -> UniversalResponse:
        """Execute fallback workflows when primary fails"""
        try:
            logger.info(f"🔄 Executing fallback workflows for route: {route.route_id}")
            
            for i, fallback_workflow in enumerate(route.fallback_workflows):
                try:
                    response = await self.execute_workflow(fallback_workflow, route)
                    
                    if response.success:
                        response.metadata['execution_strategy'] = 'fallback_workflow'
                        response.metadata['fallback_index'] = i
                        response.metadata['primary_failure'] = primary_error
                        
                        logger.debug(f"✅ Fallback workflow successful: {fallback_workflow.workflow_id}")
                        return response
                        
                except Exception as e:
                    logger.warning(f"⚠️ Fallback workflow {i} failed: {e}")
                    continue
            
            # All fallbacks failed
            return await self.create_error_response(route, f"Primary and all fallback executions failed. Primary error: {primary_error}")
            
        except Exception as e:
            logger.error(f"❌ Fallback execution failed: {e}")
            return await self.create_error_response(route, str(e))
    
    async def create_error_response(self, route: WorkflowRoute, error_message: str) -> UniversalResponse:
        """Create error response when execution fails"""
        return UniversalResponse(
            response_id=f"resp_error_{uuid.uuid4().hex[:8]}",
            workflow_id=route.selected_workflow.workflow_id,
            response_type="execution_error",
            content={
                "error": "Workflow execution failed",
                "message": "I apologize, but I encountered an error processing your request. Please try again."
            },
            success=False,
            error_details={
                "error_message": error_message,
                "route_id": route.route_id,
                "workflow_id": route.selected_workflow.workflow_id
            },
            processing_time=0.0,
            metadata={
                "error_occurred": True,
                "routing_strategy": route.routing_strategy.value,
                "route_confidence": route.route_confidence
            },
            timestamp=datetime.now()
        )
    
    def get_cached_route(self, query: str) -> Optional[WorkflowRoute]:
        """Get cached route if available and valid"""
        cache_key = hash(query.strip().lower())
        return self.route_cache.get(str(cache_key))
    
    def cache_route(self, query: str, route: WorkflowRoute) -> None:
        """Cache route result"""
        cache_key = str(hash(query.strip().lower()))
        self.route_cache[cache_key] = route
        
        # Simple cache size management
        if len(self.route_cache) > 1000:
            # Remove oldest entries (simplified approach)
            keys_to_remove = list(self.route_cache.keys())[:200]
            for key in keys_to_remove:
                del self.route_cache[key]
    
    def get_cached_execution(self, route: WorkflowRoute) -> Optional[UniversalResponse]:
        """Get cached execution result"""
        cache_key = f"{route.selected_workflow.workflow_id}_{hash(str(route.routing_metadata))}"
        return self.execution_cache.get(cache_key)
    
    def cache_execution(self, route: WorkflowRoute, response: UniversalResponse) -> None:
        """Cache execution result"""
        cache_key = f"{route.selected_workflow.workflow_id}_{hash(str(route.routing_metadata))}"
        self.execution_cache[cache_key] = response
        
        # Simple cache size management
        if len(self.execution_cache) > 500:
            keys_to_remove = list(self.execution_cache.keys())[:100]
            for key in keys_to_remove:
                del self.execution_cache[key]
    
    async def handle_multi_workflow_route(self, route: MultiWorkflowRoute) -> UniversalResponse:
        """Handle requests requiring multiple workflows"""
        # This would be implemented for complex multi-workflow scenarios
        # For now, returning a placeholder
        logger.info(f"🔄 Multi-workflow execution not yet implemented for route: {route.route_id}")
        
        return UniversalResponse(
            response_id=f"resp_{uuid.uuid4().hex[:8]}",
            workflow_id="multi_workflow_placeholder",
            response_type="multi_workflow_placeholder",
            content={"message": "Multi-workflow execution not yet implemented"},
            success=False,
            error_details={"message": "Multi-workflow execution feature in development"},
            processing_time=0.0,
            metadata={"multi_workflow": True, "placeholder": True},
            timestamp=datetime.now()
        )
    
    async def get_health_status(self) -> str:
        """Get router health status"""
        try:
            # Basic health check - ensure router can create routes
            if self.config and hasattr(self.config, 'routing_strategy'):
                return "healthy"
            else:
                return "unhealthy"
        except Exception as e:
            logger.error(f"❌ Router health check failed: {e}")
            return "unhealthy" 