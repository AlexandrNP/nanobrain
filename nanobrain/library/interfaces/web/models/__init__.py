#!/usr/bin/env python3
"""
Web Interface Models Package
Provides all Pydantic models for web interfaces, request/response handling, and universal interface architecture.

Author: NanoBrain Development Team  
Date: January 2025
Version: 1.0.0
"""

# Standard request/response models (existing framework)
from .request_models import (
    ChatRequest, ChatOptions, StreamingChatRequest
)
from .response_models import (
    ChatResponse, ErrorResponse, StreamingChatResponse, 
    HealthResponse, WorkflowStatusResponse
)

# Universal interface models (new)
from .universal_models import (
    # Enums
    IntentType, DomainType, RoutingStrategy,
    # Classification models
    IntentClassification, DomainClassification, RequestAnalysis,
    # Routing models
    WorkflowMatch, WorkflowRoute, MultiWorkflowRoute,
    # Response models
    UniversalResponse, StandardizedResponse, AggregatedResponse, StreamingResponse
)

# Workflow capability models (new)
from .workflow_models import (
    # Enums
    InputType, OutputType, InteractionPattern,
    # Capability models
    WorkflowCapabilities, WorkflowRequirements,
    # Discovery and registry models
    WorkflowDiscoveryResult, WorkflowValidationResult, WorkflowRegistryEntry,
    # Execution models
    WorkflowCompatibilityScore, WorkflowExecutionPlan
)

__all__ = [
    # Standard framework models
    "ChatRequest", "ChatOptions", "StreamingChatRequest",
    "ChatResponse", "ErrorResponse", "StreamingChatResponse", 
    "HealthResponse", "WorkflowStatusResponse",
    
    # Universal interface enums
    "IntentType", "DomainType", "RoutingStrategy",
    
    # Classification and analysis
    "IntentClassification", "DomainClassification", "RequestAnalysis",
    
    # Workflow routing
    "WorkflowMatch", "WorkflowRoute", "MultiWorkflowRoute",
    
    # Universal responses
    "UniversalResponse", "StandardizedResponse", "AggregatedResponse", "StreamingResponse",
    
    # Workflow capability enums
    "InputType", "OutputType", "InteractionPattern",
    
    # Workflow capabilities and requirements
    "WorkflowCapabilities", "WorkflowRequirements",
    
    # Workflow discovery and registry
    "WorkflowDiscoveryResult", "WorkflowValidationResult", "WorkflowRegistryEntry",
    
    # Workflow execution
    "WorkflowCompatibilityScore", "WorkflowExecutionPlan"
] 