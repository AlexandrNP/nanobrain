#!/usr/bin/env python3
"""
Universal Web Interface for NanoBrain Framework
Complete universal interface system supporting any NanoBrain workflow with natural language input.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

# Request/Response models for universal interface
from .models import (
    # Standard framework models
    ChatRequest, ChatOptions, StreamingChatRequest,
    ChatResponse, ErrorResponse, StreamingChatResponse,
    HealthResponse, WorkflowStatusResponse,
    
    # Universal interface models
    IntentType, DomainType, RoutingStrategy,
    IntentClassification, DomainClassification, RequestAnalysis,
    WorkflowMatch, WorkflowRoute, MultiWorkflowRoute,
    UniversalResponse, StandardizedResponse, AggregatedResponse, StreamingResponse,
    
    # Workflow capability models
    InputType, OutputType, InteractionPattern,
    WorkflowCapabilities, WorkflowRequirements,
    WorkflowDiscoveryResult, WorkflowValidationResult, WorkflowRegistryEntry,
    WorkflowCompatibilityScore, WorkflowExecutionPlan
)

# Universal server components
from .servers import (
    BaseUniversalServer, BaseServerConfig,
    UniversalNanoBrainServer, UniversalServerConfig,
    UniversalServerFactory, ServerFactoryConfig
)

# Request analysis and classification
from .analysis import (
    UniversalRequestAnalyzer, RequestAnalyzerConfig,
    IntentClassifier, DomainClassifier
)

# Workflow routing and discovery
from .routing import (
    WorkflowRegistry, WorkflowRegistryConfig,
    WorkflowRouter, WorkflowRouterConfig,
    RoutingStrategyConfig, BaseRoutingStrategy,
    BestMatchStrategy, WeightedScoringStrategy,
    AdaptiveStrategy, MultiCriteriaStrategy
)

# Response processing and formatting
from .processing import (
    UniversalResponseProcessor, ResponseProcessorConfig,
    FormatConverter, FormatConverterConfig,
    BaseFormatConverter, JSONConverter, CSVConverter, TextConverter,
    StreamingHandler, StreamingHandlerConfig,
    StreamSession, StreamingType
)

# Frontend API models (for backend-frontend communication)
from .api import (
    FrontendChatRequest, FrontendChatResponse,
    FrontendErrorResponse, FrontendHealthResponse,
)

# Legacy web interface (maintained for compatibility)
from .web_interface import WebInterface
from .config.web_interface_config import WebInterfaceConfig

__all__ = [
    # Models - Standard framework
    "ChatRequest", "ChatOptions", "StreamingChatRequest",
    "ChatResponse", "ErrorResponse", "StreamingChatResponse", 
    "HealthResponse", "WorkflowStatusResponse",
    
    # Models - Universal interface
    "IntentType", "DomainType", "RoutingStrategy",
    "IntentClassification", "DomainClassification", "RequestAnalysis",
    "WorkflowMatch", "WorkflowRoute", "MultiWorkflowRoute",
    "UniversalResponse", "StandardizedResponse", "AggregatedResponse", "StreamingResponse",
    
    # Models - Workflow capabilities
    "InputType", "OutputType", "InteractionPattern",
    "WorkflowCapabilities", "WorkflowRequirements",
    "WorkflowDiscoveryResult", "WorkflowValidationResult", "WorkflowRegistryEntry",
    "WorkflowCompatibilityScore", "WorkflowExecutionPlan",
    
    # Servers
    "BaseUniversalServer", "BaseServerConfig",
    "UniversalNanoBrainServer", "UniversalServerConfig",
    "UniversalServerFactory", "ServerFactoryConfig",
    
    # Analysis
    "UniversalRequestAnalyzer", "RequestAnalyzerConfig",
    "IntentClassifier", "DomainClassifier",
    
    # Routing
    "WorkflowRegistry", "WorkflowRegistryConfig",
    "WorkflowRouter", "WorkflowRouterConfig",
    "RoutingStrategyConfig", "BaseRoutingStrategy",
    "BestMatchStrategy", "WeightedScoringStrategy",
    "AdaptiveStrategy", "MultiCriteriaStrategy",
    
    # Processing
    "UniversalResponseProcessor", "ResponseProcessorConfig",
    "FormatConverter", "FormatConverterConfig",
    "BaseFormatConverter", "JSONConverter", "CSVConverter", "TextConverter",
    "StreamingHandler", "StreamingHandlerConfig",
    "StreamSession", "StreamingType",
    
    # API models
    "FrontendChatRequest", "FrontendChatResponse",
    "FrontendErrorResponse", "FrontendHealthResponse",
    
    # Legacy interfaces (maintained for compatibility)
    "WebInterface", "WebInterfaceConfig"
]

# Framework information
__version__ = "1.0.0"
__author__ = "NanoBrain Development Team"
__description__ = "Universal web interface for NanoBrain framework workflows"
__framework_compliance__ = "nanobrain-v1.0"

# Universal interface metadata
UNIVERSAL_INTERFACE_INFO = {
    "name": "NanoBrain Universal Web Interface",
    "version": __version__,
    "framework_version": "1.0.0",
    "build_date": "2025-01-01",
    "components": {
        "models": "Universal data models and schemas",
        "servers": "Configurable universal server implementations", 
        "analysis": "Natural language request analysis and classification",
        "routing": "Intelligent workflow discovery and routing",
        "processing": "Universal response processing and formatting",
        "api": "Frontend-backend communication models",
        "frontend": "Dynamic frontend adaptation components (JavaScript)"
    },
    "capabilities": [
        "Universal workflow adaptation",
        "Natural language request processing", 
        "Intelligent workflow routing",
        "Multi-format response handling",
        "Real-time streaming support",
        "Dynamic frontend adaptation",
        "Framework-compliant component assembly"
    ],
    "supported_workflows": "Any NanoBrain workflow with from_config pattern",
    "supported_formats": [
        "JSON", "CSV", "HTML", "XML", "Text", "Markdown", 
        "Streaming", "Binary", "Visualization"
    ]
} 