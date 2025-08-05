#!/usr/bin/env python3
"""
Universal Interface Models for NanoBrain Framework
Defines models for universal request analysis, workflow routing, and response processing.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from nanobrain.library.interfaces.web.models.request_models import ChatRequest
from nanobrain.library.interfaces.web.models.response_models import ChatResponse


class IntentType(str, Enum):
    """Enumeration of supported intent types for natural language requests"""
    INFORMATION_REQUEST = "information_request"
    ANALYSIS_REQUEST = "analysis_request"
    COMPARISON_REQUEST = "comparison_request"
    EXPLANATION_REQUEST = "explanation_request"
    PROCEDURE_REQUEST = "procedure_request"
    GENERAL_CONVERSATION = "general_conversation"
    UNKNOWN = "unknown"


class DomainType(str, Enum):
    """Enumeration of supported domain types for request classification"""
    VIROLOGY = "virology"
    BIOINFORMATICS = "bioinformatics"
    PROTEIN_ANALYSIS = "protein_analysis"
    GENOMICS = "genomics"
    GENERAL_SCIENCE = "general_science"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


class RoutingStrategy(str, Enum):
    """Enumeration of workflow routing strategies"""
    BEST_MATCH = "best_match"
    MULTI_WORKFLOW = "multi_workflow"
    FALLBACK_CHAIN = "fallback_chain"
    CONFIDENCE_THRESHOLD = "confidence_threshold"


class IntentClassification(BaseModel):
    """Classification result for user intent analysis"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "intent_type": "analysis_request",
                "confidence": 0.85,
                "keywords": ["protein", "analysis", "sequence"],
                "classification_method": "keyword_based"
            }
        }
    )
    
    intent_type: IntentType = Field(..., description="Classified intent type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    keywords: List[str] = Field(default_factory=list, description="Key terms that influenced classification")
    classification_method: str = Field(..., description="Method used for classification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional classification metadata")


class DomainClassification(BaseModel):
    """Classification result for domain analysis"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "domain_type": "virology",
                "confidence": 0.9,
                "indicators": ["virus", "viral", "pathogen"],
                "classification_method": "semantic_analysis"
            }
        }
    )
    
    domain_type: DomainType = Field(..., description="Classified domain type")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    indicators: List[str] = Field(default_factory=list, description="Domain indicators found in request")
    classification_method: str = Field(..., description="Method used for domain classification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional domain metadata")


class RequestAnalysis(BaseModel):
    """Complete analysis result for natural language request"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "req_123",
                "original_query": "Analyze this protein sequence for viral origins",
                "intent_classification": {
                    "intent_type": "analysis_request",
                    "confidence": 0.85
                },
                "domain_classification": {
                    "domain_type": "virology", 
                    "confidence": 0.9
                },
                "analysis_timestamp": "2025-01-01T12:00:00Z"
            }
        }
    )
    
    request_id: str = Field(..., description="Unique request identifier")
    original_query: str = Field(..., description="Original user query")
    intent_classification: IntentClassification = Field(..., description="Intent analysis result")
    domain_classification: DomainClassification = Field(..., description="Domain analysis result")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities from query")
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Query complexity assessment")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis completion time")
    analyzer_metadata: Dict[str, Any] = Field(default_factory=dict, description="Analyzer-specific metadata")


class WorkflowMatch(BaseModel):
    """Workflow matching result for request routing"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "viral_protein_analysis",
                "workflow_class": "nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow.AlphavirusWorkflow",
                "match_score": 0.95,
                "match_reasons": ["domain_match", "intent_match", "capability_match"]
            }
        }
    )
    
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_class: str = Field(..., description="Full class path for workflow")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Overall match score")
    match_reasons: List[str] = Field(default_factory=list, description="Reasons for workflow selection")
    capability_alignment: Dict[str, float] = Field(default_factory=dict, description="Capability alignment scores")
    estimated_processing_time: Optional[float] = Field(default=None, description="Estimated processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional matching metadata")


class WorkflowRoute(BaseModel):
    """Single workflow routing decision"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "route_id": "route_123",
                "selected_workflow": {
                    "workflow_id": "viral_protein_analysis",
                    "match_score": 0.95
                },
                "routing_strategy": "best_match",
                "route_confidence": 0.9
            }
        }
    )
    
    route_id: str = Field(..., description="Unique route identifier")
    selected_workflow: WorkflowMatch = Field(..., description="Selected workflow for execution")
    routing_strategy: RoutingStrategy = Field(..., description="Strategy used for routing")
    route_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in routing decision")
    fallback_workflows: List[WorkflowMatch] = Field(default_factory=list, description="Fallback workflow options")
    routing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Routing decision metadata")


class MultiWorkflowRoute(BaseModel):
    """Multi-workflow routing for complex requests"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "route_id": "multi_route_123",
                "workflows": [
                    {"workflow_id": "viral_protein_analysis", "sequence": 1},
                    {"workflow_id": "result_aggregation", "sequence": 2}
                ],
                "execution_strategy": "sequential"
            }
        }
    )
    
    route_id: str = Field(..., description="Unique multi-route identifier")
    workflows: List[WorkflowMatch] = Field(..., description="Ordered list of workflows to execute")
    execution_strategy: str = Field(..., description="Execution strategy (sequential, parallel, conditional)")
    aggregation_method: str = Field(default="merge", description="Method for aggregating results")
    route_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall route confidence")
    estimated_total_time: Optional[float] = Field(default=None, description="Estimated total processing time")
    routing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Multi-route metadata")


class UniversalResponse(BaseModel):
    """Universal response format for any workflow output"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response_id": "resp_123",
                "workflow_id": "viral_protein_analysis",
                "response_type": "analysis_results",
                "content": {"analysis": "protein analysis results"},
                "success": True
            }
        }
    )
    
    response_id: str = Field(..., description="Unique response identifier")
    workflow_id: str = Field(..., description="Source workflow identifier")
    response_type: str = Field(..., description="Type of response content")
    content: Dict[str, Any] = Field(..., description="Response content data")
    success: bool = Field(..., description="Whether workflow execution was successful")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if unsuccessful")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response generation time")


class StandardizedResponse(BaseModel):
    """Standardized response format for frontend consumption"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response_id": "std_resp_123",
                "message": "Analysis completed successfully",
                "data": {"results": "standardized results"},
                "response_format": "structured_data",
                "success": True
            }
        }
    )
    
    response_id: str = Field(..., description="Unique standardized response identifier")
    message: str = Field(..., description="Human-readable response message")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured response data")
    response_format: str = Field(..., description="Format type (text, structured_data, file, visualization)")
    success: bool = Field(..., description="Overall success status")
    error_message: Optional[str] = Field(default=None, description="Error message if unsuccessful")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    frontend_hints: Dict[str, Any] = Field(default_factory=dict, description="Hints for frontend rendering")


class AggregatedResponse(BaseModel):
    """Aggregated response from multiple workflows"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "aggregation_id": "agg_123",
                "source_responses": ["resp_1", "resp_2"],
                "aggregated_content": {"combined": "results"},
                "aggregation_method": "merge"
            }
        }
    )
    
    aggregation_id: str = Field(..., description="Unique aggregation identifier")
    source_responses: List[str] = Field(..., description="Source response identifiers")
    aggregated_content: Dict[str, Any] = Field(..., description="Combined response content")
    aggregation_method: str = Field(..., description="Method used for aggregation")
    success: bool = Field(..., description="Overall aggregation success")
    partial_failures: List[str] = Field(default_factory=list, description="Failed component responses")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Aggregation metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Aggregation completion time")


class StreamingResponse(BaseModel):
    """Streaming response container for real-time data"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stream_id": "stream_123",
                "chunk_id": "chunk_1",
                "content": {"partial": "results"},
                "is_final": False,
                "progress": 0.25
            }
        }
    )
    
    stream_id: str = Field(..., description="Unique streaming session identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: Dict[str, Any] = Field(..., description="Streaming content chunk")
    is_final: bool = Field(..., description="Whether this is the final chunk")
    progress: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Progress percentage")
    chunk_type: str = Field(default="data", description="Type of chunk (data, status, error)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Chunk generation time") 