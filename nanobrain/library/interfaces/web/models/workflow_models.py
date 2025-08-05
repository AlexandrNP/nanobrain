#!/usr/bin/env python3
"""
Workflow Capability Models for NanoBrain Framework
Defines models for describing workflow capabilities, requirements, and metadata.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class InputType(str, Enum):
    """Enumeration of supported input types for workflows"""
    NATURAL_LANGUAGE_QUERY = "natural_language_query"
    PROTEIN_SEQUENCE = "protein_sequence"
    DNA_SEQUENCE = "dna_sequence"
    VIRUS_NAME = "virus_name"
    ANALYSIS_REQUEST = "analysis_request"
    CONVERSATIONAL_QUERY = "conversational_query"
    FILE_UPLOAD = "file_upload"
    STRUCTURED_DATA = "structured_data"


class OutputType(str, Enum):
    """Enumeration of supported output types from workflows"""
    ANALYSIS_RESULTS = "analysis_results"
    CONVERSATIONAL_RESPONSE = "conversational_response"
    STRUCTURED_INFORMATION = "structured_information"
    VISUALIZATIONS = "visualizations"
    REPORTS = "reports"
    FILE_DOWNLOAD = "file_download"
    STREAMING_DATA = "streaming_data"
    JSON_DATA = "json_data"


class InteractionPattern(str, Enum):
    """Enumeration of supported interaction patterns"""
    CHAT = "chat"
    QUESTION_ANSWER = "q_and_a"
    DATA_UPLOAD = "data_upload"
    PROGRESS_TRACKING = "progress_tracking"
    FORM_BASED = "form_based"
    STREAMING = "streaming"
    BATCH_PROCESSING = "batch_processing"


class WorkflowCapabilities(BaseModel):
    """Complete description of workflow capabilities and requirements"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "viral_protein_analysis",
                "workflow_class": "nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow.AlphavirusWorkflow",
                "natural_language_input": True,
                "input_types": ["protein_sequence", "virus_name"],
                "output_types": ["analysis_results", "visualizations"],
                "interaction_patterns": ["chat", "data_upload"],
                "domains": ["virology", "bioinformatics"]
            }
        }
    )
    
    # Core identification
    workflow_id: str = Field(..., description="Unique workflow identifier")
    workflow_class: str = Field(..., description="Full Python class path")
    workflow_name: str = Field(..., description="Human-readable workflow name")
    description: str = Field(..., description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    
    # Input/Output capabilities
    natural_language_input: bool = Field(..., description="Whether workflow accepts natural language input")
    input_types: List[InputType] = Field(..., description="Supported input data types")
    output_types: List[OutputType] = Field(..., description="Produced output data types")
    interaction_patterns: List[InteractionPattern] = Field(..., description="Supported interaction patterns")
    
    # Domain and classification
    domains: List[str] = Field(..., description="Knowledge domains this workflow covers")
    keywords: List[str] = Field(default_factory=list, description="Keywords for workflow discovery")
    categories: List[str] = Field(default_factory=list, description="Workflow categories")
    
    # Technical requirements
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence for routing")
    max_processing_time: Optional[float] = Field(default=None, description="Maximum processing time in seconds")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    
    # Metadata
    author: Optional[str] = Field(default=None, description="Workflow author")
    created_date: Optional[str] = Field(default=None, description="Creation date")
    last_updated: Optional[str] = Field(default=None, description="Last update date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional workflow metadata")


class WorkflowRequirements(BaseModel):
    """Requirements derived from request analysis"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "required_input_types": ["natural_language_query"],
                "preferred_output_types": ["analysis_results"],
                "required_domains": ["virology"],
                "min_confidence": 0.8,
                "interaction_preference": "chat"
            }
        }
    )
    
    required_input_types: List[InputType] = Field(..., description="Required input type support")
    preferred_output_types: List[OutputType] = Field(default_factory=list, description="Preferred output types")
    required_domains: List[str] = Field(default_factory=list, description="Required domain coverage")
    optional_domains: List[str] = Field(default_factory=list, description="Optional domain coverage")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence requirement")
    interaction_preference: Optional[InteractionPattern] = Field(default=None, description="Preferred interaction pattern")
    max_processing_time: Optional[float] = Field(default=None, description="Maximum acceptable processing time")
    requirements_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional requirements metadata")


class WorkflowDiscoveryResult(BaseModel):
    """Result of workflow discovery process"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "discovery_id": "disc_123",
                "discovered_workflows": [
                    {"workflow_id": "viral_protein_analysis", "discovery_method": "auto_scan"}
                ],
                "discovery_method": "auto_scan",
                "success": True
            }
        }
    )
    
    discovery_id: str = Field(..., description="Unique discovery session identifier")
    discovered_workflows: List[WorkflowCapabilities] = Field(..., description="Discovered workflows")
    discovery_method: str = Field(..., description="Method used for discovery")
    discovery_paths: List[str] = Field(default_factory=list, description="Paths scanned for workflows")
    success: bool = Field(..., description="Whether discovery completed successfully")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Error details if unsuccessful")
    discovery_metadata: Dict[str, Any] = Field(default_factory=dict, description="Discovery session metadata")


class WorkflowValidationResult(BaseModel):
    """Result of workflow capability validation"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "viral_protein_analysis",
                "is_valid": True,
                "validation_checks": ["has_from_config", "has_natural_language_input"],
                "compliance_score": 0.95
            }
        }
    )
    
    workflow_id: str = Field(..., description="Validated workflow identifier")
    is_valid: bool = Field(..., description="Whether workflow passes validation")
    validation_checks: List[str] = Field(default_factory=list, description="Validation checks performed")
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Framework compliance score")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors found")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    validation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")


class WorkflowRegistryEntry(BaseModel):
    """Entry in the workflow registry"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "entry_id": "entry_123",
                "capabilities": {
                    "workflow_id": "viral_protein_analysis",
                    "natural_language_input": True
                },
                "validation_result": {
                    "is_valid": True,
                    "compliance_score": 0.95
                },
                "registration_method": "auto_discovery"
            }
        }
    )
    
    entry_id: str = Field(..., description="Unique registry entry identifier")
    capabilities: WorkflowCapabilities = Field(..., description="Workflow capabilities")
    validation_result: WorkflowValidationResult = Field(..., description="Validation result")
    registration_method: str = Field(..., description="Method used for registration")
    registration_timestamp: str = Field(..., description="Registration timestamp")
    last_accessed: Optional[str] = Field(default=None, description="Last access timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")
    status: str = Field(default="active", description="Registry entry status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Registry entry metadata")


class WorkflowCompatibilityScore(BaseModel):
    """Compatibility score between request requirements and workflow capabilities"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "viral_protein_analysis",
                "overall_score": 0.85,
                "component_scores": {
                    "input_compatibility": 0.9,
                    "domain_match": 0.8,
                    "output_suitability": 0.85
                }
            }
        }
    )
    
    workflow_id: str = Field(..., description="Workflow being scored")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall compatibility score")
    component_scores: Dict[str, float] = Field(..., description="Individual component scores")
    compatibility_reasons: List[str] = Field(default_factory=list, description="Reasons for compatibility score")
    incompatibility_issues: List[str] = Field(default_factory=list, description="Issues reducing compatibility")
    scoring_method: str = Field(..., description="Method used for scoring")
    scoring_metadata: Dict[str, Any] = Field(default_factory=dict, description="Scoring metadata")


class WorkflowExecutionPlan(BaseModel):
    """Plan for executing one or more workflows"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "plan_id": "plan_123",
                "execution_strategy": "sequential",
                "workflow_sequence": [
                    {"workflow_id": "viral_protein_analysis", "order": 1}
                ],
                "estimated_duration": 300.0
            }
        }
    )
    
    plan_id: str = Field(..., description="Unique execution plan identifier")
    execution_strategy: str = Field(..., description="Execution strategy (sequential, parallel, conditional)")
    workflow_sequence: List[Dict[str, Any]] = Field(..., description="Ordered workflow execution sequence")
    estimated_duration: Optional[float] = Field(default=None, description="Estimated total execution time")
    resource_allocation: Dict[str, Any] = Field(default_factory=dict, description="Resource allocation plan")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies required")
    fallback_plans: List[str] = Field(default_factory=list, description="Fallback execution plans")
    plan_metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution plan metadata") 