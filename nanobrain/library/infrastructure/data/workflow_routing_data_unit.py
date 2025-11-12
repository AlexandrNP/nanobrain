"""
Workflow Routing Data Unit - Intelligent Workflow Selection Results
==================================================================

Contains workflow routing decisions based on query classification and capability 
matching for composable intelligent routing architecture.

This component follows NanoBrain framework patterns:
- Inherits from DataUnitBase for framework compliance  
- Uses from_config pattern for component creation
- Provides comprehensive data schema validation
- Supports configuration-driven behavior

Usage:
    from nanobrain.library.infrastructure.data import WorkflowRoutingDataUnit
    
    # Create via from_config (framework pattern)
    data_unit = WorkflowRoutingDataUnit.from_config('config/workflow_routing_data_unit.yml')
    
    # Set routing results
    data_unit.set_data({
        'selected_workflow': 'viral_protein_analysis',
        'workflow_class': 'nanobrain.library.workflows.viral_protein_analysis.ViralAnalysisWorkflow',
        'routing_confidence': 0.89,
        'capability_match_score': 0.95
    })
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from nanobrain.core.data_unit import DataUnitBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentConfigurationError
from pydantic import Field, ConfigDict, validator


class WorkflowRoutingDataUnitConfig(ConfigBase):
    """
    Configuration schema for WorkflowRoutingDataUnit
    
    MANDATORY FIELDS:
    - All configuration classes MUST inherit from ConfigBase
    - MUST include comprehensive field documentation  
    - MUST use Pydantic V2 validation with ConfigDict
    """
    
    # REQUIRED FIELDS
    name: str = Field(..., description="Data unit identifier for logging and monitoring")
    
    # OPTIONAL FIELDS
    description: str = Field(
        default="Workflow routing selection results",
        description="Human-readable data unit description"
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable comprehensive logging and monitoring"
    )
    enable_validation: bool = Field(
        default=True,
        description="Enable data schema validation"
    )
    routing_strategies: List[str] = Field(
        default=["capability_based", "classification_based", "priority_based", "fallback"],
        description="Supported routing strategies"
    )
    confidence_thresholds: Dict[str, float] = Field(
        default={
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        },
        description="Confidence score thresholds for routing decisions"
    )
    capability_matching: Dict[str, Any] = Field(
        default={
            "enable_capability_scoring": True,
            "minimum_capability_match": 0.5,
            "capability_weight": 0.7,
            "classification_weight": 0.3
        },
        description="Capability matching configuration"
    )
    
    # MANDATORY PYDANTIC V2 CONFIGURATION
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=False,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "workflow_routing_data",
                    "description": "Workflow selection routing results",
                    "enable_logging": True,
                    "enable_validation": True,
                    "routing_strategies": ["capability_based", "classification_based", "fallback"],
                    "confidence_thresholds": {
                        "high_confidence": 0.8,
                        "medium_confidence": 0.6,
                        "low_confidence": 0.4
                    },
                    "capability_matching": {
                        "enable_capability_scoring": True,
                        "minimum_capability_match": 0.5,
                        "capability_weight": 0.7,
                        "classification_weight": 0.3
                    }
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "workflow_routing_data_unit",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class WorkflowRoutingDataUnit(DataUnitBase):
    """
    Workflow Routing Data Unit - Intelligent Workflow Selection Results
    ==================================================================
    
    Contains workflow routing decisions based on query classification and 
    capability matching for composable intelligent routing architecture.
    
    **Core Architecture:**
        This data unit enables clean separation between query classification
        and workflow execution by providing structured routing decisions that
        workflow orchestration systems can use for execution planning.
        
        * **Workflow Selection**: Primary and alternative workflow choices
        * **Capability Matching**: Detailed capability analysis and scoring
        * **Routing Metadata**: Confidence, strategy, and execution parameters
        * **Performance Optimization**: Caching and execution hints
    
    **Configuration Architecture:**
        ```yaml
        # Basic workflow routing data unit configuration
        name: "workflow_routing_data"
        description: "Workflow selection routing results"
        enable_logging: true
        enable_validation: true
        
        # Routing configuration
        routing_strategies:
          - "capability_based"
          - "classification_based"
          - "priority_based"
          - "fallback"
          
        confidence_thresholds:
          high_confidence: 0.8
          medium_confidence: 0.6
          low_confidence: 0.4
          
        capability_matching:
          enable_capability_scoring: true
          minimum_capability_match: 0.5
          capability_weight: 0.7
          classification_weight: 0.3
        ```
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.data import WorkflowRoutingDataUnit
        
        # Create data unit from configuration
        data_unit = WorkflowRoutingDataUnit.from_config('config/workflow_routing_data_unit.yml')
        
        # Set routing results
        routing_data = {
            'selected_workflow': 'viral_protein_analysis',
            'workflow_class': 'nanobrain.library.workflows.viral_protein_analysis.ViralAnalysisWorkflow',
            'config_path': 'config/viral_analysis_workflow.yml',
            'routing_confidence': 0.89,
            'capability_match_score': 0.95
        }
        data_unit.set_data(routing_data)
        
        # Get routing decision
        decision = data_unit.get_routing_decision()
        ```
    
    Attributes:
        name (str): Data unit identifier for logging and debugging
        description (str): Human-readable data unit description
        logger (logging.Logger): Data unit-specific logger instance
        config (WorkflowRoutingDataUnitConfig): Data unit configuration
        
    Note:
        This data unit follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.
    
    See Also:
        * :class:`DataUnitBase`: Base framework data unit interface
        * :class:`WorkflowRoutingDataUnitConfig`: Configuration schema
        * :class:`WorkflowRoutingStep`: Component that generates this data
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "workflow_routing_data_unit"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    # DATA SCHEMA DEFINITION
    data_schema = {
        # Primary routing decision
        'selected_workflow': 'string',        # Selected workflow identifier
        'workflow_class': 'string',           # Full class path for workflow
        'config_path': 'string',              # Path to workflow configuration file
        'workflow_type': 'string',            # Type of workflow ('conversational', 'analytical', etc.)
        
        # Routing confidence and scoring
        'routing_confidence': 'float',        # Overall routing decision confidence (0.0-1.0)
        'confidence_level': 'string',         # 'high', 'medium', 'low'
        'capability_match_score': 'float',    # How well workflow matches requirements (0.0-1.0)
        'classification_alignment': 'float',  # Alignment with classification results (0.0-1.0)
        
        # Routing strategy and method
        'routing_strategy': 'string',         # Strategy used ('capability_based', 'classification_based', etc.)
        'routing_method': 'string',           # 'primary_match', 'fallback', 'manual_override'
        'decision_factors': 'dict',           # Factors that influenced the decision
        'routing_timestamp': 'datetime',      # When routing decision was made
        
        # Alternative options
        'alternative_workflows': 'list',      # Alternative workflow options with scores
        'fallback_workflow': 'string',        # Fallback workflow if primary fails
        'backup_options': 'list',             # Additional backup workflow options
        
        # Execution parameters
        'execution_parameters': 'dict',       # Parameters for workflow execution
        'priority_level': 'float',            # Execution priority (0.0-1.0)
        'estimated_execution_time': 'float',  # Estimated execution time in seconds
        'resource_requirements': 'dict',      # Required computational resources
        
        # Capability analysis
        'required_capabilities': 'list',      # Capabilities required for the query
        'workflow_capabilities': 'dict',      # Capabilities provided by selected workflow
        'capability_gap_analysis': 'dict',    # Analysis of capability gaps
        'capability_coverage': 'float',       # Percentage of requirements covered (0.0-1.0)
        
        # Performance and optimization
        'supports_streaming': 'boolean',      # Whether workflow supports streaming responses
        'supports_caching': 'boolean',        # Whether workflow supports result caching
        'cache_key': 'string',               # Cache key for result caching
        'parallel_execution': 'boolean',      # Whether workflow supports parallel execution
        
        # Quality and validation
        'routing_quality_score': 'float',     # Overall quality of routing decision (0.0-1.0)
        'validation_status': 'string',        # 'validated', 'warning', 'error'
        'validation_messages': 'list',        # Validation messages and warnings
        'recommendation_strength': 'float',   # Strength of recommendation (0.0-1.0)
        
        # Context and metadata
        'session_id': 'string',              # Session identifier
        'query_context': 'dict',             # Original query context
        'routing_context': 'dict',           # Additional routing context
        'processing_time': 'float'           # Routing processing time in seconds
    }
    
    # MANDATORY FRAMEWORK METHODS
    @classmethod
    def _get_config_class(cls) -> type:
        """
        Return the configuration class for this data unit.
        
        MANDATORY IMPLEMENTATION - This is the ONLY method that differs
        between data unit types in the unified framework pattern.
        
        Returns:
            Configuration class type for this data unit
        """
        return WorkflowRoutingDataUnitConfig
    
    def _init_from_config(self, config: 'WorkflowRoutingDataUnitConfig') -> None:
        """
        Initialize data unit from validated configuration.
        
        MANDATORY IMPLEMENTATION - Data unit-specific initialization logic.
        
        Args:
            config: Validated configuration instance
            
        Raises:
            ComponentConfigurationError: If configuration is invalid
        """
        # Store configuration
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Initialize logging
        self.logger = get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            enable_logging=config.enable_logging
        )
        
        # Store routing configuration
        self.routing_strategies = config.routing_strategies
        self.confidence_thresholds = config.confidence_thresholds
        self.capability_matching = config.capability_matching
        self.enable_validation = config.enable_validation
        
        # Initialize data storage
        self._data: Dict[str, Any] = {}
        
        # Log component initialization
        self.logger.info(
            f"Initializing {self.__class__.__name__}",
            extra={
                "component_name": self.name,
                "component_type": self.COMPONENT_TYPE,
                "config_source": "from_config",
                "supported_strategies": len(self.routing_strategies)
            }
        )
        
        self.logger.info(
            f"{self.__class__.__name__} initialized successfully",
            extra={
                "component_name": self.name,
                "data_schema_fields": len(self.data_schema)
            }
        )
    
    # DATA VALIDATION METHODS
    def _validate_routing_data(self, data: Dict[str, Any]) -> None:
        """
        Validate routing data against schema and business rules.
        
        Args:
            data: Routing data to validate
            
        Raises:
            ComponentConfigurationError: If data is invalid
        """
        if not self.enable_validation:
            return
            
        # Validate required fields
        required_fields = ['selected_workflow', 'routing_confidence']
        for field in required_fields:
            if field not in data:
                raise ComponentConfigurationError(
                    f"Missing required field: {field}"
                )
        
        # Validate routing strategy
        routing_strategy = data.get('routing_strategy')
        if routing_strategy and routing_strategy not in self.routing_strategies:
            raise ComponentConfigurationError(
                f"Invalid routing strategy '{routing_strategy}'. "
                f"Must be one of: {self.routing_strategies}"
            )
        
        # Validate confidence scores
        for score_field in ['routing_confidence', 'capability_match_score', 'classification_alignment']:
            if score_field in data:
                score = data[score_field]
                if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                    raise ComponentConfigurationError(
                        f"Invalid {score_field}: {score}. Must be float between 0.0 and 1.0"
                    )
        
        # Validate capability coverage if present
        if 'capability_coverage' in data:
            coverage = data['capability_coverage']
            if not isinstance(coverage, (int, float)) or not (0.0 <= coverage <= 1.0):
                raise ComponentConfigurationError(
                    f"Invalid capability_coverage: {coverage}. Must be float between 0.0 and 1.0"
                )
        
        self.logger.debug(
            "Routing data validation passed",
            extra={
                "selected_workflow": data['selected_workflow'],
                "routing_confidence": data['routing_confidence'],
                "routing_strategy": data.get('routing_strategy', 'unknown')
            }
        )
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """
        Set routing data with validation.
        
        Args:
            data: Routing results dictionary
        """
        # Add timestamp if not provided
        if 'routing_timestamp' not in data:
            data['routing_timestamp'] = datetime.utcnow()
        
        # Add confidence level based on thresholds
        confidence = data.get('routing_confidence', 0.0)
        if confidence >= self.confidence_thresholds['high_confidence']:
            data['confidence_level'] = 'high'
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            data['confidence_level'] = 'medium'
        else:
            data['confidence_level'] = 'low'
        
        # Calculate overall routing quality score
        if 'routing_quality_score' not in data:
            routing_conf = data.get('routing_confidence', 0.0)
            capability_match = data.get('capability_match_score', 0.0)
            classification_align = data.get('classification_alignment', 0.0)
            
            # Weighted average based on capability matching configuration
            cap_weight = self.capability_matching.get('capability_weight', 0.7)
            class_weight = self.capability_matching.get('classification_weight', 0.3)
            
            data['routing_quality_score'] = (
                routing_conf * 0.4 + 
                capability_match * cap_weight * 0.6 + 
                classification_align * class_weight * 0.6
            )
        
        # Validate data
        self._validate_routing_data(data)
        
        # Store data
        self._data = data
        
        self.logger.info(
            "Routing data updated",
            extra={
                "selected_workflow": data.get('selected_workflow'),
                "routing_confidence": data.get('routing_confidence'),
                "confidence_level": data.get('confidence_level'),
                "routing_strategy": data.get('routing_strategy'),
                "quality_score": data.get('routing_quality_score')
            }
        )
    
    def get_routing_decision(self) -> Dict[str, Any]:
        """
        Get routing decision summary.
        
        Returns:
            Dictionary with routing decision information
        """
        if not self._data:
            return {}
            
        return {
            "selected_workflow": self._data.get('selected_workflow'),
            "workflow_class": self._data.get('workflow_class'),
            "config_path": self._data.get('config_path'),
            "routing_confidence": self._data.get('routing_confidence'),
            "confidence_level": self._data.get('confidence_level'),
            "capability_match_score": self._data.get('capability_match_score'),
            "supports_streaming": self._data.get('supports_streaming', False),
            "execution_parameters": self._data.get('execution_parameters', {})
        }
    
    def is_high_confidence_routing(self) -> bool:
        """Check if routing decision has high confidence."""
        confidence = self._data.get('routing_confidence', 0.0)
        return confidence >= self.confidence_thresholds['high_confidence']
    
    def get_alternative_options(self) -> List[Dict[str, Any]]:
        """
        Get alternative workflow options.
        
        Returns:
            List of alternative workflow options with scores
        """
        alternatives = self._data.get('alternative_workflows', [])
        fallback = self._data.get('fallback_workflow')
        
        # Include fallback as an option if available
        if fallback and fallback not in [alt.get('workflow') for alt in alternatives]:
            alternatives.append({
                "workflow": fallback,
                "type": "fallback",
                "confidence": 0.5,
                "reason": "fallback_option"
            })
        
        return alternatives
    
    def get_execution_context(self) -> Dict[str, Any]:
        """
        Get execution context for workflow runner.
        
        Returns:
            Dictionary with execution context information
        """
        return {
            "workflow_class": self._data.get('workflow_class'),
            "config_path": self._data.get('config_path'),
            "execution_parameters": self._data.get('execution_parameters', {}),
            "priority_level": self._data.get('priority_level', 0.5),
            "supports_streaming": self._data.get('supports_streaming', False),
            "supports_caching": self._data.get('supports_caching', False),
            "cache_key": self._data.get('cache_key'),
            "resource_requirements": self._data.get('resource_requirements', {}),
            "session_id": self._data.get('session_id')
        }
    
    def get_capability_analysis(self) -> Dict[str, Any]:
        """
        Get detailed capability analysis.
        
        Returns:
            Dictionary with capability analysis information
        """
        return {
            "required_capabilities": self._data.get('required_capabilities', []),
            "workflow_capabilities": self._data.get('workflow_capabilities', {}),
            "capability_coverage": self._data.get('capability_coverage', 0.0),
            "capability_gap_analysis": self._data.get('capability_gap_analysis', {}),
            "capability_match_score": self._data.get('capability_match_score', 0.0)
        } 