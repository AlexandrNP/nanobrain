"""
Query Classification Data Unit - LLM-based Query Analysis Results
================================================================

Stores results of intelligent query classification including workflow routing 
decisions, confidence scores, and capability matching for composable intelligent
routing architecture.

This component follows NanoBrain framework patterns:
- Inherits from DataUnitBase for framework compliance
- Uses from_config pattern for component creation
- Provides comprehensive data schema validation
- Supports configuration-driven behavior

Usage:
    from nanobrain.library.infrastructure.data import QueryClassificationDataUnit
    
    # Create via from_config (framework pattern)
    data_unit = QueryClassificationDataUnit.from_config('config/query_classification_data_unit.yml')
    
    # Set classification results
    data_unit.set_data({
        'query': 'Analyze chikungunya virus proteins',
        'classification': 'analytical',
        'confidence': 0.92,
        'domain_classification': ['bioinformatics', 'viral_analysis'],
        'suggested_workflow': 'viral_protein_analysis'
    })
"""

from typing import Any, Dict, List
from datetime import datetime

from nanobrain.core.data_unit import DataUnitBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentConfigurationError
from pydantic import Field, ConfigDict


class QueryClassificationDataUnitConfig(ConfigBase):
    """
    Configuration schema for QueryClassificationDataUnit
    
    MANDATORY FIELDS:
    - All configuration classes MUST inherit from ConfigBase
    - MUST include comprehensive field documentation
    - MUST use Pydantic V2 validation with ConfigDict
    """
    
    # REQUIRED FIELDS
    name: str = Field(..., description="Data unit identifier for logging and monitoring")
    
    # OPTIONAL FIELDS
    description: str = Field(
        default="Query classification analysis results",
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
    classification_domains: List[str] = Field(
        default=["conversational", "analytical", "mixed", "unknown"],
        description="Supported classification domains"
    )
    confidence_thresholds: Dict[str, float] = Field(
        default={
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        },
        description="Confidence score thresholds for decision making"
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
                    "name": "query_classification_data",
                    "description": "LLM-based query classification results",
                    "enable_logging": True,
                    "enable_validation": True,
                    "classification_domains": ["conversational", "analytical", "mixed"],
                    "confidence_thresholds": {
                        "high_confidence": 0.8,
                        "medium_confidence": 0.6,
                        "low_confidence": 0.4
                    }
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "query_classification_data_unit",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class QueryClassificationDataUnit(DataUnitBase):
    """
    Query Classification Data Unit - LLM-based Query Analysis Results
    ================================================================
    
    Stores results of intelligent query classification including workflow 
    routing decisions, confidence scores, and capability matching for 
    composable intelligent routing architecture.
    
    **Core Architecture:**
        This data unit enables clean separation between HTTP interface and 
        intelligent routing by providing structured classification results
        that downstream routing steps can use for decision making.
        
        * **Query Analysis**: Original query text and metadata
        * **Classification Results**: Domain classification with confidence
        * **Routing Metadata**: Suggested workflows and capabilities
        * **Context Preservation**: Session and temporal context data
    
    **Configuration Architecture:**
        ```yaml
        # Basic query classification data unit configuration
        name: "query_classification_data"
        description: "LLM-based query classification results"
        enable_logging: true
        enable_validation: true
        
        # Classification configuration
        classification_domains:
          - "conversational"
          - "analytical"
          - "mixed"
          - "unknown"
          
        confidence_thresholds:
          high_confidence: 0.8
          medium_confidence: 0.6
          low_confidence: 0.4
        ```
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.data import QueryClassificationDataUnit
        
        # Create data unit from configuration
        data_unit = QueryClassificationDataUnit.from_config('config/query_classification_data_unit.yml')
        
        # Set classification results
        classification_data = {
            'query': 'Analyze chikungunya virus proteins',
            'classification': 'analytical', 
            'confidence': 0.92,
            'domain_classification': ['bioinformatics', 'viral_analysis'],
            'suggested_workflow': 'viral_protein_analysis'
        }
        data_unit.set_data(classification_data)
        
        # Retrieve classification results
        results = data_unit.get_data()
        ```
    
    Attributes:
        name (str): Data unit identifier for logging and debugging
        description (str): Human-readable data unit description
        logger (logging.Logger): Data unit-specific logger instance
        config (QueryClassificationDataUnitConfig): Data unit configuration
        
    Note:
        This data unit follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.
    
    See Also:
        * :class:`DataUnitBase`: Base framework data unit interface
        * :class:`QueryClassificationDataUnitConfig`: Configuration schema
        * :class:`QueryClassificationStep`: Component that generates this data
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "query_classification_data_unit"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    # DATA SCHEMA DEFINITION
    data_schema = {
        # Original query information
        'query': 'string',                    # Original user query text
        'query_metadata': 'dict',             # Query parsing metadata
        'session_id': 'string',               # Session identifier
        'timestamp': 'datetime',              # Classification timestamp
        
        # Classification results
        'classification': 'string',           # Primary classification ('conversational', 'analytical', 'mixed', 'unknown')
        'confidence': 'float',                # Classification confidence (0.0-1.0)
        'confidence_level': 'string',         # 'high', 'medium', 'low'
        'classification_method': 'string',    # 'llm_analysis', 'keyword_match', 'pattern_recognition'
        
        # Detailed analysis
        'domain_classification': 'list',      # List of detected domains
        'matched_keywords': 'list',           # Keywords that influenced classification
        'extracted_entities': 'dict',         # Named entities extracted from query
        'intent_analysis': 'dict',            # Intent detection results
        
        # Routing recommendations
        'suggested_workflow': 'string',       # Primary workflow recommendation
        'alternative_workflows': 'list',      # Alternative workflow options
        'workflow_capabilities_required': 'dict',  # Required capabilities for routing
        'routing_priority': 'float',          # Priority score for routing (0.0-1.0)
        
        # Performance and context
        'processing_time': 'float',           # Classification processing time (seconds)
        'model_used': 'string',               # LLM model used for classification
        'supports_streaming': 'boolean',      # Whether classified workflow supports streaming
        'estimated_completion_time': 'float', # Estimated workflow completion time
        
        # Quality metrics
        'classification_quality_score': 'float',  # Overall quality assessment
        'ambiguity_score': 'float',               # Query ambiguity level (0.0-1.0)
        'requires_clarification': 'boolean',      # Whether query needs user clarification
        'clarification_questions': 'list'         # Suggested clarification questions
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
        return QueryClassificationDataUnitConfig
    
    def _init_from_config(self, config: 'QueryClassificationDataUnitConfig') -> None:
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
        
        # Store classification configuration
        self.classification_domains = config.classification_domains
        self.confidence_thresholds = config.confidence_thresholds
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
                "supported_domains": len(self.classification_domains)
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
    def _validate_classification_data(self, data: Dict[str, Any]) -> None:
        """
        Validate classification data against schema and business rules.
        
        Args:
            data: Classification data to validate
            
        Raises:
            ComponentConfigurationError: If data is invalid
        """
        if not self.enable_validation:
            return
            
        # Validate required fields
        required_fields = ['query', 'classification', 'confidence']
        for field in required_fields:
            if field not in data:
                raise ComponentConfigurationError(
                    f"Missing required field: {field}"
                )
        
        # Validate classification domain
        if data['classification'] not in self.classification_domains:
            raise ComponentConfigurationError(
                f"Invalid classification '{data['classification']}'. "
                f"Must be one of: {self.classification_domains}"
            )
        
        # Validate confidence score
        confidence = data.get('confidence', 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ComponentConfigurationError(
                f"Invalid confidence score: {confidence}. Must be float between 0.0 and 1.0"
            )
        
        self.logger.debug(
            "Classification data validation passed",
            extra={
                "classification": data['classification'],
                "confidence": confidence,
                "query_length": len(data.get('query', ''))
            }
        )
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """
        Set classification data with validation.
        
        Args:
            data: Classification results dictionary
        """
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow()
        
        # Add confidence level based on thresholds
        confidence = data.get('confidence', 0.0)
        if confidence >= self.confidence_thresholds['high_confidence']:
            data['confidence_level'] = 'high'
        elif confidence >= self.confidence_thresholds['medium_confidence']:
            data['confidence_level'] = 'medium'
        else:
            data['confidence_level'] = 'low'
        
        # Validate data
        self._validate_classification_data(data)
        
        # Store data
        self._data = data
        
        self.logger.info(
            "Classification data updated",
            extra={
                "classification": data.get('classification'),
                "confidence": data.get('confidence'),
                "confidence_level": data.get('confidence_level'),
                "suggested_workflow": data.get('suggested_workflow')
            }
        )
    
    def get_classification_summary(self) -> Dict[str, Any]:
        """
        Get summary of classification results.
        
        Returns:
            Dictionary with classification summary
        """
        if not self._data:
            return {}
            
        return {
            "classification": self._data.get('classification'),
            "confidence": self._data.get('confidence'),
            "confidence_level": self._data.get('confidence_level'),
            "suggested_workflow": self._data.get('suggested_workflow'),
            "supports_streaming": self._data.get('supports_streaming', False),
            "requires_clarification": self._data.get('requires_clarification', False)
        }
    
    def is_high_confidence(self) -> bool:
        """Check if classification has high confidence."""
        confidence = self._data.get('confidence', 0.0)
        return confidence >= self.confidence_thresholds['high_confidence']
    
    def get_routing_recommendation(self) -> Dict[str, Any]:
        """
        Get routing recommendation for workflow selection.
        
        Returns:
            Dictionary with routing information
        """
        return {
            "suggested_workflow": self._data.get('suggested_workflow'),
            "alternative_workflows": self._data.get('alternative_workflows', []),
            "workflow_capabilities_required": self._data.get('workflow_capabilities_required', {}),
            "routing_priority": self._data.get('routing_priority', 0.5),
            "confidence": self._data.get('confidence', 0.0)
        } 