"""
Session Context Data Unit - User Session and Context Management
==============================================================

Manages user session state, conversation history, and context preservation 
across multiple interactions for composable web interface architecture.

This component follows NanoBrain framework patterns:
- Inherits from DataUnitBase for framework compliance
- Uses from_config pattern for component creation  
- Provides comprehensive data schema validation
- Supports configuration-driven behavior

Usage:
    from nanobrain.library.infrastructure.data import SessionContextDataUnit
    
    # Create via from_config (framework pattern)
    data_unit = SessionContextDataUnit.from_config('config/session_context_data_unit.yml')
    
    # Set session data
    data_unit.set_data({
        'session_id': 'session_12345',
        'user_id': 'user_abc',
        'conversation_history': [],
        'context_variables': {'domain': 'bioinformatics'}
    })
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import uuid

from nanobrain.core.data_unit import DataUnitBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.core.logging_system import get_logger
from nanobrain.core.component_base import ComponentConfigurationError
from pydantic import Field, ConfigDict


class SessionContextDataUnitConfig(ConfigBase):
    """
    Configuration schema for SessionContextDataUnit
    
    MANDATORY FIELDS:
    - All configuration classes MUST inherit from ConfigBase
    - MUST include comprehensive field documentation
    - MUST use Pydantic V2 validation with ConfigDict
    """
    
    # REQUIRED FIELDS
    name: str = Field(..., description="Data unit identifier for logging and monitoring")
    
    # OPTIONAL FIELDS
    description: str = Field(
        default="User session and context management",
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
    session_config: Dict[str, Any] = Field(
        default={
            "max_conversation_history": 100,
            "session_timeout_minutes": 60,
            "auto_cleanup_expired": True,
            "enable_context_persistence": True,
            "max_context_size_mb": 10
        },
        description="Session management configuration"
    )
    privacy_config: Dict[str, Any] = Field(
        default={
            "anonymize_user_data": False,
            "log_conversation_content": True,
            "enable_data_retention": True,
            "retention_days": 30
        },
        description="Privacy and data retention configuration"
    )
    context_tracking: Dict[str, Any] = Field(
        default={
            "track_user_preferences": True,
            "track_workflow_history": True,
            "track_performance_metrics": True,
            "enable_predictive_context": True
        },
        description="Context tracking configuration"
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
                    "name": "session_context_data",
                    "description": "User session and context management",
                    "enable_logging": True,
                    "enable_validation": True,
                    "session_config": {
                        "max_conversation_history": 100,
                        "session_timeout_minutes": 60,
                        "auto_cleanup_expired": True,
                        "enable_context_persistence": True,
                        "max_context_size_mb": 10
                    },
                    "privacy_config": {
                        "anonymize_user_data": False,
                        "log_conversation_content": True,
                        "enable_data_retention": True,
                        "retention_days": 30
                    },
                    "context_tracking": {
                        "track_user_preferences": True,
                        "track_workflow_history": True,
                        "track_performance_metrics": True,
                        "enable_predictive_context": True
                    }
                }
            ],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "component_type": "session_context_data_unit",
                "config_loading_method": "from_config_only",
                "supports_recursive_references": True
            }
        }
    )


class SessionContextDataUnit(DataUnitBase):
    """
    Session Context Data Unit - User Session and Context Management
    ==============================================================
    
    Manages user session state, conversation history, and context preservation
    across multiple interactions for composable web interface architecture.
    
    **Core Architecture:**
        This data unit enables stateful interactions across multiple requests
        by maintaining session context, conversation history, and user preferences
        in a framework-compliant manner that integrates with intelligent routing.
        
        * **Session Management**: Session creation, tracking, and lifecycle
        * **Conversation History**: Multi-turn conversation preservation
        * **Context Variables**: User preferences and domain-specific context
        * **Performance Tracking**: Session-level performance and usage metrics
    
    **Configuration Architecture:**
        ```yaml
        # Basic session context data unit configuration
        name: "session_context_data"
        description: "User session and context management"
        enable_logging: true
        enable_validation: true
        
        # Session management configuration
        session_config:
          max_conversation_history: 100
          session_timeout_minutes: 60
          auto_cleanup_expired: true
          enable_context_persistence: true
          max_context_size_mb: 10
          
        # Privacy configuration
        privacy_config:
          anonymize_user_data: false
          log_conversation_content: true
          enable_data_retention: true
          retention_days: 30
          
        # Context tracking
        context_tracking:
          track_user_preferences: true
          track_workflow_history: true
          track_performance_metrics: true
          enable_predictive_context: true
        ```
    
    **Usage Patterns:**
        ```python
        from nanobrain.library.infrastructure.data import SessionContextDataUnit
        
        # Create data unit from configuration
        data_unit = SessionContextDataUnit.from_config('config/session_context_data_unit.yml')
        
        # Create new session
        session_data = data_unit.create_new_session(user_id="user123")
        
        # Add conversation turn
        data_unit.add_conversation_turn(
            user_message="Analyze chikungunya proteins",
            system_response="Analyzing proteins...",
            workflow_used="viral_protein_analysis"
        )
        
        # Get session context
        context = data_unit.get_session_context()
        ```
    
    Attributes:
        name (str): Data unit identifier for logging and debugging
        description (str): Human-readable data unit description
        logger (logging.Logger): Data unit-specific logger instance
        config (SessionContextDataUnitConfig): Data unit configuration
        
    Note:
        This data unit follows the mandatory from_config pattern and cannot be
        instantiated directly. All configurations must be loaded from YAML files
        using the from_config method.
    
    See Also:
        * :class:`DataUnitBase`: Base framework data unit interface
        * :class:`SessionContextDataUnitConfig`: Configuration schema
        * :class:`WebInterfaceStep`: Component that uses this data
    """
    
    # MANDATORY COMPONENT METADATA
    COMPONENT_TYPE: str = "session_context_data_unit"
    REQUIRED_CONFIG_FIELDS: List[str] = ['name']
    
    # DATA SCHEMA DEFINITION
    data_schema = {
        # Session identification
        'session_id': 'string',              # Unique session identifier
        'user_id': 'string',                 # User identifier (can be anonymous)
        'user_identifier': 'string',         # Human-readable user identifier
        'session_type': 'string',            # 'interactive', 'api', 'batch'
        
        # Session lifecycle
        'session_start_time': 'datetime',    # When session was created
        'last_activity': 'datetime',         # Last user interaction timestamp
        'session_duration': 'float',         # Current session duration in seconds
        'session_status': 'string',          # 'active', 'inactive', 'expired', 'terminated'
        'expiration_time': 'datetime',       # When session will expire
        
        # Conversation management
        'conversation_history': 'list',      # List of conversation turns
        'conversation_turn_count': 'integer', # Number of conversation turns
        'last_user_message': 'string',       # Last user message
        'last_system_response': 'string',    # Last system response
        'conversation_summary': 'string',    # AI-generated conversation summary
        
        # Context variables and preferences
        'context_variables': 'dict',         # User-defined context variables
        'user_preferences': 'dict',          # User preference settings
        'domain_context': 'dict',            # Domain-specific context information
        'workflow_preferences': 'dict',      # Preferred workflows and settings
        
        # Usage and behavior tracking
        'workflow_history': 'list',          # History of workflows used
        'preferred_workflows': 'list',       # Workflows user frequently uses
        'interaction_patterns': 'dict',      # User interaction patterns
        'performance_history': 'list',       # Session performance metrics
        
        # Technical metadata  
        'user_agent': 'string',             # Browser/client user agent
        'ip_address': 'string',              # Client IP address (if tracking enabled)
        'client_info': 'dict',               # Client technical information
        'session_metadata': 'dict',          # Additional session metadata
        
        # Quality and analytics
        'user_satisfaction': 'float',        # User satisfaction score (0.0-1.0)
        'session_quality_score': 'float',    # Overall session quality (0.0-1.0)
        'error_count': 'integer',            # Number of errors in session
        'warning_count': 'integer',          # Number of warnings in session
        
        # Security and compliance
        'authentication_status': 'string',   # 'authenticated', 'anonymous', 'guest'
        'authorization_level': 'string',     # User authorization level
        'privacy_settings': 'dict',          # User privacy preferences
        'consent_given': 'dict',             # User consent for data processing
        
        # Resource usage
        'memory_usage_mb': 'float',          # Session memory usage
        'processing_time_total': 'float',    # Total processing time in seconds
        'api_calls_count': 'integer',        # Number of API calls made
        'tokens_used': 'integer'             # Total tokens used (for LLM calls)
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
        return SessionContextDataUnitConfig
    
    def _init_from_config(self, config: 'SessionContextDataUnitConfig') -> None:
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
        
        # Store session configuration
        self.session_config = config.session_config
        self.privacy_config = config.privacy_config
        self.context_tracking = config.context_tracking
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
                "max_history": self.session_config.get('max_conversation_history', 100)
            }
        )
        
        self.logger.info(
            f"{self.__class__.__name__} initialized successfully",
            extra={
                "component_name": self.name,
                "data_schema_fields": len(self.data_schema)
            }
        )
    
    # SESSION MANAGEMENT METHODS
    def create_new_session(self, user_id: Optional[str] = None, session_type: str = "interactive") -> Dict[str, Any]:
        """
        Create a new session with default values.
        
        Args:
            user_id: User identifier (generates anonymous if None)
            session_type: Type of session ('interactive', 'api', 'batch')
            
        Returns:
            Dictionary with new session data
        """
        now = datetime.now(timezone.utc)
        session_id = str(uuid.uuid4())
        
        if not user_id:
            user_id = f"anonymous_{session_id[:8]}"
        
        # Calculate expiration time
        timeout_minutes = self.session_config.get('session_timeout_minutes', 60)
        expiration_time = now.replace(microsecond=0) + datetime.timedelta(minutes=timeout_minutes)
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'user_identifier': user_id,
            'session_type': session_type,
            'session_start_time': now,
            'last_activity': now,
            'session_duration': 0.0,
            'session_status': 'active',
            'expiration_time': expiration_time,
            'conversation_history': [],
            'conversation_turn_count': 0,
            'last_user_message': '',
            'last_system_response': '',
            'conversation_summary': '',
            'context_variables': {},
            'user_preferences': {},
            'domain_context': {},
            'workflow_preferences': {},
            'workflow_history': [],
            'preferred_workflows': [],
            'interaction_patterns': {},
            'performance_history': [],
            'user_agent': '',
            'ip_address': '',
            'client_info': {},
            'session_metadata': {},
            'user_satisfaction': 0.0,
            'session_quality_score': 0.0,
            'error_count': 0,
            'warning_count': 0,
            'authentication_status': 'anonymous',
            'authorization_level': 'user',
            'privacy_settings': {},
            'consent_given': {},
            'memory_usage_mb': 0.0,
            'processing_time_total': 0.0,
            'api_calls_count': 0,
            'tokens_used': 0
        }
        
        self.set_data(session_data)
        
        self.logger.info(
            "New session created",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "session_type": session_type,
                "expiration_time": expiration_time.isoformat()
            }
        )
        
        return session_data
    
    def add_conversation_turn(self, user_message: str, system_response: str, 
                            workflow_used: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        """
        Add a conversation turn to the session history.
        
        Args:
            user_message: User's message
            system_response: System's response
            workflow_used: Workflow that generated the response
            metadata: Additional metadata for the turn
        """
        if not self._data:
            raise ComponentConfigurationError("Session not initialized. Call create_new_session first.")
        
        turn_data = {
            'timestamp': datetime.now(timezone.utc),
            'turn_number': self._data['conversation_turn_count'] + 1,
            'user_message': user_message,
            'system_response': system_response,
            'workflow_used': workflow_used,
            'metadata': metadata or {}
        }
        
        # Add to conversation history
        history = self._data.get('conversation_history', [])
        history.append(turn_data)
        
        # Trim history if it exceeds maximum
        max_history = self.session_config.get('max_conversation_history', 100)
        if len(history) > max_history:
            history = history[-max_history:]
        
        # Update session data
        self._data['conversation_history'] = history
        self._data['conversation_turn_count'] = len(history)
        self._data['last_user_message'] = user_message
        self._data['last_system_response'] = system_response
        self._data['last_activity'] = datetime.now(timezone.utc)
        
        # Update workflow history
        if workflow_used:
            workflow_history = self._data.get('workflow_history', [])
            workflow_history.append({
                'workflow': workflow_used,
                'timestamp': turn_data['timestamp'],
                'turn_number': turn_data['turn_number']
            })
            self._data['workflow_history'] = workflow_history
        
        # Update session duration
        start_time = self._data.get('session_start_time')
        if start_time:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._data['session_duration'] = duration
        
        self.logger.debug(
            "Conversation turn added",
            extra={
                "session_id": self._data.get('session_id'),
                "turn_number": turn_data['turn_number'],
                "workflow_used": workflow_used,
                "message_length": len(user_message),
                "response_length": len(system_response)
            }
        )
    
    def update_context_variable(self, key: str, value: Any) -> None:
        """
        Update a context variable.
        
        Args:
            key: Context variable key
            value: Context variable value
        """
        if not self._data:
            raise ComponentConfigurationError("Session not initialized.")
        
        context_vars = self._data.get('context_variables', {})
        context_vars[key] = value
        self._data['context_variables'] = context_vars
        self._data['last_activity'] = datetime.now(timezone.utc)
        
        self.logger.debug(
            "Context variable updated",
            extra={
                "session_id": self._data.get('session_id'),
                "variable_key": key,
                "variable_type": type(value).__name__
            }
        )
    
    def get_session_context(self) -> Dict[str, Any]:
        """
        Get current session context summary.
        
        Returns:
            Dictionary with session context information
        """
        if not self._data:
            return {}
        
        return {
            "session_id": self._data.get('session_id'),
            "user_id": self._data.get('user_id'),
            "session_status": self._data.get('session_status'),
            "conversation_turn_count": self._data.get('conversation_turn_count', 0),
            "last_activity": self._data.get('last_activity'),
            "context_variables": self._data.get('context_variables', {}),
            "user_preferences": self._data.get('user_preferences', {}),
            "domain_context": self._data.get('domain_context', {}),
            "preferred_workflows": self._data.get('preferred_workflows', []),
            "session_duration": self._data.get('session_duration', 0.0)
        }
    
    def is_session_active(self) -> bool:
        """Check if session is still active (not expired)."""
        if not self._data:
            return False
        
        expiration_time = self._data.get('expiration_time')
        if not expiration_time:
            return True
        
        return datetime.now(timezone.utc) < expiration_time
    
    def get_conversation_summary(self, max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history summary.
        
        Args:
            max_turns: Maximum number of recent turns to return
            
        Returns:
            List of conversation turns
        """
        history = self._data.get('conversation_history', [])
        
        if max_turns and len(history) > max_turns:
            history = history[-max_turns:]
        
        return history
    
    def update_performance_metrics(self, processing_time: float, tokens_used: int = 0, 
                                 memory_delta: float = 0.0) -> None:
        """
        Update session performance metrics.
        
        Args:
            processing_time: Processing time for the operation
            tokens_used: Number of tokens used
            memory_delta: Change in memory usage
        """
        if not self._data:
            return
        
        # Update totals
        self._data['processing_time_total'] = self._data.get('processing_time_total', 0.0) + processing_time
        self._data['tokens_used'] = self._data.get('tokens_used', 0) + tokens_used
        self._data['memory_usage_mb'] = self._data.get('memory_usage_mb', 0.0) + memory_delta
        self._data['api_calls_count'] = self._data.get('api_calls_count', 0) + 1
        
        # Add to performance history
        perf_entry = {
            'timestamp': datetime.now(timezone.utc),
            'processing_time': processing_time,
            'tokens_used': tokens_used,
            'memory_delta': memory_delta
        }
        
        perf_history = self._data.get('performance_history', [])
        perf_history.append(perf_entry)
        
        # Keep only recent performance entries (last 100)
        if len(perf_history) > 100:
            perf_history = perf_history[-100:]
        
        self._data['performance_history'] = perf_history 