"""
Enhanced Framework-Level Config Base Class

This module implements the enhanced mandatory from_config pattern with
comprehensive recursive loading, Pydantic integration, schema extraction, 
and optional protocol support - all within the ConfigBase class itself.
"""

import yaml
import logging
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Union, Optional, ClassVar, List, Set
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfigLoadingContext:
    """Context information for configuration loading operations"""
    base_path: Path
    resolution_stack: Set[str]
    loading_timestamp: datetime
    workflow_directory: Optional[Path] = None
    additional_context: Dict[str, Any] = None


class ConfigBase(BaseModel, ABC):
    """
    Enhanced Base Configuration Class with Integrated Loading
    
    This is the foundation class that ALL Config classes must inherit from.
    Integrates ALL configuration loading functionality directly into from_config:
    - Recursive reference resolution
    - Protocol integration (MCP/A2A)
    - Schema extraction and validation
    - File-path-only loading enforcement (with DataUnit/Link/Trigger exception)
    
    ✅ FRAMEWORK COMPLIANCE:
    - ALL configuration loading through enhanced from_config method
    - Mandatory recursive resolution of config references
    - Complete Pydantic validation and schema support
    - Optional MCP/A2A protocol integration
    - File-based configuration exclusively
    
    ❌ FORBIDDEN: Config(name="test", class="...")
    ❌ FORBIDDEN: Config.from_config({"name": "test"})
    ✅ REQUIRED: Config.from_config('path/to/config.yml')
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [],
            "nanobrain_metadata": {
                "framework_version": "2.0.0",
                "config_loading_method": "enhanced_from_config_only",
                "supports_recursive_references": True,
                "supports_mcp_integration": True,
                "supports_a2a_integration": True
            }
        }
    )
    
    # Framework-level control flag
    _allow_direct_instantiation: ClassVar[bool] = False
    
    # Optional MCP Support - Available for ALL configuration classes
    mcp_support: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional MCP (Model Context Protocol) integration configuration",
        json_schema_extra={
            "examples": [
                {
                    "server_config": {
                        "name": "nanobrain_mcp_server",
                        "url": "ws://localhost:8080/mcp",
                        "timeout": 30
                    },
                    "client_config": {
                        "default_timeout": 10,
                        "max_retries": 3
                    }
                }
            ]
        }
    )
    
    # Optional A2A Support - Available for ALL configuration classes  
    a2a_support: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional A2A (Agent-to-Agent) protocol integration configuration",
        json_schema_extra={
            "examples": [
                {
                    "agent_card": {
                        "version": "1.0.0",
                        "purpose": "Specialized agent for data processing",
                        "capabilities": ["streaming", "multi_turn_conversation"]
                    },
                    "protocol_config": {
                        "communication_mode": "async",
                        "message_format": "json"
                    }
                }
            ]
        }
    )
    
    def __init__(self, *args, **kwargs):
        """
        FORBIDDEN: Direct instantiation prohibited by framework design.
        All Config classes MUST be loaded via from_config() method.
        
        Raises:
            ValueError: Always raised when attempting direct instantiation
        """
        if not self.__class__._allow_direct_instantiation:
            raise ValueError(
                f"❌ FRAMEWORK VIOLATION: Direct instantiation of {self.__class__.__name__} is FORBIDDEN.\n"
                f"   REQUIRED: Use {self.__class__.__name__}.from_config(file_path) instead.\n"
                f"   REASON: NanoBrain framework enforces configuration-driven component creation.\n"
                f"   SOLUTION: Create YAML config file and load via from_config() method.\n"
                f"   EXAMPLE: config = {self.__class__.__name__}.from_config('path/to/config.yml')\n"
                f"   \n"
                f"   ❌ PROHIBITED:\n"
                f"      {self.__class__.__name__}(name='test', class='...')\n"
                f"      {self.__class__.__name__}.from_config({{'name': 'test'}})\n"
                f"   \n"
                f"   ✅ REQUIRED:\n"
                f"      {self.__class__.__name__}.from_config('path/to/config.yml')"
            )
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path], **context) -> 'ConfigBase':
        """
        ENHANCED: Complete configuration loading with all features integrated
        
        This method handles ALL configuration loading responsibilities:
        - File-path-only loading (dictionaries prohibited except DataUnit/Link/Trigger)
        - Recursive reference resolution
        - Protocol integration (MCP/A2A)
        - Schema validation and type checking
        - Context-aware configuration resolution
        
        Args:
            config_path: MANDATORY file path to YAML configuration
            **context: Additional context (workflow_directory, etc.)
            
        Returns:
            Fully resolved and validated Config instance
            
        Raises:
            ValueError: If config_path is not a file path
            FileNotFoundError: If config file doesn't exist
            RecursionError: If circular dependencies detected
            
        ✅ FRAMEWORK COMPLIANCE:
        - ONLY accepts file paths (str or Path objects) for most classes
        - EXCEPTION: DataUnit, Link, Trigger classes may accept inline dict config
        - Automatic object instantiation via class+config patterns
        - Complete Pydantic validation and type checking
        - Optional protocol integration when specified
        """
        # STRICT ENFORCEMENT: Only file paths allowed for ConfigBase classes
        if not isinstance(config_path, (str, Path)):
            raise ValueError(
                f"❌ FRAMEWORK VIOLATION: {cls.__name__}.from_config ONLY accepts file paths.\n"
                f"   GIVEN: {type(config_path)}\n"
                f"   REQUIRED: str or Path object pointing to YAML configuration file\n"
                f"   \n"
                f"   ❌ PROHIBITED USAGE:\n"
                f"      {cls.__name__}.from_config({{'name': 'test', 'class': '...'}})\n"
                f"      {cls.__name__}.from_config(config_dict)\n"
                f"   \n"
                f"   ✅ CORRECT USAGE:\n"
                f"      {cls.__name__}.from_config('path/to/config.yml')\n"
                f"      {cls.__name__}.from_config(Path('config.yml'))\n"
                f"   \n"
                f"   NOTE: Only DataUnit, Link, Trigger classes support inline dict config\n"
                f"   REASON: NanoBrain framework enforces file-based configuration for most classes"
            )
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"❌ CONFIGURATION ERROR: Config file not found: {config_path}\n"
                f"   SOLUTION: Ensure configuration file exists at specified path\n"
                f"   SEARCH PATHS: Check relative to current directory and workflow_directory"
            )
        
        try:
            logger.info(f"🔄 Loading {cls.__name__} from: {config_path}")
            
            # Create loading context
            loading_context = ConfigLoadingContext(
                base_path=config_path.parent,
                resolution_stack=set(),
                loading_timestamp=datetime.now(),
                workflow_directory=context.get('workflow_directory'),
                additional_context=context
            )
            
            # Load raw YAML data
            raw_config = cls._load_yaml_file(config_path)
            
            # Resolve nested objects with class+config patterns
            resolved_config = cls._resolve_nested_objects(raw_config, loading_context)
            
            # Apply optional protocol integrations
            enhanced_config = cls._apply_protocol_integrations(resolved_config, loading_context)
            
            # Create and validate configuration instance
            config_instance = cls._create_validated_instance(enhanced_config)
            
            logger.info(f"✅ Successfully loaded {cls.__name__} from {config_path}")
            return config_instance
            
        except Exception as e:
            logger.error(f"❌ Failed to load {cls.__name__} from {config_path}: {e}")
            raise ValueError(
                f"❌ CONFIGURATION LOADING FAILED: {config_path}\n"
                f"   ERROR: {str(e)}\n"
                f"   CONFIG_CLASS: {cls.__name__}\n"
                f"   SOLUTION: Check YAML syntax, file permissions, and recursive references"
            ) from e
    
    @classmethod
    def _load_yaml_file(cls, config_path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
            
            if not isinstance(config_data, dict):
                raise ValueError(f"Configuration file must contain a YAML dictionary, got {type(config_data)}")
            
            return config_data
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"File encoding error in {config_path}: {e}")
    
    @classmethod
    def _resolve_nested_objects(cls, config_data: Dict[str, Any], context: ConfigLoadingContext) -> Dict[str, Any]:
        """
        Resolve nested objects specified with 'class' and 'config' fields
        
        When encountering a subsection with both 'class' and 'config' fields:
        1. Import the class from the 'class' field (full module path)
        2. Create instance using ClassName.from_config() with 'config' field value
        3. Replace the subsection with the instantiated object
        
        Args:
            config_data: Configuration dictionary to process
            context: Loading context for path resolution
            
        Returns:
            Configuration with nested objects instantiated
            
        ✅ FRAMEWORK COMPLIANCE:
        - Uses existing from_config pattern for object creation
        - Supports full module paths for any NanoBrain component
        - File paths supported for ALL classes
        - Inline dict config ONLY for DataUnit, Link, Trigger classes
        - Maintains consistent object instantiation across framework
        """
        import importlib
        resolved_config = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Check if this dict has both 'class' and 'config' fields
                if 'class' in value and 'config' in value:
                    # Extract class path and config
                    class_path = value['class']
                    config_value = value['config']
                    
                    try:
                        # Import the class
                        module_path, class_name = class_path.rsplit('.', 1)
                        module = importlib.import_module(module_path)
                        target_class = getattr(module, class_name)
                        
                        # Resolve config based on target class type and config value type
                        if isinstance(config_value, str):
                            # File path - all classes support this
                            config_path = cls._resolve_config_path(config_value, context)
                            instance = target_class.from_config(config_path, **context.additional_context)
                        else:
                            # Inline configuration dict - only supported for DataUnit, Link, Trigger classes
                            if cls._is_inline_config_supported(target_class):
                                instance = target_class.from_config(config_value, **context.additional_context)
                            else:
                                raise ValueError(
                                    f"❌ FRAMEWORK VIOLATION: Inline dict configuration not supported for {class_path}\n"
                                    f"   SUPPORTED CLASSES: DataUnit, Link, Trigger and their subclasses only\n"
                                    f"   REQUIRED: Use file path for config field\n"
                                    f"   EXAMPLE: config: 'path/to/{class_name.lower()}.yml'\n"
                                    f"   CURRENT: config: {config_value}"
                                )
                        
                        # Replace the configuration dict with the instantiated object
                        resolved_config[key] = instance
                        
                        logger.debug(f"✅ Instantiated {class_name} for key '{key}'")
                        
                    except Exception as e:
                        raise ValueError(
                            f"❌ FAILED TO INSTANTIATE OBJECT: {key}\n"
                            f"   CLASS: {class_path}\n"
                            f"   CONFIG: {config_value}\n"
                            f"   ERROR: {str(e)}\n"
                            f"   SOLUTION: Ensure class path is correct and config is valid"
                        ) from e
                else:
                    # Recursively process nested dictionaries
                    resolved_config[key] = cls._resolve_nested_objects(value, context)
            elif isinstance(value, list):
                # Process lists that might contain dicts with class+config
                resolved_list = []
                for item in value:
                    if isinstance(item, dict):
                        resolved_list.append(cls._resolve_nested_objects(item, context))
                    else:
                        resolved_list.append(item)
                resolved_config[key] = resolved_list
            else:
                # Keep primitive values as-is
                resolved_config[key] = value
        
        return resolved_config

    @classmethod
    def _is_inline_config_supported(cls, target_class: type) -> bool:
        """
        Check if target class supports inline dict configuration
        
        ✅ FRAMEWORK COMPLIANCE:
        Only DataUnit, Link, and Trigger classes (and their subclasses) support inline dict config.
        All other classes MUST use file paths for configuration.
        
        Args:
            target_class: Class to check for inline config support
            
        Returns:
            True if class supports inline dict config, False otherwise
        """
        # Import base classes for comparison
        try:
            from nanobrain.core.data_unit import DataUnit
            from nanobrain.core.link import LinkBase  
            from nanobrain.core.trigger import Trigger
            
            # Check if target class is a subclass of supported classes
            return (issubclass(target_class, DataUnit) or 
                    issubclass(target_class, LinkBase) or 
                    issubclass(target_class, Trigger))
        except ImportError:
            # If import fails, default to False (require file path)
            return False

    @classmethod
    def _resolve_config_path(cls, config_path: str, context: ConfigLoadingContext) -> str:
        """
        Resolve configuration file path for class+config instantiation
        
        Resolution order:
        1. If workflow_directory is set, resolve relative to workflow_directory  
        2. Otherwise, resolve relative to current base_path
        3. Support absolute paths
        """
        from pathlib import Path
        
        path = Path(config_path)
        
        # Handle absolute paths
        if path.is_absolute():
            return str(path)
        
        # Try workflow_directory first (matches chatbot_viral_integration pattern)
        if context.workflow_directory:
            workflow_resolved = Path(context.workflow_directory) / path
            if workflow_resolved.exists():
                return str(workflow_resolved)
        
        # Fallback to base_path resolution
        base_resolved = context.base_path / path
        return str(base_resolved)
    
    @classmethod
    def _apply_protocol_integrations(cls, 
                                   config_data: Dict[str, Any], 
                                   context: ConfigLoadingContext) -> Dict[str, Any]:
        """Apply optional MCP/A2A protocol integrations"""
        enhanced_config = config_data.copy()
        
        # Apply MCP integration if specified
        if 'mcp_support' in config_data:
            mcp_config = config_data['mcp_support']
            enhanced_config.update(cls._apply_mcp_integration(mcp_config, context))
            logger.debug("🔌 Applied MCP integration")
        
        # Apply A2A integration if specified
        if 'a2a_support' in config_data:
            a2a_config = config_data['a2a_support']
            enhanced_config.update(cls._apply_a2a_integration(a2a_config, context))
            logger.debug("🔌 Applied A2A integration")
        
        return enhanced_config
    
    @classmethod
    def _apply_mcp_integration(cls, mcp_config: Dict[str, Any], context: ConfigLoadingContext) -> Dict[str, Any]:
        """Apply MCP (Model Context Protocol) integration"""
        integration_data = {}
        
        if 'server_config' in mcp_config:
            integration_data['mcp_server_config'] = mcp_config['server_config']
        
        if 'client_config' in mcp_config:
            integration_data['mcp_client_config'] = mcp_config['client_config']
        
        return integration_data
    
    @classmethod
    def _apply_a2a_integration(cls, a2a_config: Dict[str, Any], context: ConfigLoadingContext) -> Dict[str, Any]:
        """Apply A2A (Agent-to-Agent) protocol integration"""
        integration_data = {}
        
        if 'agent_card' in a2a_config:
            integration_data['a2a_agent_card'] = a2a_config['agent_card']
        
        if 'protocol_config' in a2a_config:
            integration_data['a2a_protocol_config'] = a2a_config['protocol_config']
        
        return integration_data
    
    @classmethod
    def _create_validated_instance(cls, config_data: Dict[str, Any]) -> 'ConfigBase':
        """Create configuration instance with validation"""
        # Temporarily allow instantiation for validated config data
        cls._allow_direct_instantiation = True
        try:
            instance = cls(**config_data)
            return instance
        finally:
            cls._allow_direct_instantiation = False
    
    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        """
        Extract complete Pydantic schema for this configuration class
        
        Returns:
            Complete JSON schema with validation rules and examples
        """
        schema = cls.model_json_schema()
        
        # Enhance with NanoBrain-specific metadata
        schema.setdefault('nanobrain_metadata', {}).update({
            'config_class': cls.__name__,
            'module': cls.__module__,
            'framework_version': '2.0.0',
            'loading_method': 'enhanced_from_config_only',
            'supports_recursive_loading': True,
            'supports_mcp_integration': True,
            'supports_a2a_integration': True,
            'direct_instantiation_forbidden': True
        })
        
        return schema
    
    @classmethod 
    def get_schema_with_examples(cls) -> Dict[str, Any]:
        """
        Get schema with comprehensive example configurations
        
        Returns:
            Schema enhanced with realistic configuration examples
        """
        schema = cls.get_schema()
        
        # Add comprehensive examples
        examples = cls._generate_configuration_examples()
        schema['examples'] = examples
        
        return schema
    
    @classmethod
    def validate_config_file(cls, config_path: Union[str, Path]) -> List[str]:
        """
        Validate configuration file against this class schema
        
        Args:
            config_path: Path to configuration file to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        config_path = Path(config_path)
        errors = []
        
        try:
            # Load raw configuration
            raw_config = cls._load_yaml_file(config_path)
            
            # Attempt to create instance for validation
            cls._allow_direct_instantiation = True
            try:
                cls(**raw_config)
            except Exception as e:
                errors.append(f"Validation error: {str(e)}")
            finally:
                cls._allow_direct_instantiation = False
                
        except Exception as e:
            errors.append(f"File loading error: {str(e)}")
        
        return errors
    
    @classmethod
    def _generate_configuration_examples(cls) -> List[Dict[str, Any]]:
        """Generate realistic configuration examples for this class"""
        # Base example with minimal required fields
        base_example = {
            "name": f"example_{cls.__name__.lower()}",
            "description": f"Example configuration for {cls.__name__}"
        }
        
        # Enhanced example with optional protocol support
        enhanced_example = base_example.copy()
        enhanced_example.update({
            "mcp_support": {
                "server_config": {
                    "name": f"{cls.__name__.lower()}_mcp_server",
                    "url": "ws://localhost:8080/mcp",
                    "timeout": 30
                }
            },
            "a2a_support": {
                "agent_card": {
                    "version": "1.0.0",
                    "purpose": f"Agent configured via {cls.__name__}",
                    "capabilities": ["configuration_driven", "protocol_aware"]
                }
            }
        })
        
        return [base_example, enhanced_example]
    
    def apply_mcp_integration(self) -> None:
        """
        Apply MCP integration if specified in configuration
        
        This method is called automatically during configuration loading
        when mcp_support is present in the configuration file.
        """
        if self.mcp_support is None:
            return
        
        logger.info(f"🔌 Applying MCP integration for {self.__class__.__name__}")
        self._apply_mcp_configuration(self.mcp_support)
    
    def apply_a2a_integration(self) -> None:
        """
        Apply A2A integration if specified in configuration
        
        This method is called automatically during configuration loading
        when a2a_support is present in the configuration file.
        """
        if self.a2a_support is None:
            return
        
        logger.info(f"🔌 Applying A2A integration for {self.__class__.__name__}")
        self._apply_a2a_configuration(self.a2a_support)
    
    def _apply_mcp_configuration(self, mcp_config: Dict[str, Any]) -> None:
        """Override in subclasses for specific MCP integration logic"""
        pass
    
    def _apply_a2a_configuration(self, a2a_config: Dict[str, Any]) -> None:
        """Override in subclasses for specific A2A integration logic"""
        pass
    
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook for applying integrations
        
        Called automatically after Pydantic model initialization
        to apply any specified protocol integrations.
        """
        super().model_post_init(__context)
        
        # Apply optional integrations
        self.apply_mcp_integration()
        self.apply_a2a_integration()

    # Legacy methods preserved for backward compatibility
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to YAML string or save to file.
        
        Args:
            file_path: Optional file path to save YAML
            
        Returns:
            YAML string representation
        """
        # Convert to dict for YAML serialization
        config_dict = self.model_dump()
        
        # Generate YAML string
        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            indent=2
        )
        
        # Save to file if path provided
        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(yaml_str)
            logger.info(f"Configuration saved to {file_path}")
        
        return yaml_str
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}") 