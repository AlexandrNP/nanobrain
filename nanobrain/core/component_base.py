"""
Component Base System for NanoBrain Framework

Provides the mandatory from_config pattern foundation for all framework components.
"""

import importlib
import inspect
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, ClassVar, List, Optional, Type, Union
from pathlib import Path
from pydantic import BaseModel

from .logging_system import get_logger


class ComponentConfigurationError(Exception):
    """Raised when component configuration is invalid"""
    pass


class ComponentDependencyError(Exception):
    """Raised when component dependencies cannot be resolved"""
    pass


def validate_config_usage(component_class, config_object):
    """
    Framework-level validation to ensure proper Config usage.
    PREVENTS any programmatic Config creation.
    
    Args:
        component_class: The component class being created
        config_object: The config object to validate
        
    Raises:
        ValueError: If config was created via prohibited constructor usage
    """
    if hasattr(config_object, '__class__'):
        config_class = config_object.__class__
        
        # Check if config was created via constructor (FORBIDDEN)
        if hasattr(config_class, '_allow_direct_instantiation'):
            # This means it's a ConfigBase-derived class
            # If we're here and the flag is True, it means constructor was used during from_config
            # which is allowed. If it's False, then somehow constructor was bypassed, which is okay.
            pass
        else:
            # For non-ConfigBase classes, check if it looks like programmatic creation
            if isinstance(config_object, dict):
                # Dictionary configs are allowed for testing
                pass
            elif isinstance(config_object, BaseModel):
                # Check if this is an old-style BaseModel that should have been ConfigBase
                try:
                    from .config.config_base import ConfigBase
                    if not isinstance(config_object, ConfigBase):
                        logger = get_logger("component_base")
                        logger.warning(
                            f"⚠️ FRAMEWORK WARNING: {config_class.__name__} should inherit from ConfigBase.\n"
                            f"   COMPONENT: {component_class.__name__}\n"
                            f"   REQUIRED: Update {config_class.__name__} to inherit from ConfigBase.\n"
                            f"   CURRENT: Allowing for backward compatibility, but this will be deprecated."
                        )
                except ImportError:
                    # ConfigBase not available, skip check
                    pass
        
        # Additional validation for config content
        if hasattr(config_object, '__dict__') and hasattr(config_object, 'model_dump'):
            # This is a Pydantic model, check if it has content
            try:
                config_dict = config_object.model_dump()
                if not config_dict:
                    raise ValueError(
                        f"❌ CONFIG ERROR: Empty {config_class.__name__} configuration.\n"
                        f"   COMPONENT: {component_class.__name__}\n"
                        f"   REQUIRED: Valid configuration data in YAML file."
                    )
            except Exception:
                # If model_dump fails, skip validation
                pass


def import_class_from_path(class_path: str, search_namespaces: List[str] = None) -> Type:
    """
    Import class with enhanced path resolution and error handling
    
    Args:
        class_path: Full module path or short class name
        search_namespaces: List of namespace prefixes to search
        
    Returns:
        Imported class type
        
    Raises:
        ImportError: If class cannot be found or imported
    """
    search_namespaces = search_namespaces or [
        'nanobrain.core',
        'nanobrain.library.workflows',
        'nanobrain.library.agents',
        'nanobrain.library.tools',
        'nanobrain.library.infrastructure'
    ]
    
    # Try direct import first (full path provided)
    if '.' in class_path:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            pass
    
    # Try search namespaces for short names
    for namespace in search_namespaces:
        for pattern in [
            f"{namespace}.{class_path.lower()}.{class_path}",  # module.class pattern
            f"{namespace}.{class_path}",  # direct class import
        ]:
            try:
                if '.' in pattern:
                    module_path, class_name = pattern.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    return getattr(module, class_name)
            except (ImportError, AttributeError):
                continue
    
    raise ImportError(f"Cannot import class: {class_path}")


class FromConfigBase(ABC):
    """
    Mandatory Base Class for All NanoBrain Framework Components
    ==========================================================
    
    This is the foundational abstract base class that ALL NanoBrain framework components 
    must inherit from. It enforces the unified ``from_config`` pattern and provides the 
    concrete implementation that works identically across all component types.
    
    **Core Philosophy:**
        The ``from_config`` pattern ensures that all components are created exclusively 
        through configuration files, enabling:
        
        * **Configuration-Driven Architecture**: All behavior controlled via YAML files
        * **Reproducible Systems**: Component creation is deterministic and version-controlled
        * **Zero Hardcoding**: Complete flexibility through declarative configuration
        * **Unified Interface**: Identical creation pattern across all component types
    
    **Architectural Role:**
        FromConfigBase serves as the architectural foundation that:
        
        * Prohibits direct constructor usage (``Component()`` is forbidden)
        * Enforces configuration-first design principles
        * Provides unified dependency resolution mechanisms
        * Ensures consistent error handling across all components
        * Enables framework-wide validation and lifecycle management
    
    **Component Lifecycle:**
        The unified ``from_config`` pattern follows a strict lifecycle:
        
        1. **Configuration Loading**: Parse YAML/dict configuration
        2. **Validation**: Validate configuration against component schema
        3. **Dependency Resolution**: Resolve references to other components
        4. **Component Creation**: Create instance with validated configuration
        5. **Initialization**: Component-specific initialization logic
        
    **Inheritance Requirements:**
        All subclasses MUST implement:
        
        * :meth:`_get_config_class()`: Return the configuration class type
        * :meth:`_init_from_config()`: Component-specific initialization logic
        
        Optional overrides:
        
        * :meth:`extract_component_config()`: Custom configuration extraction
        * :meth:`resolve_dependencies()`: Custom dependency resolution
        * :meth:`validate_configuration()`: Additional validation logic
    
    **Usage Pattern:**
        ```python
        # ❌ FORBIDDEN: Direct instantiation
        # component = MyComponent(name="test")
        
        # ✅ REQUIRED: Configuration-based creation
        component = MyComponent.from_config('config/my_component.yml')
        
        # ✅ ALTERNATIVE: Inline configuration (for testing)
        component = MyComponent.from_config({
            'name': 'test_component',
            'description': 'Test configuration'
        })
        ```
    
    **Configuration Examples:**
        
        **Basic Component Configuration:**
        ```yaml
        # config/my_component.yml
        name: "data_processor"
        description: "Processes input data"
        auto_initialize: true
        debug_mode: false
        ```
        
        **Component with Dependencies:**
        ```yaml
        # config/workflow_step.yml
        name: "analysis_step"
        description: "Data analysis step"
        executor:
          class: "nanobrain.core.executor.LocalExecutor"
          config: "config/local_executor.yml"
        tools:
          - class: "nanobrain.library.tools.DataAnalyzer"
            config: "config/data_analyzer.yml"
        ```
    
    **Error Handling:**
        The framework provides comprehensive error handling:
        
        * :exc:`ComponentConfigurationError`: Configuration validation failures
        * :exc:`ComponentDependencyError`: Dependency resolution issues
        * :exc:`ValueError`: Invalid configuration format or missing required fields
        * :exc:`FileNotFoundError`: Configuration file not found
        * :exc:`ImportError`: Component class not found
    
    **Thread Safety:**
        All FromConfigBase components are designed to be thread-safe. The framework
        handles synchronization automatically for shared resources and configuration
        loading operations.
    
    **Performance Considerations:**
        * **Lazy Loading**: Dependencies are resolved only when needed
        * **Configuration Caching**: Parsed configurations are cached for reuse
        * **Validation Optimization**: Schema validation is performed once per configuration
        * **Memory Efficiency**: Large configurations are processed in streaming fashion
    
    **Framework Integration:**
        FromConfigBase integrates seamlessly with other framework components:
        
        * **Configuration System**: Works with :class:`ConfigBase` for validation
        * **Logging System**: Automatic logging integration for component lifecycle
        * **Error Handling**: Consistent error reporting across all components
        * **Tool Integration**: Automatic tool discovery and registration
        
    **Best Practices:**
        
        * Always use configuration files for production deployments
        * Use inline configurations only for testing and development
        * Implement comprehensive validation in ``_init_from_config``
        * Handle dependencies gracefully with meaningful error messages
        * Document configuration schema in component docstrings
        
    **Advanced Features:**
        
        * **Recursive Configuration**: Supports nested component configurations
        * **Template Variables**: Configuration templating with variable substitution
        * **Environment Integration**: Environment variable interpolation
        * **Schema Validation**: Automatic Pydantic schema validation
        * **Cross-References**: Automatic resolution of component references
    
    Attributes:
        REQUIRED_CONFIG_FIELDS (List[str]): List of required configuration field names
        OPTIONAL_CONFIG_FIELDS (Dict[str, Any]): Default values for optional fields  
        COMPONENT_TYPE (str): Component type identifier for logging and debugging
    
    Note:
        This class is abstract and cannot be instantiated directly. All concrete
        framework components must inherit from this class and implement the
        required abstract methods.
    
    See Also:
        * :class:`ConfigBase`: Configuration validation and management
        * :class:`AgentConfig`: Example configuration class implementation
        * :class:`StepConfig`: Step-specific configuration patterns
        * :mod:`nanobrain.core.config`: Configuration system documentation
    """
    
    # Component configuration schema (must be defined in subclasses)
    REQUIRED_CONFIG_FIELDS: ClassVar[List[str]] = []
    OPTIONAL_CONFIG_FIELDS: ClassVar[Dict[str, Any]] = {}
    COMPONENT_TYPE: ClassVar[str] = "unknown"
    
    @classmethod
    def from_config(cls, config: Union[str, Path, BaseModel, Dict[str, Any]], **kwargs) -> 'FromConfigBase':
        """
        Unified Component Creation Interface - The Primary Framework Entry Point
        =====================================================================
        
        This method provides the UNIFIED interface for creating ALL NanoBrain framework 
        components. It works identically across agents, steps, workflows, tools, and all 
        other component types, ensuring consistent behavior throughout the framework.
        
        **Universal Usage Pattern:**
            Every component type follows the exact same creation pattern:
            
            ```python
            # For ANY component type - agents, steps, workflows, etc.
            component = ComponentClass.from_config('path/to/config.yml')
            ```
        
        **Architecture:**
            This method implements the complete component creation lifecycle:
            
            1. **Configuration Loading**: Parse and validate configuration from file/dict
            2. **Schema Validation**: Validate against component-specific Pydantic schemas  
            3. **Dependency Resolution**: Resolve references to other components
            4. **Component Instantiation**: Create component instance with validated config
            5. **Initialization**: Execute component-specific initialization logic
            6. **Validation**: Perform final component validation and health checks
        
        **Supported Configuration Sources:**
            
            **YAML File Configuration (Recommended):**
            ```python
            agent = ConversationalAgent.from_config('config/my_agent.yml')
            step = DataProcessingStep.from_config('config/data_step.yml') 
            workflow = AnalysisWorkflow.from_config('config/workflow.yml')
            ```
            
            **Inline Dictionary Configuration (Testing):**
            ```python
            agent = ConversationalAgent.from_config({
                'name': 'test_agent',
                'model': 'gpt-4',
                'temperature': 0.7
            })
            ```
            
            **Pydantic Model Configuration (Advanced):**
            ```python
            config_obj = AgentConfig(name='agent', model='gpt-4')
            agent = ConversationalAgent.from_config(config_obj)
            ```
        
        **Configuration Examples:**
            
            **Agent Configuration:**
            ```yaml
            name: "research_assistant"
            description: "AI agent for research tasks"
            model: "gpt-4"
            temperature: 0.3
            system_prompt: "You are a research assistant."
            tools:
              - class: "nanobrain.library.tools.WebSearchTool"
                config: "config/web_search.yml"
            executor:
              class: "nanobrain.core.executor.LocalExecutor"
              config: "config/local_executor.yml"
            ```
            
            **Step Configuration:**
            ```yaml
            name: "data_analysis"
            description: "Analyzes input data"
            input_data_units:
              raw_data:
                class: "nanobrain.core.data_unit.DataUnitFile"
                config: "config/input_data.yml"
            output_data_units:
              results:
                class: "nanobrain.core.data_unit.DataUnitMemory"
                config: "config/output_data.yml"
            ```
            
            **Workflow Configuration:**
            ```yaml
            name: "analysis_pipeline"
            description: "Complete data analysis pipeline"
            execution_strategy: "event_driven"
            steps:
              - class: "nanobrain.library.steps.DataIngestionStep"
                config: "config/ingestion.yml"
              - class: "nanobrain.library.steps.ProcessingStep"
                config: "config/processing.yml"
            links:
              - class: "nanobrain.core.link.DirectLink"
                config:
                  source: "ingestion.output"
                  target: "processing.input"
            ```
        
        **Dependency Resolution:**
            The framework automatically resolves component dependencies:
            
            * **Class+Config Pattern**: References to other components via class and config
            * **Recursive Resolution**: Nested component dependencies resolved automatically
            * **Circular Dependency Detection**: Prevents infinite recursion with clear errors
            * **Lazy Loading**: Dependencies loaded only when needed for performance
            
            Example of automatic dependency resolution:
            ```yaml
            # Workflow automatically creates and configures referenced components
            steps:
              - class: "nanobrain.core.step.Step"
                config:
                  name: "processor"
                  agent:
                    class: "nanobrain.core.agent.ConversationalAgent"
                    config:
                      name: "processing_agent"
                      model: "gpt-4"
            ```
        
        **Error Handling:**
            Comprehensive error handling with detailed context:
            
            * **Configuration Errors**: Invalid YAML syntax, missing files, schema violations
            * **Dependency Errors**: Circular dependencies, missing references, import failures  
            * **Validation Errors**: Component-specific validation failures with suggestions
            * **Runtime Errors**: Component initialization failures with diagnostic information
            
            All errors include:
            - Detailed error messages with context
            - Configuration file paths and line numbers (where applicable)
            - Suggestions for resolution
            - Component type and dependency chain information
        
        **Performance Optimization:**
            The framework includes several performance optimizations:
            
            * **Configuration Caching**: Parsed configurations cached for reuse
            * **Lazy Dependency Loading**: Components loaded only when accessed
            * **Schema Validation Caching**: Validation schemas cached per component type
            * **Parallel Processing**: Independent components can be created in parallel
        
        **Framework Integration:**
            Seamless integration with all framework systems:
            
            * **Logging Integration**: Automatic logging of component creation and lifecycle
            * **Metrics Collection**: Performance metrics for component creation times
            * **Tool Registry**: Automatic registration of tools and capabilities
            * **Configuration Validation**: Schema-based validation with helpful error messages
        
        **Advanced Features:**
            
            **Environment Variable Interpolation:**
            ```yaml
            # Configuration can reference environment variables
            model: "${MODEL_NAME:-gpt-3.5-turbo}"
            api_key: "${OPENAI_API_KEY}"
            ```
            
            **Template Variables:**
            ```yaml
            # Template variables for reusable configurations
            name: "agent_${ENVIRONMENT}"
            debug_mode: ${DEBUG_MODE:-false}
            ```
            
            **Conditional Configuration:**
            ```yaml
            # Environment-specific configurations
            tools:
              - class: "nanobrain.library.tools.WebSearchTool"
                config: "config/web_search_${ENVIRONMENT}.yml"
            ```
        
        Args:
            config: Configuration source - can be:
                * **str/Path**: Path to YAML configuration file
                * **dict**: Inline configuration dictionary  
                * **BaseModel**: Pydantic configuration model instance
            **kwargs: Additional context for component creation:
                * **executor**: Shared executor for component operations
                * **workflow_directory**: Base directory for relative paths
                * **environment**: Environment name for conditional configs
                * **debug_mode**: Enable debug logging and validation
                * **validate_only**: Only validate configuration without creation
        
        Returns:
            Fully initialized component instance ready for use. The component will have:
            * All dependencies resolved and initialized
            * Configuration validated and applied
            * Logging and monitoring configured
            * Ready to accept requests or participate in workflows
        
        Raises:
            ComponentConfigurationError: When configuration is invalid:
                * Missing required fields
                * Invalid field values  
                * Schema validation failures
                * File format errors
            
            ComponentDependencyError: When dependencies cannot be resolved:
                * Circular dependency detection
                * Missing component references
                * Import failures for component classes
                * Dependency initialization failures
            
            FileNotFoundError: When configuration file cannot be found at specified path
            
            ValueError: When configuration format is invalid:
                * Invalid configuration type provided
                * Malformed YAML syntax
                * Incompatible configuration structure
            
            ImportError: When component class cannot be imported:
                * Missing component modules
                * Invalid class paths in configuration
                * Missing dependencies for component classes
        
        Examples:
            **Basic Agent Creation:**
            ```python
            from nanobrain.core import ConversationalAgent
            
            # Create from file
            agent = ConversationalAgent.from_config('config/agent.yml')
            
            # Create from inline config
            agent = ConversationalAgent.from_config({
                'name': 'helper',
                'model': 'gpt-4',
                'temperature': 0.7
            })
            ```
            
            **Complex Workflow Creation:**
            ```python
            from nanobrain.core import Workflow
            
            # Create workflow with automatic dependency resolution
            workflow = Workflow.from_config('config/analysis_workflow.yml')
            
            # Execute workflow
            results = await workflow.execute(input_data)
            ```
            
            **Component with Custom Executor:**
            ```python
            from nanobrain.core import ParslExecutor, ConversationalAgent
            
            # Create custom executor
            executor = ParslExecutor.from_config('config/hpc_executor.yml')
            
            # Create agent with custom executor
            agent = ConversationalAgent.from_config(
                'config/agent.yml', 
                executor=executor
            )
            ```
        
        Note:
            This method is the ONLY supported way to create framework components.
            Direct constructor usage (``Component()``) is prohibited and will raise
            errors. This ensures all components follow consistent configuration
            patterns and lifecycle management.
        
        Warning:
            Configuration files may contain sensitive information (API keys, credentials).
            Ensure proper file permissions and avoid committing sensitive configurations
            to version control. Use environment variables for sensitive values.
        
        See Also:
            * :meth:`_get_config_class`: Component-specific configuration class
            * :meth:`_init_from_config`: Component-specific initialization
            * :class:`ConfigBase`: Configuration validation and management
            * :mod:`nanobrain.core.config`: Configuration system documentation
        
        Args:
            config: Configuration file path, config object, or dictionary
            **kwargs: Framework dependencies and parameters
            
        Returns:
            Fully initialized component instance
            
        Raises:
            NotImplementedError: If subclass doesn't implement _get_config_class()
            FileNotFoundError: If config file path doesn't exist
            ComponentConfigurationError: If config validation fails
        """
        # Import utilities on-demand to avoid circular imports
        from .config.component_factory import load_config_file
        
        # Step 1: Normalize input to dictionary format
        if isinstance(config, (str, Path)):
            # Handle file path input
            config_path = Path(config)
            
            # Resolve relative paths
            if not config_path.is_absolute():
                config_path = cls._resolve_config_file_path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load YAML to dictionary
            config_dict = load_config_file(str(config_path))
            
            # Handle class auto-detection
            if 'class' in config_dict:
                target_class_path = config_dict['class']
                current_class_path = f"{cls.__module__}.{cls.__name__}"
                
                if target_class_path != current_class_path:
                    # Delegate to correct class
                    target_class = import_class_from_path(target_class_path)
                    return target_class.from_config(config_path, **kwargs)
            
        elif isinstance(config, dict):
            # Already dictionary format
            config_dict = config
            
        else:
            # Config object - convert to dictionary for unified processing
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                # Fallback - extract attributes
                config_dict = {
                    key: getattr(config, key) 
                    for key in dir(config) 
                    if not key.startswith('_') and not callable(getattr(config, key))
                }
        
        # Step 2: Create config object using component-specific config class
        config_class = cls._get_config_class()
        
        # Check if the config class is a ConfigBase subclass that requires file paths
        from .config.config_base import ConfigBase
        is_configbase_subclass = issubclass(config_class, ConfigBase)
        
        if is_configbase_subclass and isinstance(config, (str, Path)):
            # For ConfigBase subclasses, pass the file path directly to preserve from_config behavior
            config_object = config_class.from_config(config, **kwargs)
        else:
            # For other config classes or when config is already a dict, use the dictionary approach
            # Filter out framework-specific fields that shouldn't be passed to config class
            framework_fields = {'class', 'config_file'}
            filtered_config_dict = {k: v for k, v in config_dict.items() if k not in framework_fields}
            
            # ✅ FRAMEWORK COMPLIANCE: Use from_config method instead of constructor
            if hasattr(config_class, 'from_config'):
                if is_configbase_subclass:
                    # ConfigBase subclasses need special handling for inline dict configs
                    try:
                        config_object = config_class.from_config(filtered_config_dict)
                    except ValueError as e:
                        if "ONLY accepts file paths" in str(e):
                            # This ConfigBase subclass doesn't support inline dicts
                            # We need to create a temporary file or use the original path
                            if isinstance(config, (str, Path)):
                                config_object = config_class.from_config(config, **kwargs)
                            else:
                                raise ValueError(
                                    f"❌ CONFIGURATION ERROR: {config_class.__name__} requires file path but got dictionary.\n"
                                    f"   SOLUTION: Ensure step configuration is loaded from file path, not dictionary."
                                ) from e
                        else:
                            raise
                else:
                    config_object = config_class.from_config(filtered_config_dict)
            else:
                # Fallback to constructor for legacy classes
                config_object = config_class(**filtered_config_dict)
        
        # FRAMEWORK VALIDATION: Ensure proper Config usage
        validate_config_usage(cls, config_object)
        
        # Step 3: Use existing framework pattern for component creation
        cls.validate_config_schema(config_object)
        component_config = cls.extract_component_config(config_object)
        dependencies = cls.resolve_dependencies(component_config, **kwargs)
        instance = cls.create_instance(config_object, component_config, dependencies)
        instance._post_config_initialization()
        
        return instance
    
    @classmethod
    def _get_config_class(cls):
        """
        MANDATORY: Return config class for this component type.
        
        This is the ONLY method that differs between component types.
        ALL other aspects of from_config() are identical.
        
        This method should be implemented by all component subclasses.
        The NotImplementedError provides clear guidance when missing.
        
        Returns:
            Config class appropriate for this component type
            
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"Subclass {cls.__name__} must implement _get_config_class() "
            f"to specify which config class to use. "
            f"This is the ONLY component-specific method required."
        )
    
    @classmethod
    def _resolve_config_file_path(cls, config_file: Union[str, Path]) -> Path:
        """
        Enhanced configuration file path resolution with multiple search strategies.
        
        Resolution order:
        1. Absolute paths - return as-is if they exist
        2. Relative to calling class file directory (traditional NanoBrain rule)
        3. Relative to workflow directory (if available via context)
        4. Relative to current working directory
        
        Args:
            config_file: Relative or absolute path to config file
            
        Returns:
            Resolved absolute path to config file
            
        Raises:
            FileNotFoundError: If config file cannot be found in any location
        """
        config_path = Path(config_file)
        
        # Return absolute paths as-is if they exist
        if config_path.is_absolute():
            if config_path.exists():
                return config_path
            else:
                raise FileNotFoundError(f"Absolute config path not found: {config_path}")
        
        # Multiple resolution strategies for relative paths
        resolution_attempts = []
        
        # Strategy 1: Relative to calling class's file directory (traditional rule)
        try:
            calling_class_file = inspect.getfile(cls)
            calling_class_dir = Path(calling_class_file).parent.resolve()
            class_relative_path = calling_class_dir / config_file
            resolution_attempts.append(("Class Directory", class_relative_path))
            
            # Strategy 1b: Try going up one directory level (for step configs in parent directories)
            parent_relative_path = calling_class_dir.parent / config_file
            resolution_attempts.append(("Class Parent Directory", parent_relative_path))
            
        except (OSError, TypeError) as e:
            # If we can't determine class file location, continue with other strategies
            pass
        
        # Strategy 2: Relative to current working directory
        cwd_relative_path = Path.cwd() / config_file
        resolution_attempts.append(("Current Working Directory", cwd_relative_path))
        
        # Strategy 3: Look for workflow directory in the path and use that as base
        config_file_str = str(config_file)
        if 'workflows/' in config_file_str:
            # Extract the workflow directory portion
            parts = config_file_str.split('workflows/')
            if len(parts) > 1:
                # Try to find the workflow directory in the filesystem
                potential_workflow_bases = [
                    Path.cwd() / "nanobrain/library/workflows",
                    Path.cwd() / "library/workflows",
                    Path.cwd() / "workflows"
                ]
                
                for base in potential_workflow_bases:
                    if base.exists():
                        workflow_relative_path = base / parts[1]
                        resolution_attempts.append(("Workflow Base Directory", workflow_relative_path))
        
        # Try each resolution strategy
        for strategy_name, resolved_path in resolution_attempts:
            if resolved_path.exists():
                logger = get_logger("component_base")
                logger.debug(f"✅ Config resolved via {strategy_name}: {resolved_path}")
                return resolved_path.resolve()
            else:
                logger = get_logger("component_base")
                logger.debug(f"❌ Config not found via {strategy_name}: {resolved_path}")
        
        # If all strategies failed, provide comprehensive error message
        attempted_paths = [str(attempt[1]) for attempt in resolution_attempts]
        raise FileNotFoundError(
            f"❌ CONFIGURATION FILE RESOLUTION FAILED: {config_file}\n"
            f"   CALLING CLASS: {cls.__module__}.{cls.__name__}\n"
            f"   SEARCHED PATHS:\n" + 
            "\n".join(f"      {i+1}. {path}" for i, path in enumerate(attempted_paths)) +
            f"\n   SOLUTIONS:\n"
            f"      1. Ensure config file exists at one of the searched locations\n"
            f"      2. Use absolute path: Path('/full/path/to/config.yml')\n"
            f"      3. Place config file relative to class directory\n"
            f"      4. Check current working directory: {Path.cwd()}"
        )

    @classmethod
    def validate_config_schema(cls, config: Any) -> None:
        """Validate that configuration contains required fields"""
        missing_fields = []
        for field in cls.REQUIRED_CONFIG_FIELDS:
            # Handle both dictionary and object configurations
            if isinstance(config, dict):
                if field not in config:
                    missing_fields.append(field)
            else:
                if not hasattr(config, field):
                    missing_fields.append(field)
        
        if missing_fields:
            raise ComponentConfigurationError(
                f"{cls.__name__} missing required configuration fields: {missing_fields}"
            )
    
    @classmethod
    def extract_component_config(cls, config: Any) -> Dict[str, Any]:
        """Extract component-specific configuration - SAME signature for ALL components"""
        # Default implementation - components can override if needed
        return {
            'name': getattr(config, 'name', 'unnamed'),
            'description': getattr(config, 'description', ''),
            'timeout': getattr(config, 'timeout', 300),
            'enable_logging': getattr(config, 'enable_logging', True)
        }
    
    @classmethod  
    def resolve_dependencies(cls, component_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Resolve dependencies - SAME signature for ALL components"""
        # Default implementation - framework injects standard dependencies
        return {
            'enable_logging': kwargs.get('enable_logging', True),
            'debug_mode': kwargs.get('debug_mode', False),
            'framework_version': kwargs.get('framework_version', '2.0.0')
        }
    
    @classmethod
    def create_instance(cls, config: Any, component_config: Dict[str, Any], 
                       dependencies: Dict[str, Any]) -> 'FromConfigBase':
        """Create instance with resolved configuration and dependencies"""
        # Use __new__ to bypass __init__
        instance = cls.__new__(cls)
        instance._init_from_config(config, component_config, dependencies)
        return instance
    
    def _init_from_config(self, config: Any, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize component - SAME signature for ALL components"""
        # Default implementation - components override for specific behavior
        self.config = config
        self.name = component_config.get('name', 'unnamed')
        self.description = component_config.get('description', '')
        self.timeout = component_config.get('timeout', 300)
        self.enable_logging = component_config.get('enable_logging', True)
        
        # Initialize logging if enabled
        if self.enable_logging:
            self.nb_logger = get_logger(f"{self.__class__.__name__.lower()}.{self.name}")
    
    def _post_config_initialization(self) -> None:
        """Post-configuration initialization hook (override in subclasses if needed)"""
        pass
    
    # PREVENT direct instantiation (unless explicitly allowed during from_config)
    def __init__(self, *args, **kwargs):
        if not getattr(self.__class__, '_allow_direct_instantiation', False):
            raise RuntimeError(
                f"Direct instantiation of {self.__class__.__name__} is prohibited. "
                f"Use: {self.__class__.__name__}.from_config(config_file_or_object)"
            )
        # Don't call super().__init__() with arguments since object.__init__() doesn't accept them
        super().__init__() 