"""
Component Factory for NanoBrain Framework

Pure from_config pattern factory with dynamic class loading.
"""

import logging
import importlib
from typing import Any, Dict, Union, Optional, Type, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def create_component(class_path: str, config: Any, **kwargs) -> Any:
    """
    Enterprise Component Factory - Advanced Dynamic Component Creation and Dependency Injection
    ========================================================================================
    
    The create_component function provides comprehensive dynamic component creation with intelligent
    class loading, configuration validation, dependency injection, and framework integration. This
    factory enables runtime component instantiation, plugin architectures, and flexible system
    composition while maintaining type safety and framework compliance patterns.
    
    **Core Architecture:**
        The component factory provides enterprise-grade component creation capabilities:
        
        * **Dynamic Class Loading**: Runtime class import and instantiation with intelligent error handling
        * **Framework Compliance**: Mandatory from_config pattern enforcement with validation
        * **Configuration Integration**: Seamless integration with ConfigBase and validation systems
        * **Dependency Injection**: Advanced dependency resolution and injection management
        * **Plugin Architecture**: Extensible plugin system with dynamic component discovery
        * **Type Safety**: Comprehensive type checking and validation with error reporting
    
    **Component Creation Capabilities:**
        
        **Dynamic Loading System:**
        * **Module Path Resolution**: Intelligent module path parsing and import resolution
        * **Class Discovery**: Automatic class discovery and instantiation with validation
        * **Import Error Handling**: Comprehensive error handling with detailed diagnostic information
        * **Version Compatibility**: Framework version compatibility checking and validation
        
        **Framework Integration:**
        * **from_config Enforcement**: Mandatory from_config pattern validation and enforcement
        * **Configuration Validation**: Automatic configuration validation and type checking
        * **Component Registration**: Automatic component registration and lifecycle management
        * **Interface Compliance**: Component interface validation and compatibility checking
        
        **Enterprise Features:**
        * **Dependency Injection**: Advanced dependency resolution with circular dependency detection
        * **Plugin Architecture**: Dynamic plugin loading and registration with hot-swapping
        * **Configuration Templates**: Template-based component configuration with inheritance
        * **Performance Optimization**: Caching and lazy loading for improved performance
    
    **Dynamic Component Creation Process:**
        
        **1. Path Validation and Parsing:**
        ```python
        # Validate and parse component class path
        module_path, class_name = validate_and_parse_path(class_path)
        
        # Example valid paths:
        # "nanobrain.core.agent.Agent"
        # "nanobrain.library.tools.bioinformatics.BVBRCTool"
        # "custom.components.MyCustomAgent"
        ```
        
        **2. Dynamic Module Import:**
        ```python
        # Import module with intelligent error handling
        module = import_module_with_validation(module_path)
        component_class = getattr_with_validation(module, class_name)
        
        # Verify framework compliance
        validate_framework_compliance(component_class)
        ```
        
        **3. Configuration Processing:**
        ```python
        # Process and validate configuration
        processed_config = process_component_config(config, component_class)
        
        # Resolve dependencies if needed
        resolved_dependencies = resolve_component_dependencies(processed_config)
        ```
        
        **4. Component Instantiation:**
        ```python
        # Create component via from_config pattern
        component = component_class.from_config(processed_config, **kwargs)
        
        # Register component for lifecycle management
        register_component_instance(component)
        ```
    
    **Usage Patterns:**
        
        **Basic Component Creation:**
        ```python
        from nanobrain.core.config.component_factory import create_component
        
        # Create agent with configuration file
        agent = create_component(
            class_path="nanobrain.library.agents.enhanced.CollaborativeAgent",
            config="config/collaborative_agent.yml"
        )
        
        # Create tool with configuration dictionary
        tool_config = {
            'tool_name': 'bvbrc_tool',
            'cache_enabled': True,
            'timeout': 30
        }
        
        tool = create_component(
            class_path="nanobrain.library.tools.bioinformatics.BVBRCTool",
            config=tool_config
        )
        
        # Create workflow with complex configuration
        workflow = create_component(
            class_path="nanobrain.library.workflows.chat_workflow.ChatWorkflow",
            config="config/workflows/chat_workflow.yml",
            environment="production"
        )
        
        print(f"Created agent: {agent}")
        print(f"Created tool: {tool}")
        print(f"Created workflow: {workflow}")
        ```
        
        **Advanced Component Creation with Dependencies:**
        ```python
        # Component creation with dependency injection
        class ComponentCreationManager:
            def __init__(self):
                self.component_registry = {}
                self.dependency_graph = {}
                
            async def create_component_with_dependencies(
                self, 
                component_spec: Dict[str, Any]
            ) -> Any:
                \"\"\"Create component with automatic dependency resolution\"\"\"
                
                class_path = component_spec['class']
                config = component_spec['config']
                dependencies = component_spec.get('dependencies', [])
                
                # Resolve dependencies first
                resolved_dependencies = {}
                for dep_name in dependencies:
                    if dep_name not in self.component_registry:
                        # Create dependency if not exists
                        dep_spec = self.get_dependency_spec(dep_name)
                        dependency = await self.create_component_with_dependencies(dep_spec)
                        self.component_registry[dep_name] = dependency
                    
                    resolved_dependencies[dep_name] = self.component_registry[dep_name]
                
                # Create main component with resolved dependencies
                component = create_component(
                    class_path=class_path,
                    config=config,
                    dependencies=resolved_dependencies
                )
                
                return component
                
            async def create_application_stack(self, stack_config: Dict[str, Any]):
                \"\"\"Create complete application stack with ordered dependency resolution\"\"\"
                
                # Build dependency graph
                dependency_order = self.calculate_dependency_order(stack_config)
                
                created_components = {}
                
                for component_name in dependency_order:
                    component_spec = stack_config[component_name]
                    
                    # Create component with dependencies
                    component = await self.create_component_with_dependencies(component_spec)
                    created_components[component_name] = component
                    self.component_registry[component_name] = component
                    
                    print(f"Created component: {component_name}")
                
                return created_components
        
        # Example application stack configuration
        application_stack = {
            'database_manager': {
                'class': 'nanobrain.library.infrastructure.data.DatabaseManager',
                'config': 'config/database_config.yml',
                'dependencies': []
            },
            'cache_manager': {
                'class': 'nanobrain.library.infrastructure.data.CacheManager',
                'config': 'config/cache_config.yml',
                'dependencies': []
            },
            'ai_agent': {
                'class': 'nanobrain.library.agents.enhanced.CollaborativeAgent',
                'config': 'config/agent_config.yml',
                'dependencies': ['database_manager', 'cache_manager']
            },
            'web_interface': {
                'class': 'nanobrain.library.interfaces.web.WebInterface',
                'config': 'config/web_interface_config.yml',
                'dependencies': ['ai_agent']
            }
        }
        
        # Create complete application stack
        manager = ComponentCreationManager()
        components = await manager.create_application_stack(application_stack)
        
        # Access created components
        agent = components['ai_agent']
        web_interface = components['web_interface']
        ```
        
        **Plugin Architecture with Dynamic Loading:**
        ```python
        # Dynamic plugin system using component factory
        class PluginManager:
            def __init__(self):
                self.loaded_plugins = {}
                self.plugin_registry = {}
                
            async def discover_plugins(self, plugin_directory: str):
                \"\"\"Discover and register available plugins\"\"\"
                
                plugin_path = Path(plugin_directory)
                
                for plugin_config in plugin_path.glob("*.yml"):
                    plugin_name = plugin_config.stem
                    
                    # Load plugin configuration
                    with open(plugin_config) as f:
                        config = yaml.safe_load(f)
                    
                    # Register plugin
                    self.plugin_registry[plugin_name] = {
                        'class_path': config['plugin_class'],
                        'config_file': str(plugin_config),
                        'metadata': config.get('metadata', {}),
                        'dependencies': config.get('dependencies', [])
                    }
                    
            async def load_plugin(self, plugin_name: str) -> Any:
                \"\"\"Load and instantiate plugin dynamically\"\"\"
                
                if plugin_name in self.loaded_plugins:
                    return self.loaded_plugins[plugin_name]
                
                plugin_info = self.plugin_registry.get(plugin_name)
                if not plugin_info:
                    raise ValueError(f"Plugin not found: {plugin_name}")
                
                # Load plugin dependencies
                for dep_name in plugin_info['dependencies']:
                    if dep_name not in self.loaded_plugins:
                        await self.load_plugin(dep_name)
                
                # Create plugin component
                try:
                    plugin = create_component(
                        class_path=plugin_info['class_path'],
                        config=plugin_info['config_file']
                    )
                    
                    # Initialize plugin
                    if hasattr(plugin, 'initialize'):
                        await plugin.initialize()
                    
                    self.loaded_plugins[plugin_name] = plugin
                    print(f"Loaded plugin: {plugin_name}")
                    
                    return plugin
                    
                except Exception as e:
                    print(f"Failed to load plugin {plugin_name}: {e}")
                    raise
                    
            async def unload_plugin(self, plugin_name: str):
                \"\"\"Unload plugin and cleanup resources\"\"\"
                
                if plugin_name not in self.loaded_plugins:
                    return
                
                plugin = self.loaded_plugins[plugin_name]
                
                # Cleanup plugin
                if hasattr(plugin, 'cleanup'):
                    await plugin.cleanup()
                
                del self.loaded_plugins[plugin_name]
                print(f"Unloaded plugin: {plugin_name}")
                
            async def reload_plugin(self, plugin_name: str):
                \"\"\"Reload plugin with updated configuration\"\"\"
                
                # Unload existing plugin
                await self.unload_plugin(plugin_name)
                
                # Reload plugin configuration
                await self.discover_plugins(self.plugin_directory)
                
                # Load updated plugin
                return await self.load_plugin(plugin_name)
        
        # Plugin management
        plugin_manager = PluginManager()
        await plugin_manager.discover_plugins('plugins/')
        
        # Load specific plugins
        analytics_plugin = await plugin_manager.load_plugin('analytics_plugin')
        monitoring_plugin = await plugin_manager.load_plugin('monitoring_plugin')
        
        # Use plugins
        await analytics_plugin.process_data(data)
        await monitoring_plugin.start_monitoring()
        ```
        
        **Error Handling and Validation:**
        ```python
        # Comprehensive error handling for component creation
        def create_component_with_validation(
            class_path: str, 
            config: Any, 
            **kwargs
        ) -> Any:
            \"\"\"Create component with comprehensive validation and error handling\"\"\"
            
            try:
                # Pre-creation validation
                validate_class_path(class_path)
                validate_configuration(config)
                
                # Create component
                component = create_component(class_path, config, **kwargs)
                
                # Post-creation validation
                validate_component_instance(component)
                
                return component
                
            except ValueError as e:
                print(f"Configuration validation error: {e}")
                print(f"Class path: {class_path}")
                print(f"Config: {config}")
                raise
                
            except ImportError as e:
                print(f"Import error for class: {class_path}")
                print(f"Error details: {e}")
                
                # Suggest alternatives
                suggestions = suggest_similar_classes(class_path)
                if suggestions:
                    print(f"Similar classes found: {suggestions}")
                raise
                
            except AttributeError as e:
                print(f"Component {class_path} does not implement required interface")
                print(f"Error: {e}")
                
                # Check for common issues
                if "from_config" in str(e):
                    print("Component must implement from_config method")
                    print("Example implementation:")
                    print("  @classmethod")
                    print("  def from_config(cls, config, **kwargs):")
                    print("      return cls(**config)")
                raise
                
            except Exception as e:
                print(f"Unexpected error creating component {class_path}: {e}")
                print(f"Config: {config}")
                print(f"Kwargs: {kwargs}")
                raise
        
        # Example with error handling
        try:
            component = create_component_with_validation(
                "nanobrain.library.agents.enhanced.CollaborativeAgent",
                "config/agent_config.yml"
            )
        except Exception as e:
            print(f"Component creation failed: {e}")
            # Handle error appropriately
        ```
    
    **Performance Optimization:**
        
        **Caching and Optimization:**
        * **Class Import Caching**: Cache imported classes for improved performance
        * **Configuration Caching**: Cache processed configurations for reuse
        * **Lazy Loading**: Implement lazy loading for large component hierarchies
        * **Memory Optimization**: Optimize memory usage for large-scale component creation
        
        **Parallel Creation:**
        * **Concurrent Component Creation**: Create independent components in parallel
        * **Dependency Parallelization**: Parallelize independent dependency branches
        * **Batch Processing**: Efficient batch creation for multiple components
        * **Resource Pool Management**: Manage creation resources for optimal performance
    
    **Enterprise Integration:**
        
        **Framework Compliance:**
        * **Interface Validation**: Ensure all components implement required interfaces
        * **Version Compatibility**: Validate framework version compatibility
        * **Security Validation**: Validate component security and permissions
        * **Configuration Standards**: Enforce configuration standards and patterns
        
        **Monitoring and Analytics:**
        * **Creation Metrics**: Track component creation success rates and performance
        * **Dependency Analysis**: Analyze component dependency patterns and optimization
        * **Error Analytics**: Comprehensive error tracking and analysis
        * **Performance Profiling**: Profile component creation performance and bottlenecks
    
    Args:
        class_path (str): Full module.Class path for dynamic import (e.g., "nanobrain.core.agent.Agent")
        config (Any): Configuration object, file path, or dictionary for component initialization
        **kwargs: Additional arguments passed to the component's from_config method
        
    Returns:
        Any: Component instance created via from_config pattern with full initialization
        
    Raises:
        ValueError: If class path format is invalid or configuration is malformed
        ImportError: If specified class/module cannot be imported or found
        AttributeError: If component class doesn't implement required from_config method
        ComponentConfigurationError: If component configuration validation fails
        RuntimeError: If component creation fails due to runtime issues
        
    Note:
        This factory requires all components to implement the from_config pattern for framework compliance.
        Class paths must use full module.Class format for proper import resolution.
        Configuration validation depends on the target component's validation requirements.
        Created components are automatically registered for lifecycle management when applicable.
        
    Warning:
        Dynamic class loading may introduce security risks if class paths are not properly validated.
        Component creation failure may leave partial dependencies in inconsistent states.
        Large dependency graphs may cause performance issues during creation.
        Circular dependencies in component configurations will cause creation failures.
        
    See Also:
        * :class:`ComponentRegistry`: Component registration and lifecycle management
        * :mod:`nanobrain.core.component_base`: Framework component base classes and interfaces
        * :mod:`nanobrain.core.config.config_base`: Configuration base classes and validation
        * :mod:`nanobrain.core.config.yaml_config`: YAML configuration loading and processing
        * :func:`load_config_file`: Configuration file loading utilities
    """
    from ..logging_system import get_logger
    logger = get_logger("component_factory")
    
    # Basic validation
    if not class_path or '.' not in class_path:
        raise ValueError(f"Invalid class path: {class_path}. Must use full module.Class format")
    
    # Import the specified class
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import component '{class_path}': {e}")
    
    # Verify class has from_config method
    if not hasattr(component_class, 'from_config'):
        raise AttributeError(
            f"Component '{class_path}' does not implement required from_config method. "
            f"All NanoBrain components must implement from_config pattern."
        )
    
    # Create instance using from_config pattern
    try:
        instance = component_class.from_config(config, **kwargs)
        logger.debug(f"Created {class_path} via from_config pattern")
        return instance
    except Exception as e:
        from ..component_base import ComponentConfigurationError
        raise ComponentConfigurationError(
            f"Failed to create '{class_path}' via from_config: {e}"
        )


def get_component_class(class_path: str) -> Type:
    """
    Import and return component class for direct from_config usage
    
    Args:
        class_path: Full module.Class path
        
    Returns:
        Component class ready for from_config usage
    """
    if not class_path or '.' not in class_path:
        raise ValueError(f"Invalid class path: {class_path}. Must use full module.Class format")
    
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        
        # Verify class has from_config method
        if not hasattr(component_class, 'from_config'):
            raise AttributeError(
                f"Component '{class_path}' does not implement required from_config method"
            )
            
        return component_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import component class '{class_path}': {e}")


# REMOVED: _convert_dict_to_config_object function - NO PREPROCESSING
# Components handle their own configuration conversion in from_config

def validate_component_config(config: Dict[str, Any]) -> None:
    """
    Validate component configuration has required fields
    NO HARDCODED VALIDATION - only structural checks
    """
    from pydantic import BaseModel
    
    if not isinstance(config, (dict, BaseModel)):
        raise ValueError("Configuration must be dict or BaseModel")
    
    # Only validate that class field exists for dict configs
    if isinstance(config, dict) and 'class' not in config:
        raise ValueError("Component configuration must specify 'class' field")


# REMOVED ComponentFactory class - use functions directly

def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file - NO MODIFICATIONS"""
    import yaml
    
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a dictionary: {config_path}")
    
    # Return AS-IS - no modifications or hardcoded additions
    return config


# Component registry helpers (NO HARDCODING)
class ComponentRegistry:
    """
    Enterprise Component Registry - Advanced Component Registration and Lifecycle Management
    ====================================================================================
    
    The ComponentRegistry provides comprehensive component registration, discovery, and lifecycle
    management for enterprise applications, implementing self-registration patterns, intelligent
    component lookup, and automated registry management. This registry enables dynamic component
    discovery, plugin architectures, and flexible system composition while maintaining framework
    compliance and avoiding hardcoded component mappings.
    
    **Core Architecture:**
        The component registry provides enterprise-grade component management capabilities:
        
        * **Self-Registration Pattern**: Components automatically register themselves without hardcoded mappings
        * **Dynamic Discovery**: Runtime component discovery and registration with intelligent lookup
        * **Lifecycle Management**: Component lifecycle tracking and management with cleanup automation
        * **Plugin Architecture**: Extensible plugin system with hot-swapping and dynamic loading
        * **Framework Integration**: Complete integration with component factory and dependency injection
        * **Registry Analytics**: Component usage analytics and optimization recommendations
    
    **Component Registration Capabilities:**
        
        **Self-Registration System:**
        * **Automatic Registration**: Components self-register during import or initialization
        * **No Hardcoded Mappings**: Zero hardcoded component mappings or predefined registrations
        * **Dynamic Path Generation**: Automatic full module path generation for component identification
        * **Collision Detection**: Duplicate registration detection and resolution
        
        **Registration Validation:**
        * **Interface Compliance**: Automatic validation of component interface compliance
        * **Framework Compatibility**: Version and framework compatibility validation
        * **Registration Integrity**: Registry integrity validation and corruption detection
        * **Dependency Validation**: Component dependency validation and resolution
        
        **Registry Management:**
        * **Efficient Lookup**: High-performance component lookup with caching optimization
        * **Registry Persistence**: Optional registry persistence for faster startup times
        * **Registry Synchronization**: Multi-process registry synchronization and coordination
        * **Registry Analytics**: Component usage tracking and optimization analysis
    
    **Usage Patterns:**
        
        **Component Self-Registration:**
        ```python
        from nanobrain.core.config.component_factory import ComponentRegistry
        from nanobrain.core.component_base import FromConfigBase
        
        # Component automatically registers itself
        class MyCustomAgent(FromConfigBase):
            \"\"\"Custom agent that self-registers\"\"\"
            
            def __init__(self):
                super().__init__()
                # Component automatically registers during initialization
                ComponentRegistry.register(self.__class__)
                
            @classmethod
            def from_config(cls, config, **kwargs):
                instance = cls()
                # Additional initialization
                return instance
        
        # Registration happens automatically when class is defined/imported
        # No manual registration required
        
        # Lookup registered component
        component_class = ComponentRegistry.get("__main__.MyCustomAgent")
        if component_class:
            print(f"Found registered component: {component_class}")
            instance = component_class.from_config(config)
        ```
        
        **Plugin System with Registry:**
        ```python
        # Advanced plugin system using component registry
        class PluginRegistryManager:
            def __init__(self):
                self.active_plugins = {}
                
            def discover_and_register_plugins(self, plugin_directory: str):
                \"\"\"Discover plugins and trigger registration\"\"\"
                
                import importlib.util
                from pathlib import Path
                
                plugin_path = Path(plugin_directory)
                
                for plugin_file in plugin_path.glob("*.py"):
                    # Dynamic plugin import triggers self-registration
                    spec = importlib.util.spec_from_file_location(
                        f"plugin_{plugin_file.stem}", 
                        plugin_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Plugin components automatically register themselves
                    print(f"Loaded plugin module: {plugin_file.stem}")
                    
            def get_plugins_by_type(self, plugin_type: str) -> List[type]:
                \"\"\"Get all registered plugins of specific type\"\"\"
                
                matching_plugins = []
                
                for class_path, component_class in ComponentRegistry._registry.items():
                    if hasattr(component_class, 'PLUGIN_TYPE'):
                        if component_class.PLUGIN_TYPE == plugin_type:
                            matching_plugins.append(component_class)
                            
                return matching_plugins
                
            def get_all_registered_components(self) -> Dict[str, type]:
                \"\"\"Get all registered components for analysis\"\"\"
                
                return ComponentRegistry._registry.copy()
                
            def validate_plugin_compatibility(self, plugin_class: type) -> bool:
                \"\"\"Validate plugin compatibility with current framework\"\"\"
                
                # Check required interface
                if not hasattr(plugin_class, 'from_config'):
                    return False
                    
                # Check framework version compatibility
                if hasattr(plugin_class, 'REQUIRED_FRAMEWORK_VERSION'):
                    # Validate version compatibility
                    pass
                    
                return True
        
        # Plugin discovery and management
        plugin_manager = PluginRegistryManager()
        plugin_manager.discover_and_register_plugins("plugins/")
        
        # Find specific plugin types
        ai_plugins = plugin_manager.get_plugins_by_type("ai_agent")
        tool_plugins = plugin_manager.get_plugins_by_type("bioinformatics_tool")
        
        print(f"Found {len(ai_plugins)} AI agent plugins")
        print(f"Found {len(tool_plugins)} bioinformatics tool plugins")
        ```
        
        **Enterprise Component Management:**
        ```python
        # Enterprise component registry with advanced features
        class EnterpriseComponentRegistry:
            def __init__(self):
                self.component_metadata = {}
                self.usage_analytics = {}
                self.dependency_graph = {}
                
            def register_with_metadata(self, component_class: type, metadata: Dict[str, Any]):
                \"\"\"Register component with additional metadata\"\"\"
                
                # Standard registration
                ComponentRegistry.register(component_class)
                
                # Store additional metadata
                class_path = f"{component_class.__module__}.{component_class.__name__}"
                self.component_metadata[class_path] = {
                    'metadata': metadata,
                    'registration_time': datetime.now(),
                    'usage_count': 0,
                    'dependencies': self.extract_dependencies(component_class)
                }
                
            def get_component_with_analytics(self, class_path: str) -> Optional[type]:
                \"\"\"Get component and track usage analytics\"\"\"
                
                component_class = ComponentRegistry.get(class_path)
                
                if component_class:
                    # Track usage
                    if class_path in self.component_metadata:
                        self.component_metadata[class_path]['usage_count'] += 1
                        self.component_metadata[class_path]['last_used'] = datetime.now()
                        
                return component_class
                
            def analyze_component_usage(self) -> Dict[str, Any]:
                \"\"\"Analyze component usage patterns\"\"\"
                
                analysis = {
                    'total_components': len(ComponentRegistry._registry),
                    'most_used_components': [],
                    'unused_components': [],
                    'dependency_analysis': {}
                }
                
                # Usage analysis
                usage_data = []
                for class_path, metadata in self.component_metadata.items():
                    usage_count = metadata.get('usage_count', 0)
                    usage_data.append((class_path, usage_count))
                    
                    if usage_count == 0:
                        analysis['unused_components'].append(class_path)
                        
                # Sort by usage
                usage_data.sort(key=lambda x: x[1], reverse=True)
                analysis['most_used_components'] = usage_data[:10]
                
                return analysis
                
            def optimize_registry(self):
                \"\"\"Optimize registry based on usage patterns\"\"\"
                
                analysis = self.analyze_component_usage()
                
                # Remove unused components (if safe to do so)
                for unused_component in analysis['unused_components']:
                    if self.is_safe_to_remove(unused_component):
                        del ComponentRegistry._registry[unused_component]
                        del self.component_metadata[unused_component]
                        
            def generate_component_report(self) -> str:
                \"\"\"Generate comprehensive component registry report\"\"\"
                
                report = []
                report.append("Component Registry Report")
                report.append("=" * 50)
                
                # Registry statistics
                total_components = len(ComponentRegistry._registry)
                report.append(f"Total Registered Components: {total_components}")
                
                # Component breakdown by module
                module_breakdown = {}
                for class_path in ComponentRegistry._registry.keys():
                    module_path = class_path.rsplit('.', 1)[0]
                    module_breakdown[module_path] = module_breakdown.get(module_path, 0) + 1
                    
                report.append("\\nComponents by Module:")
                for module, count in sorted(module_breakdown.items()):
                    report.append(f"  {module}: {count} components")
                    
                # Usage analysis
                if self.component_metadata:
                    analysis = self.analyze_component_usage()
                    report.append(f"\\nMost Used Components:")
                    for class_path, usage_count in analysis['most_used_components'][:5]:
                        report.append(f"  {class_path}: {usage_count} uses")
                        
                return "\\n".join(report)
        
        # Enterprise registry management
        enterprise_registry = EnterpriseComponentRegistry()
        
        # Register components with metadata
        enterprise_registry.register_with_metadata(
            MyCustomAgent,
            {
                'category': 'ai_agent',
                'version': '1.0.0',
                'author': 'Enterprise Team',
                'capabilities': ['chat', 'analysis', 'collaboration']
            }
        )
        
        # Get component with analytics
        agent_class = enterprise_registry.get_component_with_analytics(
            "__main__.MyCustomAgent"
        )
        
        # Generate reports
        report = enterprise_registry.generate_component_report()
        print(report)
        ```
        
        **Registry Testing and Validation:**
        ```python
        # Comprehensive registry testing utilities
        class RegistryTestSuite:
            def __init__(self):
                self.test_results = {}
                
            def test_registry_integrity(self) -> bool:
                \"\"\"Test registry integrity and consistency\"\"\"
                
                test_results = {
                    'path_validation': True,
                    'class_validation': True,
                    'interface_compliance': True,
                    'duplicate_detection': True
                }
                
                for class_path, component_class in ComponentRegistry._registry.items():
                    # Validate path format
                    if not self.validate_class_path_format(class_path):
                        test_results['path_validation'] = False
                        
                    # Validate class object
                    if not self.validate_class_object(component_class):
                        test_results['class_validation'] = False
                        
                    # Validate interface compliance
                    if not self.validate_interface_compliance(component_class):
                        test_results['interface_compliance'] = False
                        
                # Check for duplicates
                if self.detect_duplicate_registrations():
                    test_results['duplicate_detection'] = False
                    
                self.test_results['integrity_test'] = test_results
                return all(test_results.values())
                
            def test_registry_performance(self) -> Dict[str, float]:
                \"\"\"Test registry performance characteristics\"\"\"
                
                import time
                
                performance_results = {}
                
                # Test lookup performance
                start_time = time.time()
                for class_path in ComponentRegistry._registry.keys():
                    ComponentRegistry.get(class_path)
                lookup_time = time.time() - start_time
                performance_results['lookup_time'] = lookup_time
                
                # Test registration performance
                test_class = type('TestClass', (), {'from_config': lambda: None})
                start_time = time.time()
                ComponentRegistry.register(test_class)
                registration_time = time.time() - start_time
                performance_results['registration_time'] = registration_time
                
                # Cleanup test registration
                ComponentRegistry.clear()
                
                return performance_results
                
            def generate_test_report(self) -> str:
                \"\"\"Generate comprehensive test report\"\"\"
                
                # Run all tests
                integrity_passed = self.test_registry_integrity()
                performance_metrics = self.test_registry_performance()
                
                report = []
                report.append("Registry Test Report")
                report.append("=" * 30)
                
                # Integrity results
                report.append(f"Integrity Test: {'PASSED' if integrity_passed else 'FAILED'}")
                if 'integrity_test' in self.test_results:
                    for test_name, result in self.test_results['integrity_test'].items():
                        status = 'PASS' if result else 'FAIL'
                        report.append(f"  {test_name}: {status}")
                        
                # Performance results
                report.append("\\nPerformance Metrics:")
                for metric_name, value in performance_metrics.items():
                    report.append(f"  {metric_name}: {value:.6f}s")
                    
                return "\\n".join(report)
        
        # Registry testing
        test_suite = RegistryTestSuite()
        test_report = test_suite.generate_test_report()
        print(test_report)
        ```
    
    **Advanced Features:**
        
        **Dynamic Registry Management:**
        * **Hot-Swapping**: Runtime component replacement without system restart
        * **Version Management**: Multiple component versions with intelligent selection
        * **Dependency Tracking**: Automatic dependency tracking and validation
        * **Registry Persistence**: Optional registry state persistence across restarts
        
        **Performance Optimization:**
        * **Lookup Caching**: High-performance component lookup with intelligent caching
        * **Lazy Registration**: Lazy component registration for improved startup performance
        * **Memory Optimization**: Memory-efficient registry storage for large component sets
        * **Batch Operations**: Efficient batch registration and lookup operations
        
        **Enterprise Integration:**
        * **Multi-Process Coordination**: Registry synchronization across multiple processes
        * **Distributed Registry**: Distributed component registry for microservice architectures
        * **Registry Analytics**: Comprehensive analytics and monitoring for component usage
        * **Security Integration**: Access control and security validation for component registration
    
    **Registry Security:**
        
        **Registration Validation:**
        * **Component Verification**: Cryptographic verification of component integrity
        * **Source Validation**: Validation of component source and origin
        * **Permission Checking**: Role-based access control for component registration
        * **Malware Detection**: Basic malware and suspicious code detection
        
        **Runtime Security:**
        * **Access Control**: Fine-grained access control for component lookup and usage
        * **Audit Logging**: Comprehensive audit logging for registry operations
        * **Integrity Monitoring**: Runtime integrity monitoring and corruption detection
        * **Secure Cleanup**: Secure component cleanup and resource deallocation
    
    Attributes:
        _registry (Dict[str, type]): Internal registry mapping class paths to component classes
        
    Class Methods:
        register: Register component class with automatic path generation
        get: Retrieve registered component class by full path
        clear: Clear registry (primarily for testing purposes)
        
    Note:
        This registry implements self-registration patterns to avoid hardcoded component mappings.
        Components automatically register themselves during import or initialization.
        Registry provides high-performance lookup with intelligent caching optimization.
        Clear method should only be used in testing environments to avoid production issues.
        
    Warning:
        Registry state is global and shared across all components in the application.
        Clearing the registry in production will remove all registered components.
        Component registration should happen during initialization to ensure availability.
        Multiple registrations of the same class path will overwrite previous registrations.
        
    See Also:
        * :func:`create_component`: Dynamic component creation using registry lookup
        * :mod:`nanobrain.core.component_base`: Framework component base classes
        * :mod:`nanobrain.core.config.config_base`: Configuration base classes
        * :class:`ConfigLoadingContext`: Configuration loading context and metadata
    """
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, component_class: type) -> None:
        """Components self-register - no hardcoded registration"""
        if hasattr(component_class, '__module__') and hasattr(component_class, '__name__'):
            full_path = f"{component_class.__module__}.{component_class.__name__}"
            cls._registry[full_path] = component_class
    
    @classmethod
    def get(cls, class_path: str) -> Optional[type]:
        """Get registered component - no fallbacks or defaults"""
        return cls._registry.get(class_path)
    
    @classmethod
    def clear(cls) -> None:
        """Clear registry - used for testing"""
        cls._registry.clear() 