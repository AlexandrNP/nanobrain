"""
Component Factory for NanoBrain Framework

Creates components from YAML configurations with advanced validation.
"""

import logging
import yaml
from typing import Any, Dict, List, Optional, Union, Type, Callable
from pathlib import Path
from enum import Enum
import asyncio

try:
    from ..core import (
        Step, SimpleStep, Agent, SimpleAgent,
        DataUnitMemory, DataUnitFile, DataUnitString, DataUnitStream,
        ManualTrigger, TimerTrigger, AllDataReceivedTrigger, DataUpdatedTrigger,
        DirectLink, LangChainTool,
        LocalExecutor, ThreadExecutor, ParslExecutor, ProcessExecutor
    )
except ImportError:
    # Fallback for when running from project root with src in path
    from core import (
        Step, SimpleStep, Agent, SimpleAgent,
        DataUnitMemory, DataUnitFile, DataUnitString, DataUnitStream,
        ManualTrigger, TimerTrigger, AllDataReceivedTrigger, DataUpdatedTrigger,
        DirectLink, LangChainTool,
        LocalExecutor, ThreadExecutor, ParslExecutor, ProcessExecutor
    )

# Lazy import function for library agents
def _import_library_agents():
    """Lazy import of library agents to avoid import issues during module initialization."""
    try:
        import sys
        from pathlib import Path
        library_path = Path(__file__).parent.parent.parent / "library"
        if str(library_path) not in sys.path:
            sys.path.insert(0, str(library_path))
        from agents.specialized import CodeWriterAgent, FileWriterAgent
        return CodeWriterAgent, FileWriterAgent
    except ImportError:
        # Fallback for when running from project root with src in path
        try:
            import sys
            from pathlib import Path
            library_path = Path("library")
            if str(library_path) not in sys.path:
                sys.path.insert(0, str(library_path))
            from agents.specialized import CodeWriterAgent, FileWriterAgent
            return CodeWriterAgent, FileWriterAgent
        except ImportError:
            # Return None if agents can't be imported
            return None, None
from .yaml_config import YAMLConfig
from .schema_validator import SchemaValidator, ConfigSchema, create_schema_from_yaml

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    """Types of components that can be created."""
    STEP = "step"
    WORKFLOW = "workflow"
    AGENT = "agent"
    DATA_UNIT = "data_unit"
    TRIGGER = "trigger"
    LINK = "link"
    EXECUTOR = "executor"
    TOOL = "tool"


class ComponentFactory:
    """
    Factory for creating NanoBrain components from YAML configurations.
    
    Supports advanced validation, parameter schemas, and custom validators.
    """
    
    def __init__(self):
        """Initialize the component factory."""
        self.component_registry: Dict[str, Any] = {}
        self.custom_classes: Dict[str, Type] = {}
        self.schema_validators: Dict[str, SchemaValidator] = {}
        self.config_search_paths: List[Path] = [
            Path("library/agents/specialized/config"),  # Library agent configurations
            Path("library/agents/conversational/config"),  # Library conversational agent configurations
            Path("library/agents/enhanced/config"),     # Library enhanced agent configurations
            Path("src/config/templates"),   # General templates
            Path("config/templates"),       # General templates (relative)
            Path("templates"),
            Path(".")
        ]
        
        # Register default component classes
        self._register_default_classes()
        
        logger.debug("ComponentFactory initialized")
    
    def _register_default_classes(self) -> None:
        """Register default component classes."""
        # Import library agents lazily
        CodeWriterAgent, FileWriterAgent = _import_library_agents()
        
        # Agent classes
        agent_classes = {
            "SimpleAgent": SimpleAgent,
            "Agent": Agent,
        }
        
        # Add library agents if available
        if CodeWriterAgent is not None:
            agent_classes.update({
                "CodeWriterAgent": CodeWriterAgent,
                "agents.CodeWriterAgent": CodeWriterAgent,  # Support module.class format
            })
        
        if FileWriterAgent is not None:
            agent_classes.update({
                "FileWriterAgent": FileWriterAgent,
                "agents.FileWriterAgent": FileWriterAgent,  # Support module.class format
            })
        
        self.custom_classes.update(agent_classes)
        
        # Data unit classes
        self.custom_classes.update({
            "DataUnitMemory": DataUnitMemory,
            "DataUnitFile": DataUnitFile,
            "DataUnitString": DataUnitString,
            "DataUnitStream": DataUnitStream,
        })
        
        # Trigger classes
        self.custom_classes.update({
            "TriggerManual": ManualTrigger,
            "TriggerTimer": TimerTrigger,
            "AllDataReceivedTrigger": AllDataReceivedTrigger,
        })
        
        # Other classes
        self.custom_classes.update({
            "Step": Step,
            "Link": DirectLink,
            "LangChainTool": LangChainTool,
            "ExecutorLocal": LocalExecutor,
            "ExecutorParsl": ParslExecutor,
            "ProcessExecutor": ProcessExecutor,
        })
    
    def register_class(self, name: str, cls: Type) -> None:
        """Register a custom class for component creation."""
        self.custom_classes[name] = cls
        logger.debug(f"Registered custom class: {name}")
    
    def register_schema_validator(self, component_type: str, validator: SchemaValidator) -> None:
        """Register a schema validator for a component type."""
        self.schema_validators[component_type] = validator
        logger.debug(f"Registered schema validator for: {component_type}")
    
    def load_schema_from_file(self, component_type: str, schema_path: Union[str, Path]) -> None:
        """Load and register a schema validator from file."""
        validator = SchemaValidator()
        validator.load_schema_from_yaml(schema_path)
        self.register_schema_validator(component_type, validator)
    
    def add_config_search_path(self, path: Union[str, Path]) -> None:
        """Add a path to search for configuration files."""
        path = Path(path)
        if path not in self.config_search_paths:
            self.config_search_paths.append(path)
            logger.debug(f"Added config search path: {path}")
    
    def find_config_file(self, filename: str) -> Optional[Path]:
        """Find a configuration file in the search paths."""
        for search_path in self.config_search_paths:
            config_path = search_path / filename
            if config_path.exists():
                return config_path
        return None
    
    def validate_config(self, config: Dict[str, Any], component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration using registered schema validators.
        
        Args:
            config: Configuration to validate
            component_type: Type of component (for schema selection)
            
        Returns:
            Validated configuration
        """
        if component_type and component_type in self.schema_validators:
            validator = self.schema_validators[component_type]
            return validator.validate_config(config)
        
        # Try to infer component type from config
        if 'class' in config:
            class_name = config['class']
            if class_name in ['SimpleAgent', 'CodeWriterAgent', 'FileWriterAgent', 'Agent']:
                component_type = 'agent'
            elif class_name == 'Step':
                component_type = 'step'
            
            if component_type and component_type in self.schema_validators:
                validator = self.schema_validators[component_type]
                return validator.validate_config(config)
        
        logger.debug("No schema validator found, skipping validation")
        return config
    
    def validate_parameters(self, operation: str, parameters: Dict[str, Any], component_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate operation-specific parameters.
        
        Args:
            operation: Operation name
            parameters: Parameters to validate
            component_type: Type of component (for schema selection)
            
        Returns:
            Validated parameters
        """
        if component_type and component_type in self.schema_validators:
            validator = self.schema_validators[component_type]
            return validator.validate_parameters(operation, parameters)
        
        logger.debug(f"No schema validator found for {component_type}, skipping parameter validation")
        return parameters
    
    def create_component(self, component_type: ComponentType, config: Dict[str, Any], name: Optional[str] = None) -> Any:
        """
        Create a component from configuration.
        
        Args:
            component_type: Type of component to create
            config: Component configuration
            name: Optional name for the component
            
        Returns:
            Created component instance
        """
        if isinstance(component_type, str):
            component_type = ComponentType(component_type)
        
        # Validate configuration
        validated_config = self.validate_config(config, component_type.value)
        
        # Create component based on type
        if component_type == ComponentType.AGENT:
            component = self._create_agent(validated_config, name)
        elif component_type == ComponentType.STEP:
            component = self._create_step(validated_config, name)
        elif component_type == ComponentType.WORKFLOW:
            component = self._create_workflow(validated_config, name)
        elif component_type == ComponentType.DATA_UNIT:
            component = self._create_data_unit(validated_config, name)
        elif component_type == ComponentType.TRIGGER:
            component = self._create_trigger(validated_config, name)
        elif component_type == ComponentType.LINK:
            component = self._create_link(validated_config, name)
        elif component_type == ComponentType.EXECUTOR:
            component = self._create_executor(validated_config, name)
        elif component_type == ComponentType.TOOL:
            component = self._create_tool(validated_config, name)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
        
        # Register component if name is provided
        if name:
            self.component_registry[name] = component
        
        return component
    
    def _create_agent(self, config: Dict[str, Any], name: Optional[str] = None) -> Agent:
        """Create an agent from configuration."""
        agent_class_name = config.get('class', config.get('type', 'SimpleAgent'))
        
        if agent_class_name not in self.custom_classes:
            raise ValueError(f"Unknown agent class: {agent_class_name}")
        
        agent_class = self.custom_classes[agent_class_name]
        
        # Extract agent configuration
        agent_config_dict = config.get('config', {})
        if name and 'name' not in agent_config_dict:
            agent_config_dict['name'] = name
        
        # Include prompt_templates from top level if available
        if 'prompt_templates' in config:
            agent_config_dict['prompt_templates'] = config['prompt_templates']
        
        # Create AgentConfig object
        from nanobrain.core.agent import AgentConfig
        agent_config = AgentConfig(**agent_config_dict)
        
        # Create agent instance
        agent = agent_class(config=agent_config)
        
        # Register tools if specified
        tools = config.get('tools', [])
        for tool_config in tools:
            tool = self._create_tool(tool_config)
            agent.register_tool(tool)
        
        logger.debug(f"Created agent: {agent_class_name}")
        return agent
    
    def _create_step(self, config: Dict[str, Any], name: Optional[str] = None) -> Step:
        """Create a step from configuration."""
        step_type = config.get('type', 'SimpleStep')
        step_config_dict = config.get('config', {})
        
        if name and 'name' not in step_config_dict:
            step_config_dict['name'] = name

        # Create agent if specified
        agent = None
        if 'agent' in config:
            agent_config = config['agent']
            agent = self._create_agent(agent_config)

        # Create StepConfig object
        from nanobrain.core.step import StepConfig
        step_config = StepConfig(**step_config_dict)

        # Create step instance based on type
        if step_type == 'SimpleStep':
            from nanobrain.core.step import SimpleStep
            step = SimpleStep(config=step_config, agent=agent)
        else:
            raise ValueError(f"Unknown step type: {step_type}")

        # Set up input data units
        if 'input_configs' in config:
            for input_name, input_config in config['input_configs'].items():
                data_unit = self._create_data_unit(input_config, f"{step.name}_{input_name}")
                step.register_input_data_unit(input_name, data_unit)

        # Set up output data unit
        if 'output_config' in config:
            output_data_unit = self._create_data_unit(config['output_config'], f"{step.name}_output")
            step.register_output_data_unit(output_data_unit)

        return step
    
    def _create_workflow(self, config: Dict[str, Any], name: Optional[str] = None) -> Any:
        """Create a workflow from configuration."""
        from ..workflow import Workflow, WorkflowConfig
        
        # Create WorkflowConfig object
        if name and 'name' not in config:
            config['name'] = name
        
        workflow_config = WorkflowConfig(**config)
        
        # Create workflow instance
        workflow = Workflow(workflow_config)
        
        logger.debug(f"Created workflow: {workflow_config.name}")
        return workflow
    
    def _create_data_unit(self, config: Dict[str, Any], name: Optional[str] = None) -> Any:
        """Create a data unit from configuration."""
        data_type = config.get('data_type', 'memory')
        
        # Convert string to enum if needed
        if isinstance(data_type, str):
            from nanobrain.core.data_unit import DataUnitType, DataUnitConfig
            try:
                data_type_enum = DataUnitType(data_type.lower())
            except ValueError:
                raise ValueError(f"Unknown data unit type: {data_type}")
        else:
            data_type_enum = data_type
        
        # Create DataUnitConfig object
        from nanobrain.core.data_unit import DataUnitConfig
        data_config = DataUnitConfig(**config)
        
        # Create data unit based on type
        if data_type_enum == DataUnitType.MEMORY:
            return DataUnitMemory(config=data_config, name=name or "memory_data")
        elif data_type_enum == DataUnitType.FILE:
            file_path = config.get('file_path', 'data.txt')
            return DataUnitFile(file_path=file_path, config=data_config, name=name or "file_data")
        elif data_type_enum == DataUnitType.STRING:
            initial_value = config.get('initial_value', '')
            return DataUnitString(initial_value=initial_value, config=data_config, name=name or "string_data")
        elif data_type_enum == DataUnitType.STREAM:
            return DataUnitStream(config=data_config, name=name or "stream_data")
        else:
            raise ValueError(f"Unsupported data unit type: {data_type_enum}")
    
    def _create_trigger(self, config: Dict[str, Any], name: Optional[str] = None) -> Any:
        """Create a trigger from configuration."""
        trigger_type = config.get('trigger_type', 'manual')
        
        # Convert string to enum if needed
        if isinstance(trigger_type, str):
            from nanobrain.core.trigger import TriggerType, TriggerConfig
            try:
                trigger_type_enum = TriggerType(trigger_type.lower())
            except ValueError:
                raise ValueError(f"Unknown trigger type: {trigger_type}")
        else:
            trigger_type_enum = trigger_type
        
        # Create TriggerConfig object
        from nanobrain.core.trigger import TriggerConfig
        trigger_config = TriggerConfig(**config)
        
        # Create trigger based on type
        if trigger_type_enum == TriggerType.MANUAL:
            return ManualTrigger(config=trigger_config, name=name or "manual_trigger")
        elif trigger_type_enum == TriggerType.TIMER:
            interval = config.get('interval', 60)
            return TimerTrigger(interval_ms=interval * 1000, config=trigger_config, name=name or "timer_trigger")
        elif trigger_type_enum == TriggerType.ALL_DATA_RECEIVED:
            data_units = config.get('data_units', [])
            return AllDataReceivedTrigger(data_units=data_units, config=trigger_config, name=name or "all_data_trigger")
        elif trigger_type_enum == TriggerType.DATA_UPDATED:
            data_units = config.get('data_units', [])
            return DataUpdatedTrigger(data_units=data_units, config=trigger_config, name=name or "data_updated_trigger")
        else:
            raise ValueError(f"Unsupported trigger type: {trigger_type_enum}")
    
    def _create_link(self, config: Dict[str, Any], name: Optional[str] = None) -> DirectLink:
        """Create a link from configuration."""
        # Get source and target components from registry
        source_name = config.get('source', config.get('source_id', ''))
        target_name = config.get('target', config.get('target_id', ''))
        
        source = self.component_registry.get(source_name)
        target = self.component_registry.get(target_name)
        
        if not source:
            logger.warning(f"Source component '{source_name}' not found in registry, using placeholder")
            # Create a simple placeholder object with a name attribute
            class Placeholder:
                def __init__(self, name):
                    self.name = name
            source = Placeholder(f"placeholder_source_{source_name}")
        
        if not target:
            logger.warning(f"Target component '{target_name}' not found in registry, using placeholder")
            # Create a simple placeholder object with a name attribute
            class Placeholder:
                def __init__(self, name):
                    self.name = name
            target = Placeholder(f"placeholder_target_{target_name}")
        
        # Create LinkConfig object
        from nanobrain.core.link import LinkConfig, LinkType
        link_config_dict = {k: v for k, v in config.items() if k not in ['source', 'target', 'source_id', 'target_id', 'name']}
        link_config_dict['link_type'] = LinkType.DIRECT
        link_config = LinkConfig(**link_config_dict)
        
        return DirectLink(
            source=source,
            target=target,
            config=link_config,
            name=name or "link"
        )
    
    def _create_executor(self, config: Dict[str, Any], name: Optional[str] = None) -> Any:
        """Create an executor from configuration."""
        executor_type = config.get('executor_type', 'local')
        
        # Convert string to enum if needed
        if isinstance(executor_type, str):
            from nanobrain.core.executor import ExecutorType, ExecutorConfig
            try:
                executor_type_enum = ExecutorType(executor_type.lower())
            except ValueError:
                raise ValueError(f"Unknown executor type: {executor_type}")
        else:
            executor_type_enum = executor_type
        
        # Create ExecutorConfig object
        from nanobrain.core.executor import ExecutorConfig
        executor_config = ExecutorConfig(**config)
        
        # Create executor based on type
        if executor_type_enum == ExecutorType.LOCAL:
            return LocalExecutor(config=executor_config)
        elif executor_type_enum == ExecutorType.THREAD:
            return ThreadExecutor(config=executor_config)
        elif executor_type_enum == ExecutorType.PARSL:
            return ParslExecutor(config=executor_config)
        elif executor_type_enum == ExecutorType.PROCESS:
            return ProcessExecutor(config=executor_config)
        else:
            raise ValueError(f"Unsupported executor type: {executor_type_enum}")
    
    def _create_tool(self, config: Dict[str, Any], name: Optional[str] = None) -> Any:
        """Create a tool from configuration."""
        tool_class_name = config.get('class', 'LangChainTool')
        
        if tool_class_name not in self.custom_classes:
            raise ValueError(f"Unknown tool class: {tool_class_name}")
        
        tool_class = self.custom_classes[tool_class_name]
        
        # Create tool instance
        tool_config = config.get('config', {})
        if name and 'name' not in tool_config:
            tool_config['name'] = name
        
        return tool_class(**tool_config)
    
    def create_from_yaml_file(self, yaml_path: Union[str, Path], component_name: Optional[str] = None) -> Any:
        """
        Create a component from a YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            component_name: Optional name for the component
            
        Returns:
            Created component instance
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            # Try to find the file in search paths
            found_path = self.find_config_file(yaml_path.name)
            if found_path:
                yaml_path = found_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Determine component type from config
        component_type = self._infer_component_type(config)
        
        return self.create_component(component_type, config, component_name)
    
    def _infer_component_type(self, config: Dict[str, Any]) -> ComponentType:
        """Infer component type from configuration."""
        # Check for explicit type
        if 'type' in config:
            type_value = config['type']
            # Handle class names that should map to component types
            if type_value in ['SimpleAgent', 'CodeWriterAgent', 'FileWriterAgent', 'Agent']:
                return ComponentType.AGENT
            elif type_value in ['SimpleStep', 'Step', 'TransformStep']:
                return ComponentType.STEP
            else:
                try:
                    return ComponentType(type_value)
                except ValueError:
                    # If it's not a valid ComponentType, try to infer from class name
                    pass
        
        # Infer from class name
        class_name = config.get('class', '')
        if class_name in ['SimpleAgent', 'CodeWriterAgent', 'FileWriterAgent', 'Agent']:
            return ComponentType.AGENT
        elif class_name in ['SimpleStep', 'Step', 'TransformStep']:
            return ComponentType.STEP
        elif class_name in ['Workflow'] or 'steps' in config or 'execution_strategy' in config:
            return ComponentType.WORKFLOW
        elif 'data_type' in config:
            return ComponentType.DATA_UNIT
        elif 'trigger_type' in config:
            return ComponentType.TRIGGER
        elif 'executor_type' in config:
            return ComponentType.EXECUTOR
        elif 'source_id' in config and 'target_id' in config:
            return ComponentType.LINK
        
        # Default to agent if has agent-like config
        if 'config' in config and any(key in config['config'] for key in ['model', 'system_prompt']):
            return ComponentType.AGENT
        
        # Default to step
        return ComponentType.STEP
    
    def create_workflow_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Create a complete workflow from a YAML file.
        
        Args:
            yaml_path: Path to workflow YAML file
            
        Returns:
            Dictionary containing all created components
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            found_path = self.find_config_file(yaml_path.name)
            if found_path:
                yaml_path = found_path
            else:
                raise FileNotFoundError(f"Workflow file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            workflow_config = yaml.safe_load(f)
        
        workflow = {}
        
        # Create executors
        executors = workflow_config.get('executors', [])
        if isinstance(executors, dict):
            # Handle dictionary format: {"name": {config}}
            for name, executor_config in executors.items():
                executor_config['name'] = name  # Ensure name is set
                executor = self.create_component(ComponentType.EXECUTOR, executor_config, name)
                workflow[name] = executor
                self.component_registry[name] = executor
        else:
            # Handle list format: [{"name": "...", config}]
            for executor_config in executors:
                name = executor_config.get('name', f"executor_{len(workflow)}")
                executor = self.create_component(ComponentType.EXECUTOR, executor_config, name)
                workflow[name] = executor
                self.component_registry[name] = executor
        
        # Create data units
        data_units = workflow_config.get('data_units', [])
        if isinstance(data_units, dict):
            for name, data_config in data_units.items():
                data_config['name'] = name
                data_unit = self.create_component(ComponentType.DATA_UNIT, data_config, name)
                workflow[name] = data_unit
                self.component_registry[name] = data_unit
        else:
            for data_config in data_units:
                name = data_config.get('name', f"data_{len(workflow)}")
                data_unit = self.create_component(ComponentType.DATA_UNIT, data_config, name)
                workflow[name] = data_unit
                self.component_registry[name] = data_unit
        
        # Create triggers
        triggers = workflow_config.get('triggers', [])
        if isinstance(triggers, dict):
            for name, trigger_config in triggers.items():
                trigger_config['name'] = name
                trigger = self.create_component(ComponentType.TRIGGER, trigger_config, name)
                workflow[name] = trigger
                self.component_registry[name] = trigger
        else:
            for trigger_config in triggers:
                name = trigger_config.get('name', f"trigger_{len(workflow)}")
                trigger = self.create_component(ComponentType.TRIGGER, trigger_config, name)
                workflow[name] = trigger
                self.component_registry[name] = trigger
        
        # Create agents
        agents = workflow_config.get('agents', [])
        if isinstance(agents, dict):
            for name, agent_config in agents.items():
                agent_config['name'] = name
                agent = self.create_component(ComponentType.AGENT, agent_config, name)
                workflow[name] = agent
                self.component_registry[name] = agent
        else:
            for agent_config in agents:
                name = agent_config.get('name', f"agent_{len(workflow)}")
                agent = self.create_component(ComponentType.AGENT, agent_config, name)
                workflow[name] = agent
                self.component_registry[name] = agent
        
        # Create steps
        steps = workflow_config.get('steps', [])
        if isinstance(steps, dict):
            for name, step_config in steps.items():
                step_config['name'] = name
                step = self.create_component(ComponentType.STEP, step_config, name)
                workflow[name] = step
                self.component_registry[name] = step
        else:
            for step_config in steps:
                name = step_config.get('name', f"step_{len(workflow)}")
                step = self.create_component(ComponentType.STEP, step_config, name)
                workflow[name] = step
                self.component_registry[name] = step
        
        # Create links
        links = workflow_config.get('links', [])
        if isinstance(links, dict):
            for name, link_config in links.items():
                link_config['name'] = name
                link = self.create_component(ComponentType.LINK, link_config, name)
                workflow[name] = link
                self.component_registry[name] = link
        else:
            for link_config in links:
                name = link_config.get('name', f"link_{len(workflow)}")
                link = self.create_component(ComponentType.LINK, link_config, name)
                workflow[name] = link
                self.component_registry[name] = link
        
        logger.info(f"Created workflow with {len(workflow)} components")
        return workflow
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a component from the registry."""
        return self.component_registry.get(name)
    
    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self.component_registry.keys())
    
    def clear_registry(self) -> None:
        """Clear the component registry."""
        self.component_registry.clear()
        logger.debug("Component registry cleared")
    
    def shutdown_components(self) -> None:
        """Shutdown all registered components."""
        async def _async_shutdown():
            for name, component in self.component_registry.items():
                if hasattr(component, 'shutdown'):
                    try:
                        if asyncio.iscoroutinefunction(component.shutdown):
                            await component.shutdown()
                        else:
                            component.shutdown()
                        logger.debug(f"Shutdown component: {name}")
                    except Exception as e:
                        logger.error(f"Error shutting down component {name}: {e}")
        
        # Run the async shutdown
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                asyncio.create_task(_async_shutdown())
            else:
                loop.run_until_complete(_async_shutdown())
        except RuntimeError:
            # No event loop, create a new one
            asyncio.run(_async_shutdown())
        
        # Clear the registry after shutdown
        self.clear_registry()


# Global factory instance
_global_factory: Optional[ComponentFactory] = None


def get_factory() -> ComponentFactory:
    """Get the global component factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = ComponentFactory()
    return _global_factory


def set_factory(factory: ComponentFactory) -> None:
    """Set the global component factory instance."""
    global _global_factory
    _global_factory = factory


def create_component_from_yaml(yaml_path: Union[str, Path], component_name: Optional[str] = None) -> Any:
    """Create a component from YAML using the global factory."""
    return get_factory().create_from_yaml_file(yaml_path, component_name)


def create_workflow_from_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Create a workflow from YAML using the global factory."""
    return get_factory().create_workflow_from_yaml(yaml_path) 