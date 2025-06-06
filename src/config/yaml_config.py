"""
YAML Configuration System for NanoBrain Framework

Provides loading, saving, and validation of YAML configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field, ValidationError, ConfigDict

logger = logging.getLogger(__name__)


class YAMLConfig(BaseModel):
    """
    Base configuration class with YAML serialization support.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow"
    )
    
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to YAML string or save to file.
        
        Args:
            file_path: Optional file path to save YAML
            
        Returns:
            YAML string representation
        """
        # Convert to dict, handling Pydantic models
        config_dict = self._to_serializable_dict()
        
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
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        result = {}
        
        for key, value in self.dict().items():
            if isinstance(value, BaseModel):
                result[key] = value.dict()
            elif isinstance(value, list):
                result[key] = [
                    item.dict() if isinstance(item, BaseModel) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[key] = {
                    k: v.dict() if isinstance(v, BaseModel) else v
                    for k, v in value.items()
                }
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def from_yaml(cls, yaml_content: Union[str, Path]) -> 'YAMLConfig':
        """
        Create configuration from YAML string or file.
        
        Args:
            yaml_content: YAML string or file path
            
        Returns:
            Configuration instance
        """
        if isinstance(yaml_content, (str, Path)) and Path(yaml_content).exists():
            # Load from file
            yaml_str = Path(yaml_content).read_text()
            logger.info(f"Configuration loaded from {yaml_content}")
        else:
            # Treat as YAML string
            yaml_str = str(yaml_content)
        
        # Parse YAML
        try:
            config_dict = yaml.safe_load(yaml_str)
            return cls(**config_dict)
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
    
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
    
    def merge_with(self, other: 'YAMLConfig') -> 'YAMLConfig':
        """
        Merge with another configuration.
        
        Args:
            other: Other configuration to merge
            
        Returns:
            New merged configuration
        """
        self_dict = self.dict()
        other_dict = other.dict()
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(self_dict, other_dict)
        
        return self.__class__(**merged_dict)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


class WorkflowConfig(YAMLConfig):
    """Configuration for complete workflows."""
    
    name: str
    description: str = ""
    version: str = "1.0.0"
    steps: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    agents: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    executors: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    data_units: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    triggers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    def get_step_config(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific step."""
        return self.steps.get(step_name)
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name)
    
    def add_step(self, name: str, config: Dict[str, Any]) -> None:
        """Add a step configuration."""
        self.steps[name] = config
        logger.debug(f"Added step configuration: {name}")
    
    def add_agent(self, name: str, config: Dict[str, Any]) -> None:
        """Add an agent configuration."""
        self.agents[name] = config
        logger.debug(f"Added agent configuration: {name}")
    
    def add_link(self, link_config: Dict[str, Any]) -> None:
        """Add a link configuration."""
        self.links.append(link_config)
        logger.debug(f"Added link configuration")
    
    def validate_references(self) -> List[str]:
        """
        Validate that all references between components exist.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check link references
        for i, link in enumerate(self.links):
            source = link.get('source')
            target = link.get('target')
            
            if source and source not in self.steps and source not in self.agents:
                errors.append(f"Link {i}: source '{source}' not found")
            
            if target and target not in self.steps and target not in self.agents:
                errors.append(f"Link {i}: target '{target}' not found")
        
        # Check step executor references
        for step_name, step_config in self.steps.items():
            executor_name = step_config.get('executor')
            if executor_name and executor_name not in self.executors:
                errors.append(f"Step '{step_name}': executor '{executor_name}' not found")
        
        return errors


def load_config(file_path: Union[str, Path], config_class: type = WorkflowConfig) -> YAMLConfig:
    """
    Load configuration from YAML file.
    
    Args:
        file_path: Path to YAML file
        config_class: Configuration class to use
        
    Returns:
        Configuration instance
    """
    try:
        return config_class.from_yaml(file_path)
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise


def save_config(config: YAMLConfig, file_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration instance
        file_path: Path to save file
    """
    try:
        config.to_yaml(file_path)
    except Exception as e:
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        raise


def create_example_config() -> WorkflowConfig:
    """
    Create an example workflow configuration.
    
    Returns:
        Example configuration
    """
    return WorkflowConfig(
        name="example_workflow",
        description="Example NanoBrain workflow configuration",
        version="1.0.0",
        
        # Define executors
        executors={
            "local": {
                "executor_type": "local",
                "max_workers": 4
            },
            "parsl_hpc": {
                "executor_type": "parsl",
                "max_workers": 16,
                "parsl_config": {
                    "provider": "slurm",
                    "nodes_per_block": 1,
                    "cores_per_node": 16
                }
            }
        },
        
        # Define data units
        data_units={
            "input_data": {
                "data_type": "memory",
                "persistent": False
            },
            "processed_data": {
                "data_type": "file",
                "persistent": True,
                "file_path": "/tmp/processed_data.json"
            }
        },
        
        # Define triggers
        triggers={
            "data_trigger": {
                "trigger_type": "data_updated",
                "debounce_ms": 100
            },
            "timer_trigger": {
                "trigger_type": "timer",
                "timer_interval_ms": 5000
            }
        },
        
        # Define agents
        agents={
            "code_writer": {
                "agent_type": "simple",
                "name": "code_writer",
                "description": "Agent for writing code",
                "model": "gpt-4",
                "system_prompt": "You are a helpful code writing assistant.",
                "tools": []
            },
            "file_writer": {
                "agent_type": "simple", 
                "name": "file_writer",
                "description": "Agent for file operations",
                "model": "gpt-3.5-turbo",
                "system_prompt": "You help with file operations.",
                "tools": []
            }
        },
        
        # Define steps
        steps={
            "data_processor": {
                "step_type": "transform",
                "name": "data_processor",
                "description": "Process input data",
                "executor": "local",
                "input_data_units": [{"data_type": "memory"}],
                "output_data_units": [{"data_type": "memory"}],
                "trigger_config": {
                    "trigger_type": "data_updated"
                }
            },
            "hpc_analyzer": {
                "step_type": "simple",
                "name": "hpc_analyzer", 
                "description": "Heavy computation step",
                "executor": "parsl_hpc",
                "input_data_units": [{"data_type": "memory"}],
                "output_data_units": [{"data_type": "file", "file_path": "/tmp/results.json"}],
                "trigger_config": {
                    "trigger_type": "all_data_received"
                }
            }
        },
        
        # Define links
        links=[
            {
                "link_type": "direct",
                "source": "data_processor",
                "target": "hpc_analyzer",
                "name": "processor_to_analyzer"
            }
        ]
    ) 