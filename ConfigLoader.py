from WorkingMemory import WorkingMemory
from typing import Dict
from ConfigManager import ConfigManager
from DirectoryTracer import DirectoryTracer
import os
import yaml
import asyncio



class ConfigLoader:
    """
    Loads configuration from YAML files and constructs objects.
    
    Biological analogy: Learning from external information.
    Justification: Like how the brain forms internal models from external
    information, the config loader builds system structures from configuration files.
    """
    def __init__(self):
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path())
        config = self.config_manager.get_config(self.__class__.__name__)
        
        # Initialize with config values or defaults
        memory_capacity = config.get('memory_capacity', 20)
        self.working_memory = WorkingMemory(capacity=memory_capacity)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.concept_network = {}  # Graph of related concepts
    
    @staticmethod
    async def load(config_path: str):
        """
        Recursively loads configuration from YAML file.
        
        Biological analogy: Learning hierarchical concepts from information.
        Justification: Like how the brain builds hierarchical knowledge
        structures from information, the config loader builds hierarchical
        object structures from configuration data.
        """
        loader = ConfigLoader()
        return await loader._load_internal(config_path)
    
    async def _load_internal(self, config_path: str):
        """
        Internal loading with memory and learning.
        
        Biological analogy: Memory-assisted learning.
        Justification: Like how the brain uses existing memories to assist
        in learning new information, the config loader uses cached configurations
        to assist in loading new ones.
        """
        # Check memory first
        cached = self.working_memory.retrieve(config_path)
        if cached:
            return cached
            
        # Load from file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Process configuration
        result = await self._construct_objects(config, os.path.dirname(config_path))
        
        # Store in memory
        self.working_memory.store(config_path, result)
        
        # Update concept network
        self._update_concept_network(config_path, config)
        
        return result
    
    async def _construct_objects(self, config: Dict, base_path: str):
        """
        Constructs objects from configuration.
        
        Biological analogy: Building neural representations from sensory input.
        Justification: Like how the brain constructs internal representations
        from sensory information, the config loader constructs object structures
        from configuration data.
        """
        # Analyze dependencies to determine construction order
        dependency_levels = self._analyze_dependencies(config)
        
        # Sort items by dependency level
        sorted_items = sorted(dependency_levels.items(), key=lambda x: x[1])
        
        # Construct objects in order
        constructed_objects = {}
        for name, _ in sorted_items:
            if name in config:
                constructed_objects[name] = await self._construct_single_object(
                    name, config[name], base_path)
                
        return constructed_objects
    
    async def _construct_single_object(self, name: str, config: Dict, base_path: str):
        """
        Construct a single object with error handling and learning.
        
        Biological analogy: Formation of a specific concept representation.
        Justification: Like how the brain forms specific conceptual representations
        by integrating features and relating to existing concepts, the config loader
        constructs specific objects by interpreting parameters and relating to other objects.
        """
        try:
            # Here we would instantiate the appropriate class based on config
            # For demonstration, we'll return a placeholder
            
            # Recursive loading if needed
            if 'config_path' in config:
                sub_path = os.path.join(base_path, config['config_path'])
                return await self._load_internal(sub_path)
                
            # This would be the actual instantiation based on class specified in config
            return {"name": name, "type": config.get("type", "unknown"), "config": config}
            
        except Exception as e:
            # Learn from failure
            self._learn_from_error(name, config, e)
            raise e
    
    def _analyze_dependencies(self, config: Dict) -> Dict[str, int]:
        """
        Analyzes dependencies between configuration items.
        
        Biological analogy: Understanding relationships between concepts.
        Justification: Like how the brain analyzes relationships between
        concepts to understand hierarchical knowledge, the config loader
        analyzes relationships between configuration items to understand
        construction order.
        """
        # Initialize all items at level 0
        levels = {name: 0 for name in config.keys()}
        
        # Track visited items to detect cycles
        visited = set()
        
        # Process each item
        for name in config:
            if name not in visited:
                assign_level(name)
                
        # Helper function to recursively assign levels
        def assign_level(item, level=0):
            # Mark as visited
            visited.add(item)
            
            # Get dependencies
            dependencies = []
            if item in config and isinstance(config[item], dict):
                dependencies = config[item].get('dependencies', [])
                
            # Process dependencies
            for dep in dependencies:
                if dep not in visited:
                    assign_level(dep, level + 1)
                    
            # Set level to max of current and required
            levels[item] = max(levels.get(item, 0), level)
            
        return levels
    
    def _update_concept_network(self, config_path: str, config: Dict):
        """
        Updates the internal concept network based on new configuration.
        
        Biological analogy: Updating semantic networks with new information.
        Justification: Like how the brain updates its semantic networks when
        learning new relationships between concepts, the config loader updates
        its concept network when learning new relationships between configuration elements.
        """
        # Add config path as a node
        if config_path not in self.concept_network:
            self.concept_network[config_path] = set()
            
        # Add object types as nodes connected to config path
        for name, obj_config in config.items():
            obj_type = obj_config.get('type', 'unknown')
            
            if obj_type not in self.concept_network:
                self.concept_network[obj_type] = set()
                
            # Connect config path to object type
            self.concept_network[config_path].add(obj_type)
            
            # Connect object type to config path
            self.concept_network[obj_type].add(config_path)
    
    def _learn_from_error(self, name: str, config: Dict, error: Exception):
        """
        Update internal knowledge based on errors encountered.
        
        Biological analogy: Learning from mistakes.
        Justification: Like how the brain learns from errors to avoid
        similar mistakes in the future, the config loader learns from
        configuration errors to improve future loading attempts.
        """
        # In a real implementation, this could adapt loading strategies
        # based on common error patterns
        pass