from WorkingMemory import WorkingMemory
from typing import Dict
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
        self.working_memory = WorkingMemory(capacity=20)  # Cache for loaded configurations
        self.learning_rate = 0.1  # How quickly we adapt to new configs
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
        
        Biological analogy: Building neural representations from concepts.
        Justification: Like how the brain constructs neural representations
        from perceived concepts, the config loader builds object structures
        from configuration descriptions.
        """
        constructed = {}
        
        # Process objects based on dependencies
        dependencies = self._analyze_dependencies(config)
        
        # Process in dependency order with parallel construction where possible
        for level in sorted(set(dependencies.values())):
            level_items = {k: v for k, v in dependencies.items() if v == level}
            
            # Items at same level can be constructed in parallel
            tasks = []
            for key in level_items:
                if key in config:
                    task = asyncio.create_task(self._construct_single_object(key, config[key], base_path))
                    tasks.append((key, task))
            
            # Wait for all tasks
            for key, task in tasks:
                constructed[key] = await task
        
        return constructed
    
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
        Analyze dependencies between objects.
        
        Biological analogy: Understanding relationships between concepts.
        Justification: Like how the brain analyzes relationships between
        concepts to understand their hierarchical organization, the config
        loader analyzes relationships between objects to understand their
        dependency structure.
        """
        # Extract dependency information
        direct_dependencies = {}
        for name, obj_config in config.items():
            deps = []
            
            # Look for references to other objects
            for key, value in obj_config.items():
                if isinstance(value, str) and value in config:
                    deps.append(value)
                elif key.endswith('_ref') and value in config:
                    deps.append(value)
            
            direct_dependencies[name] = deps
        
        # Assign dependency levels
        dependency_levels = {}
        
        def assign_level(item, level=0):
            if item in dependency_levels:
                dependency_levels[item] = max(dependency_levels[item], level)
            else:
                dependency_levels[item] = level
                
            # Process items that depend on this item
            for dep_name, dep_list in direct_dependencies.items():
                if item in dep_list:
                    assign_level(dep_name, level + 1)
        
        # Start with items that have no dependencies
        for name, deps in direct_dependencies.items():
            if not deps:
                assign_level(name, 0)
        
        return dependency_levels
    
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