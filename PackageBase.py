from enums import ComponentState
from ExecutorBase import ExecutorBase
from DirectoryTracer import DirectoryTracer
from ConfigManager import ConfigManager
from Runner import Runner

class PackageBase:
    """
    Base class for self-contained units with dependencies.
    
    Biological analogy: Functional module in brain organization.
    Justification: Like how the brain has functional modules that evolved to
    handle specific tasks with defined connections to other modules, packages
    are modular units with defined dependencies and interfaces.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        self.directory_tracer = DirectoryTracer(self.__class__.__module__)
        self.config_manager = ConfigManager(base_path=self.directory_tracer.get_absolute_path(), **kwargs)
        self.runner = Runner(executor, **kwargs)
        self.dependencies = []
        self.dependency_resolution_state = ComponentState.INACTIVE
    
    def check_dependencies(self) -> bool:
        """
        Checks if all dependencies are satisfied.
        
        Biological analogy: Neural module requiring correct inputs.
        Justification: Like how complex neural functions require proper inputs
        from various brain regions, packages require proper dependencies to function.
        """
        # Implementation would check if dependencies are available
        return all(dependency.check_availability() for dependency in self.dependencies)
    
    def check_availability(self) -> bool:
        """
        Checks if this package is available for use by others.
        
        Biological analogy: Neural readiness for activation.
        Justification: Like how neurons must be in the appropriate state
        to respond to incoming signals, packages must be in the appropriate
        state to be used by other components.
        """
        return (self.runner.state == ComponentState.INACTIVE or 
                self.runner.state == ComponentState.ENHANCED) and not self.runner.is_running
                
    def get_relative_path(self) -> str:
        """Delegate to directory tracer."""
        return self.directory_tracer.get_relative_path()
    
    def get_absolute_path(self) -> str:
        """Delegate to directory tracer."""
        return self.directory_tracer.get_absolute_path()
    
    def get_config(self, class_dir: str = None) -> dict:
        """
        Get configuration for this class.
        
        Biological analogy: Localized gene expression.
        Justification: Like how cells use their location to determine which
        genes to express, components use their location to find appropriate
        configuration.
        """
        # Use the class name for configuration lookup
        return self.config_manager.get_config(self.__class__.__name__)
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Delegate to config manager."""
        return self.config_manager.update_config(updates, adaptability_threshold)
    
    async def invoke(self):
        """Delegate to runner."""
        return await self.runner.invoke()