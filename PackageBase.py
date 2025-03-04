
from mixins import DirectoryTracerMixin, ConfigurableMixin, RunnableMixin
from enums import ExecutorBase, ComponentState


class PackageBase(DirectoryTracerMixin, ConfigurableMixin, RunnableMixin):
    """
    Base class for self-contained units with dependencies.
    
    Biological analogy: Functional module in brain organization.
    Justification: Like how the brain has functional modules that evolved to
    handle specific tasks with defined connections to other modules, packages
    are modular units with defined dependencies and interfaces.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor=executor, **kwargs)
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
        return (self.state == ComponentState.INACTIVE or 
                self.state == ComponentState.ENHANCED) and not self.running