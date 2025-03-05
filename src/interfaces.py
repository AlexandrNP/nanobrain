from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path


class IDirectoryTracer(ABC):
    """
    Interface for components that need to track their location in the codebase.
    
    Biological analogy: Place cells in the hippocampus that encode spatial location.
    Justification: Just as place cells allow an organism to know its location in physical space,
    this interface defines how components can know their location in the codebase's structure.
    """
    @abstractmethod
    def get_relative_path(self) -> str:
        """Returns the relative path from the package root."""
        pass
    
    @abstractmethod
    def get_absolute_path(self) -> str:
        """Returns the absolute path in the filesystem."""
        pass


class IConfigurable(ABC):
    """
    Interface for components that can be configured via external configuration.
    
    Biological analogy: Epigenetic mechanisms that control gene expression.
    Justification: Like how epigenetic modifications determine which genes are expressed
    without changing the DNA sequence, configuration parameters determine component
    behavior without changing the underlying code.
    """
    @abstractmethod
    def get_config(self, class_dir: str = None) -> Dict:
        """Get configuration from specified directory or default location."""
        pass
    
    @abstractmethod
    def update_config(self, updates: Dict, adaptability_threshold: float = 0.3) -> bool:
        """Update configuration if adaptability threshold is met."""
        pass
    
    @property
    @abstractmethod
    def adaptability(self) -> float:
        """Get the adaptability level of the component."""
        pass
    
    @adaptability.setter
    @abstractmethod
    def adaptability(self, value: float):
        """Set the adaptability level of the component."""
        pass


class IRunnable(IConfigurable):
    """
    Interface for components that can be executed.
    
    Biological analogy: Neuron with activation potential.
    Justification: Like how neurons can be activated to process and transmit
    signals, runnable components can be executed to process and transmit data.
    """
    @abstractmethod
    async def invoke(self) -> Any:
        """Execute the component and return the result."""
        pass
    
    @abstractmethod
    def check_runnable_config(self) -> bool:
        """Check if the component is properly configured to run."""
        pass
    
    @property
    @abstractmethod
    def state(self):
        """Get the current state of the runnable component."""
        pass
    
    @state.setter
    @abstractmethod
    def state(self, value):
        """Set the current state of the runnable component."""
        pass
    
    @property
    @abstractmethod
    def running(self) -> bool:
        """Get whether the component is currently running."""
        pass
    
    @running.setter
    @abstractmethod
    def running(self, value: bool):
        """Set whether the component is currently running."""
        pass 