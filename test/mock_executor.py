"""
Mock ExecutorBase for testing purposes.
"""

from typing import Any, Set


class MockExecutorBase:
    """Mock ExecutorBase for testing."""
    
    def __init__(self, **kwargs):
        """Initialize the MockExecutorBase."""
        self.runnable_types = set(["Step", "Workflow", "Agent"])
        self.energy_level = 1.0
        self.config_manager = MockConfigManager()
        self.system_modulator = MockSystemModulator()
        self.directory_tracer = MockDirectoryTracer()
    
    def can_execute(self, runnable_type: str) -> bool:
        """Check if this executor can execute the specified runnable type."""
        return runnable_type in self.runnable_types
    
    def execute(self, runnable) -> Any:
        """Execute a runnable object."""
        # Just return a mock result
        return "Mock execution result"
    
    def recover_energy(self):
        """Recover energy for this executor."""
        self.energy_level = 1.0
    
    def get_modulator_effect(self, name: str) -> float:
        """Get the effect of a modulator."""
        return 0.5
    
    def get_config(self, class_dir: str = None) -> dict:
        """Get the configuration for a class."""
        return {"mock_config": True}
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Update the configuration."""
        return True


class MockConfigManager:
    """Mock ConfigManager for testing."""
    
    def __init__(self, **kwargs):
        """Initialize the MockConfigManager."""
        self._adaptability = 0.5
    
    def get_config(self, class_name: str) -> dict:
        """Get the configuration for a class."""
        return {"mock_config": True}
    
    def update_config(self, updates: dict, adaptability_threshold: float = 0.3) -> bool:
        """Update the configuration."""
        return True
    
    @property
    def adaptability(self) -> float:
        """Get the adaptability level."""
        return self._adaptability
    
    @adaptability.setter
    def adaptability(self, value: float):
        """Set the adaptability level."""
        self._adaptability = max(0.0, min(1.0, value))


class MockSystemModulator:
    """Mock SystemModulator for testing."""
    
    def __init__(self, **kwargs):
        """Initialize the MockSystemModulator."""
        self.modulators = {
            "energy": 1.0,
            "attention": 0.8,
            "plasticity": 0.6
        }
    
    def get_modulator(self, name: str) -> float:
        """Get the value of a modulator."""
        return self.modulators.get(name, 0.5)
    
    def set_modulator(self, name: str, value: float):
        """Set the value of a modulator."""
        self.modulators[name] = max(0.0, min(1.0, value))
    
    def update_from_event(self, event: str):
        """Update modulators based on an event."""
        pass
    
    def apply_regulation(self):
        """Apply regulation to keep modulators within acceptable ranges."""
        pass


class MockDirectoryTracer:
    """Mock DirectoryTracer for testing."""
    
    def __init__(self, **kwargs):
        """Initialize the MockDirectoryTracer."""
        self.module_name = "mock_module"
        self.relative_path = "mock/relative/path"
    
    def get_relative_path(self) -> str:
        """Get the relative path from the package root."""
        return self.relative_path
    
    def get_absolute_path(self) -> str:
        """Get the absolute path in the filesystem."""
        return f"/absolute/{self.relative_path}" 