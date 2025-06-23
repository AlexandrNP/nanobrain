"""
Tests for Simplified Component Factory System

Tests the modern from_config pattern and simplified component creation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Import the components we need to test
from nanobrain.core.config.component_factory import ComponentFactory, import_and_create_from_config
from nanobrain.core.executor import LocalExecutor, ExecutorConfig, ExecutorType


class TestSimplifiedComponentFactory:
    """Test the simplified ComponentFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create a component factory for testing."""
        return ComponentFactory()
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_import_and_create_from_config_success(self):
        """Test successful component creation using import_and_create_from_config."""
        # Test with LocalExecutor - use proper ExecutorConfig
        executor_config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=2
        )
        
        executor = import_and_create_from_config(
            "nanobrain.core.executor.LocalExecutor",
            executor_config
        )
        
        assert executor is not None
        assert hasattr(executor, 'name')
        # LocalExecutor name comes from class name, not config
        assert executor.name == "LocalExecutor"
        assert executor.config.max_workers == 2
    
    def test_import_and_create_from_config_invalid_path(self):
        """Test error handling for invalid import paths."""
        executor_config = ExecutorConfig(executor_type=ExecutorType.LOCAL)
        
        # Test short name (not allowed)
        with pytest.raises(ValueError, match="must be a full import path"):
            import_and_create_from_config("LocalExecutor", executor_config)
        
        # Test invalid module
        with pytest.raises(ImportError, match="Cannot import module"):
            import_and_create_from_config("nonexistent.module.Class", executor_config)
        
        # Test invalid class
        with pytest.raises(ImportError, match="Class .* not found"):
            import_and_create_from_config("nanobrain.core.executor.NonexistentClass", executor_config)
    
    def test_import_and_create_from_config_no_from_config_method(self):
        """Test error handling when class doesn't implement from_config."""
        # Test with a class that doesn't have from_config
        with pytest.raises(ValueError, match="must implement from_config method"):
            import_and_create_from_config("builtins.dict", {})
    
    def test_factory_create_component_from_config(self, factory):
        """Test ComponentFactory.create_component_from_config method."""
        executor_config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=3
        )
        
        executor = factory.create_component_from_config(
            "nanobrain.core.executor.LocalExecutor",
            executor_config
        )
        
        assert executor is not None
        assert executor.name == "LocalExecutor"
        assert executor.config.max_workers == 3
    
    def test_factory_create_from_yaml_file(self, factory, temp_config_dir):
        """Test creating component from YAML file."""
        # Create a test YAML config file with proper structure
        config_data = {
            "executor_type": "local",
            "max_workers": 4,
            "timeout": 30.0
        }
        
        yaml_file = temp_config_dir / "test_executor.yml"
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create component from YAML
        executor = factory.create_from_yaml_file(
            yaml_file,
            "nanobrain.core.executor.LocalExecutor"
        )
        
        assert executor is not None
        assert executor.name == "LocalExecutor"
        assert executor.config.max_workers == 4
        assert executor.config.timeout == 30.0
    
    def test_factory_create_from_yaml_file_not_found(self, factory):
        """Test error handling for missing YAML file."""
        with pytest.raises(FileNotFoundError):
            factory.create_from_yaml_file(
                "nonexistent.yml",
                "nanobrain.core.executor.LocalExecutor"
            )


class TestModernFromConfigPattern:
    """Test direct from_config usage pattern (recommended approach)."""
    
    def test_direct_from_config_usage(self):
        """Test the preferred direct from_config pattern."""
        # This is the modern, recommended approach
        executor_config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=8
        )
        
        # Direct usage - no factory needed
        executor = LocalExecutor.from_config(executor_config)
        
        assert executor is not None
        assert executor.name == "LocalExecutor"  # Name comes from class
        assert executor.config.max_workers == 8
    
    def test_from_config_with_dependencies(self):
        """Test from_config pattern with additional dependencies."""
        executor_config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=2
        )
        
        # Example of passing additional dependencies
        executor = LocalExecutor.from_config(
            executor_config,
            custom_dependency="test_value"
        )
        
        assert executor is not None
        assert executor.name == "LocalExecutor"
        assert executor.config.max_workers == 2


class TestMigrationPatterns:
    """Test migration patterns from old to new API."""
    
    def test_migration_from_old_factory_pattern(self):
        """Demonstrate migration from old factory pattern to new approach."""
        
        # OLD PATTERN (no longer supported):
        # factory = get_factory()
        # component = factory.create_component(ComponentType.EXECUTOR, config, "name")
        
        # NEW PATTERN (simplified factory):
        factory = ComponentFactory()
        config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=2
        )
        component = factory.create_component_from_config(
            "nanobrain.core.executor.LocalExecutor",
            config
        )
        
        # BEST PATTERN (direct from_config):
        best_component = LocalExecutor.from_config(config)
        
        assert component is not None
        assert best_component is not None
        assert component.name == best_component.name
        assert component.config.max_workers == best_component.config.max_workers
    
    def test_migration_yaml_loading(self):
        """Demonstrate YAML loading migration."""
        
        # OLD PATTERN (no longer supported):
        # component = create_component_from_yaml("config.yml", "component_name")
        
        # NEW PATTERN (specify class path explicitly):
        config_dict = {
            "executor_type": "local",
            "max_workers": 2,
            "timeout": 15.0
        }
        
        # Method 1: Use factory
        factory = ComponentFactory()
        component1 = factory.create_component_from_config(
            "nanobrain.core.executor.LocalExecutor",
            config_dict
        )
        
        # Method 2: Use direct function
        component2 = import_and_create_from_config(
            "nanobrain.core.executor.LocalExecutor",
            config_dict
        )
        
        # Method 3: Direct from_config (best)
        from nanobrain.core.executor import ExecutorConfig
        config_obj = ExecutorConfig(**config_dict)
        component3 = LocalExecutor.from_config(config_obj)
        
        assert all(comp.name == "LocalExecutor" for comp in [component1, component2, component3])
        assert all(comp.config.max_workers == 2 for comp in [component1, component2, component3])


class TestRealWorldWorkflowCreation:
    """Test real-world workflow creation patterns."""
    
    def test_workflow_component_creation(self):
        """Test creating workflow components using modern pattern."""
        # Create executor for workflow steps
        executor_config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=4
        )
        executor = LocalExecutor.from_config(executor_config)
        
        # In a real workflow, you'd create steps like this:
        # step_config = StepConfig(name="workflow_step")
        # step = SomeStepClass.from_config(step_config, executor=executor)
        
        assert executor is not None
        assert executor.name == "LocalExecutor"
        assert executor.config.max_workers == 4
    
    def test_batch_component_creation(self):
        """Test creating multiple components efficiently."""
        components = []
        
        for i in range(3):
            config = ExecutorConfig(
                executor_type=ExecutorType.LOCAL,
                max_workers=i+1
            )
            executor = LocalExecutor.from_config(config)
            components.append(executor)
        
        assert len(components) == 3
        # All have the same name (LocalExecutor) since that's how the class works
        assert all("LocalExecutor" == comp.name for comp in components)
        assert all(comp.config.max_workers == i+1 for i, comp in enumerate(components))


# Integration test to ensure the simplified system works with existing framework
class TestFrameworkIntegration:
    """Test that simplified factory integrates properly with existing framework."""
    
    def test_integration_with_existing_from_config(self):
        """Test that our simplified factory works with existing from_config implementations."""
        # This tests that our approach is compatible with the framework's from_config pattern
        executor_config = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=2
        )
        
        # Both approaches should work and produce equivalent results
        direct_executor = LocalExecutor.from_config(executor_config)
        factory_executor = import_and_create_from_config(
            "nanobrain.core.executor.LocalExecutor", 
            executor_config
        )
        
        assert direct_executor.name == factory_executor.name
        assert type(direct_executor) == type(factory_executor)
        assert direct_executor.config.max_workers == factory_executor.config.max_workers


class TestConfigurationHandling:
    """Test various configuration formats and conversions."""
    
    def test_dict_to_config_conversion(self):
        """Test that dict configurations are properly converted to ExecutorConfig."""
        # This demonstrates how the factory handles dictionary input
        config_dict = {
            "executor_type": "local",
            "max_workers": 6,
            "timeout": 45.0
        }
        
        executor = import_and_create_from_config(
            "nanobrain.core.executor.LocalExecutor",
            config_dict
        )
        
        assert executor.name == "LocalExecutor"
        assert executor.config.max_workers == 6
        assert executor.config.timeout == 45.0
    
    def test_enum_handling(self):
        """Test that ExecutorType enums are handled correctly."""
        # Test with enum value
        config_enum = ExecutorConfig(
            executor_type=ExecutorType.LOCAL,
            max_workers=3
        )
        
        # Test with string value
        config_string = {
            "executor_type": "local",
            "max_workers": 3
        }
        
        executor1 = LocalExecutor.from_config(config_enum)
        executor2 = import_and_create_from_config(
            "nanobrain.core.executor.LocalExecutor",
            config_string
        )
        
        assert executor1.config.executor_type == executor2.config.executor_type
        assert executor1.config.max_workers == executor2.config.max_workers 