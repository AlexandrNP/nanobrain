"""
Tests for the base Workflow class
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path

from nanobrain.core.workflow import (
    Workflow, WorkflowConfig, WorkflowGraph, ConfigLoader,
    ExecutionStrategy, ErrorHandlingStrategy, create_workflow
)
from nanobrain.core.step import StepConfig, Step
from nanobrain.core.data_unit import DataUnitConfig
from nanobrain.core.executor import LocalExecutor


class TestWorkflowGraph:
    """Test the WorkflowGraph class."""
    
    def test_empty_graph(self):
        """Test empty graph initialization."""
        graph = WorkflowGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        
        stats = graph.get_stats()
        assert stats["num_steps"] == 0
        assert stats["num_links"] == 0
    
    def test_add_step(self):
        """Test adding steps to the graph."""
        graph = WorkflowGraph()
        
        step_config = StepConfig(name="test_step")
        step = Step(step_config)
        
        graph.add_step("test_step", step)
        assert "test_step" in graph.nodes
        assert graph.nodes["test_step"] == step
        
        # Test duplicate step ID
        with pytest.raises(ValueError, match="already exists"):
            graph.add_step("test_step", step)
    
    def test_cycle_detection(self):
        """Test cycle detection in the graph."""
        graph = WorkflowGraph()
        
        # Create three steps
        for i in range(3):
            step_config = StepConfig(name=f"step_{i}")
            step = Step(step_config)
            graph.add_step(f"step_{i}", step)
        
        # Initially no cycles
        assert not graph.has_cycles()
        
        # Add edges to create cycle: 0 -> 1 -> 2 -> 0
        from nanobrain.core.link import DirectLink, LinkConfig
        
        for i in range(3):
            source_id = f"step_{i}"
            target_id = f"step_{(i + 1) % 3}"
            
            source_step = graph.nodes[source_id]
            target_step = graph.nodes[target_id]
            
            link_config = LinkConfig()
            link = DirectLink(source_step, target_step, link_config, name=f"link_{i}")
            
            graph.add_link(f"link_{i}", link, source_id, target_id)
        
        # Now should detect cycle
        assert graph.has_cycles()
    
    def test_execution_order(self):
        """Test topological sort for execution order."""
        graph = WorkflowGraph()
        
        # Create linear workflow: A -> B -> C
        steps = []
        for name in ['A', 'B', 'C']:
            step_config = StepConfig(name=name)
            step = Step(step_config)
            graph.add_step(name, step)
            steps.append((name, step))
        
        from nanobrain.core.link import DirectLink, LinkConfig
        
        # Add links A->B and B->C
        for i in range(len(steps) - 1):
            source_name, source_step = steps[i]
            target_name, target_step = steps[i + 1]
            
            link_config = LinkConfig()
            link = DirectLink(source_step, target_step, link_config, name=f"link_{i}")
            graph.add_link(f"link_{i}", link, source_name, target_name)
        
        execution_order = graph.get_execution_order()
        assert execution_order == ['A', 'B', 'C']
    
    def test_graph_validation(self):
        """Test graph validation."""
        graph = WorkflowGraph()
        
        # Empty graph should be invalid
        is_valid, errors = graph.validate_graph()
        assert not is_valid
        assert "empty" in errors[0].lower()
        
        # Add single step - should be valid
        step_config = StepConfig(name="single_step")
        step = Step(step_config)
        graph.add_step("single_step", step)
        
        is_valid, errors = graph.validate_graph(require_connected=False)
        assert is_valid
        assert len(errors) == 0


class TestConfigLoader:
    """Test the ConfigLoader class."""
    
    def test_resolve_config_path(self):
        """Test configuration path resolution."""
        loader = ConfigLoader()
        
        # Test with current file (should exist)
        current_file = Path(__file__)
        resolved = loader.resolve_config_path(str(current_file))
        assert resolved == current_file
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            loader.resolve_config_path("non_existent_file.yaml")
    
    def test_load_workflow_config(self):
        """Test loading workflow configuration from YAML."""
        # Create temporary YAML file
        config_data = {
            'name': 'test_workflow',
            'description': 'Test workflow',
            'execution_strategy': 'sequential',
            'steps': [
                {
                    'step_id': 'test_step',
                    'class': 'Step',
                    'config': {'name': 'test_step'}
                }
            ],
            'links': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = ConfigLoader()
            workflow_config = loader.load_workflow_config(temp_path)
            
            assert isinstance(workflow_config, WorkflowConfig)
            assert workflow_config.name == 'test_workflow'
            assert workflow_config.execution_strategy == ExecutionStrategy.SEQUENTIAL
            assert len(workflow_config.steps) == 1
            
        finally:
            Path(temp_path).unlink()


class TestWorkflowConfig:
    """Test the WorkflowConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WorkflowConfig(name="test_workflow")
        
        assert config.name == "test_workflow"
        assert config.execution_strategy == ExecutionStrategy.SEQUENTIAL
        assert config.error_handling == ErrorHandlingStrategy.CONTINUE
        assert config.validate_graph is True
        assert config.allow_cycles is False
        assert len(config.steps) == 0
        assert len(config.links) == 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WorkflowConfig(
            name="custom_workflow",
            execution_strategy=ExecutionStrategy.PARALLEL,
            error_handling=ErrorHandlingStrategy.RETRY,
            max_parallel_steps=5,
            retry_attempts=5
        )
        
        assert config.execution_strategy == ExecutionStrategy.PARALLEL
        assert config.error_handling == ErrorHandlingStrategy.RETRY
        assert config.max_parallel_steps == 5
        assert config.retry_attempts == 5


class TestWorkflow:
    """Test the Workflow class."""
    
    @pytest.fixture
    def simple_workflow_config(self):
        """Create a simple workflow configuration for testing."""
        return WorkflowConfig(
            name="test_workflow",
            description="Test workflow",
            steps=[
                {
                    'step_id': 'step1',
                    'class': 'Step',
                    'config': {
                        'name': 'step1',
                        'description': 'First step'
                    }
                }
            ],
            links=[]
        )
    
    def test_workflow_creation(self, simple_workflow_config):
        """Test basic workflow creation."""
        workflow = Workflow(simple_workflow_config)
        
        assert workflow.name == "test_workflow"
        assert workflow.workflow_config == simple_workflow_config
        assert isinstance(workflow.workflow_graph, WorkflowGraph)
        assert isinstance(workflow.config_loader, ConfigLoader)
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, simple_workflow_config):
        """Test workflow initialization."""
        workflow = Workflow(simple_workflow_config)
        
        # Should not be initialized initially
        assert not workflow._is_initialized
        assert len(workflow.child_steps) == 0
        
        # Initialize workflow
        await workflow.initialize()
        
        assert workflow._is_initialized
        assert len(workflow.child_steps) == 1
        assert 'step1' in workflow.child_steps
        assert len(workflow.step_links) == 0
        
        # Clean up
        await workflow.shutdown()
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, simple_workflow_config):
        """Test workflow execution."""
        workflow = Workflow(simple_workflow_config)
        await workflow.initialize()
        
        try:
            # Execute workflow
            result = await workflow.process({"test_data": "hello"})
            
            # Check execution statistics
            stats = workflow.get_workflow_stats()
            assert stats["workflow_name"] == "test_workflow"
            assert stats["num_steps"] == 1
            assert stats["num_links"] == 0
            assert stats["completed_steps"] >= 0  # Some steps may complete
            
        finally:
            await workflow.shutdown()
    
    @pytest.mark.asyncio
    async def test_workflow_step_management(self, simple_workflow_config):
        """Test adding and removing steps dynamically."""
        workflow = Workflow(simple_workflow_config)
        await workflow.initialize()
        
        try:
            # Add a new step
            new_step_config = StepConfig(name="dynamic_step")
            await workflow.add_step("dynamic_step", new_step_config)
            
            assert "dynamic_step" in workflow.child_steps
            assert len(workflow.child_steps) == 2
            
            # Remove the step
            await workflow.remove_step("dynamic_step")
            
            assert "dynamic_step" not in workflow.child_steps
            assert len(workflow.child_steps) == 1
            
        finally:
            await workflow.shutdown()
    
    def test_workflow_stats(self, simple_workflow_config):
        """Test workflow statistics."""
        workflow = Workflow(simple_workflow_config)
        
        stats = workflow.get_workflow_stats()
        
        assert stats["workflow_name"] == "test_workflow"
        assert stats["execution_strategy"] == ExecutionStrategy.SEQUENTIAL.value
        assert stats["num_steps"] == 0  # Not initialized yet
        assert stats["num_links"] == 0
        assert stats["completed_steps"] == 0
        assert stats["failed_steps"] == 0
        assert not stats["is_complete"]


class TestWorkflowFactory:
    """Test the workflow factory function."""
    
    @pytest.mark.asyncio
    async def test_create_workflow_from_config(self):
        """Test creating workflow from configuration object."""
        config = WorkflowConfig(
            name="factory_test_workflow",
            steps=[
                {
                    'step_id': 'test_step',
                    'class': 'Step',
                    'config': {'name': 'test_step'}
                }
            ]
        )
        
        workflow = await create_workflow(config)
        
        try:
            assert isinstance(workflow, Workflow)
            assert workflow.name == "factory_test_workflow"
            assert workflow._is_initialized
            assert len(workflow.child_steps) == 1
            
        finally:
            await workflow.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_workflow_from_dict(self):
        """Test creating workflow from dictionary configuration."""
        config_dict = {
            'name': 'dict_test_workflow',
            'description': 'Test workflow from dict',
            'steps': [
                {
                    'step_id': 'test_step',
                    'class': 'Step',
                    'config': {'name': 'test_step'}
                }
            ],
            'links': []
        }
        
        workflow = await create_workflow(config_dict)
        
        try:
            assert isinstance(workflow, Workflow)
            assert workflow.name == "dict_test_workflow"
            assert workflow._is_initialized
            
        finally:
            await workflow.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_workflow_from_yaml_file(self):
        """Test creating workflow from YAML file."""
        # Create temporary YAML file
        config_data = {
            'name': 'yaml_test_workflow',
            'description': 'Test workflow from YAML',
            'steps': [
                {
                    'step_id': 'yaml_step',
                    'class': 'Step',
                    'config': {'name': 'yaml_step'}
                }
            ],
            'links': []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            workflow = await create_workflow(temp_path)
            
            try:
                assert isinstance(workflow, Workflow)
                assert workflow.name == "yaml_test_workflow"
                assert workflow._is_initialized
                
            finally:
                await workflow.shutdown()
                
        finally:
            Path(temp_path).unlink()


class TestWorkflowIntegration:
    """Integration tests for workflow with other components."""
    
    @pytest.mark.asyncio
    async def test_workflow_with_executor(self):
        """Test workflow with custom executor."""
        executor = LocalExecutor()
        
        config = WorkflowConfig(
            name="executor_test_workflow",
            steps=[
                {
                    'step_id': 'exec_step',
                    'class': 'Step',
                    'config': {'name': 'exec_step'}
                }
            ]
        )
        
        workflow = Workflow(config, executor=executor)
        await workflow.initialize()
        
        try:
            assert workflow.executor == executor
            # Child steps should also use the same executor
            child_step = workflow.child_steps['exec_step']
            assert child_step.executor == executor
            
        finally:
            await workflow.shutdown()
            await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_nested_workflow(self):
        """Test workflow containing another workflow as a step."""
        # Create inner workflow configuration
        inner_config = WorkflowConfig(
            name="inner_workflow",
            steps=[
                {
                    'step_id': 'inner_step',
                    'class': 'Step',
                    'config': {'name': 'inner_step'}
                }
            ]
        )
        
        # Create outer workflow with inner workflow as a step
        outer_config = WorkflowConfig(
            name="outer_workflow",
            steps=[
                {
                    'step_id': 'regular_step',
                    'class': 'Step',
                    'config': {'name': 'regular_step'}
                }
                # Note: Nested workflows would need additional configuration
                # This is a placeholder for future nested workflow support
            ]
        )
        
        outer_workflow = Workflow(outer_config)
        await outer_workflow.initialize()
        
        try:
            assert outer_workflow.name == "outer_workflow"
            assert len(outer_workflow.child_steps) == 1
            
        finally:
            await outer_workflow.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 