#!/usr/bin/env python3
"""
Tests for YAML Tool Loading in NanoBrain Framework

Tests the new YAML-based tool loading functionality for agents.
"""

import pytest
import asyncio
import tempfile
import os
import yaml
import warnings
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent import Agent, AgentConfig
from agents.code_writer import CodeWriterAgent
from agents.file_writer import FileWriterAgent


class TestYAMLToolLoading:
    """Test YAML-based tool loading functionality."""
    
    @pytest.fixture
    def sample_tools_config(self):
        """Create a sample tools configuration."""
        return {
            'tools': [
                {
                    'name': 'test_file_writer',
                    'tool_type': 'agent',
                    'class': 'agents.file_writer.FileWriterAgent',
                    'description': 'Test file writer tool',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'input': {
                                'type': 'string',
                                'description': 'Input text'
                            }
                        },
                        'required': ['input']
                    },
                    'config': {
                        'name': 'TestFileWriter',
                        'description': 'Test file writer agent',
                        'model': 'gpt-3.5-turbo',
                        'temperature': 0.3
                    }
                }
            ]
        }
    
    @pytest.fixture
    def temp_tools_config_file(self, sample_tools_config):
        """Create a temporary tools configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(sample_tools_config, f)
            temp_file_path = f.name
        
        yield temp_file_path
        
        # Cleanup
        os.unlink(temp_file_path)
    
    @pytest.mark.asyncio
    async def test_yaml_tool_loading_basic(self, temp_tools_config_file):
        """Test basic YAML tool loading functionality."""
        # Create agent config with tools_config_path
        config = AgentConfig(
            name="TestAgent",
            description="Test agent with YAML tools",
            tools_config_path=temp_tools_config_file,
            debug_mode=True
        )
        
        # Create a simple test agent
        class TestAgent(Agent):
            async def process(self, input_text: str, **kwargs) -> str:
                return f"Processed: {input_text}"
        
        agent = TestAgent(config)
        
        try:
            # Initialize agent (should load tools from YAML)
            await agent.initialize()
            
            # Check that tools were loaded
            available_tools = agent.available_tools
            assert 'test_file_writer' in available_tools
            
            # Verify tool registry
            tool = agent.tool_registry.get('test_file_writer')
            assert tool is not None
            assert tool.name == 'test_file_writer'
            assert tool.description == 'Test file writer tool'
            
        finally:
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_yaml_config_path_resolution(self):
        """Test configuration path resolution."""
        config = AgentConfig(
            name="TestAgent",
            description="Test agent",
            tools_config_path="nonexistent.yml"
        )
        
        class TestAgent(Agent):
            async def process(self, input_text: str, **kwargs) -> str:
                return f"Processed: {input_text}"
        
        agent = TestAgent(config)
        
        # Should raise FileNotFoundError for nonexistent file
        with pytest.raises(FileNotFoundError):
            await agent.initialize()
    
    @pytest.mark.asyncio
    async def test_yaml_tool_loading_with_code_writer(self, temp_tools_config_file):
        """Test YAML tool loading with CodeWriterAgent."""
        config = AgentConfig(
            name="CodeWriterWithYAMLTools",
            description="Code writer with YAML tools",
            tools_config_path=temp_tools_config_file,
            debug_mode=True
        )
        
        code_writer = CodeWriterAgent(config)
        
        try:
            # Mock the LLM client to avoid API calls
            with patch.object(code_writer, '_initialize_llm_client'):
                await code_writer.initialize()
                
                # Check that tools were loaded
                available_tools = code_writer.available_tools
                assert 'test_file_writer' in available_tools
                
        finally:
            await code_writer.shutdown()
    
    def test_programmatic_methods_removed(self):
        """Test that programmatic tool registration methods have been removed."""
        config = AgentConfig(name="TestAgent", description="Test agent")
        
        class TestAgent(Agent):
            async def process(self, input_text: str, **kwargs) -> str:
                return f"Processed: {input_text}"
        
        agent = TestAgent(config)
        
        # Verify that programmatic methods no longer exist
        assert not hasattr(agent, 'register_agent_tool')
        assert not hasattr(agent, 'register_function_tool')
        
        # Verify CodeWriterAgent no longer has register_file_writer_tool
        code_writer = CodeWriterAgent(config)
        assert not hasattr(code_writer, 'register_file_writer_tool')
    
    @pytest.mark.asyncio
    async def test_yaml_tool_config_standardization(self):
        """Test tool configuration standardization."""
        config = AgentConfig(
            name="TestAgent",
            description="Test agent"
        )
        
        class TestAgent(Agent):
            async def process(self, input_text: str, **kwargs) -> str:
                return f"Processed: {input_text}"
        
        agent = TestAgent(config)
        
        # Test standardization
        tool_config = {
            'name': 'test_tool',
            'class': 'some.module.SomeClass',
            'description': 'Test tool',
            'custom_field': 'custom_value'
        }
        
        standardized = agent._standardize_tool_config(tool_config)
        
        assert standardized['name'] == 'test_tool'
        assert standardized['class_path'] == 'some.module.SomeClass'
        assert standardized['description'] == 'Test tool'
        assert standardized['tool_type'] == 'agent'  # Default
        assert standardized['custom_field'] == 'custom_value'
    
    @pytest.mark.asyncio
    async def test_yaml_tool_loading_empty_config(self):
        """Test YAML tool loading with empty or missing tools config."""
        # Create empty config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({}, f)  # Empty config
            temp_file_path = f.name
        
        try:
            config = AgentConfig(
                name="TestAgent",
                description="Test agent",
                tools_config_path=temp_file_path
            )
            
            class TestAgent(Agent):
                async def process(self, input_text: str, **kwargs) -> str:
                    return f"Processed: {input_text}"
            
            agent = TestAgent(config)
            
            try:
                await agent.initialize()
                
                # Should have no tools loaded
                assert len(agent.available_tools) == 0
                
            finally:
                await agent.shutdown()
                
        finally:
            os.unlink(temp_file_path)
    
    def test_import_class_from_path(self):
        """Test class import functionality."""
        config = AgentConfig(name="TestAgent", description="Test agent")
        
        class TestAgent(Agent):
            async def process(self, input_text: str, **kwargs) -> str:
                return f"Processed: {input_text}"
        
        agent = TestAgent(config)
        
        # Test valid import
        imported_class = agent._import_class_from_path('agents.code_writer.CodeWriterAgent')
        assert imported_class == CodeWriterAgent
        
        # Test invalid module
        with pytest.raises(ImportError):
            agent._import_class_from_path('nonexistent.module.Class')
        
        # Test invalid class
        with pytest.raises(AttributeError):
            agent._import_class_from_path('agents.code_writer.NonexistentClass')
        
        # Test invalid format
        with pytest.raises(Exception):
            agent._import_class_from_path('invalid_format')


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 