"""
Tests for Component Factory System

Tests the YAML-based component creation system.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Import the components we need to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.component_factory import ComponentFactory, ComponentType, get_factory
from core.agent import Agent, SimpleAgent
from agents.code_writer import CodeWriterAgent
from core.step import Step, SimpleStep
from core.data_unit import DataUnitMemory
from core.trigger import DataUpdatedTrigger
from core.link import DirectLink
from core.executor import LocalExecutor


class TestComponentFactory:
    """Test the ComponentFactory class."""
    
    @pytest.fixture
    def factory(self):
        """Create a component factory for testing."""
        return ComponentFactory()
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_agent_config(self):
        """Sample agent configuration."""
        return {
            "name": "TestAgent",
            "description": "Test agent for unit tests",
            "class": "SimpleAgent",
            "config": {
                "name": "TestAgent",
                "description": "Test agent",
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "system_prompt": "You are a test agent.",
                "auto_initialize": False  # Don't auto-initialize to avoid OpenAI calls
            }
        }
    
    @pytest.fixture
    def sample_step_config(self):
        """Sample step configuration."""
        return {
            "name": "TestStep",
            "description": "Test step for unit tests",
            "class": "SimpleStep",
            "config": {
                "name": "TestStep",
                "description": "Test step",
                "debug_mode": True
            },
            "input_configs": {
                "input": {
                    "data_type": "memory",  # Fixed: lowercase
                    "name": "input",
                    "description": "Test input"
                }
            },
            "output_config": {
                "data_type": "memory",  # Fixed: lowercase
                "name": "output",
                "description": "Test output"
            }
        }
    
    @pytest.fixture
    def sample_workflow_config(self):
        """Sample workflow configuration."""
        return {
            "name": "TestWorkflow",
            "description": "Test workflow",
            "version": "1.0.0",
            "executors": {
                "local": {
                    "executor_type": "local",  # Fixed: lowercase
                    "name": "local",
                    "description": "Local executor",
                    "max_workers": 2
                }
            },
            "data_units": {
                "test_data": {
                    "data_type": "memory",  # Fixed: lowercase
                    "name": "test_data",
                    "description": "Test data unit"
                }
            },
            "triggers": {
                "test_trigger": {
                    "trigger_type": "data_updated",  # Fixed: lowercase
                    "name": "test_trigger",
                    "description": "Test trigger"
                }
            },
            "agents": {
                "test_agent": {
                    "class": "SimpleAgent",
                    "config": {
                        "name": "test_agent",
                        "description": "Test agent",
                        "model": "gpt-3.5-turbo",
                        "auto_initialize": False  # Don't auto-initialize
                    }
                }
            },
            "steps": {
                "test_step": {
                    "class": "SimpleStep",
                    "config": {
                        "name": "test_step",
                        "description": "Test step"
                    }
                }
            },
            "links": [
                {
                    "name": "test_link",
                    "link_type": "direct",  # Fixed: lowercase
                    "source": "test_data",
                    "target": "test_step"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_create_agent_from_config(self, factory):
        """Test creating an agent from configuration."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            config = {
                "type": "SimpleAgent",
                "config": {
                    "name": "TestAgent",
                    "description": "Test agent for unit tests",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.5,
                    "debug_mode": True
                }
            }
            
            agent = factory.create_component(
                ComponentType.AGENT, 
                config, 
                "test_agent"
            )
            
            assert isinstance(agent, SimpleAgent)
            assert agent.name == "TestAgent"
            assert agent.description == "Test agent for unit tests"
            assert agent.config.model == "gpt-3.5-turbo"
            assert agent.config.temperature == 0.5
            
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_step_from_config(self, factory):
        """Test creating a step from configuration."""
        config = {
            "type": "SimpleStep",
            "config": {
                "name": "TestStep",
                "description": "Test step for unit tests",
                "debug_mode": True
            },
            "input_configs": {
                "input": {
                    "data_type": "memory",
                    "persistent": False,
                    "cache_size": 1000
                }
            },
            "output_config": {
                "data_type": "memory",
                "persistent": False,
                "cache_size": 1000
            }
        }
        
        step = factory.create_component(
            ComponentType.STEP, 
            config, 
            "test_step"
        )
        
        assert isinstance(step, SimpleStep)
        assert step.name == "TestStep"
        assert step.description == "Test step for unit tests"
        assert len(step.input_data_units) == 1
        assert step.output_data_unit is not None
        
        await step.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_data_unit_from_config(self, factory):
        """Test creating a data unit from configuration."""
        config = {
            "data_type": "memory",
            "persistent": False,
            "cache_size": 1000
        }
        
        data_unit = factory.create_component(
            ComponentType.DATA_UNIT, 
            config, 
            "test_data_unit"
        )
        
        assert isinstance(data_unit, DataUnitMemory)
        assert data_unit.name == "test_data_unit"
        
    @pytest.mark.asyncio
    async def test_create_trigger_from_config(self, factory):
        """Test creating a trigger from configuration."""
        config = {
            "trigger_type": "data_updated",
            "conditions": {
                "data_units": ["input1", "input2"],
                "require_all": True
            }
        }
        
        trigger = factory.create_component(
            ComponentType.TRIGGER, 
            config, 
            "test_trigger"
        )
        
        assert isinstance(trigger, DataUpdatedTrigger)
        assert trigger.config.trigger_type == "data_updated"
    
    @pytest.mark.asyncio
    async def test_create_executor_from_config(self, factory):
        """Test creating an executor from configuration."""
        config = {
            "executor_type": "local",
            "max_workers": 2,
            "timeout": 30.0
        }
        
        executor = factory.create_component(
            ComponentType.EXECUTOR, 
            config, 
            "test_executor"
        )
        
        assert isinstance(executor, LocalExecutor)
        assert executor.config.executor_type == "local"
        assert executor.config.max_workers == 2
        assert executor.config.timeout == 30.0
        
        await executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_from_yaml_file(self, factory, temp_config_dir):
        """Test creating components from YAML files."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Create a test YAML file
            config_data = {
                "type": "SimpleAgent",
                "description": "Test agent from YAML",
                "config": {
                    "name": "YAMLAgent",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.3,
                    "debug_mode": True
                }
            }
            
            config_file = temp_config_dir / "test_agent.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            agent = factory.create_from_yaml_file(config_file)
            
            assert isinstance(agent, SimpleAgent)
            assert agent.name == "YAMLAgent"
            assert agent.config.temperature == 0.3
            
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_create_workflow_from_config(self, factory, temp_config_dir):
        """Test creating a complete workflow from configuration."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            workflow_config = {
                "name": "TestWorkflow",
                "description": "Test workflow for unit tests",
                "executors": {
                    "local": {
                        "executor_type": "local",
                        "name": "local",
                        "description": "Local executor",
                        "max_workers": 2
                    }
                },
                "data_units": {
                    "test_data": {
                        "data_type": "memory",
                        "name": "test_data",
                        "description": "Test data unit"
                    }
                },
                "triggers": {
                    "test_trigger": {
                        "trigger_type": "data_updated",
                        "name": "test_trigger",
                        "description": "Test trigger"
                    }
                },
                "agents": {
                    "test_agent": {
                        "class": "SimpleAgent",
                        "config": {
                            "name": "test_agent",
                            "description": "Test agent",
                            "model": "gpt-3.5-turbo",
                            "auto_initialize": False
                        }
                    }
                },
                "steps": {
                    "test_step": {
                        "class": "SimpleStep",
                        "config": {
                            "name": "test_step",
                            "description": "Test step"
                        }
                    }
                },
                "links": [
                    {
                        "name": "test_link",
                        "link_type": "direct",
                        "source": "test_data",
                        "target": "test_step"
                    }
                ]
            }
            
            workflow_file = temp_config_dir / "test_workflow.yml"
            with open(workflow_file, 'w') as f:
                yaml.dump(workflow_config, f)
            
            workflow = factory.create_workflow_from_yaml(workflow_file)
            
            assert "local" in workflow
            assert "test_data" in workflow
            assert "test_trigger" in workflow
            assert "test_agent" in workflow
            assert "test_step" in workflow
            
            # Cleanup
            for component in workflow.values():
                if hasattr(component, 'shutdown'):
                    await component.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_registry(self, factory):
        """Test component registry functionality."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            config = {
                "type": "SimpleAgent",
                "config": {
                    "name": "RegistryAgent",
                    "debug_mode": True
                }
            }
            
            # Create and register component
            agent = factory.create_component(
                ComponentType.AGENT, 
                config, 
                "registry_agent"
            )
            
            # Check registry
            assert factory.get_component("registry_agent") is agent
            
            # List components
            components = factory.list_components()
            assert "registry_agent" in components
            
            await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_component_caching(self, factory, temp_config_dir):
        """Test configuration caching."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Create a test YAML file
            config_data = {
                "type": "SimpleAgent",
                "config": {
                    "name": "CachedAgent",
                    "debug_mode": True
                }
            }
            
            config_file = temp_config_dir / "cached_agent.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load twice to test caching
            agent1 = factory.create_from_yaml_file(config_file)
            agent2 = factory.create_from_yaml_file(config_file)
            
            # Both should be created successfully (caching is for config, not instances)
            assert isinstance(agent1, SimpleAgent)
            assert isinstance(agent2, SimpleAgent)
            assert agent1 is not agent2  # Different instances
            
            await agent1.shutdown()
            await agent2.shutdown()
    
    def test_custom_class_registration(self, factory):
        """Test registering custom classes."""
        class CustomAgent(SimpleAgent):
            pass
        
        class CustomStep(SimpleStep):
            pass
        
        factory.register_class("CustomAgent", CustomAgent)
        factory.register_class("CustomStep", CustomStep)
        
        # Verify registration
        assert "CustomAgent" in factory.custom_classes
        assert "CustomStep" in factory.custom_classes
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_file(self, factory):
        """Test error handling for missing configuration files."""
        with pytest.raises(FileNotFoundError):
            factory.create_from_yaml_file("nonexistent_file.yml")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_class(self, factory):
        """Test error handling for invalid class names."""
        config = {
            "type": "NonexistentAgent",
            "config": {
                "name": "TestAgent",
                "debug_mode": True
            }
        }
        
        with pytest.raises((ValueError, KeyError)):
            factory.create_component(ComponentType.AGENT, config, "test")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_config(self, factory):
        """Test error handling for invalid configurations."""
        config = {
            "type": "SimpleAgent",
            "config": {
                # Missing required 'name' field
                "model": "invalid-model"
            }
        }
        
        with pytest.raises(Exception):  # Could be ValidationError or other
            await factory.create_component(ComponentType.AGENT, config, "test")
    
    @pytest.mark.asyncio
    async def test_shutdown_all(self, factory):
        """Test shutting down all components."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Create multiple components
            agent_config = {
                "type": "SimpleAgent",
                "config": {
                    "name": "ShutdownAgent",
                    "debug_mode": True
                }
            }
            
            executor_config = {
                "executor_type": "local",
                "max_workers": 1
            }
            
            factory.create_component(ComponentType.AGENT, agent_config, "shutdown_agent")
            factory.create_component(ComponentType.EXECUTOR, executor_config, "shutdown_executor")
            
            # Shutdown all
            factory.shutdown_components()
            
            # Registry should be empty
            components = factory.list_components()
            assert len(components) == 0

    def test_code_writer_yaml_config_loading(self, factory):
        """Test that CodeWriterAgent properly loads configuration from YAML."""
        # Create a test YAML configuration for CodeWriterAgent
        test_config = {
            "class": "CodeWriterAgent",
            "config": {
                "name": "TestCodeWriter",
                "description": "Test code writer agent",
                "model": "gpt-4-turbo",
                "temperature": 0.1,
                "max_tokens": 2000,
                "system_prompt": "Custom system prompt for testing YAML configuration loading."
            }
        }
        
        # Create agent from config
        agent = factory.create_component(ComponentType.AGENT, test_config, "test_code_writer")
        
        # Verify it's the correct type
        assert isinstance(agent, CodeWriterAgent)
        
        # Verify configuration was loaded from YAML
        assert agent.config.name == "TestCodeWriter"
        assert agent.config.description == "Test code writer agent"
        assert agent.config.model == "gpt-4-turbo"
        assert agent.config.temperature == 0.1
        assert agent.config.max_tokens == 2000
        assert agent.config.system_prompt == "Custom system prompt for testing YAML configuration loading."
        
        # Verify agent is registered
        assert "test_code_writer" in factory.component_registry
        assert factory.get_component("test_code_writer") is agent

    def test_code_writer_default_prompt_fallback(self, factory):
        """Test that CodeWriterAgent requires prompts from YAML configuration."""
        # Create config without system_prompt or prompt_templates
        test_config = {
            "class": "CodeWriterAgent",
            "config": {
                "name": "DefaultPromptAgent",
                "description": "Agent with no prompts - should use empty defaults",
                "model": "gpt-4"
            }
        }
        
        # Create agent from config
        agent = factory.create_component(ComponentType.AGENT, test_config)
        
        # Verify it's a CodeWriterAgent but has no hardcoded prompts
        assert isinstance(agent, CodeWriterAgent)
        
        # Verify system prompt is empty (no hardcoded defaults)
        assert agent.config.system_prompt == ""
        
        # Verify prompt templates are empty (no hardcoded defaults)
        assert agent.prompt_templates == {}
        
        # This demonstrates that all prompts must now come from YAML configuration
        print("✅ CodeWriterAgent correctly requires all prompts to be defined in YAML configuration")

    def test_step_coder_yaml_template(self, factory):
        """Test that step_coder.yml template properly creates CodeWriterAgent."""
        # Load the actual step_coder.yml template
        agent = factory.create_from_yaml_file(
            "nanobrain/src/agents/config/step_coder.yml",
            component_name="step_coder_test"
        )
        
        # Verify it's a CodeWriterAgent
        assert isinstance(agent, CodeWriterAgent)
        
        # Verify configuration from YAML
        assert agent.config.name == "StepCoder"
        assert agent.config.model == "gpt-4-turbo"
        assert agent.config.temperature == 0.2
        assert agent.config.max_tokens == 4000
        
        # Verify the system prompt was loaded from YAML
        assert "specialized code generation agent for the NanoBrain framework" in agent.config.system_prompt
        assert "Follow PEP 8 style guidelines for Python" in agent.config.system_prompt
        assert "Include comprehensive docstrings" in agent.config.system_prompt
        
        # Verify agent is registered
        assert "step_coder_test" in factory.component_registry

    @pytest.mark.asyncio
    async def test_code_writer_prompt_templates_from_yaml(self):
        """Test that CodeWriterAgent loads all prompt templates from YAML configuration."""
        factory = get_factory()
        
        # Load CodeWriterAgent from step_coder.yml
        agent = factory.create_from_yaml_file(
            "nanobrain/src/agents/config/step_coder.yml",
            component_name="test_prompt_templates"
        )
        
        # Verify it's a CodeWriterAgent
        assert isinstance(agent, CodeWriterAgent)
        
        # Verify prompt templates are loaded
        assert hasattr(agent, 'prompt_templates')
        assert isinstance(agent.prompt_templates, dict)
        
        # Verify all expected prompt templates are present
        expected_templates = [
            'enhanced_input',
            'python_function', 
            'python_class',
            'nanobrain_step',
            'write_code_to_file'
        ]
        
        for template_name in expected_templates:
            assert template_name in agent.prompt_templates, f"Missing template: {template_name}"
            assert isinstance(agent.prompt_templates[template_name], str)
            assert len(agent.prompt_templates[template_name].strip()) > 0, f"Empty template: {template_name}"
        
        # Verify templates contain expected placeholders
        assert '{input_text}' in agent.prompt_templates['enhanced_input']
        assert '{available_tools}' in agent.prompt_templates['enhanced_input']
        
        assert '{function_name}' in agent.prompt_templates['python_function']
        assert '{description}' in agent.prompt_templates['python_function']
        assert '{parameters}' in agent.prompt_templates['python_function']
        assert '{return_type}' in agent.prompt_templates['python_function']
        
        assert '{class_name}' in agent.prompt_templates['python_class']
        assert '{description}' in agent.prompt_templates['python_class']
        assert '{methods}' in agent.prompt_templates['python_class']
        
        assert '{step_name}' in agent.prompt_templates['nanobrain_step']
        assert '{description}' in agent.prompt_templates['nanobrain_step']
        assert '{input_types}' in agent.prompt_templates['nanobrain_step']
        assert '{output_types}' in agent.prompt_templates['nanobrain_step']
        
        assert '{tool_name}' in agent.prompt_templates['write_code_to_file']
        assert '{file_path}' in agent.prompt_templates['write_code_to_file']
        assert '{code}' in agent.prompt_templates['write_code_to_file']
        
        # Verify templates are loaded from YAML config, not hardcoded defaults
        # The YAML templates should contain specific text that differs from defaults
        assert 'Code Generation Request:' in agent.prompt_templates['enhanced_input']
        assert 'Generate a Python function with the following specifications:' in agent.prompt_templates['python_function']
        assert 'Generate a Python class with the following specifications:' in agent.prompt_templates['python_class']
        assert 'Generate a NanoBrain Step class with the following specifications:' in agent.prompt_templates['nanobrain_step']
        assert 'Please use the' in agent.prompt_templates['write_code_to_file']
        
        print("✅ All prompt templates loaded correctly from YAML configuration")

    @pytest.mark.asyncio
    async def test_code_writer_methods_use_yaml_templates(self):
        """Test that CodeWriterAgent methods use templates from YAML instead of hardcoded prompts."""
        factory = get_factory()
        
        # Create agent with custom prompt templates to verify they're being used
        custom_config = {
            "class": "CodeWriterAgent",
            "config": {
                "name": "TestCodeWriter",
                "description": "Test agent for template verification",
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "system_prompt": "Test system prompt",
                "prompt_templates": {
                    "enhanced_input": "CUSTOM_ENHANCED: {input_text} | TOOLS: {available_tools}",
                    "python_function": "CUSTOM_FUNCTION: {function_name} - {description} ({parameters}) -> {return_type}",
                    "python_class": "CUSTOM_CLASS: {class_name}{base_classes} - {description} | Methods: {methods}",
                    "nanobrain_step": "CUSTOM_STEP: {step_name} - {description} | IN: {input_types} | OUT: {output_types}",
                    "write_code_to_file": "CUSTOM_WRITE: {tool_name} -> {file_path} | {description} | CODE: {code}"
                }
            }
        }
        
        agent = factory.create_component("agent", custom_config, "test_custom_templates")
        
        # Verify custom templates are loaded
        assert agent.prompt_templates["enhanced_input"].startswith("CUSTOM_ENHANCED:")
        assert agent.prompt_templates["python_function"].startswith("CUSTOM_FUNCTION:")
        assert agent.prompt_templates["python_class"].startswith("CUSTOM_CLASS:")
        assert agent.prompt_templates["nanobrain_step"].startswith("CUSTOM_STEP:")
        assert agent.prompt_templates["write_code_to_file"].startswith("CUSTOM_WRITE:")
        
        print("✅ CodeWriterAgent correctly uses custom prompt templates from YAML")


class TestYAMLConfigIntegration:
    """Test integration with YAML configuration system."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_workflow_config_integration(self, temp_config_dir):
        """Test integration with workflow configuration."""
        with patch('openai.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_openai.return_value = mock_client
            
            # Create a comprehensive workflow configuration
            workflow_config = {
                "name": "IntegrationWorkflow",
                "description": "Integration test workflow",
                "metadata": {
                    "version": "1.0",
                    "author": "test"
                },
                "executors": [
                    {
                        "name": "local_exec",
                        "executor_type": "local",
                        "max_workers": 2
                    }
                ],
                "data_units": [
                    {
                        "name": "memory_store",
                        "data_type": "memory",
                        "cache_size": 500
                    }
                ],
                "triggers": [
                    {
                        "name": "data_trigger",
                        "trigger_type": "data_updated",
                        "conditions": {
                            "data_units": ["memory_store"],
                            "require_all": True
                        }
                    }
                ],
                "agents": [
                    {
                        "name": "main_agent",
                        "type": "SimpleAgent",
                        "config": {
                            "name": "IntegrationAgent",
                            "model": "gpt-3.5-turbo",
                            "debug_mode": True
                        }
                    }
                ],
                "steps": [
                    {
                        "name": "main_step",
                        "type": "SimpleStep",
                        "config": {
                            "name": "IntegrationStep",
                            "debug_mode": True
                        },
                        "input_configs": {
                            "input": {
                                "data_type": "memory",
                                "cache_size": 100
                            }
                        }
                    }
                ]
            }
            
            config_file = temp_config_dir / "integration_workflow.yml"
            with open(config_file, 'w') as f:
                yaml.dump(workflow_config, f)
            
            factory = ComponentFactory()
            workflow = factory.create_workflow_from_yaml(config_file)
            
            # Verify all components were created
            assert "local_exec" in workflow
            assert "memory_store" in workflow
            assert "data_trigger" in workflow
            assert "main_agent" in workflow
            assert "main_step" in workflow
            
            # Verify component types
            assert isinstance(workflow["local_exec"], LocalExecutor)
            assert isinstance(workflow["memory_store"], DataUnitMemory)
            assert isinstance(workflow["data_trigger"], DataUpdatedTrigger)
            assert isinstance(workflow["main_agent"], SimpleAgent)
            assert isinstance(workflow["main_step"], SimpleStep)
            
            # Cleanup
            factory.shutdown_components()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 