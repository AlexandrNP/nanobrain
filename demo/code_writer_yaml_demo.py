#!/usr/bin/env python3
"""
CodeWriter YAML Configuration Demo

This demo shows how CodeWriterAgent properly loads prompts and parameters
from YAML configuration files, demonstrating the complete YAML-to-agent pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import NanoBrain components
from nanobrain.config.component_factory import get_factory
import sys
from pathlib import Path
current_file = Path(__file__) if '__file__' in globals() else Path.cwd() / 'demo' / 'code_writer_yaml_demo.py'
sys.path.insert(0, str(current_file.parent.parent / "library"))
from agents.specialized import CodeWriterAgent


async def demo_yaml_config_loading():
    """Demonstrate CodeWriterAgent loading configuration from YAML."""
    print("🧠 CodeWriter YAML Configuration Demo")
    print("=" * 60)
    
    factory = get_factory()
    
    # Demo 1: Load CodeWriterAgent from step_coder.yml template
    print("\n📋 DEMO 1: Loading CodeWriterAgent from step_coder.yml")
    print("-" * 50)
    
    try:
        # Load agent from YAML template
        agent = factory.create_from_yaml_file(
            "nanobrain/src/agents/config/step_coder.yml",
            component_name="yaml_code_writer"
        )
        
        print(f"✅ Successfully created: {type(agent).__name__}")
        print(f"   Name: {agent.config.name}")
        print(f"   Description: {agent.config.description}")
        print(f"   Model: {agent.config.model}")
        print(f"   Temperature: {agent.config.temperature}")
        print(f"   Max Tokens: {agent.config.max_tokens}")
        print(f"   System Prompt Preview: {agent.config.system_prompt[:100]}...")
        
        # Verify it's actually a CodeWriterAgent
        assert isinstance(agent, CodeWriterAgent), f"Expected CodeWriterAgent, got {type(agent)}"
        print("✅ Confirmed: Agent is CodeWriterAgent instance")
        
        # Verify YAML configuration was loaded
        assert agent.config.model == "gpt-4-turbo", "Model should be loaded from YAML"
        assert agent.config.temperature == 0.2, "Temperature should be loaded from YAML"
        assert agent.config.max_tokens == 4000, "Max tokens should be loaded from YAML"
        assert "specialized code generation agent" in agent.config.system_prompt, "System prompt should be loaded from YAML"
        print("✅ Confirmed: All YAML configuration loaded correctly")
        
    except Exception as e:
        print(f"❌ Error loading from YAML: {e}")
        return
    
    # Demo 2: Create custom YAML configuration
    print("\n📋 DEMO 2: Creating CodeWriterAgent with custom YAML config")
    print("-" * 50)
    
    custom_config = {
        "class": "CodeWriterAgent",
        "config": {
            "name": "CustomCodeWriter",
            "description": "Custom code writer with specialized prompt",
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 2000,
            "system_prompt": """You are a specialized Python code generator.
            
Focus on:
- Writing clean, Pythonic code
- Adding comprehensive docstrings
- Including type hints
- Following PEP 8 standards
- Creating efficient algorithms

Always explain your code choices."""
        }
    }
    
    try:
        # Create agent from custom config
        custom_agent = factory.create_component(
            "agent", 
            custom_config, 
            "custom_code_writer"
        )
        
        print(f"✅ Successfully created: {type(custom_agent).__name__}")
        print(f"   Name: {custom_agent.config.name}")
        print(f"   Model: {custom_agent.config.model}")
        print(f"   Temperature: {custom_agent.config.temperature}")
        print(f"   Max Tokens: {custom_agent.config.max_tokens}")
        print(f"   Custom System Prompt: {custom_agent.config.system_prompt[:80]}...")
        
        # Verify custom configuration
        assert isinstance(custom_agent, CodeWriterAgent), "Should be CodeWriterAgent"
        assert custom_agent.config.model == "gpt-3.5-turbo", "Should use custom model"
        assert custom_agent.config.temperature == 0.1, "Should use custom temperature"
        assert "specialized Python code generator" in custom_agent.config.system_prompt, "Should use custom prompt"
        print("✅ Confirmed: Custom configuration applied correctly")
        
    except Exception as e:
        print(f"❌ Error creating custom agent: {e}")
        return
    
    # Demo 3: Compare with default CodeWriterAgent
    print("\n📋 DEMO 3: Comparing with default CodeWriterAgent")
    print("-" * 50)
    
    try:
        # Create default CodeWriterAgent (no YAML config)
        default_agent = CodeWriterAgent()
        
        print(f"Default Agent:")
        print(f"   Name: {default_agent.config.name}")
        print(f"   Model: {default_agent.config.model}")
        print(f"   Temperature: {default_agent.config.temperature}")
        print(f"   System Prompt Preview: {default_agent.config.system_prompt[:80]}...")
        
        print(f"\nYAML-Configured Agent:")
        print(f"   Name: {agent.config.name}")
        print(f"   Model: {agent.config.model}")
        print(f"   Temperature: {agent.config.temperature}")
        print(f"   System Prompt Preview: {agent.config.system_prompt[:80]}...")
        
        # Show the difference
        print(f"\n🔍 Configuration Differences:")
        print(f"   Model: {default_agent.config.model} → {agent.config.model}")
        print(f"   Temperature: {default_agent.config.temperature} → {agent.config.temperature}")
        print(f"   Max Tokens: {default_agent.config.max_tokens} → {agent.config.max_tokens}")
        print(f"   Prompt Length: {len(default_agent.config.system_prompt)} → {len(agent.config.system_prompt)} chars")
        
    except Exception as e:
        print(f"❌ Error creating default agent: {e}")
        return
    
    # Demo 4: Test agent functionality (if API key available)
    print("\n📋 DEMO 4: Testing agent functionality")
    print("-" * 50)
    
    try:
        # Simple test request
        test_request = "Write a simple Python function that calculates the factorial of a number"
        print(f"Test Request: {test_request}")
        
        # Note: This will likely fail without OpenAI API key, but shows the interface
        response = await agent.process(test_request)
        print(f"✅ Agent Response: {response[:200]}...")
        
    except Exception as e:
        print(f"⚠️  Agent test skipped (likely missing API key): {e}")
    
    # Demo 5: Registry and cleanup
    print("\n📋 DEMO 5: Component registry and cleanup")
    print("-" * 50)
    
    # Show registered components
    components = factory.list_components()
    print(f"📋 Registered components: {len(components)}")
    for name in components:
        component = factory.get_component(name)
        print(f"   - {name}: {type(component).__name__}")
    
    # Cleanup
    print("\n🧹 Cleaning up components...")
    factory.shutdown_components()
    print("✅ All components shutdown successfully")
    
    print("\n" + "=" * 60)
    print("✅ CodeWriter YAML Configuration Demo completed successfully!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print("   ✅ CodeWriterAgent loads configuration from YAML files")
    print("   ✅ System prompts, model parameters, and settings are configurable")
    print("   ✅ Custom configurations can be created programmatically")
    print("   ✅ Factory system properly handles CodeWriterAgent creation")
    print("   ✅ Component registry and lifecycle management working")


if __name__ == "__main__":
    asyncio.run(demo_yaml_config_loading()) 