#!/usr/bin/env python3
"""
Test script to verify agent setup works correctly with mandatory card validation.
"""

import asyncio
import sys
import os

# Add both src and parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules
sys.path.insert(0, parent_dir)                          # for demo modules

from nanobrain.core.config.component_factory import ComponentFactory
from nanobrain.core.agent import SimpleAgent

async def test_setup():
    print('🧪 Testing agent setup with mandatory card validation...')
    factory = ComponentFactory()
    
    try:
        # Create agent with proper agent card from default configuration
        agent = factory.create_from_yaml_file(
            'nanobrain/library/config/defaults/agent.yml',
            'nanobrain.core.agent.SimpleAgent'
        )
        print('✅ Agent created successfully with mandatory card!')
        
        # Test agent initialization
        await agent.initialize()
        print('✅ Agent initialized successfully!')
        
        # Test basic processing
        result = await agent.process("Hello, this is a test message")
        print(f'✅ Agent processed message: {result[:100] if result else "No response"}...')
        
        # Verify agent has proper card data
        assert hasattr(agent, '_a2a_card_data')
        assert agent._a2a_card_data is not None
        assert 'version' in agent._a2a_card_data
        print('✅ Agent card validation successful!')
        
        # Cleanup
        await agent.shutdown()
        print('✅ Agent shutdown completed successfully!')
        return True
    except Exception as e:
        print(f'❌ Setup failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_setup())
    sys.exit(0 if result else 1) 