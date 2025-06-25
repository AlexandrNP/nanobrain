#!/usr/bin/env python3
"""
Test script to verify agent setup works correctly with mocked OpenAI client and mandatory cards.
"""

import asyncio
import sys
import os
from unittest.mock import patch, AsyncMock

# Add both src and parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules
sys.path.insert(0, parent_dir)                          # for demo modules

from nanobrain.core.config.component_factory import ComponentFactory
from nanobrain.core.agent import SimpleAgent

async def test_setup_with_mock():
    print('🧪 Testing agent setup with mocked OpenAI client and mandatory cards...')
    
    with patch('openai.AsyncOpenAI') as mock_openai:
        # Mock the OpenAI client and its response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock a successful response
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        mock_response.choices[0].message.content = "Hello! This is a mocked response from the agent."
        mock_response.usage.total_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        
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
            
            # Test basic processing with mocked LLM
            result = await agent.process("Hello, this is a test message")
            print(f'✅ Agent processed message with mock: {result[:100] if result else "No response"}...')
            
            # Verify agent has proper card data
            assert hasattr(agent, '_a2a_card_data')
            assert agent._a2a_card_data is not None
            assert 'version' in agent._a2a_card_data
            print('✅ Agent card validation successful!')
            
            # Test that LLM client was properly mocked
            if mock_client.chat.completions.create.called:
                print('✅ Mocked OpenAI client was called as expected!')
            else:
                print('⚠️ Mocked OpenAI client was not called (agent may have used fallback)')
            
            await agent.shutdown()
            print('✅ Agent shutdown completed successfully!')
            return True
            
        except Exception as e:
            print(f'❌ Setup failed: {e}')
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    result = asyncio.run(test_setup_with_mock())
    sys.exit(0 if result else 1) 