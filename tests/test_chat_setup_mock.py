#!/usr/bin/env python3
"""
Test script to verify chat workflow setup works correctly with mocked OpenAI client.
"""

import asyncio
import sys
import os
from unittest.mock import patch, AsyncMock

# Add both src and parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules
sys.path.insert(0, parent_dir)                          # for demo modules

from demo.chat_workflow_demo import ChatWorkflow

async def test_setup_with_mock():
    print('🧪 Testing chat workflow setup with mocked OpenAI client...')
    
    with patch('openai.AsyncOpenAI') as mock_openai:
        # Mock the OpenAI client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        workflow = ChatWorkflow()
        try:
            await workflow.setup()
            print('✅ Setup completed successfully!')
            
            # Test that all components were created
            expected_components = [
                'user_input_du', 'agent_input_du', 'agent_output_du',
                'agent_step', 'user_trigger', 'agent_trigger', 'output_trigger',
                'user_to_agent_link', 'agent_input_to_step_link', 'step_to_output_link'
            ]
            
            for component in expected_components:
                if component not in workflow.components:
                    print(f'❌ Missing component: {component}')
                    return False
                else:
                    print(f'✅ Component created: {component}')
            
            # Verify CLI was created
            if workflow.cli is None:
                print('❌ CLI interface not created')
                return False
            else:
                print('✅ CLI interface created')
            
            await workflow.shutdown()
            print('✅ Shutdown completed successfully!')
            return True
            
        except Exception as e:
            print(f'❌ Setup failed: {e}')
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    result = asyncio.run(test_setup_with_mock())
    sys.exit(0 if result else 1) 