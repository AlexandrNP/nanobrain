#!/usr/bin/env python3
"""
Test script to verify chat workflow setup works correctly.
"""

import asyncio
import sys
import os

# Add both src and parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules
sys.path.insert(0, parent_dir)                          # for demo modules

from demo.chat_workflow_demo import ChatWorkflow

async def test_setup():
    print('🧪 Testing chat workflow setup...')
    workflow = ChatWorkflow()
    try:
        await workflow.setup()
        print('✅ Setup completed successfully!')
        await workflow.shutdown()
        print('✅ Shutdown completed successfully!')
        return True
    except Exception as e:
        print(f'❌ Setup failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_setup())
    sys.exit(0 if result else 1) 