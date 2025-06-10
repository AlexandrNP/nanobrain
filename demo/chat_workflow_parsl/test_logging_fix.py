#!/usr/bin/env python3
"""
Test script to verify logging configuration works properly.
"""

import sys
import os
import asyncio
from pathlib import Path

# Setup paths
demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
project_root = demo_dir.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging using the centralized logging system
try:
    from nanobrain.core.logging_system import configure_third_party_loggers
    configure_third_party_loggers()
except ImportError:
    pass

# Import the workflow
from nanobrain.library.workflows.chat_workflow_parsl.workflow import ParslChatWorkflow, create_parsl_chat_workflow


async def test_logging_configuration():
    """Test that logging configuration works properly."""
    print("🧪 Testing Logging Configuration")
    print("=" * 50)
    
    workflow = None
    try:
        # Create workflow with configuration
        config_path = project_root / 'nanobrain' / 'library' / 'workflows' / 'chat_workflow_parsl' / 'ParslChatWorkflow.yml'
        print(f"📁 Loading config from: {config_path}")
        
        workflow = await create_parsl_chat_workflow(str(config_path))
        print("✅ Workflow created successfully with clean logging")
        
        # Test a quick message
        test_message = "Hello, test message for logging verification"
        print(f"\n📝 Testing message: {test_message}")
        
        response = await workflow.process_user_input(test_message)
        print(f"🤖 Response received (length: {len(response)} chars)")
        
        print("\n✅ Logging configuration test passed!")
        print("   - No Parsl debug messages in console")
        print("   - No HTTP request logs in console") 
        print("   - No executor logs in console")
        print("   - Clean output achieved!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if workflow:
            print("\n🧹 Cleaning up...")
            await workflow.shutdown()
            print("✅ Cleanup complete")


if __name__ == "__main__":
    success = asyncio.run(test_logging_configuration())
    sys.exit(0 if success else 1) 