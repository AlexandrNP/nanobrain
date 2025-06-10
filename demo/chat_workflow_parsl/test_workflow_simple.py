#!/usr/bin/env python3
"""
Simple test script for Parsl Chat Workflow

Tests the basic functionality without interactive input to verify the implementation.
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


async def test_workflow():
    """Test the Parsl chat workflow functionality."""
    print("🧪 Testing NanoBrain Parsl Chat Workflow")
    print("=" * 50)
    
    workflow = None
    try:
        # Create workflow with configuration
        config_path = project_root / 'nanobrain' / 'library' / 'workflows' / 'chat_workflow_parsl' / 'ParslChatWorkflow.yml'
        print(f"📁 Loading config from: {config_path}")
        
        workflow = await create_parsl_chat_workflow(str(config_path))
        print("✅ Workflow created successfully")
        
        # Display workflow status
        status = workflow.get_workflow_status()
        print(f"📊 Workflow Status:")
        print(f"   - Name: {status['name']}")
        print(f"   - Initialized: {status['initialized']}")
        print(f"   - Agents: {status['agent_count']}")
        print(f"   - Parsl Executor: {'✅' if status['parsl_executor'] else '❌'}")
        print(f"   - Data Units: {len(status['data_units'])}")
        print(f"   - Parsl Apps: {len(status['parsl_apps'])}")
        
        # Test message processing
        test_message = "Hello! Can you tell me about distributed computing with Parsl?"
        print(f"\n📝 Testing message: {test_message}")
        
        response = await workflow.process_user_input(test_message)
        print(f"🤖 Response: {response}")
        
        # Test performance stats
        stats = await workflow.get_performance_stats()
        print(f"\n📈 Performance Stats: {stats}")
        
        print("\n✅ All tests passed!")
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
    success = asyncio.run(test_workflow())
    sys.exit(0 if success else 1) 