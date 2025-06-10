#!/usr/bin/env python3
"""
Simple test for the Parsl Chat Workflow

This test bypasses library import issues by directly importing the workflow.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Direct import of the workflow module
sys.path.insert(0, str(project_root / 'library' / 'workflows' / 'chat_workflow_parsl'))

try:
    from workflow import ParslChatWorkflow
    print("✅ Successfully imported ParslChatWorkflow")
except ImportError as e:
    print(f"❌ Failed to import ParslChatWorkflow: {e}")
    sys.exit(1)


async def test_workflow():
    """Test the Parsl chat workflow."""
    print("🚀 Testing NanoBrain Parsl Chat Workflow")
    print("=" * 50)
    
    workflow = None
    
    try:
        # Create workflow
        print("🔧 Creating workflow...")
        workflow = ParslChatWorkflow()
        
        # Initialize workflow
        print("🔧 Initializing workflow...")
        await workflow.initialize()
        
        print("✅ Workflow initialized successfully!")
        
        # Get status
        status = workflow.get_workflow_status()
        print(f"📊 Status: {status}")
        
        # Test a simple message
        print("\n📝 Testing message processing...")
        test_message = "Hello, this is a test of the Parsl chat workflow!"
        
        response = await workflow.process_user_input(test_message)
        print(f"🤖 Response: {response}")
        
        # Get performance stats
        print("\n📊 Performance Statistics:")
        stats = await workflow.get_performance_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if workflow:
            print("\n🧹 Shutting down workflow...")
            try:
                await workflow.shutdown()
                print("✅ Shutdown complete!")
            except Exception as e:
                print(f"⚠️  Shutdown error: {e}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: No OPENAI_API_KEY found.")
        print("   The test will use mock responses.")
        print()
    
    # Run the test
    asyncio.run(test_workflow()) 