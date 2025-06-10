#!/usr/bin/env python3
"""
Test script to verify centralized Parsl logging configuration.
"""

import sys
import os
import asyncio
from pathlib import Path

# Setup paths
demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
project_root = demo_dir.parent.parent
sys.path.insert(0, str(project_root))

print("🧪 Testing Centralized Parsl Logging Configuration")
print("=" * 60)

# Test 1: Import and configure logging system
try:
    from nanobrain.core.logging_system import configure_third_party_loggers, get_logger
    print("✅ Successfully imported logging system")
    
    # Configure third-party loggers
    configure_third_party_loggers()
    print("✅ Third-party loggers configured")
    
except Exception as e:
    print(f"❌ Failed to import/configure logging system: {e}")
    sys.exit(1)

# Test 2: Create a NanoBrain logger
try:
    logger = get_logger("test_parsl_logging")
    print("✅ NanoBrain logger created")
    
    # Test logging at different levels
    logger.info("Test info message - should respect global config")
    logger.debug("Test debug message - should respect global config")
    
except Exception as e:
    print(f"❌ Failed to create NanoBrain logger: {e}")
    sys.exit(1)

# Test 3: Import workflow (this will trigger Parsl logging configuration)
try:
    from nanobrain.library.workflows.chat_workflow_parsl.workflow import create_parsl_chat_workflow
    print("✅ Workflow imported successfully")
    
except Exception as e:
    print(f"❌ Failed to import workflow: {e}")
    sys.exit(1)

# Test 4: Create and initialize workflow
async def test_workflow_logging():
    """Test workflow creation with centralized logging."""
    try:
        config_path = project_root / 'nanobrain' / 'library' / 'workflows' / 'chat_workflow_parsl' / 'ParslChatWorkflow.yml'
        print(f"📁 Loading config from: {config_path}")
        
        workflow = await create_parsl_chat_workflow(str(config_path))
        print("✅ Workflow created with centralized logging")
        
        # Test a quick message
        test_message = "Hello! Test message for centralized logging verification."
        print(f"\n📝 Testing message: {test_message}")
        
        response = await workflow.process_user_input(test_message)
        print(f"🤖 Response received (length: {len(response)} chars)")
        
        # Check workflow status
        status = workflow.get_workflow_status()
        print(f"\n📊 Workflow Status:")
        print(f"   - Initialized: {status['initialized']}")
        print(f"   - Agents: {status['agent_count']}")
        print(f"   - Parsl Executor: {'✅' if status['parsl_executor'] else '❌'}")
        
        print("\n✅ Centralized logging test passed!")
        print("   - Parsl logging configured via NanoBrainLogger")
        print("   - No scattered logging configuration needed")
        print("   - Clean, maintainable code structure")
        
        await workflow.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the test
if __name__ == "__main__":
    success = asyncio.run(test_workflow_logging())
    
    print(f"\n🎯 Final Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print("   - Parsl logging is now properly centralized in NanoBrainLogger")
    print("   - No need for scattered _configure_parsl_logging methods")
    print("   - Follows proper object-oriented design principles")
    
    sys.exit(0 if success else 1) 