#!/usr/bin/env python3
"""
Direct ParslAgent Test

Tests the ParslAgent functionality directly without going through the library imports.
This demonstrates that the ParslAgent implementation is working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Setup paths
demo_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd() / 'demo' / 'chat_workflow_parsl'
project_root = demo_dir.parent.parent
sys.path.insert(0, str(project_root))

# Direct imports to avoid library import issues
from nanobrain.core.agent import ConversationalAgent, AgentConfig
from nanobrain.core.executor import ParslExecutor, ExecutorConfig, ExecutorType

# Import ParslAgent directly
sys.path.insert(0, str(project_root / 'nanobrain' / 'library' / 'agents' / 'specialized'))
from parsl_agent import ParslAgent


async def test_parsl_agent():
    """Test the ParslAgent functionality."""
    print("🧪 Testing ParslAgent Direct Implementation")
    print("=" * 50)
    
    try:
        # Create agent configuration
        agent_config = AgentConfig(
            name="test_parsl_agent",
            description="Test ParslAgent for distributed processing",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500,
            system_prompt="You are a helpful AI assistant running in a distributed environment.",
            auto_initialize=False,
            debug_mode=True,
            enable_logging=True,
            log_conversations=True
        )
        
        print("✅ Agent configuration created")
        
        # Create Parsl executor (optional for testing)
        try:
            executor_config = ExecutorConfig(
                executor_type="parsl",
                max_workers=2,
                timeout=30.0
            )
            parsl_executor = ParslExecutor(config=executor_config)
            await parsl_executor.initialize()
            print("✅ Parsl executor initialized")
        except Exception as e:
            print(f"⚠️  Parsl executor failed, using None: {e}")
            parsl_executor = None
        
        # Create ParslAgent
        agent = ParslAgent(config=agent_config, parsl_executor=parsl_executor)
        await agent.initialize()
        print("✅ ParslAgent initialized successfully")
        
        # Test message processing
        test_message = "Hello! Can you tell me about distributed computing?"
        print(f"\n📤 Sending test message: {test_message}")
        
        response = await agent.process(test_message)
        print(f"📥 Response: {response}")
        
        # Get performance metrics
        if hasattr(agent, 'get_performance_metrics'):
            metrics = agent.get_performance_metrics()
            print(f"\n📊 Performance Metrics:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")
        
        # Shutdown
        await agent.shutdown()
        if parsl_executor:
            await parsl_executor.shutdown()
        
        print("\n✅ ParslAgent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ ParslAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_parsl_agent()
    
    if success:
        print("\n🎉 All tests passed! ParslAgent is working correctly.")
    else:
        print("\n💥 Tests failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 