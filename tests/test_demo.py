#!/usr/bin/env python3
"""
Test script to verify the chat workflow demo components are working.
"""

import asyncio
import sys
import os

# Add nanobrain src to path
# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def test_configuration():
    """Test the configuration system."""
    print("🔧 Testing Configuration System...")
    
    try:
        from nanobrain.config import get_config_manager, get_api_key
        
        config_manager = get_config_manager()
        print("   ✅ Configuration manager loaded")
        
        # Test API key loading
        openai_key = get_api_key('openai')
        anthropic_key = get_api_key('anthropic')
        
        print(f"   OpenAI key available: {'✅' if openai_key else '❌'}")
        print(f"   Anthropic key available: {'✅' if anthropic_key else '❌'}")
        
        available_providers = config_manager.get_available_providers()
        print(f"   Available providers: {available_providers}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

async def test_core_components():
    """Test core NanoBrain components."""
    print("\n🧠 Testing Core Components...")
    
    try:
        from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
        from nanobrain.core.executor import LocalExecutor, ExecutorConfig
        from nanobrain.core.agent import ConversationalAgent, AgentConfig
        
        # Test executor
        executor_config = ExecutorConfig(executor_type="local", max_workers=2)
        executor = LocalExecutor(executor_config)
        await executor.initialize()
        print("   ✅ Executor initialized")
        
        # Test data unit
        du_config = DataUnitConfig(name="test", data_type="memory")
        data_unit = DataUnitMemory(du_config)
        await data_unit.set({"test": "data"})
        result = await data_unit.get()
        print(f"   ✅ Data unit working: {result}")
        
        # Test agent (only if API key is available)
        if os.getenv('OPENAI_API_KEY'):
            agent_config = AgentConfig(
                name="TestAgent",
                model="gpt-3.5-turbo",
                temperature=0.7,
                auto_initialize=False
            )
            agent = ConversationalAgent(agent_config, executor=executor)
            await agent.initialize()
            print("   ✅ Agent initialized")
        else:
            print("   ⚠️  Skipping agent test (no API key)")
        
        await executor.shutdown()
        print("   ✅ All components working")
        return True
        
    except Exception as e:
        print(f"   ❌ Component error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_threading_simulation():
    """Test threading behavior similar to the demo."""
    print("\n🧵 Testing Threading Simulation...")
    
    try:
        import threading
        from concurrent.futures import Future
        
        # Store the event loop
        event_loop = asyncio.get_running_loop()
        
        # Create a queue for communication
        input_queue = asyncio.Queue()
        
        def thread_function():
            """Simulate the input thread."""
            try:
                # Simulate putting data into the queue from a thread
                future = asyncio.run_coroutine_threadsafe(
                    input_queue.put("test_message"), 
                    event_loop
                )
                future.result(timeout=1.0)
                return True
            except Exception as e:
                print(f"   ❌ Thread error: {e}")
                return False
        
        # Start the thread
        thread = threading.Thread(target=thread_function)
        thread.start()
        
        # Wait for the message in the main async context
        try:
            message = await asyncio.wait_for(input_queue.get(), timeout=2.0)
            print(f"   ✅ Threading communication working: {message}")
            thread.join()
            return True
        except asyncio.TimeoutError:
            print("   ❌ Threading communication timeout")
            thread.join()
            return False
            
    except Exception as e:
        print(f"   ❌ Threading test error: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Testing NanoBrain Chat Workflow Demo Components")
    print("=" * 60)
    
    results = []
    
    # Test configuration
    config_ok = await test_configuration()
    results.append(("Configuration", config_ok))
    
    # Test core components
    components_ok = await test_core_components()
    results.append(("Core Components", components_ok))
    
    # Test threading
    threading_ok = await test_threading_simulation()
    results.append(("Threading", threading_ok))
    
    # Summary
    print("\n📊 Test Results:")
    print("=" * 30)
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 30)
    if all_passed:
        print("🎉 All tests passed! The demo should work correctly.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main()) 