#!/usr/bin/env python3
"""
Debug script to test workflow data flow.
"""

import asyncio
import sys
import os

# Add paths
# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def test_data_flow():
    """Test basic data flow through triggers and links."""
    print("🔍 Testing Data Flow...")
    
    from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
    from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig
    from nanobrain.core.link import DirectLink, LinkConfig
    
    # Create data units
    input_du = DataUnitMemory(DataUnitConfig(name="input"))
    output_du = DataUnitMemory(DataUnitConfig(name="output"))
    
    print("   ✅ Data units created")
    
    # Create link
    link_config = LinkConfig(link_type="direct")
    link = DirectLink(input_du, output_du, link_config, name="test_link")
    
    print("   ✅ Link created")
    
    # Create trigger
    trigger_config = TriggerConfig(trigger_type="data_updated")
    trigger = DataUpdatedTrigger([input_du], trigger_config)
    
    print("   ✅ Trigger created")
    
    # Add callback
    await trigger.add_callback(link.transfer)
    await trigger.start_monitoring()
    
    print("   ✅ Trigger monitoring started")
    
    # Test data flow
    print("   📤 Setting input data...")
    await input_du.set({"test": "data"})
    
    # Wait for trigger to process
    await asyncio.sleep(1.0)
    
    result = await output_du.get()
    print(f"   📥 Output result: {result}")
    
    await trigger.stop_monitoring()
    
    return result is not None

async def test_agent_step():
    """Test agent step processing."""
    print("\n🤖 Testing Agent Step...")
    
    from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
    from nanobrain.core.executor import LocalExecutor, ExecutorConfig
    from nanobrain.core.agent import ConversationalAgent, AgentConfig
    from nanobrain.core.step import Step, StepConfig
    from src.config import get_api_key
    
    # Check if we have API key
    if not get_api_key('openai'):
        print("   ⚠️  No OpenAI API key, skipping agent test")
        return False
    
    # Create executor
    executor_config = ExecutorConfig(executor_type="local", max_workers=2)
    executor = LocalExecutor(executor_config)
    await executor.initialize()
    
    print("   ✅ Executor created")
    
    # Create agent
    agent_config = AgentConfig(
        name="TestAgent",
        model="gpt-3.5-turbo",
        temperature=0.7,
        auto_initialize=False
    )
    agent = ConversationalAgent(agent_config, executor=executor)
    await agent.initialize()
    
    print("   ✅ Agent created")
    
    # Test direct agent processing
    try:
        response = await agent.process("Hello, what day is today?")
        print(f"   📥 Agent response: {response[:100]}...")
        
        await executor.shutdown()
        return True
        
    except Exception as e:
        print(f"   ❌ Agent error: {e}")
        await executor.shutdown()
        return False

async def test_step_execution():
    """Test step execution with data units."""
    print("\n⚙️  Testing Step Execution...")
    
    from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
    from nanobrain.core.step import Step, StepConfig
    
    # Create a simple test step
    class TestStep(Step):
        async def process(self, inputs):
            user_input = inputs.get('user_input', '')
            return {'response': f'Echo: {user_input}'}
    
    # Create data units
    input_du = DataUnitMemory(DataUnitConfig(name="input"))
    output_du = DataUnitMemory(DataUnitConfig(name="output"))
    
    # Create step
    step_config = StepConfig(name="test_step")
    step = TestStep(step_config)
    
    # Register data units
    step.register_input_data_unit('user_input', input_du)
    step.register_output_data_unit(output_du)
    
    await step.initialize()
    
    print("   ✅ Step created and initialized")
    
    # Set input data
    await input_du.set({'user_input': 'Hello world'})
    
    # Execute step
    await step.execute()
    
    # Check output
    result = await output_du.get()
    print(f"   📥 Step output: {result}")
    
    await step.shutdown()
    
    return result is not None and 'response' in result

async def main():
    """Run all debug tests."""
    print("🚀 Debugging NanoBrain Workflow")
    print("=" * 50)
    
    results = []
    
    # Test data flow
    data_flow_ok = await test_data_flow()
    results.append(("Data Flow", data_flow_ok))
    
    # Test agent step
    agent_ok = await test_agent_step()
    results.append(("Agent Processing", agent_ok))
    
    # Test step execution
    step_ok = await test_step_execution()
    results.append(("Step Execution", step_ok))
    
    # Summary
    print("\n📊 Debug Results:")
    print("=" * 30)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All components working! Issue might be in workflow integration.")
    else:
        print("\n⚠️  Some components failed. Check the errors above.")

if __name__ == "__main__":
    asyncio.run(main()) 