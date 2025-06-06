#!/usr/bin/env python3
"""
Test that simulates the exact workflow from the demo.
"""
import sys
import os
import asyncio

# Add paths
# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def test_workflow_simulation():
    print("🔍 Testing Complete Workflow Simulation...")
    
    try:
        from core.data_unit import DataUnitMemory, DataUnitConfig
        from core.trigger import DataUpdatedTrigger, TriggerConfig
        from core.link import DirectLink, LinkConfig
        from core.step import Step, StepConfig
        from core.executor import LocalExecutor, ExecutorConfig
        print("   ✅ Imports successful")
        
        # Create a simple test step
        class TestStep(Step):
            async def process(self, input_data):
                user_input = input_data.get('user_input', '')
                return {'agent_response': f'Echo: {user_input}'}
        
        # 1. Create data units (same as demo)
        user_input_du = DataUnitMemory(DataUnitConfig(name="user_input"))
        agent_input_du = DataUnitMemory(DataUnitConfig(name="agent_input"))
        agent_output_du = DataUnitMemory(DataUnitConfig(name="agent_output"))
        print("   ✅ Data units created")
        
        # 2. Create test step
        # Create executor for the step
        executor_config = ExecutorConfig(executor_type="local", max_workers=1)
        executor = LocalExecutor(executor_config)
        
        step_config = StepConfig(name="test_step")
        test_step = TestStep(step_config, executor=executor)
        test_step.register_input_data_unit('user_input', agent_input_du)
        test_step.register_output_data_unit(agent_output_du)
        await test_step.initialize()
        print("   ✅ Test step created")
        
        # 3. Create triggers
        user_trigger = DataUpdatedTrigger([user_input_du], TriggerConfig(name="user_trigger"))
        agent_trigger = DataUpdatedTrigger([agent_input_du], TriggerConfig(name="agent_trigger"))
        print("   ✅ Triggers created")
        
        # 4. Create links
        user_to_agent_link = DirectLink(user_input_du, agent_input_du, LinkConfig(), name="user_to_agent")
        await user_to_agent_link.start()
        print("   ✅ Links created and started")
        
        # 5. Set up callbacks
        async def execute_step_callback(data):
            """Wrapper to execute step without passing data as positional argument."""
            await test_step.execute()
        
        await user_trigger.add_callback(user_to_agent_link.transfer)
        await agent_trigger.add_callback(execute_step_callback)
        print("   ✅ Callbacks set up")
        
        # 6. Start monitoring
        await user_trigger.start_monitoring()
        await agent_trigger.start_monitoring()
        print("   ✅ Monitoring started")
        
        # 7. Test the workflow
        print("   📤 Setting user input...")
        await user_input_du.set({'user_input': 'Hello World'})
        
        # Wait for processing
        print("   ⏳ Waiting for processing...")
        await asyncio.sleep(2.0)
        
        # Check results
        agent_output = await agent_output_du.get()
        print(f"   📥 Agent output: {agent_output}")
        
        # Cleanup
        await user_trigger.stop_monitoring()
        await agent_trigger.stop_monitoring()
        await test_step.shutdown()
        
        if agent_output and 'agent_response' in agent_output:
            print("   ✅ Workflow simulation successful!")
            return True
        else:
            print("   ❌ Workflow simulation failed - no output")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow_simulation())
    print(f"\n🎯 Workflow Test {'PASSED' if success else 'FAILED'}") 