#!/usr/bin/env python3
"""
Minimal test for trigger functionality.
"""
import sys
import os
import asyncio

# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def test_minimal():
    print("🔍 Minimal Trigger Test...")
    
    try:
        from core.data_unit import DataUnitMemory, DataUnitConfig
        from core.trigger import DataUpdatedTrigger, TriggerConfig
        print("   ✅ Imports successful")
        
        # Create data unit
        data_unit = DataUnitMemory(DataUnitConfig(name="test"))
        print("   ✅ Data unit created")
        
        # Create trigger
        trigger = DataUpdatedTrigger([data_unit], TriggerConfig(name="test_trigger"))
        print("   ✅ Trigger created")
        
        # Track callback calls
        callback_called = False
        callback_data = None
        
        async def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
            print(f"   📞 Callback called with data: {data}")
        
        # Add callback and start monitoring
        await trigger.add_callback(test_callback)
        await trigger.start_monitoring()
        print("   ✅ Monitoring started")
        
        # Set data (should trigger callback)
        test_data = {"message": "test"}
        print("   📤 Setting data...")
        await data_unit.set(test_data)
        
        # Wait for callback
        print("   ⏳ Waiting for callback...")
        for i in range(20):  # Wait up to 2 seconds
            if callback_called:
                break
            await asyncio.sleep(0.1)
        
        # Cleanup
        await trigger.stop_monitoring()
        
        if callback_called:
            print(f"   ✅ Success! Callback called with: {callback_data}")
            return True
        else:
            print("   ❌ Callback was not called")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_minimal())
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}") 