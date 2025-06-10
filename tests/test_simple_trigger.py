#!/usr/bin/env python3
"""
Simple test to isolate trigger and link functionality.
"""
import sys
import os
import asyncio

# Add paths
# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def test_trigger_and_link():
    print("🔍 Testing Trigger and Link Isolation...")
    
    try:
        from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
        from nanobrain.core.trigger import DataUpdatedTrigger, TriggerConfig
        from nanobrain.core.link import DirectLink, LinkConfig
        print("   ✅ Imports successful")
        
        # Create data units
        source_du = DataUnitMemory(DataUnitConfig(name="source"))
        target_du = DataUnitMemory(DataUnitConfig(name="target"))
        print("   ✅ Data units created")
        
        # Create and start link
        link = DirectLink(source_du, target_du, LinkConfig(), name="test_link")
        await link.start()
        print("   ✅ Link created and started")
        
        # Test 1: Direct link transfer
        print("   📤 Test 1: Direct link transfer")
        test_data = {"message": "Direct transfer test"}
        await link.transfer(test_data)
        result = await target_du.get()
        print(f"      Result: {result}")
        
        # Clear target for next test
        await target_du.clear()
        
        # Test 2: Trigger with callback
        print("   📤 Test 2: Trigger with callback")
        trigger = DataUpdatedTrigger([source_du], TriggerConfig(name="test_trigger"))
        
        # Add callback
        await trigger.add_callback(link.transfer)
        await trigger.start_monitoring()
        print("      Trigger monitoring started")
        
        # Set data in source (should trigger callback)
        test_data2 = {"message": "Trigger test"}
        await source_du.set(test_data2)
        
        # Wait for trigger processing
        await asyncio.sleep(1.0)
        
        result2 = await target_du.get()
        print(f"      Result: {result2}")
        
        # Cleanup
        await trigger.stop_monitoring()
        
        # Check results
        if result == {"message": "Direct transfer test"} and result2 == {"message": "Trigger test"}:
            print("   ✅ Both tests successful!")
            return True
        else:
            print("   ❌ One or both tests failed")
            print(f"      Expected direct: {{'message': 'Direct transfer test'}}, got: {result}")
            print(f"      Expected trigger: {{'message': 'Trigger test'}}, got: {result2}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_trigger_and_link())
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}") 