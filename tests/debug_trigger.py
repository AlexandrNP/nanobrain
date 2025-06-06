#!/usr/bin/env python3
"""
Debug trigger monitoring.
"""
import sys
import os
import asyncio

# Add paths
# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def debug_trigger():
    print("🔍 Debug Trigger Monitoring...")
    
    try:
        from core.data_unit import DataUnitMemory, DataUnitConfig
        print("   ✅ Imports successful")
        
        # Create data unit
        data_unit = DataUnitMemory(DataUnitConfig(name="test"))
        await data_unit.initialize()
        print("   ✅ Data unit created and initialized")
        
        # Check initial metadata
        initial_time = await data_unit.get_metadata('last_updated', 0.0)
        print(f"   📊 Initial last_updated: {initial_time}")
        
        # Set some data
        test_data = {"message": "test"}
        print("   📤 Setting data...")
        await data_unit.set(test_data)
        
        # Check metadata after set
        after_set_time = await data_unit.get_metadata('last_updated', 0.0)
        print(f"   📊 After set last_updated: {after_set_time}")
        
        # Verify data was set
        retrieved_data = await data_unit.get()
        print(f"   📥 Retrieved data: {retrieved_data}")
        
        # Simulate trigger monitoring logic
        print("   🔍 Simulating trigger monitoring...")
        last_update_time = 0.0
        
        for i in range(5):
            current_update_time = await data_unit.get_metadata('last_updated', 0.0)
            print(f"   📊 Check {i+1}: current={current_update_time}, last={last_update_time}")
            
            if current_update_time > last_update_time:
                print(f"   ✅ Update detected! Time changed from {last_update_time} to {current_update_time}")
                data = await data_unit.get()
                print(f"   📥 Data: {data}")
                last_update_time = current_update_time
                break
            
            await asyncio.sleep(0.1)
        else:
            print("   ❌ No update detected in 5 checks")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_trigger())
    print(f"\n🎯 Debug {'PASSED' if success else 'FAILED'}") 