#!/usr/bin/env python3
import sys
import os
import asyncio

# Add paths for imports from tests directory
parent_dir = os.path.dirname(os.path.dirname(__file__))  # nanobrain directory
sys.path.insert(0, os.path.join(parent_dir, 'src'))     # for src modules

async def test_basic_flow():
    print("🔍 Testing Basic Data Flow...")
    
    try:
        from nanobrain.core.data_unit import DataUnitMemory, DataUnitConfig
        from nanobrain.core.link import DirectLink, LinkConfig
        print("   ✅ Imports successful")
        
        # Create data units
        input_du = DataUnitMemory(DataUnitConfig(name="input"))
        output_du = DataUnitMemory(DataUnitConfig(name="output"))
        print("   ✅ Data units created")
        
        # Create and start link
        link_config = LinkConfig(link_type="direct")
        link = DirectLink(input_du, output_du, link_config, name="test_link")
        await link.start()
        print("   ✅ Link created and started")
        
        # Test data transfer
        test_data = {"message": "Hello World"}
        await input_du.set(test_data)
        await link.transfer(test_data)
        
        result = await output_du.get()
        print(f"   📥 Transfer result: {result}")
        
        if result == test_data:
            print("   ✅ Data transfer successful!")
            return True
        else:
            print("   ❌ Data transfer failed - data mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_flow())
    print(f"\n🎯 Test {'PASSED' if success else 'FAILED'}") 