#!/usr/bin/env python3
"""
Example demonstrating reactive data transfer using Link classes with triggers.

This example shows how to set up links that automatically transfer data
when trigger conditions are met.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import necessary modules
from src.DataUnitBase import DataUnitBase
from src.LinkDirect import LinkDirect
from src.LinkFile import LinkFile
from src.TriggerDataUpdated import TriggerDataUpdated
from src.TriggerStart import TriggerStart


class SimpleDataUnit(DataUnitBase):
    """Simple implementation of DataUnitBase for testing."""
    def __init__(self, name="DataUnit", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        
    def get(self):
        """Get the data from this data unit."""
        return self.data
        
    def set(self, data):
        """Set the data in this data unit."""
        self.data = data
        print(f"{self.name}: Data set to {data}")
        return True


async def example_reactive_link():
    """Demonstrate a link that reacts to data changes."""
    print("\n=== Example: Reactive Link with Data Change Trigger ===")
    
    # Create two data units
    source = SimpleDataUnit(name="Source")
    destination = SimpleDataUnit(name="Destination")
    
    # Create a trigger that activates when data changes
    trigger = TriggerDataUpdated()
    
    # Create a direct link with the trigger
    link = LinkDirect(input_data=source, output_data=destination, trigger=trigger)
    
    # Start monitoring for data changes
    print("Starting to monitor for data changes...")
    await link.start_monitoring()
    
    # Change the data in the source, which should trigger the link to transfer
    for i in range(3):
        value = f"Value {i}"
        print(f"\nSetting source to '{value}'")
        source.set(value)
        
        # Wait a moment to allow the trigger to detect the change
        await asyncio.sleep(0.2)
        
        # Verify the data was transferred
        print(f"Destination value: {destination.get()}")
        
    print("\nReactive link example completed.")


async def example_multiple_links():
    """Demonstrate multiple links with different triggers."""
    print("\n=== Example: Multiple Links with Different Triggers ===")
    
    # Create data units
    source1 = SimpleDataUnit(name="Source1")
    source2 = SimpleDataUnit(name="Source2")
    middle = SimpleDataUnit(name="Middle")
    destination = SimpleDataUnit(name="Destination")
    
    # Create triggers
    trigger1 = TriggerDataUpdated()  # Reacts to data changes
    trigger2 = TriggerStart()  # Always activates (like a continuous connection)
    
    # Create links
    link1 = LinkDirect(input_data=source1, output_data=middle, trigger=trigger1)
    link2 = LinkDirect(input_data=middle, output_data=destination, trigger=trigger2)
    
    # Start monitoring for both links
    print("Starting link monitoring...")
    await link1.start_monitoring()
    await link2.start_monitoring()
    
    # Test the chain reaction: source1 -> middle -> destination
    print("\nTesting chain reaction:")
    print("Setting Source1, which should trigger Link1 to transfer to Middle,")
    print("then Link2 should automatically transfer from Middle to Destination")
    
    # Set data in source1
    value = "Chain reaction test"
    print(f"\nSetting Source1 to '{value}'")
    source1.set(value)
    
    # Wait a moment for the chain to complete
    await asyncio.sleep(0.5)
    
    # Check the results
    print(f"Middle value: {middle.get()}")
    print(f"Destination value: {destination.get()}")
    
    print("\nMultiple links example completed.")


async def main():
    """Run all the examples."""
    print("Link Trigger Examples")
    print("====================")
    print("Demonstrating reactive data transfer using triggers with Link classes.")
    
    # Run the examples
    await example_reactive_link()
    await example_multiple_links()
    
    print("\nAll examples completed.")


if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main()) 