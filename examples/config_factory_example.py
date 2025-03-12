#!/usr/bin/env python3
"""
Example script for using ConfigManager as a Factory.

This script demonstrates how to use the ConfigManager to automatically
create instances of classes based on configuration files.
"""

import os
import sys
import asyncio

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import setup_paths to ensure paths are set up correctly
try:
    import setup_paths
    setup_paths.verify_paths()
except ImportError:
    print("Could not import setup_paths module.")

from src.ConfigManager import ConfigManager


async def main():
    """
    Demonstrate using ConfigManager as a Factory.
    """
    print("ConfigManager Factory Demo")
    print("-------------------------")
    
    # Create a ConfigManager instance
    config_manager = ConfigManager()
    
    try:
        # Create an ExecutorBase instance
        print("\nCreating ExecutorBase instance:")
        executor = config_manager.create_instance("ExecutorBase", energy_level=0.9)
        print(f"  Type: {type(executor).__name__}")
        print(f"  Energy Level: {executor.energy_level}")
        print(f"  Energy Per Execution: {executor.energy_per_execution}")
        print(f"  Recovery Rate: {executor.recovery_rate}")
        
        # Create a Step instance
        print("\nCreating Step instance:")
        step = config_manager.create_instance("Step", executor=executor)
        print(f"  Type: {type(step).__name__}")
        print(f"  State: {step._state}")
        
        # Create a DataUnitBase instance
        print("\nCreating DataUnitBase instance:")
        data_unit = config_manager.create_instance("DataUnitBase")
        print(f"  Type: {type(data_unit).__name__}")
        
        print("\nSuccess! All instances created correctly.")
        
    except Exception as e:
        print(f"Error: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 