import asyncio
import sys
import traceback
from typing import Any, Dict, List, Optional
import os

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ExecutorBase import ExecutorBase
from builder.WorkflowSteps import CreateStep
from builder.debug_tools import apply_monitoring

def print_banner():
    """Print a banner with information about the NanoBrain framework."""
    banner = """
  _   _                  ____              _       
 | \ | | __ _ _ __   ___|  _ \   _ __ __ _(_)_ __  
 |  \| |/ _` | '_ \ / _ \ |_) | | '__/ _` | | '_ \ 
 | |\  | (_| | | | | (_) | __<  | | | (_| | | | | |
 |_| \_|\__,_|_| |_|\___/\_`_`\ |_|  \__,_|_|_| |_|
                                                 
 Biologically Inspired Framework
 ===============================
 Using Unified Link Interface & Trigger-based Data Flow
"""
    print(banner)

async def main():
    """
    Main entry point for the NanoBrain Builder.
    
    This sets up the workflow creation process and handles errors.
    """
    print_banner()
    print("Starting NanoBrain Builder...")
    
    try:
        # Create executor
        print("Initializing Executor...")
        executor = ExecutorBase()
        
        # Execute CreateStep directly with appropriate parameters
        print("Creating workflow step...")
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        
        print("\nSetting up workflow builder...")
        # Call CreateStep.execute with the builder and step name
        result = await CreateStep.execute(builder, "MyFirstStep")
        
        # Apply monitoring was here, but we don't need it as monitoring
        # is handled inside the CreateStep.execute method
        
        # Print data flow information
        print("\nData Flow Architecture:")
        print("------------------------")
        print("1. User Input -> DataStorageCommandLine")
        print("2. DataStorageCommandLine -> LinkDirect (via TriggerDataUpdated)")
        print("3. LinkDirect -> AgentWorkflowBuilder")
        print("4. AgentWorkflowBuilder processes input and returns result")
        print("\nBackup Direct Path:")
        print("DataStorageCommandLine -> AgentWorkflowBuilder (direct method call)")
        
        print("Builder terminated.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error in main process: {e}")
        traceback.print_exc()
    finally:
        # Perform cleanup
        print("Cleaning up...")
        
if __name__ == "__main__":
    # Run the main function
    try:
        # Create any necessary directories
        os.makedirs("logs", exist_ok=True)
        
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        # Exit cleanly
        sys.exit(0) 