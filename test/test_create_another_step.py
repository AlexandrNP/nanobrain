#!/usr/bin/env python3
"""
Test script for creating a step with NanoBrainBuilder.

This script creates a step using NanoBrainBuilder without user interaction.
"""

import os
import sys
import asyncio
import shutil
import signal
import time
import threading
import subprocess
from pathlib import Path

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import setup_paths to ensure paths are set up correctly
try:
    import setup_paths
    setup_paths.verify_paths()
except ImportError:
    print("Could not import setup_paths module.")

# Import necessary modules
from builder.NanoBrainBuilder import NanoBrainBuilder

def run_with_timeout(func, timeout=30):
    """Run a function with a timeout."""
    result = {"success": False, "error": "Timeout"}
    
    def target():
        """Target function to run with timeout."""
        try:
            result["result"] = func()
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    
    try:
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            # If the thread is still running after the timeout, return the default result
            print(f"Function timed out after {timeout} seconds.")
            return result
    except Exception as e:
        result["error"] = str(e)
    
    return result

async def create_step():
    """Create a step programmatically."""
    # Initialize the builder
    builder = NanoBrainBuilder()
    
    # Define workflow and step
    workflow_name = "test_workflow"
    step_name = "NewTestStep"
    step_description = "A test step created programmatically"
    
    # Get workflow path
    workflow_path = builder.get_workflow_path(workflow_name)
    if not workflow_path:
        # Create the workflow if it doesn't exist
        result = builder.create_workflow(workflow_name)
        if not result.get("success", False):
            print(f"Failed to create workflow: {result.get('error', 'Unknown error')}")
            return False
        workflow_path = builder.get_workflow_path(workflow_name)
    
    # Set as current workflow
    builder.push_workflow(workflow_path)
    print(f"Using workflow: {workflow_name}")
    
    # Check if step directory already exists
    class_name = f"Step{step_name}"
    step_dir = os.path.join(workflow_path, 'src', class_name)
    if os.path.exists(step_dir):
        print(f"Removing existing step directory: {step_dir}")
        shutil.rmtree(step_dir)
    
    # Run the step creation in a separate process
    print(f"Creating step: {step_name}")
    
    # Start a detached process to create the step
    cmd = f"cd {parent_dir} && python -c 'from builder.NanoBrainBuilder import NanoBrainBuilder; import asyncio; builder = NanoBrainBuilder(); builder.push_workflow(\"{workflow_path}\"); asyncio.run(builder.create_step(\"{step_name}\", \"Step\", \"{step_description}\"))'"
    
    process = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    
    # Give it some time to start
    time.sleep(5)
    
    # Check if the process is running
    if process.poll() is None:
        # Process is running, send the 'finish' command
        print("Sending 'finish' command to the step creation process...")
        finish_cmd = f"echo 'finish' > /tmp/nanobrain_input.txt"
        subprocess.run(finish_cmd, shell=True)
        
        # Wait for the process to finish
        print("Waiting for the step creation process to finish...")
        process.wait(timeout=60)
        
        # Check if the step was created
        if os.path.exists(step_dir):
            print(f"Step directory created: {step_dir}")
            
            # List files in the step directory
            print("\nFiles created:")
            for root, dirs, files in os.walk(step_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, step_dir)
                    print(f"  {rel_path}")
            
            return True
        else:
            print(f"Step directory not found: {step_dir}")
            return False
    else:
        # Process exited before we could send the 'finish' command
        print("Step creation process exited unexpectedly.")
        stdout, stderr = process.communicate()
        print(f"Process output: {stdout.decode('utf-8')}")
        print(f"Process error: {stderr.decode('utf-8')}")
        return False
    
    # Try to check if the step directory was created
    return os.path.exists(step_dir)

if __name__ == "__main__":
    print("Testing programmatic step creation...\n")
    
    # Run with a timeout
    result = run_with_timeout(lambda: asyncio.run(create_step()), timeout=60)
    
    # Check the result
    if result.get("success", False):
        if result.get("result", False):
            print("\nStep creation successful!")
            sys.exit(0)
        else:
            print("\nStep creation failed!")
            sys.exit(1)
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")
        sys.exit(1) 