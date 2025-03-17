#!/usr/bin/env python3
"""
Test script to diagnose import issues.
"""

import os
import sys

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {PROJECT_ROOT}")

# Add the project root to the Python path
sys.path.insert(0, PROJECT_ROOT)
print(f"Python path: {sys.path}")

# Check if the builder directory exists
builder_dir = os.path.join(PROJECT_ROOT, "builder")
print(f"Builder directory: {builder_dir}")
print(f"Builder directory exists: {os.path.exists(builder_dir)}")

# Check if builder/__init__.py exists
init_file = os.path.join(builder_dir, "__init__.py")
print(f"Builder __init__.py exists: {os.path.exists(init_file)}")

# Check if builder/NanoBrainBuilder.py exists
builder_file = os.path.join(builder_dir, "NanoBrainBuilder.py")
print(f"NanoBrainBuilder.py exists: {os.path.exists(builder_file)}")

# Try to import the builder module
try:
    print("\nTrying to import builder module...")
    import builder
    print("Successfully imported builder module")
    
    print("\nTrying to import NanoBrainBuilder class...")
    from builder.NanoBrainBuilder import NanoBrainBuilder
    print("Successfully imported NanoBrainBuilder class")
    
    # Create an instance
    print("\nTrying to create NanoBrainBuilder instance...")
    builder = NanoBrainBuilder()
    print("Successfully created NanoBrainBuilder instance")
    
except ImportError as e:
    print(f"Import error: {e}")
    
    # Check for import errors in dependencies
    print("\nChecking for import errors in dependencies...")
    try:
        print("Importing src.Workflow...")
        import src.Workflow
        print("Success")
    except ImportError as e:
        print(f"Error importing src.Workflow: {e}")
        
    try:
        print("Importing src.Agent...")
        import src.Agent
        print("Success")
    except ImportError as e:
        print(f"Error importing src.Agent: {e}")
        
    try:
        print("Importing src.ExecutorBase...")
        import src.ExecutorBase
        print("Success")
    except ImportError as e:
        print(f"Error importing src.ExecutorBase: {e}")
    
except Exception as e:
    print(f"Other error: {e}") 