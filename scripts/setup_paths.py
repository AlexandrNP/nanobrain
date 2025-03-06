#!/usr/bin/env python3
"""
Path setup for NanoBrain

This module sets up the necessary Python import paths for the NanoBrain project.
It should be imported at the beginning of scripts to ensure all modules can be found.
"""

import os
import sys
from pathlib import Path

# Determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if not PROJECT_ROOT.exists():
    raise RuntimeError(f"Project root directory {PROJECT_ROOT} does not exist")

# Add the project root to sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add src directory to sys.path
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Add builder directory to sys.path
BUILDER_DIR = PROJECT_ROOT / "builder"
if str(BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_DIR))

# Set PYTHONPATH environment variable
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

def verify_paths():
    """Print path information and check if key directories and files exist."""
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python path: {sys.path}")
    
    # Check that key directories exist
    print("\nChecking directories:")
    dirs_to_check = {
        "src": SRC_DIR,
        "builder": BUILDER_DIR,
        "tools_common": PROJECT_ROOT / "tools_common",
        "prompts": PROJECT_ROOT / "prompts"
    }
    
    for name, path in dirs_to_check.items():
        exists = path.exists()
        has_init = (path / "__init__.py").exists() if exists else False
        print(f"  {name:<12}: {'✓' if exists else '✗'} (exists) {'✓' if has_init else '✗'} (__init__.py)")
    
    # Check key files
    print("\nChecking key files:")
    files_to_check = {
        "ConfigManager.py": SRC_DIR / "ConfigManager.py",
        "NanoBrainBuilder.py": BUILDER_DIR / "NanoBrainBuilder.py",
        "Agent.py": SRC_DIR / "Agent.py",
        "Step.py": SRC_DIR / "Step.py"
    }
    
    for name, path in files_to_check.items():
        exists = path.exists()
        print(f"  {name:<20}: {'✓' if exists else '✗'}")
    
    return True

if __name__ == "__main__":
    # When run as a script, verify the paths
    verify_paths() 