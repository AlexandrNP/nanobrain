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
PROJECT_ROOT = Path(__file__).resolve().parent
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

# Load global configuration
def load_global_config():
    """Load the global configuration and set up the environment."""
    try:
        from src.GlobalConfig import GlobalConfig
        
        # Initialize the global configuration
        global_config = GlobalConfig()
        
        # Load configuration from file
        global_config.load_config()
        
        # Load configuration from environment variables
        global_config.load_from_env()
        
        # Set up the environment based on the configuration
        global_config.setup_environment()
        
        # Check if we're in testing mode
        if os.environ.get('NANOBRAIN_TESTING', '0') == '1' and global_config.get('models.use_mock_in_testing', True):
            # Set the environment variable for testing mode
            os.environ['NANOBRAIN_TESTING'] = '1'
            print("Running in testing mode with mock models.")
        
        return global_config
    except ImportError as e:
        print(f"Warning: Could not load global configuration: {e}")
        return None

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
        "Step.py": SRC_DIR / "Step.py",
        "GlobalConfig.py": SRC_DIR / "GlobalConfig.py"
    }
    
    for name, path in files_to_check.items():
        exists = path.exists()
        print(f"  {name:<20}: {'✓' if exists else '✗'}")
    
    return True

# Load global configuration when the module is imported
global_config = load_global_config()

if __name__ == "__main__":
    # When run as a script, verify the paths and print configuration info
    verify_paths()
    
    if global_config:
        print("\nGlobal configuration loaded:")
        print(f"  Config path: {global_config.config_path or 'Default configuration'}")
        print(f"  API keys configured: {', '.join(k for k, v in global_config.get('api_keys', {}).items() if v) or 'None'}")
        print(f"  Default model: {global_config.get('models.default', 'Not set')}")
        print(f"  Log level: {global_config.get('framework.log_level', 'Not set')}")
    else:
        print("\nGlobal configuration not loaded.") 