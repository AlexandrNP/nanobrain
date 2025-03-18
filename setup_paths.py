#!/usr/bin/env python3
"""
Setup paths for NanoBrain framework.

This module sets up the Python import paths for the NanoBrain framework
to ensure all modules can be imported properly.
"""

import os
import sys

def setup_paths():
    """
    Set up the Python import paths for the NanoBrain framework.
    This ensures that all modules can be imported correctly.
    """
    # Get the current directory (should be the project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the project root to the path if not already there
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Add specific module directories to the path
    paths_to_add = [
        os.path.join(current_dir, 'prompts'),
        os.path.join(current_dir, 'src'),
        os.path.join(current_dir, 'builder'),
        os.path.join(current_dir, 'test'),
        os.path.join(current_dir, 'integration_tests'),
        os.path.join(current_dir, 'tools_common'),
    ]
    
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    return True

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
    print(f"Project root: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Python path: {sys.path}")
    
    # Check that key directories exist
    print("\nChecking directories:")
    dirs_to_check = {
        "src": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'),
        "builder": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'builder'),
        "tools_common": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools_common'),
        "prompts": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts'),
        "builder/config": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'builder', 'config')
    }
    
    for name, path in dirs_to_check.items():
        exists = os.path.exists(path)
        has_init = (os.path.exists(os.path.join(path, '__init__.py')) or os.path.exists(os.path.join(path, '__main__.py'))) if exists else False
        print(f"  {name:<12}: {'✓' if exists else '✗'} (exists) {'✓' if has_init else '✗'} (__init__.py)")
    
    # Check key files
    print("\nChecking key files:")
    files_to_check = {
        "ConfigManager.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'ConfigManager.py'),
        "NanoBrainBuilder.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'builder', 'NanoBrainBuilder.py'),
        "Agent.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'Agent.py'),
        "Step.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'Step.py'),
        "GlobalConfig.py": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'GlobalConfig.py'),
        "tools.yml": os.path.join(os.path.dirname(os.path.abspath(__file__)), 'builder', 'config', 'tools.yml')
    }
    
    for name, path in files_to_check.items():
        exists = os.path.exists(path)
        print(f"  {name:<20}: {'✓' if exists else '✗'}")
    
    # Check if tools.yml exists and print its content
    tools_yml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'builder', 'config', 'tools.yml')
    if os.path.exists(tools_yml_path):
        print(f"\nFound tools.yml at: {tools_yml_path}")
        try:
            import yaml
            with open(tools_yml_path, 'r') as f:
                tools_config = yaml.safe_load(f)
                print(f"  Contains {len(tools_config.get('tools', []))} tool definitions")
                for i, tool in enumerate(tools_config.get('tools', [])):
                    print(f"    {i+1}. {tool.get('name', 'Unnamed')}: {tool.get('class', 'No class')}")
        except Exception as e:
            print(f"  Error reading tools.yml: {e}")
    else:
        print(f"\nNo tools.yml found at: {tools_yml_path}")
    
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