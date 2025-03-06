#!/usr/bin/env python3
"""
NanoBrain Command Line Interface

This script provides a command-line interface for the NanoBrain framework,
allowing users to create, configure, and run NanoBrain components from the command line.

Usage:
    nanobrain [command] [options]

Commands:
    builder         Start the interactive workflow builder
    build-docs      Build documentation from source code and configuration files
    create          Create a new NanoBrain component
    run             Run a NanoBrain workflow
    config          Manage NanoBrain configurations
    help            Show help information
"""

import os
import sys
import argparse
import subprocess
import pickle
from pathlib import Path

# Set up the Python import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script directory to path to find setup_paths module
sys.path.insert(0, script_dir)

try:
    import setup_paths
    print("Successfully imported setup_paths")
    
    # Get the global configuration
    global_config = setup_paths.global_config
    if global_config:
        print(f"Global configuration loaded from: {global_config.config_path or 'default configuration'}")
    else:
        print("Warning: Global configuration not loaded.")
except ImportError as e:
    print(f"Warning: Could not import setup_paths module: {e}")
    # Fall back to manually setting up paths
    PROJECT_ROOT = script_dir
    sys.path.insert(0, PROJECT_ROOT)
    os.environ["PYTHONPATH"] = PROJECT_ROOT
    global_config = None


def build_workflow(args):
    """Start the interactive workflow builder."""
    print("Starting NanoBrain workflow builder...")
    
    try:
        # Print path information for debugging
        if args.verbose:
            print(f"Python path: {sys.path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            
        # Try to import the setup_paths module to ensure paths are set up correctly
        try:
            import setup_paths
            if args.verbose:
                setup_paths.verify_paths()
        except ImportError:
            if args.verbose:
                print("Could not import setup_paths module.")
        
        # Import the builder
        from builder.NanoBrainBuilder import NanoBrainBuilder
        
        # Check if we should load an existing session
        if args.load_session:
            try:
                with open(args.load_session, 'rb') as f:
                    builder = pickle.load(f)
                print(f"Loaded session from {args.load_session}")
            except Exception as e:
                print(f"Error loading session: {e}")
                print("Starting a new session instead.")
                builder = NanoBrainBuilder()
        else:
            # Create a new builder
            builder = NanoBrainBuilder()
        
        # Start the builder
        if args.headless:
            # Run in non-interactive mode
            print("Running in non-interactive mode...")
            # TODO: Implement non-interactive mode
        else:
            # Run in interactive mode
            import asyncio
            asyncio.run(builder.main())
        
        # Save the session if requested
        if args.save_session:
            try:
                with open(args.save_session, 'wb') as f:
                    pickle.dump(builder, f)
                print(f"Session saved to {args.save_session}")
            except Exception as e:
                print(f"Error saving session: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
        print("The builder module could not be found. Make sure it's installed and in your Python path.")
        print(f"Current Python path includes: {sys.path}")
        
        # Try to help diagnose the issue
        builder_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builder")
        print(f"Builder directory exists: {os.path.exists(builder_dir)}")
        init_file = os.path.join(builder_dir, "__init__.py")
        print(f"Builder __init__.py exists: {os.path.exists(init_file)}")
        builder_file = os.path.join(builder_dir, "NanoBrainBuilder.py")
        print(f"NanoBrainBuilder.py exists: {os.path.exists(builder_file)}")
        
        sys.exit(1)
    except Exception as e:
        print(f"Error starting builder: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def build_docs(args):
    """Build documentation from source code and configuration files."""
    print("Building documentation...")
    # TODO: Implement this function


def create_component(args):
    """Create a new NanoBrain component."""
    print(f"Creating new {args.type} component...")
    # TODO: Implement this function


def run_workflow(args):
    """Run a NanoBrain workflow."""
    print(f"Running workflow from {args.file}...")
    # TODO: Implement this function


def manage_config(args):
    """Manage NanoBrain configurations."""
    print("Managing configurations...")
    
    if args.action == "list":
        # List all configuration settings
        if global_config:
            print("\nCurrent configuration:")
            print(f"  Config path: {global_config.config_path or 'Default configuration'}")
            
            # Print API keys (masked for security)
            print("\nAPI Keys:")
            for provider, key in global_config.get('api_keys', {}).items():
                masked_key = "Not set"
                if key:
                    masked_key = key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "****"
                print(f"  {provider}: {masked_key}")
            
            # Print model settings
            print("\nModel Settings:")
            print(f"  Default model: {global_config.get('models.default', 'Not set')}")
            print(f"  Use mock in testing: {global_config.get('models.use_mock_in_testing', 'Not set')}")
            
            # Print framework settings
            print("\nFramework Settings:")
            print(f"  Log level: {global_config.get('framework.log_level', 'Not set')}")
            print(f"  Temp directory: {global_config.get('framework.temp_dir', 'Not set')}")
            print(f"  Enable telemetry: {global_config.get('framework.enable_telemetry', 'Not set')}")
            
            # Print development settings
            print("\nDevelopment Settings:")
            print(f"  Debug mode: {global_config.get('development.debug', 'Not set')}")
            print(f"  Verbose output: {global_config.get('development.verbose', 'Not set')}")
        else:
            print("Global configuration not loaded.")
    
    elif args.action == "edit":
        # Edit the configuration
        if not global_config:
            print("Global configuration not loaded.")
            return
        
        if args.key and args.value:
            # Set a specific configuration value
            global_config.set(args.key, args.value)
            global_config.save_config()
            print(f"Configuration updated: {args.key} = {args.value}")
        elif args.key:
            # Get a specific configuration value
            value = global_config.get(args.key, "Not set")
            print(f"{args.key} = {value}")
        else:
            # Open the configuration file in an editor
            config_path = global_config.config_path
            if not config_path:
                config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yml")
                global_config.save_config(config_path)
            
            # Try to open the configuration file in the default editor
            try:
                if sys.platform == "win32":
                    os.startfile(config_path)
                elif sys.platform == "darwin":
                    subprocess.run(["open", config_path])
                else:
                    subprocess.run(["xdg-open", config_path])
                print(f"Opened configuration file: {config_path}")
            except Exception as e:
                print(f"Error opening configuration file: {e}")
                print(f"You can edit the file manually at: {config_path}")
    
    elif args.action == "reset":
        # Reset the configuration to defaults
        if global_config:
            global_config._config = global_config._get_default_config()
            global_config.save_config()
            print("Configuration reset to defaults.")
        else:
            print("Global configuration not loaded.")


def show_help(args):
    """Show help information."""
    print(__doc__)
    print("For more information, use: nanobrain [command] --help")


def main():
    """Main entry point for the NanoBrain CLI."""
    parser = argparse.ArgumentParser(
        description="NanoBrain Command Line Interface",
        usage="nanobrain [command] [options]"
    )
    
    # Add version information
    parser.add_argument('--version', action='version', version='NanoBrain CLI v0.1.0')
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Build command
    build_parser = subparsers.add_parser('builder', help='Start the interactive workflow builder')
    build_parser.add_argument('--headless', action='store_true', help='Run in non-interactive mode')
    build_parser.add_argument('--load-session', type=str, help='Load a saved builder session')
    build_parser.add_argument('--save-session', type=str, help='Save the builder session to a file')
    build_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    build_parser.set_defaults(func=build_workflow)
    
    # Build-docs command
    docs_parser = subparsers.add_parser('build-docs', help='Build documentation')
    docs_parser.add_argument('--output', type=str, default='docs', help='Output directory')
    docs_parser.set_defaults(func=build_docs)
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new component')
    create_parser.add_argument('type', choices=['step', 'agent', 'workflow'], help='Type of component to create')
    create_parser.add_argument('name', help='Name of the component')
    create_parser.set_defaults(func=create_component)
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a workflow')
    run_parser.add_argument('file', help='Workflow file to run')
    run_parser.set_defaults(func=run_workflow)
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configurations')
    config_parser.add_argument('action', choices=['list', 'edit', 'reset'], help='Configuration action')
    config_parser.add_argument('--key', help='Configuration key (e.g., "api_keys.openai")')
    config_parser.add_argument('--value', help='Configuration value')
    config_parser.set_defaults(func=manage_config)
    
    # Help command
    help_parser = subparsers.add_parser('help', help='Show help information')
    help_parser.set_defaults(func=show_help)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        show_help(args)
        return
    
    # Execute the selected command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        show_help(args)


if __name__ == "__main__":
    main() 