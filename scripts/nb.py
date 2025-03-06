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

# Debug print
print(f"Script path: {__file__}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Current working directory: {os.getcwd()}")
print(f"Initial sys.path: {sys.path}")

# Set up the Python import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script directory to path to find setup_paths module
sys.path.insert(0, script_dir)

try:
    import setup_paths
    print("Successfully imported setup_paths")
except ImportError as e:
    print(f"Warning: Could not import setup_paths module: {e}")
    # Fall back to manually setting up paths
    PROJECT_ROOT = script_dir
    sys.path.insert(0, PROJECT_ROOT)
    os.environ["PYTHONPATH"] = PROJECT_ROOT

# Debug print
print(f"Updated sys.path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

# Debug print for modules
print("\nChecking loaded modules:")
for name, module in sorted(sys.modules.items()):
    if hasattr(module, '__file__') and module.__file__:
        print(f"  {name}: {module.__file__}")
    else:
        print(f"  {name}: (built-in)")

# Debug print for argparse
def debug_main():
    """Debug version of main to understand what's happening."""
    print("Entering debug_main()")
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
    
    # Parse arguments
    print(f"sys.argv: {sys.argv}")
    args = parser.parse_args()
    print(f"Parsed args: {args}")
    
    # Print help and exit
    parser.print_help()
    return

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
    # TODO: Implement this function


def show_help(args):
    """Show help information."""
    print(__doc__)
    print("For more information, use: nanobrain [command] --help")


if __name__ == "__main__":
    # Use debug_main instead of main for debugging
    debug_main() 