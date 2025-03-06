#!/usr/bin/env python3
"""
NanoBrain Command Line Interface

This script provides a command-line interface for the NanoBrain framework,
allowing users to create, configure, and run NanoBrain components from the command line.

Usage:
    python nanobrain_cli.py [command] [options]

Commands:
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
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def build_docs(args):
    """Build documentation from source code and configuration files."""
    print("Building documentation...")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "build_docs.sh")
    subprocess.run(["bash", script_path], check=True)
    print("Documentation built successfully!")


def create_component(args):
    """Create a new NanoBrain component."""
    component_type = args.type
    component_name = args.name
    
    print(f"Creating new {component_type}: {component_name}...")
    # TODO: Implement component creation logic
    print(f"{component_type} '{component_name}' created successfully!")


def run_workflow(args):
    """Run a NanoBrain workflow."""
    workflow_file = args.file
    
    print(f"Running workflow from {workflow_file}...")
    # TODO: Implement workflow execution logic
    print("Workflow execution completed!")


def manage_config(args):
    """Manage NanoBrain configurations."""
    action = args.action
    config_file = args.file
    
    if action == "show":
        print(f"Showing configuration from {config_file}...")
        # TODO: Implement configuration display logic
    elif action == "edit":
        print(f"Editing configuration {config_file}...")
        # TODO: Implement configuration editing logic
    elif action == "validate":
        print(f"Validating configuration {config_file}...")
        # TODO: Implement configuration validation logic
    
    print("Configuration management completed!")


def show_help(args):
    """Show help information."""
    parser.print_help()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="NanoBrain Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1]
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Build docs command
    docs_parser = subparsers.add_parser("build-docs", help="Build documentation")
    docs_parser.set_defaults(func=build_docs)
    
    # Create component command
    create_parser = subparsers.add_parser("create", help="Create a new component")
    create_parser.add_argument("type", choices=["agent", "step", "workflow", "executor", "link", "dataunit"],
                              help="Type of component to create")
    create_parser.add_argument("name", help="Name of the component")
    create_parser.set_defaults(func=create_component)
    
    # Run workflow command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("file", help="Workflow file to run")
    run_parser.set_defaults(func=run_workflow)
    
    # Config management command
    config_parser = subparsers.add_parser("config", help="Manage configurations")
    config_parser.add_argument("action", choices=["show", "edit", "validate"],
                              help="Action to perform on the configuration")
    config_parser.add_argument("file", help="Configuration file to manage")
    config_parser.set_defaults(func=manage_config)
    
    # Help command
    help_parser = subparsers.add_parser("help", help="Show help information")
    help_parser.set_defaults(func=show_help)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate function
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 