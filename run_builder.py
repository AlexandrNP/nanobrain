#!/usr/bin/env python3
"""
Run NanoBrainBuilder

This script runs the NanoBrainBuilder directly, bypassing the command-line interface.
"""

import os
import sys
import asyncio
import argparse
import json
import pickle

# Set up the Python import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the script directory to path to find setup_paths module
sys.path.insert(0, script_dir)

# Define the project root
PROJECT_ROOT = script_dir

try:
    import setup_paths
    print("Successfully imported setup_paths")
except ImportError as e:
    print(f"Warning: Could not import setup_paths module: {e}")
    # Fall back to manually setting up paths
    sys.path.insert(0, PROJECT_ROOT)
    os.environ["PYTHONPATH"] = PROJECT_ROOT

# Set the NANOBRAIN_TESTING environment variable
os.environ["NANOBRAIN_TESTING"] = "1"

# Import the builder
from builder.NanoBrainBuilder import NanoBrainBuilder
from builder.AgentWorkflowBuilder import AgentWorkflowBuilder
from src.ExecutorBase import ExecutorBase
from src.Workflow import Workflow
from src.Step import Step

async def create_workflow(name):
    """Create a workflow with the given name."""
    # Create an executor
    executor = ExecutorBase()
    
    # Create a workflow directly
    workflow = Workflow(name=name, executor=executor)
    print(f"Created workflow '{name}'")
    
    # Save the workflow
    save_path = os.path.join(PROJECT_ROOT, "workflows", f"{name}.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(workflow, f)
    print(f"Saved workflow to {save_path}")
    
    return workflow

async def create_step(name, workflow_name=None):
    """Create a step with the given name."""
    # Create an executor
    executor = ExecutorBase()
    
    # Create a step
    step = Step(name=name, executor=executor)
    print(f"Created step '{name}'")
    
    # Save the step
    save_path = os.path.join(PROJECT_ROOT, "steps", f"{name}.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(step, f)
    print(f"Saved step to {save_path}")
    
    # Add the step to the workflow if specified
    if workflow_name:
        # Load the workflow
        workflow_path = os.path.join(PROJECT_ROOT, "workflows", f"{workflow_name}.pkl")
        if os.path.exists(workflow_path):
            with open(workflow_path, "rb") as f:
                workflow = pickle.load(f)
            
            # Add the step to the workflow's steps list
            workflow.steps.append(step)
            print(f"Added step '{name}' to workflow '{workflow_name}'")
            
            # Save the workflow
            with open(workflow_path, "wb") as f:
                pickle.dump(workflow, f)
            print(f"Saved updated workflow to {workflow_path}")
        else:
            print(f"Workflow '{workflow_name}' not found")
    
    return step

async def list_workflows():
    """List all workflows."""
    workflows_dir = os.path.join(PROJECT_ROOT, "workflows")
    if os.path.exists(workflows_dir):
        workflows = [f[:-4] for f in os.listdir(workflows_dir) if f.endswith(".pkl")]
        if workflows:
            print("Available workflows:")
            for workflow in workflows:
                print(f"  - {workflow}")
        else:
            print("No workflows found")
    else:
        print("No workflows directory found")

async def list_steps():
    """List all steps."""
    steps_dir = os.path.join(PROJECT_ROOT, "steps")
    if os.path.exists(steps_dir):
        steps = [f[:-4] for f in os.listdir(steps_dir) if f.endswith(".pkl")]
        if steps:
            print("Available steps:")
            for step in steps:
                print(f"  - {step}")
        else:
            print("No steps found")
    else:
        print("No steps directory found")

async def main():
    """Run the NanoBrainBuilder."""
    parser = argparse.ArgumentParser(description="Run NanoBrainBuilder")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create workflow command
    create_workflow_parser = subparsers.add_parser("create-workflow", help="Create a new workflow")
    create_workflow_parser.add_argument("name", help="Name of the workflow")
    
    # Create step command
    create_step_parser = subparsers.add_parser("create-step", help="Create a new step")
    create_step_parser.add_argument("name", help="Name of the step")
    create_step_parser.add_argument("--workflow", help="Name of the workflow to add the step to")
    
    # List workflows command
    list_workflows_parser = subparsers.add_parser("list-workflows", help="List all workflows")
    
    # List steps command
    list_steps_parser = subparsers.add_parser("list-steps", help="List all steps")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "create-workflow":
        await create_workflow(args.name)
    elif args.command == "create-step":
        await create_step(args.name, args.workflow)
    elif args.command == "list-workflows":
        await list_workflows()
    elif args.command == "list-steps":
        await list_steps()
    else:
        # Create a new builder
        builder = NanoBrainBuilder()
        
        # Start the builder
        await builder.main()

if __name__ == "__main__":
    asyncio.run(main()) 