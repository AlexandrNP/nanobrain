#!/usr/bin/env python3
"""
NanoBrain Workflow Builder

This script provides an interactive interface for building NanoBrain workflows.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
import traceback

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

# Check if we're in testing mode
TESTING_MODE = os.environ.get("NANOBRAIN_TESTING", "0") == "1"
if TESTING_MODE:
    print("Running in testing mode with mock models.")

# Import the GlobalConfig
try:
    from src.GlobalConfig import GlobalConfig
    print("Successfully imported setup_paths")
    # Initialize the global configuration
    config = GlobalConfig()
    print(f"Global configuration loaded from: {config.config_path}")
except ImportError as e:
    print(f"Error importing GlobalConfig: {e}")
    config = None

# Terminal colors and formatting
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Prompt templates
PROMPTS = {
    'welcome': f"{TermColors.HEADER}🧠 Welcome to the NanoBrain Interactive Builder! 🧠{TermColors.ENDC}\n\nThis tool will guide you through the process of creating workflows and steps.",
    'menu': f"\n{TermColors.BOLD}What would you like to do?{TermColors.ENDC}",
    'menu_options': [
        "Create a new workflow",
        "Create a new step",
        "Test a step",
        "Save a step",
        "Link steps together",
        "Save a workflow",
        "List available workflows",
        "List available steps",
        "Exit"
    ],
    'choice': f"\nEnter your choice (1-9): ",
    'workflow_name': f"Enter the name for your new workflow: ",
    'username': f"Enter your username (optional, press Enter to skip): ",
    'step_name': f"Enter the name for your new step: ",
    'add_to_workflow': f"Would you like to add this step to a workflow? (y/n): ",
    'workflow_choice': f"Enter the name of the workflow: ",
    'step_choice': f"Enter the name of the step to test: ",
    'source_step': f"Enter the name of the source step: ",
    'target_step': f"Enter the name of the target step: ",
    'link_type': "Enter the link type (1-{}, default is 1): ",
    'continue': f"\n{TermColors.BOLD}Press Enter to continue...{TermColors.ENDC}",
    'goodbye': f"\n{TermColors.HEADER}Thank you for using the NanoBrain Interactive Builder! Goodbye! 👋{TermColors.ENDC}\n",
    'success': f"{TermColors.GREEN}✅ {{0}}{TermColors.ENDC}",
    'error': f"{TermColors.FAIL}❌ Error: {{0}}{TermColors.ENDC}",
    'warning': f"{TermColors.WARNING}⚠️ Warning: {{0}}{TermColors.ENDC}",
    'info': f"{TermColors.CYAN}ℹ️ {{0}}{TermColors.ENDC}",
}

# Helper function to ensure prompt templates are converted to strings
def ensure_string(prompt_or_template, **kwargs):
    """
    Ensure that a prompt or template is converted to a string.
    If it's a PromptTemplate, format it with the provided kwargs.
    If it's already a string, return it as is.
    """
    # Check if it's a PromptTemplate-like object with a format method
    if hasattr(prompt_or_template, 'format') and callable(prompt_or_template.format):
        try:
            # Check if it's a PromptTemplate with a template attribute
            if hasattr(prompt_or_template, 'template'):
                # If kwargs are provided, format the template
                if kwargs:
                    return prompt_or_template.format(**kwargs)
                # Otherwise, return the template string
                return prompt_or_template.template
                
            # If it's a string with a format method
            # Check if the format string uses positional arguments (e.g., {0})
            if isinstance(prompt_or_template, str) and '{0}' in prompt_or_template:
                # If we have a 'message' parameter, use it as the positional argument
                if 'message' in kwargs:
                    return prompt_or_template.format(kwargs['message'])
                # If we have an 'error' parameter, use it as the positional argument
                elif 'error' in kwargs:
                    return prompt_or_template.format(kwargs['error'])
                # If we have a 'context' parameter, use it as the positional argument
                elif 'context' in kwargs:
                    return prompt_or_template.format(kwargs['context'])
                # If no suitable parameter is found, return the template as is
                return prompt_or_template
            
            # If kwargs are provided, format the string
            if kwargs:
                return prompt_or_template.format(**kwargs)
                
            # Otherwise, convert to string
            return str(prompt_or_template)
        except Exception as e:
            print(f"Warning: Error formatting prompt template: {e}")
            return str(prompt_or_template)
    # If it's already a string or another type, convert to string
    return str(prompt_or_template)

# Error handling
def handle_error(error, context=""):
    """Handle errors gracefully with helpful messages."""
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        print(ensure_string(PROMPTS['error'], message=f"{context}: {error_type} - {error_msg}"))
    else:
        print(ensure_string(PROMPTS['error'], message=f"{error_type} - {error_msg}"))
    
    # Check if we're in debug mode
    debug_mode = os.environ.get("NANOBRAIN_DEBUG", "0") == "1"
    
    # User input errors are typically ValueError with specific messages
    is_user_input_error = (
        error_type == "ValueError" and 
        any(msg in error_msg.lower() for msg in [
            "empty", "cannot be empty", "required", "invalid choice", 
            "not found", "does not exist"
        ])
    )
    
    # In debug mode, always print stack trace
    if debug_mode:
        print("\n" + ensure_string(PROMPTS['info'], message="Stack trace (debug mode):"))
        # If there's no traceback (e.g., for manually created errors), create one
        if not hasattr(error, "__traceback__") or error.__traceback__ is None:
            try:
                # Raise the error to generate a traceback
                raise error
            except Exception as e:
                traceback.print_exc()
        else:
            traceback.print_exc()
    # For non-user input errors, print stack trace even in non-debug mode
    elif not is_user_input_error:
        print("\n" + ensure_string(PROMPTS['info'], message="Stack trace for debugging:"))
        # If there's no traceback (e.g., for manually created errors), create one
        if not hasattr(error, "__traceback__") or error.__traceback__ is None:
            try:
                # Raise the error to generate a traceback
                raise error
            except Exception as e:
                traceback.print_exc()
        else:
            traceback.print_exc()
    
    return {"success": False, "error": error_msg, "error_type": error_type}

async def create_workflow(name, username=None):
    """Create a new workflow."""
    try:
        # Simulate an error for testing purposes
        if name == "error_test":
            raise ValueError("This is a simulated error for testing purposes")
            
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        result = builder.create_workflow(name)
        return result
    except Exception as e:
        return handle_error(e, "Failed to create workflow")

async def create_step(name, workflow=None):
    """Create a new step."""
    try:
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        
        # If a workflow is specified, check if it exists and set it as current
        if workflow:
            workflow_path = builder.get_workflow_path(workflow)
            if not workflow_path:
                return {"success": False, "error": f"Workflow '{workflow}' not found"}
            builder.push_workflow(workflow_path)
        
        # Ensure description is a string
        description = "A custom step"
        
        result = await builder.create_step(name, "Step", description)
        return result
    except Exception as e:
        return handle_error(e, "Failed to create step")

async def test_step(name):
    """Test a step."""
    try:
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        result = await builder.test_step(name)
        return result
    except Exception as e:
        return handle_error(e, f"Failed to test step '{name}'")

async def save_step(name):
    """Save a step."""
    try:
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        result = await builder.save_step(name)
        return result
    except Exception as e:
        return handle_error(e, f"Failed to save step '{name}'")

async def link_steps(source, target, link_type="LinkDirect"):
    """Link two steps."""
    try:
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        result = await builder.link_steps(source, target, link_type)
        return result
    except Exception as e:
        return handle_error(e, f"Failed to link steps '{source}' and '{target}'")

async def save_workflow():
    """Save a workflow."""
    try:
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        result = await builder.save_workflow()
        return result
    except Exception as e:
        return handle_error(e, "Failed to save workflow")

async def list_workflows():
    """List all available workflows."""
    try:
        from builder.NanoBrainBuilder import NanoBrainBuilder
        builder = NanoBrainBuilder()
        workflows = builder.list_workflows()
        
        if not workflows:
            print(ensure_string(PROMPTS['warning'], message="No workflows found"))
            return {"success": True, "workflows": []}
        
        print("\nAvailable workflows:")
        for i, workflow in enumerate(workflows, 1):
            print(f"{i}. {workflow}")
        
        return {"success": True, "workflows": workflows}
    except Exception as e:
        return handle_error(e, "Failed to list workflows")

async def list_steps():
    """List all steps."""
    try:
        steps_dir = os.path.join(os.getcwd(), "steps")
        if os.path.exists(steps_dir):
            steps = [f[:-4] for f in os.listdir(steps_dir) if f.endswith(".pkl")]
            if steps:
                print(ensure_string(PROMPTS['info'], message="Available steps:"))
                for i, step in enumerate(steps, 1):
                    print(f"{i}. {step}")
            else:
                print(ensure_string(PROMPTS['warning'], message="No steps found"))
        else:
            print(ensure_string(PROMPTS['warning'], message="No steps directory found"))
            # Create the directory
            os.makedirs(steps_dir, exist_ok=True)
            print(ensure_string(PROMPTS['info'], message=f"Created steps directory at {steps_dir}"))
        return {"success": True, "message": "Steps listed successfully"}
    except Exception as e:
        return handle_error(e, "Failed to list steps")

async def interactive_builder():
    """Run the interactive builder."""
    print(PROMPTS['welcome'])
    
    # Set debug mode for interactive mode to always show stack traces
    os.environ["NANOBRAIN_DEBUG"] = "1"
    
    while True:
        print(PROMPTS['menu'])
        for i, option in enumerate(PROMPTS['menu_options'], 1):
            print(f"{i}. {option}")
        
        try:
            choice = input(PROMPTS['choice'])
            
            if choice == "1":  # Create a new workflow
                try:
                    name = input(PROMPTS['workflow_name'])
                    if not name:
                        handle_error(ValueError("Workflow name cannot be empty"))
                        continue
                    
                    username = input(PROMPTS['username']) or None
                    result = await create_workflow(name, username)
                    
                    if result.get("success", False):
                        print(ensure_string(PROMPTS['success'], message=result.get('message', 'Workflow created successfully')))
                    else:
                        print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
                except EOFError:
                    print("\n" + ensure_string(PROMPTS['warning'], message="End of input reached"))
                    print(PROMPTS['goodbye'])
                    return
            
            elif choice == "2":  # Create a new step
                name = input(PROMPTS['step_name'])
                if not name:
                    handle_error(ValueError("Step name cannot be empty"))
                    continue
                
                add_to_workflow = input(PROMPTS['add_to_workflow']).lower() == 'y'
                
                if add_to_workflow:
                    # List available workflows
                    await list_workflows()
                    workflow = input(PROMPTS['workflow_choice'])
                    if not workflow:
                        handle_error(ValueError("Workflow name cannot be empty"))
                        continue
                    result = await create_step(name, workflow)
                else:
                    result = await create_step(name)
                    
                if result.get("success", False):
                    print(ensure_string(PROMPTS['success'], message=result.get('message', 'Step created successfully')))
                else:
                    print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
            
            elif choice == "3":  # Test a step
                # List available steps
                await list_steps()
                name = input(PROMPTS['step_choice'])
                if not name:
                    handle_error(ValueError("Step name cannot be empty"))
                    continue
                
                result = await test_step(name)
                if result.get("success", False):
                    print(ensure_string(PROMPTS['success'], message=result.get('message', 'Step tested successfully')))
                else:
                    print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
            
            elif choice == "4":  # Save a step
                # List available steps
                await list_steps()
                name = input(PROMPTS['step_choice'])
                if not name:
                    handle_error(ValueError("Step name cannot be empty"))
                    continue
                
                result = await save_step(name)
                if result.get("success", False):
                    print(ensure_string(PROMPTS['success'], message=result.get('message', 'Step saved successfully')))
                else:
                    print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
            
            elif choice == "5":  # Link steps together
                # List available steps
                await list_steps()
                source = input(PROMPTS['source_step'])
                if not source:
                    handle_error(ValueError("Source step name cannot be empty"))
                    continue
                
                target = input(PROMPTS['target_step'])
                if not target:
                    handle_error(ValueError("Target step name cannot be empty"))
                    continue
                
                link_types = ["LinkDirect", "LinkFile", "LinkCustom"]
                print(ensure_string(PROMPTS['info'], message="Available link types:"))
                for i, link_type in enumerate(link_types, 1):
                    print(f"{i}. {link_type}")
                
                link_choice = input(ensure_string(PROMPTS['link_type'], len(link_types)))
                try:
                    link_type = link_types[int(link_choice) - 1] if link_choice else "LinkDirect"
                except (ValueError, IndexError):
                    link_type = "LinkDirect"
                    print(ensure_string(PROMPTS['warning'], message=f"Invalid choice, using default link type: {link_type}"))
                    
                result = await link_steps(source, target, link_type)
                if result.get("success", False):
                    print(ensure_string(PROMPTS['success'], message=result.get('message', 'Steps linked successfully')))
                else:
                    print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
            
            elif choice == "6":  # Save a workflow
                result = await save_workflow()
                if result.get("success", False):
                    print(ensure_string(PROMPTS['success'], message=result.get('message', 'Workflow saved successfully')))
                else:
                    print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
            
            elif choice == "7":  # List available workflows
                await list_workflows()
            
            elif choice == "8":  # List available steps
                await list_steps()
            
            elif choice == "9":  # Exit
                print(PROMPTS['goodbye'])
                break
            
            else:
                handle_error(ValueError("Invalid choice. Please enter a number between 1 and 9."))
            
        except KeyboardInterrupt:
            print("\n" + ensure_string(PROMPTS['warning'], message="Operation cancelled by user"))
            print(PROMPTS['goodbye'])
            break
        except EOFError:
            print("\n" + ensure_string(PROMPTS['warning'], message="End of input reached"))
            print(PROMPTS['goodbye'])
            break
        except Exception as e:
            handle_error(e, "Unexpected error")
        
        # Pause before showing the menu again
        try:
            input(PROMPTS['continue'])
        except (KeyboardInterrupt, EOFError):
            print("\n" + ensure_string(PROMPTS['warning'], message="Operation cancelled by user"))
            print(PROMPTS['goodbye'])
            break

async def main():
    """Run the NanoBrainBuilder."""
    parser = argparse.ArgumentParser(description="Run NanoBrainBuilder")
    parser.add_argument("--command", help="Command to execute (optional, defaults to interactive mode)")
    parser.add_argument("--name", help="Name for workflow or step")
    parser.add_argument("--workflow", help="Name of the workflow")
    parser.add_argument("--source", help="Source step for linking")
    parser.add_argument("--target", help="Target step for linking")
    parser.add_argument("--link-type", default="LinkDirect", help="Type of link to create")
    parser.add_argument("--username", help="Username for workflow creation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        os.environ["NANOBRAIN_DEBUG"] = "1"
    
    # If a command is specified, run it
    if args.command:
        try:
            if args.command == "create-workflow":
                if not args.name:
                    error = ValueError("Name is required for create-workflow command")
                    handle_error(error)
                    return
                result = await create_workflow(args.name, args.username)
            elif args.command == "create-step":
                if not args.name:
                    error = ValueError("Name is required for create-step command")
                    handle_error(error)
                    return
                result = await create_step(args.name, args.workflow)
            elif args.command == "test-step":
                if not args.name:
                    error = ValueError("Name is required for test-step command")
                    handle_error(error)
                    return
                result = await test_step(args.name)
            elif args.command == "save-step":
                if not args.name:
                    error = ValueError("Name is required for save-step command")
                    handle_error(error)
                    return
                result = await save_step(args.name)
            elif args.command == "link-steps":
                if not args.source or not args.target:
                    error = ValueError("Source and target are required for link-steps command")
                    handle_error(error)
                    return
                result = await link_steps(args.source, args.target, args.link_type)
            elif args.command == "save-workflow":
                result = await save_workflow()
            elif args.command == "list-workflows":
                result = await list_workflows()
            elif args.command == "list-steps":
                result = await list_steps()
            else:
                error = ValueError(f"Unknown command: {args.command}")
                handle_error(error)
                parser.print_help()
                return
                
            # Print the result
            if result.get("success", False):
                print(ensure_string(PROMPTS['success'], message=result.get('message', 'Command executed successfully')))
            else:
                print(ensure_string(PROMPTS['error'], message=result.get('error', 'Unknown error')))
        except Exception as e:
            handle_error(e, f"Error executing command '{args.command}'")
    else:
        # Run in interactive mode
        await interactive_builder()

if __name__ == "__main__":
    asyncio.run(main()) 