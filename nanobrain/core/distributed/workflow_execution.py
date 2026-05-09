"""
Parsl applications for distributed workflow execution.

This module contains Parsl apps that enable NanoBrain workflows
to execute on distributed workers with proper configuration loading.
"""

from parsl.app.app import python_app
from typing import Dict, Any
import os
import asyncio


@python_app
def execute_workflow_distributed(
    config_path: str,
    input_data: Dict[str, Any],
    main_working_directory: str
) -> Dict[str, Any]:
    """
    Execute NanoBrain workflow on distributed Parsl worker.
    
    This function runs on remote Parsl workers and handles the working
    directory adjustment necessary for proper configuration file loading.
    
    Args:
        config_path: Relative path to workflow configuration file
        input_data: Input data dictionary for workflow execution
        main_working_directory: Main process working directory for config resolution
        
    Returns:
        Dictionary containing execution results:
        - success: Boolean indicating execution success
        - result: Workflow execution result (if successful)
        - error: Error message (if failed)
        - worker_info: Metadata about worker execution
    """
    import socket
    import traceback
    
    # Store original worker directory
    original_working_directory = os.getcwd()
    
    try:
        # Change to main process working directory for configuration loading
        os.chdir(main_working_directory)

        # Import NanoBrain components (must happen after directory change)
        from nanobrain.core.workflow import Workflow

        # Load workflow configuration using existing NanoBrain patterns
        # Pass the main working directory as workflow_directory context
        # This ensures step configs are resolved relative to the main directory
        workflow = Workflow.from_config(config_path, workflow_directory=main_working_directory)

        # Restore worker directory for execution
        os.chdir(original_working_directory)
        
        # Execute workflow using standard NanoBrain interface
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            execution_result = loop.run_until_complete(workflow.process(input_data))
            
            return {
                'success': True,
                'result': execution_result,
                'worker_info': {
                    'hostname': socket.gethostname(),
                    'config_path': config_path,
                    'main_directory': main_working_directory,
                    'worker_directory': original_working_directory
                }
            }
            
        finally:
            loop.close()
            
    except Exception as execution_error:
        return {
            'success': False,
            'error': str(execution_error),
            'error_type': type(execution_error).__name__,
            'traceback': traceback.format_exc(),
            'worker_info': {
                'hostname': socket.gethostname(),
                'config_path': config_path,
                'main_directory': main_working_directory,
                'worker_directory': original_working_directory
            }
        }
        
    finally:
        # Always restore original working directory
        try:
            os.chdir(original_working_directory)
        except OSError as restore_error:
            # Log restoration failure but don't mask original error
            print(f"Warning: Failed to restore working directory {original_working_directory}: {restore_error}")
