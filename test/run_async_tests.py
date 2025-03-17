#!/usr/bin/env python3
"""
Script to run all async-related unit tests.

This ensures that the ExecutorFunc.execute_async method and the _safe_execute methods
in both AgentWorkflowBuilder and AgentCodeWriter work correctly.
"""

import sys
import os
import pytest

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == '__main__':
    # Specify test files to run
    test_files = [
        "test/test_executor_func.py",
        "test/test_agent_workflow_builder_async.py",
        "test/test_agent_code_writer_async.py"
    ]
    
    # Run tests
    exit_code = pytest.main(["-v", "--asyncio-mode=strict"] + test_files)
    
    # Exit with error if any tests failed
    sys.exit(exit_code) 