#!/usr/bin/env python3
"""
Test runner for NanoBrainBuilder tests.

This script sets up appropriate mocks and runs the builder tests.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the mock classes
from test.mock_tools import (
    StepFileWriter, 
    StepPlanner, 
    StepCoder, 
    StepGitInit, 
    StepContextSearch, 
    StepWebSearch
)
from test.mock_agent import Agent
from test.mock_executor import MockExecutorBase, MockConfigManager, MockSystemModulator, MockDirectoryTracer
from test.mock_builder import (
    NanoBrainBuilder,
    CreateWorkflowStep,
    CreateStepStep,
    TestStepStep,
    SaveStepStep,
    LinkStepsStep,
    SaveWorkflowStep
)

def setup_mocks():
    """Set up mocks for the tests."""
    # Mock the tools_common module
    tools_common_mock = MagicMock()
    tools_common_mock.StepFileWriter = StepFileWriter
    tools_common_mock.StepPlanner = StepPlanner
    tools_common_mock.StepCoder = StepCoder
    tools_common_mock.StepGitInit = StepGitInit
    tools_common_mock.StepContextSearch = StepContextSearch
    tools_common_mock.StepWebSearch = StepWebSearch
    sys.modules['tools_common'] = tools_common_mock
    
    # Mock the src.Agent module
    agent_mock = MagicMock()
    agent_mock.Agent = Agent
    sys.modules['src.Agent'] = agent_mock
    
    # Mock the src.ExecutorBase module
    executor_mock = MagicMock()
    executor_mock.ExecutorBase = MockExecutorBase
    sys.modules['src.ExecutorBase'] = executor_mock
    
    # Mock the src.ConfigManager module
    config_manager_mock = MagicMock()
    config_manager_mock.ConfigManager = MockConfigManager
    sys.modules['src.ConfigManager'] = config_manager_mock
    
    # Mock the src.regulations module
    regulations_mock = MagicMock()
    regulations_mock.SystemModulator = MockSystemModulator
    sys.modules['src.regulations'] = regulations_mock
    
    # Mock the src.DirectoryTracer module
    directory_tracer_mock = MagicMock()
    directory_tracer_mock.DirectoryTracer = MockDirectoryTracer
    sys.modules['src.DirectoryTracer'] = directory_tracer_mock
    
    # Mock the builder module
    builder_mock = MagicMock()
    builder_mock.NanoBrainBuilder = NanoBrainBuilder
    sys.modules['builder'] = builder_mock
    
    # Mock the builder.WorkflowSteps module
    workflow_steps_mock = MagicMock()
    workflow_steps_mock.CreateWorkflowStep = CreateWorkflowStep
    workflow_steps_mock.CreateStepStep = CreateStepStep
    workflow_steps_mock.TestStepStep = TestStepStep
    workflow_steps_mock.SaveStepStep = SaveStepStep
    workflow_steps_mock.LinkStepsStep = LinkStepsStep
    workflow_steps_mock.SaveWorkflowStep = SaveWorkflowStep
    sys.modules['builder.WorkflowSteps'] = workflow_steps_mock
    
    # Mock subprocess for testing workflow saving
    subprocess_mock = MagicMock()
    subprocess_mock.run.return_value.returncode = 0
    sys.modules['subprocess'] = subprocess_mock


def run_tests():
    """Run the builder tests."""
    # Set up mocks
    setup_mocks()
    
    # Import and run the simple tests first
    from test.test_builder_simple import TestNanoBrainBuilder as SimpleTests
    simple_suite = unittest.TestLoader().loadTestsFromTestCase(SimpleTests)
    
    # Set up other test suites
    actual_tests_suite = unittest.TestSuite()
    workflow_execution_suite = unittest.TestSuite()
    
    # Try to import and add the actual tests if available
    try:
        from test.test_builder_actual import TestNanoBrainBuilderActual
        actual_tests_suite = unittest.TestLoader().loadTestsFromTestCase(TestNanoBrainBuilderActual)
    except ImportError as e:
        print(f"Warning: Could not import actual builder tests: {e}")
    
    # Try to import and add the workflow execution tests if available
    try:
        from test.test_workflow_execution import TestWorkflowExecution
        workflow_execution_suite = unittest.TestLoader().loadTestsFromTestCase(TestWorkflowExecution)
    except ImportError as e:
        print(f"Warning: Could not import workflow execution tests: {e}")
    
    # Combine all test suites
    all_tests = unittest.TestSuite([
        simple_suite,
        actual_tests_suite,
        workflow_execution_suite
    ])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    run_tests() 