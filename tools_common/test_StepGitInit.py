import unittest
import os
import shutil
import tempfile
import asyncio
from unittest.mock import MagicMock, patch

from src.ExecutorBase import ExecutorBase
from tools_common.StepGitInit import StepGitInit


class TestStepGitInit(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock executor
        self.executor = MagicMock(spec=ExecutorBase)
        
        # Create the StepGitInit instance
        self.git_init_tool = StepGitInit(executor=self.executor)
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch('subprocess.run')
    def test_process_success(self, mock_run):
        # Configure the mock to return a successful result
        mock_run.return_value.stdout = "Initialized empty Git repository"
        
        # Call the process method
        result = asyncio.run(self.git_init_tool.process([self.test_dir, "test-repo", "test-user"]))
        
        # Check that the result is successful
        self.assertTrue(result["success"])
        self.assertIn("Git repository initialized", result["message"])
        
        # Check that the subprocess.run was called with the correct arguments
        mock_run.assert_any_call(
            ["git", "init", "-b", "main"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check that the user was configured
        mock_run.assert_any_call(
            ["git", "config", "user.name", "test-user"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check that the .gitignore file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, ".gitignore")))
        
        # Check that the README.md file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "README.md")))
    
    @patch('subprocess.run')
    def test_process_failure(self, mock_run):
        # Configure the mock to raise an exception
        mock_run.side_effect = Exception("Git command failed")
        
        # Call the process method
        result = asyncio.run(self.git_init_tool.process([self.test_dir]))
        
        # Check that the result is not successful
        self.assertFalse(result["success"])
        self.assertIn("Failed to initialize git repository", result["error"])
    
    def test_process_missing_inputs(self):
        # Call the process method with no inputs
        result = asyncio.run(self.git_init_tool.process([]))
        
        # Check that the result contains an error message
        self.assertIn("error", result)
        self.assertIn("Missing required input", result["error"])


if __name__ == '__main__':
    unittest.main() 