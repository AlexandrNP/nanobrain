from typing import List, Any, Optional
import os
import subprocess
from pathlib import Path

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepGitInit(Step):
    """
    Tool for initializing a git repository.
    
    Biological analogy: Motor cortex for tool manipulation.
    Justification: Like how the motor cortex controls precise tool use,
    this step controls the precise initialization of git repositories.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.default_branch = kwargs.get('default_branch', 'main')
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Initialize a git repository.
        
        Args:
            inputs: List containing:
                - repo_path: Path where the repository should be initialized
                - repo_name: Name of the repository (optional)
                - username: Username for the repository (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        # Extract inputs
        if not inputs or len(inputs) < 1:
            return {"error": "Missing required input: repo_path"}
        
        repo_path = inputs[0]
        repo_name = inputs[1] if len(inputs) > 1 else None
        username = inputs[2] if len(inputs) > 2 else None
        
        # Create the repository path if it doesn't exist
        try:
            os.makedirs(repo_path, exist_ok=True)
        except Exception as e:
            return {"error": f"Failed to create directory: {e}"}
        
        # Initialize the git repository
        try:
            # Change to the repository directory
            original_dir = os.getcwd()
            os.chdir(repo_path)
            
            # Initialize the repository
            result = subprocess.run(
                ["git", "init", "-b", self.default_branch],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Configure the repository if username is provided
            if username:
                subprocess.run(
                    ["git", "config", "user.name", username],
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Create a .gitignore file
            with open(".gitignore", "w") as f:
                f.write("# Python\n")
                f.write("__pycache__/\n")
                f.write("*.py[cod]\n")
                f.write("*$py.class\n")
                f.write("*.so\n")
                f.write(".Python\n")
                f.write("env/\n")
                f.write("build/\n")
                f.write("develop-eggs/\n")
                f.write("dist/\n")
                f.write("downloads/\n")
                f.write("eggs/\n")
                f.write(".eggs/\n")
                f.write("lib/\n")
                f.write("lib64/\n")
                f.write("parts/\n")
                f.write("sdist/\n")
                f.write("var/\n")
                f.write("*.egg-info/\n")
                f.write(".installed.cfg\n")
                f.write("*.egg\n")
                f.write("\n")
                f.write("# Virtual environments\n")
                f.write("venv/\n")
                f.write("ENV/\n")
                f.write("\n")
                f.write("# IDE files\n")
                f.write(".idea/\n")
                f.write(".vscode/\n")
                f.write("*.swp\n")
                f.write("*.swo\n")
                f.write("\n")
                f.write("# Session files\n")
                f.write("*.pkl\n")
            
            # Create a README.md file if repo_name is provided
            if repo_name:
                with open("README.md", "w") as f:
                    f.write(f"# {repo_name}\n\n")
                    f.write("A NanoBrain workflow created with the NanoBrain builder tool.\n")
            
            # Change back to the original directory
            os.chdir(original_dir)
            
            return {
                "success": True,
                "message": f"Git repository initialized at {repo_path}",
                "output": result.stdout
            }
        
        except subprocess.CalledProcessError as e:
            # Change back to the original directory
            os.chdir(original_dir)
            
            return {
                "success": False,
                "error": f"Git command failed: {e.stderr}"
            }
        
        except Exception as e:
            # Change back to the original directory
            os.chdir(original_dir)
            
            return {
                "success": False,
                "error": f"Failed to initialize git repository: {e}"
            } 