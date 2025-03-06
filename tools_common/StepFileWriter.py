from typing import List, Any, Optional, Dict, Union
import os
from pathlib import Path

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepFileWriter(Step):
    """
    Tool for creating or modifying files with provided content.
    
    Biological analogy: Fine motor control for writing.
    Justification: Like how fine motor control allows for precise writing
    movements, this tool allows for precise file writing operations.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.backup_enabled = kwargs.get('backup_enabled', True)
        self.backup_extension = kwargs.get('backup_extension', '.bak')
        self.create_directories = kwargs.get('create_directories', True)
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Create or modify a file with the provided content.
        
        Args:
            inputs: List containing:
                - file_path: Path to the file to create or modify
                - content: Content to write to the file
                - mode: Write mode ('w' for write, 'a' for append)
                - make_backup: Whether to create a backup of the file (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        # Extract inputs
        if not inputs or len(inputs) < 2:
            return {
                "success": False,
                "error": "Missing required inputs: file_path and content are required"
            }
        
        file_path = inputs[0]
        content = inputs[1]
        mode = inputs[2] if len(inputs) > 2 else 'w'
        make_backup = inputs[3] if len(inputs) > 3 else self.backup_enabled
        
        # Validate mode
        if mode not in ['w', 'a']:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Must be 'w' (write) or 'a' (append)"
            }
        
        try:
            # Create directories if necessary
            if self.create_directories:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Create a backup if the file exists and backups are enabled
            if os.path.exists(file_path) and make_backup:
                backup_path = f"{file_path}{self.backup_extension}"
                try:
                    with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                        dst.write(src.read())
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to create backup: {e}"
                    }
            
            # Write or append to the file
            with open(file_path, mode) as f:
                f.write(content)
            
            # Return success result
            return {
                "success": True,
                "message": f"File {file_path} {'created' if mode == 'w' else 'appended'} successfully",
                "file_path": file_path,
                "mode": mode
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write to file: {e}"
            }
    
    async def create_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Create a new file or overwrite an existing one.
        
        Args:
            file_path: Path to the file to create
            content: Content to write to the file
        
        Returns:
            Dictionary with the result of the operation
        """
        return await self.process([file_path, content, 'w'])
    
    async def append_to_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Append content to an existing file.
        
        Args:
            file_path: Path to the file to append to
            content: Content to append to the file
        
        Returns:
            Dictionary with the result of the operation
        """
        return await self.process([file_path, content, 'a'])
    
    async def create_python_module(self, module_path: str, class_name: str, base_class: str, content: str) -> Dict[str, Any]:
        """
        Create a new Python module with proper imports and class definition.
        
        Args:
            module_path: Path to the module to create
            class_name: Name of the class to define
            base_class: Name of the base class to inherit from
            content: Content of the module (class methods, etc.)
        
        Returns:
            Dictionary with the result of the operation
        """
        # Create the module content
        module_content = f"""from typing import List, Dict, Any, Optional, Union
import os
from pathlib import Path

from src.{base_class} import {base_class}


class {class_name}({base_class}):
{content}
"""
        
        # Create the module
        return await self.create_file(module_path, module_content)
    
    async def create_yaml_config(self, config_path: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new YAML configuration file.
        
        Args:
            config_path: Path to the configuration file to create
            config_data: Configuration data to write
        
        Returns:
            Dictionary with the result of the operation
        """
        try:
            import yaml
            
            # Convert the configuration data to YAML
            yaml_content = yaml.dump(config_data, default_flow_style=False)
            
            # Create the configuration file
            return await self.create_file(config_path, yaml_content)
            
        except ImportError:
            return {
                "success": False,
                "error": "Failed to import yaml module"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create YAML configuration: {e}"
            } 