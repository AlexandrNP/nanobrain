from typing import List, Dict, Any, Optional, Union
import os
import re

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepGitExclude(Step):
    """
    Tool for managing .gitignore files and exclusion patterns.
    
    Biological analogy: Immune system memory.
    Justification: Like how the immune system remembers and filters out
    harmful elements, this tool helps filter out files that should not
    be tracked in version control.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.gitignore_path = kwargs.get('gitignore_path', '.gitignore')
        self.common_patterns = {
            'python': [
                '# Byte-compiled / optimized / DLL files',
                '__pycache__/',
                '*.py[cod]',
                '*$py.class',
                '*.so',
                '.Python',
                'build/',
                'develop-eggs/',
                'dist/',
                'downloads/',
                'eggs/',
                '.eggs/',
                'lib/',
                'lib64/',
                'parts/',
                'sdist/',
                'var/',
                'wheels/',
                '*.egg-info/',
                '.installed.cfg',
                '*.egg',
                'MANIFEST',
                '# Environments',
                '.env',
                '.venv',
                'env/',
                'venv/',
                'ENV/',
                'env.bak/',
                'venv.bak/',
                '# IDE',
                '.idea/',
                '.vscode/',
                '*.swp',
                '*.swo'
            ],
            'node': [
                '# Dependency directories',
                'node_modules/',
                'jspm_packages/',
                '# Logs',
                'logs',
                '*.log',
                'npm-debug.log*',
                'yarn-debug.log*',
                'yarn-error.log*',
                '# Build',
                'dist/',
                'build/',
                '.next/',
                'out/',
                '.nuxt/',
                '.cache/'
            ],
            'macos': [
                '.DS_Store',
                '.AppleDouble',
                '.LSOverride',
                'Icon',
                '._*',
                '.DocumentRevisions-V100',
                '.fseventsd',
                '.Spotlight-V100',
                '.TemporaryItems',
                '.Trashes',
                '.VolumeIcon.icns',
                '.com.apple.timemachine.donotpresent'
            ],
            'windows': [
                'Thumbs.db',
                'Thumbs.db:encryptable',
                'ehthumbs.db',
                'ehthumbs_vista.db',
                '*.stackdump',
                '[Dd]esktop.ini',
                '$RECYCLE.BIN/',
                '*.cab',
                '*.msi',
                '*.msix',
                '*.msm',
                '*.msp',
                '*.lnk'
            ]
        }
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Process gitignore patterns.
        
        Args:
            inputs: List containing:
                - action: Action to perform (add, remove, check, generate)
                - patterns: Patterns to add/remove/check
                - options: Optional dictionary with additional options
        
        Returns:
            Dictionary with the result of the operation
        """
        # Extract inputs
        if not inputs or len(inputs) < 1:
            return {
                "success": False,
                "error": "Missing required input: action is required"
            }
        
        action = inputs[0]
        patterns = inputs[1] if len(inputs) > 1 else []
        options = inputs[2] if len(inputs) > 2 else {}
        
        # Set options
        gitignore_path = options.get('gitignore_path', self.gitignore_path)
        
        # Process the action
        try:
            if action == 'add':
                return await self._add_patterns(patterns, gitignore_path)
            elif action == 'remove':
                return await self._remove_patterns(patterns, gitignore_path)
            elif action == 'check':
                return await self._check_patterns(patterns, gitignore_path)
            elif action == 'generate':
                return await self._generate_gitignore(patterns, gitignore_path)
            else:
                return {
                    "success": False,
                    "error": f"Invalid action: {action}. Must be one of: add, remove, check, generate"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing gitignore: {str(e)}"
            }
    
    async def _add_patterns(self, patterns: List[str], gitignore_path: str) -> Dict[str, Any]:
        """Add patterns to .gitignore."""
        # Read existing patterns
        existing_patterns = self._read_gitignore(gitignore_path)
        
        # Add new patterns
        added_patterns = []
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern and pattern not in existing_patterns:
                existing_patterns.append(pattern)
                added_patterns.append(pattern)
        
        # Write back to .gitignore
        self._write_gitignore(gitignore_path, existing_patterns)
        
        return {
            "success": True,
            "added_patterns": added_patterns,
            "gitignore_path": gitignore_path
        }
    
    async def _remove_patterns(self, patterns: List[str], gitignore_path: str) -> Dict[str, Any]:
        """Remove patterns from .gitignore."""
        # Read existing patterns
        existing_patterns = self._read_gitignore(gitignore_path)
        
        # Remove patterns
        removed_patterns = []
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern and pattern in existing_patterns:
                existing_patterns.remove(pattern)
                removed_patterns.append(pattern)
        
        # Write back to .gitignore
        self._write_gitignore(gitignore_path, existing_patterns)
        
        return {
            "success": True,
            "removed_patterns": removed_patterns,
            "gitignore_path": gitignore_path
        }
    
    async def _check_patterns(self, patterns: List[str], gitignore_path: str) -> Dict[str, Any]:
        """Check if patterns are in .gitignore."""
        # Read existing patterns
        existing_patterns = self._read_gitignore(gitignore_path)
        
        # Check patterns
        pattern_status = {}
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern:
                pattern_status[pattern] = pattern in existing_patterns
        
        return {
            "success": True,
            "pattern_status": pattern_status,
            "gitignore_path": gitignore_path
        }
    
    async def _generate_gitignore(self, templates: List[str], gitignore_path: str) -> Dict[str, Any]:
        """Generate a .gitignore file from templates."""
        # Get patterns for templates
        patterns = []
        for template in templates:
            template = template.lower().strip()
            if template in self.common_patterns:
                patterns.extend(self.common_patterns[template])
        
        # Add patterns to .gitignore
        result = await self._add_patterns(patterns, gitignore_path)
        
        return {
            "success": True,
            "templates": templates,
            "added_patterns": result.get("added_patterns", []),
            "gitignore_path": gitignore_path
        }
    
    def _read_gitignore(self, gitignore_path: str) -> List[str]:
        """Read patterns from .gitignore."""
        patterns = []
        
        # Check if .gitignore exists
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
        
        return patterns
    
    def _write_gitignore(self, gitignore_path: str, patterns: List[str]) -> None:
        """Write patterns to .gitignore."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(gitignore_path)), exist_ok=True)
        
        # Write patterns to .gitignore
        with open(gitignore_path, 'w') as f:
            for pattern in patterns:
                f.write(f"{pattern}\n")
    
    async def add_patterns(self, patterns: List[str], gitignore_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Add patterns to .gitignore.
        
        Args:
            patterns: Patterns to add
            gitignore_path: Path to .gitignore (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        return await self.process(['add', patterns, {'gitignore_path': gitignore_path or self.gitignore_path}])
    
    async def remove_patterns(self, patterns: List[str], gitignore_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Remove patterns from .gitignore.
        
        Args:
            patterns: Patterns to remove
            gitignore_path: Path to .gitignore (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        return await self.process(['remove', patterns, {'gitignore_path': gitignore_path or self.gitignore_path}])
    
    async def check_patterns(self, patterns: List[str], gitignore_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if patterns are in .gitignore.
        
        Args:
            patterns: Patterns to check
            gitignore_path: Path to .gitignore (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        return await self.process(['check', patterns, {'gitignore_path': gitignore_path or self.gitignore_path}])
    
    async def generate_gitignore(self, templates: List[str], gitignore_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a .gitignore file from templates.
        
        Args:
            templates: Templates to use (python, node, macos, windows)
            gitignore_path: Path to .gitignore (optional)
        
        Returns:
            Dictionary with the result of the operation
        """
        return await self.process(['generate', templates, {'gitignore_path': gitignore_path or self.gitignore_path}]) 