from typing import List, Dict, Any, Optional, Union
import os
import re
import ast
import importlib
import pkg_resources

from src.Step import Step
from src.ExecutorBase import ExecutorBase


class StepDependencySearch(Step):
    """
    Tool for analyzing code to identify and manage dependencies.
    
    Biological analogy: Neural pathway mapping.
    Justification: Like how the brain maps connections between neurons to understand
    relationships, this tool maps dependencies between code components to understand
    their relationships and requirements.
    """
    def __init__(self, executor: ExecutorBase, **kwargs):
        super().__init__(executor, **kwargs)
        
        # Tool-specific attributes
        self.include_standard_lib = kwargs.get('include_standard_lib', False)
        self.include_local_imports = kwargs.get('include_local_imports', True)
        self.check_installed = kwargs.get('check_installed', True)
        self.standard_libs = self._get_standard_libs()
    
    async def process(self, inputs: List[Any]) -> Any:
        """
        Analyze code to identify dependencies.
        
        Args:
            inputs: List containing:
                - code: String containing code to analyze
                - context_provider: Object providing context (AgentWorkflowBuilder)
                - options: Optional dictionary with analysis options
        
        Returns:
            Dictionary with dependency analysis results
        """
        # Extract inputs
        if not inputs or len(inputs) < 1:
            return {
                "success": False,
                "error": "Missing required input: code is required"
            }
        
        code = inputs[0]
        context_provider = inputs[1] if len(inputs) > 1 else None
        options = inputs[2] if len(inputs) > 2 else {}
        
        # Set options
        include_standard_lib = options.get('include_standard_lib', self.include_standard_lib)
        include_local_imports = options.get('include_local_imports', self.include_local_imports)
        check_installed = options.get('check_installed', self.check_installed)
        
        # Analyze the code
        try:
            # Parse imports
            imports = self._parse_imports(code)
            
            # Filter imports based on options
            filtered_imports = self._filter_imports(
                imports, 
                include_standard_lib=include_standard_lib,
                include_local_imports=include_local_imports
            )
            
            # Check if dependencies are installed
            if check_installed:
                installed_status = self._check_installed_dependencies(filtered_imports)
            else:
                installed_status = {}
            
            # Generate requirements.txt content
            requirements_txt = self._generate_requirements_txt(filtered_imports, installed_status)
            
            # Generate dependency graph
            dependency_graph = self._generate_dependency_graph(filtered_imports)
            
            return {
                "success": True,
                "imports": imports,
                "filtered_imports": filtered_imports,
                "installed_status": installed_status,
                "requirements_txt": requirements_txt,
                "dependency_graph": dependency_graph
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error analyzing dependencies: {str(e)}"
            }
    
    def _parse_imports(self, code: str) -> List[Dict[str, Any]]:
        """Parse imports from code."""
        imports = []
        
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Find all import statements
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append({
                            "type": "import",
                            "module": name.name,
                            "alias": name.asname,
                            "is_standard_lib": name.name in self.standard_libs,
                            "is_local": self._is_local_import(name.name)
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.append({
                            "type": "from",
                            "module": module,
                            "name": name.name,
                            "alias": name.asname,
                            "level": node.level,
                            "is_standard_lib": module in self.standard_libs,
                            "is_local": self._is_local_import(module)
                        })
        except SyntaxError:
            # If the code has syntax errors, try a regex-based approach
            import_pattern = r'^(?:from\s+([\w.]+)\s+)?import\s+([\w.]+)(?:\s+as\s+([\w.]+))?'
            for line in code.split('\n'):
                match = re.match(import_pattern, line.strip())
                if match:
                    from_module, import_name, alias = match.groups()
                    if from_module:
                        imports.append({
                            "type": "from",
                            "module": from_module,
                            "name": import_name,
                            "alias": alias,
                            "level": 0,
                            "is_standard_lib": from_module in self.standard_libs,
                            "is_local": self._is_local_import(from_module)
                        })
                    else:
                        imports.append({
                            "type": "import",
                            "module": import_name,
                            "alias": alias,
                            "is_standard_lib": import_name in self.standard_libs,
                            "is_local": self._is_local_import(import_name)
                        })
        
        return imports
    
    def _filter_imports(self, imports: List[Dict[str, Any]], include_standard_lib: bool = False, include_local_imports: bool = True) -> List[Dict[str, Any]]:
        """Filter imports based on options."""
        filtered = []
        
        for imp in imports:
            # Skip standard library imports if not included
            if imp.get("is_standard_lib", False) and not include_standard_lib:
                continue
            
            # Skip local imports if not included
            if imp.get("is_local", False) and not include_local_imports:
                continue
            
            filtered.append(imp)
        
        return filtered
    
    def _check_installed_dependencies(self, imports: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Check if dependencies are installed."""
        installed = {}
        
        for imp in imports:
            # Skip local imports
            if imp.get("is_local", False):
                continue
            
            # Get the top-level package name
            if imp["type"] == "import":
                package = imp["module"].split('.')[0]
            else:
                package = imp["module"].split('.')[0]
            
            # Skip if already checked
            if package in installed:
                continue
            
            # Check if installed
            try:
                importlib.import_module(package)
                installed[package] = True
            except ImportError:
                installed[package] = False
        
        return installed
    
    def _generate_requirements_txt(self, imports: List[Dict[str, Any]], installed_status: Dict[str, bool]) -> str:
        """Generate requirements.txt content."""
        requirements = []
        
        # Get unique top-level packages
        packages = set()
        for imp in imports:
            # Skip local imports
            if imp.get("is_local", False):
                continue
            
            # Skip standard library imports
            if imp.get("is_standard_lib", False):
                continue
            
            # Get the top-level package name
            if imp["type"] == "import":
                package = imp["module"].split('.')[0]
            else:
                package = imp["module"].split('.')[0]
            
            packages.add(package)
        
        # Add packages to requirements with versions if installed
        for package in sorted(packages):
            if package in installed_status and installed_status[package]:
                try:
                    version = pkg_resources.get_distribution(package).version
                    requirements.append(f"{package}=={version}")
                except:
                    requirements.append(package)
            else:
                requirements.append(package)
        
        return "\n".join(requirements)
    
    def _generate_dependency_graph(self, imports: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate a dependency graph."""
        graph = {}
        
        for imp in imports:
            if imp["type"] == "import":
                module = imp["module"]
                parts = module.split('.')
                
                # Add each part to the graph
                for i in range(len(parts)):
                    parent = '.'.join(parts[:i]) if i > 0 else None
                    child = '.'.join(parts[:i+1])
                    
                    if parent:
                        if parent not in graph:
                            graph[parent] = []
                        if child not in graph[parent]:
                            graph[parent].append(child)
                    
                    if child not in graph:
                        graph[child] = []
            elif imp["type"] == "from":
                module = imp["module"]
                name = imp["name"]
                
                if module not in graph:
                    graph[module] = []
                
                full_name = f"{module}.{name}"
                if full_name not in graph[module]:
                    graph[module].append(full_name)
        
        return graph
    
    def _get_standard_libs(self) -> List[str]:
        """Get a list of standard library modules."""
        import sys
        import sysconfig
        
        # Get standard library paths
        stdlib_paths = [
            sysconfig.get_path('stdlib'),
            sysconfig.get_path('platstdlib')
        ]
        
        # Get all modules in sys.modules that are in the standard library paths
        stdlib_modules = []
        for module_name, module in sys.modules.items():
            if hasattr(module, '__file__') and module.__file__:
                if any(module.__file__.startswith(path) for path in stdlib_paths):
                    stdlib_modules.append(module_name.split('.')[0])
        
        # Add common standard library modules
        common_stdlib = [
            'abc', 'argparse', 'ast', 'asyncio', 'base64', 'collections', 'concurrent',
            'contextlib', 'copy', 'csv', 'datetime', 'decimal', 'difflib', 'enum',
            'functools', 'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
            'importlib', 'inspect', 'io', 'itertools', 'json', 'logging', 'math',
            'multiprocessing', 'operator', 'os', 'pathlib', 'pickle', 'platform',
            'pprint', 'random', 're', 'shutil', 'signal', 'socket', 'sqlite3',
            'statistics', 'string', 'struct', 'subprocess', 'sys', 'tempfile',
            'threading', 'time', 'traceback', 'types', 'typing', 'unittest', 'urllib',
            'uuid', 'warnings', 'weakref', 'xml', 'zipfile'
        ]
        
        # Combine and deduplicate
        return list(set(stdlib_modules + common_stdlib))
    
    def _is_local_import(self, module_name: str) -> bool:
        """Check if an import is local."""
        # Consider imports starting with src, tools, builder as local
        local_prefixes = ['src', 'tools', 'builder']
        return any(module_name.startswith(prefix) for prefix in local_prefixes)
    
    async def analyze_code(self, code: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze code to identify dependencies.
        
        Args:
            code: String containing code to analyze
            options: Optional dictionary with analysis options
        
        Returns:
            Dictionary with dependency analysis results
        """
        return await self.process([code, None, options or {}])
    
    async def generate_requirements(self, code: str) -> Dict[str, Any]:
        """
        Generate requirements.txt content from code.
        
        Args:
            code: String containing code to analyze
        
        Returns:
            Dictionary with requirements.txt content
        """
        options = {
            "include_standard_lib": False,
            "include_local_imports": False,
            "check_installed": True
        }
        
        result = await self.process([code, None, options])
        
        if result.get("success", False):
            return {
                "success": True,
                "requirements_txt": result.get("requirements_txt", "")
            }
        else:
            return result
    
    async def check_dependencies(self, code: str) -> Dict[str, Any]:
        """
        Check if dependencies are installed.
        
        Args:
            code: String containing code to analyze
        
        Returns:
            Dictionary with dependency status
        """
        options = {
            "include_standard_lib": False,
            "include_local_imports": False,
            "check_installed": True
        }
        
        result = await self.process([code, None, options])
        
        if result.get("success", False):
            return {
                "success": True,
                "installed_status": result.get("installed_status", {})
            }
        else:
            return result 