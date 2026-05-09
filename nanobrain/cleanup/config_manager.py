"""
Configuration management system for the Nanobrain framework cleanup.

This module provides functionality to standardize configuration patterns including:
- Hardcoded path detection and replacement
- Environment variable substitution
- JSON Schema validation
- Template configuration creation
- YAML structure standardization
"""

import re
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Optional dependency for JSON schema validation
try:
    import jsonschema
    from jsonschema import validate, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception  # Fallback for type hints


@dataclass
class ConfigIssue:
    """Represents a configuration-related issue found during analysis."""
    file_path: Path
    issue_type: str  # 'hardcoded_path', 'invalid_schema', 'inconsistent_structure'
    description: str
    line_number: Optional[int] = None
    suggested_fix: Optional[str] = None
    severity: str = 'medium'  # 'low', 'medium', 'high'


@dataclass
class HardcodedPath:
    """Represents a hardcoded path in configuration."""
    file_path: Path
    line_number: int
    original_path: str
    suggested_replacement: str
    context: str  # The full line containing the path


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    file_path: Path
    is_valid: bool
    schema_errors: List[str]
    warnings: List[str]
    schema_used: Optional[str] = None


class ConfigManager:
    """
    Manages configuration standardization for the Nanobrain framework.
    
    This class analyzes YAML configuration files to detect and fix:
    - Hardcoded paths that should be environment variables
    - Invalid configuration structures
    - Inconsistent naming conventions
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.issues: List[ConfigIssue] = []
        self.config_files: List[Path] = []
        self.schemas: Dict[str, Dict[str, Any]] = {}
        
        # Common hardcoded path patterns to detect
        self.hardcoded_patterns = [
            r'/lus/flare/projects/FoundEpidem',
            r'/home/[^/\s]+',
            r'C:\\[^\\s]*',
            r'/Users/[^/\s]+',
            r'/opt/[^/\s]*',
            r'/usr/local/[^/\s]*',
            r'/tmp/[^/\s]*',
        ]
        
        # Load default schemas
        self._load_default_schemas()
    
    def analyze_configurations(self) -> 'CleanupResult':
        """
        Analyze all configuration files for issues.
        
        Returns:
            CleanupResult with analysis results and detected issues
        """
        try:
            self.logger.info("Starting configuration analysis")
            self.issues.clear()
            self.config_files.clear()
            
            # Find all configuration files
            config_files = self._find_config_files()
            self.config_files = config_files
            self.logger.info(f"Found {len(config_files)} configuration files to analyze")
            
            # Analyze each file
            for config_file in config_files:
                self._analyze_config_file(config_file)
                
            self.logger.info(f"Configuration analysis complete. Found {len(self.issues)} issues")
            
            from .models import CleanupResult
            return CleanupResult(
                success=True,
                message=f"Configuration analysis complete. Found {len(self.issues)} issues",
                files_processed=len(config_files),
                details={
                    'issues_found': len(self.issues),
                    'hardcoded_paths': len([i for i in self.issues if i.issue_type == 'hardcoded_path']),
                    'schema_violations': len([i for i in self.issues if i.issue_type == 'invalid_schema']),
                    'structure_issues': len([i for i in self.issues if i.issue_type == 'inconsistent_structure'])
                }
            )
            
        except Exception as e:
            error_msg = f"Configuration analysis failed: {str(e)}"
            self.logger.error(error_msg)
            from .models import CleanupResult
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def fix_hardcoded_paths(self) -> 'CleanupResult':
        """
        Replace hardcoded paths with environment variables or relative paths.
        
        Returns:
            CleanupResult with fix results
        """
        try:
            hardcoded_issues = [i for i in self.issues if i.issue_type == 'hardcoded_path']
            if not hardcoded_issues:
                from .models import CleanupResult
                return CleanupResult(
                    success=True,
                    message="No hardcoded paths to fix"
                )
                
            self.logger.info(f"Fixing {len(hardcoded_issues)} hardcoded paths")
            fixed_count = 0
            
            # Group issues by file for efficient processing
            issues_by_file = {}
            for issue in hardcoded_issues:
                if issue.file_path not in issues_by_file:
                    issues_by_file[issue.file_path] = []
                issues_by_file[issue.file_path].append(issue)
            
            for file_path, file_issues in issues_by_file.items():
                if self._fix_hardcoded_paths_in_file(file_path, file_issues):
                    fixed_count += len(file_issues)
                    
            from .models import CleanupResult
            return CleanupResult(
                success=True,
                message=f"Fixed {fixed_count}/{len(hardcoded_issues)} hardcoded paths",
                files_processed=len(issues_by_file),
                details={'fixed_count': fixed_count, 'total_issues': len(hardcoded_issues)}
            )
            
        except Exception as e:
            error_msg = f"Hardcoded path fixing failed: {str(e)}"
            self.logger.error(error_msg)
            from .models import CleanupResult
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def validate_configurations(self) -> 'CleanupResult':
        """
        Validate configuration files against schemas.
        
        Returns:
            CleanupResult with validation results
        """
        try:
            self.logger.info("Validating configuration files")
            
            validation_results = []
            for config_file in self.config_files:
                result = self._validate_config_file(config_file)
                validation_results.append(result)
            
            valid_count = sum(1 for r in validation_results if r.is_valid)
            invalid_count = len(validation_results) - valid_count
            
            from .models import CleanupResult
            return CleanupResult(
                success=True,
                message=f"Configuration validation complete. {valid_count} valid, {invalid_count} invalid",
                files_processed=len(validation_results),
                details={
                    'valid_configs': valid_count,
                    'invalid_configs': invalid_count,
                    'validation_results': validation_results
                }
            )
            
        except Exception as e:
            error_msg = f"Configuration validation failed: {str(e)}"
            self.logger.error(error_msg)
            from .models import CleanupResult
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def create_template_configurations(self) -> 'CleanupResult':
        """
        Create template configurations for common use cases.
        
        Returns:
            CleanupResult with template creation results
        """
        try:
            self.logger.info("Creating template configurations")
            
            templates_dir = self.project_root / "config" / "templates"
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            templates_created = []
            
            # Create workflow template
            workflow_template = self._create_workflow_template()
            workflow_path = templates_dir / "workflow_template.yaml"
            with open(workflow_path, 'w') as f:
                yaml.dump(workflow_template, f, default_flow_style=False, sort_keys=False)
            templates_created.append(workflow_path)
            
            # Create executor template
            executor_template = self._create_executor_template()
            executor_path = templates_dir / "executor_template.yaml"
            with open(executor_path, 'w') as f:
                yaml.dump(executor_template, f, default_flow_style=False, sort_keys=False)
            templates_created.append(executor_path)
            
            # Create environment template
            env_template = self._create_environment_template()
            env_path = templates_dir / "environment_template.yaml"
            with open(env_path, 'w') as f:
                yaml.dump(env_template, f, default_flow_style=False, sort_keys=False)
            templates_created.append(env_path)
            
            from .models import CleanupResult
            return CleanupResult(
                success=True,
                message=f"Created {len(templates_created)} template configurations",
                files_processed=len(templates_created),
                details={'templates_created': [str(p) for p in templates_created]}
            )
            
        except Exception as e:
            error_msg = f"Template creation failed: {str(e)}"
            self.logger.error(error_msg)
            from .models import CleanupResult
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def standardize_yaml_structure(self) -> 'CleanupResult':
        """
        Standardize YAML structure and naming conventions.
        
        Returns:
            CleanupResult with standardization results
        """
        try:
            self.logger.info("Standardizing YAML structure and naming")
            
            standardized_count = 0
            
            for config_file in self.config_files:
                if self._standardize_yaml_file(config_file):
                    standardized_count += 1
                    
            from .models import CleanupResult
            return CleanupResult(
                success=True,
                message=f"Standardized {standardized_count}/{len(self.config_files)} configuration files",
                files_processed=standardized_count,
                details={'standardized_count': standardized_count, 'total_files': len(self.config_files)}
            )
            
        except Exception as e:
            error_msg = f"YAML standardization failed: {str(e)}"
            self.logger.error(error_msg)
            from .models import CleanupResult
            return CleanupResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def get_issues(self) -> List[ConfigIssue]:
        """Get all detected configuration issues."""
        return self.issues.copy()
    
    def _find_config_files(self) -> List[Path]:
        """Find all configuration files in the project."""
        config_files = []
        
        # Common configuration file patterns
        patterns = ['*.yaml', '*.yml', '*.json']
        
        # Search in common configuration directories
        search_dirs = [
            self.project_root / "config",
            self.project_root / "configs",
            self.project_root / "envs",
            self.project_root / "examples",
            self.project_root / "demos",
        ]
        
        # Also search in root directory
        search_dirs.append(self.project_root)
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for pattern in patterns:
                    config_files.extend(search_dir.rglob(pattern))
        
        # Filter out common non-config files
        filtered_files = []
        exclude_patterns = ['node_modules', '.git', '__pycache__', 'build', 'dist']
        
        for config_file in config_files:
            if not any(exclude in str(config_file) for exclude in exclude_patterns):
                filtered_files.append(config_file)
        
        return filtered_files
    
    def _analyze_config_file(self, config_file: Path) -> None:
        """Analyze a single configuration file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Detect hardcoded paths
            self._detect_hardcoded_paths_in_content(config_file, content)
            
            # Parse and validate structure
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                try:
                    config_data = yaml.safe_load(content)
                    self._analyze_yaml_structure(config_file, config_data)
                except yaml.YAMLError as e:
                    self._add_issue(config_file, 'invalid_schema', f"Invalid YAML syntax: {str(e)}")
            elif config_file.suffix.lower() == '.json':
                try:
                    config_data = json.loads(content)
                    self._analyze_json_structure(config_file, config_data)
                except json.JSONDecodeError as e:
                    self._add_issue(config_file, 'invalid_schema', f"Invalid JSON syntax: {str(e)}")
                    
        except (UnicodeDecodeError, IOError) as e:
            self.logger.warning(f"Could not read {config_file}: {str(e)}")
    
    def _detect_hardcoded_paths_in_content(self, config_file: Path, content: str) -> None:
        """Detect hardcoded paths in configuration content."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Only process string lines, skip empty lines
            if not isinstance(line, str) or not line.strip():
                continue
                
            for pattern in self.hardcoded_patterns:
                try:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        hardcoded_path = match.group(0)
                        suggested_replacement = self._suggest_path_replacement(hardcoded_path)
                        
                        issue = ConfigIssue(
                            file_path=config_file,
                            issue_type='hardcoded_path',
                            description=f"Hardcoded path detected: {hardcoded_path}",
                            line_number=line_num,
                            suggested_fix=f"Replace with: {suggested_replacement}",
                            severity='high'
                        )
                        self.issues.append(issue)
                except TypeError:
                    # Skip non-string values that can't be processed by regex
                    continue
    
    def _suggest_path_replacement(self, hardcoded_path: str) -> str:
        """Suggest a replacement for a hardcoded path."""
        if '/lus/flare/projects/FoundEpidem' in hardcoded_path:
            return "${NANOBRAIN_PROJECT_ROOT}"
        elif '/home/' in hardcoded_path:
            return "${HOME}/nanobrain"
        elif 'C:\\' in hardcoded_path:
            return "${NANOBRAIN_HOME}"
        elif '/tmp/' in hardcoded_path:
            return "${TMPDIR:-/tmp}/nanobrain"
        else:
            return "${NANOBRAIN_ROOT}"
    
    def _analyze_yaml_structure(self, config_file: Path, config_data: Any) -> None:
        """Analyze YAML structure for consistency issues."""
        if not isinstance(config_data, dict):
            return
            
        # Check for common structure issues
        self._check_naming_conventions(config_file, config_data)
        self._check_required_sections(config_file, config_data)
    
    def _analyze_json_structure(self, config_file: Path, config_data: Any) -> None:
        """Analyze JSON structure for consistency issues."""
        if not isinstance(config_data, dict):
            return
            
        # Similar checks as YAML
        self._check_naming_conventions(config_file, config_data)
    
    def _check_naming_conventions(self, config_file: Path, config_data: dict) -> None:
        """Check naming conventions in configuration."""
        def check_keys(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Check for inconsistent naming (camelCase vs snake_case)
                    if re.search(r'[a-z][A-Z]', key):  # camelCase
                        if any('_' in k for k in obj.keys()):  # mixed with snake_case
                            self._add_issue(
                                config_file,
                                'inconsistent_structure',
                                f"Mixed naming conventions at {path}.{key}",
                                suggested_fix="Use consistent snake_case naming"
                            )
                    
                    check_keys(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_keys(item, f"{path}[{i}]")
        
        check_keys(config_data)
    
    def _check_required_sections(self, config_file: Path, config_data: dict) -> None:
        """Check for required sections in configuration files."""
        # Define expected sections based on file type/location
        if 'workflow' in config_file.name.lower():
            required_sections = ['name', 'steps', 'executor']
            for section in required_sections:
                if section not in config_data:
                    self._add_issue(
                        config_file,
                        'inconsistent_structure',
                        f"Missing required section: {section}",
                        suggested_fix=f"Add {section} section to configuration"
                    )
    
    def _add_issue(self, file_path: Path, issue_type: str, description: str, 
                   line_number: Optional[int] = None, suggested_fix: Optional[str] = None,
                   severity: str = 'medium') -> None:
        """Add a configuration issue."""
        issue = ConfigIssue(
            file_path=file_path,
            issue_type=issue_type,
            description=description,
            line_number=line_number,
            suggested_fix=suggested_fix,
            severity=severity
        )
        self.issues.append(issue)
    
    def _fix_hardcoded_paths_in_file(self, file_path: Path, issues: List[ConfigIssue]) -> bool:
        """Fix hardcoded paths in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort issues by line number in reverse order to avoid offset issues
            sorted_issues = sorted(issues, key=lambda x: x.line_number or 0, reverse=True)
            
            modified = False
            for issue in sorted_issues:
                if issue.line_number and issue.line_number <= len(lines):
                    line = lines[issue.line_number - 1]
                    
                    # Apply suggested fix
                    for pattern in self.hardcoded_patterns:
                        if re.search(pattern, line):
                            # Extract the hardcoded path
                            match = re.search(pattern, line)
                            if match:
                                hardcoded_path = match.group(0)
                                replacement = self._suggest_path_replacement(hardcoded_path)
                                new_line = line.replace(hardcoded_path, replacement)
                                lines[issue.line_number - 1] = new_line
                                modified = True
                                break
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                self.logger.info(f"Fixed hardcoded paths in {file_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to fix hardcoded paths in {file_path}: {str(e)}")
            
        return False
    
    def _validate_config_file(self, config_file: Path) -> ConfigValidationResult:
        """Validate a configuration file against appropriate schema."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(content)
            elif config_file.suffix.lower() == '.json':
                config_data = json.loads(content)
            else:
                return ConfigValidationResult(
                    file_path=config_file,
                    is_valid=False,
                    schema_errors=["Unsupported file format"],
                    warnings=[]
                )
            
            # Check if jsonschema is available
            if not HAS_JSONSCHEMA:
                return ConfigValidationResult(
                    file_path=config_file,
                    is_valid=True,
                    schema_errors=[],
                    warnings=["jsonschema not available - skipping schema validation"],
                    schema_used=None
                )
            
            # Determine appropriate schema
            schema_name = self._determine_schema(config_file, config_data)
            
            if schema_name and schema_name in self.schemas:
                try:
                    validate(instance=config_data, schema=self.schemas[schema_name])
                    return ConfigValidationResult(
                        file_path=config_file,
                        is_valid=True,
                        schema_errors=[],
                        warnings=[],
                        schema_used=schema_name
                    )
                except ValidationError as e:
                    return ConfigValidationResult(
                        file_path=config_file,
                        is_valid=False,
                        schema_errors=[str(e)],
                        warnings=[],
                        schema_used=schema_name
                    )
            else:
                return ConfigValidationResult(
                    file_path=config_file,
                    is_valid=True,  # No schema to validate against
                    schema_errors=[],
                    warnings=["No appropriate schema found for validation"],
                    schema_used=None
                )
                
        except Exception as e:
            return ConfigValidationResult(
                file_path=config_file,
                is_valid=False,
                schema_errors=[f"Validation error: {str(e)}"],
                warnings=[]
            )
    
    def _determine_schema(self, config_file: Path, config_data: Any) -> Optional[str]:
        """Determine which schema to use for validation."""
        if not isinstance(config_data, dict):
            return None
            
        # Check file name patterns
        if 'workflow' in config_file.name.lower():
            return 'workflow'
        elif 'executor' in config_file.name.lower():
            return 'executor'
        elif 'env' in config_file.name.lower():
            return 'environment'
            
        # Check content patterns
        if 'steps' in config_data and 'name' in config_data:
            return 'workflow'
        elif 'provider' in config_data and 'max_workers' in config_data:
            return 'executor'
            
        return None
    
    def _standardize_yaml_file(self, config_file: Path) -> bool:
        """Standardize a single YAML file."""
        try:
            if config_file.suffix.lower() not in ['.yaml', '.yml']:
                return False
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not isinstance(config_data, dict):
                return False
            
            # Standardize naming conventions
            standardized_data = self._standardize_naming(config_data)
            
            # Write back with consistent formatting
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(standardized_data, f, default_flow_style=False, 
                         sort_keys=False, indent=2, width=120)
            
            self.logger.info(f"Standardized {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to standardize {config_file}: {str(e)}")
            return False
    
    def _standardize_naming(self, obj: Any) -> Any:
        """Recursively standardize naming conventions."""
        if isinstance(obj, dict):
            standardized = {}
            for key, value in obj.items():
                # Only apply regex to string keys
                if isinstance(key, str):
                    # Convert camelCase to snake_case
                    snake_key = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', key).lower()
                else:
                    snake_key = key
                standardized[snake_key] = self._standardize_naming(value)
            return standardized
        elif isinstance(obj, list):
            return [self._standardize_naming(item) for item in obj]
        else:
            # Return non-dict, non-list objects as-is (including integers, floats, booleans, etc.)
            return obj
    
    def _load_default_schemas(self) -> None:
        """Load default JSON schemas for validation."""
        # Workflow schema
        self.schemas['workflow'] = {
            "type": "object",
            "required": ["name", "steps"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "config": {"type": "object"}
                        }
                    }
                },
                "executor": {"type": "object"}
            }
        }
        
        # Executor schema
        self.schemas['executor'] = {
            "type": "object",
            "required": ["provider"],
            "properties": {
                "provider": {"type": "string"},
                "max_workers": {"type": "integer", "minimum": 1},
                "config": {"type": "object"}
            }
        }
        
        # Environment schema
        self.schemas['environment'] = {
            "type": "object",
            "properties": {
                "variables": {"type": "object"},
                "paths": {"type": "object"},
                "modules": {"type": "array"}
            }
        }
    
    def _create_workflow_template(self) -> Dict[str, Any]:
        """Create a workflow configuration template."""
        return {
            "name": "example_workflow",
            "description": "Example workflow configuration",
            "steps": [
                {
                    "name": "data_acquisition",
                    "type": "data_step",
                    "config": {
                        "source": "${DATA_SOURCE}",
                        "output_dir": "${WORKFLOW_OUTPUT_DIR}/data"
                    }
                },
                {
                    "name": "processing",
                    "type": "compute_step",
                    "config": {
                        "input_dir": "${WORKFLOW_OUTPUT_DIR}/data",
                        "output_dir": "${WORKFLOW_OUTPUT_DIR}/results"
                    }
                }
            ],
            "executor": {
                "provider": "parsl",
                "config": {
                    "max_workers": 4,
                    "working_dir": "${WORKFLOW_WORKING_DIR}"
                }
            },
            "environment": {
                "variables": {
                    "DATA_SOURCE": "/path/to/data",
                    "WORKFLOW_OUTPUT_DIR": "${HOME}/nanobrain/outputs",
                    "WORKFLOW_WORKING_DIR": "${TMPDIR:-/tmp}/nanobrain"
                }
            }
        }
    
    def _create_executor_template(self) -> Dict[str, Any]:
        """Create an executor configuration template."""
        return {
            "provider": "parsl",
            "max_workers": 8,
            "config": {
                "working_dir": "${NANOBRAIN_WORKING_DIR}",
                "log_dir": "${NANOBRAIN_LOG_DIR}",
                "monitoring": {
                    "enabled": True,
                    "database_url": "${NANOBRAIN_MONITORING_DB}"
                }
            },
            "resources": {
                "cores_per_worker": 1,
                "memory_per_worker": "2GB",
                "walltime": "01:00:00"
            }
        }
    
    def _create_environment_template(self) -> Dict[str, Any]:
        """Create an environment configuration template."""
        return {
            "variables": {
                "NANOBRAIN_ROOT": "${HOME}/nanobrain",
                "NANOBRAIN_DATA_DIR": "${NANOBRAIN_ROOT}/data",
                "NANOBRAIN_OUTPUT_DIR": "${NANOBRAIN_ROOT}/outputs",
                "NANOBRAIN_LOG_DIR": "${NANOBRAIN_ROOT}/logs",
                "NANOBRAIN_WORKING_DIR": "${TMPDIR:-/tmp}/nanobrain",
                "NANOBRAIN_CONFIG_DIR": "${NANOBRAIN_ROOT}/config"
            },
            "paths": {
                "python_path": ["${NANOBRAIN_ROOT}", "${NANOBRAIN_ROOT}/nanobrain"],
                "library_path": ["${NANOBRAIN_ROOT}/lib"]
            },
            "modules": [
                "python/3.9",
                "gcc/9.3.0"
            ],
            "conda": {
                "environment": "nanobrain",
                "channels": ["conda-forge", "bioconda"]
            }
        }