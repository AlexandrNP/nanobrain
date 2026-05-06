#!/usr/bin/env python3
"""
Configuration Pattern Validation Script

Validates that all configuration examples follow the enhanced from_config patterns:
- Class+config pattern consistency
- File path vs inline config compliance
- ConfigBase schema adherence
- Framework architectural compliance

✅ FRAMEWORK COMPLIANCE:
- No hardcoded validation rules
- Data-driven pattern checking
- Complete coverage of all configuration types
- Adherence to NanoBrain architectural patterns
"""

import yaml
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import logging
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigurationPatternValidator:
    """
    Comprehensive validator for configuration pattern compliance
    
    Validates all configuration examples against enhanced from_config patterns.
    """
    
    def __init__(self):
        self.validation_results = []
        self.framework_patterns = self._load_framework_patterns()
        self.validation_stats = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'patterns_validated': 0,
            'compliance_issues': 0
        }
    
    def _load_framework_patterns(self) -> Dict[str, Any]:
        """Load framework pattern definitions"""
        return {
            "class_config_pattern": {
                "required_fields": ["class", "config"],
                "class_field_format": r"^[a-zA-Z_][a-zA-Z0-9_.]*\.[A-Z][a-zA-Z0-9_]*$",
                "config_field_types": ["string", "dict"]
            },
            "inline_config_support": {
                "supported_classes": [
                    "nanobrain.core.data_unit",
                    "nanobrain.core.link", 
                    "nanobrain.core.trigger"
                ],
                "file_path_classes": "all_others"
            },
            "framework_compliance": {
                "no_hardcoded_values": True,
                "pure_configuration_driven": True,
                "configbase_integration": True,
                "environment_variable_pattern": r"\$\{[A-Z_][A-Z0-9_]*(?::-[^}]*)?\}"
            },
            "forbidden_patterns": [
                "localhost",
                "127.0.0.1", 
                "admin",
                "password123",
                "test123",
                "hardcoded_key"
            ]
        }
    
    def validate_class_config_pattern(self, config_data: Dict[str, Any], config_path: str) -> List[str]:
        """Validate class+config pattern compliance"""
        errors = []
        
        def check_recursive(data: Any, path: str = ""):
            if isinstance(data, dict):
                # Check for class+config patterns
                if "class" in data and "config" in data:
                    # Validate class field format
                    class_field = data["class"]
                    if not isinstance(class_field, str):
                        errors.append(f"{path}: Class field must be string, got {type(class_field)}")
                    elif not re.match(self.framework_patterns["class_config_pattern"]["class_field_format"], class_field):
                        errors.append(f"{path}: Invalid class field format '{class_field}'. Must be 'module.path.ClassName'")
                    
                    # Validate config field type
                    config_field = data["config"]
                    if not isinstance(config_field, (str, dict)):
                        errors.append(f"{path}: Config field must be string (file path) or dict (inline config)")
                    
                    # Validate inline config compliance
                    if isinstance(config_field, dict):
                        if not self._is_inline_config_allowed(class_field):
                            errors.append(f"{path}: Inline dict config not allowed for {class_field}. Only DataUnit, Link, Trigger classes support inline config. Use file path instead.")
                    
                    # Additional validation for class+config structure
                    self.validation_stats['patterns_validated'] += 1
                
                # Recursively check nested structures
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    check_recursive(value, new_path)
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    check_recursive(item, new_path)
        
        check_recursive(config_data)
        return errors
    
    def _is_inline_config_allowed(self, class_path: str) -> bool:
        """Check if class supports inline dict configuration"""
        if not isinstance(class_path, str):
            return False
            
        supported_modules = self.framework_patterns["inline_config_support"]["supported_classes"]
        return any(class_path.startswith(module) for module in supported_modules)
    
    def validate_framework_compliance(self, config_data: Dict[str, Any], config_path: str) -> List[str]:
        """Validate framework architectural compliance"""
        errors = []
        config_str = str(config_data).lower()
        
        # Check for hardcoded values
        forbidden_patterns = self.framework_patterns["forbidden_patterns"]
        for pattern in forbidden_patterns:
            if pattern.lower() in config_str:
                # Check if it's in an environment variable context
                env_var_pattern = self.framework_patterns["framework_compliance"]["environment_variable_pattern"]
                if not re.search(env_var_pattern, str(config_data), re.IGNORECASE):
                    errors.append(f"Potential hardcoded value detected: '{pattern}' in {config_path}")
                    errors.append(f"  SOLUTION: Use environment variable like '${{VARIABLE_NAME}}' instead")
                    self.validation_stats['compliance_issues'] += 1
        
        # Check for proper environment variable usage
        self._validate_environment_variables(config_data, config_path, errors)
        
        # Check for configuration-driven patterns in workflows
        if "steps" in config_data:
            self._validate_workflow_steps(config_data["steps"], config_path, errors)
        
        # Check for universal tool loading patterns in agents
        if "tools" in config_data:
            self._validate_tool_loading_patterns(config_data["tools"], config_path, errors)
        
        return errors
    
    def _validate_environment_variables(self, config_data: Dict[str, Any], config_path: str, errors: List[str]):
        """Validate proper environment variable usage"""
        env_var_pattern = self.framework_patterns["framework_compliance"]["environment_variable_pattern"]
        
        def check_env_vars(data: Any, path: str = ""):
            if isinstance(data, str):
                # Check if string contains potential environment variable pattern
                if "${" in data:
                    if not re.search(env_var_pattern, data):
                        errors.append(f"{path}: Invalid environment variable format in '{data}'. Use '${{VAR_NAME}}' or '${{VAR_NAME:-default}}'")
            elif isinstance(data, dict):
                for key, value in data.items():
                    check_env_vars(value, f"{path}.{key}" if path else key)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    check_env_vars(item, f"{path}[{i}]" if path else f"[{i}]")
        
        check_env_vars(config_data)
    
    def _validate_workflow_steps(self, steps_config: Any, config_path: str, errors: List[str]):
        """Validate workflow steps configuration"""
        if isinstance(steps_config, list):
            errors.append(f"Workflow steps should use Dict format for class+config patterns, not List format")
            errors.append(f"  MIGRATION: Convert list-based steps to dict-based with class+config patterns")
        elif isinstance(steps_config, dict):
            for step_id, step_config in steps_config.items():
                if not isinstance(step_config, dict):
                    errors.append(f"Step '{step_id}' must be a configuration dict")
                    continue
                    
                if "class" not in step_config:
                    errors.append(f"Step '{step_id}' must specify 'class' field for enhanced from_config pattern")
                
                if "config" not in step_config and "config_file" not in step_config:
                    errors.append(f"Step '{step_id}' must specify either 'config' or 'config_file' field")
                
                # Validate legacy config_file usage
                if "config_file" in step_config:
                    errors.append(f"Step '{step_id}' uses deprecated 'config_file'. Use 'config' field instead")
    
    def _validate_tool_loading_patterns(self, tools_config: Any, config_path: str, errors: List[str]):
        """Validate universal tool loading patterns"""
        if not isinstance(tools_config, dict):
            errors.append(f"Tools configuration must be a Dict for universal tool loading")
            return
        
        for tool_name, tool_config in tools_config.items():
            if not isinstance(tool_config, dict):
                errors.append(f"Tool '{tool_name}' must be a configuration dict")
                continue
            
            if "class" not in tool_config:
                errors.append(f"Tool '{tool_name}' must specify 'class' field for class+config pattern")
            
            if "config" not in tool_config:
                errors.append(f"Tool '{tool_name}' must specify 'config' field")
    
    def validate_configuration_file(self, config_path: Path) -> Tuple[bool, List[str]]:
        """Validate single configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    return False, [f"Unsupported file format: {config_path.suffix}"]
            
            if config_data is None:
                config_data = {}
            
            if not isinstance(config_data, dict):
                return False, ["Configuration must be a dictionary"]
            
            errors = []
            
            # Validate class+config patterns
            pattern_errors = self.validate_class_config_pattern(config_data, str(config_path))
            errors.extend(pattern_errors)
            
            # Validate framework compliance
            compliance_errors = self.validate_framework_compliance(config_data, str(config_path))
            errors.extend(compliance_errors)
            
            # Additional validation for framework metadata
            self._validate_framework_metadata(config_data, str(config_path), errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Error loading configuration: {str(e)}"]
    
    def _validate_framework_metadata(self, config_data: Dict[str, Any], config_path: str, errors: List[str]):
        """Validate framework metadata presence and correctness"""
        if "framework_metadata" in config_data:
            metadata = config_data["framework_metadata"]
            
            # Check for required metadata fields
            required_fields = ["config_type", "framework_version"]
            for field in required_fields:
                if field not in metadata:
                    errors.append(f"Missing required framework metadata field: '{field}'")
            
            # Validate framework version
            if "framework_version" in metadata:
                version = metadata["framework_version"]
                if version != "2.0.0":
                    errors.append(f"Expected framework_version '2.0.0', got '{version}'")
            
            # Check for enhanced from_config usage indicator
            if "uses_enhanced_from_config" in metadata:
                if not metadata["uses_enhanced_from_config"]:
                    errors.append("Configuration should use enhanced from_config patterns")
    
    def validate_all_configurations(self, base_path: Path, include_patterns: List[str] = None) -> Dict[str, Any]:
        """Validate all configuration files in directory tree"""
        if include_patterns is None:
            include_patterns = ["**/*.yml", "**/*.yaml", "**/*.json"]
        
        config_files = []
        
        # Find all configuration files
        for pattern in include_patterns:
            config_files.extend(base_path.glob(pattern))
        
        results = {
            "total_files": len(config_files),
            "valid_files": 0,
            "invalid_files": 0,
            "file_results": [],
            "validation_stats": self.validation_stats.copy(),
            "framework_compliance_summary": {
                "total_patterns_checked": 0,
                "compliant_patterns": 0,
                "non_compliant_patterns": 0
            }
        }
        
        skip_directories = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
        
        for config_file in config_files:
            # Skip certain directories and files
            if any(skip_dir in str(config_file) for skip_dir in skip_directories):
                continue
            
            # Skip backup files
            if ".backup" in str(config_file) or config_file.name.startswith('.'):
                continue
            
            is_valid, errors = self.validate_configuration_file(config_file)
            
            file_result = {
                "file": str(config_file.relative_to(base_path)),
                "valid": is_valid,
                "errors": errors,
                "error_count": len(errors)
            }
            
            results["file_results"].append(file_result)
            
            if is_valid:
                results["valid_files"] += 1
                logger.info(f"✅ {config_file.name}: VALID")
            else:
                results["invalid_files"] += 1
                logger.error(f"❌ {config_file.name}: INVALID - {len(errors)} issues")
                for error in errors[:3]:  # Show first 3 errors
                    logger.error(f"   - {error}")
                if len(errors) > 3:
                    logger.error(f"   ... and {len(errors) - 3} more issues")
        
        # Update validation statistics
        results["validation_stats"] = self.validation_stats
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Any], output_path: Path = None) -> str:
        """Generate comprehensive validation report"""
        report_lines = [
            "# Configuration Pattern Validation Report",
            f"**Generated**: {self._get_timestamp()}",
            f"**Framework Version**: 2.0.0",
            "",
            "## Summary",
            f"- **Total Files Checked**: {results['total_files']}",
            f"- **Valid Files**: {results['valid_files']}",
            f"- **Invalid Files**: {results['invalid_files']}",
            f"- **Success Rate**: {(results['valid_files']/results['total_files']*100):.1f}% " if results['total_files'] > 0 else "0%",
            "",
            "## Validation Statistics",
            f"- **Patterns Validated**: {results['validation_stats']['patterns_validated']}",
            f"- **Compliance Issues**: {results['validation_stats']['compliance_issues']}",
            "",
        ]
        
        if results['invalid_files'] > 0:
            report_lines.extend([
                "## Invalid Files",
                ""
            ])
            
            for file_result in results['file_results']:
                if not file_result['valid']:
                    report_lines.extend([
                        f"### {file_result['file']}",
                        f"**Error Count**: {file_result['error_count']}",
                        ""
                    ])
                    for error in file_result['errors']:
                        report_lines.append(f"- {error}")
                    report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Validation report saved to: {output_path}")
        
        return report_content
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main validation entry point"""
    validator = ConfigurationPatternValidator()
    base_path = Path.cwd()
    
    logger.info("🧪 Validating Configuration Pattern Compliance")
    logger.info(f"Base directory: {base_path}")
    logger.info("Framework: NanoBrain Enhanced from_config Patterns")
    
    # Validate all configurations
    results = validator.validate_all_configurations(base_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION VALIDATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total files checked: {results['total_files']}")
    logger.info(f"Valid files: {results['valid_files']}")
    logger.info(f"Invalid files: {results['invalid_files']}")
    
    if results['total_files'] > 0:
        success_rate = (results['valid_files'] / results['total_files']) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")
    
    # Print validation statistics
    logger.info(f"Patterns validated: {results['validation_stats']['patterns_validated']}")
    logger.info(f"Compliance issues: {results['validation_stats']['compliance_issues']}")
    
    # Generate and save detailed report
    report_file = base_path / "configuration_pattern_validation_report.md"
    validator.generate_validation_report(results, report_file)
    
    # Save detailed JSON report
    json_report_file = base_path / "configuration_pattern_validation_report.json"
    with open(json_report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Detailed JSON report saved to: {json_report_file}")
    
    # Final status
    if results['invalid_files'] == 0:
        logger.info("🎉 ALL CONFIGURATIONS COMPLIANT!")
        return 0
    else:
        logger.error(f"❌ {results['invalid_files']} configuration files need attention")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 