#!/usr/bin/env python3
"""
Configuration Migration Validation Script

Validates that all config file references follow new organizational rules.
NO HARDCODING - discovers all components and validates their config patterns.

This script implements comprehensive validation for the CONFIG_PATH_RESOLUTION_AND_MIGRATION_PLAN.md
following Nanobrain's data-driven, configurable architecture principles.
"""

import os
import sys
import inspect
import importlib
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Type, Set
from dataclasses import dataclass, field

# Add nanobrain to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanobrain.core.component_base import FromConfigBase


@dataclass
class ValidationViolation:
    """Represents a configuration validation violation"""
    component: str
    error: str
    violation_type: str
    file_path: Optional[str] = None
    suggested_action: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: str
    total_components_checked: int = 0
    total_config_references: int = 0
    successful_validations: int = 0
    violations: List[ValidationViolation] = field(default_factory=list)
    migration_readiness: bool = False
    action_items: List[str] = field(default_factory=list)
    
    @property
    def violation_count(self) -> int:
        return len(self.violations)
    
    @property
    def success_rate(self) -> float:
        if self.total_components_checked == 0:
            return 0.0
        return (self.successful_validations / self.total_components_checked) * 100


class ConfigMigrationValidator:
    """
    Validates configuration migration compliance following Nanobrain architectural patterns.
    
    This validator implements data-driven discovery and validation, avoiding any hardcoded
    assumptions about component locations or configurations.
    """
    
    def __init__(self, nanobrain_root: Optional[Path] = None):
        """
        Initialize validator with configurable root directory.
        
        Args:
            nanobrain_root: Root directory of nanobrain framework
        """
        self.nanobrain_root = nanobrain_root or Path(__file__).parent.parent
        self.violations: List[ValidationViolation] = []
        self.success_count = 0
        self.total_references = 0
        self.discovered_components: Set[Type[FromConfigBase]] = set()
        self.config_files: Set[Path] = set()
        
    def run_comprehensive_validation(self) -> ValidationReport:
        """
        Execute comprehensive validation of configuration migration.
        
        Returns:
            Complete validation report
        """
        print("🔍 Starting Configuration Migration Validation...")
        print(f"   Nanobrain Root: {self.nanobrain_root}")
        
        try:
            # Phase 1: Discovery
            print("\n📋 Phase 1: Component and Configuration Discovery")
            self._discover_framework_components()
            self._discover_config_files()
            
            # Phase 2: Component Validation
            print("\n🔧 Phase 2: Component Configuration Validation")
            self._validate_component_config_organization()
            
            # Phase 3: Configuration Content Validation
            print("\n📝 Phase 3: Configuration Content Validation")
            self._validate_component_configuration_completeness()
            self._validate_config_reference_completeness()
            
            # Phase 4: Workflow Structure Validation
            print("\n🔗 Phase 4: Workflow Structure Validation")
            self._validate_workflow_structure()
            
            # Phase 5: Path Resolution Validation
            print("\n🗂️  Phase 5: Path Resolution Validation")
            self._validate_config_file_references()
            
            # Generate comprehensive report
            print("\n📊 Generating Validation Report...")
            report = self._generate_migration_report()
            
            return report
            
        except Exception as e:
            # Add critical error to violations
            self.violations.append(ValidationViolation(
                component="validation_framework",
                error=f"Critical validation error: {e}",
                violation_type="validation_failure",
                suggested_action="Check validator implementation and nanobrain framework structure"
            ))
            return self._generate_migration_report()
    
    def _discover_framework_components(self) -> None:
        """
        Discover all FromConfigBase subclasses in the framework.
        
        Uses data-driven discovery to avoid hardcoded component lists.
        """
        print("   🔍 Discovering framework components...")
        
        # Search patterns for component modules - fixed to use proper recursive glob
        search_dirs = [
            self.nanobrain_root / "nanobrain" / "core",
            self.nanobrain_root / "nanobrain" / "library",
        ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            # Use proper recursive glob
            module_files = list(search_dir.rglob("*.py"))
            
            for module_file in module_files:
                if module_file.name.startswith('__'):
                    continue
                    
                try:
                    # Convert file path to module path
                    relative_path = module_file.relative_to(self.nanobrain_root)
                    module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
                    
                    # Import and inspect module
                    module = importlib.import_module(module_path)
                    
                    # Find FromConfigBase subclasses
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, FromConfigBase) and 
                            obj != FromConfigBase and 
                            obj.__module__ == module_path):
                            self.discovered_components.add(obj)
                            print(f"      Found component: {obj.__module__}.{obj.__name__}")
                            
                except (ImportError, ModuleNotFoundError, ValueError) as e:
                    # Skip modules that can't be imported (may have dependencies)
                    continue
                except Exception as e:
                    self.violations.append(ValidationViolation(
                        component=f"module_discovery:{module_file}",
                        error=f"Error discovering components in module: {e}",
                        violation_type="discovery_error",
                        suggested_action="Check module structure and imports"
                    ))
        
        print(f"   ✅ Discovered {len(self.discovered_components)} framework components")
    
    def _discover_config_files(self) -> None:
        """
        Discover relevant YAML configuration files in the framework.
        Filters out backup directories, node_modules, and other irrelevant files.
        """
        print("   🔍 Discovering configuration files...")
        
        # Focus on relevant directories only
        relevant_dirs = [
            self.nanobrain_root / "nanobrain",
            self.nanobrain_root / "config",
            self.nanobrain_root / "demo",
        ]
        
        exclude_patterns = [
            "**/node_modules/**",
            "**/backup*/**",
            "**/*backup*/**",
            "**/venv/**",
            "**/.*/**",  # Hidden directories
            "**/cache/**",
            "**/results/**",
            "**/data/**",
        ]
        
        for base_dir in relevant_dirs:
            if not base_dir.exists():
                continue
                
            # Find all YAML files
            for pattern in ["**/*.yml", "**/*.yaml"]:
                config_files = list(base_dir.glob(pattern))
                
                # Filter out excluded patterns
                for config_file in config_files:
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if config_file.match(exclude_pattern):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        self.config_files.add(config_file)
        
        print(f"   ✅ Discovered {len(self.config_files)} configuration files")
    
    def _validate_component_config_organization(self) -> None:
        """
        Validate each component's config organization against new rules.
        """
        print("   🔧 Validating component configuration organization...")
        
        for component_class in self.discovered_components:
            try:
                # Test if component can resolve config files from its directory
                self._validate_component_config_access(component_class)
                self.success_count += 1
                
            except Exception as e:
                self.violations.append(ValidationViolation(
                    component=f"{component_class.__module__}.{component_class.__name__}",
                    error=str(e),
                    violation_type="config_access",
                    suggested_action="Ensure component config files are co-located with component class"
                ))
        
        print(f"   ✅ Validated {self.success_count} components successfully")
    
    def _validate_component_config_access(self, component_class: Type[FromConfigBase]) -> None:
        """
        Test if component can access its config files from class directory.
        
        Args:
            component_class: Component class to test
            
        Raises:
            Exception: If component cannot access expected config files
        """
        try:
            # Get component's file directory
            component_file = inspect.getfile(component_class)
            component_dir = Path(component_file).parent
            
            # Look for component-specific config files
            component_configs = list(component_dir.glob("**/*.yml")) + list(component_dir.glob("**/*.yaml"))
            
            if component_configs:
                # Test path resolution using new method
                for config_file in component_configs:
                    relative_path = config_file.relative_to(component_dir)
                    try:
                        resolved_path = component_class._resolve_config_file_path(relative_path)
                        if not resolved_path.exists():
                            raise FileNotFoundError(f"Resolved path does not exist: {resolved_path}")
                    except Exception as e:
                        raise Exception(f"Path resolution failed for {relative_path}: {e}")
            
        except Exception as e:
            raise Exception(f"Component config access validation failed: {e}")
    
    def _validate_component_configuration_completeness(self) -> None:
        """
        Check if configurations are complete for their components.
        """
        print("   📝 Validating component configuration completeness...")
        
        for config_file in self.config_files:
            try:
                violations = self._validate_component_config_completeness(config_file)
                if violations:
                    for violation in violations:
                        self.violations.append(ValidationViolation(
                            component="config_completeness",
                            error=violation,
                            violation_type="incomplete_component_configuration",
                            file_path=str(config_file),
                            suggested_action="Ensure config contains all required parameters for its component"
                        ))
                        
            except Exception as e:
                self.violations.append(ValidationViolation(
                    component="config_completeness",
                    error=f"Error validating config completeness: {e}",
                    violation_type="validation_error",
                    file_path=str(config_file),
                    suggested_action="Check config file format and accessibility"
                ))
    
    def _validate_component_config_completeness(self, config_file: Path) -> List[str]:
        """
        Validate that a configuration file is complete for its component.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            List of completeness violations
        """
        violations = []
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                violations.append("Configuration file is empty")
                return violations
            
            # Check for required fields
            required_fields = ['name', 'class']
            for field in required_fields:
                if field not in config_data:
                    violations.append(f"Missing required field: {field}")
            
            # If class is specified, check if it's a valid component
            if 'class' in config_data:
                class_path = config_data['class']
                try:
                    # Try to import the class to validate it exists
                    module_path, class_name = class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    component_class = getattr(module, class_name)
                    
                    # Check if it's a FromConfigBase subclass
                    if not issubclass(component_class, FromConfigBase):
                        violations.append(f"Class {class_path} is not a FromConfigBase subclass")
                        
                except (ImportError, AttributeError, ValueError) as e:
                    violations.append(f"Cannot import specified class {class_path}: {e}")
                    
        except yaml.YAMLError as e:
            violations.append(f"YAML parsing error: {e}")
        except Exception as e:
            violations.append(f"Error reading config file: {e}")
            
        return violations
    
    def _validate_config_reference_completeness(self) -> None:
        """
        Check if referenced configurations are complete for their components.
        """
        print("   🔗 Validating configuration reference completeness...")
        
        for config_file in self.config_files:
            try:
                reference_violations = self._validate_config_references_complete(config_file)
                if reference_violations:
                    for violation in reference_violations:
                        self.violations.append(ValidationViolation(
                            component="config_references",
                            error=violation,
                            violation_type="incomplete_reference_configuration",
                            file_path=str(config_file),
                            suggested_action="Ensure referenced config files are complete for their components"
                        ))
                        
            except Exception as e:
                self.violations.append(ValidationViolation(
                    component="config_references",
                    error=f"Error validating config references: {e}",
                    violation_type="validation_error",
                    file_path=str(config_file),
                    suggested_action="Check config file format and reference paths"
                ))
    
    def _validate_config_references_complete(self, config_file: Path) -> List[str]:
        """
        Validate that all config_file references point to complete configurations.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            List of reference completeness violations
        """
        violations = []
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return violations
            
            # Find all config_file references
            config_references = self._find_config_file_references(config_data)
            
            for ref_path in config_references:
                self.total_references += 1
                
                # Resolve reference path relative to current config
                try:
                    if Path(ref_path).is_absolute():
                        referenced_config = Path(ref_path)
                    else:
                        referenced_config = config_file.parent / ref_path
                    
                    if not referenced_config.exists():
                        violations.append(f"Referenced config file does not exist: {ref_path}")
                        continue
                    
                    # Validate referenced config completeness
                    ref_violations = self._validate_component_config_completeness(referenced_config)
                    if ref_violations:
                        violations.append(f"Referenced config {ref_path} is incomplete: {'; '.join(ref_violations)}")
                        
                except Exception as e:
                    violations.append(f"Error validating reference {ref_path}: {e}")
                    
        except yaml.YAMLError as e:
            violations.append(f"YAML parsing error: {e}")
        except Exception as e:
            violations.append(f"Error reading config file: {e}")
            
        return violations
    
    def _find_config_file_references(self, config_data: Any, references: Optional[List[str]] = None) -> List[str]:
        """
        Recursively find all config_file references in configuration data.
        
        Args:
            config_data: Configuration data to search
            references: List to accumulate references (for recursion)
            
        Returns:
            List of config_file reference paths
        """
        if references is None:
            references = []
        
        if isinstance(config_data, dict):
            for key, value in config_data.items():
                if key == 'config_file' and isinstance(value, str):
                    references.append(value)
                else:
                    self._find_config_file_references(value, references)
        elif isinstance(config_data, list):
            for item in config_data:
                self._find_config_file_references(item, references)
        
        return references
    
    def _validate_workflow_structure(self) -> None:
        """
        Validate workflow config directory structure.
        """
        print("   🔗 Validating workflow structure...")
        
        # Find workflow classes
        workflow_classes = [cls for cls in self.discovered_components 
                           if 'workflow' in cls.__name__.lower()]
        
        for workflow_class in workflow_classes:
            try:
                workflow_dir = self._get_workflow_directory(workflow_class)
                if workflow_dir and workflow_dir.exists():
                    violations = self._check_workflow_config_structure(workflow_dir)
                    self.violations.extend(violations)
                    
            except Exception as e:
                self.violations.append(ValidationViolation(
                    component=f"{workflow_class.__module__}.{workflow_class.__name__}",
                    error=f"Error validating workflow structure: {e}",
                    violation_type="workflow_structure",
                    suggested_action="Check workflow directory organization"
                ))
        
        print(f"   ✅ Validated {len(workflow_classes)} workflows")
    
    def _get_workflow_directory(self, workflow_class: Type[FromConfigBase]) -> Optional[Path]:
        """
        Get workflow directory for a workflow class.
        
        Args:
            workflow_class: Workflow class
            
        Returns:
            Path to workflow directory if found
        """
        try:
            workflow_file = inspect.getfile(workflow_class)
            return Path(workflow_file).parent
        except Exception:
            return None
    
    def _check_workflow_config_structure(self, workflow_dir: Path) -> List[ValidationViolation]:
        """
        Check workflow config directory structure.
        
        Args:
            workflow_dir: Workflow directory path
            
        Returns:
            List of structure violations
        """
        violations = []
        
        # Check for config directory
        config_dir = workflow_dir / "config"
        if not config_dir.exists():
            violations.append(ValidationViolation(
                component=f"workflow_structure:{workflow_dir.name}",
                error="Missing config directory",
                violation_type="workflow_structure",
                file_path=str(workflow_dir),
                suggested_action="Create config directory for workflow configurations"
            ))
            return violations
        
        # Check for step-specific subdirectories
        step_dirs = [d for d in config_dir.iterdir() if d.is_dir() and 'Step' in d.name]
        if not step_dirs:
            violations.append(ValidationViolation(
                component=f"workflow_structure:{workflow_dir.name}",
                error="No step-specific config directories found",
                violation_type="workflow_structure",
                file_path=str(config_dir),
                suggested_action="Create step-specific config directories (e.g., DataAcquisitionStep/)"
            ))
        
        return violations
    
    def _validate_config_file_references(self) -> None:
        """
        Check all config_file references in YAML files.
        """
        print("   🗂️  Validating config file references...")
        
        for config_file in self.config_files:
            try:
                violations = self._validate_config_references(config_file)
                self.violations.extend(violations)
                
            except Exception as e:
                self.violations.append(ValidationViolation(
                    component="config_references",
                    error=f"Error validating references: {e}",
                    violation_type="reference_validation_error",
                    file_path=str(config_file),
                    suggested_action="Check config file format and reference syntax"
                ))
        
        print(f"   ✅ Validated {self.total_references} configuration references")
    
    def _validate_config_references(self, config_file: Path) -> List[ValidationViolation]:
        """
        Validate config file references in a configuration file.
        
        Args:
            config_file: Configuration file to validate
            
        Returns:
            List of reference violations
        """
        violations = []
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return violations
            
            # Check all config_file references
            references = self._find_config_file_references(config_data)
            
            for ref_path in references:
                try:
                    # Test path resolution
                    if Path(ref_path).is_absolute():
                        resolved_path = Path(ref_path)
                    else:
                        resolved_path = config_file.parent / ref_path
                    
                    if not resolved_path.exists():
                        violations.append(ValidationViolation(
                            component="config_reference",
                            error=f"Referenced config file not found: {ref_path}",
                            violation_type="missing_reference",
                            file_path=str(config_file),
                            suggested_action=f"Create missing config file or fix reference path: {ref_path}"
                        ))
                        
                except Exception as e:
                    violations.append(ValidationViolation(
                        component="config_reference",
                        error=f"Error resolving reference {ref_path}: {e}",
                        violation_type="reference_resolution_error",
                        file_path=str(config_file),
                        suggested_action="Check reference path format and accessibility"
                    ))
                    
        except yaml.YAMLError as e:
            violations.append(ValidationViolation(
                component="config_parsing",
                error=f"YAML parsing error: {e}",
                violation_type="yaml_error",
                file_path=str(config_file),
                suggested_action="Fix YAML syntax errors"
            ))
        except Exception as e:
            violations.append(ValidationViolation(
                component="config_access",
                error=f"Error reading config file: {e}",
                violation_type="file_access_error",
                file_path=str(config_file),
                suggested_action="Check file permissions and accessibility"
            ))
            
        return violations
    
    def _generate_migration_report(self) -> ValidationReport:
        """
        Generate detailed migration status report.
        
        Returns:
            Comprehensive validation report
        """
        action_items = self._generate_action_items()
        
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_components_checked=len(self.discovered_components),
            total_config_references=self.total_references,
            successful_validations=self.success_count,
            violations=self.violations,
            migration_readiness=len(self.violations) == 0,
            action_items=action_items
        )
        
        return report
    
    def _generate_action_items(self) -> List[str]:
        """
        Generate actionable items based on violations.
        
        Returns:
            List of prioritized action items
        """
        action_items = []
        
        # Group violations by type
        violation_groups = {}
        for violation in self.violations:
            if violation.violation_type not in violation_groups:
                violation_groups[violation.violation_type] = []
            violation_groups[violation.violation_type].append(violation)
        
        # Generate action items based on violation patterns
        for violation_type, violations in violation_groups.items():
            count = len(violations)
            if violation_type == "config_access":
                action_items.append(f"Fix {count} component config access issues - co-locate configs with classes")
            elif violation_type == "incomplete_component_configuration":
                action_items.append(f"Complete {count} component configurations - add missing required fields")
            elif violation_type == "missing_reference":
                action_items.append(f"Resolve {count} missing config file references - create or fix paths")
            elif violation_type == "workflow_structure":
                action_items.append(f"Restructure {count} workflow configurations - follow new directory patterns")
            else:
                action_items.append(f"Address {count} {violation_type} issues")
        
        # Add general recommendations
        if self.total_references > 0:
            action_items.append(f"Validate all {self.total_references} config file references follow new path resolution rules")
        
        if len(self.discovered_components) > 0:
            action_items.append(f"Ensure all {len(self.discovered_components)} components follow component configuration completeness rules")
        
        return action_items


def print_validation_report(report: ValidationReport) -> None:
    """
    Print comprehensive validation report to console.
    
    Args:
        report: Validation report to print
    """
    print("\n" + "="*80)
    print("🔍 CONFIGURATION MIGRATION VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 SUMMARY")
    print(f"   Timestamp: {report.timestamp}")
    print(f"   Components Checked: {report.total_components_checked}")
    print(f"   Config References: {report.total_config_references}")
    print(f"   Successful Validations: {report.successful_validations}")
    print(f"   Violations Found: {report.violation_count}")
    print(f"   Success Rate: {report.success_rate:.1f}%")
    print(f"   Migration Ready: {'✅ YES' if report.migration_readiness else '❌ NO'}")
    
    if report.violations:
        print(f"\n❌ VIOLATIONS ({len(report.violations)})")
        
        # Group violations by type
        violation_groups = {}
        for violation in report.violations:
            if violation.violation_type not in violation_groups:
                violation_groups[violation.violation_type] = []
            violation_groups[violation.violation_type].append(violation)
        
        for violation_type, violations in violation_groups.items():
            print(f"\n   {violation_type.replace('_', ' ').title()} ({len(violations)} issues):")
            for i, violation in enumerate(violations[:3], 1):  # Show first 3 of each type
                print(f"      {i}. {violation.component}")
                print(f"         Error: {violation.error}")
                if violation.file_path:
                    print(f"         File: {violation.file_path}")
                if violation.suggested_action:
                    print(f"         Action: {violation.suggested_action}")
                print()
            
            if len(violations) > 3:
                print(f"      ... and {len(violations) - 3} more {violation_type} issues")
                print()
    
    if report.action_items:
        print(f"\n📋 ACTION ITEMS ({len(report.action_items)})")
        for i, item in enumerate(report.action_items, 1):
            print(f"   {i}. {item}")
    
    print(f"\n🎯 NEXT STEPS")
    if report.migration_readiness:
        print("   ✅ Configuration migration validation PASSED")
        print("   ✅ All components follow new organizational rules")
        print("   ✅ Ready to proceed with full migration implementation")
    else:
        print("   ❌ Configuration migration validation FAILED")
        print("   ❌ Fix violations before proceeding with migration")
        print("   📋 Complete action items listed above")
    
    print("="*80)


def main():
    """
    Main entry point for configuration migration validation.
    """
    print("🚀 Configuration Migration Validation")
    print("   Following Nanobrain architectural patterns and data-driven execution")
    
    # Initialize validator with auto-discovered nanobrain root
    validator = ConfigMigrationValidator()
    
    # Run comprehensive validation
    report = validator.run_comprehensive_validation()
    
    # Print detailed report
    print_validation_report(report)
    
    # Save detailed report to file
    report_file = Path(__file__).parent / f"config_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import json
        
        # Convert report to serializable format
        report_dict = {
            'timestamp': report.timestamp,
            'total_components_checked': report.total_components_checked,
            'total_config_references': report.total_config_references,
            'successful_validations': report.successful_validations,
            'violation_count': report.violation_count,
            'success_rate': report.success_rate,
            'migration_readiness': report.migration_readiness,
            'violations': [
                {
                    'component': v.component,
                    'error': v.error,
                    'violation_type': v.violation_type,
                    'file_path': v.file_path,
                    'suggested_action': v.suggested_action
                }
                for v in report.violations
            ],
            'action_items': report.action_items
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
    except Exception as e:
        print(f"\n⚠️  Could not save detailed report: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if report.migration_readiness else 1)


if __name__ == "__main__":
    main() 