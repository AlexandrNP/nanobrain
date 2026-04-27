#!/usr/bin/env python3
"""
Import Path Validation Script

Validates that all workflow configurations use only full import paths
and that all referenced classes implement the from_config pattern.
"""

import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


class ImportPathValidator:
    """Validate component import paths and configurations"""
    
    def __init__(self):
        self.validation_log = []
        self.errors = []
        self.warnings = []
        self.stats = {
            'files_processed': 0,
            'steps_validated': 0,
            'full_paths_found': 0,
            'short_names_found': 0,
            'from_config_validated': 0,
            'from_config_missing': 0
        }
    
    def validate_class_import_path(self, class_path: str) -> List[str]:
        """Validate that a class import path is valid"""
        issues = []
        
        if not class_path:
            issues.append("Class path cannot be empty")
            return issues
        
        # ENFORCE full import path format - NO EXCEPTIONS
        if '.' not in class_path:
            issues.append(f"Class path must be full import path: {class_path}. Short names and built-in types are no longer supported.")
            self.stats['short_names_found'] += 1
            return issues
        
        self.stats['full_paths_found'] += 1
        
        try:
            # Attempt to import the class
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            
            if not hasattr(module, class_name):
                issues.append(f"Class '{class_name}' not found in module '{module_path}'")
                return issues
            
            class_obj = getattr(module, class_name)
            
            # Validate from_config implementation
            if not hasattr(class_obj, 'from_config'):
                issues.append(f"Class '{class_path}' must implement from_config method")
                self.stats['from_config_missing'] += 1
            else:
                self.stats['from_config_validated'] += 1
            
        except ImportError as e:
            issues.append(f"Cannot import module '{module_path}': {e}")
        except Exception as e:
            issues.append(f"Error validating class path '{class_path}': {e}")
        
        return issues
    
    def validate_workflow_file(self, workflow_file: Path) -> List[str]:
        """Validate all class paths in a workflow file"""
        print(f"🔍 Validating: {workflow_file}")
        file_issues = []
        
        try:
            with open(workflow_file, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            if not isinstance(workflow_data, dict):
                file_issues.append("Invalid YAML structure")
                return file_issues
            
            # Check if this is a workflow file
            if 'steps' not in workflow_data:
                print(f"  ℹ️  Skipping non-workflow file: {workflow_file}")
                return []
            
            steps_in_file = 0
            valid_steps = 0
            
            for step in workflow_data.get('steps', []):
                if not isinstance(step, dict):
                    file_issues.append(f"Invalid step structure: {step}")
                    continue
                
                step_id = step.get('step_id', 'unknown')
                class_path = step.get('class')
                steps_in_file += 1
                
                if not class_path:
                    file_issues.append(f"Step '{step_id}': Missing class field")
                    continue
                
                step_issues = self.validate_class_import_path(class_path)
                if step_issues:
                    for issue in step_issues:
                        file_issues.append(f"Step '{step_id}': {issue}")
                else:
                    valid_steps += 1
                    print(f"  ✅ Valid: {step_id} → {class_path}")
            
            print(f"  📊 Summary: {valid_steps}/{steps_in_file} steps valid")
            self.stats['steps_validated'] += steps_in_file
            
        except Exception as e:
            file_issues.append(f"Error validating workflow file: {e}")
        
        return file_issues
    
    def validate_all_workflows(self) -> None:
        """Validate all workflow files in the framework"""
        print("🚀 Starting validation of all workflow files...")
        print("=" * 60)
        
        # Find all workflow files
        workflow_files = []
        
        # Core workflow directories
        base_paths = [
            Path("nanobrain/library/workflows"),
            Path("demo/config"),
            Path("config"),
        ]
        
        for base_path in base_paths:
            if base_path.exists():
                # Find all YAML files recursively
                for yaml_file in base_path.rglob("*.yml"):
                    workflow_files.append(yaml_file)
                for yaml_file in base_path.rglob("*.yaml"):
                    workflow_files.append(yaml_file)
        
        print(f"Found {len(workflow_files)} configuration files to validate")
        print("-" * 60)
        
        for workflow_file in workflow_files:
            file_issues = self.validate_workflow_file(workflow_file)
            if file_issues:
                self.errors.extend([f"{workflow_file}: {issue}" for issue in file_issues])
            self.stats['files_processed'] += 1
            print()  # Empty line for readability
        
        self.print_validation_summary()
    
    def validate_specific_files(self, file_paths: List[str]) -> None:
        """Validate specific workflow files"""
        print(f"🚀 Starting validation of {len(file_paths)} specific files...")
        print("=" * 60)
        
        for file_path in file_paths:
            workflow_file = Path(file_path)
            if workflow_file.exists():
                file_issues = self.validate_workflow_file(workflow_file)
                if file_issues:
                    self.errors.extend([f"{workflow_file}: {issue}" for issue in file_issues])
                self.stats['files_processed'] += 1
            else:
                self.errors.append(f"File not found: {workflow_file}")
            print()  # Empty line for readability
        
        self.print_validation_summary()
    
    def generate_validation_report(self) -> str:
        """Generate detailed validation report"""
        report_lines = [
            "# Import Path Validation Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary Statistics",
            f"- Files processed: {self.stats['files_processed']}",
            f"- Steps validated: {self.stats['steps_validated']}",
            f"- Full import paths: {self.stats['full_paths_found']}",
            f"- Short names found: {self.stats['short_names_found']}",
            f"- from_config implemented: {self.stats['from_config_validated']}",
            f"- from_config missing: {self.stats['from_config_missing']}",
            "",
        ]
        
        if self.stats['short_names_found'] == 0 and self.stats['from_config_missing'] == 0:
            report_lines.extend([
                "## ✅ VALIDATION PASSED",
                "All workflow configurations use full import paths and implement from_config pattern.",
                ""
            ])
        else:
            report_lines.extend([
                "## ❌ VALIDATION FAILED",
                "Issues found that need to be resolved:",
                ""
            ])
        
        if self.errors:
            report_lines.extend([
                "## Errors",
                ""
            ])
            for error in self.errors:
                report_lines.append(f"- {error}")
            report_lines.append("")
        
        if self.warnings:
            report_lines.extend([
                "## Warnings",
                ""
            ])
            for warning in self.warnings:
                report_lines.append(f"- {warning}")
            report_lines.append("")
        
        report_lines.extend([
            "## Recommendations",
            "",
            "### For Short Class Names",
            "Update to full import paths using the migration script:",
            "```bash",
            "python scripts/migrate_class_paths.py --all",
            "```",
            "",
            "### For Missing from_config",
            "Update classes to implement the from_config pattern:",
            "```python",
            "@classmethod",
            "def from_config(cls, config: StepConfig, **kwargs) -> 'YourClass':",
            "    # Implementation here",
            "    pass",
            "```",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def print_validation_summary(self) -> None:
        """Print validation summary"""
        print("=" * 60)
        print("📊 IMPORT PATH VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Steps validated: {self.stats['steps_validated']}")
        print(f"Full import paths: {self.stats['full_paths_found']}")
        print(f"Short names found: {self.stats['short_names_found']}")
        print(f"from_config implemented: {self.stats['from_config_validated']}")
        print(f"from_config missing: {self.stats['from_config_missing']}")
        
        # Overall status
        if self.stats['short_names_found'] == 0 and self.stats['from_config_missing'] == 0:
            print("\n✅ VALIDATION PASSED")
            print("All configurations use full import paths and implement from_config pattern.")
        else:
            print("\n❌ VALIDATION FAILED")
            print("Issues found that need to be resolved.")
        
        if self.errors:
            print(f"\n❌ Errors: {len(self.errors)}")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
        
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:  # Show first 5 warnings
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more warnings")
        
        print("\n🎯 NEXT STEPS:")
        if self.stats['short_names_found'] > 0:
            print("1. Run migration script to update short class names to full import paths")
        if self.stats['from_config_missing'] > 0:
            print("2. Update classes to implement from_config pattern")
        print("3. Re-run validation to verify all issues are resolved")
        print("4. Run comprehensive tests to ensure functionality")


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate import paths in workflow configurations")
    parser.add_argument("--all", action="store_true", help="Validate all workflow files")
    parser.add_argument("--files", nargs="+", help="Specific files to validate")
    parser.add_argument("--report", type=str, help="Generate detailed report to file")
    
    args = parser.parse_args()
    
    validator = ImportPathValidator()
    
    if args.all:
        validator.validate_all_workflows()
    elif args.files:
        validator.validate_specific_files(args.files)
    else:
        # Default: validate key workflow files
        key_files = [
            "config/example_workflow.yaml",
            "nanobrain/library/workflows/chat_workflow_parsl/ParslChatWorkflow.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
        ]
        validator.validate_specific_files(key_files)
    
    # Generate report if requested
    if args.report:
        report_content = validator.generate_validation_report()
        with open(args.report, 'w') as f:
            f.write(report_content)
        print(f"\n📄 Detailed report saved to: {args.report}")


if __name__ == "__main__":
    main() 