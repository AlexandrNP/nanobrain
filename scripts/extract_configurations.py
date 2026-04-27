#!/usr/bin/env python3
"""
Configuration Extraction Script

Extracts inline step configurations to separate files for modular configuration.
"""

import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ConfigurationExtractor:
    """Extract inline configurations to separate files with full import paths"""
    
    def __init__(self):
        self.extraction_log = []
        self.errors = []
        self.stats = {
            'workflows_processed': 0,
            'workflows_updated': 0,
            'configs_extracted': 0,
            'backups_created': 0
        }
    
    def extract_workflow_configurations(self, workflow_file: Path) -> None:
        """Extract all step configurations from workflow file"""
        print(f"🔄 Extracting configurations from: {workflow_file}")
        
        if not workflow_file.exists():
            self.errors.append(f"File not found: {workflow_file}")
            return
        
        try:
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = workflow_file.with_suffix(f'.yml.backup_extraction_{timestamp}')
            shutil.copy2(workflow_file, backup_file)
            print(f"📁 Created backup: {backup_file}")
            self.stats['backups_created'] += 1
            
            # Load workflow configuration
            with open(workflow_file, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            # Create config directory structure
            config_dir = workflow_file.parent / "config"
            config_dir.mkdir(exist_ok=True)
            (config_dir / "steps").mkdir(exist_ok=True)
            (config_dir / "shared").mkdir(exist_ok=True)
            print(f"📁 Created config directories: {config_dir}")
            
            # Track changes
            changes_made = False
            configs_extracted = 0
            
            # Process each step
            updated_steps = []
            for step in workflow_data.get('steps', []):
                if 'config' in step:
                    # Extract configuration to separate file
                    config_file_path = self._extract_step_config(step, config_dir)
                    
                    # Update step reference (preserve full import path)
                    updated_step = {
                        'step_id': step['step_id'],
                        'class': step['class'],  # Keep full import path
                        'config_file': config_file_path
                    }
                    
                    # Preserve step-level metadata
                    for field in ['name', 'description', 'estimated_time']:
                        if field in step:
                            updated_step[field] = step[field]
                    
                    updated_steps.append(updated_step)
                    changes_made = True
                    configs_extracted += 1
                    print(f"  ✅ Extracted config for: {step['step_id']}")
                else:
                    updated_steps.append(step)
                    print(f"  ℹ️  No inline config for: {step['step_id']}")
            
            # Update workflow file if changes were made
            if changes_made:
                workflow_data['steps'] = updated_steps
                self._save_updated_workflow(workflow_file, workflow_data)
                print(f"✅ Extracted {configs_extracted} configurations from: {workflow_file}")
                self.stats['workflows_updated'] += 1
                self.stats['configs_extracted'] += configs_extracted
            else:
                print(f"ℹ️  No inline configurations found: {workflow_file}")
            
            self.stats['workflows_processed'] += 1
            
        except Exception as e:
            error_msg = f"Error extracting from {workflow_file}: {e}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
    
    def _extract_step_config(self, step: Dict[str, Any], config_dir: Path) -> str:
        """Extract step configuration to separate file"""
        # Extract class name from full import path for file naming
        full_class_path = step['class']
        if '.' in full_class_path:
            class_name = full_class_path.split('.')[-1]
        else:
            class_name = full_class_path
        
        config_data = step['config'].copy()
        
        # Add metadata to configuration
        config_data['name'] = step.get('name', step['step_id'])
        config_data['description'] = step.get('description', f"Configuration for {class_name}")
        if 'estimated_time' in step:
            config_data['estimated_time'] = step['estimated_time']
        
        # Add configuration metadata
        config_data['_metadata'] = {
            'extracted_from': str(step['step_id']),
            'extraction_timestamp': datetime.now().isoformat(),
            'class_path': full_class_path
        }
        
        # Create configuration file
        config_filename = f"{class_name}.yml"
        config_path = config_dir / "steps" / config_filename
        
        # Handle duplicate filenames
        counter = 1
        while config_path.exists():
            config_filename = f"{class_name}_{counter}.yml"
            config_path = config_dir / "steps" / config_filename
            counter += 1
        
        # Save configuration
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        # Log the extraction
        self.extraction_log.append(f"{step['step_id']} → {config_filename}")
        
        # Return relative path from workflow directory
        return f"config/steps/{config_filename}"
    
    def _save_updated_workflow(self, workflow_file: Path, workflow_data: Dict[str, Any]) -> None:
        """Save updated workflow file with external config references"""
        # Add metadata about extraction
        if '_metadata' not in workflow_data:
            workflow_data['_metadata'] = {}
        
        workflow_data['_metadata']['config_extraction'] = {
            'extracted': True,
            'extraction_timestamp': datetime.now().isoformat(),
            'pattern': 'modular_configuration_v1'
        }
        
        with open(workflow_file, 'w') as f:
            yaml.dump(workflow_data, f, default_flow_style=False, sort_keys=False)
    
    def extract_all_workflows(self) -> None:
        """Extract configurations from all workflow files"""
        print("🚀 Starting configuration extraction for all workflows...")
        print("=" * 60)
        
        # Find all workflow files with inline configurations
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
                    if self._has_inline_configs(yaml_file):
                        workflow_files.append(yaml_file)
                for yaml_file in base_path.rglob("*.yaml"):
                    if self._has_inline_configs(yaml_file):
                        workflow_files.append(yaml_file)
        
        print(f"Found {len(workflow_files)} workflow files with inline configurations")
        print("-" * 60)
        
        for workflow_file in workflow_files:
            self.extract_workflow_configurations(workflow_file)
            print()  # Empty line for readability
        
        self.print_extraction_summary()
    
    def _has_inline_configs(self, yaml_file: Path) -> bool:
        """Check if YAML file has steps with inline configurations"""
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, dict) or 'steps' not in data:
                return False
            
            for step in data['steps']:
                if isinstance(step, dict) and 'config' in step:
                    return True
            
            return False
        except Exception:
            return False
    
    def extract_specific_files(self, file_paths: List[str]) -> None:
        """Extract configurations from specific workflow files"""
        print(f"🚀 Starting configuration extraction for {len(file_paths)} specific files...")
        print("=" * 60)
        
        for file_path in file_paths:
            workflow_file = Path(file_path)
            self.extract_workflow_configurations(workflow_file)
            print()  # Empty line for readability
        
        self.print_extraction_summary()
    
    def print_extraction_summary(self) -> None:
        """Print extraction summary"""
        print("=" * 60)
        print("📊 CONFIGURATION EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Workflows processed: {self.stats['workflows_processed']}")
        print(f"Workflows updated: {self.stats['workflows_updated']}")
        print(f"Configurations extracted: {self.stats['configs_extracted']}")
        print(f"Backups created: {self.stats['backups_created']}")
        
        if self.errors:
            print(f"\n❌ Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.extraction_log:
            print(f"\n✅ Successful extractions: {len(self.extraction_log)}")
            for log_entry in self.extraction_log:
                print(f"  - {log_entry}")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Review extracted configuration files")
        print("2. Test workflows to ensure they load external configs correctly")
        print("3. Optimize configuration files for reusability")
        print("4. Create shared configuration templates")


def main():
    """Main extraction function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract inline configurations to separate files")
    parser.add_argument("--all", action="store_true", help="Extract from all workflow files")
    parser.add_argument("--files", nargs="+", help="Specific files to extract from")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be extracted without making changes")
    
    args = parser.parse_args()
    
    extractor = ConfigurationExtractor()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("=" * 60)
        # TODO: Implement dry run logic
        return
    
    if args.all:
        extractor.extract_all_workflows()
    elif args.files:
        extractor.extract_specific_files(args.files)
    else:
        # Default: extract from example workflow
        key_files = [
            "config/example_workflow.yaml"
        ]
        extractor.extract_specific_files(key_files)


if __name__ == "__main__":
    main() 