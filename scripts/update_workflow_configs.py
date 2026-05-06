#!/usr/bin/env python3
"""
Workflow Configuration Updater

Updates existing workflow configurations to use external default configuration 
files and the modular configuration pattern with full import paths.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowConfigUpdater:
    """Updates workflow configurations to use modular pattern"""
    
    def __init__(self, registry_file: str = "config_registry.json"):
        self.registry = self._load_registry(registry_file)
        self.component_to_config = self._build_component_config_mapping()
        self.updated_files = []
        self.backup_suffix = f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _load_registry(self, registry_file: str) -> Dict[str, Any]:
        """Load configuration registry"""
        with open(registry_file, 'r') as f:
            return json.load(f)
    
    def _build_component_config_mapping(self) -> Dict[str, str]:
        """Build mapping from component class to config file"""
        mapping = {}
        for component, config_info in self.registry["components"].items():
            mapping[component] = config_info["config_file"]
        return mapping
    
    def update_workflow_file(self, workflow_file: Path) -> bool:
        """Update a single workflow file to use modular configuration"""
        logger.info(f"🔄 Updating workflow: {workflow_file}")
        
        if not workflow_file.exists():
            logger.error(f"❌ Workflow file not found: {workflow_file}")
            return False
        
        try:
            # Create backup
            backup_file = workflow_file.with_suffix(f"{workflow_file.suffix}{self.backup_suffix}")
            workflow_file.rename(backup_file)
            logger.info(f"📁 Created backup: {backup_file}")
            
            # Load workflow data
            with open(backup_file, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            if not workflow_data:
                logger.warning(f"⚠️  Empty workflow file: {workflow_file}")
                # Restore from backup
                backup_file.rename(workflow_file)
                return False
            
            # Handle different workflow data types
            if not isinstance(workflow_data, dict):
                logger.warning(f"⚠️  Workflow data is not a dictionary, skipping: {workflow_file}")
                # Restore from backup
                backup_file.rename(workflow_file)
                return False
            
            # Update workflow configuration
            updated_data = self._update_workflow_data(workflow_data, workflow_file)
            
            # Save updated workflow
            with open(workflow_file, 'w') as f:
                yaml.dump(updated_data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"✅ Successfully updated: {workflow_file}")
            self.updated_files.append(str(workflow_file))
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating {workflow_file}: {e}")
            # Restore backup if it exists
            if backup_file.exists():
                backup_file.rename(workflow_file)
            return False
    
    def _update_workflow_data(self, workflow_data: Dict[str, Any], workflow_file: Path) -> Dict[str, Any]:
        """Update workflow data structure"""
        if not isinstance(workflow_data, dict):
            return workflow_data
        
        updated_data = workflow_data.copy()
        
        # Update steps if present - handle different step formats
        if 'steps' in updated_data:
            if isinstance(updated_data['steps'], list):
                # List format: standard workflow steps
                updated_steps = []
                for step in updated_data['steps']:
                    if isinstance(step, dict):
                        updated_step = self._update_step_configuration(step, workflow_file)
                        updated_steps.append(updated_step)
                    else:
                        updated_steps.append(step)  # Keep non-dict steps as-is
                updated_data['steps'] = updated_steps
            elif isinstance(updated_data['steps'], dict):
                # Dict format: step_id -> step_config mapping
                updated_steps = {}
                for step_id, step_config in updated_data['steps'].items():
                    if isinstance(step_config, dict):
                        # Add step_id to step config for processing
                        step_with_id = step_config.copy()
                        step_with_id['step_id'] = step_id
                        updated_step = self._update_step_configuration(step_with_id, workflow_file)
                        # Remove step_id if it was added
                        if 'step_id' in updated_step and step_id == updated_step['step_id']:
                            del updated_step['step_id']
                        updated_steps[step_id] = updated_step
                    else:
                        updated_steps[step_id] = step_config  # Keep non-dict configs as-is
                updated_data['steps'] = updated_steps
        
        # Add metadata about the update
        if '_metadata' not in updated_data:
            updated_data['_metadata'] = {}
        elif not isinstance(updated_data['_metadata'], dict):
            updated_data['_metadata'] = {}
        
        updated_data['_metadata'].update({
            'last_updated': datetime.now().isoformat(),
            'updated_by': 'modular_config_migration',
            'migration_version': '1.0.0',
            'uses_modular_config': True
        })
        
        return updated_data
    
    def _update_step_configuration(self, step: Dict[str, Any], workflow_file: Path) -> Dict[str, Any]:
        """Update individual step configuration"""
        if not isinstance(step, dict):
            return step
        
        updated_step = step.copy()
        
        # Check if step has a class field
        if 'class' not in step:
            logger.warning(f"⚠️  Step without class field: {step.get('step_id', 'unknown')}")
            return updated_step
        
        step_class = step['class']
        
        # If step already has config_file, keep it
        if 'config_file' in step:
            logger.info(f"  Step '{step.get('step_id')}' already uses external config")
            return updated_step
        
        # Check if we have a default config for this component
        if step_class in self.component_to_config:
            default_config_file = self.component_to_config[step_class]
            
            # If step has inline config, we'll move it to external file
            if 'config' in step and isinstance(step['config'], dict):
                # Create step-specific config file
                step_config_file = self._create_step_specific_config(
                    step, default_config_file, workflow_file
                )
                if step_config_file:
                    updated_step['config_file'] = step_config_file
                    # Remove inline config
                    del updated_step['config']
                    logger.info(f"  ✅ Moved inline config to: {step_config_file}")
            else:
                # Use default configuration
                updated_step['config_file'] = default_config_file
                logger.info(f"  ✅ Added default config: {default_config_file}")
        else:
            logger.warning(f"  ⚠️  No default config found for class: {step_class}")
        
        return updated_step
    
    def _create_step_specific_config(self, step: Dict[str, Any], 
                                   default_config_file: str, 
                                   workflow_file: Path) -> Union[str, None]:
        """Create step-specific configuration file"""
        try:
            # Load default configuration
            with open(default_config_file, 'r') as f:
                default_config = yaml.safe_load(f)
            
            if not isinstance(default_config, dict):
                logger.warning(f"⚠️  Default config is not a dict: {default_config_file}")
                return None
            
            # Merge with step-specific config
            step_config = step.get('config', {})
            if not isinstance(step_config, dict):
                logger.warning(f"⚠️  Step config is not a dict for step: {step.get('step_id')}")
                return None
            
            merged_config = self._merge_configs(default_config, step_config)
            
            # Update merged config metadata
            merged_config['name'] = step.get('name', step.get('step_id', 'Unnamed Step'))
            merged_config['description'] = step.get('description', merged_config.get('description', ''))
            
            if '_metadata' not in merged_config:
                merged_config['_metadata'] = {}
            elif not isinstance(merged_config['_metadata'], dict):
                merged_config['_metadata'] = {}
            
            merged_config['_metadata'].update({
                'source_workflow': str(workflow_file),
                'step_id': step.get('step_id'),
                'created_from_inline': True,
                'based_on_default': default_config_file,
                'created_date': datetime.now().isoformat()
            })
            
            # Create step-specific config file path
            workflow_dir = workflow_file.parent
            config_dir = workflow_dir / "config" / "steps"
            config_dir.mkdir(parents=True, exist_ok=True)
            
            step_id = step.get('step_id', 'unknown_step')
            config_filename = f"{step_id}.yml"
            step_config_path = config_dir / config_filename
            
            # Save step-specific configuration
            with open(step_config_path, 'w') as f:
                yaml.dump(merged_config, f, default_flow_style=False, sort_keys=False)
            
            # Return relative path from workflow file
            relative_path = f"config/steps/{config_filename}"
            return relative_path
            
        except Exception as e:
            logger.error(f"❌ Error creating step-specific config: {e}")
            return None
    
    def _merge_configs(self, default_config: Dict[str, Any], 
                      step_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge step-specific config with default config"""
        if not isinstance(default_config, dict) or not isinstance(step_config, dict):
            return default_config if isinstance(default_config, dict) else {}
        
        merged = default_config.copy()
        
        # Deep merge step config into default config
        def deep_merge(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged, step_config)
        return merged
    
    def update_all_workflows(self) -> bool:
        """Update all known workflow files"""
        logger.info("🚀 Starting workflow configuration update...")
        logger.info("=" * 50)
        
        # Known workflow files
        workflow_files = [
            Path("nanobrain/library/workflows/chat_workflow_parsl/ParslChatWorkflow.yml"),
            Path("nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"),
            Path("nanobrain/library/workflows/viral_protein_analysis/config/CleanWorkflow.yml"),
            Path("nanobrain/library/workflows/chat_workflow/chat_workflow.yml"),
            Path("nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml"),
            Path("demo/config/chat_workflow.yml"),
            Path("config/example_workflow.yaml")
        ]
        
        # Also find any other YAML files that might be workflows
        additional_workflows = []
        search_patterns = ["**/*workflow*.yml", "**/*workflow*.yaml", "**/workflows/**/*.yml", "**/workflows/**/*.yaml"]
        for pattern in search_patterns:
            try:
                additional_workflows.extend(Path(".").glob(pattern))
            except Exception as e:
                logger.warning(f"⚠️  Error searching with pattern {pattern}: {e}")
        
        # Remove duplicates and filter out already known files
        known_paths = {f.resolve() for f in workflow_files if f.exists()}
        additional_workflows = [f for f in additional_workflows 
                              if f.resolve() not in known_paths and f.exists() and f.is_file()]
        
        # Filter existing files only
        existing_workflow_files = [f for f in workflow_files if f.exists()]
        all_workflows = existing_workflow_files + additional_workflows
        
        logger.info(f"📋 Found {len(all_workflows)} workflow files to update")
        
        success_count = 0
        for workflow_file in all_workflows:
            if self.update_workflow_file(workflow_file):
                success_count += 1
        
        logger.info(f"\n📊 Update Summary:")
        logger.info(f"  Total files: {len(all_workflows)}")
        logger.info(f"  Successfully updated: {success_count}")
        logger.info(f"  Failed: {len(all_workflows) - success_count}")
        
        if success_count == len(all_workflows):
            logger.info("🎉 All workflow files updated successfully!")
            return True
        else:
            logger.warning("⚠️  Some workflow files failed to update")
            return False
    
    def generate_update_report(self) -> Dict[str, Any]:
        """Generate update report"""
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "updated_files_count": len(self.updated_files),
                "migration_version": "1.0.0"
            },
            "updated_files": self.updated_files,
            "component_mappings": self.component_to_config,
            "registry_info": {
                "total_components": len(self.registry["components"]),
                "registry_version": self.registry["metadata"]["version"]
            }
        }
        
        with open("workflow_update_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📄 Update report saved to: workflow_update_report.json")
        return report
    
    def validate_updates(self) -> bool:
        """Validate that updated workflows are valid"""
        logger.info("🔍 Validating updated workflows...")
        
        validation_errors = []
        for workflow_file in self.updated_files:
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                
                # Basic validation
                if not workflow_data:
                    validation_errors.append(f"{workflow_file}: Empty workflow")
                    continue
                
                if not isinstance(workflow_data, dict):
                    validation_errors.append(f"{workflow_file}: Workflow data is not a dictionary")
                    continue
                
                # Check that steps have either config or config_file (if steps exist)
                if 'steps' in workflow_data:
                    steps = workflow_data['steps']
                    if isinstance(steps, list):
                        for step in steps:
                            if isinstance(step, dict):
                                step_id = step.get('step_id', 'unknown')
                                if 'config' not in step and 'config_file' not in step:
                                    validation_errors.append(
                                        f"{workflow_file}: Step '{step_id}' has no config or config_file"
                                    )
                    elif isinstance(steps, dict):
                        for step_id, step_config in steps.items():
                            if isinstance(step_config, dict):
                                if 'config' not in step_config and 'config_file' not in step_config:
                                    validation_errors.append(
                                        f"{workflow_file}: Step '{step_id}' has no config or config_file"
                                    )
                
                logger.info(f"  ✅ {workflow_file}: Validation passed")
                
            except Exception as e:
                validation_errors.append(f"{workflow_file}: Validation error - {e}")
                logger.error(f"  ❌ {workflow_file}: Validation error - {e}")
        
        if validation_errors:
            logger.error("❌ Validation errors found:")
            for error in validation_errors:
                logger.error(f"  {error}")
            return False
        
        logger.info("✅ All updated workflows validated successfully")
        return True


def main():
    """Main function"""
    try:
        updater = WorkflowConfigUpdater()
        
        # Update all workflows
        success = updater.update_all_workflows()
        
        # Generate report
        updater.generate_update_report()
        
        # Validate updates
        validation_success = updater.validate_updates()
        
        overall_success = success and validation_success
        
        if overall_success:
            logger.info("\n🎉 WORKFLOW UPDATE COMPLETED SUCCESSFULLY!")
            logger.info("✅ All workflows now use modular configuration pattern")
        else:
            logger.error("\n❌ WORKFLOW UPDATE COMPLETED WITH ISSUES")
            logger.error("🔧 Please review the errors and fix manually")
        
        exit(0 if overall_success else 1)
        
    except Exception as e:
        logger.error(f"❌ Workflow update failed: {e}")
        exit(1)


if __name__ == "__main__":
    main() 