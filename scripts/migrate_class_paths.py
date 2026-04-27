#!/usr/bin/env python3
"""
Class Path Migration Script

Migrates all workflow configurations from short class names and built-in types
to full import paths, completely removing backward compatibility.
"""

import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ClassPathMigrator:
    """Migrate short class names to full import paths"""
    
    CLASS_PATH_MAPPING = {
        # Chatbot Viral Integration Steps
        'QueryClassificationStep': 'nanobrain.library.workflows.chatbot_viral_integration.steps.query_classification_step.QueryClassificationStep',
        'AnnotationJobStep': 'nanobrain.library.workflows.chatbot_viral_integration.steps.annotation_job_step.AnnotationJobStep',
        'ConversationalResponseStep': 'nanobrain.library.workflows.chatbot_viral_integration.steps.conversational_response_step.ConversationalResponseStep',
        'ResponseFormattingStep': 'nanobrain.library.workflows.chatbot_viral_integration.steps.response_formatting_step.ResponseFormattingStep',
        
        # Viral Protein Analysis Steps
        'BVBRCDataAcquisitionStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep',
        'SequenceCurationStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.sequence_curation_step.SequenceCurationStep',
        'ClusteringStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.clustering_step.ClusteringStep',
        'AnnotationMappingStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.annotation_mapping_step.AnnotationMappingStep',
        'PSSMCreationStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.pssm_creation_step.PSSMCreationStep',
        'PSSMAnalysisStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.pssm_analysis_step.PSSMAnalysisStep',
        'ReportGenerationStep': 'nanobrain.library.workflows.viral_protein_analysis.steps.report_generation_step.ReportGenerationStep',
        
        # Core Framework Steps (migrate to full paths)
        'Step': 'nanobrain.core.step.Step',
        'SimpleStep': 'nanobrain.core.step.Step',  # Legacy mapping to Step
        'TransformStep': 'nanobrain.core.step.TransformStep',
        'Workflow': 'nanobrain.core.workflow.Workflow',
        
        # Built-in type aliases (WILL BE REMOVED - migration only)
        'simple': 'nanobrain.core.step.Step',
        'step': 'nanobrain.core.step.Step', 
        'transform': 'nanobrain.core.step.TransformStep',
        'workflow': 'nanobrain.core.workflow.Workflow'
    }
    
    def __init__(self):
        self.migration_log = []
        self.errors = []
        self.stats = {
            'files_processed': 0,
            'files_migrated': 0,
            'steps_updated': 0,
            'backups_created': 0
        }
    
    def migrate_workflow_file(self, workflow_file: Path) -> None:
        """Migrate a single workflow file to use full import paths"""
        print(f"🔄 Migrating: {workflow_file}")
        
        if not workflow_file.exists():
            self.errors.append(f"File not found: {workflow_file}")
            return
        
        try:
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = workflow_file.with_suffix(f'.yml.backup_{timestamp}')
            shutil.copy2(workflow_file, backup_file)
            print(f"📁 Created backup: {backup_file}")
            self.stats['backups_created'] += 1
            
            # Load workflow configuration
            with open(workflow_file, 'r') as f:
                workflow_data = yaml.safe_load(f)
            
            # Track changes
            changes_made = False
            steps_updated = 0
            
            # Update step class paths
            if 'steps' in workflow_data:
                for step in workflow_data['steps']:
                    if 'class' in step:
                        old_class = step['class']
                        if old_class in self.CLASS_PATH_MAPPING:
                            new_class = self.CLASS_PATH_MAPPING[old_class]
                            step['class'] = new_class
                            print(f"  ✅ Updated: {old_class} → {new_class}")
                            self.migration_log.append(f"{workflow_file}: {old_class} → {new_class}")
                            changes_made = True
                            steps_updated += 1
                        elif '.' not in old_class:
                            # Unmapped short name - this will cause an error later
                            print(f"  ⚠️  Unknown short class name: {old_class} (will need manual migration)")
                            self.errors.append(f"{workflow_file}: Unknown short class '{old_class}'")
                        else:
                            print(f"  ✓ Already full path: {old_class}")
            
            # Save updated workflow if changes were made
            if changes_made:
                with open(workflow_file, 'w') as f:
                    yaml.dump(workflow_data, f, default_flow_style=False, sort_keys=False)
                print(f"✅ Migrated {steps_updated} steps in: {workflow_file}")
                self.stats['files_migrated'] += 1
                self.stats['steps_updated'] += steps_updated
            else:
                print(f"ℹ️  No changes needed: {workflow_file}")
            
            self.stats['files_processed'] += 1
            
        except Exception as e:
            error_msg = f"Error migrating {workflow_file}: {e}"
            print(f"❌ {error_msg}")
            self.errors.append(error_msg)
    
    def migrate_all_workflows(self) -> None:
        """Migrate all workflow files in the framework"""
        print("🚀 Starting migration of all workflow files...")
        print("=" * 60)
        
        # Find all workflow files
        workflow_files = []
        
        # Core workflow directories
        base_paths = [
            Path("nanobrain/library/workflows"),
            Path("config"),
            Path("demo/config"),
        ]
        
        for base_path in base_paths:
            if base_path.exists():
                # Find all YAML files recursively
                for yaml_file in base_path.rglob("*.yml"):
                    workflow_files.append(yaml_file)
                for yaml_file in base_path.rglob("*.yaml"):
                    workflow_files.append(yaml_file)
        
        print(f"Found {len(workflow_files)} configuration files to check")
        print("-" * 60)
        
        for workflow_file in workflow_files:
            self.migrate_workflow_file(workflow_file)
            print()  # Empty line for readability
        
        self.print_migration_summary()
    
    def migrate_specific_files(self, file_paths: List[str]) -> None:
        """Migrate specific workflow files"""
        print(f"🚀 Starting migration of {len(file_paths)} specific files...")
        print("=" * 60)
        
        for file_path in file_paths:
            workflow_file = Path(file_path)
            self.migrate_workflow_file(workflow_file)
            print()  # Empty line for readability
        
        self.print_migration_summary()
    
    def print_migration_summary(self) -> None:
        """Print migration summary"""
        print("=" * 60)
        print("📊 MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files migrated: {self.stats['files_migrated']}")
        print(f"Steps updated: {self.stats['steps_updated']}")
        print(f"Backups created: {self.stats['backups_created']}")
        
        if self.errors:
            print(f"\n❌ Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.migration_log:
            print(f"\n✅ Successful migrations: {len(self.migration_log)}")
            for log_entry in self.migration_log:
                print(f"  - {log_entry}")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Review any errors and manually migrate unknown class names")
        print("2. Test workflows to ensure they work with full import paths")
        print("3. Remove any remaining references to built-in types in code")
        print("4. Run validation script to verify migration completeness")


def main():
    """Main migration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate workflow configurations to full import paths")
    parser.add_argument("--all", action="store_true", help="Migrate all workflow files")
    parser.add_argument("--files", nargs="+", help="Specific files to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    
    args = parser.parse_args()
    
    migrator = ClassPathMigrator()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print("=" * 60)
        # TODO: Implement dry run logic
        return
    
    if args.all:
        migrator.migrate_all_workflows()
    elif args.files:
        migrator.migrate_specific_files(args.files)
    else:
        # Default: migrate key workflow files
        key_files = [
            "nanobrain/library/workflows/chatbot_viral_integration/ChatbotViralWorkflow.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml",
            "nanobrain/library/workflows/viral_protein_analysis/config/CleanWorkflow.yml",
            "nanobrain/library/workflows/chat_workflow/chat_workflow.yml",
            "nanobrain/library/workflows/chat_workflow_parsl/ParslChatWorkflow.yml",
            "config/example_workflow.yaml"
        ]
        migrator.migrate_specific_files(key_files)


if __name__ == "__main__":
    main() 