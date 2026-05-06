#!/usr/bin/env python3
"""
Configuration Registry Creator

Creates a comprehensive registry mapping all from_config components 
to their default configuration files for the modular configuration system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List


class ConfigurationRegistry:
    """Registry for mapping components to their default configurations"""
    
    def __init__(self):
        self.registry = {}
        self.config_base_paths = [
            Path("nanobrain/core/config/defaults"),
            Path("nanobrain/library/config/defaults")
        ]
    
    def build_component_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive component to configuration mapping"""
        
        registry = {
            # Core Framework Components
            "nanobrain.core.step.Step": {
                "config_file": "nanobrain/core/config/defaults/step.yml",
                "config_type": "StepConfig",
                "category": "core",
                "description": "Base step component configuration"
            },
            
            "nanobrain.core.step.TransformStep": {
                "config_file": "nanobrain/core/config/defaults/step.yml",
                "config_type": "StepConfig", 
                "category": "core",
                "description": "Transform step component configuration"
            },
            
            # Executor Components
            "nanobrain.core.executor.LocalExecutor": {
                "config_file": "nanobrain/core/config/defaults/executor.yml",
                "config_type": "ExecutorConfig",
                "category": "core",
                "description": "Local executor configuration"
            },
            
            "nanobrain.core.executor.ThreadExecutor": {
                "config_file": "nanobrain/core/config/defaults/executor.yml",
                "config_type": "ExecutorConfig",
                "category": "core",
                "description": "Thread-based executor configuration"
            },
            
            "nanobrain.core.executor.ProcessExecutor": {
                "config_file": "nanobrain/core/config/defaults/executor.yml",
                "config_type": "ExecutorConfig",
                "category": "core",
                "description": "Process-based executor configuration"
            },
            
            "nanobrain.core.executor.ParslExecutor": {
                "config_file": "nanobrain/core/config/defaults/executor.yml",
                "config_type": "ExecutorConfig",
                "category": "core",
                "description": "Parsl-based executor configuration"
            },
            
            # Data Unit Components
            "nanobrain.core.data_unit.DataUnitMemory": {
                "config_file": "nanobrain/core/config/defaults/data_unit.yml",
                "config_type": "DataUnitConfig",
                "category": "core",
                "description": "Memory-based data unit configuration"
            },
            
            "nanobrain.core.data_unit.DataUnitFile": {
                "config_file": "nanobrain/core/config/defaults/data_unit.yml",
                "config_type": "DataUnitConfig",
                "category": "core",
                "description": "File-based data unit configuration"
            },
            
            "nanobrain.core.data_unit.DataUnitString": {
                "config_file": "nanobrain/core/config/defaults/data_unit.yml",
                "config_type": "DataUnitConfig",
                "category": "core",
                "description": "String-based data unit configuration"
            },
            
            "nanobrain.core.data_unit.DataUnitStream": {
                "config_file": "nanobrain/core/config/defaults/data_unit.yml",
                "config_type": "DataUnitConfig",
                "category": "core",
                "description": "Stream-based data unit configuration"
            },
            
            # Trigger Components
            "nanobrain.core.trigger.DataUpdatedTrigger": {
                "config_file": "nanobrain/core/config/defaults/trigger.yml",
                "config_type": "TriggerConfig",
                "category": "core",
                "description": "Data update trigger configuration"
            },
            
            "nanobrain.core.trigger.AllDataReceivedTrigger": {
                "config_file": "nanobrain/core/config/defaults/trigger.yml",
                "config_type": "TriggerConfig",
                "category": "core",
                "description": "All data received trigger configuration"
            },
            
            "nanobrain.core.trigger.TimerTrigger": {
                "config_file": "nanobrain/core/config/defaults/trigger.yml",
                "config_type": "TriggerConfig",
                "category": "core",
                "description": "Timer-based trigger configuration"
            },
            
            "nanobrain.core.trigger.ManualTrigger": {
                "config_file": "nanobrain/core/config/defaults/trigger.yml",
                "config_type": "TriggerConfig",
                "category": "core",
                "description": "Manual trigger configuration"
            },
            
            # Agent Components
            "nanobrain.library.agents.specialized.base.SimpleSpecializedAgent": {
                "config_file": "nanobrain/library/config/defaults/agent.yml",
                "config_type": "AgentConfig",
                "category": "library",
                "description": "Simple specialized agent configuration"
            },
            
            "nanobrain.library.agents.specialized.base.ConversationalSpecializedAgent": {
                "config_file": "nanobrain/library/config/defaults/agent.yml",
                "config_type": "AgentConfig",
                "category": "library",
                "description": "Conversational specialized agent configuration"
            },
            
            "nanobrain.library.agents.enhanced.collaborative_agent.CollaborativeAgent": {
                "config_file": "nanobrain/library/config/defaults/agent.yml",
                "config_type": "AgentConfig",
                "category": "library",
                "description": "Collaborative agent configuration"
            },
            
            "nanobrain.library.agents.conversational.enhanced_collaborative_agent.EnhancedCollaborativeAgent": {
                "config_file": "nanobrain/library/config/defaults/agent.yml",
                "config_type": "AgentConfig",
                "category": "library",
                "description": "Enhanced collaborative agent configuration"
            },
            
            # Bioinformatics Tool Components
            "nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool": {
                "config_file": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
                "config_type": "BVBRCConfig",
                "category": "library",
                "description": "BV-BRC tool configuration"
            },
            
            "nanobrain.library.tools.bioinformatics.mmseqs_tool.MMseqs2Tool": {
                "config_file": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
                "config_type": "MMseqs2Config",
                "category": "library",
                "description": "MMseqs2 tool configuration"
            },
            
            "nanobrain.library.tools.bioinformatics.muscle_tool.MUSCLETool": {
                "config_file": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
                "config_type": "MUSCLEConfig",
                "category": "library",
                "description": "MUSCLE tool configuration"
            },
            
            "nanobrain.library.tools.bioinformatics.pubmed_client.PubMedClient": {
                "config_file": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
                "config_type": "PubMedConfig",
                "category": "library",
                "description": "PubMed client configuration"
            },
            
            # Interface Components
            "nanobrain.library.interfaces.web.web_interface.WebInterface": {
                "config_file": "nanobrain/library/config/defaults/web_interface.yml",
                "config_type": "WebInterfaceConfig",
                "category": "library",
                "description": "Web interface configuration"
            },
            
            # Workflow Components
            "nanobrain.library.workflows.chat_workflow.chat_workflow.ChatWorkflow": {
                "config_file": "nanobrain/library/config/defaults/chat_workflow.yml",
                "config_type": "Dict[str, Any]",
                "category": "library",
                "description": "Chat workflow configuration"
            },
            
            # Viral Protein Analysis Workflow Steps
            "nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "BV-BRC data acquisition step configuration"
            },
            
            "nanobrain.library.workflows.viral_protein_analysis.steps.sequence_curation_step.SequenceCurationStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Sequence curation step configuration"
            },
            
            "nanobrain.library.workflows.viral_protein_analysis.steps.clustering_step.ClusteringStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Clustering step configuration"
            },
            
            "nanobrain.library.workflows.viral_protein_analysis.steps.annotation_mapping_step.AnnotationMappingStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Annotation mapping step configuration"
            },
            
            "nanobrain.library.workflows.viral_protein_analysis.steps.alignment_step.AlignmentStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Alignment step configuration"
            },
            
            "nanobrain.library.workflows.viral_protein_analysis.steps.pssm_analysis_step.PSSMAnalysisStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "PSSM analysis step configuration"
            },
            
            "nanobrain.library.workflows.viral_protein_analysis.steps.report_generation_step.ReportGenerationStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Report generation step configuration"
            },
            
            # Chatbot Workflow Steps (short names - these need full path migration)
            "QueryClassificationStep": {
                "config_file": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Query classification step configuration"
            },
            
            "AnnotationJobStep": {
                "config_file": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Annotation job step configuration"
            },
            
            "ConversationalResponseStep": {
                "config_file": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Conversational response step configuration"
            },
            
            "ResponseFormattingStep": {
                "config_file": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Response formatting step configuration"
            },
            
            # Generic step types that might be used
            "GenomeFilteringStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Genome filtering step configuration"
            },
            
            "ProteinExtractionStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Protein extraction step configuration"
            },
            
            "SequenceAnnotationStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Sequence annotation step configuration"
            },
            
            "AnnotationStandardizationStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Annotation standardization step configuration"
            },
            
            "GenomeSchematicStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Genome schematic step configuration"
            },
            
            "LengthAnalysisStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Length analysis step configuration"
            },
            
            "AlignmentStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Alignment step configuration"
            },
            
            "PSSMGenerationStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "PSSM generation step configuration"
            },
            
            "ConservationAnalysisStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Conservation analysis step configuration"
            },
            
            "QualityAssessmentStep": {
                "config_file": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
                "config_type": "StepConfig",
                "category": "library",
                "description": "Quality assessment step configuration"
            }
        }
        
        return registry
    
    def save_registry(self, output_file: str = "config_registry.json") -> None:
        """Save the component registry to a JSON file"""
        registry = self.build_component_registry()
        
        # Add metadata
        registry_data = {
            "metadata": {
                "version": "1.0.0",
                "description": "NanoBrain Framework Component Configuration Registry",
                "total_components": len(registry),
                "categories": list(set(comp["category"] for comp in registry.values()))
            },
            "components": registry
        }
        
        with open(output_file, 'w') as f:
            json.dump(registry_data, f, indent=2, sort_keys=True)
        
        print(f"✅ Configuration registry saved to: {output_file}")
        print(f"📊 Total components registered: {len(registry)}")
    
    def save_registry_yaml(self, output_file: str = "config_registry.yml") -> None:
        """Save the component registry to a YAML file"""
        registry = self.build_component_registry()
        
        registry_data = {
            "metadata": {
                "version": "1.0.0",
                "description": "NanoBrain Framework Component Configuration Registry",
                "total_components": len(registry),
                "categories": list(set(comp["category"] for comp in registry.values()))
            },
            "components": registry
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(registry_data, f, default_flow_style=False, sort_keys=True)
        
        print(f"✅ Configuration registry (YAML) saved to: {output_file}")
    
    def validate_config_files(self) -> bool:
        """Validate that all referenced configuration files exist"""
        registry = self.build_component_registry()
        missing_files = []
        
        for component, config_info in registry.items():
            config_file = Path(config_info["config_file"])
            if not config_file.exists():
                missing_files.append((component, config_file))
        
        if missing_files:
            print("❌ Missing configuration files:")
            for component, file_path in missing_files:
                print(f"  - {component}: {file_path}")
            return False
        else:
            print("✅ All configuration files exist")
            return True
    
    def generate_config_index(self) -> None:
        """Generate an index of all configuration files"""
        registry = self.build_component_registry()
        
        # Group by category
        by_category = {}
        for component, config_info in registry.items():
            category = config_info["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((component, config_info))
        
        print("\n📋 CONFIGURATION FILES INDEX")
        print("=" * 50)
        
        for category, components in by_category.items():
            print(f"\n🔧 {category.upper()} COMPONENTS:")
            print("-" * 30)
            
            # Group by config file
            by_config_file = {}
            for component, config_info in components:
                config_file = config_info["config_file"]
                if config_file not in by_config_file:
                    by_config_file[config_file] = []
                by_config_file[config_file].append((component, config_info))
            
            for config_file, file_components in by_config_file.items():
                print(f"\n📄 {config_file}")
                for component, config_info in file_components:
                    class_name = component.split('.')[-1]
                    print(f"  ✅ {class_name} ({config_info['config_type']})")
    
    def generate_migration_helper(self) -> None:
        """Generate helper script for migrating to default configs"""
        registry = self.build_component_registry()
        
        migration_script = """#!/usr/bin/env python3
'''
Auto-generated migration helper script for default configurations.
'''

from pathlib import Path
import yaml

COMPONENT_CONFIG_MAPPING = {
"""
        
        for component, config_info in registry.items():
            migration_script += f'    "{component}": "{config_info["config_file"]}",\n'
        
        migration_script += """
}

def get_default_config_for_component(component_class_path: str) -> str:
    '''Get the default configuration file path for a component.'''
    return COMPONENT_CONFIG_MAPPING.get(component_class_path)

def load_default_config(component_class_path: str) -> dict:
    '''Load default configuration for a component.'''
    config_file = get_default_config_for_component(component_class_path)
    if not config_file:
        raise ValueError(f"No default configuration found for {component_class_path}")
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("Component Configuration Mapping:")
    for component, config_file in COMPONENT_CONFIG_MAPPING.items():
        print(f"  {component} -> {config_file}")
"""
        
        with open("scripts/config_migration_helper.py", 'w') as f:
            f.write(migration_script)
        
        print("✅ Migration helper script created: scripts/config_migration_helper.py")


def main():
    """Main function to create configuration registry"""
    print("🚀 Creating Configuration Registry...")
    print("=" * 50)
    
    registry = ConfigurationRegistry()
    
    # Validate configuration files exist
    print("\n1. Validating configuration files...")
    if not registry.validate_config_files():
        print("⚠️  Some configuration files are missing. Please ensure all default configs are created.")
        return
    
    # Save registry in multiple formats
    print("\n2. Saving configuration registry...")
    registry.save_registry("config_registry.json")
    registry.save_registry_yaml("config_registry.yml")
    
    # Generate index
    print("\n3. Generating configuration index...")
    registry.generate_config_index()
    
    # Generate migration helper
    print("\n4. Creating migration helper...")
    registry.generate_migration_helper()
    
    print("\n🎉 Configuration registry creation complete!")
    print("\n📋 Summary:")
    print("  ✅ Configuration registry: config_registry.json, config_registry.yml")
    print("  ✅ Migration helper: scripts/config_migration_helper.py")
    print("  ✅ All default configuration files validated")


if __name__ == "__main__":
    main() 