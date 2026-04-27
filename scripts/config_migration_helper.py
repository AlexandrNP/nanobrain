#!/usr/bin/env python3
'''
Auto-generated migration helper script for default configurations.
'''

from pathlib import Path
import yaml

COMPONENT_CONFIG_MAPPING = {
    "nanobrain.core.step.Step": "nanobrain/core/config/defaults/step.yml",
    "nanobrain.core.step.TransformStep": "nanobrain/core/config/defaults/step.yml",
    "nanobrain.core.executor.LocalExecutor": "nanobrain/core/config/defaults/executor.yml",
    "nanobrain.core.executor.ThreadExecutor": "nanobrain/core/config/defaults/executor.yml",
    "nanobrain.core.executor.ProcessExecutor": "nanobrain/core/config/defaults/executor.yml",
    "nanobrain.core.executor.ParslExecutor": "nanobrain/core/config/defaults/executor.yml",
    "nanobrain.core.data_unit.DataUnitMemory": "nanobrain/core/config/defaults/data_unit.yml",
    "nanobrain.core.data_unit.DataUnitFile": "nanobrain/core/config/defaults/data_unit.yml",
    "nanobrain.core.data_unit.DataUnitString": "nanobrain/core/config/defaults/data_unit.yml",
    "nanobrain.core.data_unit.DataUnitStream": "nanobrain/core/config/defaults/data_unit.yml",
    "nanobrain.core.trigger.DataUpdatedTrigger": "nanobrain/core/config/defaults/trigger.yml",
    "nanobrain.core.trigger.AllDataReceivedTrigger": "nanobrain/core/config/defaults/trigger.yml",
    "nanobrain.core.trigger.TimerTrigger": "nanobrain/core/config/defaults/trigger.yml",
    "nanobrain.core.trigger.ManualTrigger": "nanobrain/core/config/defaults/trigger.yml",
    "nanobrain.library.agents.specialized.base.SimpleSpecializedAgent": "nanobrain/library/config/defaults/agent.yml",
    "nanobrain.library.agents.specialized.base.ConversationalSpecializedAgent": "nanobrain/library/config/defaults/agent.yml",
    "nanobrain.library.agents.enhanced.collaborative_agent.CollaborativeAgent": "nanobrain/library/config/defaults/agent.yml",
    "nanobrain.library.agents.conversational.enhanced_collaborative_agent.EnhancedCollaborativeAgent": "nanobrain/library/config/defaults/agent.yml",
    "nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
    "nanobrain.library.tools.bioinformatics.mmseqs_tool.MMseqs2Tool": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
    "nanobrain.library.tools.bioinformatics.muscle_tool.MUSCLETool": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
    "nanobrain.library.tools.bioinformatics.pubmed_client.PubMedClient": "nanobrain/library/config/defaults/bioinformatics_tools.yml",
    "nanobrain.library.interfaces.web.web_interface.WebInterface": "nanobrain/library/config/defaults/web_interface.yml",
    "nanobrain.library.workflows.chat_workflow.chat_workflow.ChatWorkflow": "nanobrain/library/config/defaults/chat_workflow.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.sequence_curation_step.SequenceCurationStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.clustering_step.ClusteringStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.annotation_mapping_step.AnnotationMappingStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.alignment_step.AlignmentStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.pssm_analysis_step.PSSMAnalysisStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "nanobrain.library.workflows.viral_protein_analysis.steps.report_generation_step.ReportGenerationStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "QueryClassificationStep": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
    "AnnotationJobStep": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
    "ConversationalResponseStep": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
    "ResponseFormattingStep": "nanobrain/library/config/defaults/chatbot_workflow_steps.yml",
    "GenomeFilteringStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "ProteinExtractionStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "SequenceAnnotationStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "AnnotationStandardizationStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "GenomeSchematicStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "LengthAnalysisStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "AlignmentStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "PSSMGenerationStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "ConservationAnalysisStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",
    "QualityAssessmentStep": "nanobrain/library/config/defaults/viral_analysis_steps.yml",

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
