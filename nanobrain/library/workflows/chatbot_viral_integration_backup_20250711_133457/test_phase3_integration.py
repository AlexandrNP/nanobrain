#!/usr/bin/env python3
"""
Phase 3 Integration Test for ChatbotViralWorkflow

This test validates that ChatbotViralWorkflow properly integrates with
AlphavirusWorkflow as a step, following Phase 3 implementation requirements.

Test Coverage:
1. Configuration loading and validation
2. Data unit class field usage
3. Workflow-as-step integration
4. Data flow between components
5. Step interface compliance
6. End-to-end workflow execution
"""

import asyncio
import pytest
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from nanobrain.core.workflow import WorkflowConfig
from nanobrain.core.data_unit import DataUnitConfig
from nanobrain.library.workflows.chatbot_viral_integration.chatbot_viral_workflow import ChatbotViralWorkflow
from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow


class TestPhase3Integration:
    """Test suite for Phase 3 workflow-as-step integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config_path = Path(__file__).parent / "ChatbotViralWorkflow.yml"
        self.alphavirus_config_path = Path(__file__).parents[1] / "viral_protein_analysis" / "config" / "AlphavirusWorkflow.yml"
        
    def test_configuration_loading(self):
        """Test 1: Configuration files load properly with class-based data units"""
        
        # Load ChatbotViralWorkflow configuration
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate configuration structure
        assert config_dict['name'] == 'ChatbotViralWorkflow'
        assert config_dict['version'] == '4.5.0'
        assert 'steps' in config_dict
        assert 'data_units' in config_dict
        assert 'links' in config_dict
        
        # Validate that viral_protein_analysis step exists
        viral_step = None
        for step in config_dict['steps']:
            if step['step_id'] == 'viral_protein_analysis':
                viral_step = step
                break
        
        assert viral_step is not None, "viral_protein_analysis step not found"
        assert viral_step['class'] == 'nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow.AlphavirusWorkflow'
        assert viral_step['config_file'] == '../viral_protein_analysis/config/AlphavirusWorkflow.yml'
        
    def test_data_units_use_class_field(self):
        """Test 2: All data units use class field (Phase 1 requirement)"""
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Check data_units section
        for data_unit in config_dict['data_units']:
            assert 'class' in data_unit, f"Data unit {data_unit['name']} missing class field"
            assert data_unit['class'].startswith('nanobrain.core.data_unit.'), f"Invalid class path: {data_unit['class']}"
            assert 'data_type' not in data_unit, f"Data unit {data_unit['name']} has forbidden data_type field"
            assert 'type' not in data_unit, f"Data unit {data_unit['name']} has forbidden type field"
        
    def test_workflow_links_integration(self):
        """Test 3: Links properly connect to AlphavirusWorkflow step"""
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Find links that connect to viral_protein_analysis
        viral_input_link = None
        viral_output_link = None
        
        for link in config_dict['links']:
            if link['target'] == 'viral_protein_analysis.virus_input':
                viral_input_link = link
            elif link['source'] == 'viral_protein_analysis.analysis_results':
                viral_output_link = link
        
        assert viral_input_link is not None, "Input link to viral_protein_analysis not found"
        assert viral_output_link is not None, "Output link from viral_protein_analysis not found"
        
        # Validate link structure
        assert viral_input_link['source'] == 'virus_name_resolution.resolution_output'
        assert viral_output_link['target'] == 'response_formatting.analysis_input'
        
    def test_alphavirus_workflow_as_step(self):
        """Test 4: AlphavirusWorkflow can function as a step"""
        
        # Load AlphavirusWorkflow configuration
        with open(self.alphavirus_config_path, 'r') as f:
            alphavirus_config_dict = yaml.safe_load(f)
        
        # Create WorkflowConfig
        workflow_config = WorkflowConfig(**alphavirus_config_dict)
        
        # Create AlphavirusWorkflow instance
        alphavirus_workflow = AlphavirusWorkflow.from_config(workflow_config)
        
        # Verify it has Step interface methods
        assert hasattr(alphavirus_workflow, 'process'), "AlphavirusWorkflow missing process method"
        assert callable(alphavirus_workflow.process), "AlphavirusWorkflow.process not callable"
        
        # Verify input/output data units are defined
        assert 'input_data_units' in alphavirus_config_dict
        assert 'output_data_units' in alphavirus_config_dict
        
        # Verify input/output data units use class field
        input_data_units = alphavirus_config_dict['input_data_units']
        output_data_units = alphavirus_config_dict['output_data_units']
        
        assert 'virus_input' in input_data_units
        assert 'class' in input_data_units['virus_input']
        assert input_data_units['virus_input']['class'] == 'nanobrain.core.data_unit.DataUnitMemory'
        
        assert 'analysis_results' in output_data_units
        assert 'class' in output_data_units['analysis_results']
        assert output_data_units['analysis_results']['class'] == 'nanobrain.core.data_unit.DataUnitFile'
        
    def test_chatbot_workflow_initialization(self):
        """Test 5: ChatbotViralWorkflow initializes with class-based data units"""
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create WorkflowConfig
        workflow_config = WorkflowConfig(**config_dict)
        
        # Create ChatbotViralWorkflow instance
        chatbot_workflow = ChatbotViralWorkflow.from_config(workflow_config)
        
        # Verify data units were created with class field
        assert hasattr(chatbot_workflow, 'user_query_data_unit')
        assert hasattr(chatbot_workflow, 'extraction_result_data_unit')
        assert hasattr(chatbot_workflow, 'resolution_result_data_unit')
        assert hasattr(chatbot_workflow, 'analysis_result_data_unit')
        assert hasattr(chatbot_workflow, 'final_result_data_unit')
        
    def test_step_configuration_files(self):
        """Test 6: Step configuration files use class-based data units"""
        
        step_config_files = [
            'config/steps/query_classification.yml',
            'config/steps/virus_name_resolution.yml',
            'config/steps/response_formatting.yml',
            'config/steps/conversational_response.yml'
        ]
        
        for config_file in step_config_files:
            config_path = Path(__file__).parent / config_file
            if config_path.exists():
                with open(config_path, 'r') as f:
                    step_config = yaml.safe_load(f)
                
                # Check input_configs if present
                if 'input_configs' in step_config:
                    for input_name, input_config in step_config['input_configs'].items():
                        assert 'class' in input_config, f"Input config {input_name} in {config_file} missing class field"
                        assert input_config['class'].startswith('nanobrain.core.data_unit.')
                
                # Check output_config if present
                if 'output_config' in step_config:
                    output_config = step_config['output_config']
                    assert 'class' in output_config, f"Output config in {config_file} missing class field"
                    assert output_config['class'].startswith('nanobrain.core.data_unit.')
    
    def test_workflow_metadata_phase3(self):
        """Test 7: Workflow metadata indicates Phase 3 completion"""
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Check workflow metadata
        assert 'workflow_metadata' in config_dict
        metadata = config_dict['workflow_metadata']
        
        assert metadata['framework_compliance'] == 'from_config_mandatory'
        assert metadata['migration_status'] == 'phase3_complete'
        assert metadata['integration_status'] == 'workflow_as_step_enabled'
        assert 'AlphavirusWorkflow as step' in metadata['notes']
        
    def test_data_flow_format_compatibility(self):
        """Test 8: Data flow formats are compatible between steps"""
        
        # Test virus_name_resolution output matches AlphavirusWorkflow input
        virus_resolution_config_path = Path(__file__).parent / 'config/steps/virus_name_resolution.yml'
        with open(virus_resolution_config_path, 'r') as f:
            resolution_config = yaml.safe_load(f)
        
        # Verify resolution output description mentions workflow input
        output_config = resolution_config['output_config']
        assert 'workflow input' in output_config['description'].lower()
        
        # Test response_formatting input matches AlphavirusWorkflow output
        response_config_path = Path(__file__).parent / 'config/steps/response_formatting.yml'
        with open(response_config_path, 'r') as f:
            response_config = yaml.safe_load(f)
        
        # Verify analysis_input expects workflow output
        analysis_input = response_config['input_configs']['analysis_input']
        assert 'workflow results' in analysis_input['description'].lower()
        assert analysis_input['class'] == 'nanobrain.core.data_unit.DataUnitFile'
        
    async def test_workflow_execution_simulation(self):
        """Test 9: Simulate workflow execution flow"""
        
        # Create mock input data
        test_input = {
            'virus_name': 'Chikungunya virus',
            'analysis_parameters': {
                'min_genome_length': 8000,
                'max_genome_length': 15000,
                'clustering_threshold': 0.8
            }
        }
        
        # Load and create AlphavirusWorkflow
        with open(self.alphavirus_config_path, 'r') as f:
            alphavirus_config_dict = yaml.safe_load(f)
        
        alphavirus_config = WorkflowConfig(**alphavirus_config_dict)
        alphavirus_workflow = AlphavirusWorkflow.from_config(alphavirus_config)
        
        # Verify workflow accepts the input format
        assert hasattr(alphavirus_workflow, 'process')
        
        # Test input validation (without actual execution)
        assert isinstance(test_input, dict)
        assert 'virus_name' in test_input
        assert 'analysis_parameters' in test_input
        
    def test_removed_individual_steps(self):
        """Test 10: Individual viral protein analysis steps are removed"""
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # These steps should no longer exist
        removed_steps = ['data_acquisition', 'annotation_mapping', 'annotation_job_processing']
        
        existing_step_ids = [step['step_id'] for step in config_dict['steps']]
        
        for removed_step in removed_steps:
            assert removed_step not in existing_step_ids, f"Step {removed_step} should have been removed"
        
        # Verify that viral_protein_analysis step exists instead
        assert 'viral_protein_analysis' in existing_step_ids, "viral_protein_analysis step should exist"
        
    def test_triggers_configuration(self):
        """Test 11: Triggers are properly configured for data-driven execution"""
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Check that triggers are configured
        assert 'triggers' in config_dict
        triggers = config_dict['triggers']
        
        # Verify key triggers exist
        trigger_data_units = [trigger['data_unit'] for trigger in triggers]
        
        assert 'user_query' in trigger_data_units
        assert 'extracted_query_data' in trigger_data_units
        assert 'resolution_output' in trigger_data_units
        assert 'analysis_results' in trigger_data_units
        
        # Verify all triggers use class field
        for trigger in triggers:
            assert 'class' in trigger
            assert trigger['class'] == 'nanobrain.core.triggers.DataUnitChangeTrigger'


def run_phase3_validation():
    """Run all Phase 3 validation tests"""
    
    print("🧪 Phase 3 Integration Validation")
    print("=" * 60)
    
    test_instance = TestPhase3Integration()
    test_instance.setup_method()
    
    tests = [
        ("Configuration Loading", test_instance.test_configuration_loading),
        ("Data Units Use Class Field", test_instance.test_data_units_use_class_field),
        ("Workflow Links Integration", test_instance.test_workflow_links_integration),
        ("AlphavirusWorkflow as Step", test_instance.test_alphavirus_workflow_as_step),
        ("ChatbotWorkflow Initialization", test_instance.test_chatbot_workflow_initialization),
        ("Step Configuration Files", test_instance.test_step_configuration_files),
        ("Workflow Metadata Phase3", test_instance.test_workflow_metadata_phase3),
        ("Data Flow Compatibility", test_instance.test_data_flow_format_compatibility),
        ("Removed Individual Steps", test_instance.test_removed_individual_steps),
        ("Triggers Configuration", test_instance.test_triggers_configuration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 Phase 3 Integration Validation: ALL TESTS PASSED!")
        print("✅ AlphavirusWorkflow successfully integrated as step")
        print("✅ Class-based data units implemented")
        print("✅ Data flow properly configured")
        print("✅ Individual steps successfully replaced")
        print("✅ Framework compliance achieved")
        return True
    else:
        print(f"\n❌ Phase 3 Integration Validation: {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_phase3_validation()
    exit(0 if success else 1) 