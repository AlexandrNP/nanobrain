#!/usr/bin/env python3
"""
Integration Test for Viral Protein Analysis Workflow
=====================================================

Full integration test for the viral protein analysis workflow after
framework compliance updates.

This test validates:
- Workflow initialization
- Component loading
- Data flow
- Execution readiness
"""

import asyncio
import pytest
import logging

from nanobrain.library.workflows.viral_protein_analysis.test_helper import ViralProteinWorkflowTester

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestViralProteinWorkflow:
    """Integration tests for viral protein analysis workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_creation(self):
        """Test that workflow can be created from configuration."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        assert workflow is not None, "Workflow creation failed"
        assert workflow.name == "AlphavirusWorkflow", f"Unexpected workflow name: {workflow.name}"
        assert hasattr(workflow, 'steps'), "Workflow missing steps attribute"
        
    @pytest.mark.asyncio
    async def test_workflow_components(self):
        """Test that all workflow components are properly loaded."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        validation = ViralProteinWorkflowTester.validate_workflow_components(workflow)
        
        # Check all validation results
        for check, result in validation.items():
            assert result, f"Validation failed for: {check}"
            
        # Specific component checks
        assert len(workflow.steps) == 8, f"Expected 8 steps, got {len(workflow.steps)}"
        
        # Expected step names
        expected_steps = [
            'data_acquisition',
            'annotation_mapping',
            'sequence_curation',
            'clustering_analysis',
            'alignment',
            'pssm_analysis',
            'data_aggregation',
            'result_collection'
        ]
        
        for step_name in expected_steps:
            assert step_name in workflow.steps, f"Missing expected step: {step_name}"
    
    @pytest.mark.asyncio
    async def test_links_property(self):
        """Test that the new workflow_links property accessor works."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        # Test that workflow_links property is accessible
        assert hasattr(workflow, 'workflow_links'), "Workflow missing workflow_links property"
        
        links = workflow.workflow_links
        assert isinstance(links, dict), f"Links should be a dict, got {type(links)}"
        
        # Check expected number of links
        expected_link_count = 9
        actual_link_count = len(links)
        assert actual_link_count == expected_link_count, \
            f"Expected {expected_link_count} links, got {actual_link_count}"
        
        # Get link information
        link_info = ViralProteinWorkflowTester.get_link_info(workflow)
        logger.info(f"Found {len(link_info)} links with info")
        
    @pytest.mark.asyncio
    async def test_graph_validation(self):
        """Test that the public graph validation method works."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        # Test that validate_graph is accessible
        assert hasattr(workflow, 'validate_graph'), "Workflow missing validate_graph method"
        
        # Validate the graph
        is_valid = workflow.validate_graph()
        assert is_valid, "Workflow graph validation failed"
        
    @pytest.mark.asyncio
    async def test_data_unit_access(self):
        """Test that workflow data units are accessible via inherited Step properties."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        # Check that workflow inherits Step data unit properties
        assert hasattr(workflow, 'step_input_data_units'), \
            "Workflow missing step_input_data_units (inherited from Step)"
        assert hasattr(workflow, 'step_output_data_units'), \
            "Workflow missing step_output_data_units (inherited from Step)"
        
        # Check specific data units
        assert 'workflow_input' in workflow.step_input_data_units, \
            "Missing workflow_input data unit"
        assert 'workflow_output' in workflow.step_output_data_units, \
            "Missing workflow_output data unit"
        
    @pytest.mark.asyncio
    async def test_set_input_data(self):
        """Test setting input data to the workflow."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        test_data = {
            "virus_name": "Chikungunya virus",
            "confidence_threshold": 0.95,
            "validation_criteria": {
                "species_match_threshold": 0.95,
                "taxonomic_verification_threshold": 0.90,
                "require_zero_contamination": True
            }
        }
        
        success = await ViralProteinWorkflowTester.set_test_input(workflow, test_data)
        assert success, "Failed to set test input data"
        
        # Verify data was set
        input_unit = workflow.step_input_data_units.get('workflow_input')
        assert input_unit is not None, "workflow_input data unit not found"
        
        retrieved_data = await input_unit.get()
        assert retrieved_data is not None, "Failed to retrieve set data"
        assert retrieved_data.get('virus_name') == "Chikungunya virus", \
            "Retrieved data doesn't match set data"
        
    @pytest.mark.asyncio
    async def test_minimal_execution_readiness(self):
        """Test that workflow is ready for execution with minimal test data."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        # Run minimal test
        test_passed = await ViralProteinWorkflowTester.run_minimal_test(workflow)
        assert test_passed, "Minimal execution readiness test failed"
        
    @pytest.mark.asyncio  
    async def test_step_configuration_loading(self):
        """Test that step configurations are properly loaded."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        # Check a specific step
        data_acq_step = workflow.steps.get('data_acquisition')
        assert data_acq_step is not None, "data_acquisition step not found"
        
        # Verify step has required methods
        assert hasattr(data_acq_step, 'execute'), "Step missing execute method"
        assert hasattr(data_acq_step, 'initialize'), "Step missing initialize method"
        
        # Check step configuration
        assert hasattr(data_acq_step, 'config'), "Step missing config attribute"
        
    @pytest.mark.asyncio
    async def test_workflow_configuration_compliance(self):
        """Test that workflow follows framework configuration patterns."""
        workflow = await ViralProteinWorkflowTester.create_test_workflow()
        
        # Check that workflow was created via from_config
        assert hasattr(workflow, 'config'), "Workflow missing config attribute"
        assert hasattr(workflow, 'workflow_config'), "Workflow missing workflow_config attribute"
        
        # Verify execution strategy is set
        assert workflow.workflow_config.execution_strategy is not None, \
            "Execution strategy not configured"
        
        # Verify workflow directory is set
        assert workflow.workflow_config.workflow_directory is not None, \
            "Workflow directory not configured"


async def run_full_integration_test():
    """
    Run full integration test suite for viral protein workflow.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("=" * 80)
    print("VIRAL PROTEIN WORKFLOW - FULL INTEGRATION TEST")
    print("=" * 80)
    
    test_results = {}
    test_class = TestViralProteinWorkflow()
    
    # List of test methods
    tests = [
        ("Workflow Creation", test_class.test_workflow_creation),
        ("Component Loading", test_class.test_workflow_components),
        ("Links Property", test_class.test_links_property),
        ("Graph Validation", test_class.test_graph_validation),
        ("Data Unit Access", test_class.test_data_unit_access),
        ("Set Input Data", test_class.test_set_input_data),
        ("Execution Readiness", test_class.test_minimal_execution_readiness),
        ("Step Configuration", test_class.test_step_configuration_loading),
        ("Framework Compliance", test_class.test_workflow_configuration_compliance)
    ]
    
    # Run each test
    for test_name, test_method in tests:
        print(f"\nRunning: {test_name}...")
        try:
            await test_method()
            test_results[test_name] = "PASSED"
            print(f"   ✅ {test_name}: PASSED")
        except AssertionError as e:
            test_results[test_name] = f"FAILED: {e}"
            print(f"   ❌ {test_name}: FAILED - {e}")
        except Exception as e:
            test_results[test_name] = f"ERROR: {e}"
            print(f"   ❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in test_results.values() if r == "PASSED")
    failed = len(test_results) - passed
    
    print(f"\nTotal Tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for test_name, result in test_results.items():
            if result != "PASSED":
                print(f"  - {test_name}: {result}")
    
    all_passed = failed == 0
    if all_passed:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("The viral protein analysis workflow is fully functional and compliant.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. See details above.")
    
    return all_passed


if __name__ == "__main__":
    # Run integration tests
    success = asyncio.run(run_full_integration_test())
    exit(0 if success else 1)
