#!/usr/bin/env python3
"""
Test Helper for Viral Protein Analysis Workflow
================================================

Helper utilities for testing the viral protein analysis workflow after
framework compliance updates.

This module provides easy-to-use functions for:
- Creating and initializing test workflows
- Setting test input data
- Validating workflow components
- Running test executions
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import AlphavirusWorkflow
from nanobrain.core.workflow import Workflow

logger = logging.getLogger(__name__)


class ViralProteinWorkflowTester:
    """
    Helper class for testing viral protein analysis workflow.
    
    This class provides utility methods for testing the AlphavirusWorkflow
    after framework compliance updates, making it easier to validate
    workflow functionality and debug issues.
    """
    
    @staticmethod
    async def create_test_workflow(config_path: Optional[str] = None) -> AlphavirusWorkflow:
        """
        Create and initialize test workflow.
        
        Args:
            config_path: Optional path to workflow configuration file.
                        Defaults to the standard AlphavirusWorkflow config.
                        
        Returns:
            Initialized AlphavirusWorkflow instance ready for testing
            
        Example:
            >>> workflow = await ViralProteinWorkflowTester.create_test_workflow()
            >>> assert workflow is not None
        """
        if config_path is None:
            # Use default config path relative to workflow directory
            config_path = "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
            
        logger.info(f"Creating test workflow from config: {config_path}")
        
        try:
            # Create workflow using from_config pattern
            workflow = AlphavirusWorkflow.from_config(config_path)
            
            # Initialize the workflow
            await workflow.initialize()
            
            logger.info(f"Successfully created and initialized workflow: {workflow.name}")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to create test workflow: {e}")
            raise
    
    @staticmethod
    async def set_test_input(workflow: Workflow, test_data: Dict[str, Any]) -> bool:
        """
        Set test input data to workflow.
        
        Note: Workflow inherits from Step, so we use step_input_data_units
        to access the workflow-level input data units.
        
        Args:
            workflow: The workflow instance to set input data on
            test_data: Dictionary containing test input data
            
        Returns:
            bool: True if input was set successfully, False otherwise
            
        Example:
            >>> test_data = {"virus_name": "Chikungunya virus"}
            >>> success = await ViralProteinWorkflowTester.set_test_input(workflow, test_data)
            >>> assert success
        """
        try:
            # Workflow inherits from Step, use step_input_data_units
            if hasattr(workflow, 'step_input_data_units'):
                input_unit = workflow.step_input_data_units.get('workflow_input')
                if input_unit:
                    await input_unit.set(test_data)
                    logger.info(f"Successfully set test input data: {list(test_data.keys())}")
                    return True
                else:
                    logger.warning("workflow_input data unit not found in step_input_data_units")
            else:
                logger.error("Workflow missing step_input_data_units attribute")
                
        except Exception as e:
            logger.error(f"Failed to set test input: {e}")
            
        return False
    
    @staticmethod
    async def get_workflow_output(workflow: Workflow) -> Optional[Any]:
        """
        Get workflow output data.
        
        Args:
            workflow: The workflow instance to get output from
            
        Returns:
            The workflow output data, or None if not available
            
        Example:
            >>> output = await ViralProteinWorkflowTester.get_workflow_output(workflow)
            >>> if output:
            >>>     print(f"Got output: {output}")
        """
        try:
            # Workflow inherits from Step, use step_output_data_units
            if hasattr(workflow, 'step_output_data_units'):
                output_unit = workflow.step_output_data_units.get('workflow_output')
                if output_unit:
                    output_data = await output_unit.get()
                    logger.info("Successfully retrieved workflow output")
                    return output_data
                else:
                    logger.warning("workflow_output data unit not found")
            else:
                logger.error("Workflow missing step_output_data_units attribute")
                
        except Exception as e:
            logger.error(f"Failed to get workflow output: {e}")
            
        return None
    
    @staticmethod
    def validate_workflow_components(workflow: Workflow) -> Dict[str, bool]:
        """
        Validate workflow components are properly loaded.
        
        Args:
            workflow: The workflow instance to validate
            
        Returns:
            Dictionary with validation results for different components
            
        Example:
            >>> validation = ViralProteinWorkflowTester.validate_workflow_components(workflow)
            >>> assert all(validation.values()), f"Some components failed: {validation}"
        """
        validation_results = {}
        
        # Check steps
        validation_results['has_steps'] = bool(workflow.steps) if hasattr(workflow, 'steps') else False
        validation_results['expected_step_count'] = len(workflow.steps) == 8 if hasattr(workflow, 'steps') else False
        
        # Check links using new property accessor
        validation_results['has_links'] = bool(workflow.workflow_links) if hasattr(workflow, 'workflow_links') else False
        expected_link_count = 9
        actual_link_count = len(workflow.workflow_links) if hasattr(workflow, 'workflow_links') else 0
        validation_results['expected_link_count'] = actual_link_count == expected_link_count
        
        if actual_link_count != expected_link_count:
            logger.warning(f"Expected {expected_link_count} links, found {actual_link_count}")
        
        # Check graph validation using new public method
        validation_results['graph_valid'] = workflow.validate_graph() if hasattr(workflow, 'validate_graph') else False
        
        # Check data units
        validation_results['has_input_units'] = bool(workflow.step_input_data_units) if hasattr(workflow, 'step_input_data_units') else False
        validation_results['has_output_units'] = bool(workflow.step_output_data_units) if hasattr(workflow, 'step_output_data_units') else False
        
        # Check specific required components
        if hasattr(workflow, 'step_input_data_units'):
            validation_results['has_workflow_input'] = 'workflow_input' in workflow.step_input_data_units
        else:
            validation_results['has_workflow_input'] = False
            
        if hasattr(workflow, 'step_output_data_units'):
            validation_results['has_workflow_output'] = 'workflow_output' in workflow.step_output_data_units
        else:
            validation_results['has_workflow_output'] = False
        
        # Log validation summary
        logger.info(f"Validation results: {validation_results}")
        failed_checks = [k for k, v in validation_results.items() if not v]
        if failed_checks:
            logger.warning(f"Failed validation checks: {failed_checks}")
        else:
            logger.info("All validation checks passed!")
            
        return validation_results
    
    @staticmethod
    def get_step_names(workflow: Workflow) -> list:
        """
        Get list of step names in the workflow.
        
        Args:
            workflow: The workflow instance
            
        Returns:
            List of step names
        """
        if hasattr(workflow, 'steps'):
            return list(workflow.steps.keys())
        return []
    
    @staticmethod
    def get_link_info(workflow: Workflow) -> Dict[str, Dict[str, Any]]:
        """
        Get information about workflow links.
        
        Args:
            workflow: The workflow instance
            
        Returns:
            Dictionary with link information
        """
        link_info = {}
        
        if hasattr(workflow, 'workflow_links'):
            for link_id, link in workflow.workflow_links.items():
                info = {
                    'type': type(link).__name__,
                    'has_source': hasattr(link, 'source'),
                    'has_target': hasattr(link, 'target')
                }
                
                if hasattr(link, 'source'):
                    info['source'] = str(link.source)
                if hasattr(link, 'target'):
                    info['target'] = str(link.target)
                    
                link_info[link_id] = info
                
        return link_info
    
    @staticmethod
    async def run_minimal_test(workflow: Workflow) -> bool:
        """
        Run a minimal test of the workflow with test data.
        
        Args:
            workflow: The workflow instance to test
            
        Returns:
            bool: True if test passed, False otherwise
        """
        try:
            # Create minimal test input
            test_input = {
                "virus_name": "test_alphavirus",
                "confidence_threshold": 0.9,
                "validation_criteria": {
                    "species_match_threshold": 0.95,
                    "taxonomic_verification_threshold": 0.90,
                    "require_zero_contamination": True
                }
            }
            
            # Set input
            success = await ViralProteinWorkflowTester.set_test_input(workflow, test_input)
            if not success:
                logger.error("Failed to set test input")
                return False
                
            logger.info("Minimal test setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Minimal test failed: {e}")
            return False


# Convenience function for quick testing
async def test_viral_workflow():
    """
    Quick test function for viral protein workflow.
    
    This function provides a simple way to test the workflow
    with all the fixes applied.
    
    Example:
        >>> import asyncio
        >>> asyncio.run(test_viral_workflow())
    """
    print("=" * 80)
    print("VIRAL PROTEIN WORKFLOW TEST (WITH FIXES)")
    print("=" * 80)
    
    tester = ViralProteinWorkflowTester()
    
    # Create workflow
    print("\n1. Creating workflow...")
    workflow = await tester.create_test_workflow()
    print(f"   ✅ Workflow created: {workflow.name}")
    
    # Validate components
    print("\n2. Validating components...")
    validation = tester.validate_workflow_components(workflow)
    
    for check, result in validation.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}: {result}")
    
    # Check specific improvements
    print("\n3. Testing improvements...")
    
    # Test links accessor
    if hasattr(workflow, 'workflow_links'):
        print(f"   ✅ Links property accessible: {len(workflow.workflow_links)} links")
        link_info = tester.get_link_info(workflow)
        for link_id in list(link_info.keys())[:3]:  # Show first 3
            print(f"      - {link_id}: {link_info[link_id]['type']}")
    else:
        print("   ❌ Links property not accessible")
    
    # Test graph validation
    if hasattr(workflow, 'validate_graph'):
        valid = workflow.validate_graph()
        status = "✅" if valid else "⚠️"
        print(f"   {status} Graph validation accessible: {valid}")
    else:
        print("   ❌ Graph validation not accessible")
    
    # Test data unit access
    print("\n4. Testing data unit access...")
    test_passed = await tester.run_minimal_test(workflow)
    status = "✅" if test_passed else "❌"
    print(f"   {status} Data unit access test: {test_passed}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = all(validation.values()) and test_passed
    if all_passed:
        print("\n✅ ALL TESTS PASSED! Workflow is ready for use.")
    else:
        print("\n⚠️ Some tests failed. See details above.")
        
    return all_passed


if __name__ == "__main__":
    # Run the test when script is executed directly
    asyncio.run(test_viral_workflow())
