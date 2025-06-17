"""
EEEV Production Integration Tests - Phase 4

Comprehensive integration tests for the complete EEEV workflow,
testing end-to-end functionality with real data simulation.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.0.0
"""

import pytest
import asyncio
import time
from pathlib import Path

from nanobrain.library.workflows.viral_protein_analysis.production_eeev_workflow import (
    ProductionEEEVWorkflow,
    EEEVAnalysisResult,
    EEEVProteinData,
    BoundaryPrediction,
    run_eeev_analysis
)


class TestEEEVProductionIntegration:
    """Integration tests for EEEV production workflow"""
    
    @pytest.fixture
    def workflow(self):
        """Create workflow instance for testing"""
        workflow = ProductionEEEVWorkflow()
        return workflow
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, workflow):
        """Test workflow initialization"""
        assert workflow is not None
        assert workflow.config is not None
        assert workflow.logger is not None
        assert 'eeev' in workflow.config
        assert 'expected_proteins' in workflow.config['eeev']
    
    @pytest.mark.asyncio
    async def test_complete_eeev_workflow_execution(self, workflow):
        """Test complete EEEV workflow execution"""
        # Execute complete workflow
        result = await workflow.execute_complete_eeev_analysis(
            organism="Eastern equine encephalitis virus",
            environment="testing"
        )
        
        # Validate basic result structure
        assert isinstance(result, EEEVAnalysisResult)
        assert result.analysis_id is not None
        assert result.success is True
        assert result.results is not None
        assert result.execution_time is not None
        assert result.execution_time > 0
        
        # Validate analysis summary
        analysis_summary = result.results.get('analysis_summary', {})
        assert 'total_predictions' in analysis_summary
        assert 'high_confidence_predictions' in analysis_summary
        assert 'analysis_timestamp' in analysis_summary
        assert analysis_summary['total_predictions'] > 0
        
        # Validate boundary predictions
        boundary_predictions = result.results.get('boundary_predictions', [])
        assert isinstance(boundary_predictions, list)
        assert len(boundary_predictions) > 0
        
        # Validate individual predictions
        for prediction in boundary_predictions:
            assert 'protein_id' in prediction
            assert 'protein_type' in prediction
            assert 'predicted_start' in prediction
            assert 'predicted_end' in prediction
            assert 'confidence' in prediction
            assert 'literature_support' in prediction
            
            # Validate data types
            assert isinstance(prediction['protein_id'], str)
            assert isinstance(prediction['protein_type'], str)
            assert isinstance(prediction['predicted_start'], int)
            assert isinstance(prediction['predicted_end'], int)
            assert isinstance(prediction['confidence'], (int, float))
            assert isinstance(prediction['literature_support'], bool)
            
            # Validate confidence range
            assert 0.0 <= prediction['confidence'] <= 1.0
            
            # Validate position logic
            assert prediction['predicted_start'] < prediction['predicted_end']
    
    @pytest.mark.asyncio
    async def test_eeev_specific_protein_detection(self, workflow):
        """Test EEEV-specific protein detection"""
        result = await workflow.execute_complete_eeev_analysis()
        
        assert result.success
        
        boundary_predictions = result.results.get('boundary_predictions', [])
        found_protein_types = [p['protein_type'] for p in boundary_predictions]
        
        # Check for expected EEEV structural proteins
        expected_eeev_proteins = [
            'capsid protein',
            'envelope protein E2'
        ]
        
        for expected_protein in expected_eeev_proteins:
            # Check if any found protein matches expected type
            found_match = any(
                expected_protein.lower() in ptype.lower() 
                for ptype in found_protein_types
            )
            assert found_match, f"Expected protein '{expected_protein}' not found in {found_protein_types}"
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self):
        """Test workflow error handling"""
        # Create workflow with invalid config to trigger error
        workflow = ProductionEEEVWorkflow("nonexistent_config.yml")
        
        # Mock an error in the workflow
        original_method = workflow._mock_data_acquisition
        
        async def failing_method():
            raise Exception("Simulated data acquisition failure")
        
        workflow._mock_data_acquisition = failing_method
        
        # Execute workflow and expect graceful failure
        result = await workflow.execute_complete_eeev_analysis()
        
        assert result.success is False
        assert result.error is not None
        assert "Simulated data acquisition failure" in result.error
        assert result.execution_time is not None
        
        # Restore original method
        workflow._mock_data_acquisition = original_method
    
    @pytest.mark.asyncio
    async def test_workflow_performance_requirements(self, workflow):
        """Test workflow meets performance requirements"""
        start_time = time.time()
        
        result = await workflow.execute_complete_eeev_analysis()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Workflow should complete in reasonable time (< 30 seconds for testing)
        assert total_time < 30, f"Workflow took {total_time:.2f} seconds, expected < 30"
        
        assert result.success
        assert result.execution_time is not None
        assert result.execution_time > 0
        
        # Execution time should be reasonably close to measured time
        time_difference = abs(result.execution_time - total_time)
        assert time_difference < 1.0, f"Execution time mismatch: {time_difference:.2f} seconds"
    
    @pytest.mark.asyncio
    async def test_mock_data_acquisition(self, workflow):
        """Test mock data acquisition step"""
        proteins = await workflow._mock_data_acquisition()
        
        assert isinstance(proteins, list)
        assert len(proteins) > 0
        
        for protein in proteins:
            assert isinstance(protein, EEEVProteinData)
            assert protein.protein_id is not None
            assert protein.protein_type is not None
            assert protein.sequence is not None
            assert protein.length > 0
            assert protein.genome_id is not None
            assert protein.start_position >= 0
            assert protein.end_position > protein.start_position
            assert protein.strand in ['+', '-']
    
    @pytest.mark.asyncio
    async def test_mock_clustering(self, workflow):
        """Test mock protein clustering step"""
        # Get test proteins
        proteins = await workflow._mock_data_acquisition()
        
        # Test clustering
        clusters = await workflow._mock_clustering(proteins)
        
        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        
        # Verify all proteins are clustered
        total_clustered = sum(len(cluster_proteins) for cluster_proteins in clusters.values())
        assert total_clustered == len(proteins)
        
        # Verify cluster structure
        for cluster_name, cluster_proteins in clusters.items():
            assert isinstance(cluster_name, str)
            assert isinstance(cluster_proteins, list)
            assert len(cluster_proteins) > 0
            
            for protein in cluster_proteins:
                assert isinstance(protein, EEEVProteinData)
    
    @pytest.mark.asyncio
    async def test_mock_boundary_detection(self, workflow):
        """Test mock boundary detection step"""
        # Get test data
        proteins = await workflow._mock_data_acquisition()
        clusters = await workflow._mock_clustering(proteins)
        
        # Test boundary detection
        predictions = await workflow._mock_boundary_detection(clusters)
        
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        
        for prediction in predictions:
            assert isinstance(prediction, BoundaryPrediction)
            assert prediction.protein_id is not None
            assert prediction.protein_type is not None
            assert prediction.predicted_start >= 0
            assert prediction.predicted_end > prediction.predicted_start
            assert 0.0 <= prediction.confidence <= 1.0
            assert isinstance(prediction.literature_support, bool)
            assert isinstance(prediction.supporting_references, list)
            assert prediction.pssm_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_results_processing(self, workflow):
        """Test results processing step"""
        # Get test data
        proteins = await workflow._mock_data_acquisition()
        clusters = await workflow._mock_clustering(proteins)
        predictions = await workflow._mock_boundary_detection(clusters)
        
        # Test results processing
        results = await workflow._process_results(predictions)
        
        assert isinstance(results, dict)
        assert 'analysis_summary' in results
        assert 'boundary_predictions' in results
        
        # Validate analysis summary
        summary = results['analysis_summary']
        assert 'total_predictions' in summary
        assert 'high_confidence_predictions' in summary
        assert 'analysis_timestamp' in summary
        assert summary['total_predictions'] == len(predictions)
        
        # Validate boundary predictions format
        boundary_preds = results['boundary_predictions']
        assert len(boundary_preds) == len(predictions)
        
        for pred in boundary_preds:
            assert 'protein_id' in pred
            assert 'protein_type' in pred
            assert 'predicted_start' in pred
            assert 'predicted_end' in pred
            assert 'confidence' in pred
            assert 'literature_support' in pred
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self, workflow):
        """Test progress tracking functionality"""
        progress_updates = []
        
        # Override progress update method to capture calls
        original_update = workflow._update_progress
        
        async def capture_progress(analysis_id, message, percentage):
            progress_updates.append((analysis_id, message, percentage))
            await original_update(analysis_id, message, percentage)
        
        workflow._update_progress = capture_progress
        
        # Execute workflow
        result = await workflow.execute_complete_eeev_analysis()
        
        assert result.success
        assert len(progress_updates) > 0
        
        # Validate progress updates
        for i, (analysis_id, message, percentage) in enumerate(progress_updates):
            assert isinstance(analysis_id, str)
            assert isinstance(message, str)
            assert isinstance(percentage, int)
            assert 0 <= percentage <= 100
            
            # Progress should be non-decreasing
            if i > 0:
                prev_percentage = progress_updates[i-1][2]
                assert percentage >= prev_percentage
        
        # Should reach 100% completion
        final_percentage = progress_updates[-1][2]
        assert final_percentage == 100
        
        # Restore original method
        workflow._update_progress = original_update
    
    @pytest.mark.asyncio
    async def test_analysis_id_generation(self, workflow):
        """Test analysis ID generation"""
        # Generate multiple IDs
        ids = []
        for _ in range(10):
            analysis_id = workflow._generate_analysis_id()
            ids.append(analysis_id)
            
            # Validate ID format
            assert isinstance(analysis_id, str)
            assert analysis_id.startswith('eeev_analysis_')
            assert len(analysis_id) > len('eeev_analysis_')
        
        # All IDs should be unique
        assert len(set(ids)) == len(ids)
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self):
        """Test concurrent workflow execution"""
        # Create multiple workflow instances
        workflows = [ProductionEEEVWorkflow() for _ in range(3)]
        
        # Execute workflows concurrently
        tasks = [
            workflow.execute_complete_eeev_analysis(organism=f"EEEV_test_{i}")
            for i, workflow in enumerate(workflows)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All workflows should succeed
        for result in results:
            assert result.success
            assert result.results is not None
            
        # All analysis IDs should be unique
        analysis_ids = [result.analysis_id for result in results]
        assert len(set(analysis_ids)) == len(analysis_ids)


class TestEEEVConvenienceFunction:
    """Test the convenience function for EEEV analysis"""
    
    @pytest.mark.asyncio
    async def test_run_eeev_analysis_function(self):
        """Test the run_eeev_analysis convenience function"""
        result = await run_eeev_analysis()
        
        assert isinstance(result, EEEVAnalysisResult)
        assert result.success
        assert result.results is not None
        assert result.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_run_eeev_analysis_with_custom_organism(self):
        """Test run_eeev_analysis with custom organism"""
        custom_organism = "Custom EEEV strain test"
        result = await run_eeev_analysis(organism=custom_organism)
        
        assert result.success
        assert result.results is not None


class TestEEEVDataStructures:
    """Test EEEV data structure validation"""
    
    def test_eeev_protein_data_creation(self):
        """Test EEEVProteinData creation and validation"""
        protein = EEEVProteinData(
            protein_id="test_001",
            protein_type="capsid protein",
            sequence="MQKVHVQPYHV",
            length=11,
            genome_id="test_genome",
            product="test capsid",
            start_position=100,
            end_position=200,
            strand="+"
        )
        
        assert protein.protein_id == "test_001"
        assert protein.protein_type == "capsid protein"
        assert protein.length == 11
        assert protein.start_position == 100
        assert protein.end_position == 200
        assert protein.strand == "+"
    
    def test_boundary_prediction_creation(self):
        """Test BoundaryPrediction creation and validation"""
        prediction = BoundaryPrediction(
            protein_id="test_001",
            protein_type="capsid protein",
            predicted_start=100,
            predicted_end=200,
            confidence=0.85,
            literature_support=True,
            supporting_references=["12345", "67890"],
            pssm_score=0.90
        )
        
        assert prediction.protein_id == "test_001"
        assert prediction.confidence == 0.85
        assert prediction.literature_support is True
        assert len(prediction.supporting_references) == 2
        assert prediction.pssm_score == 0.90
    
    def test_eeev_analysis_result_creation(self):
        """Test EEEVAnalysisResult creation and validation"""
        result = EEEVAnalysisResult(
            analysis_id="test_analysis_123",
            success=True,
            results={"test": "data"},
            execution_time=10.5
        )
        
        assert result.analysis_id == "test_analysis_123"
        assert result.success is True
        assert result.results == {"test": "data"}
        assert result.execution_time == 10.5
        assert result.error is None


@pytest.mark.integration
@pytest.mark.slow
class TestEEEVProductionReady:
    """Production readiness tests for EEEV workflow"""
    
    @pytest.mark.asyncio
    async def test_workflow_resilience_to_failures(self):
        """Test workflow resilience to component failures"""
        workflow = ProductionEEEVWorkflow()
        
        # Test with various failure scenarios
        failure_scenarios = [
            ("data_acquisition", "Data acquisition failed"),
            ("clustering", "Clustering failed"),
            ("boundary_detection", "Boundary detection failed")
        ]
        
        for component, error_message in failure_scenarios:
            # Create a new workflow instance for each test
            test_workflow = ProductionEEEVWorkflow()
            
            # Mock the component to fail
            if component == "data_acquisition":
                async def failing_method():
                    raise Exception(error_message)
                test_workflow._mock_data_acquisition = failing_method
            
            # Execute and verify graceful failure
            result = await test_workflow.execute_complete_eeev_analysis()
            
            assert result.success is False
            assert error_message in result.error
            assert result.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_large_scale_workflow_execution(self):
        """Test workflow with larger scale data simulation"""
        workflow = ProductionEEEVWorkflow()
        
        # Override mock data to simulate larger dataset
        original_method = workflow._mock_data_acquisition
        
        async def large_dataset_method():
            proteins = []
            for i in range(50):  # Simulate 50 proteins
                protein = EEEVProteinData(
                    protein_id=f"EEEV_{i:03d}",
                    protein_type=f"protein_type_{i % 5}",  # 5 different types
                    sequence="MQKVHVQPYHV" * (10 + i % 10),  # Variable length
                    length=(10 + i % 10) * 11,
                    genome_id="large_eeev_genome",
                    product=f"test protein {i}",
                    start_position=i * 300,
                    end_position=(i + 1) * 300 - 1,
                    strand="+" if i % 2 == 0 else "-"
                )
                proteins.append(protein)
            return proteins
        
        workflow._mock_data_acquisition = large_dataset_method
        
        # Execute workflow
        result = await workflow.execute_complete_eeev_analysis()
        
        assert result.success
        assert len(result.results['boundary_predictions']) == 50
        
        # Restore original method
        workflow._mock_data_acquisition = original_method
    
    @pytest.mark.asyncio
    async def test_workflow_memory_efficiency(self):
        """Test workflow memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        workflow = ProductionEEEVWorkflow()
        result = await workflow.execute_complete_eeev_analysis()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert result.success
        # Memory increase should be reasonable (< 100MB for test workflow)
        assert memory_increase < 100, f"Memory increase: {memory_increase:.2f} MB"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"]) 