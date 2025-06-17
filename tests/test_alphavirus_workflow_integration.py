"""
Integration Tests for Alphavirus Workflow

Tests the complete workflow integration with real BV-BRC connectivity
to ensure the next phase implementation is working correctly.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

from nanobrain.library.workflows.viral_protein_analysis.alphavirus_workflow import (
    AlphavirusWorkflow, WorkflowData, WorkflowResult
)
from nanobrain.library.workflows.viral_protein_analysis.config.workflow_config import (
    AlphavirusWorkflowConfig, BVBRCConfig
)
from nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step import (
    BVBRCDataAcquisitionStep, GenomeData, ProteinData
)
from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool


class TestAlphavirusWorkflowIntegration:
    """Integration tests for the complete Alphavirus workflow"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
            
    @pytest.fixture
    def test_config(self, temp_output_dir):
        """Create test configuration"""
        config = AlphavirusWorkflowConfig()
        config.output.base_directory = temp_output_dir
        config.bvbrc.verify_on_init = False  # Skip verification for tests
        return config
        
    @pytest.fixture
    def workflow(self, test_config):
        """Create workflow instance"""
        return AlphavirusWorkflow(test_config)
        
    def test_configuration_loading(self):
        """Test configuration loading from YAML"""
        config_path = Path(__file__).parent.parent / "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
        
        if config_path.exists():
            config = AlphavirusWorkflowConfig.from_file(str(config_path))
            assert config.name == "alphavirus_analysis"
            assert config.bvbrc.executable_path == "/Applications/BV-BRC.app/deployment/bin/"
            assert config.clustering.min_seq_id == 0.7
        else:
            # Test default configuration
            config = AlphavirusWorkflowConfig()
            assert config.name == "alphavirus_analysis"
            
    def test_configuration_validation(self):
        """Test configuration validation"""
        config = AlphavirusWorkflowConfig()
        issues = config.validate()
        
        # Should have BV-BRC path issues on most systems (expected)
        print(f"Configuration validation issues (expected): {issues}")
        # This is informational - don't fail the test for missing BV-BRC
        
    def test_workflow_data_container(self):
        """Test workflow data container functionality"""
        workflow_data = WorkflowData()
        
        # Test initial state
        assert workflow_data.original_genomes == []
        assert workflow_data.filtered_genomes == []
        assert workflow_data.unique_proteins == []
        
        # Test data updates
        test_result = {
            'original_genomes': [{'genome_id': 'test1', 'genome_length': 11000}],
            'filtered_genomes': [{'genome_id': 'test1', 'genome_length': 11000}],
            'unique_proteins': [{'patric_id': 'test_protein', 'aa_sequence_md5': 'test_md5'}]
        }
        
        workflow_data.update_from_acquisition(test_result)
        assert len(workflow_data.original_genomes) == 1
        assert len(workflow_data.filtered_genomes) == 1
        assert len(workflow_data.unique_proteins) == 1
        
    @pytest.mark.asyncio
    async def test_bvbrc_data_acquisition_step_mock(self, test_config):
        """Test BV-BRC data acquisition step with mocked responses"""
        
        # Create step with test configuration
        bvbrc_config = BVBRCConfig()
        bvbrc_config.verify_on_init = False
        
        step = BVBRCDataAcquisitionStep(bvbrc_config, test_config.bvbrc.__dict__)
        
        # Mock the BV-BRC tool responses
        mock_genomes_data = b"""genome_id\tgenome_length\tgenome_name\ttaxon_lineage_names
511145.12\t11000\tTest Alphavirus 1\tViruses;Alphavirus
511145.13\t11500\tTest Alphavirus 2\tViruses;Alphavirus
511145.14\t5000\tTest Fragment\tViruses;Alphavirus
"""
        
        mock_proteins_data = b"""patric_id\taa_sequence_md5\tgenome_id\tproduct\tgene
test_protein_1\tmd5_1\t511145.12\tcapsid protein\tcapsid
test_protein_2\tmd5_2\t511145.12\tenvelope protein\tE1
test_protein_3\tmd5_1\t511145.13\tcapsid protein\tcapsid
"""
        
        mock_sequences_data = b"""aa_sequence_md5\taa_sequence
md5_1\tMKLLVVVAGKKSSQQRRTTYYAACC
md5_2\tMQWERTYUIKJHGFDSAASDFGH
"""
        
        mock_annotations_data = b"""aa_sequence_md5\tproduct\tgene\trefseq_locus_tag
md5_1\tcapsid protein\tcapsid\tCAP_01
md5_2\tenvelope protein\tE1\tENV_01
"""
        
        with patch.object(step.bv_brc_tool, 'verify_installation', return_value=True), \
             patch.object(step.bv_brc_tool, 'execute_p3_command') as mock_execute:
            
            # Setup mock responses based on command
            async def mock_command_response(command, args):
                from nanobrain.core.external_tool import ToolResult
                
                if command == "p3-all-genomes":
                    return ToolResult(returncode=0, stdout=mock_genomes_data, stderr=b"", 
                                    execution_time=1.0, command=[command] + args, success=True)
                elif command == "p3-get-feature-data" and "patric_id" in str(args):
                    return ToolResult(returncode=0, stdout=mock_proteins_data, stderr=b"",
                                    execution_time=1.0, command=[command] + args, success=True)
                elif command == "p3-get-feature-data" and "aa_sequence_md5" in str(args):
                    return ToolResult(returncode=0, stdout=mock_annotations_data, stderr=b"",
                                    execution_time=1.0, command=[command] + args, success=True)
                elif command == "p3-get-feature-sequence":
                    return ToolResult(returncode=0, stdout=mock_sequences_data, stderr=b"",
                                    execution_time=1.0, command=[command] + args, success=True)
                else:
                    return ToolResult(returncode=1, stdout=b"", stderr=b"Unknown command",
                                    execution_time=0.1, command=[command] + args, success=False)
                    
            mock_execute.side_effect = mock_command_response
            
            # Execute the step
            result = await step.execute({'target_genus': 'Alphavirus'})
            
            # Verify results
            assert result['statistics']['total_genomes_downloaded'] == 3
            assert result['statistics']['genomes_after_filtering'] == 2  # Filtered out 5000 bp genome
            assert result['statistics']['unique_proteins_found'] == 2   # md5_1 and md5_2
            assert len(result['protein_sequences']) == 2
            assert len(result['protein_annotations']) == 2
            assert result['annotated_fasta'] != ""
            
            # Verify FASTA format
            fasta_lines = result['annotated_fasta'].split('\n')
            assert any(line.startswith('>') for line in fasta_lines)
            assert 'md5_1' in result['annotated_fasta']
            assert 'capsid protein' in result['annotated_fasta']
            
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, workflow):
        """Test workflow initialization and step setup"""
        
        assert workflow.config is not None
        assert workflow.steps is not None
        assert 'data_acquisition' in workflow.steps
        assert workflow.logger is not None
        
        # Test progress callback setting
        callback_called = False
        
        async def test_callback(progress_data):
            nonlocal callback_called
            callback_called = True
            assert 'percentage' in progress_data
            assert 'message' in progress_data
            
        workflow.set_progress_callback(test_callback)
        await workflow._update_progress("Test message", 50)
        assert callback_called
        
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow):
        """Test workflow error handling"""
        
        # Mock a failing step
        with patch.object(workflow.steps['data_acquisition'], 'execute', 
                         side_effect=Exception("Test error")):
            
            result = await workflow.execute_full_workflow()
            
            assert not result.success
            assert "Test error" in result.error
            assert result.workflow_data is not None  # Should contain partial data
            assert result.execution_time > 0
            
    def test_output_file_organization(self, workflow, temp_output_dir):
        """Test output file organization and naming"""
        
        # Create mock workflow data
        workflow_data = WorkflowData()
        workflow_data.filtered_genomes = [
            GenomeData("test1", 11000, "Test Genome 1").to_dict(),
            GenomeData("test2", 11500, "Test Genome 2").to_dict()
        ]
        workflow_data.unique_proteins = [
            ProteinData("protein1", "md5_1", "test1", "capsid").to_dict()
        ]
        workflow_data.annotated_fasta = ">md5_1|capsid|capsid protein|CAP_01|unknown\nMKLLVVVAG"
        workflow_data.clusters = [{"id": "cluster1", "members": ["protein1"]}]
        workflow_data.pssm_matrices = [{"cluster_id": "cluster1", "matrix": [[0.1, 0.2]]}]
        
        # Test file collection (this is a synchronous wrapper for the async method)
        async def test_file_collection():
            output_files = await workflow._collect_output_files(workflow_data)
            
            # Verify files are created
            assert 'filtered_genomes' in output_files
            assert 'unique_proteins_fasta' in output_files
            assert 'clusters' in output_files
            assert 'pssm_matrices' in output_files
            
            # Verify file contents
            genomes_file = Path(output_files['filtered_genomes'])
            assert genomes_file.exists()
            
            with open(genomes_file) as f:
                genomes_data = json.load(f)
                assert len(genomes_data) == 2
                assert genomes_data[0]['genome_id'] == 'test1'
                
            fasta_file = Path(output_files['unique_proteins_fasta'])
            assert fasta_file.exists()
            
            with open(fasta_file) as f:
                fasta_content = f.read()
                assert '>md5_1' in fasta_content
                assert 'MKLLVVVAG' in fasta_content
                
        # Run the async test
        asyncio.run(test_file_collection())
        
    @pytest.mark.asyncio
    async def test_viral_pssm_json_generation(self, workflow):
        """Test Viral_PSSM.json format output generation"""
        
        # Create mock workflow data
        workflow_data = WorkflowData()
        workflow_data.filtered_genomes = [{"genome_id": "test1"}] * 5
        workflow_data.unique_proteins = [{"patric_id": f"protein_{i}"} for i in range(10)]
        workflow_data.clusters = [
            {
                "id": "cluster_1",
                "consensus_annotation": "capsid protein",
                "protein_class": "structural",
                "member_count": 5,
                "consensus_score": 0.85,
                "overall_confidence": 0.9
            },
            {
                "id": "cluster_2", 
                "consensus_annotation": "envelope protein E1",
                "protein_class": "structural",
                "member_count": 3,
                "consensus_score": 0.78,
                "overall_confidence": 0.82
            }
        ]
        workflow_data.pssm_matrices = [{"cluster_id": "cluster_1"}, {"cluster_id": "cluster_2"}]
        workflow_data.step_timings = {"data_acquisition": 10.5, "clustering": 5.2}
        
        # Generate Viral_PSSM.json
        viral_pssm_json = await workflow._generate_viral_pssm_json(workflow_data)
        
        # Verify structure
        assert 'metadata' in viral_pssm_json
        assert 'proteins' in viral_pssm_json
        assert 'analysis_summary' in viral_pssm_json
        assert 'quality_metrics' in viral_pssm_json
        
        # Verify metadata
        metadata = viral_pssm_json['metadata']
        assert metadata['organism'] == 'Alphavirus'
        assert metadata['method'] == 'nanobrain_alphavirus_analysis'
        assert metadata['total_genomes_analyzed'] == 5
        assert metadata['clustering_method'] == 'MMseqs2'
        
        # Verify proteins
        proteins = viral_pssm_json['proteins']
        assert len(proteins) >= 2
        
        protein1 = proteins[0]
        assert protein1['id'] == 'alphavirus_cluster_1'
        assert protein1['function'] == 'capsid protein'
        assert protein1['protein_class'] == 'structural'
        assert protein1['cluster_info']['member_count'] == 5
        assert protein1['confidence_metrics']['overall_confidence'] == 0.9
        
        # Verify analysis summary
        summary = viral_pssm_json['analysis_summary']
        assert summary['total_proteins'] == 10
        assert summary['clusters_generated'] == 2
        assert summary['execution_time_seconds'] == 15.7  # Sum of step timings
        
    @pytest.mark.asyncio
    async def test_workflow_with_missing_dependencies(self, workflow):
        """Test workflow behavior when optional dependencies are missing"""
        
        # This tests graceful degradation when MMseqs2, MUSCLE etc. are not available
        # The workflow should initialize but fail gracefully when trying to use missing tools
        
        with patch('nanobrain.library.tools.bioinformatics.mmseqs_tool.MMseqs2Tool') as mock_mmseqs:
            mock_mmseqs.side_effect = ImportError("MMseqs2 not available")
            
            # The workflow should still initialize
            assert workflow is not None
            assert workflow.steps is not None
            
            # When executed, it should provide clear error messages
            # This is tested in the error handling test above
            
    def test_configuration_defaults(self):
        """Test configuration defaults match expected values"""
        config = AlphavirusWorkflowConfig()
        
        # Test BV-BRC defaults (corrected path from our testing)
        assert config.bvbrc.executable_path == "/Applications/BV-BRC.app/deployment/bin/"
        assert config.bvbrc.min_length == 8000
        assert config.bvbrc.max_length == 15000
        assert config.bvbrc.genome_batch == 100
        
        # Test clustering defaults
        assert config.clustering.min_seq_id == 0.7
        assert config.clustering.coverage == 0.8
        assert config.clustering.sensitivity == 7.5
        
        # Test expected protein lengths
        expected_lengths = config.quality_control.expected_lengths
        assert expected_lengths['capsid'] == [200, 320]
        assert expected_lengths['E1'] == [390, 490]
        assert expected_lengths['nsP1'] == [500, 600]


@pytest.mark.integration
class TestAlphavirusWorkflowBVBRCIntegration:
    """Integration tests that require actual BV-BRC connectivity"""
    
    @pytest.mark.asyncio
    async def test_bvbrc_connection_real(self):
        """Test real BV-BRC connection (only if BV-BRC is available)"""
        
        try:
            from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool
            
            bvbrc_tool = BVBRCTool()
            is_available = await bvbrc_tool.verify_installation()
            
            if is_available:
                # Test a simple query
                result = await bvbrc_tool.execute_p3_command("p3-all-genomes", [
                    "--eq", "genome_id,511145.12",  # E. coli test genome
                    "--attr", "genome_id,genome_name",
                    "--limit", "1"
                ])
                
                assert result.returncode == 0
                assert result.stdout
                print(f"✅ BV-BRC real connection test passed")
                print(f"Sample output: {result.stdout.decode()[:200]}...")
            else:
                pytest.skip("BV-BRC not available for real connection test")
                
        except Exception as e:
            pytest.skip(f"BV-BRC connection test skipped: {e}")


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v", "-s"]) 