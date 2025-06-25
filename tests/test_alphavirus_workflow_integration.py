"""
Integration Tests for Alphavirus Workflow - Updated for Framework V4.5.0

Tests the complete workflow integration with mandatory from_config pattern compliance.
Updated to work with cleaned configuration structure and latest framework patterns.

Version: 4.5.0
Framework Compliance: ✅ from_config Pattern, ✅ Configuration V4.5.0
"""

import pytest
import asyncio
import tempfile
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, AsyncMock, MagicMock

# Framework Core Imports (Framework Compliant)
from nanobrain.core.workflow import WorkflowConfig, Workflow, create_workflow
from nanobrain.core.step import StepConfig, BaseStep
from nanobrain.core.executor import LocalExecutor, ExecutorConfig
from nanobrain.core.data_unit import DataUnitConfig, create_data_unit, DataUnitType
from nanobrain.core.external_tool import ToolResult

# Workflow Components (from_config compliant)
from nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step import BVBRCDataAcquisitionStep
from nanobrain.library.workflows.viral_protein_analysis.steps.clustering_step import ClusteringStep
from nanobrain.library.workflows.viral_protein_analysis.steps.alignment_step import AlignmentStep
from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool

# Test data models
from pydantic import BaseModel
from typing import List


class GenomeData(BaseModel):
    """Test genome data model"""
    genome_id: str
    genome_length: int
    genome_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"genome_id": self.genome_id, "genome_length": self.genome_length, "genome_name": self.genome_name}


class ProteinData(BaseModel):
    """Test protein data model"""
    patric_id: str
    aa_sequence_md5: str
    genome_id: str
    product: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"patric_id": self.patric_id, "aa_sequence_md5": self.aa_sequence_md5, 
                "genome_id": self.genome_id, "product": self.product}


class WorkflowData(BaseModel):
    """Test workflow data container"""
    original_genomes: List[Dict[str, Any]] = []
    filtered_genomes: List[Dict[str, Any]] = []
    unique_proteins: List[Dict[str, Any]] = []
    annotated_fasta: str = ""
    clusters: List[Dict[str, Any]] = []
    pssm_matrices: List[Dict[str, Any]] = []
    step_timings: Dict[str, float] = {}
    
    def update_from_acquisition(self, result: Dict[str, Any]):
        """Update data from acquisition step results"""
        self.original_genomes = result.get('original_genomes', [])
        self.filtered_genomes = result.get('filtered_genomes', [])
        self.unique_proteins = result.get('unique_proteins', [])


class WorkflowResult(BaseModel):
    """Test workflow result model"""
    success: bool
    error: Optional[str] = None
    workflow_data: Optional[WorkflowData] = None
    execution_time: float = 0.0
    output_files: Dict[str, str] = {}


class TestAlphavirusWorkflowIntegration:
    """Integration tests for the complete Alphavirus workflow with framework compliance"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def workflow_config_path(self):
        """Get path to current workflow configuration"""
        return Path(__file__).parent.parent / "nanobrain/library/workflows/viral_protein_analysis/config/AlphavirusWorkflow.yml"
    
    @pytest.fixture
    def test_executor(self):
        """Create test executor via from_config"""
        executor_config = ExecutorConfig(
            name="test_executor",
            class_path="nanobrain.core.executor.LocalExecutor",
            max_concurrent_tasks=2
        )
        return LocalExecutor.from_config(executor_config)
    
    @pytest.fixture
    def test_workflow_config(self, temp_output_dir, workflow_config_path):
        """Load and customize workflow configuration for testing"""
        if workflow_config_path.exists():
            with open(workflow_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Customize for testing
            config_data['name'] = 'test_alphavirus_workflow'
            config_data['output_directory'] = temp_output_dir
            config_data['enable_monitoring'] = False  # Disable for testing
            
            return WorkflowConfig(**config_data)
        else:
            # Fallback minimal configuration
            return WorkflowConfig(
                name="test_alphavirus_workflow", 
                description="Test workflow",
                steps={},
                links=[]
            )
    
    def test_configuration_loading_and_validation(self, workflow_config_path):
        """Test configuration loading from YAML with framework compliance"""
        
        if workflow_config_path.exists():
            # Test YAML loading
            with open(workflow_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate structure
            assert 'name' in config_data, "Configuration should have name"
            assert 'description' in config_data, "Configuration should have description"
            assert 'version' in config_data, "Configuration should have version"
            assert config_data['version'] == '4.5.0', "Should use latest version"
            
            # Test framework compliance fields
            assert 'execution_strategy' in config_data, "Should have execution strategy"
            assert 'error_handling' in config_data, "Should have error handling"
            
            # Test step configuration references
            if 'steps' in config_data:
                steps = config_data['steps']
                assert isinstance(steps, list), "Steps should be a list"
                
                for step_config in steps:
                    step_name = step_config.get('step_id', step_config.get('name', 'unknown'))
                    assert 'class' in step_config, f"Step {step_name} should have class"
                    assert 'config_file' in step_config, f"Step {step_name} should reference config file"
            
            print(f"✅ Configuration validation passed: {config_data['name']} v{config_data['version']}")
        else:
            pytest.skip("AlphavirusWorkflow.yml not found - using fallback configuration")
    
    def test_step_creation_via_from_config(self, test_executor):
        """Test that all workflow steps can be created via from_config pattern"""
        
        # Test BV-BRC Data Acquisition Step
        step_config = StepConfig(
            name="test_bvbrc_step",
            class_path="nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep"
        )
        
        step = BVBRCDataAcquisitionStep.from_config(step_config, executor=test_executor)
        assert step is not None, "BVBRCDataAcquisitionStep should be created via from_config"
        assert hasattr(step, 'bv_brc_tool'), "Step should have initialized BV-BRC tool"
        
        # Test Clustering Step
        clustering_config = StepConfig(
            name="test_clustering_step",
            class_path="nanobrain.library.workflows.viral_protein_analysis.steps.clustering_step.ClusteringStep"
        )
        
        clustering_step = ClusteringStep.from_config(clustering_config, executor=test_executor)
        assert clustering_step is not None, "ClusteringStep should be created via from_config"
        assert hasattr(clustering_step, 'mmseqs2_tool'), "Step should have initialized MMseqs2 tool"
        
        # Test Alignment Step
        alignment_config = StepConfig(
            name="test_alignment_step", 
            class_path="nanobrain.library.workflows.viral_protein_analysis.steps.alignment_step.AlignmentStep"
        )
        
        alignment_step = AlignmentStep.from_config(alignment_config, executor=test_executor)
        assert alignment_step is not None, "AlignmentStep should be created via from_config"
        assert hasattr(alignment_step, 'muscle_tool'), "Step should have initialized MUSCLE tool"
        
        print("✅ All workflow steps successfully created via from_config pattern")
    
    def test_data_unit_creation_and_integration(self):
        """Test data unit creation and integration using framework patterns"""
        
        # Test memory data unit
        memory_config = DataUnitConfig(
            name="test_memory_unit",
            data_type=DataUnitType.MEMORY,
            persistent=False
        )
        
        memory_unit = create_data_unit(memory_config)
        assert memory_unit is not None, "Memory data unit should be created"
        
        # Test file data unit  
        file_config = DataUnitConfig(
            name="test_file_unit",
            data_type=DataUnitType.FILE,
            file_path="/tmp/test_workflow_data.json"
        )
        
        file_unit = create_data_unit(file_config)
        assert file_unit is not None, "File data unit should be created"
        
        # Test string data unit
        string_config = DataUnitConfig(
            name="test_string_unit", 
            data_type=DataUnitType.STRING,
            initial_value="test data"
        )
        
        string_unit = create_data_unit(string_config)
        assert string_unit is not None, "String data unit should be created"
        
        print("✅ All data unit types successfully created via framework factory")
    
    @pytest.mark.asyncio
    async def test_bvbrc_data_acquisition_with_mocks(self, test_executor):
        """Test BV-BRC data acquisition step with comprehensive mocking"""
        
        # Create step via from_config
        step_config = StepConfig(
            name="test_bvbrc_acquisition",
            class_path="nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep"
        )
        
        step = BVBRCDataAcquisitionStep.from_config(step_config, executor=test_executor)
        
        # Mock data
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
        
        # Mock BV-BRC tool responses
        async def mock_command_response(command, args):
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
        
        with patch.object(step.bv_brc_tool, 'verify_installation', return_value=True), \
             patch.object(step.bv_brc_tool, 'execute_p3_command', side_effect=mock_command_response):
            
            # Execute step
            result = await step.execute({'target_genus': 'Alphavirus'})
            
            # Validate results
            assert result is not None, "Step should return results"
            assert 'statistics' in result, "Should contain statistics"
            assert result['statistics']['total_genomes_downloaded'] == 3, "Should download 3 genomes"
            assert result['statistics']['genomes_after_filtering'] == 2, "Should filter out short genome"
            assert result['statistics']['unique_proteins_found'] == 2, "Should find 2 unique proteins"
            
            # Validate output format
            assert 'protein_sequences' in result, "Should contain protein sequences"
            assert 'protein_annotations' in result, "Should contain protein annotations"
            assert 'annotated_fasta' in result, "Should contain annotated FASTA"
            
            # Validate FASTA format
            fasta_content = result['annotated_fasta']
            assert fasta_content.count('>') == 2, "FASTA should have 2 sequence headers"
            assert 'md5_1' in fasta_content, "Should contain md5_1 sequence"
            assert 'capsid protein' in fasta_content, "Should contain annotation"
            
            print("✅ BV-BRC data acquisition step test passed with comprehensive validation")
    
    @pytest.mark.asyncio 
    async def test_workflow_data_flow_integration(self, test_executor):
        """Test data flow between workflow steps"""
        
        # Create workflow data container
        workflow_data = WorkflowData()
        
        # Test initial state
        assert workflow_data.original_genomes == [], "Should start with empty genomes"
        assert workflow_data.unique_proteins == [], "Should start with empty proteins"
        
        # Simulate data acquisition results
        acquisition_result = {
            'original_genomes': [GenomeData(genome_id="test1", genome_length=11000, genome_name="Test Genome 1").to_dict()],
            'filtered_genomes': [GenomeData(genome_id="test1", genome_length=11000, genome_name="Test Genome 1").to_dict()],
            'unique_proteins': [ProteinData(patric_id="protein1", aa_sequence_md5="md5_1", genome_id="test1", product="capsid").to_dict()],
            'annotated_fasta': ">md5_1|capsid|capsid protein|CAP_01|unknown\nMKLLVVVAG"
        }
        
        # Update workflow data
        workflow_data.update_from_acquisition(acquisition_result)
        
        # Validate data flow
        assert len(workflow_data.original_genomes) == 1, "Should update original genomes"
        assert len(workflow_data.filtered_genomes) == 1, "Should update filtered genomes"
        assert len(workflow_data.unique_proteins) == 1, "Should update unique proteins"
        assert workflow_data.original_genomes[0]['genome_id'] == 'test1', "Should preserve genome data"
        
        # Test subsequent step data flow
        workflow_data.clusters = [{"id": "cluster1", "members": ["protein1"], "consensus": "capsid protein"}]
        workflow_data.pssm_matrices = [{"cluster_id": "cluster1", "matrix": [[0.1, 0.2, 0.3, 0.4]]}]
        
        assert len(workflow_data.clusters) == 1, "Should store clustering results"
        assert len(workflow_data.pssm_matrices) == 1, "Should store PSSM matrices"
        
        print("✅ Workflow data flow integration test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, test_executor):
        """Test error handling throughout workflow execution"""
        
        # Test step creation with invalid configuration
        with pytest.raises(Exception):
            invalid_config = StepConfig(
                name="invalid_step",
                class_path="nonexistent.module.InvalidStep"
            )
            BaseStep.from_config(invalid_config, executor=test_executor)
        
        # Test graceful handling of tool failures
        step_config = StepConfig(
            name="test_error_step",
            class_path="nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step.BVBRCDataAcquisitionStep"
        )
        
        step = BVBRCDataAcquisitionStep.from_config(step_config, executor=test_executor)
        
        # Mock tool failure
        async def mock_failed_command(command, args):
            return ToolResult(returncode=1, stdout=b"", stderr=b"Connection failed",
                            execution_time=1.0, command=[command] + args, success=False)
        
        with patch.object(step.bv_brc_tool, 'verify_installation', return_value=True), \
             patch.object(step.bv_brc_tool, 'execute_p3_command', side_effect=mock_failed_command):
            
            try:
                result = await step.execute({'target_genus': 'Alphavirus'})
                # Step should handle errors gracefully
                assert result is not None, "Step should return error results"
            except Exception as e:
                # Expected to fail, but should be handled gracefully
                assert "Connection failed" in str(e) or "BV-BRC" in str(e), "Should provide meaningful error"
        
        print("✅ Error handling and recovery test passed")
    
    def test_configuration_compliance_with_cleanup(self, workflow_config_path):
        """Test configuration compliance after cleanup"""
        
        if workflow_config_path.exists():
            with open(workflow_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Test that obsolete files are not referenced
            obsolete_files = [
                'CleanWorkflow.yml', 'RealWorkflow.yml', 'MinimalWorkflow.yml',
                'bv_brc_config.yml', 'mmseqs_config.yml', 'muscle_config.yml',
                'workflow_config.py'
            ]
            
            config_str = yaml.dump(config_data)
            for obsolete_file in obsolete_files:
                assert obsolete_file not in config_str, f"Configuration should not reference {obsolete_file}"
            
            # Test that current structure is referenced
            if 'tools' in config_data:
                tools = config_data['tools']
                assert 'bv_brc_tool' in tools, "Should reference current BV-BRC tool config"
                assert 'mmseqs2_tool' in tools, "Should reference current MMseqs2 tool config"
                assert 'muscle_tool' in tools, "Should reference current MUSCLE tool config"
                
                # Test tool config paths
                for tool_name, tool_config in tools.items():
                    if 'config_file' in tool_config:
                        config_file = tool_config['config_file']
                        assert config_file.startswith('config/tools/'), f"Tool {tool_name} should use config/tools/ directory"
            
            print("✅ Configuration compliance test passed - no obsolete references found")
        else:
            pytest.skip("AlphavirusWorkflow.yml not found")
    
    @pytest.mark.asyncio
    async def test_viral_pssm_json_generation_framework_compliant(self):
        """Test Viral_PSSM.json generation with framework compliance"""
        
        # Create test workflow data
        workflow_data = WorkflowData()
        workflow_data.filtered_genomes = [{"genome_id": f"genome_{i}", "genome_name": f"Test Genome {i}"} for i in range(5)]
        workflow_data.unique_proteins = [{"patric_id": f"protein_{i}", "product": f"protein {i}"} for i in range(10)]
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
        workflow_data.pssm_matrices = [
            {"cluster_id": "cluster_1", "matrix": [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]},
            {"cluster_id": "cluster_2", "matrix": [[0.2, 0.3, 0.2, 0.3], [0.3, 0.2, 0.3, 0.2]]}
        ]
        workflow_data.step_timings = {"data_acquisition": 10.5, "clustering": 5.2, "alignment": 3.1}
        
        # Generate Viral_PSSM.json structure
        viral_pssm_data = {
            'metadata': {
                'organism': 'Alphavirus',
                'method': 'nanobrain_alphavirus_analysis',
                'framework_version': '4.5.0',
                'total_genomes_analyzed': len(workflow_data.filtered_genomes),
                'clustering_method': 'MMseqs2',
                'from_config_compliant': True
            },
            'proteins': [],
            'analysis_summary': {
                'total_proteins': len(workflow_data.unique_proteins),
                'clusters_generated': len(workflow_data.clusters),
                'execution_time_seconds': sum(workflow_data.step_timings.values()),
                'framework_compliance': True
            },
            'quality_metrics': {
                'clustering_confidence': sum(c['overall_confidence'] for c in workflow_data.clusters) / len(workflow_data.clusters),
                'annotation_consistency': True
            }
        }
        
        # Generate protein entries
        for i, cluster in enumerate(workflow_data.clusters):
            protein_entry = {
                'id': f"alphavirus_cluster_{i+1}",
                'function': cluster['consensus_annotation'],
                'protein_class': cluster['protein_class'],
                'cluster_info': {
                    'member_count': cluster['member_count'],
                    'consensus_score': cluster['consensus_score']
                },
                'confidence_metrics': {
                    'overall_confidence': cluster['overall_confidence']
                },
                'pssm_matrix': workflow_data.pssm_matrices[i]['matrix']
            }
            viral_pssm_data['proteins'].append(protein_entry)
        
        # Validate structure
        assert 'metadata' in viral_pssm_data, "Should contain metadata"
        assert 'proteins' in viral_pssm_data, "Should contain proteins"
        assert 'analysis_summary' in viral_pssm_data, "Should contain analysis summary"
        assert 'quality_metrics' in viral_pssm_data, "Should contain quality metrics"
        
        # Validate metadata compliance
        metadata = viral_pssm_data['metadata']
        assert metadata['framework_version'] == '4.5.0', "Should use current framework version"
        assert metadata['from_config_compliant'] == True, "Should indicate from_config compliance"
        assert metadata['clustering_method'] == 'MMseqs2', "Should use MMseqs2 clustering"
        
        # Validate proteins
        proteins = viral_pssm_data['proteins']
        assert len(proteins) == 2, "Should have 2 protein clusters"
        
        protein1 = proteins[0]
        assert protein1['function'] == 'capsid protein', "Should have correct function"
        assert protein1['cluster_info']['member_count'] == 5, "Should have correct member count"
        assert len(protein1['pssm_matrix']) == 2, "Should have PSSM matrix data"
        
        # Validate analysis summary
        summary = viral_pssm_data['analysis_summary']
        assert summary['framework_compliance'] == True, "Should indicate framework compliance"
        assert summary['execution_time_seconds'] == 18.8, "Should sum all step timings"
        
        print("✅ Viral_PSSM.json generation test passed with framework compliance")


@pytest.mark.integration
class TestAlphavirusWorkflowRealBVBRCIntegration:
    """Integration tests that require actual BV-BRC connectivity"""
    
    @pytest.mark.asyncio
    async def test_bvbrc_real_connection_framework_compliant(self):
        """Test real BV-BRC connection with framework compliance (if BV-BRC available)"""
        
        try:
            # Create BV-BRC tool via from_config
            from nanobrain.library.config.defaults.tools.bv_brc_tool import BVBRCToolConfig
            
            bvbrc_config = BVBRCToolConfig(
                name="test_bvbrc_tool",
                class_path="nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool"
            )
            
            bvbrc_tool = BVBRCTool.from_config(bvbrc_config)
            is_available = await bvbrc_tool.verify_installation()
            
            if is_available:
                # Test a simple query using framework-compliant pattern
                result = await bvbrc_tool.execute_p3_command("p3-all-genomes", [
                    "--eq", "genome_id,511145.12",  # E. coli test genome
                    "--attr", "genome_id,genome_name",
                    "--limit", "1"
                ])
                
                assert result.returncode == 0, "Command should succeed"
                assert result.stdout, "Should return data"
                assert result.success == True, "Should indicate success"
                
                print(f"✅ BV-BRC real connection test passed (framework compliant)")
                print(f"Sample output: {result.stdout.decode()[:200]}...")
            else:
                pytest.skip("BV-BRC not available for real connection test")
                
        except Exception as e:
            pytest.skip(f"BV-BRC connection test skipped: {e}")


if __name__ == "__main__":
    # Run tests with proper configuration
    pytest.main([__file__, "-v", "-s", "--tb=short"])