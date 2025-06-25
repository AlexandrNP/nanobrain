"""
Working Tests for External Bioinformatics Tools

This module tests all external tools with proper async handling and correct signatures.
"""

import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import the tools to test
from nanobrain.core.external_tool import (
    ExternalTool, 
    ExternalToolConfig, 
    ToolResult,
    InstallationStatus,
    DiagnosticReport,
    ToolInstallationError,
    ToolExecutionError
)
from nanobrain.library.tools.bioinformatics.bv_brc_tool import (
    BVBRCTool, BVBRCConfig, GenomeData, ProteinData, 
    BVBRCInstallationError, BVBRCDataError
)
from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config
from nanobrain.library.tools.bioinformatics.muscle_tool import MUSCLETool, MUSCLEConfig
# from nanobrain.library.tools.bioinformatics.pssm_generator_tool import (
#     PSSMGeneratorTool, PSSMConfig
# )
from nanobrain.library.infrastructure.data.email_manager import EmailManager
from nanobrain.library.infrastructure.data.cache_manager import CacheManager


class TestExternalToolBase:
    """Test the base external tool functionality"""
    
    def test_external_tool_config_creation(self):
        """Test creating external tool configuration"""
        config = ExternalToolConfig(
            tool_name="test_tool",
            installation_path="/usr/local/bin",
            executable_path="/usr/local/bin/test_tool"
        )
        
        assert config.tool_name == "test_tool"
        assert config.installation_path == "/usr/local/bin"
        assert config.executable_path == "/usr/local/bin/test_tool"
    
    def test_installation_status_creation(self):
        """Test creating installation status"""
        status = InstallationStatus()
        status.found = True
        status.installation_type = "conda"
        status.version = "1.0.0"
        
        assert status.found == True
        assert status.installation_type == "conda"
        assert status.version == "1.0.0"
    
    def test_tool_result_creation(self):
        """Test creating tool results with correct signature"""
        result = ToolResult(
            returncode=0,
            stdout=b"test output",
            stderr=b"",
            execution_time=1.5,
            command=["test", "command"],
            success=True
        )
        
        assert result.success
        assert result.stdout_text == "test output"
        assert result.stderr_text == ""
        assert result.returncode == 0
        assert result.command == ["test", "command"]


class TestExternalToolCore:
    """Test core external tool functionality that was moved from bioinformatics module"""
    
    def test_diagnostic_report_creation(self):
        """Test creating diagnostic report"""
        status = InstallationStatus()
        status.found = True
        status.installation_type = "local"
        
        report = DiagnosticReport(
            tool_name="test_tool",
            installation_status=status
        )
        
        assert report.tool_name == "test_tool"
        assert report.installation_status.found == True
        assert report.installation_status.installation_type == "local"
    
    def test_error_classes_hierarchy(self):
        """Test that error classes have correct inheritance"""
        # Test ToolInstallationError inheritance
        error = ToolInstallationError("Test installation error")
        assert isinstance(error, Exception)
        assert str(error) == "Test installation error"
        
        # Test ToolExecutionError inheritance
        error = ToolExecutionError("Test execution error")
        assert isinstance(error, Exception)
        assert str(error) == "Test execution error"


class TestBVBRCTool:
    """Test BV-BRC tool functionality"""
    
    @pytest.fixture
    def bvbrc_config(self):
        """Create test BV-BRC configuration"""
        return BVBRCConfig(
            tool_name="bv_brc",
            installation_path="/Applications/BV-BRC.app/",
            executable_path="/Applications/BV-BRC.app/Contents/Resources/deployment/bin/",
            genome_batch_size=10,
            md5_batch_size=5,
            verify_on_init=False  # Disable async initialization for testing
        )
    
    @pytest.fixture
    def mock_email_manager(self):
        """Create mock email manager"""
        mock_manager = Mock(spec=EmailManager)
        mock_manager.get_email_for_service.return_value = None  # Anonymous access
        mock_manager.should_authenticate.return_value = False
        return mock_manager
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager"""
        mock_manager = AsyncMock(spec=CacheManager)
        mock_manager.get_cached_response.return_value = None
        mock_manager.cache_response.return_value = None
        return mock_manager
    
    @pytest.fixture
    def bvbrc_tool(self, bvbrc_config, mock_email_manager, mock_cache_manager):
        """Create BV-BRC tool instance"""
        return BVBRCTool(bvbrc_config)
    
    def test_genome_data_creation(self):
        """Test genome data container"""
        genome = GenomeData(
            genome_id="12345.1",
            genome_length=11700,
            genome_name="Test Alphavirus",
            taxon_lineage="Viruses; Togaviridae; Alphavirus"
        )
        
        assert genome.genome_id == "12345.1"
        assert genome.genome_length == 11700
        assert "Alphavirus" in genome.taxon_lineage
    
    def test_protein_data_creation(self):
        """Test protein data container"""
        protein = ProteinData(
            patric_id="fig|12345.1.peg.100",
            aa_sequence_md5="abc123def456",
            product="capsid protein",
            aa_sequence="MKLLVVVAG"
        )
        
        assert protein.patric_id == "fig|12345.1.peg.100"
        assert protein.product == "capsid protein"
        assert protein.aa_sequence == "MKLLVVVAG"
    
    @pytest.mark.asyncio
    async def test_installation_verification_missing(self, bvbrc_tool):
        """Test installation verification when BV-BRC is missing"""
        with patch('pathlib.Path.exists', return_value=False):
            # verify_installation returns False, doesn't raise exception
            result = await bvbrc_tool.verify_installation()
            assert result == False
    
    @pytest.mark.asyncio
    async def test_installation_verification_success(self, bvbrc_tool):
        """Test successful installation verification"""
        mock_result = ToolResult(
            returncode=0,
            stdout=b"genome_id\tgenome_name\n511145.12\tEscherichia coli",
            stderr=b"",
            execution_time=1.0,
            command=["p3-all-genomes", "--help"],
            success=True
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(bvbrc_tool, 'execute_p3_command', return_value=mock_result):
                result = await bvbrc_tool.verify_installation()
                assert result == True
    
    @pytest.mark.asyncio
    async def test_installation_verification_headers_only(self, bvbrc_tool):
        """Test installation verification with headers only (no data)"""
        mock_result = ToolResult(
            returncode=0,
            stdout=b"genome_id\tgenome_name\n",  # Only headers
            stderr=b"",
            execution_time=1.0,
            command=["p3-all-genomes", "--help"],
            success=True
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(bvbrc_tool, 'execute_p3_command', return_value=mock_result):
                # verify_installation returns False, doesn't raise exception
                result = await bvbrc_tool.verify_installation()
                assert result == False
    
    @pytest.mark.asyncio
    async def test_alphavirus_genomes_download(self, bvbrc_tool):
        """Test downloading Alphavirus genomes"""
        mock_output = (
            b"genome_id\tgenome_length\tgenome_name\ttaxon_lineage_names\n"
            b"12345.1\t11700\tEastern Equine Encephalitis Virus\tViruses; Togaviridae; Alphavirus\n"
            b"12346.1\t11800\tWestern Equine Encephalitis Virus\tViruses; Togaviridae; Alphavirus\n"
        )
        
        mock_result = ToolResult(
            returncode=0,
            stdout=mock_output,
            stderr=b"",
            execution_time=2.0,
            command=["p3-all-genomes"],
            success=True
        )
        
        with patch.object(bvbrc_tool, 'execute_p3_command', return_value=mock_result):
            genomes = await bvbrc_tool.download_alphavirus_genomes()
            
            assert len(genomes) == 2
            assert genomes[0].genome_id == "12345.1"
            assert genomes[0].genome_length == 11700
            assert "Alphavirus" in genomes[0].taxon_lineage
    
    @pytest.mark.asyncio
    async def test_genome_filtering_by_size(self, bvbrc_tool):
        """Test filtering genomes by size"""
        test_genomes = [
            GenomeData("12345.1", 11700, "EEEV", "Alphavirus"),  # Valid
            GenomeData("12346.1", 5000, "Fragment", "Alphavirus"),  # Too small
            GenomeData("12347.1", 20000, "Large", "Alphavirus"),  # Too large
            GenomeData("12348.1", 11800, "WEE", "Alphavirus")  # Valid
        ]
        
        filtered = await bvbrc_tool.filter_genomes_by_size(test_genomes)
        
        assert len(filtered) == 2
        assert all(8000 <= g.genome_length <= 15000 for g in filtered)


class TestToolConfiguration:
    """Test tool configuration handling"""
    
    def test_mmseqs_config_creation(self):
        """Test MMseqs2 configuration"""
        config = MMseqs2Config(
            tool_name="mmseqs2",
            executable_path="/usr/local/bin/mmseqs",
            min_seq_id=0.7,
            coverage=0.8,
            sensitivity=7.5,
            verify_on_init=False
        )
        
        assert config.tool_name == "mmseqs2"
        assert config.min_seq_id == 0.7
        assert config.coverage == 0.8
        assert config.sensitivity == 7.5
    
    def test_muscle_config_creation(self):
        """Test MUSCLE configuration"""
        config = MUSCLEConfig(
            tool_name="muscle",
            executable_path="/usr/local/bin/muscle",
            max_iterations=16,
            verify_on_init=False
        )
        
        assert config.tool_name == "muscle"
        assert config.max_iterations == 16
    
    # def test_pssm_config_creation(self):
    #     """Test PSSM configuration"""
    #     config = PSSMConfig(
    #         tool_name="pssm_generator",
    #         amino_acid_alphabet="ACDEFGHIKLMNPQRSTVWY",
    #         pseudocount=0.01,
    #         verify_on_init=False
    #     )
    #     
    #     assert config.tool_name == "pssm_generator"
    #     assert config.amino_acid_alphabet == "ACDEFGHIKLMNPQRSTVWY"
    #     assert config.pseudocount == 0.01


class TestToolMockVerification:
    """Test that tools can be properly mocked for testing"""
    
    def test_bvbrc_tool_with_disabled_init(self):
        """Test BV-BRC tool creation with disabled initialization"""
        config = BVBRCConfig(
            tool_name="bv_brc_test",
            verify_on_init=False
        )
        
        # This should not raise an error
        tool = BVBRCTool(config)
        assert tool.name == "bv_brc_test"
    
    def test_mmseqs_tool_with_disabled_init(self):
        """Test MMseqs2 tool creation with disabled initialization"""
        config = MMseqs2Config(
            tool_name="mmseqs2_test",
            verify_on_init=False
        )
        
        # This should not raise an error
        tool = MMseqs2Tool(config)
        assert tool.name == "mmseqs2_test"
    
    def test_muscle_tool_with_disabled_init(self):
        """Test MUSCLE tool creation with disabled initialization"""
        config = MUSCLEConfig(
            tool_name="muscle_test",
            verify_on_init=False
        )
        
        # This should not raise an error
        tool = MUSCLETool(config)
        assert tool.name == "muscle_test"
    
    # def test_pssm_tool_with_disabled_init(self):
    #     """Test PSSM tool creation with disabled initialization"""
    #     config = PSSMConfig(
    #         tool_name="pssm_test",
    #         verify_on_init=False
    #     )
    #     
    #     # This should not raise an error
    #     tool = PSSMGeneratorTool(config)
    #     assert tool.tool_name == "pssm_test"


class TestToolIntegration:
    """Test integration between different tools"""
    
    @pytest.mark.asyncio
    async def test_workflow_simulation(self):
        """Test a simple workflow simulation"""
        # Create tools with disabled initialization
        bvbrc_config = BVBRCConfig(tool_name="bv_brc_sim", verify_on_init=False)
        mmseqs_config = MMseqs2Config(tool_name="mmseqs_sim", verify_on_init=False)
        muscle_config = MUSCLEConfig(tool_name="muscle_sim", verify_on_init=False)
        # pssm_config = PSSMConfig(tool_name="pssm_sim", verify_on_init=False)
        
        bvbrc_tool = BVBRCTool(bvbrc_config)
        mmseqs_tool = MMseqs2Tool(mmseqs_config)
        muscle_tool = MUSCLETool(muscle_config)
        # pssm_tool = PSSMGeneratorTool(pssm_config)
        
        # Test that all tools can be created
        assert bvbrc_tool.name == "bv_brc_sim"
        assert mmseqs_tool.name == "mmseqs_sim"
        assert muscle_tool.name == "muscle_sim"
        # assert pssm_tool.tool_name == "pssm_sim"
        
        # Test basic configuration access
        assert bvbrc_tool.bio_config.genome_batch_size >= 0
        assert mmseqs_tool.mmseqs_config.min_seq_id > 0
        assert muscle_tool.muscle_config.max_iterations > 0
        # assert pssm_tool.pssm_config.pseudocount > 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 