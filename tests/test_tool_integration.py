"""
Integration Tests for Bioinformatics Tools

Test realistic scenarios to ensure tools are working and properly configured.
"""

import pytest
import asyncio
import os
from pathlib import Path

from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config
from nanobrain.library.tools.bioinformatics.muscle_tool import MUSCLETool, MUSCLEConfig
from nanobrain.library.tools.bioinformatics.pssm_generator_tool import PSSMGeneratorTool, PSSMConfig


class TestToolInstallations:
    """Test actual tool installations and accessibility"""
    
    def test_bvbrc_installation_path_exists(self):
        """Test if BV-BRC installation path exists"""
        bvbrc_path = Path("/Applications/BV-BRC.app/")
        
        if bvbrc_path.exists():
            print(f"✅ BV-BRC found at {bvbrc_path}")
            
            # Check for executable path
            exec_path = bvbrc_path / "Contents/Resources/deployment/bin/"
            if exec_path.exists():
                print(f"✅ BV-BRC executables found at {exec_path}")
                
                # List some expected executables
                expected_tools = ["p3-all-genomes", "p3-get-feature-data", "p3-get-feature-sequence"]
                for tool in expected_tools:
                    tool_path = exec_path / tool
                    if tool_path.exists():
                        print(f"✅ Found {tool}")
                    else:
                        print(f"⚠️ Missing {tool}")
            else:
                print(f"❌ BV-BRC executables not found at {exec_path}")
        else:
            print(f"❌ BV-BRC not found at {bvbrc_path}")
            print("ℹ️ Install BV-BRC from https://www.bv-brc.org/")
    
    def test_mmseqs_installation(self):
        """Test if MMseqs2 is available"""
        import shutil
        
        mmseqs_path = shutil.which("mmseqs")
        if mmseqs_path:
            print(f"✅ MMseqs2 found at {mmseqs_path}")
        else:
            print("❌ MMseqs2 not found in PATH")
            print("ℹ️ Install with: conda install -c conda-forge mmseqs2")
    
    def test_muscle_installation(self):
        """Test if MUSCLE is available"""
        import shutil
        
        muscle_path = shutil.which("muscle")
        if muscle_path:
            print(f"✅ MUSCLE found at {muscle_path}")
        else:
            print("❌ MUSCLE not found in PATH")
            print("ℹ️ Install with: conda install -c bioconda muscle")
    
    def test_python_dependencies(self):
        """Test if required Python dependencies are available"""
        required_packages = [
            "numpy",
            "biopython", 
            "pandas",
            "aiohttp",
            "asyncio"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} available")
            except ImportError:
                print(f"❌ {package} not available")


class TestToolConfigurations:
    """Test tool configurations and basic functionality"""
    
    def test_bvbrc_config_creation(self):
        """Test BV-BRC configuration setup"""
        config = BVBRCConfig(
            tool_name="bv_brc_test",
            verify_on_init=False
        )
        
        assert config.tool_name == "bv_brc_test"
        assert config.anonymous_access == True
        assert config.genome_batch_size > 0
        assert config.md5_batch_size > 0
        print("✅ BV-BRC configuration test passed")
    
    def test_mmseqs_config_creation(self):
        """Test MMseqs2 configuration setup"""
        config = MMseqs2Config(
            tool_name="mmseqs_test",
            verify_on_init=False,
            min_seq_id=0.7,
            coverage=0.8
        )
        
        assert config.tool_name == "mmseqs_test"
        assert config.min_seq_id == 0.7
        assert config.coverage == 0.8
        print("✅ MMseqs2 configuration test passed")
    
    def test_muscle_config_creation(self):
        """Test MUSCLE configuration setup"""
        config = MUSCLEConfig(
            tool_name="muscle_test",
            verify_on_init=False,
            max_iterations=16
        )
        
        assert config.tool_name == "muscle_test"
        assert config.max_iterations == 16
        print("✅ MUSCLE configuration test passed")
    
    def test_pssm_config_creation(self):
        """Test PSSM generator configuration setup"""
        config = PSSMConfig(
            tool_name="pssm_test",
            verify_on_init=False,
            pseudocount=0.01
        )
        
        assert config.tool_name == "pssm_test"
        assert config.pseudocount == 0.01
        print("✅ PSSM configuration test passed")


class TestToolInitialization:
    """Test tool initialization without async issues"""
    
    def test_bvbrc_tool_creation(self):
        """Test BV-BRC tool can be created without errors"""
        config = BVBRCConfig(
            tool_name="bv_brc_init_test",
            verify_on_init=False
        )
        
        try:
            tool = BVBRCTool(config)
            assert tool.tool_name == "bv_brc_init_test"
            assert hasattr(tool, 'bv_brc_config')
            print("✅ BV-BRC tool initialization test passed")
        except Exception as e:
            print(f"❌ BV-BRC tool initialization failed: {e}")
            raise
    
    def test_mmseqs_tool_creation(self):
        """Test MMseqs2 tool can be created without errors"""
        config = MMseqs2Config(
            tool_name="mmseqs_init_test",
            verify_on_init=False
        )
        
        try:
            tool = MMseqs2Tool(config)
            assert tool.tool_name == "mmseqs_init_test"
            assert hasattr(tool, 'mmseqs_config')
            print("✅ MMseqs2 tool initialization test passed")
        except Exception as e:
            print(f"❌ MMseqs2 tool initialization failed: {e}")
            raise
    
    def test_muscle_tool_creation(self):
        """Test MUSCLE tool can be created without errors"""
        config = MUSCLEConfig(
            tool_name="muscle_init_test",
            verify_on_init=False
        )
        
        try:
            tool = MUSCLETool(config)
            assert tool.tool_name == "muscle_init_test"
            assert hasattr(tool, 'muscle_config')
            print("✅ MUSCLE tool initialization test passed")
        except Exception as e:
            print(f"❌ MUSCLE tool initialization failed: {e}")
            raise
    
    def test_pssm_tool_creation(self):
        """Test PSSM tool can be created without errors"""
        config = PSSMConfig(
            tool_name="pssm_init_test",
            verify_on_init=False
        )
        
        try:
            tool = PSSMGeneratorTool(config)
            assert tool.tool_name == "pssm_init_test"
            assert hasattr(tool, 'pssm_config')
            print("✅ PSSM tool initialization test passed")
        except Exception as e:
            print(f"❌ PSSM tool initialization failed: {e}")
            raise


class TestWorkflowCompatibility:
    """Test workflow compatibility and data flow"""
    
    def test_data_containers(self):
        """Test data container compatibility"""
        from nanobrain.library.tools.bioinformatics.bv_brc_tool import GenomeData, ProteinData
        
        # Test GenomeData creation
        genome = GenomeData(
            genome_id="test.1",
            genome_length=11700,
            genome_name="Test Genome",
            taxon_lineage="Test Lineage"
        )
        
        assert genome.genome_id == "test.1"
        assert genome.genome_length == 11700
        
        # Test ProteinData creation
        protein = ProteinData(
            patric_id="fig|test.1.peg.1",
            aa_sequence_md5="test_md5",
            product="test protein",
            gene="testA",
            aa_sequence="MKLLVVVAG"
        )
        
        assert protein.patric_id == "fig|test.1.peg.1"
        assert protein.aa_sequence_md5 == "test_md5"
        
        print("✅ Data container compatibility test passed")
    
    def test_configuration_file_structure(self):
        """Test configuration file structure compatibility"""
        from pathlib import Path
        
        # Check config directories exist
        config_dirs = [
            "nanobrain/library/workflows/viral_protein_analysis/config",
            "nanobrain/config"
        ]
        
        for config_dir in config_dirs:
            config_path = Path(config_dir)
            if config_path.exists():
                print(f"✅ Configuration directory found: {config_dir}")
            else:
                print(f"ℹ️ Configuration directory not found: {config_dir}")
    
    def test_logging_integration(self):
        """Test logging system integration"""
        from nanobrain.core.logging_system import get_logger
        
        # Test logger creation
        logger = get_logger("test_integration")
        assert logger is not None
        
        # Test logging functionality
        logger.info("Integration test logging verification")
        print("✅ Logging integration test passed")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements 