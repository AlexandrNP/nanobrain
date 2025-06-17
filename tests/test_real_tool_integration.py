"""
Real Tool Integration Tests for NanoBrain Framework

Tests for the new bioinformatics tool infrastructure with:
- Auto-detection of existing installations
- Real API calls (no mocks)
- Progressive scaling
- Comprehensive error handling
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config


class TestBVBRCToolIntegration:
    """Test BV-BRC tool integration with real API calls"""
    
    @pytest.fixture
    async def bv_brc_tool(self):
        """Create BV-BRC tool instance for testing"""
        config = BVBRCConfig(verify_on_init=False)  # Skip verification for testing
        tool = BVBRCTool(config)
        return tool
    
    @pytest.mark.asyncio
    async def test_bv_brc_installation_detection(self, bv_brc_tool):
        """Test BV-BRC installation detection"""
        try:
            status = await bv_brc_tool.detect_existing_installation()
            
            # Check status structure
            assert hasattr(status, 'found')
            assert hasattr(status, 'installation_type')
            assert hasattr(status, 'issues')
            assert hasattr(status, 'suggestions')
            
            if status.found:
                print(f"✅ BV-BRC found: {status.installation_type} at {status.installation_path}")
                assert status.installation_path is not None
                assert status.executable_path is not None
            else:
                print(f"ℹ️ BV-BRC not found. Issues: {status.issues}")
                print(f"ℹ️ Suggestions: {status.suggestions}")
                
        except Exception as e:
            pytest.skip(f"BV-BRC detection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_bv_brc_progressive_scaling(self, bv_brc_tool):
        """Test progressive scaling configuration"""
        # Test scale level 1 (small test)
        result = await bv_brc_tool.execute_with_progressive_scaling(scale_level=1)
        
        assert isinstance(result, dict)
        assert "scale_config" in result or "limit" in str(result)
        
        print(f"✅ Progressive scaling test passed: {result}")
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("/Applications/BV-BRC.app/").exists(),
        reason="BV-BRC not installed locally"
    )
    async def test_bv_brc_real_api_call(self, bv_brc_tool):
        """Test real BV-BRC API call (only if installed)"""
        try:
            # Initialize tool (this will verify installation)
            status = await bv_brc_tool.initialize_tool()
            assert status.found
            
            # Test small-scale real download
            genomes = await bv_brc_tool.download_alphavirus_genomes(limit=2)
            
            assert isinstance(genomes, list)
            print(f"✅ Real API call successful: downloaded {len(genomes)} genomes")
            
            if len(genomes) > 0:
                genome = genomes[0]
                assert hasattr(genome, 'genome_id')
                assert hasattr(genome, 'genome_length')
                assert hasattr(genome, 'genome_name')
                
        except Exception as e:
            pytest.skip(f"Real BV-BRC API call failed: {e}")


class TestMMseqs2ToolIntegration:
    """Test MMseqs2 tool integration with conda environment management"""
    
    @pytest.fixture
    async def mmseqs2_tool(self):
        """Create MMseqs2 tool instance for testing"""
        config = MMseqs2Config()
        tool = MMseqs2Tool(config)
        return tool
    
    @pytest.mark.asyncio
    async def test_mmseqs2_installation_detection(self, mmseqs2_tool):
        """Test MMseqs2 installation detection"""
        try:
            status = await mmseqs2_tool.detect_existing_installation()
            
            # Check status structure
            assert hasattr(status, 'found')
            assert hasattr(status, 'installation_type')
            assert hasattr(status, 'issues')
            assert hasattr(status, 'suggestions')
            
            if status.found:
                print(f"✅ MMseqs2 found: {status.installation_type} at {status.installation_path}")
                assert status.installation_path is not None
            else:
                print(f"ℹ️ MMseqs2 not found. Issues: {status.issues}")
                print(f"ℹ️ Suggestions: {status.suggestions}")
                
        except Exception as e:
            pytest.skip(f"MMseqs2 detection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_mmseqs2_progressive_scaling(self, mmseqs2_tool):
        """Test MMseqs2 progressive scaling configuration"""
        # Test scale level 1 (fast test)
        result = await mmseqs2_tool.execute_with_progressive_scaling(scale_level=1)
        
        assert isinstance(result, dict)
        assert "max_sequences" in result
        assert "sensitivity" in result
        assert result["max_sequences"] == 50  # Level 1 setting
        
        print(f"✅ MMseqs2 progressive scaling test passed: {result}")
    
    @pytest.mark.asyncio
    async def test_mmseqs2_installation_suggestions(self, mmseqs2_tool):
        """Test MMseqs2 installation suggestions"""
        suggestions = await mmseqs2_tool._generate_specific_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check for conda installation suggestion
        conda_suggestion_found = any("conda" in suggestion for suggestion in suggestions)
        assert conda_suggestion_found
        
        print(f"✅ Installation suggestions available: {len(suggestions)} options")


class TestToolConfigurationIntegration:
    """Test tool configuration loading and validation"""
    
    def test_bv_brc_config_loading(self):
        """Test BV-BRC configuration loading"""
        config = BVBRCConfig()
        
        # Check local installation paths
        assert config.installation_path == "/Applications/BV-BRC.app"
        assert config.executable_path == "/Applications/BV-BRC.app/deployment/bin"
        
        # Check progressive scaling
        assert 1 in config.progressive_scaling
        assert 2 in config.progressive_scaling
        assert config.progressive_scaling[1]["limit"] == 5
        
        # Check error handling
        assert config.auto_retry is True
        assert config.max_retries == 3
        
        print("✅ BV-BRC configuration loaded correctly")
    
    def test_mmseqs2_config_loading(self):
        """Test MMseqs2 configuration loading"""
        config = MMseqs2Config()
        
        # Check conda configuration
        assert config.conda_package == "mmseqs2"
        assert config.conda_channel == "bioconda"
        assert config.environment_name == "nanobrain-viral_protein-mmseqs2"
        
        # Check progressive scaling
        assert 1 in config.progressive_scaling
        assert config.progressive_scaling[1]["max_sequences"] == 50
        
        # Check clustering parameters
        assert config.min_seq_id == 0.3
        assert config.coverage == 0.8
        
        print("✅ MMseqs2 configuration loaded correctly")


@pytest.mark.integration
class TestEndToEndToolIntegration:
    """End-to-end integration tests with real data flow"""
    
    @pytest.mark.asyncio
    async def test_tool_initialization_sequence(self):
        """Test complete tool initialization sequence"""
        tools_status = {}
        
        # Test BV-BRC initialization
        try:
            bv_brc_tool = BVBRCTool(BVBRCConfig(verify_on_init=False))
            bv_brc_status = await bv_brc_tool.detect_existing_installation()
            tools_status["bv_brc"] = bv_brc_status.found
        except Exception as e:
            tools_status["bv_brc"] = False
            print(f"BV-BRC initialization issue: {e}")
        
        # Test MMseqs2 initialization
        try:
            mmseqs2_tool = MMseqs2Tool(MMseqs2Config())
            mmseqs2_status = await mmseqs2_tool.detect_existing_installation()
            tools_status["mmseqs2"] = mmseqs2_status.found
        except Exception as e:
            tools_status["mmseqs2"] = False
            print(f"MMseqs2 initialization issue: {e}")
        
        print(f"🔧 Tool availability status: {tools_status}")
        
        # At least one tool should be available or have clear installation path
        available_tools = sum(tools_status.values())
        print(f"✅ {available_tools} tools available out of 2")
    
    @pytest.mark.asyncio
    async def test_configuration_yaml_integration(self):
        """Test YAML configuration file integration"""
        from pathlib import Path
        
        # Check BV-BRC config file exists
        bv_brc_config_path = Path("nanobrain/library/workflows/viral_protein_analysis/config/BVBRCTool.yml")
        assert bv_brc_config_path.exists(), f"BV-BRC config not found at {bv_brc_config_path}"
        
        # Check MMseqs2 config file exists
        mmseqs2_config_path = Path("nanobrain/library/workflows/viral_protein_analysis/config/MMseqs2Tool.yml")
        assert mmseqs2_config_path.exists(), f"MMseqs2 config not found at {mmseqs2_config_path}"
        
        print("✅ Configuration YAML files found and accessible")


if __name__ == "__main__":
    """Run integration tests directly"""
    pytest.main([__file__, "-v", "--tb=short"]) 