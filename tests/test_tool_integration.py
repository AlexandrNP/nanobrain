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
from nanobrain.library.tools.bioinformatics.pubmed_client import PubMedClient


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
                # Try alternative path
                alt_exec_path = bvbrc_path / "deployment/bin/"
                if alt_exec_path.exists():
                    print(f"✅ BV-BRC executables found at {alt_exec_path}")
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


class TestToolInitialization:
    """Test tool initialization and configuration"""
    
    @pytest.mark.asyncio
    async def test_bvbrc_tool_creation(self):
        """Test BV-BRC tool creation via from_config with mandatory card"""
        try:
            from nanobrain.core.config.component_factory import ComponentFactory
            factory = ComponentFactory()
            
            # Use the default configuration file with tool_card
            tool = factory.create_from_yaml_file(
                'nanobrain/library/config/defaults/tools/BVBRCTool.yml',
                'nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool'
            )
            assert tool is not None
            print("✅ BV-BRC tool created successfully with tool card")
        except Exception as e:
            print(f"❌ BV-BRC tool creation failed: {e}")
            pytest.fail(f"BV-BRC tool creation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_mmseqs2_tool_creation(self):
        """Test MMseqs2 tool creation via from_config with mandatory card"""
        try:
            from nanobrain.core.config.component_factory import ComponentFactory
            factory = ComponentFactory()
            
            # Use the default configuration file with tool_card
            tool = factory.create_from_yaml_file(
                'nanobrain/library/config/defaults/tools/MMseqs2Tool.yml',
                'nanobrain.library.tools.bioinformatics.mmseqs_tool.MMseqs2Tool'
            )
            assert tool is not None
            print("✅ MMseqs2 tool created successfully with tool card")
        except Exception as e:
            print(f"❌ MMseqs2 tool creation failed: {e}")
            pytest.fail(f"MMseqs2 tool creation failed: {e}")
    
    @pytest.mark.asyncio
    async def test_muscle_tool_creation(self):
        """Test MUSCLE tool creation via from_config with mandatory card"""
        try:
            from nanobrain.core.config.component_factory import ComponentFactory
            factory = ComponentFactory()
            
            # Use the default configuration file with tool_card
            tool = factory.create_from_yaml_file(
                'nanobrain/library/config/defaults/tools/MUSCLETool.yml',
                'nanobrain.library.tools.bioinformatics.muscle_tool.MUSCLETool'
            )
            assert tool is not None
            print("✅ MUSCLE tool created successfully with tool card")
        except Exception as e:
            print(f"❌ MUSCLE tool creation failed: {e}")
            pytest.fail(f"MUSCLE tool creation failed: {e}")


class TestToolAutoInstallation:
    """Test auto-installation functionality"""
    
    @pytest.mark.asyncio
    async def test_tool_detection_and_installation_bvbrc(self):
        """Test BV-BRC detection and installation guidance"""
        try:
            from nanobrain.core.config.component_factory import ComponentFactory
            factory = ComponentFactory()
            
            # Use the default configuration file with tool_card
            tool = factory.create_from_yaml_file(
                'nanobrain/library/config/defaults/tools/BVBRCTool.yml',
                'nanobrain.library.tools.bioinformatics.bv_brc_tool.BVBRCTool'
            )
            
            # Test installation detection
            status = await tool.detect_existing_installation()
            
            print(f"BV-BRC Detection Results:")
            print(f"  Found: {status.found}")
            print(f"  Installation Type: {status.installation_type}")
            print(f"  Installation Path: {status.installation_path}")
            print(f"  Executable Path: {status.executable_path}")
            print(f"  Issues: {status.issues}")
            print(f"  Suggestions: {status.suggestions}")
            
            assert hasattr(status, 'found')
            assert hasattr(status, 'installation_type')
            assert hasattr(status, 'issues')
            assert hasattr(status, 'suggestions')
            
        except Exception as e:
            print(f"❌ BV-BRC detection test failed: {e}")
            pytest.fail(f"BV-BRC detection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_tool_detection_and_installation_mmseqs2(self):
        """Test MMseqs2 detection and installation guidance"""
        try:
            from nanobrain.core.config.component_factory import ComponentFactory
            factory = ComponentFactory()
            
            # Use the default configuration file with tool_card
            tool = factory.create_from_yaml_file(
                'nanobrain/library/config/defaults/tools/MMseqs2Tool.yml',
                'nanobrain.library.tools.bioinformatics.mmseqs_tool.MMseqs2Tool'
            )
            
            # Test installation detection
            status = await tool.detect_existing_installation()
            
            print(f"MMseqs2 Detection Results:")
            print(f"  Found: {status.found}")
            print(f"  Installation Type: {status.installation_type}")
            print(f"  Installation Path: {status.installation_path}")
            print(f"  Executable Path: {status.executable_path}")
            print(f"  Issues: {status.issues}")
            print(f"  Suggestions: {status.suggestions}")
            
            assert hasattr(status, 'found')
            assert hasattr(status, 'installation_type')
            assert hasattr(status, 'issues')
            assert hasattr(status, 'suggestions')
            
        except Exception as e:
            print(f"❌ MMseqs2 detection test failed: {e}")
            pytest.fail(f"MMseqs2 detection failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 