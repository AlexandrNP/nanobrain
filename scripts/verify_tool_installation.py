#!/usr/bin/env python3
"""
Comprehensive Tool Installation Verification for NanoBrain Framework

This script verifies:
1. Current tool installation status
2. Auto-detection capabilities
3. Auto-installation functionality
4. Tool accessibility and functionality
5. Installation guidance for missing tools
"""

import asyncio
import sys
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add nanobrain to path for imports
sys.path.insert(0, os.path.abspath('.'))

from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool, MMseqs2Config
from nanobrain.library.tools.bioinformatics.muscle_tool import MUSCLETool, MUSCLEConfig
from nanobrain.library.tools.bioinformatics.pubmed_client import PubMedClient, PubMedConfig
from nanobrain.core.logging_system import get_logger


class ToolInstallationVerifier:
    """Comprehensive tool installation verification system"""
    
    def __init__(self):
        self.logger = get_logger("tool_verifier")
        self.results = {}
        
    async def verify_all_tools(self) -> Dict[str, Any]:
        """Verify installation status of all tools"""
        print("🔧 NanoBrain Tool Installation Verification")
        print("=" * 50)
        
        # Test each tool
        tools_to_test = [
            ("BV-BRC", self.verify_bvbrc_tool),
            ("MMseqs2", self.verify_mmseqs2_tool),
            ("MUSCLE", self.verify_muscle_tool),
            ("PubMed Client", self.verify_pubmed_tool),
        ]
        
        for tool_name, verify_func in tools_to_test:
            print(f"\n🔍 Verifying {tool_name}...")
            try:
                result = await verify_func()
                self.results[tool_name] = result
                self._print_tool_status(tool_name, result)
            except Exception as e:
                self.results[tool_name] = {
                    "status": "error",
                    "error": str(e),
                    "installation_detected": False,
                    "auto_installation_available": False
                }
                print(f"❌ {tool_name} verification failed: {e}")
        
        # Generate summary report
        await self.generate_summary_report()
        return self.results
    
    async def verify_bvbrc_tool(self) -> Dict[str, Any]:
        """Verify BV-BRC tool installation and functionality"""
        result = {
            "status": "unknown",
            "installation_detected": False,
            "auto_installation_available": False,
            "installation_path": None,
            "executable_path": None,
            "installation_type": None,
            "issues": [],
            "suggestions": [],
            "version": None
        }
        
        try:
            # Create tool instance
            config = BVBRCConfig(verify_on_init=False)
            tool = BVBRCTool.from_config(config)
            
            # Test installation detection
            status = await tool.detect_existing_installation()
            result.update({
                "installation_detected": status.found,
                "installation_path": status.installation_path,
                "executable_path": status.executable_path,
                "installation_type": status.installation_type,
                "issues": status.issues,
                "suggestions": status.suggestions,
                "version": status.version
            })
            
            # Test auto-installation capability
            if hasattr(tool, 'install_if_missing'):
                result["auto_installation_available"] = True
            
            if status.found:
                result["status"] = "available"
                # Test basic functionality
                try:
                    await tool.initialize_tool()
                    result["status"] = "functional"
                except Exception as e:
                    result["status"] = "detected_but_non_functional"
                    result["issues"].append(f"Functionality test failed: {e}")
            else:
                result["status"] = "not_found"
                
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(str(e))
            
        return result
    
    async def verify_mmseqs2_tool(self) -> Dict[str, Any]:
        """Verify MMseqs2 tool installation and functionality"""
        result = {
            "status": "unknown",
            "installation_detected": False,
            "auto_installation_available": False,
            "installation_path": None,
            "executable_path": None,
            "installation_type": None,
            "issues": [],
            "suggestions": [],
            "version": None
        }
        
        try:
            # Create tool instance
            config = MMseqs2Config(tool_name="mmseqs2_verify")
            tool = MMseqs2Tool.from_config(config)
            
            # Test installation detection
            status = await tool.detect_existing_installation()
            result.update({
                "installation_detected": status.found,
                "installation_path": status.installation_path,
                "executable_path": status.executable_path,
                "installation_type": status.installation_type,
                "issues": status.issues,
                "suggestions": status.suggestions,
                "version": status.version
            })
            
            # Test auto-installation capability
            if hasattr(tool, 'install_if_missing'):
                result["auto_installation_available"] = True
            
            # Check if tool is in PATH
            mmseqs_path = shutil.which("mmseqs")
            if mmseqs_path:
                result["system_path_available"] = True
                result["system_path_location"] = mmseqs_path
                if not status.found:
                    result["installation_detected"] = True
                    result["installation_type"] = "system_path"
                    result["executable_path"] = mmseqs_path
            
            if result["installation_detected"]:
                result["status"] = "available"
            else:
                result["status"] = "not_found"
                result["suggestions"].extend([
                    "Install via conda: conda install -c bioconda mmseqs2",
                    "Install via bioconda: conda install -c conda-forge mmseqs2"
                ])
                
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(str(e))
            
        return result
    
    async def verify_muscle_tool(self) -> Dict[str, Any]:
        """Verify MUSCLE tool installation and functionality"""
        result = {
            "status": "unknown",
            "installation_detected": False,
            "auto_installation_available": False,
            "installation_path": None,
            "executable_path": None,
            "installation_type": None,
            "issues": [],
            "suggestions": [],
            "version": None
        }
        
        try:
            # Create tool instance
            config = MUSCLEConfig(tool_name="muscle_verify")
            tool = MUSCLETool.from_config(config)
            
            # Test installation detection
            status = await tool.detect_existing_installation()
            result.update({
                "installation_detected": status.found,
                "installation_path": status.installation_path,
                "executable_path": status.executable_path,
                "installation_type": status.installation_type,
                "issues": status.issues,
                "suggestions": status.suggestions,
                "version": status.version
            })
            
            # Test auto-installation capability
            if hasattr(tool, 'install_if_missing'):
                result["auto_installation_available"] = True
            
            # Check if tool is in PATH
            muscle_path = shutil.which("muscle")
            if muscle_path:
                result["system_path_available"] = True
                result["system_path_location"] = muscle_path
                if not status.found:
                    result["installation_detected"] = True
                    result["installation_type"] = "system_path"
                    result["executable_path"] = muscle_path
            
            if result["installation_detected"]:
                result["status"] = "available"
            else:
                result["status"] = "not_found"
                result["suggestions"].extend([
                    "Install via conda: conda install -c bioconda muscle",
                    "Download from: https://drive5.com/muscle/"
                ])
                
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(str(e))
            
        return result
    
    async def verify_pubmed_tool(self) -> Dict[str, Any]:
        """Verify PubMed client functionality"""
        result = {
            "status": "unknown",
            "installation_detected": False,
            "auto_installation_available": True,  # Python-based, no external deps
            "installation_path": "nanobrain.library.tools.bioinformatics.pubmed_client",
            "executable_path": None,
            "installation_type": "python_module",
            "issues": [],
            "suggestions": [],
            "version": None
        }
        
        try:
            # Test module import
            from nanobrain.library.tools.bioinformatics.pubmed_client import PubMedClient, PubMedConfig
            result["installation_detected"] = True
            result["status"] = "available"
            
            # Test basic functionality with proper configuration
            config = PubMedConfig(
                tool_name="pubmed_verify",
                email="test@example.com"
            )
            client = PubMedClient.from_config(config)
            result["status"] = "functional"
            
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(str(e))
            
        return result
    
    def _print_tool_status(self, tool_name: str, result: Dict[str, Any]):
        """Print formatted tool status"""
        status = result["status"]
        
        if status == "functional":
            print(f"  ✅ {tool_name}: Fully functional")
        elif status == "available":
            print(f"  ✅ {tool_name}: Available")
        elif status == "detected_but_non_functional":
            print(f"  ⚠️  {tool_name}: Detected but not functional")
        elif status == "not_found":
            print(f"  ❌ {tool_name}: Not found")
        elif status == "error":
            print(f"  💥 {tool_name}: Error during verification")
        
        # Print details
        if result.get("installation_detected"):
            print(f"     Installation Type: {result.get('installation_type', 'unknown')}")
            if result.get("installation_path"):
                print(f"     Installation Path: {result['installation_path']}")
            if result.get("executable_path"):
                print(f"     Executable Path: {result['executable_path']}")
        
        # Print auto-installation info
        if result.get("auto_installation_available"):
            print(f"     Auto-installation: Available")
        else:
            print(f"     Auto-installation: Not available")
        
        # Print issues
        if result.get("issues"):
            for issue in result["issues"]:
                print(f"     ⚠️  Issue: {issue}")
        
        # Print top suggestion
        if result.get("suggestions"):
            print(f"     💡 Suggestion: {result['suggestions'][0]}")
    
    async def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"\n📋 INSTALLATION SUMMARY REPORT")
        print("=" * 40)
        
        # Count statuses
        functional_count = sum(1 for r in self.results.values() if r["status"] == "functional")
        available_count = sum(1 for r in self.results.values() if r["status"] == "available")
        detected_count = sum(1 for r in self.results.values() if r["status"] == "detected_but_non_functional")
        missing_count = sum(1 for r in self.results.values() if r["status"] == "not_found")
        error_count = sum(1 for r in self.results.values() if r["status"] == "error")
        
        print(f"✅ Fully Functional: {functional_count}")
        print(f"✅ Available: {available_count}")
        print(f"⚠️  Detected but Non-functional: {detected_count}")
        print(f"❌ Not Found: {missing_count}")
        print(f"💥 Errors: {error_count}")
        
        # Auto-installation capabilities
        auto_install_count = sum(1 for r in self.results.values() if r.get("auto_installation_available"))
        print(f"\n🔧 Auto-installation Available: {auto_install_count}/{len(self.results)}")
        
        # Installation recommendations
        print(f"\n💡 INSTALLATION RECOMMENDATIONS")
        print("-" * 30)
        
        for tool_name, result in self.results.items():
            if result["status"] in ["not_found", "error"]:
                print(f"\n{tool_name}:")
                for suggestion in result.get("suggestions", ["Manual installation required"]):
                    print(f"  • {suggestion}")
        
        # Overall status
        total_tools = len(self.results)
        working_tools = functional_count + available_count
        
        print(f"\n📊 OVERALL STATUS")
        print("-" * 20)
        if working_tools == total_tools:
            print("🎉 ALL TOOLS READY FOR PRODUCTION!")
        elif working_tools >= total_tools * 0.75:
            print("✅ MAJORITY OF TOOLS AVAILABLE - Production ready with limitations")
        elif working_tools >= total_tools * 0.5:
            print("⚠️  PARTIAL TOOL AVAILABILITY - Some workflows may not work")
        else:
            print("❌ SIGNIFICANT SETUP REQUIRED - Most tools unavailable")
        
        print(f"   Working: {working_tools}/{total_tools} ({working_tools/total_tools*100:.1f}%)")
    
    async def test_auto_installation(self, tool_name: str) -> bool:
        """Test auto-installation for a specific tool"""
        print(f"\n🔄 Testing auto-installation for {tool_name}...")
        
        try:
            if tool_name.lower() == "mmseqs2":
                config = MMseqs2Config(tool_name="mmseqs2_auto_install_test")
                tool = MMseqs2Tool.from_config(config)
                
                # Check if installation is needed
                status = await tool.detect_existing_installation()
                if status.found:
                    print(f"  ✅ {tool_name} already installed")
                    return True
                
                # Attempt auto-installation
                print(f"  🔄 Attempting auto-installation...")
                success = await tool.install_if_missing()
                
                if success:
                    print(f"  ✅ {tool_name} auto-installation successful")
                    return True
                else:
                    print(f"  ❌ {tool_name} auto-installation failed")
                    return False
                    
            else:
                print(f"  ℹ️  Auto-installation not implemented for {tool_name}")
                return False
                
        except Exception as e:
            print(f"  ❌ Auto-installation error: {e}")
            return False


async def main():
    """Main verification function"""
    verifier = ToolInstallationVerifier()
    
    try:
        # Run comprehensive verification
        results = await verifier.verify_all_tools()
        
        # Optionally test auto-installation for missing tools
        print(f"\n🧪 TESTING AUTO-INSTALLATION CAPABILITIES")
        print("=" * 45)
        
        missing_tools = [name for name, result in results.items() 
                        if result["status"] == "not_found" and result.get("auto_installation_available")]
        
        if missing_tools:
            print(f"Testing auto-installation for: {', '.join(missing_tools)}")
            for tool in missing_tools:
                await verifier.test_auto_installation(tool)
        else:
            print("No missing tools with auto-installation capability found.")
        
        return results
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main()) 