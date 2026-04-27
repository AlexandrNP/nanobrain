#!/usr/bin/env python3
"""
NanoBrain Tools Verification Script

This script verifies that all bioinformatics tools are properly installed
and configured for the NanoBrain viral protein analysis workflow.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
import importlib

def check_python_dependencies():
    """Check required Python packages"""
    print("🔍 Checking Python Dependencies...")
    
    required_packages = {
        'numpy': '✅ Required for numerical computations',
        'pandas': '✅ Required for data manipulation', 
        'aiohttp': '✅ Required for async HTTP requests',
        'biopython': '⚠️ Required for sequence analysis',
        'pytest': '✅ Required for testing'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}: Available")
        except ImportError:
            print(f"  ❌ {package}: Missing - {description}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_bvbrc_installation():
    """Check BV-BRC installation"""
    print("\n🔍 Checking BV-BRC Installation...")
    
    app_path = Path("/Applications/BV-BRC.app/")
    if not app_path.exists():
        print("  ❌ BV-BRC application not found at /Applications/BV-BRC.app/")
        print("  📥 Install from: https://www.bv-brc.org/")
        return False
    
    print(f"  ✅ BV-BRC application found at {app_path}")
    
    # Check executables
    exec_paths = [
        app_path / "deployment/bin/",
        app_path / "Contents/Resources/deployment/bin/"
    ]
    
    for exec_path in exec_paths:
        if exec_path.exists():
            p3_all_genomes = exec_path / "p3-all-genomes"
            if p3_all_genomes.exists():
                print(f"  ✅ BV-BRC executables found at {exec_path}")
                print(f"  ✅ p3-all-genomes available")
                return True
    
    print("  ❌ BV-BRC executables not found")
    return False

def check_bioinformatics_tools():
    """Check other bioinformatics tools"""
    print("\n🔍 Checking Bioinformatics Tools...")
    
    tools = {
        'mmseqs': 'MMseqs2 for sequence clustering',
        'muscle': 'MUSCLE for multiple sequence alignment'
    }
    
    all_available = True
    missing_tools = []
    
    for tool, description in tools.items():
        tool_path = shutil.which(tool)
        if tool_path:
            print(f"  ✅ {tool}: Found at {tool_path}")
        else:
            print(f"  ❌ {tool}: Not found - {description}")
            all_available = False
            missing_tools.append(tool)
    
    return all_available, missing_tools

def check_nanobrain_framework():
    """Check NanoBrain framework components"""
    print("\n🔍 Checking NanoBrain Framework...")
    
    try:
        # Test core imports
        from nanobrain.core.logging_system import get_logger
        print("  ✅ Core logging system")
        
        from nanobrain.library.tools.bioinformatics.bv_brc_tool import BVBRCTool, BVBRCConfig
        print("  ✅ BV-BRC tool wrapper")
        
        from nanobrain.library.tools.bioinformatics.mmseqs_tool import MMseqs2Tool
        print("  ✅ MMseqs2 tool wrapper")
        
        from nanobrain.library.tools.bioinformatics.muscle_tool import MUSCLETool
        print("  ✅ MUSCLE tool wrapper")
        
        from nanobrain.library.tools.bioinformatics.pssm_generator_tool import PSSMGeneratorTool
        print("  ✅ PSSM generator tool")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def run_quick_tests():
    """Run quick functionality tests"""
    print("\n🔍 Running Quick Tests...")
    
    try:
        # Run the test suite
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_external_tools.py',
            'tests/test_tool_integration.py',
            '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ All tests passed!")
            return True
        else:
            print("  ❌ Some tests failed:")
            print(result.stdout[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️ Tests timed out")
        return False
    except Exception as e:
        print(f"  ❌ Test execution failed: {e}")
        return False

def print_installation_instructions(missing_packages, missing_tools):
    """Print installation instructions for missing components"""
    print("\n📋 INSTALLATION INSTRUCTIONS")
    print("=" * 50)
    
    if missing_packages:
        print("\n🐍 Install Missing Python Packages:")
        print("   Using conda (recommended):")
        conda_packages = []
        for pkg in missing_packages:
            if pkg == 'biopython':
                conda_packages.append('bioconda::biopython')
            else:
                conda_packages.append(pkg)
        print(f"   conda install {' '.join(conda_packages)}")
        
        print("\n   Using pip (alternative):")
        print(f"   pip install {' '.join(missing_packages)}")
    
    if missing_tools:
        print("\n🧬 Install Missing Bioinformatics Tools:")
        for tool in missing_tools:
            if tool == 'mmseqs':
                print("   conda install -c conda-forge mmseqs2")
            elif tool == 'muscle':
                print("   conda install -c bioconda muscle")

def main():
    """Main verification function"""
    print("🧬 NanoBrain Bioinformatics Tools Verification")
    print("=" * 50)
    
    all_good = True
    missing_packages = []
    missing_tools = []
    
    # Check Python dependencies
    py_ok, missing_py = check_python_dependencies()
    if not py_ok:
        all_good = False
        missing_packages = missing_py
    
    # Check BV-BRC
    bvbrc_ok = check_bvbrc_installation()
    if not bvbrc_ok:
        all_good = False
    
    # Check bioinformatics tools
    tools_ok, missing_biotools = check_bioinformatics_tools()
    if not tools_ok:
        missing_tools = missing_biotools
        # Note: Tools are optional for framework testing
    
    # Check NanoBrain framework
    framework_ok = check_nanobrain_framework()
    if not framework_ok:
        all_good = False
    
    # Run tests if basic components are available
    if framework_ok:
        tests_ok = run_quick_tests()
        if not tests_ok:
            print("  ℹ️ Tests failed but framework components are available")
    
    # Summary
    print("\n📋 VERIFICATION SUMMARY")
    print("=" * 30)
    
    if all_good and not missing_tools:
        print("🎉 ALL SYSTEMS READY!")
        print("   You can run the complete viral protein analysis workflow.")
    elif framework_ok and bvbrc_ok:
        print("✅ CORE FRAMEWORK READY!")
        print("   BV-BRC workflows can be executed.")
        if missing_tools:
            print("   Install MMseqs2/MUSCLE for full functionality.")
    else:
        print("⚠️ SETUP INCOMPLETE")
        print("   Some required components are missing.")
    
    if missing_packages or missing_tools:
        print_installation_instructions(missing_packages, missing_tools)
    
    print(f"\n📊 Status: {'✅ READY' if all_good else '🔧 NEEDS SETUP'}")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main()) 